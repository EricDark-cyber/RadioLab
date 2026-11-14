"""RTTY, PSK31/63 et CW : émissions synthétiques reçues par la chaîne complète (I/Q -> BLU -> décodeur)."""
import numpy as np
import pytest
from radiolab.dsp.receiver import Receiver
from radiolab.dsp.hfdigi import VARICODE, MORSE, _LETTERS, _FIGURES

FS = 250e3
OFF = 30e3
LETTERS = {c: i for i, c in enumerate(_LETTERS)}
FIGURES = {c: i for i, c in enumerate(_FIGURES)}


def add_noise(x, snr_db, seed=0):
    """Bruit blanc tel que le rapport signal/bruit dans 2,5 kHz vaille snr_db."""
    r = np.random.default_rng(seed)
    p = np.mean(np.abs(x) ** 2)
    sigma = np.sqrt(p * 10 ** (-snr_db / 10) * FS / 2500.0)
    return (x + sigma * (r.standard_normal(len(x)) + 1j * r.standard_normal(len(x))) / np.sqrt(2)).astype(np.complex64)


def psk_signal(text, baud, f_audio):
    bits = [1] * 0 + [0] * 40                                    # préambule : inversions continues
    for ch in text:
        bits += [int(b) for b in VARICODE[ord(ch)]] + [0, 0]
    bits += [0] * 20
    levels = np.cumprod([1.0] + [1.0 if b else -1.0 for b in bits])           # 0 = inversion de phase
    sps = int(FS / baud)
    amp = np.repeat(levels, sps).astype(np.float64)
    k = np.hanning(sps); amp = np.convolve(amp, k / k.sum(), "same")          # transitions en cosinus
    t = np.arange(len(amp)) / FS
    return 0.02 * amp * np.exp(2j * np.pi * (OFF + f_audio) * t)


def rtty_signal(text, baud, f_audio, mark_high=True, shift=170.0):
    bits = [1] * 30
    figs = False
    for ch in text:
        if ch in LETTERS and ch not in "\x00\x0e\x0f":
            if figs and ch != " ": bits += [0] + [(31 >> i) & 1 for i in range(5)] + [1, 1]; figs = False
            code = LETTERS[ch]
        else:
            if not figs: bits += [0] + [(27 >> i) & 1 for i in range(5)] + [1, 1]; figs = True
            code = FIGURES[ch]
        if ch == " ": figs = False
        bits += [0] + [(code >> i) & 1 for i in range(5)] + [1, 1]            # départ, 5 bits, arrêt (2 bits)
    bits += [1] * 20
    n = int(len(bits) * FS / baud)
    b = np.array(bits)[np.minimum((np.arange(n) * baud / FS).astype(np.int64), len(bits) - 1)]
    dev = (shift / 2) * (2 * b - 1) * (1 if mark_high else -1)
    phase = 2 * np.pi * np.cumsum(OFF + f_audio + dev) / FS
    return 0.02 * np.exp(1j * phase)


def cw_signal(text, wpm, f_rf):
    inv = {v: k for k, v in MORSE.items()}
    dot = int(FS * 1.2 / wpm)
    key = [0] * (10 * dot)
    for word in text.split(" "):
        for ch in word:
            for el in inv[ch]:
                key += [1] * (dot if el == "." else 3 * dot) + [0] * dot
            key += [0] * (2 * dot)
        key += [0] * (4 * dot)
    key = np.array(key + [0] * (10 * dot), dtype=np.float64)
    k = np.hanning(int(FS * 0.005)); key = np.convolve(key, k / k.sum(), "same")   # fronts adoucis (5 ms)
    return 0.02 * key * np.exp(2j * np.pi * f_rf * np.arange(len(key)) / FS)


def decode(iq, mode, kind, center=1000.0, reverse=False, chunk=32768):
    rx = Receiver(FS); rx.set_mode(mode); rx.set_offset(OFF)
    rx.set_digi(kind, center, reverse)
    for i in range(0, len(iq), chunk):
        rx.process(iq[i:i + chunk])
    return "".join(rx.digi_text)


@pytest.mark.parametrize("kind,baud,snr", [("PSK31", 31.25, 30), ("PSK31", 31.25, 3), ("PSK63", 62.5, 6)])
def test_psk(kind, baud, snr):
    text = "CQ CQ de F4TEST F4TEST pse k\n"
    out = decode(add_noise(psk_signal(text, baud, 1000.0), snr), "USB", kind)
    assert text.strip() in out, repr(out)


def test_psk_afc_and_lsb():
    text = "RadioLab PSK31 test 599 73"
    out = decode(add_noise(psk_signal(text, 31.25, 1006.0), 15), "USB", "PSK31")       # 6 Hz à côté
    assert text in out, repr(out)
    out = decode(add_noise(np.conj(psk_signal(text, 31.25, 1000.0)) * np.exp(4j * np.pi * OFF * np.arange(len(psk_signal(text, 31.25, 1000.0))) / FS), 15), "LSB", "PSK31")
    assert text in out, repr(out)                                                      # le PSK ne dépend pas de la bande latérale


@pytest.mark.parametrize("kind,baud,snr", [("RTTY45", 45.45, 30), ("RTTY45", 45.45, 8), ("RTTY50", 50.0, 12)])
def test_rtty(kind, baud, snr):
    text = "RYRYRY CQ CQ DE F4TEST F4TEST 599 73 K"
    out = decode(add_noise(rtty_signal(text, baud, 1000.0), snr), "USB", kind)
    assert "CQ CQ DE F4TEST F4TEST 599 73 K" in out, repr(out)      # les tout premiers caractères peuvent manquer


def test_rtty_polarity():
    text = "CQ DE F4TEST 12345"
    sig = rtty_signal("RYRY " + text, 45.45, 1000.0, mark_high=False)
    assert text not in decode(add_noise(sig, 20), "USB", "RTTY45")
    assert text in decode(add_noise(sig, 20), "USB", "RTTY45", reverse=True)
    # en BLI, le sens normal est inversé automatiquement
    lsb = rtty_signal("RYRY " + text, 45.45, -1000.0)           # même émetteur (mark en haut), reçu 1 kHz sous la porteuse
    assert text in decode(add_noise(lsb, 20), "LSB", "RTTY45")


@pytest.mark.parametrize("wpm,snr", [(20, 30), (12, 15), (30, 12), (20, 6)])
def test_cw(wpm, snr):
    text = "CQ CQ DE F4TEST F4TEST K"
    out = decode(add_noise(cw_signal(text, wpm, OFF), snr, seed=wpm), "CW", "CW")      # mode CW : note de 700 Hz
    assert "DE F4TEST F4TEST K" in out, repr(out)                                      # les premiers signes servent au réglage de vitesse


def test_cw_in_usb_and_speed():
    rx = Receiver(FS); rx.set_mode("USB"); rx.set_offset(OFF); rx.set_digi("CW", 1200.0)
    iq = add_noise(cw_signal("PARIS PARIS PARIS PARIS", 25, OFF + 1200.0), 20)
    for i in range(0, len(iq), 32768):
        rx.process(iq[i:i + 32768])
    assert "PARIS PARIS PARIS" in "".join(rx.digi_text) and abs(rx.digi.wpm() - 25) < 3


@pytest.mark.parametrize("kind", ["PSK31", "RTTY45", "CW"])
def test_silent_on_noise(kind):
    out = decode(add_noise(np.zeros(int(FS * 12), dtype=np.complex64) + 1e-9, -60), "USB", kind)
    assert len(out.replace(" ", "")) <= 3, repr(out)


def test_decoder_off_outside_ssb():
    rx = Receiver(FS); rx.set_digi("PSK31")
    assert rx.digi is not None
    rx.set_mode("NFM"); assert rx.digi is None
    rx.set_mode("LSB"); assert rx.digi is not None
    rx.set_digi(None); assert rx.digi is None


def test_cw_manual_speed_and_threshold():
    text = "CQ CQ DE F4TEST F4TEST K"
    iq = add_noise(cw_signal(text, 20, OFF), 20)
    rx = Receiver(FS); rx.set_mode("CW"); rx.set_offset(OFF); rx.set_digi("CW")
    for i in range(0, len(iq), 32768):
        rx.process(iq[i:i + 32768])
    auto_thr = rx.digi.thr_db
    assert auto_thr is not None and len(rx.digi.history) > 500 and max(rx.digi.history) > auto_thr > min(rx.digi.history)
    rx = Receiver(FS); rx.set_mode("CW"); rx.set_offset(OFF); rx.set_digi("CW")
    rx.set_cw_options(wpm=20, threshold_db=auto_thr)          # tout en manuel, avec les bonnes valeurs
    for i in range(0, len(iq), 32768):
        rx.process(iq[i:i + 32768])
    out = "".join(rx.digi_text)
    assert text in out and rx.digi.wpm() == 20 and abs(rx.digi.thr_db - auto_thr) < 0.01, repr(out)
    rx.set_mode("USB"); rx.set_mode("CW")                      # les réglages survivent à un changement de mode
    assert rx.digi.auto_speed is False and rx.digi.manual_thr is not None
    rx.set_cw_options(); assert rx.digi.auto_speed and rx.digi.manual_thr is None
    rx = Receiver(FS); rx.set_mode("CW"); rx.set_offset(OFF); rx.set_digi("CW")
    rx.set_cw_options(threshold_db=auto_thr + 40)              # seuil trop haut : plus rien n'est décodé
    for i in range(0, len(iq), 32768):
        rx.process(iq[i:i + 32768])
    assert "".join(rx.digi_text).strip() == ""


def test_cw_silent_between_transmissions():
    a = cw_signal("CQ CQ DE PE1NCP PE1NCP K", 25, OFF)
    gap = np.zeros(int(FS * 15), dtype=np.complex128)           # 15 s de bruit seul entre deux appels
    iq = add_noise(np.concatenate([a, gap, a]), 25)
    rx = Receiver(FS); rx.set_mode("CW"); rx.set_offset(OFF); rx.set_digi("CW")
    marks = {}
    for i in range(0, len(iq), 32768):
        rx.process(iq[i:i + 32768])
        t = i / FS
        if abs(t - (len(a) / FS + 1.0)) < 0.07: marks["start"] = len("".join(rx.digi_text))
        if abs(t - (len(a) / FS + 14.0)) < 0.07: marks["end"] = len("".join(rx.digi_text))
    out = "".join(rx.digi_text)
    assert out.count("PE1NCP PE1NCP K") == 2, repr(out)
    assert len(out[marks["start"]:marks["end"]].strip()) <= 1, repr(out[marks["start"]:marks["end"]])


def test_psk_constellation():
    """Signal propre : les points se regroupent en deux taches, à -1 et +1 sur l'axe horizontal."""
    rx = Receiver(FS); rx.set_mode("USB"); rx.set_offset(OFF); rx.set_digi("PSK31", 1000.0)
    iq = add_noise(psk_signal("CQ CQ de F4TEST constellation", 31.25, 1000.0), 25)
    for i in range(0, len(iq), 32768):
        rx.process(iq[i:i + 32768])
    pts = np.array(rx.digi.points)[-150:]
    assert len(pts) == 150 and np.mean(np.abs(np.abs(pts.real) - 1.0) < 0.35) > 0.9 and np.mean(np.abs(pts.imag)) < 0.2
    assert (pts.real > 0).any() and (pts.real < 0).any()
