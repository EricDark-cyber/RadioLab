"""Packet AX.25 (APRS 1200 bauds en FM, packet HF 300 bauds en BLU), anti-fading, binaural, forme du filtre."""
import numpy as np
import pytest
from radiolab.dsp.receiver import Receiver
from radiolab.dsp.packet import crc16_x25, parse_ax25

FS = 250e3
OFF = 30e3


def test_crc_reference_value():
    assert crc16_x25(b"123456789") == 0x906E                 # valeur de contrôle publiée du CRC-16/X.25


def ax25_frame(src, dest, path, info):
    def addr(call, last):
        name, _, ssid = call.partition("-")
        return bytes((ord(c) << 1) for c in name.ljust(6)) + bytes([0x60 | (int(ssid or 0) << 1) | int(last)])
    calls = [dest, src] + path
    body = b"".join(addr(c, i == len(calls) - 1) for i, c in enumerate(calls)) + b"\x03\xf0" + info.encode()
    fcs = crc16_x25(body)
    return body + bytes([fcs & 0xFF, fcs >> 8])


def afsk_audio(frames, baud, f_mark, f_space, fs=FS):
    bits = []
    flag = [0, 1, 1, 1, 1, 1, 1, 0]
    for frame in frames:
        bits += flag * 30
        ones = 0
        for byte in frame:
            for k in range(8):
                b = (byte >> k) & 1
                bits.append(b)
                ones = ones + 1 if b else 0
                if ones == 5:
                    bits.append(0); ones = 0                 # bit de bourrage
        bits += flag * 4
    tone, level = [], 1
    for b in bits:                                           # NRZI : un 0 change de tonalité
        if b == 0:
            level ^= 1
        tone.append(level)
    n = int(len(tone) * fs / baud)
    idx = np.minimum((np.arange(n) * baud / fs).astype(np.int64), len(tone) - 1)
    f = np.where(np.array(tone)[idx] == 1, f_mark, f_space)
    return np.sin(2 * np.pi * np.cumsum(f) / fs)


def add_noise(x, snr_db, seed=0):
    r = np.random.default_rng(seed)
    sigma = np.sqrt(np.mean(np.abs(x) ** 2) * 10 ** (-snr_db / 10) * FS / 2500.0)
    return (x + sigma * (r.standard_normal(len(x)) + 1j * r.standard_normal(len(x))) / np.sqrt(2)).astype(np.complex64)


FRAMES = [ax25_frame("F4TEST-9", "APRS", ["WIDE1-1", "WIDE2-2"], "!4852.00N/00220.00E>RadioLab"),
          ax25_frame("F1ABC", "APDR16", [], ">Bonjour de Paris")]
EXPECTED = ["F4TEST-9>APRS,WIDE1-1,WIDE2-2:!4852.00N/00220.00E>RadioLab", "F1ABC>APDR16:>Bonjour de Paris"]


def test_parse_ax25():
    assert parse_ax25(FRAMES[0][:-2]) == EXPECTED[0] and parse_ax25(b"\x00" * 20) is None


@pytest.mark.parametrize("noise", [0.002, 0.02])
def test_aprs_1200_in_nfm(noise):
    audio = afsk_audio(FRAMES, 1200, 1200.0, 2200.0)
    audio = np.concatenate([np.zeros(int(FS * 0.3)), audio, np.zeros(int(FS * 0.3))])
    phase = 2 * np.pi * OFF * np.arange(len(audio)) / FS + 2 * np.pi * 3000 * np.cumsum(audio) / FS
    r = np.random.default_rng(2)
    iq = (0.1 * np.exp(1j * phase) + noise * (r.standard_normal(len(audio)) + 1j * r.standard_normal(len(audio)))).astype(np.complex64)
    rx = Receiver(FS); rx.set_mode("NFM"); rx.set_offset(OFF); rx.set_digi("APRS1200")
    assert rx.digi is not None
    for i in range(0, len(iq), 32768):
        rx.process(iq[i:i + 32768])
    assert "".join(rx.digi_text).splitlines() == EXPECTED
    rx.set_mode("USB"); assert rx.digi is None                # ce décodeur n'a de sens qu'en FM étroite


@pytest.mark.parametrize("mode", ["USB", "LSB"])
def test_packet_300_in_ssb(mode):
    audio = afsk_audio(FRAMES, 300, 1600.0, 1800.0)
    t = np.arange(len(audio)) / FS
    # émission BLU : chaque tonalité audio f devient une raie à (porteuse +/- f)
    sign = 1 if mode == "USB" else -1
    f = sign * tone_track(FRAMES)
    iq = 0.02 * np.exp(1j * (2 * np.pi * OFF * t + 2 * np.pi * np.cumsum(f) / FS))
    rx = Receiver(FS); rx.set_mode(mode); rx.set_offset(OFF); rx.set_digi("PACKET300", 1700.0)
    iq = add_noise(iq, 15)
    for i in range(0, len(iq), 32768):
        rx.process(iq[i:i + 32768])
    assert "".join(rx.digi_text).splitlines() == EXPECTED


def tone_track(frames, baud=300, f_mark=1600.0, f_space=1800.0):
    """Fréquence audio instantanée de l'émission packet (même codage que afsk_audio)."""
    probe = afsk_audio(frames, baud, 1.0, 2.0)               # on ne garde que la longueur
    bits, flag = [], [0, 1, 1, 1, 1, 1, 1, 0]
    for frame in frames:
        bits += flag * 30
        ones = 0
        for byte in frame:
            for k in range(8):
                b = (byte >> k) & 1
                bits.append(b); ones = ones + 1 if b else 0
                if ones == 5:
                    bits.append(0); ones = 0
        bits += flag * 4
    tone, level = [], 1
    for b in bits:
        if b == 0:
            level ^= 1
        tone.append(level)
    idx = np.minimum((np.arange(len(probe)) * baud / FS).astype(np.int64), len(tone) - 1)
    return np.where(np.array(tone)[idx] == 1, f_mark, f_space)


def test_packet_silent_on_noise():
    r = np.random.default_rng(5)
    n = int(FS * 6)
    iq = (0.01 * (r.standard_normal(n) + 1j * r.standard_normal(n))).astype(np.complex64)
    for mode, kind in (("NFM", "APRS1200"), ("USB", "PACKET300")):
        rx = Receiver(FS); rx.set_mode(mode); rx.set_offset(OFF); rx.set_digi(kind, 1700.0)
        for i in range(0, n, 32768):
            rx.process(iq[i:i + 32768])
        assert "".join(rx.digi_text) == ""


def level_at(audio, f, fs=48000, skip=0.4):
    a = audio[int(skip * fs):]
    w = np.hanning(len(a))
    return 20 * np.log10(abs(np.sum(a * w * np.exp(-2j * np.pi * f * np.arange(len(a)) / fs))) / w.sum() * 2 + 1e-12)


def run(rx, iq):
    parts = [rx.process(iq[i:i + 32768]) for i in range(0, len(iq), 32768)]
    return np.concatenate([p for p in parts if len(p)])


def am_rx(**kw):
    rx = Receiver(FS); rx.set_mode("AM"); rx.set_offset(OFF); rx.set_agc(False); rx.volume = 1.0
    rx.set_am_sync(True, kw.get("anti_fading", False))
    return rx


def test_anti_fading_picks_the_surviving_sideband():
    n = int(FS * 1.5)
    t = np.arange(n) / FS
    carrier = np.exp(2j * np.pi * OFF * t)
    upper, lower = np.exp(2j * np.pi * (OFF + 1000) * t), np.exp(2j * np.pi * (OFF - 1000) * t)
    normal = (0.02 * (carrier + 0.25 * upper + 0.25 * lower)).astype(np.complex64)          # AM modulée à 50 %
    faded = (0.02 * (carrier + 0.25 * upper + 0.01 * lower)).astype(np.complex64)           # bande basse évanouie (-28 dB)
    ref = level_at(run(am_rx(), normal), 1000)
    rx = am_rx(anti_fading=True)
    assert abs(level_at(run(rx, normal), 1000) - ref) < 0.5 and rx.fade_choice == "les deux"
    rx = am_rx(anti_fading=True)
    kept = level_at(run(rx, faded), 1000)
    assert rx.fade_choice == "haute" and abs(kept - ref) < 0.7                              # niveau rétabli
    assert ref - level_at(run(am_rx(), faded), 1000) > 5                                    # sans anti-fading : -5,7 dB


def test_binaural_puts_quadrature_on_the_two_ears():
    n = int(FS * 1.2)
    iq = (0.01 * np.exp(2j * np.pi * (OFF + 1000) * np.arange(n) / FS)).astype(np.complex64)
    rx = Receiver(FS); rx.set_mode("USB"); rx.set_offset(OFF); rx.set_agc(False); rx.volume = 1.0; rx.set_binaural(True)
    out = run(rx, iq)
    assert out.ndim == 2
    seg = out[24000:48000]
    w = np.hanning(len(seg))
    ph = [np.angle(np.sum(seg[:, k] * w * np.exp(-2j * np.pi * 1000 * np.arange(len(seg)) / 48000))) for k in (0, 1)]
    assert abs(abs((ph[0] - ph[1] + np.pi) % (2 * np.pi) - np.pi) - np.pi / 2) < 0.05       # 90 degrés entre les oreilles
    assert abs(level_at(out[:, 0], 1000) - level_at(out[:, 1], 1000)) < 0.3


def test_filter_shape_changes_selectivity():
    n = int(FS * 1.0)
    inside = (0.01 * np.exp(2j * np.pi * (OFF + 1500) * np.arange(n) / FS)).astype(np.complex64)
    edge = (0.01 * np.exp(2j * np.pi * (OFF + 3100) * np.arange(n) / FS)).astype(np.complex64)   # 300 Hz hors du filtre

    def rejection(window, order):
        vals = []
        for sig, f in ((inside, 1500), (edge, 3100)):
            rx = Receiver(FS); rx.set_mode("USB"); rx.set_offset(OFF); rx.set_agc(False); rx.volume = 1.0
            rx.set_filter_shape(window, order)
            vals.append(level_at(run(rx, sig), f))
        return vals[0] - vals[1]
    assert rejection(None, None) > 60                          # réglage automatique
    assert rejection("Blackman-Harris", 31) < 20               # filtre court : flancs mous
    assert rejection("Blackman-Harris", 1001) > 60             # filtre long : flancs raides
    assert rejection("Hamming", 401) > rejection("Hamming", 101)
