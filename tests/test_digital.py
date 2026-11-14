"""DCS, POCSAG, corrections audio : émetteurs synthétiques et mots de référence publiés."""
import numpy as np
import pytest
from radiolab.dsp.receiver import Receiver
from radiolab.dsp.digital import (dcs_codeword, DCS_CODES, pocsag_syndrome, pocsag_encode, pocsag_correct,
                                  POCSAG_SYNC, POCSAG_IDLE, BitSlicer)
from radiolab.dsp.audiofx import AudioFX

FS = 250e3
OFF = 20e3


def fm(audio, dev=5e3, noise=0.003, seed=1):
    n = len(audio)
    phase = 2 * np.pi * OFF * np.arange(n) / FS + 2 * np.pi * dev * np.cumsum(audio) / FS
    r = np.random.default_rng(seed)
    return (0.1 * np.exp(1j * phase) + noise * (r.standard_normal(n) + 1j * r.standard_normal(n))).astype(np.complex64)


def nfm_rx():
    rx = Receiver(FS); rx.set_mode("NFM"); rx.set_offset(OFF)
    return rx


def run(rx, iq, block=32768):
    return [rx.process(iq[i:i + block]) for i in range(0, len(iq), block)]


def test_dcs_reference_codewords():
    # mots publiés dans la littérature (table Midland) : parité, « 100 », code
    assert dcs_codeword(0o023) == 0b11101100011100000010011
    assert dcs_codeword(0o346) == 0b01110101001100011100110
    assert len(DCS_CODES) == 83


@pytest.mark.parametrize("code,inverted", [(0o023, False), (0o754, False), (0o346, True)])
def test_dcs(code, inverted):
    n = int(FS * 3)
    t = np.arange(n) / FS
    word = dcs_codeword(code)
    bit_index = (t * 134.4).astype(np.int64) % 23
    nrz = ((word >> bit_index) & 1) * 2.0 - 1.0                     # bit de poids faible émis en premier
    if inverted:
        nrz = -nrz
    k = int(FS / 300)
    sub = np.convolve(nrz, np.ones(k) / k, "same")                  # filtrage subaudible de l'émetteur
    voice = 0.5 * np.sin(2 * np.pi * 1100 * t) * (np.sin(2 * np.pi * 2.5 * t) > 0)
    rx = nfm_rx()
    run(rx, fm(voice + 0.15 * sub))
    assert rx.dcs_label == "D%03o%s" % (code, "I" if inverted else "N")
    rx = nfm_rx()
    run(rx, fm(voice))
    assert rx.dcs_label is None and rx.ctcss_hz is None


def test_pocsag_code_matches_standard_words():
    assert pocsag_syndrome(POCSAG_SYNC) == 0 and pocsag_syndrome(POCSAG_IDLE) == 0
    assert pocsag_encode(POCSAG_IDLE >> 11) == POCSAG_IDLE and pocsag_encode(POCSAG_SYNC >> 11) == POCSAG_SYNC
    assert pocsag_correct(POCSAG_IDLE ^ (1 << 19)) == POCSAG_IDLE and pocsag_correct(POCSAG_IDLE ^ 0b101) is None


def pocsag_bits(ric, text, numeric=False):
    """Préambule + lots POCSAG portant un message."""
    if numeric:
        table = "0123456789*U -)("
        stream = []
        for ch in text:
            v = table.index(ch); stream += [(v >> k) & 1 for k in range(4)]
        while len(stream) % 20: stream += [0, 0, 1, 1]              # remplissage par des espaces
    else:
        stream = []
        for ch in text:
            stream += [(ord(ch) >> k) & 1 for k in range(7)]
        while len(stream) % 20: stream.append(0)
    words = [pocsag_encode((0 << 20) | ((ric >> 3) << 2) | (0 if numeric else 3))]
    for i in range(0, len(stream), 20):
        words.append(pocsag_encode((1 << 20) | int("".join(map(str, stream[i:i + 20])), 2)))
    slots = [POCSAG_IDLE] * (2 * (ric & 7)) + words
    while len(slots) % 16: slots.append(POCSAG_IDLE)
    slots += [POCSAG_IDLE] * 16
    bits = [1, 0] * 288
    for b in range(0, len(slots), 16):
        for w in [POCSAG_SYNC] + slots[b:b + 16]:
            bits += [(w >> k) & 1 for k in range(31, -1, -1)]
    return np.array(bits)


@pytest.mark.parametrize("baud,inverted", [(1200, False), (512, False), (2400, False), (1200, True)])
def test_pocsag(baud, inverted):
    bits = np.concatenate([pocsag_bits(1234567, "BONJOUR DE RADIOLAB 73"), pocsag_bits(42, "0612345678", numeric=True)])
    n = int(FS * (len(bits) / baud + 0.6))
    idx = ((np.arange(n) / FS - 0.3) * baud).astype(np.int64)
    level = np.where((idx >= 0) & (idx < len(bits)), 1.0 - 2.0 * bits[np.clip(idx, 0, len(bits) - 1)], 0.0)   # 1 = fréquence basse
    if inverted:
        level = -level
    k = max(int(FS / baud / 4), 1)
    rx = nfm_rx(); rx.set_pocsag(True)
    run(rx, fm(np.convolve(level, np.ones(k) / k, "same"), dev=4.5e3, noise=0.01))
    got = [(m["ric"], m["text"], m["baud"]) for m in rx.pocsag_messages]
    assert got == [(1234567, "BONJOUR DE RADIOLAB 73", baud), (42, "0612345678", baud)], got


def test_pocsag_silent_on_noise():
    r = np.random.default_rng(4)
    rx = nfm_rx(); rx.set_pocsag(True)
    run(rx, (0.01 * (r.standard_normal(int(FS * 4)) + 1j * r.standard_normal(int(FS * 4)))).astype(np.complex64))
    assert len(rx.pocsag_messages) == 0 and rx.dcs_label is None


def tone_db(x, f, fs):
    w = np.hanning(len(x))
    return 20 * np.log10(abs(np.sum(x * w * np.exp(-2j * np.pi * f * np.arange(len(x)) / fs))) / w.sum() * 2 + 1e-12)


def test_notch_and_equalizer():
    fs = 12500.0
    t = np.arange(int(fs * 2)) / fs
    x = (np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 1300 * t)
         + np.sin(2 * np.pi * 4500 * t)).astype(np.float32) * 0.2
    fx = AudioFX(fs)
    assert not fx.active and fx.process(x) is x
    fx.configure([(1000, 100)], bass_db=6.0, treble_db=-6.0)
    y = np.concatenate([fx.process(x[i:i + 700]) for i in range(0, len(x), 700)])[2000:]
    ref = x[2000:]
    assert tone_db(ref, 1000, fs) - tone_db(y, 1000, fs) > 40          # porteuse supprimée
    assert abs(tone_db(y, 1300, fs) - tone_db(ref, 1300, fs)) < 0.5     # voisine intacte
    assert abs(tone_db(y, 100, fs) - tone_db(ref, 100, fs) - 6.0) < 0.7
    assert abs(tone_db(y, 4500, fs) - tone_db(ref, 4500, fs) + 6.0) < 0.7
    fx.configure([], 0.0, 0.0); assert not fx.active


def test_notch_in_receiver_and_raw_output():
    FSI, off = 2.5e6, 123456.0
    n = int(FSI)
    iq = (0.01 * np.exp(2j * np.pi * (off + 1000) * np.arange(n) / FSI) + 0.01 * np.exp(2j * np.pi * (off + 1700) * np.arange(n) / FSI)).astype(np.complex64)
    rx = Receiver(FSI); rx.set_mode("USB"); rx.set_offset(off); rx.set_agc(False); rx.volume = 1.0
    rx.set_audio_fx(notches=[(1000, 120)])
    out = np.concatenate(run(rx, iq, 65536))[24000:]
    assert tone_db(out, 1700, 48000) - tone_db(out, 1000, 48000) > 40
    rx = nfm_rx(); rx.set_raw_output(True)
    assert rx.fs2 == 25000.0
    t = np.arange(int(FS * 1)) / FS
    out = np.concatenate(run(rx, fm(0.5 * np.sin(2 * np.pi * 6000 * t))))[24000:]
    assert tone_db(out, 6000, 48000) > -22                             # 6 kHz passe sans atténuation notable
    rx.set_raw_output(False); assert rx.fs2 == 12500.0
