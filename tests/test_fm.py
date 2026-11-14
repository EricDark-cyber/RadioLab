"""FM de radiodiffusion (stéréo, RDS) et CTCSS, sur un émetteur synthétique."""
import numpy as np
import pytest
from radiolab.dsp.receiver import Receiver
from radiolab.dsp.fmradio import rds_checkword, ToneDetector, CTCSS_TONES

FS = 1.0e6
OFF = 150e3
OFFSET = {"A": 0x0FC, "B": 0x198, "C": 0x168, "D": 0x1B4}


def rds_bits(ps="TESTFM 1", rt="BONJOUR DE RADIOLAB", pi=0xF123, pty=10, seconds=5.0):
    groups = []
    rt = (rt + "\r").ljust(64)[:64]
    for i in range(4):
        groups.append((pi, (0 << 12) | (pty << 5) | i, 0xE0CD, (ord(ps[2 * i]) << 8) | ord(ps[2 * i + 1])))
    for i in range(6):
        seg = rt[4 * i:4 * i + 4]
        groups.append((pi, (2 << 12) | (pty << 5) | i, (ord(seg[0]) << 8) | ord(seg[1]), (ord(seg[2]) << 8) | ord(seg[3])))
    bits = []
    while len(bits) < seconds * 1187.5:
        for g in groups:
            for word, name in zip(g, "ABCD"):
                block = (word << 10) | (rds_checkword(word) ^ OFFSET[name])
                bits += [(block >> k) & 1 for k in range(25, -1, -1)]
    return np.array(bits, dtype=np.int64)


def fm_station(seconds, left_hz=1000.0, right_hz=None, stereo=True, rds=True, snr_db=40.0):
    n = int(FS * seconds)
    t = np.arange(n) / FS
    wp = 2 * np.pi * 19000.0 * t
    left = 0.8 * np.sin(2 * np.pi * left_hz * t) if left_hz else np.zeros(n)
    right = 0.8 * np.sin(2 * np.pi * right_hz * t) if right_hz else np.zeros(n)
    mpx = 0.45 * (left + right)
    if stereo:
        mpx += 0.45 * (left - right) * np.sin(2 * wp) + 0.09 * np.sin(wp)
    if rds:
        bits = rds_bits(seconds=seconds)
        diff = np.cumsum(bits) % 2                                   # codage différentiel
        chips = np.stack([2 * diff - 1, 1 - 2 * diff], axis=1).ravel()   # biphase : deux moitiés opposées
        base = chips[np.minimum((t * 2375.0).astype(np.int64), len(chips) - 1)].astype(float)
        k = int(FS / 2375 / 2)
        base = np.convolve(base, np.ones(k) / k, "same")             # adoucit les fronts
        mpx += 0.05 * base * np.sin(3 * wp)
    phase = 2 * np.pi * OFF * t + 2 * np.pi * 75e3 * np.cumsum(mpx) / FS
    r = np.random.default_rng(0)
    sigma = 0.1 * 10 ** (-snr_db / 20)
    return (0.1 * np.exp(1j * phase) + sigma * (r.standard_normal(n) + 1j * r.standard_normal(n))).astype(np.complex64)


def run(rx, iq, block=32768):
    parts = [rx.process(iq[i:i + block]) for i in range(0, len(iq), block)]
    return parts


def tone_level(x, f, fs=48000):
    w = np.hanning(len(x))
    return 20 * np.log10(abs(np.sum(x * w * np.exp(-2j * np.pi * f * np.arange(len(x)) / fs))) / w.sum() * 2 + 1e-12)


def make_rx():
    rx = Receiver(FS); rx.set_mode("WFM"); rx.set_offset(OFF); rx.volume = 1.0
    return rx


def test_stereo_separation():
    rx = make_rx()
    parts = run(rx, fm_station(1.5, left_hz=1000.0, right_hz=None, rds=False))
    assert rx.is_stereo and parts[-1].ndim == 2
    audio = np.concatenate([p for p in parts if p.ndim == 2])[-24000:]
    left, right = tone_level(audio[:, 0], 1000), tone_level(audio[:, 1], 1000)
    assert abs(left - 20 * np.log10(0.8 * 0.9 * 0.5)) < 1.5 and left - right > 30, (left, right)
    assert tone_level(audio[:, 0], 19000) < left - 45                # le pilote ne passe pas dans l'audio


def test_right_channel_and_swap():
    rx = make_rx()
    audio = np.concatenate([p for p in run(rx, fm_station(1.5, left_hz=None, right_hz=3000.0, rds=False)) if p.ndim == 2])[-24000:]
    assert tone_level(audio[:, 1], 3000) - tone_level(audio[:, 0], 3000) > 30


def test_mono_station_stays_mono():
    rx = make_rx()
    parts = run(rx, fm_station(1.0, stereo=False, rds=False))
    assert not rx.is_stereo and parts[-1].ndim == 1
    rx = make_rx(); rx.stereo_enabled = False
    parts = run(rx, fm_station(1.0, rds=False))
    assert not rx.is_stereo and parts[-1].ndim == 1


@pytest.mark.parametrize("snr_db", [40.0, 22.0])
def test_rds(snr_db):
    rx = make_rx()
    run(rx, fm_station(5.0, snr_db=snr_db))
    info = rx.rds_info()
    assert info["synced"] and info["pi"] == 0xF123, (info, rx.rds.groups_ok)
    assert info["ps"] == "TESTFM 1" and info["rt"] == "BONJOUR DE RADIOLAB" and info["pty"] == "Pop"
    assert rx.rds.groups_ok > 30


def test_rds_checkword_detects_errors():
    w = 0xF123
    block = (w << 10) | (rds_checkword(w) ^ 0x0FC)
    assert rds_checkword(block >> 10) ^ (block & 0x3FF) == 0x0FC
    assert rds_checkword((block ^ (1 << 17)) >> 10) ^ (block & 0x3FF) != 0x0FC


@pytest.mark.parametrize("tone", [67.0, 88.5, 123.0, 250.3])
def test_ctcss(tone):
    fs = 250e3
    n = int(fs * 3)
    t = np.arange(n) / fs
    audio = 0.6 * np.sin(2 * np.pi * 1000 * t) * (np.sin(2 * np.pi * 3 * t) > 0) + 0.15 * np.sin(2 * np.pi * tone * t)
    phase = 2 * np.pi * 20e3 * t + 2 * np.pi * 5e3 * np.cumsum(audio) / fs
    r = np.random.default_rng(1)
    iq = (0.1 * np.exp(1j * phase) + 0.003 * (r.standard_normal(n) + 1j * r.standard_normal(n))).astype(np.complex64)
    rx = Receiver(fs); rx.set_mode("NFM"); rx.set_offset(20e3)
    run(rx, iq)
    assert rx.ctcss_hz == tone
    rx = Receiver(fs); rx.set_mode("NFM"); rx.set_offset(20e3)
    run(rx, iq * 0 + 0.003 * (r.standard_normal(n) + 1j * r.standard_normal(n)).astype(np.complex64))
    assert rx.ctcss_hz is None                                       # pas de fausse détection sur du bruit
