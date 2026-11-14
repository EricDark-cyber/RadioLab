"""WSPR : balises synthétiques, décodées à partir du son (12,5 kHz) avec une horloge simulée."""
import numpy as np
import pytest
from radiolab.dsp import wspr
from radiolab.dsp.hfdigi import make_decoder

FS = 12500.0


def beacon(call, loc, dbm, f_audio, start_s, total_s=120.0):
    """Son d'une balise : 162 symboles à quatre tonalités, à partir de start_s."""
    sym = wspr.encode(call, loc, dbm)
    t = np.arange(int(total_s * FS)) / FS
    k = np.floor((t - start_s) / wspr.SYMBOL_S).astype(np.int64)
    on = (k >= 0) & (k < wspr.SYMBOLS)
    f = f_audio + (sym[np.clip(k, 0, wspr.SYMBOLS - 1)] - 1.5) * wspr.TONE_HZ
    return np.where(on, np.cos(2 * np.pi * np.cumsum(f) / FS), 0.0)


def noise(n, snr_db, seed):
    """Bruit blanc tel qu'une balise d'amplitude 1 ait le rapport signal/bruit voulu dans 2,5 kHz (convention WSPR)."""
    sigma = np.sqrt(0.5 * 10 ** (-snr_db / 10) * (FS / 2) / 2500.0)
    return sigma * np.random.default_rng(seed).standard_normal(n)


def receive(audio, t_start=0.0):
    """Fait défiler le son dans le décodeur, l'horloge avançant au rythme des échantillons."""
    out = []
    now = [t_start]
    d = wspr.WSPRDecoder(FS, out.append, clock=lambda: now[0])
    d.threaded = False
    for i in range(0, len(audio), 4096):
        now[0] = t_start + (i + 4096) / FS
        d.process(audio[i:i + 4096])
    return d, "".join(out)


def test_message_coding():
    for call, loc, dbm in (("F4ABC", "JN18", 23), ("K1JT", "FN20", 37), ("DL0XYZ", "JO62", 0), ("G4AB", "IO91", 60)):
        bits = wspr.pack(call, loc, dbm)
        assert len(bits) == 50 and wspr.unpack(bits) == (call, loc, dbm)
        sym = wspr.encode(call, loc, dbm)
        assert len(sym) == 162 and set(sym) <= {0, 1, 2, 3} and np.array_equal(sym % 2, wspr.SYNC)
        coded = np.array(wspr.convolve(bits + [0] * 31))
        assert wspr.sequential_decode(3.0 * (2 * coded - 1.0)) == bits
        flipped = 3.0 * (2 * coded - 1.0); flipped[[5, 40, 41, 90, 133]] *= -1          # cinq bits faux : corrigés
        assert wspr.sequential_decode(flipped) == bits
    assert sorted(wspr.INTERLEAVE) == list(range(162))
    assert wspr.unpack([1] * 50) is None                                               # contenu invraisemblable : rejeté
    assert wspr.sequential_decode(4 * np.random.default_rng(0).standard_normal(162)) is None


@pytest.mark.parametrize("snr", [-10, -20])
def test_two_beacons(snr):
    a = beacon("F4ABC", "JN18", 23, 1480.0, 1.0)
    b = 0.7 * beacon("DL0XYZ", "JO62", 37, 1532.7, 1.6)          # seconde balise, 0,6 s en retard, à 53 Hz de la première
    audio = a + b + noise(len(a), snr, 1)
    d, out = receive(audio)
    assert "F4ABC  JN18 23 dBm" in out and "DL0XYZ JO62 37 dBm" in out, out
    lines = out.strip().split("\n")
    assert len(lines) == 2 and lines[0].startswith("00:00") and "1480.0 Hz" in lines[0] or "1479" in lines[0] or "1480" in lines[0]
    assert d.decodes == 2 and "2 balises" in d.status()


def test_noise_only_and_next_slot():
    """Du bruit seul ne produit aucun faux message ; la période suivante est décodée à son tour."""
    quiet = noise(int(120 * FS), -20, 2)
    second = beacon("G4AB", "IO91", 10, 1555.0, 1.0) + noise(int(120 * FS), -18, 3)
    d, out = receive(np.concatenate([quiet, second]))
    lines = out.strip().split("\n")
    assert len(lines) == 1 and "G4AB" in lines[0] and "IO91 10 dBm" in lines[0] and lines[0].startswith("00:02"), out


def test_wiring():
    d = make_decoder("WSPR", FS, 1000.0, lambda s: None)
    assert isinstance(d, wspr.WSPRDecoder) and abs(d.fs_d - 378.8) < 0.1
