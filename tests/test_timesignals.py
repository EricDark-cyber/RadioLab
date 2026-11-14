"""Signaux horaires et Hellschreiber : porteuses synthétiques, décodées à partir du son (12,5 kHz)."""
import numpy as np
import pytest
from radiolab.dsp import timesignals as ts
from radiolab.dsp.hfdigi import make_decoder
from radiolab.dsp.pictures import HellDecoder

FS = 12500.0
TONE = 1000.0


def noise(n, sigma, seed=0):
    return sigma * np.random.default_rng(seed).standard_normal(n)


def run(kind, audio):
    out = []
    d = make_decoder(kind, FS, TONE, out.append)
    for i in range(0, len(audio), 4096):
        d.process(audio[i:i + 4096])
    return d, "".join(out)


def seconds_stream(minutes, first_second):
    """Suite (bit ou None pour la seconde 59) à partir de la seconde first_second de la première minute."""
    stream = []
    for bits in minutes:
        stream += list(bits) + [None]
    return stream[first_second:]


def test_minute_coding():
    bits = ts.encode_minute(42, 15, 12, 4, 11, 2026, summer=False)
    assert len(bits) == 59 and ts.decode_minute(bits) == dict(minute=42, hour=15, day=12, weekday=4, month=11, year=2026, summer=False)
    assert ts.decode_minute(ts.encode_minute(7, 23, 31, 7, 8, 2025, summer=True))["summer"] is True
    for k in (22, 30, 40, 55):                                   # un bit faux : la parité le révèle
        bad = list(bits); bad[k] ^= 1
        assert ts.decode_minute(bad) is None
    assert ts.decode_minute(bits[:58]) is None


def dcf77_audio(stream, depth=0.15, f=TONE):
    n = int(len(stream) * FS)
    amp = np.ones(n)
    for s, bit in enumerate(stream):
        if bit is not None:
            amp[int(s * FS):int((s + (0.2 if bit else 0.1)) * FS)] = depth
    k = np.hanning(int(0.004 * FS)); amp = np.convolve(amp, k / k.sum(), "same")
    return amp * np.cos(2 * np.pi * f * np.arange(n) / FS)


@pytest.mark.parametrize("sigma,f", [(0.0, TONE), (0.5, TONE + 6.0)])
def test_dcf77(sigma, f):
    m1 = ts.encode_minute(42, 15, 12, 4, 11, 2026)
    m2 = ts.encode_minute(43, 15, 12, 4, 11, 2026)
    stream = seconds_stream([m1, m1, m2], 50) + [0, 1]            # pris à la seconde 50 : dix secondes inutilisables
    d, out = run("DCF77", 0.3 * dcf77_audio(stream, f=f) + 0.3 * noise(int(len(stream) * FS), sigma))
    assert "jeudi 12/11/2026 15:42 (heure d'hiver)" in out and "jeudi 12/11/2026 15:43" in out, out
    assert d.minutes == 2 and "".join(map(str, m1)) in out and "15:43" in d.status()


def test_dcf77_bad_minute():
    m1 = ts.encode_minute(5, 8, 1, 1, 6, 2026, summer=True)
    stream = seconds_stream([m1, m1], 55) + [0, 1, 0]
    audio = 0.3 * dcf77_audio(stream)
    audio[int(20.0 * FS):int(21.3 * FS)] *= 0.05                  # évanouissement d'une seconde : un bit manque
    d, out = run("DCF77", audio)
    assert d.minutes == 0 and ("erronée" in out or "?" in out), out


def als162_audio(stream, f=TONE):
    n = int(len(stream) * FS)
    phase = np.zeros(n)
    ramp = int(0.025 * FS)
    one = np.concatenate([np.linspace(0, 1, ramp, False), np.linspace(1, -1, 2 * ramp, False), np.linspace(-1, 0, ramp, False)])
    for s, bit in enumerate(stream):
        if bit is not None:
            a = int(s * FS)
            phase[a:a + len(one)] = one
            if bit:
                phase[a + len(one):a + 2 * len(one)] = one
    return np.cos(2 * np.pi * f * np.arange(n) / FS + phase)


@pytest.mark.parametrize("sigma,f", [(0.0, TONE), (0.4, TONE - 3.5)])
def test_als162(sigma, f):
    m1 = ts.encode_minute(59, 23, 31, 4, 12, 2026)
    m2 = ts.encode_minute(0, 0, 1, 5, 1, 2027)
    stream = [0, 1] + seconds_stream([m1, m1, m2], 52) + [0, 1, 0]
    audio = 0.3 * als162_audio(stream, f=f) + 0.3 * noise(int(len(stream) * FS), sigma, 3)
    d, out = run("ALS162", audio[int(0.37 * FS):])                # début quelconque dans la seconde
    assert "jeudi 31/12/2026 23:59 (heure d'hiver)" in out and "vendredi 01/01/2027 00:00" in out, out
    assert d.minutes == 2 and "".join(map(str, m1)) in out


GLYPHS = {"bloc": ["#####"] * 7, "barre": ["..#.."] * 7, "angle": ["#...."] * 6 + ["#####"],
          "trait": ["....."] * 3 + ["#####"] + ["....."] * 3}


def hell_audio(names, f=TONE):
    """Chaque dessin : 5 colonnes de 7 points, émises de bas en haut, plus deux colonnes vides."""
    dot = 1.0 / (17.5 * 7)
    keys = []
    for name in names:
        g = GLYPHS[name]
        for col in range(5):
            keys += [1.0 if g[6 - row][col] == "#" else 0.0 for row in range(7)]
        keys += [0.0] * 14
    n = int(len(keys) * dot * FS)
    key = np.array(keys)[np.minimum((np.arange(n) / (dot * FS)).astype(np.int64), len(keys) - 1)]
    k = np.hanning(int(0.002 * FS)); key = np.convolve(key, k / k.sum(), "same")
    return key * np.cos(2 * np.pi * f * np.arange(n) / FS)


def test_hell():
    names = ["bloc", "barre", "angle", "trait"] * 4
    audio = np.concatenate([np.zeros(int(0.2 * FS)), 0.3 * hell_audio(names)]) + noise(int(0.2 * FS) + len(hell_audio(names)), 0.01)
    d, _ = run("HELL", audio)
    assert isinstance(d, HellDecoder) and abs(d.count - (3.5 + 7 * 16)) <= 2
    tape = d.tape()
    assert tape.shape == (2 * 14 * 3, 420) and tape.dtype == np.uint8
    used = tape[:, :d.count]
    black = used < 100
    share = black.mean()
    expected = np.mean([[c == "#" for c in "".join(GLYPHS[n])] for n in names]) * 5 / 7
    assert abs(share - expected) < 0.05, (share, expected)        # autant de noir que dans les dessins émis
    assert np.array_equal(black[:42], black[42:])                 # tracé en double : deux bandes identiques
    cols = black[:42].mean(axis=0)
    assert (cols > 0.9).sum() >= 4 * 5 - 2 and (cols < 0.05).sum() >= 16 * 2 - 4      # colonnes pleines (blocs) et vides (espaces)
    assert tape[:, 300:].min() == 255 and "colonnes" in d.status()                     # le reste de la bande est blanc
