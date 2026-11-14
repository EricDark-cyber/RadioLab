"""Banc d'essai du traitement du signal sur signaux synthétiques (pytest)."""
import time
import numpy as np
import pytest
from radiolab.dsp.receiver import Receiver
from radiolab.dsp.filters import LinearResampler, PolyDDC
from radiolab.dsp.noise import SpectralNR, LineEnhancer, NoiseBlanker, expint_e1

FS = 2.5e6
OFF = 123456.0          # fréquence écoutée, par rapport au centre


def tone(f, n, fs=FS, amp=0.01):
    return (amp * np.exp(2j * np.pi * f * np.arange(n) / fs)).astype(np.complex64)


def noise(n, sigma, seed=0):
    r = np.random.default_rng(seed)
    return (sigma * (r.standard_normal(n) + 1j * r.standard_normal(n)) / np.sqrt(2)).astype(np.complex64)


def run(rx, iq, block=65536):
    return np.concatenate([rx.process(iq[i:i + block]) for i in range(0, len(iq), block)])


def level_at(audio, f, fs=48000, skip=0.3):
    """Niveau (dB) de la raie à la fréquence f, après le régime transitoire."""
    a = audio[int(skip * fs):]
    w = np.hanning(len(a))
    return 20 * np.log10(abs(np.sum(a * w * np.exp(-2j * np.pi * f * np.arange(len(a)) / fs))) / w.sum() * 2 + 1e-12)


def make_rx(mode, **kw):
    rx = Receiver(FS)
    rx.set_mode(mode)
    rx.set_offset(OFF)
    rx.set_agc(kw.get("agc", False))
    rx.volume = 1.0
    return rx


N = int(FS * 1.0)


def test_usb_selects_upper_sideband():
    rx = make_rx("USB")
    up = level_at(run(rx, tone(OFF + 1000, N)), 1000)
    rx = make_rx("USB")
    down = level_at(run(rx, tone(OFF - 1000, N)), 1000)
    assert up > -45 and up - down > 55, (up, down)


def test_lsb_selects_lower_sideband():
    up = level_at(run(make_rx("LSB"), tone(OFF + 1000, N)), 1000)
    down = level_at(run(make_rx("LSB"), tone(OFF - 1000, N)), 1000)
    assert down - up > 55, (up, down)


def test_adjacent_signal_rejected():
    inband = level_at(run(make_rx("USB"), tone(OFF + 1500, N)), 1500)
    for df in (4000, 9000, 30000, 49000, 51500, 400000):
        out = run(make_rx("USB"), tone(OFF + df, N))
        rms = 20 * np.log10(np.sqrt(np.mean(out[14400:] ** 2)) + 1e-12)
        assert inband - rms > 55, (df, inband, rms)


def test_cw_pitch():
    out = run(make_rx("CW"), tone(OFF, N))
    assert level_at(out, 700) - level_at(out, 1400) > 50


def test_am():
    n = np.arange(N)
    iq = tone(OFF, N) * (1 + 0.5 * np.cos(2 * np.pi * 1000 * n / FS)).astype(np.float32)
    out = run(make_rx("AM"), iq)
    assert abs(level_at(out, 1000) - 20 * np.log10(0.005 * 0.5)) < 1.5       # 0,5 : gain fixe sans AGC


def test_nfm():
    n = np.arange(N)
    ph = 2 * np.pi * OFF * n / FS + (3000 / 1000) * np.sin(2 * np.pi * 1000 * n / FS)
    out = run(make_rx("NFM"), (0.01 * np.exp(1j * ph)).astype(np.complex64))
    assert level_at(out, 1000) > -25 and level_at(out, 1000) - level_at(out, 2000) > 30


def test_wfm():
    n = np.arange(N)
    ph = 2 * np.pi * OFF * n / FS + (50e3 / 1000) * np.sin(2 * np.pi * 1000 * n / FS)
    out = run(make_rx("WFM"), (0.01 * np.exp(1j * ph)).astype(np.complex64))
    assert abs(level_at(out, 1000) - 20 * np.log10(50 / 75 * 0.5)) < 1.5
    assert level_at(out, 1000) - level_at(out, 3000) > 40


@pytest.mark.parametrize("mode", ["USB", "NFM", "WFM"])
def test_block_size_does_not_change_output(mode):
    iq = tone(OFF + 1000, N // 2) + noise(N // 2, 0.003)
    a = run(make_rx(mode), iq, block=65536)
    r = np.random.default_rng(1)
    rx, parts, i = make_rx(mode), [], 0
    while i < len(iq):
        k = int(r.integers(1, 40000))
        parts.append(rx.process(iq[i:i + k])); i += k
    b = np.concatenate(parts)
    n = min(len(a), len(b))
    assert abs(len(a) - len(b)) <= 2
    assert np.max(np.abs(a[:n] - b[:n])) < 2e-4 * (np.max(np.abs(a)) + 1e-9) + 1e-6


def test_noise_blanker():
    sig = tone(OFF + 1000, N, amp=0.002) + noise(N, 0.002)
    imp = sig.copy()
    imp[::25000] += 3.0                                   # 100 impulsions par seconde, très fortes
    clean = run(make_rx("USB"), sig)
    dirty = run(make_rx("USB"), imp)
    rx = make_rx("USB"); rx.set_nb(True, 4.0)
    fixed = run(rx, imp)
    n = min(len(clean), len(fixed))
    err = lambda y: 10 * np.log10(np.mean((y[9600:n] - clean[9600:n]) ** 2) + 1e-20)
    assert err(dirty) - err(fixed) > 20, (err(dirty), err(fixed))
    assert rx.nb.blanked > 0
    rx = make_rx("USB"); rx.set_nb(True, 4.0)             # sans impulsion : quasi transparent
    same = run(rx, sig)
    assert rx.nb.blanked < 0.001 * N / 50
    assert err(same) < err(dirty) - 30


def test_auto_notch_removes_carrier():
    fs = 12500.0
    r = np.random.default_rng(3)
    speech_like = 0.1 * r.standard_normal(int(fs * 4))
    x = (speech_like + 1.0 * np.sin(2 * np.pi * 1000 * np.arange(len(speech_like)) / fs)).astype(np.float32)
    anf = LineEnhancer(delay=62, mode="notch"); anf.enabled = True
    y = np.concatenate([anf.process(x[i:i + 500]) for i in range(0, len(x), 500)])
    assert level_at(x, 1000, fs, skip=2.0) - level_at(y, 1000, fs, skip=2.0) > 30
    t0 = int(fs * 2)
    assert abs(10 * np.log10(np.mean(y[t0:] ** 2) / np.mean(speech_like[t0:] ** 2))) < 2.0   # le reste est conservé


def voiced(fs, seconds, seed=5):
    """Signal « type voix » : harmoniques d'un fondamental, par bouffées."""
    t = np.arange(int(fs * seconds)) / fs
    r = np.random.default_rng(seed)
    s = sum(r.uniform(0.3, 1) * np.sin(2 * np.pi * 140 * k * t + r.uniform(0, 6)) for k in range(2, 16))
    env = (np.sin(2 * np.pi * 1.5 * t) > -0.2) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    return (0.1 * s * env).astype(np.float32)


@pytest.mark.parametrize("method,min_gain", [("lsa", 6.5), ("wiener", 6.5), ("specsub", 4.5)])
def test_spectral_nr_improves_snr(method, min_gain):
    fs = 12500.0
    s = voiced(fs, 8)
    nz = np.random.default_rng(7).standard_normal(len(s)).astype(np.float32)
    nz *= np.sqrt(np.mean(s ** 2) / np.mean(nz ** 2)) * 10 ** (-5 / 20)        # SNR d'entrée : 5 dB
    nr = SpectralNR(fs, method=method, reduction_db=18); nr.enabled = True
    y = np.concatenate([nr.process((s + nz)[i:i + 777]) for i in range(0, len(s), 777)])
    d = nr.nfft - nr.hop                                                       # retard du traitement
    ref, y = s[int(fs * 2):len(y) - d], y[int(fs * 2) + d:]
    snr_out = 10 * np.log10(np.mean(ref ** 2) / np.mean((y - ref) ** 2))
    assert snr_out - 5.0 > min_gain, snr_out


def test_spectral_nr_transparent_when_gain_is_one():
    fs = 12500.0
    s = voiced(fs, 2)
    nr = SpectralNR(fs, reduction_db=0.0); nr.enabled = True                   # plancher 0 dB -> gain 1
    y = nr.process(s)
    d = nr.nfft - nr.hop
    assert np.max(np.abs(y[d + 300:] - s[300:len(y) - d])) < 1e-4


def test_lms_nr_improves_snr():
    fs = 12500.0
    s = voiced(fs, 8)
    nz = np.random.default_rng(8).standard_normal(len(s)).astype(np.float32)
    nz *= np.sqrt(np.mean(s ** 2) / np.mean(nz ** 2)) * 10 ** (-5 / 20)
    lms = LineEnhancer(ntaps=128, delay=1, mu=0.3, leak=1e-4, mode="enhance"); lms.enabled = True
    y = lms.process(s + nz)
    a = int(fs * 2)
    g = np.dot(y[a:], s[a:]) / np.dot(s[a:], s[a:])                            # gain appliqué au signal
    snr_out = 10 * np.log10(np.mean((g * s[a:]) ** 2) / np.mean((y[a:] - g * s[a:]) ** 2))
    assert snr_out - 5.0 > 2.5, snr_out


def test_expint():
    exp1 = pytest.importorskip("scipy.special").exp1
    x = np.logspace(-6, 2, 500)
    assert np.max(np.abs(expint_e1(x) / exp1(x) - 1)) < 1e-5


def test_resampler():
    fs = 12500.0
    x = np.sin(2 * np.pi * 1000 * np.arange(int(fs)) / fs).astype(np.float32)
    rs = LinearResampler(fs, 48000)
    y = np.concatenate([rs.process(x[i:i + 333]) for i in range(0, len(x), 333)])
    assert abs(len(y) - 48000) <= 4
    assert abs(level_at(y, 1000, skip=0.1)) < 0.3


def test_realtime_margin():
    iq = tone(OFF + 1000, int(FS * 2)) + noise(int(FS * 2), 0.003)
    rx = make_rx("USB", agc=True)
    rx.set_nb(True); rx.set_nr(True); rx.set_anf(True)
    t = time.perf_counter(); run(rx, iq); dt = time.perf_counter() - t
    print("\nUSB + NB + NR + ANF : %.0f %% d'un coeur à 2,5 MS/s" % (dt / 2 * 100))
    rx = make_rx("WFM")
    t = time.perf_counter(); run(rx, iq); dt2 = time.perf_counter() - t
    print("WFM : %.0f %% d'un coeur à 2,5 MS/s" % (dt2 / 2 * 100))
    assert dt / 2 < 0.9 and dt2 / 2 < 0.9


def harmonic_ratio(out, f=1000):
    """Niveau de l'harmonique 2 par rapport au fondamental, en dB."""
    return level_at(out, 2 * f) - level_at(out, f)


def test_synchronous_am_survives_carrier_fading():
    n = np.arange(N)
    mod = 0.5 * np.cos(2 * np.pi * 1000 * n / FS)
    faded = tone(OFF, N) * (0.2 + mod).astype(np.float32)          # porteuse affaiblie : surmodulation à 250 %
    env = run(make_rx("AM"), faded)
    rx = make_rx("AM"); rx.set_am_sync(True)
    sync = run(rx, faded)
    assert rx.sync_locked
    assert harmonic_ratio(env) > -12                                # l'enveloppe déforme fortement
    assert harmonic_ratio(sync) < -35                               # la détection synchrone reste propre
    normal = tone(OFF, N) * (1 + mod).astype(np.float32)
    rx = make_rx("AM"); rx.set_am_sync(True)
    assert abs(level_at(run(rx, normal), 1000) - level_at(run(make_rx("AM"), normal), 1000)) < 0.5
    rx = make_rx("AM"); rx.set_am_sync(True)                        # sans porteuse : retour à l'enveloppe, sans erreur
    run(rx, noise(N // 2, 0.003)); assert not rx.sync_locked


def test_dsb_and_raw_modes():
    both = tone(OFF + 1000, N) + tone(OFF - 1500, N)
    out = run(make_rx("DSB"), both)
    assert level_at(out, 1000) > -50 and level_at(out, 1500) > -50   # les deux bandes latérales passent
    rx = make_rx("RAW")
    out = np.concatenate([p for p in [rx.process(both[i:i + 65536]) for i in range(0, len(both), 65536)] if len(p)])
    assert out.ndim == 2 and out.shape[1] == 2
    z = out[24000:, 0] + 1j * out[24000:, 1]                         # I à gauche, Q à droite : le signe des fréquences est conservé
    spec = np.abs(np.fft.fft(z * np.hanning(len(z))))
    f = np.fft.fftfreq(len(z), 1 / 48000)
    top = sorted(f[np.argsort(spec)[-2:]])
    assert abs(top[0] + 1500) < 5 and abs(top[1] - 1000) < 5


def test_cw_pitch_is_adjustable():
    rx = make_rx("CW"); rx.set_cw_pitch(500)
    assert (rx.lo, rx.hi) == (250.0, 750.0) and rx.digi_audio_center() == 500
    out = run(rx, tone(OFF, N))
    assert level_at(out, 500) - level_at(out, 700) > 40
    rx.set_mode("USB"); rx.set_mode("CW")
    assert (rx.lo, rx.hi) == (250.0, 750.0)                          # la note choisie est conservée
    rx = make_rx("USB"); rx.set_cw_pitch(5000); assert rx.cw_pitch == 1200
