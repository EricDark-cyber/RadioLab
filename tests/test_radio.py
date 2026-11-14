"""Essais du chef d'orchestre et des fichiers, sans écran ni carte son."""
import time
import wave
import numpy as np
from radiolab.radio import Radio
from radiolab.audio import WavWriter, NullAudio
from radiolab.sources.iqfile import IQFileSource
from radiolab import cli

FS = 250000


def make_iq_file(path, seconds=2.0, tone_hz=21000.0, amp=0.2):
    n = int(FS * seconds)
    iq = (amp * np.exp(2j * np.pi * tone_hz * np.arange(n) / FS)).astype(np.complex64)
    w = WavWriter(str(path), FS, iq=True)
    w.write(iq); w.close()
    return iq


class Capture(NullAudio):
    def __init__(self):
        self.parts = []

    def write(self, audio):
        self.parts.append(audio)


def test_iq_file_roundtrip(tmp_path):
    p = tmp_path / "RadioLab_20260101_000000Z_7100000Hz_IQ.wav"
    iq = make_iq_file(p)
    src = IQFileSource(str(p), loop=False, realtime=False)
    assert src.sample_rate == FS and src.center_hz == 7100000.0
    src.rewind()
    got = []
    while (b := src.read_block()) is not None:
        got.append(b)
    got = np.concatenate(got)
    assert len(got) == len(iq) and np.max(np.abs(got - iq)) < 1e-4


def test_float_wav(tmp_path):
    import struct
    p = tmp_path / "f_14200000Hz.wav"
    x = np.arange(2000, dtype="<f4") / 4000.0
    with open(p, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + x.nbytes) + b"WAVE" + b"fmt " +
                struct.pack("<IHHIIHH", 16, 3, 2, 96000, 96000 * 8, 8, 32) + b"data" + struct.pack("<I", x.nbytes))
        f.write(x.tobytes())
    src = IQFileSource(str(p), loop=False, realtime=False)
    src.rewind()
    assert src.sample_rate == 96000 and np.allclose(src.read_block().view(np.float32), x)


def test_radio_plays_file_and_records(tmp_path):
    p = tmp_path / "RadioLab_20260101_000000Z_7100000Hz_IQ.wav"
    make_iq_file(p, seconds=3.0)                       # porteuse à 7 121 000 Hz
    cap = Capture()
    radio = Radio(audio_factory=lambda: cap)
    radio.set_source(IQFileSource(str(p), loop=False, realtime=True))
    radio.set_mode("USB")
    radio.tune(7_120_000)                              # 1 kHz sous la porteuse -> note de 1 kHz en BLU sup.
    assert radio.passband() == (7_120_100.0, 7_122_800.0)
    radio.rx.set_nr(True); radio.rx.set_nb(True); radio.rx.set_anf(False)
    radio.start()
    rec = radio.start_recording(str(tmp_path), iq=False)
    time.sleep(1.5)
    f, db = radio.spectrum(1024)
    assert abs(f[np.argmax(db)] - 7_121_000) < 2 * FS / 1024 and db.max() > -20
    radio.tune(7_500_000)                              # hors du fichier : ramené dans la bande
    assert radio.vfo_hz <= 7_100_000 + 0.48 * FS
    radio.tune(7_120_000)
    time.sleep(1.0)
    radio.stop()
    assert radio.error is None and radio.lost_blocks == 0
    audio = np.concatenate(cap.parts)
    assert len(audio) > 48000
    seg = audio[24000:60000]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    assert abs(np.argmax(spec) * 48000 / len(seg) - 1000) < 5
    with wave.open(rec) as w:
        assert w.getframerate() == 48000 and w.getnframes() > 24000


def test_mode_switch_while_running(tmp_path):
    p = tmp_path / "x_100000000Hz_IQ.wav"
    make_iq_file(p, seconds=2.0)
    radio = Radio()
    radio.set_source(IQFileSource(str(p), loop=True, realtime=True))
    radio.start()
    for mode in ("AM", "LSB", "CW", "NFM", "WFM", "USB"):
        radio.set_mode(mode); radio.rx.set_filter(radio.rx.lo, radio.rx.hi)
        time.sleep(0.15)
    radio.stop()
    assert radio.error is None


def test_cli(tmp_path):
    p = tmp_path / "RadioLab_20260101_000000Z_7100000Hz_IQ.wav"
    make_iq_file(p, seconds=1.0)
    out = tmp_path / "a.wav"
    assert cli.main([str(p), "--freq", "7120000", "--mode", "USB", "--nr", "--nb", "--anf", "-o", str(out)]) == 0
    with wave.open(str(out)) as w:
        assert abs(w.getnframes() - 48000) < 2000


def test_second_vfo_and_split(tmp_path):
    n = FS * 3
    t = np.arange(n) / FS
    iq = (0.2 * np.exp(2j * np.pi * 21_000 * t) + 0.2 * np.exp(2j * np.pi * -40_500 * t)).astype(np.complex64)
    p = tmp_path / "v_7100000Hz_IQ.wav"
    w = WavWriter(str(p), FS, iq=True); w.write(iq); w.close()
    cap = Capture()
    radio = Radio(audio_factory=lambda: cap)
    radio.set_source(IQFileSource(str(p), loop=False, realtime=True))
    radio.set_mode("USB"); radio.rx.set_agc(False); radio.tune(7_120_000)          # VFO A : note de 1000 Hz
    radio.enable_second(True); radio.rx2.set_mode("USB"); radio.rx2.set_agc(False)
    radio.tune2(7_059_000)                                                         # VFO B : note de 500 Hz
    radio.split_ears = True
    radio.start(); time.sleep(1.5)
    radio.split_ears = False; time.sleep(1.0)
    radio.tune2(9_000_000); assert not radio.rx2_in_band                           # hors bande : VFO B muet, sans erreur
    time.sleep(0.3)
    radio.stop()
    assert radio.error is None
    parts = [p for p in cap.parts if p.ndim == 2 and len(p)]
    audio = np.concatenate(parts)

    def level(x, f):
        w = np.hanning(len(x))
        return 20 * np.log10(abs(np.sum(x * w * np.exp(-2j * np.pi * f * np.arange(len(x)) / 48000))) / w.sum() * 2 + 1e-12)
    seg = audio[24000:60000]                                                       # oreilles séparées
    assert level(seg[:, 0], 1000) - level(seg[:, 0], 500) > 40 and level(seg[:, 1], 500) - level(seg[:, 1], 1000) > 40
    seg = audio[80000:110000]                                                      # mélangés
    assert abs(level(seg[:, 0], 1000) - level(seg[:, 0], 500)) < 3 and np.allclose(seg[:, 0], seg[:, 1])
    radio.enable_second(False); assert radio.rx2 is None


def test_timeshift_replays_older_blocks():
    import threading

    class Src:
        kind, sample_rate, center_hz, sample_rates = "file", 100_000, 7_100_000.0, [100_000]
        def close(self): pass
        def stop(self): pass
    radio = Radio(); radio.set_source(Src()); radio.audio = NullAudio()
    radio.set_timeshift(True, 0.0)
    radio.running = True
    th = threading.Thread(target=radio._work, daemon=True); th.start()

    def feed(values):
        for v in values:
            radio.blocks.put(np.full(10_000, v, dtype=np.complex64))               # un bloc = 0,1 s
        time.sleep(0.4)
    feed(range(1, 11))
    assert radio.last_block[0] == 10 and abs(radio.history_seconds() - 1.0) < 1e-6  # en direct
    radio.set_timeshift(True, 0.5)                                                  # 0,5 s en arrière = 5 blocs
    feed([11]); assert radio.last_block[0] == 6
    feed([12, 13]); assert radio.last_block[0] == 8                                 # le retard reste constant
    radio.set_timeshift(True, 60.0); feed([14]); assert radio.last_block[0] == 1    # plus loin que la mémoire : le plus ancien
    radio.HISTORY_BYTES = 8 * 10_000 * 5                                            # mémoire limitée à 5 blocs
    radio.set_timeshift(True, 0.0); feed([15, 16])
    assert len(radio.history) == 5 and abs(radio.history_seconds() - 0.5) < 1e-6
    radio.tune(7_101_000); assert radio.history_seconds() == 0.5                    # même fréquence centrale : on garde
    radio.set_timeshift(False); assert radio.history_seconds() == 0
    radio.running = False; th.join()
    assert radio.error is None
