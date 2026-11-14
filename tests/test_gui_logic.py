"""Logique de l'interface et liaison ctypes Airspy, avec des doublures (pas d'écran, pas de récepteur)."""
import ctypes
import os
import subprocess
import sys
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fakes"))
import fakeqt
FileDialog, MessageBox = fakeqt.install()
from radiolab import gui                                    # noqa: E402
from radiolab.radio import Radio, load_config               # noqa: E402
from radiolab.sources.airspy import AirspySource            # noqa: E402
from test_radio import make_iq_file                         # noqa: E402


@pytest.fixture(scope="session")
def fake_lib(tmp_path_factory):
    import shutil
    if shutil.which("gcc") is None or sys.platform == "win32":
        pytest.skip("gcc absent : essai de la liaison Airspy ignoré")
    out = tmp_path_factory.mktemp("lib") / "libfakeairspy.so"
    src = os.path.join(os.path.dirname(__file__), "fakes", "fake_airspy.c")
    subprocess.check_call(["gcc", "-shared", "-fPIC", "-O1", "-o", str(out), src, "-lpthread", "-lm"])
    return str(out)


@pytest.fixture
def window(tmp_path):
    MessageBox.messages.clear()
    cfg = load_config(str(tmp_path / "none.json"))
    cfg["record_dir"] = str(tmp_path)
    gui.save_config = lambda cfg, path=None: None           # ne pas écrire dans le dossier personnel
    win = gui.MainWindow(Radio(), cfg)
    yield win
    win.radio.close()


def test_airspy_binding(fake_lib):
    src = AirspySource(fake_lib)
    assert src.sample_rates == [2_500_000, 10_000_000] and src.sample_rate == 2_500_000
    lib = ctypes.CDLL(fake_lib)
    lib.fake_last_freq.restype = ctypes.c_uint32
    src.set_spyverter(True)
    src.set_center(7_100_000)
    assert lib.fake_last_freq() == 127_100_000 and lib.fake_last_bias() == 0     # le décalage seul n'alimente rien
    src.set_bias_tee(True); assert lib.fake_last_bias() == 1
    src.set_spyverter(False); assert lib.fake_last_bias() == 1                   # les deux réglages sont indépendants
    src.set_bias_tee(False); assert lib.fake_last_bias() == 0
    src.set_gain(15); assert lib.fake_last_gain() == 15
    src.set_gain(9, "sensitivity"); assert lib.fake_last_gain() == 109
    src.set_spyverter(False)
    with pytest.raises(Exception):
        src.set_center(7_100_000)                           # la HF sans convertisseur est refusée
    src.set_center(100_000_000)
    blocks = []
    src.start(blocks.append)
    time.sleep(0.3)
    src.stop(); src.close()
    assert blocks and blocks[0].dtype == np.complex64 and len(blocks[0]) == 65536
    spec = np.abs(np.fft.fft(blocks[0][:4096]))
    assert abs(np.argmax(spec) * 2.5e6 / 4096 - 100e3) < 1300 and abs(abs(blocks[0][10]) - 0.1) < 0.01
    assert src.dropped >= 7


def test_tuning_steps(window):
    win = window
    labels = [it[0] for it in win.step.items]
    for wanted in ("6,25 kHz", "12,5 kHz", "25 kHz", "8,33 kHz", "9 kHz", "1 Hz", "1 MHz"):
        assert wanted in labels, labels
    win.step.setCurrentIndex(labels.index("6,25 kHz"))
    win.freq.setValue(145_500_000); win.freq.stepBy(1); assert win.freq.value() == 145_506_250
    win.freq.stepBy(-3); assert win.freq.value() == 145_487_500
    win.freq.setValue(145_501_000); win.freq.stepBy(1); assert win.freq.value() == 145_506_250   # retour sur la grille
    win.freq.setValue(145_501_000); win.freq.stepBy(-1); assert win.freq.value() == 145_500_000
    win.step.setCurrentIndex(labels.index("8,33 kHz"))
    win.freq.setValue(118_000_000)
    seen = []
    for _ in range(3):
        win.freq.stepBy(1); seen.append(win.freq.value())
    assert seen == [118_008_333, 118_016_667, 118_025_000]                                        # pas de dérive
    assert gui.snap(7_103_400, 5000) == 7_105_000 and gui.snap(7_102_400, 5000) == 7_100_000

    class Wheel:                                               # molette sur le spectre ou la cascade
        def __init__(self, delta, ctrl=False): self.delta, self.ctrl, self.accepted = delta, ctrl, False
        def modifiers(self): return type("M", (), {"__and__": lambda s, other: self.ctrl})()
        def angleDelta(self): return type("P", (), {"y": lambda s: self.delta})()
        def accept(self): self.accepted = True
    win.step.setCurrentIndex(labels.index("12,5 kHz")); win.freq.setValue(145_500_000)
    ev = Wheel(120); win.plot.wheelEvent(ev); assert win.freq.value() == 145_512_500 and ev.accepted
    win.wf_plot.wheelEvent(Wheel(-240)); assert win.freq.value() == 145_487_500
    win.plot.wheelEvent(Wheel(60)); assert win.freq.value() == 145_487_500        # demi-cran : rien encore
    win.plot.wheelEvent(Wheel(60)); assert win.freq.value() == 145_500_000
    win.plot.wheelEvent(Wheel(120, ctrl=True)); assert win.freq.value() == 145_500_000   # Ctrl : zoom, pas d'accord
    assert win.cfg["step_hz"] == 12500                                                   # le dernier pas choisi est mémorisé


def test_gui_with_file(window, tmp_path):
    p = tmp_path / "RadioLab_20260101_000000Z_7100000Hz_IQ.wav"
    make_iq_file(p, seconds=2.0)
    FileDialog.path = str(p)
    win = window
    win.open_file()
    r = win.radio
    assert r.running and r.vfo_hz == 7_100_000 and win.freq.value() == 7_100_000
    time.sleep(0.4)
    win.on_timer()
    for mode in ("AM", "USB", "CW", "NFM", "LSB"):
        win.mode_buttons[mode].clicked.emit(False)
        assert r.rx.mode == mode and win.bw.value() == int(r.rx.hi - r.rx.lo)
    win.bw.setValue(1800); assert (r.rx.lo, r.rx.hi) == (-1900.0, -100.0)
    win.freq.setValue(7_120_000); assert r.vfo_hz == 7_120_000
    win.freq.setValue(9_000_000); assert win.freq.value() == int(7_100_000 + 0.48 * 250000)   # borné au fichier
    win.nb.setChecked(True); win.nr.setChecked(True); win.anf.setChecked(True); win.nr_level.setValue(24)
    assert r.rx.nb.enabled and r.rx.nr.enabled and r.rx.anf.enabled and r.rx.nr.reduction_db == 24
    win.nr_method.setCurrentIndex(3); assert r.rx.lms.enabled and not r.rx.nr.enabled
    win.volume.setValue(80); win.squelch.setValue(-100); assert r.rx.volume == 0.8 and r.rx.squelch_db == -100
    win.rec_audio.setChecked(True); time.sleep(0.3); win.rec_audio.setChecked(False)
    win.rec_iq.setChecked(True); time.sleep(0.3); win.rec_iq.setChecked(False)
    assert len([f for f in os.listdir(tmp_path) if f.startswith("RadioLab_") and f.endswith(("AF.wav", "IQ.wav"))]) >= 3
    n = len(win.cfg["memories"]); win.on_mem_add(); assert len(win.cfg["memories"]) == n + 1
    win.mode_buttons["WFM"].clicked.emit(False)             # 250 kHz de bande : la FM large est acceptée
    win.on_recall(1); assert r.rx.mode == "LSB"
    time.sleep(0.2); win.on_timer()
    assert r.error is None and MessageBox.messages == []
    win.closeEvent(fakeqt.Any())
    assert not r.running


def test_gui_with_airspy(window, fake_lib):
    win = window
    win.cfg["dll_path"] = fake_lib
    win.freq.setValue(7_100_000)
    win.mode_buttons["LSB"].clicked.emit(False)
    win.toggle_airspy()
    r = win.radio
    assert MessageBox.messages == [] and r.running and r.source.kind == "airspy"
    assert r.vfo_hz == 7_100_000 and r.center_hz == 7_100_000 - 312_500 and r.source.converter_offset == 120_000_000
    assert r.rx.mode == "LSB"                               # le mode choisi avant le démarrage est appliqué
    time.sleep(0.5); win.on_timer()
    win.freq.setValue(14_200_000)                           # hors de l'écran : la fréquence centrale suit
    assert r.vfo_hz == 14_200_000 and abs(r.center_hz - 14_200_000) < 0.4 * 2.5e6
    win.gain.setValue(5)
    lib = ctypes.CDLL(fake_lib)
    lib.fake_last_freq.restype = ctypes.c_uint32
    before = lib.fake_last_freq()
    assert win.sv_ppm.isEnabled()
    win.sv_ppm.setValue(1.5)                                # +1,5 ppm sur 120 MHz = +180 Hz demandés à l'Airspy
    assert lib.fake_last_freq() - before == 180 and r.vfo_hz == 14_200_000 and win.cfg["sv_ppm"] == 1.5
    win.sv_ppm.setValue(-2.0); assert lib.fake_last_freq() - before == -240
    win.sv_ppm.setValue(0.0); assert lib.fake_last_freq() == before
    assert win.bias_tee.isChecked() and lib.fake_last_bias() == 1               # repris de l'ancienne case unique
    win.bias_tee.setChecked(False)
    assert lib.fake_last_bias() == 0 and r.source.converter_offset == 120_000_000 and r.vfo_hz == 14_200_000
    win.bias_tee.setChecked(True); assert lib.fake_last_bias() == 1
    win.rate.setCurrentIndex(1)                             # passage à 10 MS/s en marche
    assert r.sample_rate == 10_000_000 and r.running and r.vfo_hz == 14_200_000
    assert [it[1] for it in win.decim.items] == [1, 2, 4, 8, 16, 32]
    win.decim.setCurrentIndex(4)                            # 10 MS/s décimé par 16, comme dans SDR#
    assert r.sample_rate == 625_000 and r.running and r.vfo_hz == 14_200_000 and win.cfg["decimation"] == 16
    assert abs(r.center_hz - (14_200_000 - 0.125 * 625_000)) < 1 and win.span_label.text_ == "500 kHz (625 kS/s)"
    time.sleep(0.6); win._spec_next = 0.0; win.on_timer()
    assert r.error is None and r.lost_blocks == 0 and win.resolution_hz == 625_000 / 4096
    win.freq.setValue(14_600_000)                           # sortie de la bande décimée : la fréquence centrale suit
    assert r.vfo_hz == 14_600_000 and abs(r.vfo_hz - r.center_hz) < 0.4 * 625_000
    win.freq.setValue(14_200_000)
    win.rate.setCurrentIndex(0)                             # retour à 2,5 MS/s : la décimation par 16 reste permise
    assert r.source.sample_rate == 2_500_000 and r.decimation == 16 and r.sample_rate == 156_250
    win.decim.setCurrentIndex(0); assert r.sample_rate == 2_500_000 and r.decimator is None
    win.spyverter.setChecked(False)                         # sans convertisseur, 14 MHz est hors plage
    assert MessageBox.messages and "SpyVerter" in MessageBox.messages[-1][0]
    assert lib.fake_last_bias() == 1                        # décocher SpyVerter ne coupe pas le bias-tee
    assert not win.sv_ppm.isEnabled()
    MessageBox.messages.clear()
    win.freq.setValue(100_000_000); win.mode_buttons["WFM"].clicked.emit(False)
    assert r.vfo_hz == 100_000_000 and r.rx.mode == "WFM" and MessageBox.messages == []
    time.sleep(0.5); win.on_timer()
    assert r.error is None and r.lost_blocks == 0
    win.toggle_airspy(); assert not r.running
    win.toggle_airspy(); assert r.running
    win.closeEvent(fakeqt.Any())


def test_gui_scanner_lock_and_info(window, tmp_path):
    from test_fm import fm_station
    from radiolab.audio import WavWriter
    p = tmp_path / "fm_98000000Hz_IQ.wav"
    w = WavWriter(str(p), 1_000_000, iq=True); w.write(fm_station(4.0) * 5); w.close()   # station à 98,150 MHz
    FileDialog.path = str(p)
    win = window
    win.open_file()
    r = win.radio
    win.mode_buttons["WFM"].clicked.emit(False)
    win.freq.setValue(98_150_000)
    deadline = time.time() + 12
    while time.time() < deadline and win.rds_ps.text_ != "TESTFM 1":
        time.sleep(0.25)
        for _ in range(5):
            win.on_timer()
    assert win.rds_ps.text_ == "TESTFM 1" and win.stereo_label.text_ == "Stéréo", (win.rds_ps.text_, r.error)
    assert win.rds_misc.text_.startswith("F123") and win.band_label.text_ == "Radiodiffusion FM"
    assert r.audio_spectrum() is not None and r.error is None
    win.stereo.setChecked(False); time.sleep(0.3); win.update_info(); assert win.stereo_label.text_ == "Mono"
    # scanner : refusé tant que le squelch n'est pas réglé
    win.scan_range.setChecked(True)
    assert not win.scanner.active and not win.scan_range.isChecked() and "squelch" in MessageBox.messages[-1][1]
    MessageBox.messages.clear()
    win.squelch.setValue(-10)
    win.step.setCurrentIndex([it[0] for it in win.step.items].index("50 kHz"))
    win.scan_lo.setValue(97_800_000); win.scan_hi.setValue(98_400_000)
    win.scan_range.setChecked(True)
    assert win.scanner.active and len(win.scanner.channels) == 13 and r.vfo_hz == 97_800_000
    deadline = time.time() + 15
    while time.time() < deadline and not win.scanner.holding:
        time.sleep(0.05); win.on_timer()
    assert win.scanner.holding and abs(r.vfo_hz - 98_150_000) <= 100_000, r.vfo_hz      # arrêt sur la station
    assert win.freq.value() == r.vfo_hz
    win.scan_mem.setChecked(True)                           # l'autre scanner prend la main
    assert not win.scan_range.isChecked() and win.scanner.active and len(win.scanner.channels) == len(win.cfg["memories"])
    win.freq.setValue(98_000_000)                           # un accord manuel arrête le scanner
    assert not win.scanner.active and not win.scan_mem.isChecked() and r.vfo_hz == 98_000_000
    win.lock.setChecked(True)
    win.freq.setValue(98_300_000)
    assert r.vfo_hz == 98_000_000 and win.freq.value() == 98_000_000                      # verrouillée
    win.scan_range.setChecked(True); assert not win.scanner.active
    win.lock.setChecked(False); win.freq.setValue(98_300_000); assert r.vfo_hz == 98_300_000
    win.show_bands.setChecked(False)
    win.closeEvent(fakeqt.Any())


def test_gui_tools(window, tmp_path):
    import datetime, socket
    from test_digital import pocsag_bits, fm, FS as PFS
    from radiolab.audio import WavWriter
    bits = pocsag_bits(1234567, "BONJOUR DE RADIOLAB 73")
    n = int(PFS * (len(bits) / 1200 + 0.6))
    idx = ((np.arange(n) / PFS - 0.3) * 1200).astype(np.int64)
    level = np.where((idx >= 0) & (idx < len(bits)), 1.0 - 2.0 * bits[np.clip(idx, 0, len(bits) - 1)], 0.0)
    p = tmp_path / "pocsag_466000000Hz_IQ.wav"
    w = WavWriter(str(p), int(PFS), iq=True); w.write(fm(np.convolve(level, np.ones(50) / 50, "same"), dev=4.5e3) * 5); w.close()
    FileDialog.path = str(p)
    win = window
    win.pocsag.setChecked(True)                               # coché avant même d'avoir une source
    win.open_file()
    r = win.radio
    win.mode_buttons["NFM"].clicked.emit(False)
    win.freq.setValue(466_020_000)
    assert r.rx.pocsag_enabled
    deadline = time.time() + 12
    while time.time() < deadline and win.pocsag_list.count() == 0:
        time.sleep(0.2); win.update_info()
    assert win.pocsag_list.count() >= 1 and "BONJOUR DE RADIOLAB 73" in win.pocsag_list.rows[0] and "1234567" in win.pocsag_list.rows[0]
    win.on_pocsag_clear(); assert win.pocsag_list.count() == 0
    # filtres audio
    fakeqt.InputDialog.number = 1200
    win.on_notch_add(); win.bass.setValue(6)
    assert r.rx.notches == [(1200, 100)] and r.rx.bass_db == 6 and r.rx.fx.active
    win.on_notch_del(); win.bass.setValue(0); assert not r.rx.fx.active
    win.raw.setChecked(True); assert r.rx.raw_output and r.rx.fs2 == 25000.0
    win.raw.setChecked(False)
    win.audio_device.currentIndexChanged.emit(0); assert r.running and r.error is None
    # commande réseau (port libre choisi par le système pour l'essai)
    win.net_port = 0
    win.net_box.setChecked(True)
    s = socket.create_connection(("127.0.0.1", win.net.port), timeout=2); f = s.makefile("rw", newline="\n")
    f.write("f\n"); f.flush(); assert f.readline().strip() == "466020000"
    f.write("F 466050000\nM AM 0\n"); f.flush(); assert f.readline().strip() == "RPRT 0" and f.readline().strip() == "RPRT 0"
    time.sleep(0.1); win.on_timer()
    assert r.vfo_hz == 466_050_000 and win.freq.value() == 466_050_000 and r.rx.mode == "AM" and win.mode_buttons["AM"].isChecked()
    s.close(); win.net_box.setChecked(False); assert win.net is None
    # mémoires CSV
    FileDialog.save_path = str(tmp_path / "mem.csv"); win.on_mem_export()
    n_mem = len(win.cfg["memories"]); FileDialog.path = FileDialog.save_path; win.on_mem_import()
    assert len(win.cfg["memories"]) == 2 * n_mem
    # enregistrement programmé
    now = datetime.datetime.now()
    fakeqt.InputDialog.text = "%s;0.02;466020000;NFM;audio" % now.strftime("%Y-%m-%d %H:%M")
    win.on_job_add(); assert len(win.cfg["jobs"]) == 1 and win.job_list.count() == 1 and MessageBox.messages == []
    win.run_scheduler(now.replace(second=0, microsecond=0) + datetime.timedelta(seconds=0.5))
    assert r.audio_rec is not None and r.rx.mode == "NFM" and r.vfo_hz == 466_020_000
    time.sleep(0.3)
    win.run_scheduler(now.replace(second=0) + datetime.timedelta(seconds=5))
    assert r.audio_rec is None and win.cfg["jobs"] == []
    assert any(f.endswith("AF.wav") for f in os.listdir(tmp_path))
    fakeqt.InputDialog.text = "n'importe quoi"; win.on_job_add(); assert MessageBox.messages and win.cfg["jobs"] == []
    MessageBox.messages.clear()
    # journal d'activité du scanner
    win.log_hits.setChecked(True); win.squelch.setValue(-40)
    win.scan_lo.setValue(466_000_000); win.scan_hi.setValue(466_040_000)
    win.step.setCurrentIndex([it[0] for it in win.step.items].index("20 kHz"))
    win.scan_range.setChecked(True)
    deadline = time.time() + 10
    while time.time() < deadline and not any("journal" in f for f in os.listdir(tmp_path)):
        time.sleep(0.05); win.on_timer()
    log = [f for f in os.listdir(tmp_path) if "journal" in f]
    assert log and ";NFM;" in open(tmp_path / log[0], encoding="utf-8-sig").read()
    assert r.error is None and MessageBox.messages == []
    win.closeEvent(fakeqt.Any())


def test_gui_hf_digital(window, tmp_path):
    from test_hfdigi import psk_signal, add_noise, FS as HFS, OFF
    from radiolab.audio import WavWriter
    p = tmp_path / "psk_14070000Hz_IQ.wav"
    w = WavWriter(str(p), int(HFS), iq=True); w.write(add_noise(psk_signal("CQ CQ de F4TEST pse k", 31.25, 1000.0), 20) * 10); w.close()
    FileDialog.path = str(p)
    win = window
    win.digi_kind.setCurrentIndex(3)                           # PSK31 choisi avant d'ouvrir la source
    win.open_file()
    r = win.radio
    win.mode_buttons["USB"].clicked.emit(False)
    assert r.rx.digi_kind == "PSK31" and r.rx.digi is not None
    signal_hz = 14_070_000 + OFF + 1000                         # là où le signal apparaît sur la cascade
    assert win.signal_to_vfo(signal_hz) == 14_070_000 + OFF
    win.freq.setValue(int(win.signal_to_vfo(signal_hz)))
    deadline = time.time() + 20
    while time.time() < deadline and "CQ CQ de F4TEST pse k" not in win.digi_out.text_:
        time.sleep(0.2); win.update_info()
    assert "CQ CQ de F4TEST pse k" in win.digi_out.text_ and "Qualité" in win.digi_status.text_
    win.digi_center.setValue(1500); assert r.rx.digi_center == 1500 and win.signal_to_vfo(signal_hz) == signal_hz - 1500
    win.mode_buttons["LSB"].clicked.emit(False); assert win.signal_to_vfo(signal_hz) == signal_hz + 1500
    win.digi_kind.setCurrentIndex(5); win.mode_buttons["CW"].clicked.emit(False)
    assert r.rx.digi_kind == "CW" and r.rx.digi_audio_center() == 700.0 and win.signal_to_vfo(signal_hz) is None
    win.mode_buttons["NFM"].clicked.emit(False); win.update_info()
    assert r.rx.digi is None and "USB" in win.digi_status.text_
    assert win.digi_out.text_ != ""
    win.digi_clear.click(); assert win.digi_out.text_ == "" and win._digi_buf == ""        # bouton du bandeau
    win._digi_buf = "abc"; win.digi_out.setPlainText("abc"); r.rx.digi_text.append("x")
    win.digi_clear2.click(); win.update_info()                                             # bouton de l'onglet
    assert win.digi_out.text_ == "" and len(r.rx.digi_text) == 0
    win.mode_buttons["CW"].clicked.emit(False)
    win.cw_auto_speed.setChecked(False)
    assert win.cw_wpm.isEnabled() and r.rx.cw_wpm == win.cw_wpm.value() and not r.rx.digi.auto_speed
    win.cw_wpm.setValue(25); assert r.rx.digi.wpm() == 25
    win.cw_auto_thr.setChecked(False); win.cw_thr.setValue(-70)
    assert r.rx.cw_threshold_db == -70.0 and r.rx.digi.manual_thr is not None
    win.cw_auto_speed.setChecked(True); win.cw_auto_thr.setChecked(True)
    assert r.rx.cw_wpm is None and r.rx.cw_threshold_db is None and r.rx.digi.auto_speed
    time.sleep(0.5); win.update_info()
    win.digi_kind.setCurrentIndex(0); assert r.rx.digi_kind is None
    assert r.error is None and MessageBox.messages == []
    win.closeEvent(fakeqt.Any())


def test_gui_zoom_spectrum(window, tmp_path):
    from radiolab.audio import WavWriter
    fs = 1_000_000
    n = fs * 3
    tt = np.arange(n) / fs
    iq = (0.2 * np.exp(2j * np.pi * 100_000 * tt) + 0.2 * np.exp(2j * np.pi * 100_150 * tt)).astype(np.complex64)
    p = tmp_path / "z_14000000Hz_IQ.wav"
    w = WavWriter(str(p), fs, iq=True); w.write(iq); w.close()
    FileDialog.path = str(p)
    win = window
    data = {}

    class Curve:
        def setData(self, f, d): data.update(f=f, d=d)
    win.curve = Curve()
    win.open_file()
    vb = win.plot.getViewBox()

    class View:                                                  # accès commode à la bande affichée
        def __getitem__(self, key): return list(vb.range)
        def __setitem__(self, key, value): vb.range = list(value)
    view = View()
    assert view["range"] == [14_000_000 - fs / 2, 14_000_000 + fs / 2]
    r = win.radio
    time.sleep(1.5)

    def draw():
        win._spec_next = 0.0
        win.on_timer()
    draw()
    assert len(data["f"]) == gui.WF_COLS and win.wf_range == tuple(view["range"]) and win.resolution_hz == fs / 4096
    col = int((14_100_000 - view["range"][0]) / fs * gui.WF_COLS)
    assert win.wf[0, col - 2:col + 3].max() > -20 and win.wf[0, 100] < -60
    view["range"] = [14_099_000.0, 14_101_000.0]                # zoom sur 2 kHz : la finesse suit
    draw()
    assert win.resolution_hz < 2.0 and win.wf_range == (14_099_000.0, 14_101_000.0)
    f, d = data["f"], data["d"]
    strong = d > d.max() - 20
    assert int(np.sum(np.diff(strong.astype(int)) == 1) + strong[0]) == 2          # les deux porteuses sont séparées
    assert win.wf[1].max() > -20 and win.wf.shape == (gui.WF_ROWS, gui.WF_COLS)    # l'ancienne ligne a été recalée
    # échelle du spectre, niveaux de la cascade et curseur de zoom
    seen = {}

    class Img(fakeqt.Any):
        def setImage(self, data, autoLevels=False, levels=None): seen["levels"] = levels
    win.img = Img()
    yr = {}
    win.plot.setYRange = lambda a, b, padding=0: yr.update(lo=a, hi=b)
    win.offset_db.setValue(-20); win.range_db.setValue(90)
    assert yr == {"lo": -110.0, "hi": -20.0} and win.cfg["range_db"] == 90
    for _ in range(40): draw()
    lo, hi = seen["levels"]
    assert win.wf_auto.isChecked() and abs(hi - lo - 60) < 1e-6 and abs(lo + 5 - win._floor_db) < 1e-6
    win.contrast.setValue(100); draw(); assert abs(seen["levels"][1] - seen["levels"][0] - 20) < 1e-6
    win.contrast.setValue(50)
    win.wf_auto.setChecked(False)                               # manuel : on repart des niveaux automatiques
    assert win.wf_min.isEnabled() and abs(win.wf_min.value() - lo) <= 1.5 and win.wf_max.value() - win.wf_min.value() == 60
    win.wf_min.setValue(-120); win.wf_max.setValue(-40); draw()
    assert seen["levels"] == (-120.0, -40.0) and win.cfg["wf_min"] == -120
    win.wf_max.setValue(-130); draw(); assert seen["levels"] == (-120.0, -115.0)   # max sous le min : écart minimal garanti
    win.wf_auto.setChecked(True)
    win.freq.setValue(14_100_000)
    win.zoom.setValue(16)                                       # 16 crans : vue 4 fois plus serrée, centrée sur la fréquence
    assert view["range"] == [14_100_000 - fs / 8, 14_100_000 + fs / 8]
    win.zoom.setValue(0); assert view["range"] == [14_000_000 - fs / 2, 14_000_000 + fs / 2]
    view["range"] = [14_099_000.0, 14_101_000.0]

    # fenêtres d'analyse : spectre FI (autour de la fréquence écoutée) et spectre audio
    got = {}

    class Grab:
        def __init__(self, key): self.key = key
        def setData(self, x, y): got[self.key] = (np.asarray(x), np.asarray(y))
    win.if_curve, win.af_curve = Grab("if"), Grab("af")
    win.mode_buttons["USB"].clicked.emit(False); win.freq.setValue(14_099_000)     # porteuses à +1000 et +1150 Hz
    time.sleep(0.5)
    win.update_analysis(); assert got == {}                                        # cases décochées : aucun calcul
    win.show_if.setChecked(True); win.show_af.setChecked(True)
    assert win.cfg["show_if"] and win.cfg["show_af"]
    win.update_analysis()
    x, y = got["if"]
    assert x[0] < -4000 and x[-1] > 4000 and abs(x[np.argmax(y)] - 1075) < 120      # axe relatif à la fréquence écoutée
    x, y = got["af"]
    assert x[-1] <= 4000 and abs(x[np.argmax(y)] - 1075) < 120                      # la même chose, vue côté audio
    win.show_if.setChecked(False); win.show_af.setChecked(False)
    win.fft_size.setCurrentIndex(1); draw()                      # finesse imposée : 4096 points
    assert win.resolution_hz == fs / 4096
    win.fft_window.setCurrentIndex(win.fft_window.findData("Hamming")); draw()
    assert win.cfg["fft_window"] == "Hamming" and r.error is None
    win.fft_size.setCurrentIndex(0)
    view["range"] = [14_099_000.0, 14_101_000.0]
    win.freq.setValue(14_100_000)                                # accord au milieu de la vue : rien ne bouge
    draw(); assert view["range"] == [14_099_000.0, 14_101_000.0]
    win.freq.setValue(14_100_950); draw()                        # près du bord droit : la vue glisse à droite
    assert view["range"] == [14_099_150.0, 14_101_150.0]
    win.freq.setValue(14_103_000); draw()                        # sortie de l'écran : la vue suit, même largeur
    assert view["range"] == [14_101_200.0, 14_103_200.0]
    win.freq.setValue(14_098_000); draw()                        # vers le bas : la vue glisse à gauche
    assert view["range"] == [14_097_800.0, 14_099_800.0]
    view["range"] = [14_200_000.0, 14_202_000.0]; draw()         # vue déplacée à la main : elle n'est pas ramenée
    assert view["range"] == [14_200_000.0, 14_202_000.0]
    win.freq.setValue(14_499_900); draw()                        # fichier : l'accord est borné à 14,480 MHz, la vue suit
    assert r.vfo_hz == 14_480_000 and view["range"] == [14_478_200.0, 14_480_200.0]
    win.closeEvent(fakeqt.Any())


def test_gui_radio_options(window, tmp_path, fake_lib):
    win = window
    win.cfg["dll_path"] = fake_lib
    lib = ctypes.CDLL(fake_lib); lib.fake_last_freq.restype = ctypes.c_uint32
    win.spyverter.setChecked(False)
    win.shift_hz.setValue(-125_000_000); win.shift_on.setChecked(True)      # convertisseur HF à 125 MHz
    win.freq.setValue(7_100_000); win.mode_buttons["LSB"].clicked.emit(False)
    win.toggle_airspy()
    r = win.radio
    assert MessageBox.messages == [] and r.running and r.vfo_hz == 7_100_000
    assert lib.fake_last_freq() == round(r.center_hz) + 125_000_000
    win.shift_on.setChecked(False); assert MessageBox.messages and "Décalage" in MessageBox.messages[-1][0]   # 7 MHz hors plage sans décalage
    MessageBox.messages.clear()
    win.shift_on.setChecked(True); assert lib.fake_last_freq() == round(r.center_hz) + 125_000_000
    win.invert.setChecked(True); win.dc_fix.setChecked(True)
    assert r.invert_iq and r.remove_dc and win.cfg["invert_iq"] and win.cfg["remove_dc"]
    time.sleep(0.5)
    f, db, _ = r.spectrum_view(r.center_hz - 1.25e6, r.center_hz + 1.25e6, nfft=4096)
    assert abs(f[np.argmax(db)] - (r.center_hz - 100_000)) < 2000              # la porteuse de +100 kHz passe à -100 kHz
    win.invert.setChecked(False)
    # modes et options d'écoute
    for mode in ("DSB", "RAW", "AM"):
        win.mode_buttons[mode].clicked.emit(False); time.sleep(0.2)
        assert r.rx.mode == mode and r.error is None
    win.am_sync.setChecked(True); assert r.rx.am_sync and win.cfg["am_sync"]
    win.mode_buttons["CW"].clicked.emit(False)
    win.cw_pitch.setValue(500)
    assert r.rx.cw_pitch == 500 and (r.rx.lo, r.rx.hi) == (250.0, 750.0) and r.passband() == (r.vfo_hz - 250.0, r.vfo_hz + 250.0)
    win.bw.setValue(200); assert (r.rx.lo, r.rx.hi) == (400.0, 600.0)
    # calage sur la grille
    win.step.setCurrentIndex([it[0] for it in win.step.items].index("5 kHz"))
    win.freq.setValue(7_101_300); win.freq.stepBy(1); assert win.freq.value() == 7_105_000
    win.snap_box.setChecked(False)
    win.freq.setValue(7_101_300); win.freq.stepBy(1); assert win.freq.value() == 7_106_300
    win.plot.wheelEvent(type("W", (), {"modifiers": lambda s: type("M", (), {"__and__": lambda a, b: False})(),
                                       "angleDelta": lambda s: type("P", (), {"y": lambda q: -120})(), "accept": lambda s: None})())
    assert win.freq.value() == 7_101_300
    # groupes de mémoires
    win.cfg["memories"][:] = [{"name": "A", "hz": 7_100_000, "mode": "LSB", "group": "HF"},
                              {"name": "B", "hz": 14_200_000, "mode": "USB", "group": "HF"},
                              {"name": "C", "hz": 145_500_000, "mode": "NFM", "group": "VHF"},
                              {"name": "D", "hz": 98_000_000, "mode": "WFM"}]
    win.refresh_memories()
    assert [it[0] for it in win.mem_group.items] == ["Tous les groupes", "HF", "VHF"] and len(win.mem.rows) == 4
    win.mem_group.setCurrentIndex(2); assert win.mem.rows == ["C — 145 500 000 Hz — NFM [VHF]"]
    win.on_recall(0); assert r.vfo_hz == 145_500_000 and r.rx.mode == "NFM"
    win.mem._row = 0; win.on_mem_del()
    assert [m["name"] for m in win.cfg["memories"]] == ["A", "B", "D"] and [it[0] for it in win.mem_group.items] == ["Tous les groupes", "HF"]
    win.mem_group.setCurrentIndex(1); win.squelch.setValue(-50); win.scan_mem.setChecked(True)
    assert win.scanner.active and len(win.scanner.channels) == 2               # le scanner ne parcourt que le groupe affiché
    win.stop_scan()
    assert r.error is None and MessageBox.messages == []
    win.closeEvent(fakeqt.Any())


def test_gui_small_and_medium_additions(window, tmp_path):
    from radiolab.audio import WavWriter
    fs = 250_000
    tt = np.arange(fs * 4) / fs
    iq = (0.2 * np.exp(2j * np.pi * 21_000 * tt) + 0.2 * np.exp(2j * np.pi * -40_500 * tt)).astype(np.complex64)
    p = tmp_path / "s_7100000Hz_IQ.wav"
    w = WavWriter(str(p), fs, iq=True); w.write(iq); w.close()
    FileDialog.path = str(p)
    win = window
    win.open_file()
    r = win.radio
    win.mode_buttons["USB"].clicked.emit(False)
    assert (win.edge_lo.value(), win.edge_hi.value(), win.bw.value()) == (100, 2800, 2700)
    win.edge_lo.setValue(300); win.edge_hi.setValue(2400)             # filtre asymétrique
    assert (r.rx.lo, r.rx.hi) == (300.0, 2400.0) and win.bw.value() == 2100
    win.edge_hi.setValue(310); assert (r.rx.lo, r.rx.hi) == (300.0, 2400.0) and win.edge_hi.value() == 2400   # trop étroit : refusé
    win.bw.setValue(2000); assert (win.edge_lo.value(), win.edge_hi.value()) == (100, 2100)
    win.filter_window.setCurrentIndex(1); win.filter_order.setValue(501)
    assert r.rx.filter_window == "Blackman-Harris" and r.rx.filter_order == 501 and len(r.rx.chan.taps) == 501
    win.filter_order.setValue(0); assert r.rx.filter_order is None
    win.binaural.setChecked(True); assert r.rx.binaural and win.cfg["binaural"]
    win.binaural.setChecked(False)
    assert not win.anti_fading.isEnabled()
    win.am_sync.setChecked(True); win.anti_fading.setChecked(True)
    assert win.anti_fading.isEnabled() and r.rx.am_sync and r.rx.anti_fading
    # second récepteur
    win.freq.setValue(7_120_000)
    win.vfo2_freq.setValue(7_059_000); win.vfo2_on.setChecked(True)
    assert r.rx2 is not None and r.vfo2_hz == 7_059_000 and r.rx2.mode == "USB" and r.rx2_in_band
    win.vfo2_mode.setCurrentIndex(1); win.vfo2_split.setChecked(True); win.vfo2_volume.setValue(30)
    assert r.rx2.mode == "LSB" and r.split_ears and r.rx2.volume == 0.3
    win.on_vfo2_copy(); assert r.vfo2_hz == 7_120_000 and r.rx2.mode == "USB"
    time.sleep(0.4); win.update_info(); assert r.error is None
    win.vfo2_on.setChecked(False); assert r.rx2 is None
    # TimeShift
    win.ts_on.setChecked(True); assert r.timeshift
    time.sleep(1.3); win.update_info()
    assert win.ts_delay._hi >= 1 and "en mémoire" in win.ts_label.text_ and win.ts_label.text_.startswith("En direct")
    win.ts_delay.setValue(1); win.update_info()
    assert r.delay_s == 1.0 and win.ts_label.text_.startswith("1 s en arrière")
    win.ts_live.click(); assert r.delay_s == 0.0
    win.ts_on.setChecked(False); assert not r.timeshift and r.history_seconds() == 0 and win.ts_label.text_ == "En direct"
    # packet : choix du décodeur selon le mode
    kinds = [it[1] for it in win.digi_kind.items]
    win.digi_kind.setCurrentIndex(kinds.index("APRS1200")); win.update_info()
    assert r.rx.digi is None and "NFM" in win.digi_status.text_ and win.signal_to_vfo(7_100_000) is None
    win.mode_buttons["NFM"].clicked.emit(False); win.update_info()
    assert r.rx.digi is not None and "Trames valides" in win.digi_status.text_ and win.signal_to_vfo(7_100_000) is None
    win.digi_kind.setCurrentIndex(kinds.index("PACKET300")); win.mode_buttons["USB"].clicked.emit(False)
    assert r.rx.digi is not None and win.signal_to_vfo(7_101_700) == 7_101_700 - win.digi_center.value()
    assert r.error is None and MessageBox.messages == []
    win.closeEvent(fakeqt.Any())


def test_gui_tetra_mode(window, tmp_path):
    from test_tetra import tetra_bits, modulate, FS as TFS, OFF
    from radiolab.audio import WavWriter
    bits, _ = tetra_bits(250)
    p = tmp_path / "t_390000000Hz_IQ.wav"
    w = WavWriter(str(p), int(TFS), iq=True); w.write(modulate(bits, OFF + 150.0, 22.0) * 5); w.close()
    FileDialog.path = str(p)
    win = window
    win.open_file()
    r = win.radio
    win.mode_buttons["TETRA"].clicked.emit(False)
    win.freq.setValue(int(390_000_000 + OFF))
    assert r.rx.mode == "TETRA" and r.rx.digi is not None
    deadline = time.time() + 15
    while time.time() < deadline and "Porteuse TETRA" not in win.digi_status.text_:
        time.sleep(0.3); win.update_info()
    assert "Porteuse TETRA" in win.digi_status.text_ and "Porteuse TETRA" in win.digi_out.text_, win.digi_status.text_
    assert abs(r.rx.digi.freq_error_hz - 150) < 40
    win.mode_buttons["USB"].clicked.emit(False); win.update_info()
    assert r.rx.digi is None and r.error is None and MessageBox.messages == []
    win.closeEvent(fakeqt.Any())


def test_new_layout(window, tmp_path):
    """Refonte 0.13 : grand afficheur chiffre par chiffre, coupure du son, sections repliables."""
    win = window
    p = tmp_path / "RadioLab_20260101_000000Z_7100000Hz_IQ.wav"
    make_iq_file(p, seconds=2.0)
    FileDialog.path = str(p); win.open_file()
    r = win.radio
    win.snap_box.setChecked(False)
    start = win.freq.value()
    win.freq_digits.labels[-4].on_delta(1000)               # chiffre des kilohertz : +1 kHz
    assert win.freq.value() == start + 1000 and r.vfo_hz == start + 1000
    assert win.freq_digits.text == "%010d" % (start + 1000)
    win.on_digit(-1000); assert win.freq.value() == start
    win.lock.setChecked(True); win.on_digit(1000); assert win.freq.value() == start      # verrou respecté
    win.lock.setChecked(False)
    win.volume.setValue(60); assert r.rx.volume == 0.6
    win.mute.setChecked(True); assert r.rx.volume == 0.0 and win.cfg["volume"] == 0.6     # le réglage est conservé
    win.mute.setChecked(False); assert r.rx.volume == 0.6
    heads = win.panels.headers
    assert heads["Source"].isChecked() and heads["Radio"].isChecked() and not heads["Scanner"].isChecked()
    heads["Scanner"].setChecked(True); assert win.cfg["panels"]["Scanner"] is True
    heads["Radio"].setChecked(False); assert win.cfg["panels"]["Radio"] is False
    win.btn_side.setChecked(False); assert win.cfg["side_visible"] is False
    import json; json.dumps(win.cfg["panels"])                # l'état des sections s'enregistre dans la configuration


def test_s_units():
    assert gui.s_units(-73.0)[0] == "S9" and gui.s_units(-79.0)[0] == "S8" and gui.s_units(-121.0)[0] == "S1"
    assert gui.s_units(-53.0)[0] == "S9+20 dB" and gui.s_units(-150.0) == ("S0", 0.0)
    assert gui.s_units(-93.0, vhf=True)[0] == "S9" and gui.s_units(-13.0)[1] == 1.0
    assert 0.4 < gui.s_units(-73.0)[1] < 0.5                  # S9 un peu avant le milieu de la barre


def test_comfort(window, tmp_path):
    """0.14 : filtre à la souris, clavier, étiquettes de mémoires, crêtes et moyennage, lecteur, charge."""
    win = window
    p = tmp_path / "RadioLab_20260101_000000Z_7100000Hz_IQ.wav"
    make_iq_file(p, seconds=4.0)
    FileDialog.path = str(p); win.open_file()
    r = win.radio
    win.snap_box.setChecked(False)
    win.mode_buttons["USB"].clicked.emit(False)
    win.freq.setValue(7_110_000)
    # --- zone du filtre tirée à la souris
    lo, hi = r.passband()
    win.on_band_dragged(lo, hi + 400.0)                       # bord droit : filtre plus large
    assert abs(r.rx.hi - 3200.0) < 1 and abs(r.rx.lo - 100.0) < 1 and win.edge_hi.value() == 3200 and win.bw.value() == 3100
    win.on_band_dragged(lo + 200.0, hi + 400.0)               # bord gauche
    assert abs(r.rx.lo - 300.0) < 1 and r.vfo_hz == 7_110_000
    lo, hi = r.passband()
    win.on_band_dragged(lo + 1500.0, hi + 1500.0)             # zone entière : la fréquence suit, le filtre ne change pas
    assert r.vfo_hz == 7_111_500 and abs(r.rx.lo - 300.0) < 1 and abs(r.rx.hi - 3200.0) < 1
    win.mode_buttons["CW"].clicked.emit(False)
    lo, hi = r.passband()
    win.on_band_dragged(lo - 100.0, hi + 100.0)               # en CW la zone est centrée sur la fréquence affichée
    assert abs((r.rx.hi - r.rx.lo) - (hi - lo + 200.0)) < 1 and abs((r.rx.lo + r.rx.hi) / 2 - r.rx.cw_pitch) < 1
    win._band_set = True; win.on_band_signal(); win._band_set = False     # déplacement fait par le programme : ignoré
    # --- clavier
    win.mode_buttons["USB"].clicked.emit(False); win.step.setCurrentIndex(gui.STEP_VALUES.index(1000))
    win.freq.setValue(7_100_000)
    win.key_actions["Right"](); assert r.vfo_hz == 7_101_000
    win.key_actions["Ctrl+Left"](); assert r.vfo_hz == 7_091_000
    win.key_actions["PgUp"](); assert r.rx.mode == "WFM" or MessageBox.messages      # mode suivant (refusé si la bande est trop étroite)
    MessageBox.messages.clear()
    win.mode_buttons["USB"].clicked.emit(False)
    win.key_actions["PgDown"](); assert r.rx.mode == "LSB" and win.mode_buttons["LSB"].isChecked()
    win.key_actions["Space"](); assert win.mute.isChecked() and r.rx.volume == 0.0
    win.key_actions["Space"](); assert not win.mute.isChecked()
    win.key_actions["L"](); assert win.lock.isChecked()
    win.key_actions["L"]()
    win.key_actions["F1"](); assert MessageBox.messages[-1][0] == "Raccourcis clavier" and "Espace" in MessageBox.messages[-1][1]
    MessageBox.messages.clear()
    assert [k for k, _ in gui.SHORTCUTS] == list(win.key_actions)
    # --- étiquettes des mémoires
    win.cfg["memories"] = [{"name": "A", "hz": 7_050_000, "mode": "LSB"}, {"name": "B", "hz": 7_120_000, "mode": "LSB"},
                           {"name": "Loin", "hz": 14_000_000, "mode": "USB"}]
    assert win.visible_memories(7_000_000, 7_200_000) == [(7_050_000.0, "A"), (7_120_000.0, "B")]
    win.cfg["memories"] = [{"name": str(i), "hz": 7_000_000 + i, "mode": "LSB"} for i in range(100)]
    assert len(win.visible_memories(0, 8e6)) == gui.MEM_LABELS
    win.update_mem_labels(); win.show_mem.setChecked(False); win.update_mem_labels(); assert win.cfg["show_mem"] is False
    # --- crêtes et moyennage dans la boucle d'affichage, S-mètre, charge
    win.peak_hold.setChecked(True); win.avg_n.setValue(8)
    time.sleep(0.6)
    for _ in range(3):
        win._spec_next = 0.0; win.on_timer()
    assert win.trace.peak is not None and win.trace.avg is not None and win.cfg["avg_n"] == 8
    assert np.all(win.trace.peak >= 10 * np.log10(win.trace.avg + 1e-30) - 1e-6)
    win.peak_hold.setChecked(False); win._spec_next = 0.0; win.on_timer(); assert win.trace.peak is None
    assert win.s_text.text_.startswith("S") and "dBm" in win.smeter.text_
    assert 0.0 < r.load < 5.0
    # --- lecteur : pause, reprise, déplacement
    src = r.source
    assert abs(src.duration_s - 4.0) < 0.01 and win.is_file()
    win.key_actions["P"](); assert src.paused and win.play_pause.text_ == "▶"
    time.sleep(0.3); a = src.position(); time.sleep(0.3); assert src.position() == a       # à l'arrêt
    win.play_pos.setValue(250); time.sleep(0.05)
    assert abs(src.position() - 0.25) < 0.01                                               # déplacé pendant la pause
    win.play_pause.setChecked(False); time.sleep(0.5)
    assert not src.paused and 0.3 < src.position() < 0.5                                   # la lecture a repris de là
    win.update_player(); assert win.play_time.text_.endswith("/ 0:04") and abs(win.play_pos.value() - 1000 * src.position()) < 60
    win.play_pos.setValue(900); time.sleep(0.1); assert src.position() > 0.89


def test_calibration(window, fake_lib, tmp_path):
    win = window
    p = tmp_path / "RadioLab_20260101_000000Z_7100000Hz_IQ.wav"
    make_iq_file(p, seconds=1.0)
    FileDialog.path = str(p); win.open_file()
    win.on_calibrate()                                        # sur un fichier : refusé, avec une explication
    assert MessageBox.messages and "SpyVerter" in MessageBox.messages[-1][1]
    MessageBox.messages.clear()
    win.radio.stop()
    win.cfg["dll_path"] = fake_lib
    win.freq.setValue(7_100_000); win.toggle_airspy()
    r = win.radio
    assert r.source.kind == "airspy" and MessageBox.messages == []
    time.sleep(1.2)                                           # le faux Airspy émet une porteuse à 100 kHz du centre
    carrier = r.center_hz + 100_000
    found = r.measure_carrier(carrier + 700.0)
    assert found is not None and abs(found[0] - carrier) < 1.0 and found[1] > 40
    fakeqt.InputDialog.number = int(carrier) - 120            # « la vraie fréquence est 120 Hz plus bas » : +1 ppm
    win.on_calibrate()
    assert MessageBox.messages == [] and abs(win.sv_ppm.value() - 1.0) < 0.02 and abs(r.source.sv_ppm - 1.0) < 0.02
    fakeqt.InputDialog.number = int(carrier) + 20_000         # rien à cet endroit
    win.on_calibrate(); assert "Aucune porteuse" in MessageBox.messages[-1][1] and abs(win.sv_ppm.value() - 1.0) < 0.02
    MessageBox.messages.clear()
    fakeqt.InputDialog.number = 1000


def test_pictures_in_gui(window, tmp_path):
    """Images : les lignes du décodeur s'accumulent, une image terminée est enregistrée, une nouvelle efface l'ancienne."""
    win = window
    p = tmp_path / "RadioLab_20260101_000000Z_7100000Hz_IQ.wav"
    make_iq_file(p, seconds=1.0)
    FileDialog.path = str(p); win.open_file()
    kinds = [it[1] for it in win.digi_kind.items]
    assert {"NAVTEX", "FAX", "SSTV"} <= set(kinds)
    win.mode_buttons["USB"].clicked.emit(False)
    win.digi_kind.setCurrentIndex(kinds.index("SSTV"))
    d = win.radio.rx.digi
    assert type(d).__name__ == "SSTVDecoder" and win.signal_to_vfo(7_101_900.0) == 7_100_000.0
    win.update_info(); assert "attente" in win.digi_status.text_ and win.pic_rows == []
    d.image_id = 1
    d.rows.extend(np.full((320, 3), k, dtype=np.uint8) for k in range(20))
    win.update_info()
    assert len(win.pic_rows) == 20 and win.picture().shape == (20, 320, 3)
    d.rows.extend(np.full((320, 3), 200, dtype=np.uint8) for _ in range(4)); d.finished = 1
    win.update_info()
    saved = [f for f in os.listdir(tmp_path) if f.endswith("_SSTV.png")]
    assert len(saved) == 1 and len(win.pic_rows) == 24 and open(tmp_path / saved[0], "rb").read(4) == b"\x89PNG"
    d.image_id = 2; d.rows.append(np.zeros((320, 3), dtype=np.uint8)); win.update_info()
    assert len(win.pic_rows) == 1                                  # nouvelle image : l'ancienne a quitté l'écran
    win.digi_kind.setCurrentIndex(kinds.index("FAX"))
    d = win.radio.rx.digi
    row = np.zeros(1810, dtype=np.uint8); row[905:] = 255
    d.rows.extend(row.copy() for _ in range(10)); win.update_info()
    assert win.picture().shape == (10, 1810) and win.picture()[0, 0] == 0
    win.pic_shift.setValue(50); assert win.picture()[0, 0] == 255 and win.picture()[0, 1000] == 0     # cadrage manuel
    win.pic_clear.click(); assert win.picture() is None and win.save_picture() is None
    win.digi_kind.setCurrentIndex(kinds.index("PSK31")); win.update_info()
    assert hasattr(win.radio.rx.digi, "points")


def test_new_decoders_in_gui(window, tmp_path):
    """0.15 : WSPR, Hellschreiber et signaux horaires sont proposés et branchés."""
    win = window
    p = tmp_path / "RadioLab_20260101_000000Z_7100000Hz_IQ.wav"
    make_iq_file(p, seconds=1.0)
    FileDialog.path = str(p); win.open_file()
    kinds = [it[1] for it in win.digi_kind.items]
    win.mode_buttons["USB"].clicked.emit(False)
    for kind, cls in (("WSPR", "WSPRDecoder"), ("HELL", "HellDecoder"), ("DCF77", "DCF77Decoder"), ("ALS162", "ALS162Decoder")):
        win.digi_kind.setCurrentIndex(kinds.index(kind))
        d = win.radio.rx.digi
        assert type(d).__name__ == cls
        time.sleep(0.15); win.update_info()
        assert win.digi_status.text_ == d.status() or win.digi_status.text_.split("dans")[0] == d.status().split("dans")[0]
    assert win.signal_to_vfo(14_097_100.0) == 14_097_100.0 - 1000.0
    win.digi_kind.setCurrentIndex(kinds.index("WSPR")); assert win.signal_to_vfo(14_097_100.0) == 14_095_600.0
    win.digi_kind.setCurrentIndex(kinds.index("HELL"))
    d = win.radio.rx.digi
    d.columns.extend(np.full(14, 1.0) for _ in range(30)); d.count = 30
    win.update_info()
    assert win.picture().shape == (84, 420) and win.picture()[0, 0] == 0 and win.picture()[0, 400] == 255
    assert win.save_picture().endswith("_HELL.png")
