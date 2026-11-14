"""Scanner et plan de bandes (sans matériel : radio et horloge simulées)."""
import pytest
from radiolab.scanner import Scanner
from radiolab.bandplan import band_at, BANDS


class FakeRx:
    mode, level_db = "NFM", -100.0


class FakeRadio:
    def __init__(self):
        self.rx, self.vfo_hz, self.center_hz, self.tuned = FakeRx(), 0, 0.0, []

    def tune(self, hz):
        if hz == 145_512_500:
            raise ValueError("hors plage")
        self.vfo_hz = hz; self.tuned.append(hz)

    def set_mode(self, mode):
        self.rx.mode = mode


def test_scanner_stops_on_signal_and_resumes():
    radio, now = FakeRadio(), [0.0]
    sc = Scanner(radio, clock=lambda: now[0])
    sc.start_range(145_500_000, 145_550_000, 12500, threshold_db=-60)
    assert radio.vfo_hz == 145_500_000 and len(sc.channels) == 5
    sc.tick(); assert radio.vfo_hz == 145_500_000                     # mesure pas encore stable
    now[0] += 0.3; sc.tick(); assert radio.vfo_hz == 145_525_000      # canal vide -> suivant ; 145,5125 est sauté
    radio.rx.level_db = -40.0
    for _ in range(20):
        now[0] += 0.3; sc.tick()
    assert radio.vfo_hz == 145_525_000 and sc.holding                 # signal présent : on reste
    radio.rx.level_db = -100.0
    now[0] += 1.0; sc.tick(); assert radio.vfo_hz == 145_525_000      # maintien 2 s après la fin du signal
    now[0] += 1.5; sc.tick(); assert radio.vfo_hz == 145_537_500
    for _ in range(3):
        now[0] += 0.3; sc.tick()
    assert radio.vfo_hz == 145_525_000                                # la liste boucle
    sc.stop(); now[0] += 5; sc.tick(); assert radio.vfo_hz == 145_525_000


def test_scanner_memories_change_mode():
    radio, now = FakeRadio(), [0.0]
    sc = Scanner(radio, clock=lambda: now[0])
    sc.start([(7_100_000, "LSB"), (87_800_000, "WFM")], -60)
    assert radio.rx.mode == "LSB"
    now[0] += 1; sc.tick(); assert radio.rx.mode == "WFM" and radio.vfo_hz == 87_800_000
    with pytest.raises(ValueError):
        sc.start_range(1e6, 1.8e9, 1.0, -60)
    with pytest.raises(ValueError):
        sc.start([], -60)


def test_bandplan():
    assert band_at(7_100_000) == "40 m amateur" and band_at(98_000_000) == "Radiodiffusion FM"
    assert band_at(125_000_000) == "Aviation (AM)" and band_at(300_000_000) == ""
    assert all(lo < hi for lo, hi, _, _ in BANDS)
