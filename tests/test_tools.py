"""Commande réseau, mémoires CSV, journal et enregistrements programmés."""
import datetime
import socket
import pytest
from radiolab.netcontrol import RigCtlServer
from radiolab import planner

MODES = ("AM", "LSB", "USB", "CW", "NFM", "WFM")


def test_rigctl_protocol():
    state = [145_800_000.0, "NFM", 12000.0]
    srv = RigCtlServer(lambda: tuple(state), port=0)
    s = socket.create_connection(("127.0.0.1", srv.port), timeout=2)
    f = s.makefile("rw", newline="\n")

    def ask(cmd, lines=1):
        f.write(cmd + "\n"); f.flush()
        return [f.readline().strip() for _ in range(lines)]
    assert ask("f") == ["145800000"]
    assert ask("F 145812345") == ["RPRT 0"] and srv.commands.get(timeout=1) == ("freq", 145812345.0)
    assert ask("F  437505000.000000") == ["RPRT 0"] and srv.commands.get(timeout=1) == ("freq", 437505000.0)
    assert ask("m", 2) == ["FM", "12000"]
    assert ask("M USB 2400") == ["RPRT 0"] and srv.commands.get(timeout=1) == ("mode", "USB")
    assert ask("\\get_freq") == ["145800000"] and ask("\\set_mode FM 0") == ["RPRT 0"]
    assert srv.commands.get(timeout=1) == ("mode", "NFM")
    assert ask("M RTTY 0") == ["RPRT -1"] and ask("F") == ["RPRT -1"] and ask("\\get_rit") == ["RPRT -11"]
    assert ask("t") == ["0"] and ask("v") == ["VFOA"] and ask("\\chk_vfo") == ["CHKVFO 0"]
    assert ask("\\get_powerstat") == ["1"] and ask("s", 2) == ["0", "VFOA"] and ask("S 0 VFOA") == ["RPRT 0"]
    dump = ask("\\dump_state", 34)
    assert dump[:3] == ["0", "2", "1"] and dump[4] == "0 0 0 0 0 0 0" and dump[-1] == "0" and "" not in dump
    assert ask("f") == ["145800000"]                                   # la suite reste synchronisée
    f.write("q\n"); f.flush()
    assert f.readline() == ""
    s.close(); srv.close()


def test_memories_csv_roundtrip(tmp_path):
    mem = [{"name": "France Inter", "hz": 87_800_000, "mode": "WFM", "group": "FM"},
           {"name": "Relais R1", "hz": 145_625_000, "mode": "NFM"}]
    p = tmp_path / "m.csv"
    planner.export_memories(str(p), mem)
    back = planner.import_memories(str(p), MODES)
    assert [(m["name"], m["hz"], m["mode"], m["group"]) for m in back] == [
        ("France Inter", 87_800_000, "WFM", "FM"), ("Relais R1", 145_625_000, "NFM", "")]
    q = tmp_path / "other.csv"
    q.write_text("Name,Frequency,Mode\nTour,118 700 000,AM\nRelais,145600000.0,FM\nBad,abc,AM\nX,7100000,PSK\n", encoding="utf-8")
    back = planner.import_memories(str(q), MODES)
    assert [(m["name"], m["hz"], m["mode"]) for m in back] == [("Tour", 118_700_000, "AM"), ("Relais", 145_600_000, "NFM"), ("X", 7_100_000, "USB")]


def test_activity_log(tmp_path):
    now = datetime.datetime(2026, 11, 12, 21, 5, 9)
    p = planner.log_activity(str(tmp_path), 145_500_000, "NFM", -42.3, "88.5 Hz", now)
    planner.log_activity(str(tmp_path), 145_525_000, "NFM", -50, "", now)
    lines = open(p, encoding="utf-8-sig").read().splitlines()
    assert p.endswith("RadioLab_journal_20261112.csv") and len(lines) == 3
    assert lines[1] == "21:05:09;145500000;NFM;-42;88.5 Hz"


def test_scheduler():
    jobs = [planner.parse_job("2026-11-12 21:00;30;7100000;lsb;audio", MODES),
            planner.parse_job("2026-11-12 22:00;1.5;98000000;WFM;IQ", MODES),
            planner.parse_job("2026-11-11 10:00;10;7100000;LSB;audio", MODES)]
    assert planner.job_label(jobs[1]) == "2026-11-12 22:00 — 1.5 min — 98000000 Hz WFM — I/Q"
    sch = planner.Scheduler(jobs)
    t = lambda h, m, s=0: datetime.datetime(2026, 11, 12, h, m, s)
    assert sch.tick(t(20, 59)) == [] and len(jobs) == 2                 # le créneau de la veille est retiré
    a = sch.tick(t(21, 0)); assert a == [("start", jobs[0])]
    first = jobs[0]
    assert sch.tick(t(21, 15)) == []
    assert sch.tick(t(21, 30)) == [("stop", first)] and len(jobs) == 1
    assert [x[0] for x in sch.tick(t(22, 0, 30))] == ["start"]
    assert [x[0] for x in sch.tick(t(22, 1, 30))] == ["stop"] and jobs == []
    for bad in ("demain;30;7100000;LSB;audio", "2026-11-12 21:00;30;7100000;LSB", "2026-11-12 21:00;0;7100000;LSB;audio",
                "2026-11-12 21:00;30;7100000;PSK;audio"):
        with pytest.raises(ValueError):
            planner.parse_job(bad, MODES)
