"""Fax météo, SSTV, NAVTEX : émissions synthétiques reçues par la chaîne complète (I/Q -> BLU -> décodeur)."""
import zlib
import numpy as np
import pytest
from radiolab.dsp.receiver import Receiver
from radiolab.dsp import navtex
from radiolab.dsp.pictures import write_png, SSTV_MODES, SSTV_BY_VIS, ycc_to_rgb
from radiolab.dsp.spectrum import TraceMemory

FS = 250e3
OFF = 30e3


def to_rf(freq_hz, snr_db=None, seed=0):
    """Signal I/Q dont la note audio (en USB sur OFF) suit freq_hz, un point par échantillon."""
    phase = 2 * np.pi * np.cumsum(OFF + freq_hz) / FS
    x = 0.02 * np.exp(1j * phase)
    if snr_db is not None:
        r = np.random.default_rng(seed)
        sigma = np.sqrt(np.mean(np.abs(x) ** 2) * 10 ** (-snr_db / 10) * FS / 2500.0)
        x = x + sigma * (r.standard_normal(len(x)) + 1j * r.standard_normal(len(x))) / np.sqrt(2)
    return x.astype(np.complex64)


def run(iq, kind, center=1000.0, chunk=32768):
    rx = Receiver(FS); rx.set_mode("USB"); rx.set_offset(OFF)
    rx.set_digi(kind, center)
    rows = []
    for i in range(0, len(iq), chunk):
        rx.process(iq[i:i + chunk])
        if hasattr(rx.digi, "take_rows"):
            rows += rx.digi.take_rows()
    return rx, rows


def test_png(tmp_path):
    def read(path):
        data = open(path, "rb").read()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        pos, out, head = 8, b"", None
        while pos < len(data):
            n = int.from_bytes(data[pos:pos + 4], "big"); tag = data[pos + 4:pos + 8]; body = data[pos + 8:pos + 8 + n]
            assert zlib.crc32(tag + body) & 0xFFFFFFFF == int.from_bytes(data[pos + 8 + n:pos + 12 + n], "big")
            if tag == b"IHDR": head = body
            if tag == b"IDAT": out += body
            pos += 12 + n
        return head, zlib.decompress(out)
    gray = (np.arange(6 * 5).reshape(6, 5) * 8).astype(np.uint8)
    write_png(str(tmp_path / "g.png"), gray)
    head, raw = read(tmp_path / "g.png")
    assert head == (5).to_bytes(4, "big") + (6).to_bytes(4, "big") + bytes([8, 0, 0, 0, 0])
    assert np.array_equal(np.frombuffer(raw, np.uint8).reshape(6, 6)[:, 1:], gray)
    rgb = np.random.default_rng(1).integers(0, 255, (4, 7, 3)).astype(np.uint8)
    write_png(str(tmp_path / "c.png"), rgb)
    head, raw = read(tmp_path / "c.png")
    assert head[8:10] == bytes([8, 2]) and np.array_equal(np.frombuffer(raw, np.uint8).reshape(4, 22)[:, 1:].reshape(4, 7, 3), rgb)


def test_trace_memory():
    t = TraceMemory()
    f = np.linspace(0, 1000, 50)
    r = np.random.default_rng(0)
    noisy = [10 * np.log10(r.exponential(1.0, 50)) for _ in range(60)]
    for db in noisy:
        avg, peak = t.update(f, db, average=20, hold=True)
    assert np.std(avg) < 0.5 * np.std(noisy[-1]) and np.all(peak >= avg - 1e-9)       # bruit lissé, crêtes au-dessus
    avg, peak = t.update(f, noisy[0], average=1, hold=False)
    assert np.allclose(avg, noisy[0]) and peak is None
    t.update(f, noisy[0], 20, True)
    avg, peak = t.update(f + 500, noisy[1], 20, True)                                 # la vue a bougé : tout repart
    assert np.allclose(avg, noisy[1]) and np.allclose(peak, noisy[1])


def navtex_signal(text, f_audio=1000.0, high_is_one=True):
    bits = []
    for code in navtex.encode(text):
        bits += [(code >> k) & 1 for k in range(6, -1, -1)]
    n = int(len(bits) * FS / 100.0)
    b = np.array(bits)[np.minimum((np.arange(n) * 100.0 / FS).astype(np.int64), len(bits) - 1)]
    return f_audio + 85.0 * (2 * b - 1) * (1 if high_is_one else -1)


@pytest.mark.parametrize("snr,high", [(30, True), (10, True), (20, False)])
def test_navtex(snr, high):
    text = "ZCZC SA12\r\n121530 UTC NOV 26\r\nCROSS CORSEN GALE WARNING NR 345\r\nNNNN"
    rx, _ = run(to_rf(navtex_signal(text, high_is_one=high), snr), "NAVTEX")
    out = "".join(rx.digi_text)
    assert "GALE WARNING NR 345" in out and "ZCZC SA12" in out and "121530 UTC" in out, repr(out)
    assert rx.digi.synced and rx.digi.q > 0.9


def test_navtex_repairs_errors():
    """Un caractère abîmé dans un des deux exemplaires est rattrapé par l'autre."""
    text = "ZCZC AB01 SECURITE NAVIGATION 12345 NNNN"
    slots = navtex.encode(text)
    start = 40
    for i in range(start + 6, len(slots) - 12, 14):
        slots[i] ^= 0b0000001                                   # 5 bits à 1 au lieu de 4 : erreur détectable
    bits = [b for code in slots for b in [(code >> k) & 1 for k in range(6, -1, -1)]]
    n = int(len(bits) * FS / 100.0)
    b = np.array(bits)[np.minimum((np.arange(n) * 100.0 / FS).astype(np.int64), len(bits) - 1)]
    rx, _ = run(to_rf(1000.0 + 85.0 * (2 * b - 1), 25), "NAVTEX")
    out = "".join(rx.digi_text)
    assert "SECURITE NAVIGATION 12345 NNNN" in out, repr(out)


def fax_picture(lines):
    """Image d'essai : dégradé horizontal, bande noire à gauche, bande blanche au quart."""
    x = np.linspace(0, 1, 1810)
    row = 0.2 + 0.6 * x
    row[:60] = 0.0
    row[450:500] = 1.0
    return np.tile(row, (lines, 1))


def fax_signal(picture, start=True, phasing=20, stop=True, lead=0.37):
    spl = int(FS * 0.5)                                          # échantillons par ligne (120 lignes par minute)
    parts = [np.full(int(lead * FS), 1900.0)]
    t = np.arange(int(3.0 * FS)) / FS
    if start:
        parts.append(np.where(np.sin(2 * np.pi * 300 * t) > 0, 2300.0, 1500.0))
    one = np.full(spl, 2300.0); one[:int(0.05 * spl)] = 1500.0   # ligne blanche, impulsion noire de 5 % au bord
    parts += [one] * phasing
    for row in picture:
        parts.append(1500.0 + 800.0 * np.interp(np.arange(spl) * len(row) / spl, np.arange(len(row)), row))
    if stop:
        parts.append(np.where(np.sin(2 * np.pi * 450 * t) > 0, 2300.0, 1500.0))
    parts.append(np.full(int(0.6 * FS), 1900.0))
    return np.concatenate(parts)


def test_fax():
    pic = fax_picture(24)
    rx, rows = run(to_rf(fax_signal(pic), 25), "FAX")
    d = rx.digi
    assert d.image_id == 1 and d.finished == 1 and d.phased, d.status()
    img = np.array(rows[-20:], dtype=np.float64) / 255.0
    assert 22 <= d.lines <= 26 and img.shape[1] == 1810, d.lines
    mean = img.mean(axis=0)
    # le bord de l'image est calé sur le milieu de l'impulsion (2,5 % de ligne) : bandes et dégradé sont en place
    shift = -0.025 * 1810
    mean = np.roll(mean, 100); shift += 100                      # marge : la bande noire de gauche chevauche le bord
    black = mean[int(shift) + 5:int(shift) + 50]; white = mean[int(450 + shift) + 5:int(500 + shift) - 5]
    assert black.max() < 0.15 and white.min() > 0.85, (black.max(), white.min())
    assert abs(mean[1000] - (0.2 + 0.6 * (1000 - shift) / 1810)) < 0.06 and abs(mean[1600] - mean[700] - 0.6 * 900 / 1810) < 0.05
    text = "".join(rx.digi_text)
    assert "début" in text and "fin de l'image" in text


def test_fax_caught_midway():
    """Pris en cours d'émission (ni départ ni mise en phase) : l'image se trace quand même, sans cadrage."""
    rx, rows = run(to_rf(fax_signal(fax_picture(16), start=False, phasing=0, stop=False), 25), "FAX")
    assert len(rows) >= 14 and not rx.digi.phased and "prise en cours" in rx.digi.status()
    img = np.array(rows[2:14], dtype=np.float64) / 255.0
    assert img.std(axis=0).mean() < 0.05                         # lignes identiques : le rythme des lignes est tenu


def sstv_picture(width, height):
    x = np.linspace(0, 1, width)[None, :]; y = np.linspace(0, 1, height)[:, None]
    r = 255 * x * np.ones_like(y); g = 255 * y * np.ones_like(x); b = 255 * (1 - x) * np.ones_like(y)
    img = np.stack([r, g, b], axis=-1)
    img[:, width // 4:width // 4 + width // 16] = (255, 255, 255)
    img[:, width // 2:width // 2 + width // 16] = (0, 0, 0)
    return np.round(img).astype(np.uint8)


def rgb_to_ycc(rgb):
    r, g, b = [rgb[..., i].astype(np.float64) for i in range(3)]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y, 128 + 0.713 * (r - y), 128 + 0.564 * (b - y)


def sstv_signal(mode, img, clock_error=0.0):
    """Suite (fréquence, durée en ms) puis conversion en échantillons ; clock_error simule une horloge d'émetteur fausse."""
    seq = [(1900.0, 300.0), (1200.0, 10.0), (1900.0, 300.0), (1200.0, 30.0)]
    bits = [(mode["vis"] >> k) & 1 for k in range(7)]
    bits.append(sum(bits) % 2)
    seq += [(1100.0 if b else 1300.0, 30.0) for b in bits] + [(1200.0, 30.0)]
    w = mode["width"]

    def scan(values, dur):
        return [(1500.0 + 800.0 * v / 255.0, dur / len(values)) for v in values]
    name = mode["name"]
    if name.startswith("Martin"):
        dur = mode["scans"][0][2]
        for row in img:
            seq += [(1200.0, 4.862), (1500.0, 0.572)]
            for ch in (1, 2, 0):
                seq += scan(row[:, ch], dur) + [(1500.0, 0.572)]
    elif name.startswith("Scottie"):
        dur = mode["scans"][0][2]
        seq.append((1200.0, 9.0))
        for row in img:
            seq += [(1500.0, 1.5)] + scan(row[:, 1], dur) + [(1500.0, 1.5)] + scan(row[:, 2], dur)
            seq += [(1200.0, 9.0), (1500.0, 1.5)] + scan(row[:, 0], dur)
    elif name.startswith("PD"):
        dur = mode["scans"][0][2]
        y, cr, cb = rgb_to_ycc(img)
        for k in range(0, len(img), 2):
            seq += [(1200.0, 20.0), (1500.0, 2.08)] + scan(y[k], dur) + scan((cr[k] + cr[k + 1]) / 2, dur)
            seq += scan((cb[k] + cb[k + 1]) / 2, dur) + scan(y[k + 1], dur)
    else:                                                        # Robot 36
        y, cr, cb = rgb_to_ycc(img)
        for k, row in enumerate(img):
            c = (cr if k % 2 == 0 else cb)[k].reshape(-1, 2).mean(axis=1)
            seq += [(1200.0, 9.0), (1500.0, 3.0)] + scan(y[k], 88.0)
            seq += [(1500.0 if k % 2 == 0 else 2300.0, 4.5), (1900.0, 1.5)] + scan(c, 44.0)
    seq = [(1900.0, 400.0)] + seq + [(1900.0, 300.0)]
    freqs = np.array([f for f, _ in seq]); ends = np.cumsum([d for _, d in seq]) * 1e-3 * (1.0 + clock_error)
    t = np.arange(int(ends[-1] * FS)) / FS
    return freqs[np.minimum(np.searchsorted(ends, t, side="right"), len(freqs) - 1)]


def check_sstv(mode, rows, img, tol):
    out = np.array(rows, dtype=np.float64)
    assert out.shape == img.shape, (out.shape, img.shape)
    err = np.abs(out - img)
    w = img.shape[1]
    keep = np.ones(w, dtype=bool)
    for edge in (w // 4, w // 4 + w // 16, w // 2, w // 2 + w // 16):          # les fronts raides sont adoucis par les filtres
        keep[max(edge - w // 40, 0):edge + w // 40] = False
    keep[:w // 40] = False; keep[-(w // 40):] = False
    assert np.mean(err[2:-2, keep]) < tol, (mode["name"], np.mean(err[2:-2, keep]))
    mid = out[len(out) // 2]
    assert mid[w // 4 + w // 32].min() > 215 and mid[w // 2 + w // 32].max() < 40     # barres blanche et noire à leur place


@pytest.mark.parametrize("name,lines,tol", [("Martin M2", 24, 9), ("Scottie S2", 24, 9), ("Martin M1", 12, 7),
                                            ("Scottie S1", 12, 7), ("PD120", 24, 12), ("PD90", 12, 9), ("Robot 36", 24, 12)])
def test_sstv(name, lines, tol):
    mode = dict(next(m for m in SSTV_MODES if m["name"] == name))
    img = sstv_picture(mode["width"], lines)
    mode_lines = lines // 2 if mode["kind"] == "pd" else lines
    saved = SSTV_BY_VIS[mode["vis"]]["lines"]
    SSTV_BY_VIS[mode["vis"]]["lines"] = mode_lines               # image raccourcie : l'essai reste rapide
    try:
        rx, rows = run(to_rf(sstv_signal(mode, img), 30), "SSTV")
    finally:
        SSTV_BY_VIS[mode["vis"]]["lines"] = saved
    assert rx.digi.image_id == 1 and rx.digi.finished == 1 and name in "".join(rx.digi_text)
    check_sstv(mode, rows, img, tol)


def test_sstv_clock_error():
    """Horloge de l'émetteur fausse de 300 ppm : le recalage sur les impulsions de ligne évite l'image penchée."""
    mode = next(m for m in SSTV_MODES if m["name"] == "Martin M2")
    img = sstv_picture(320, 120)
    saved, mode["lines"] = mode["lines"], 120
    try:
        rx, rows = run(to_rf(sstv_signal(mode, img, clock_error=300e-6), 30), "SSTV")
    finally:
        mode["lines"] = saved
    out = np.array(rows, dtype=np.float64)
    top = np.argmax(np.diff(out[5, 60:120, 0])); bottom = np.argmax(np.diff(out[114, 60:120, 0]))
    assert abs(int(top) - 19) <= 1 and abs(int(bottom) - 19) <= 1, (top, bottom)   # sans recalage : 36 points d'écart en bas


def test_timing_tables():
    by = {m["name"]: m for m in SSTV_MODES}
    assert abs(by["Martin M1"]["period"] - 446.446) < 1e-6 and abs(by["Martin M2"]["period"] - 226.798) < 1e-6
    assert abs(by["Scottie S1"]["period"] - 428.220) < 1e-6 and abs(by["Scottie S2"]["period"] - 277.692) < 1e-6
    assert abs(by["PD120"]["period"] - 508.480) < 1e-6 and abs(by["PD180"]["period"] - 754.240) < 1e-6
    assert abs(by["PD90"]["period"] - 703.040) < 1e-6 and len({m["vis"] for m in SSTV_MODES}) == len(SSTV_MODES)
    assert np.array_equal(ycc_to_rgb(np.array([128]), np.array([128]), np.array([128])), [[128, 128, 128]])


def test_sstv_in_fm():
    """SSTV en FM étroite (comme depuis la station spatiale) : la note audio module la fréquence de la porteuse."""
    mode = next(m for m in SSTV_MODES if m["name"] == "PD120")
    img = sstv_picture(640, 16)
    saved, mode["lines"] = mode["lines"], 8
    try:
        tone = sstv_signal(mode, img)
        audio = np.cos(2 * np.pi * np.cumsum(tone) / FS)
        iq = (0.05 * np.exp(1j * (2 * np.pi * OFF * np.arange(len(audio)) / FS + 2 * np.pi * 3000.0 * np.cumsum(audio) / FS)))
        rx = Receiver(FS); rx.set_mode("NFM"); rx.set_offset(OFF); rx.set_digi("SSTV")
        rows = []
        for i in range(0, len(iq), 32768):
            rx.process(iq[i:i + 32768].astype(np.complex64)); rows += rx.digi.take_rows()
    finally:
        mode["lines"] = saved
    assert rx.digi.finished == 1
    check_sstv(mode, rows, img, 14)
