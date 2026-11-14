"""TETRA, premier étage : démodulation pi/4-DQPSK et repérage des salves, sur un signal fabriqué."""
import numpy as np
import pytest
from radiolab.dsp.receiver import Receiver
from radiolab.dsp.tetra import (TetraDecoder, rrc_taps, bits_to_phase_steps, TRAIN_NORMAL_1, TRAIN_NORMAL_2,
                                TRAIN_SYNC, NORMAL_TRAIN_START, SYNC_TRAIN_START, BURST_BITS, BSCH_SCRAMBLING,
                                BSCH_ORDER, decode_bsch, crc16_ok)

SPS = 16
FS = 18000 * SPS                      # 288 kHz
OFF = 40e3


def encode_bsch(mcc, mnc, colour, timeslot, frame, multiframe):
    """Fabrique les 120 bits du bloc de synchronisation, par les opérations inverses du décodeur."""
    def put(value, n):
        return [(value >> (n - 1 - k)) & 1 for k in range(n)]
    info = put(0, 4) + put(colour, 6) + put(timeslot - 1, 2) + put(frame, 5) + put(multiframe, 6) + [0] * 8 \
        + put(mcc, 10) + put(mnc, 14) + [1, 1, 0, 0, 1]
    crc = 0xFFFF
    for b in info:
        x = ((crc >> 15) & 1) ^ b
        crc = (crc << 1) & 0xFFFF
        if x:
            crc ^= 0x1021
    crc ^= 0xFFFF
    bits = info + put(crc, 16) + [0, 0, 0, 0]
    assert crc16_ok(bits[:76])
    coded, hist = [], [0, 0, 0, 0]
    for t, b in enumerate(bits):                               # G1 = 1 + D + D^4, G2 = 1 + D^2 + D^3 + D^4
        coded.append(b ^ hist[0] ^ hist[3])
        if t % 2 == 0:
            coded.append(b ^ hist[1] ^ hist[2] ^ hist[3])
        hist = [b] + hist[:3]
    block = np.zeros(120, dtype=np.int8)
    block[BSCH_ORDER] = coded                                  # entrelacement
    return block ^ BSCH_SCRAMBLING


def tetra_bits(n_bursts, seed=0, cell=None):
    """Suite de salves descendantes : contenu aléatoire, vraies séquences d'apprentissage à leur place."""
    r = np.random.default_rng(seed)
    out, kinds = [], []
    for k in range(n_bursts):
        b = r.integers(0, 2, BURST_BITS)
        if k % 18 == 17:                                       # de temps en temps, une salve de synchronisation
            b[SYNC_TRAIN_START:SYNC_TRAIN_START + 38] = TRAIN_SYNC; kinds.append("synchro")
            if cell:
                b[94:214] = encode_bsch(*cell, timeslot=1 + k % 4, frame=1 + (k // 4) % 18, multiframe=1 + (k // 72) % 60)
        elif k % 3 == 0:
            b[NORMAL_TRAIN_START:NORMAL_TRAIN_START + 22] = TRAIN_NORMAL_2; kinds.append("normale 2")
        else:
            b[NORMAL_TRAIN_START:NORMAL_TRAIN_START + 22] = TRAIN_NORMAL_1; kinds.append("normale 1")
        out.append(b)
    return np.concatenate(out), kinds


def modulate(bits, freq_hz, snr_db=25.0, seed=1):
    phase = np.cumsum(bits_to_phase_steps(bits))
    up = np.zeros(len(phase) * SPS, dtype=np.complex128)
    up[::SPS] = np.exp(1j * phase)
    x = np.convolve(up, rrc_taps(SPS) * SPS, "same")
    t = np.arange(len(x)) / FS
    x = 0.05 * x * np.exp(2j * np.pi * freq_hz * t)
    r = np.random.default_rng(seed)
    sigma = np.sqrt(np.mean(np.abs(x) ** 2) * 10 ** (-snr_db / 10) * FS / 25000.0)   # SNR dans 25 kHz
    return (x + sigma * (r.standard_normal(len(x)) + 1j * r.standard_normal(len(x))) / np.sqrt(2)).astype(np.complex64)


def run(rx, iq, block=20000):
    for i in range(0, len(iq), block):
        rx.process(iq[i:i + block])


def test_rrc_pair_has_no_intersymbol_interference():
    h = np.convolve(rrc_taps(SPS), rrc_taps(SPS))
    centre = len(h) // 2
    others = h[centre % SPS::SPS]
    assert np.sort(np.abs(others))[-2] < 0.02 * abs(h[centre])            # quasi nul aux instants des autres symboles


@pytest.mark.parametrize("snr_db,offset_hz", [(30.0, 0.0), (18.0, 350.0), (14.0, -900.0)])
def test_demodulated_bits_match(snr_db, offset_hz):
    bits, _ = tetra_bits(40)
    got = []
    rx = Receiver(FS); rx.set_mode("TETRA"); rx.set_offset(OFF)
    dec = rx.digi
    original = dec._bits
    dec._bits = lambda new: (got.append(new.copy()), original(new))[1]
    run(rx, modulate(bits, OFF + offset_hz, snr_db))
    out = np.concatenate(got)
    assert abs(dec.freq_error_hz - offset_hz) < 25 and dec.quality > 0.6
    ref = bits[2000:12000]                                                # on cherche ce passage dans la sortie
    best = max(np.mean(out[s:s + len(ref)] == ref) for s in range(0, 6000, 2))
    assert best > 0.999 if snr_db >= 18 else best > 0.99                  # moins d'un bit faux sur 1000


def test_bursts_are_found_and_counted():
    bits, kinds = tetra_bits(180)                                          # 2,55 s de signal
    rx = Receiver(FS); rx.set_mode("TETRA"); rx.set_offset(OFF)
    run(rx, modulate(bits, OFF + 200.0, 20.0))
    dec = rx.digi
    assert dec.locked and dec.slot_phase is not None
    for name in ("normale 1", "normale 2", "synchro"):
        assert abs(dec.counts[name] - kinds.count(name)) <= 0.08 * kinds.count(name) + 2, (name, dec.counts)
    assert "Porteuse TETRA" in dec.status() and "Porteuse TETRA" in "".join(rx.digi_text)
    assert len(rx.process(np.zeros(20000, dtype=np.complex64))) > 0        # le flux audio (muet) continue


def test_no_false_lock_on_noise_or_fm():
    r = np.random.default_rng(3)
    n = int(FS * 3)
    noise = (0.01 * (r.standard_normal(n) + 1j * r.standard_normal(n))).astype(np.complex64)
    t = np.arange(n) / FS
    fm = (0.05 * np.exp(1j * (2 * np.pi * OFF * t + 2.5 * np.sin(2 * np.pi * 1000 * t)))).astype(np.complex64)
    for sig in (noise, fm + noise):
        rx = Receiver(FS); rx.set_mode("TETRA"); rx.set_offset(OFF)
        run(rx, sig)
        assert not rx.digi.locked and "".join(rx.digi_text) == "" and "Pas de porteuse" in rx.digi.status()


def test_tetra_mode_needs_enough_bandwidth():
    rx = Receiver(48000.0)
    with pytest.raises(ValueError):
        rx.set_mode("TETRA")
    rx = Receiver(2.5e6); rx.set_mode("TETRA")
    assert rx.fs1 > 80e3 and isinstance(rx.digi, TetraDecoder)
    rx.set_mode("USB"); assert rx.digi is None


def test_bsch_block_roundtrip_and_error_correction():
    block = encode_bsch(208, 1234, 43, timeslot=3, frame=17, multiframe=42)
    info = decode_bsch(block)
    assert (info["mcc"], info["mnc"], info["colour"], info["timeslot"], info["frame"], info["multiframe"]) == (208, 1234, 43, 3, 17, 42)
    assert info["corrected"] == 0
    noisy = block.copy(); noisy[[5, 47, 99]] ^= 1                # trois bits faux : le code les corrige
    info = decode_bsch(noisy)
    assert info is not None and info["mnc"] == 1234 and info["corrected"] == 3
    r = np.random.default_rng(0)
    assert all(decode_bsch(r.integers(0, 2, 120)) is None for _ in range(200))   # du hasard ne passe pas le CRC


def test_cell_identity_is_read_from_the_signal():
    bits, _ = tetra_bits(200, cell=(262, 1001, 17))
    rx = Receiver(FS); rx.set_mode("TETRA"); rx.set_offset(OFF)
    run(rx, modulate(bits, OFF - 400.0, 18.0))
    dec = rx.digi
    assert dec.locked and dec.bsch_ok >= 3 and dec.bsch_bad == 0
    assert (dec.cell["mcc"], dec.cell["mnc"], dec.cell["colour"]) == (262, 1001, 17)
    text = "".join(rx.digi_text)
    assert "Cellule TETRA identifiée : pays 262, réseau 1001, code couleur 17" in text
    assert "réseau 1001" in dec.status() and "trame" in dec.status()
    r = dec.burst_rates()
    assert 55 < r["normale 1"] + r["normale 2"] + r["synchro"] < 75      # environ 70 salves par seconde
