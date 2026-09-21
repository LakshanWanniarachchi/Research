"""
Tests for the USB-serial acquisition path (ecg_serial_monitor.py).

No hardware and no trained model are needed: lines are synthesised in the
exact format the firmware prints, and the one test that touches the model
package uses only preprocess(), which is pure numpy.
"""

import numpy as np

import ecg_serial_monitor as mon


def _samples(n, start_seq=0, adc=None, lead_off_at=(), period_us=10_000):
    """Yield Sample objects as the firmware would stream them."""
    for i in range(n):
        seq = start_seq + i
        value = 2000 + int(300 * np.sin(i / 7)) if adc is None else adc
        lo = 1 if i in lead_off_at else 0
        yield mon.Sample(seq, 1_000_000 + seq * period_us, value, lo, 0)


# ---------------------------------------------------------------------------
# parse_line
# ---------------------------------------------------------------------------
def test_parses_a_sample_line():
    kind, s = mon.parse_line("42,231551,2048,0,1\n")
    assert kind == "sample"
    assert (s.seq, s.t_us, s.adc, s.lo_plus, s.lo_minus) == (42, 231551, 2048, 0, 1)
    assert s.lead_off


def test_status_boot_and_corrupt_lines_are_never_samples():
    assert mon.parse_line("# columns: seq,t_us,ecg_adc,lo_plus,lo_minus")[0] == "status"
    assert mon.parse_line("rst:0x1 (POWERON_RESET),boot:0x13")[0] == "text"
    assert mon.parse_line("12,3456,20")[0] == "malformed"          # truncated
    assert mon.parse_line("12,3456,5000,0,0")[0] == "malformed"    # > 12-bit
    assert mon.parse_line("12,3456,200,2,0")[0] == "malformed"     # LO not 0/1
    assert mon.parse_line("   ") is None


# ---------------------------------------------------------------------------
# WindowBuilder
# ---------------------------------------------------------------------------
def test_clean_stream_gives_one_1000_sample_window():
    # require_beat=False: these three tests check window assembly, not signal
    # content, and their synthetic sine deliberately contains no QRS complex.
    wb = mon.WindowBuilder(require_beat=False)
    windows = [w for s in _samples(1000) if (w := wb.add(s)) is not None]
    assert len(windows) == 1
    w = windows[0]
    assert w.values.shape == (1000,)
    assert w.first_seq == 0
    assert abs(w.measured_rate_hz - 100.0) < 1e-6


def test_lead_off_sample_restarts_the_window():
    wb = mon.WindowBuilder()
    out = [wb.add(s) for s in _samples(1500, lead_off_at={600})]
    windows = [w for w in out if w is not None]
    # 600 good samples are discarded; the next window starts at seq 601
    # and needs until seq 1600 to fill, which this stream does not reach.
    assert windows == []
    assert wb.rejected["lead_off"] == 1
    assert wb.progress == 1500 - 601


def test_sequence_gap_restarts_the_window():
    wb = mon.WindowBuilder(require_beat=False)
    stream = list(_samples(500)) + list(_samples(1000, start_seq=501))
    windows = [w for s in stream if (w := wb.add(s)) is not None]
    assert wb.rejected["sequence_gap"] == 1
    assert len(windows) == 1 and windows[0].first_seq == 501


def test_saturated_window_is_refused():
    wb = mon.WindowBuilder()
    windows = [w for s in _samples(1000, adc=4095) if (w := wb.add(s)) is not None]
    assert windows == []
    assert wb.rejected["saturated"] == 1


def test_window_matches_the_model_input_shape():
    """The window goes into the model's own preprocess(), giving (1,1000,1)."""
    import model
    wb = mon.WindowBuilder(require_beat=False)
    w = [w for s in _samples(1000) if (w := wb.add(s)) is not None][0]
    x = model.preprocess(w.values)
    assert x.shape == (1, 1000, 1)
    assert abs(float(x.mean())) < 1e-4 and abs(float(x.std()) - 1) < 1e-3


def test_signal_quality_flags():
    q = mon.signal_quality([0, 4095] * 100)
    assert q["saturated"] and q["rail_fraction"] == 1.0
    assert mon.signal_quality([2000] * 200)["flat"]


def test_diagnose_names_the_wiring_fault_first():
    # AD8232 unpowered or OUT not on VP: the ADC is pinned AND the lead-off
    # pins float high, because nothing is driving them.
    ok, msg = mon.diagnose([0] * 200, lead_off=True)
    assert not ok and "0 V" in msg
    ok, msg = mon.diagnose([4095] * 200, lead_off=True)
    assert not ok and "3.3 V" in msg
    ok, msg = mon.diagnose([2000 + (i % 7) * 40 for i in range(200)], lead_off=True)
    assert not ok and "LEAD OFF" in msg
    ok, msg = mon.diagnose([0, 4095] * 100, lead_off=False)
    assert not ok and "saturated" in msg
    ok, msg = mon.diagnose([2000 + (i % 7) * 40 for i in range(200)], lead_off=False)
    assert ok
    assert not mon.diagnose([2000] * 200, lead_off=False, link_ok=False)[0]


def _synthetic_ecg(n=1500, fs=100, bpm=72, amp=300, base=2600):
    """A crude beat train: one narrow spike per cardiac cycle, plus noise."""
    rng = np.random.default_rng(3)
    x = np.full(n, float(base)) + rng.normal(0, 4, n)
    period = int(fs * 60 / bpm)
    for i in range(0, n, period):
        if i + 3 < n:
            x[i:i + 3] += [amp * 0.4, amp, amp * 0.3]
    return x


def test_detect_beat_finds_a_beat_and_rejects_mains_noise():
    bpm, strength = mon.detect_beat(_synthetic_ecg(bpm=72))
    assert 65 < bpm < 80 and strength >= mon.BEAT_MIN_STRENGTH

    # 50 Hz interference sampled at 100 Hz: alternating, no heartbeat
    n = np.arange(1500)
    mains = 2600 + 300 * (-1.0) ** n + np.random.default_rng(0).normal(0, 20, 1500)
    _, strength = mon.detect_beat(mains)
    assert strength < mon.BEAT_MIN_STRENGTH


def test_window_without_a_heartbeat_is_not_scored():
    n = np.arange(1000)
    mains = 2600 + 300 * (-1.0) ** n
    wb = mon.WindowBuilder()
    out = [w for i, v in enumerate(mains)
           if (w := wb.add(mon.Sample(i, 1_000_000 + i * 10_000, float(v), 0, 0)))]
    assert out == [] and wb.rejected["no_heartbeat"] == 1

    # the same signal is scored when the gate is switched off
    wb2 = mon.WindowBuilder(require_beat=False)
    out2 = [w for i, v in enumerate(mains)
            if (w := wb2.add(mon.Sample(i, 1_000_000 + i * 10_000, float(v), 0, 0)))]
    assert len(out2) == 1


def test_diagnose_reports_a_missing_heartbeat():
    n = np.arange(1200)
    mains = 2600 + 300 * (-1.0) ** n
    ok, msg = mon.diagnose(mains[-200:], lead_off=False, long_adc=mains)
    assert not ok and "No heartbeat" in msg
    ok, msg = mon.diagnose(_synthetic_ecg()[-200:], lead_off=False,
                           long_adc=_synthetic_ecg())
    assert ok and "bpm" in msg


def test_a_railed_input_blames_the_electrodes_when_the_leads_are_on():
    """Two different faults look identical in the samples alone.

    Pinned at 0 with the leads reported OFF is a wiring or power fault. Pinned
    at 0 with the leads reported ON is the amplifier railed by baseline drift -
    seen for real on 2026-09-23, when a good ECG slid to the bottom of the ADC
    range over about thirty seconds. Telling someone to check the wiring then
    sends them to the wrong place.
    """
    ok, msg = mon.diagnose([0] * 200, lead_off=True)
    assert not ok and "OUT -> VP" in msg

    ok, msg = mon.diagnose([0] * 200, lead_off=False)
    assert not ok and "Re-seat the electrodes" in msg and "OUT -> VP" not in msg

    ok, msg = mon.diagnose([4095] * 200, lead_off=False)
    assert not ok and "Re-seat the electrodes" in msg
