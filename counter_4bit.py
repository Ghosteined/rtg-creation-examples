#!/usr/bin/env python3
"""4-bit counter for Road to Gramby: counts +1 every second while switch A is on.

Inputs : A (switch), INIT (button - press ONCE after spawning, then leave it)
Outputs: Q0-Q3 (count, Q0 = least significant bit, wraps 15 -> 0), CLK (blinking clock light)

Built from 4 real Switch parts that flip on a rising edge (ripple counter) + a Delayer clock.

Usage: python3 counter_4bit.py [switch] [output_file]
"""
import random
import sys

from rtg_circuit import Circuit, Dual, Layout, Seq, build
import rtg_sim

N = 4
TAU = 0.25            # latch build: clock time unit (period 4*TAU = 1 s)
HOLD = 0.5            # a count must stay on the lights this long to count as "settled" (ripple glitches are shorter)
HALF = 0.5            # switch build: clock half period (period 1 s)


def make_switch_counter():
    """Down-counting ripple of toggle switches, displayed inverted = up counter.
    INIT also rising-edges the first switch, which ripples through all four -> they all start ON,
    i.e. the inverted display starts at 0."""
    c = Circuit()
    init = c.button("INIT", init=True)
    seq = Seq(c, init)
    a = c.switch("A")
    clk = seq.oscillator(HALF)
    # tick = rising edge of AND(A, short pulse after each clock rise).  (Sampling A in a latch was tried
    # and rejected: an asynchronous input hitting a latch just as its window closes can leave the latch
    # loop oscillating.)
    # a 0.1 s pulse right after each rising edge of the clock; A only lets it through while it is on, so
    # there is at most one tick per clock period and it is aligned to the clock.
    pulse = c.AND(clk, c.NOT(c.OR(c.delay(clk, 0.1), init)))
    trig = c.OR(c.AND(a, pulse), init)
    bits = []
    for i in range(N):
        t = c.toggle(trig)                    # flips on every rising edge of its trigger
        bits.append(t)
        trig = t                              # next switch flips when this one turns ON
    for i, t in enumerate(bits):
        c.output(f"Q{i}", c.NOT(c.OR(t, init)))
    c.output("CLK", clk)
    return c


MAKERS = {"switch": make_switch_counter}



# ---------------------------------------------------------------------------------------------
def count(o):
    return sum(o[f"Q{i}"] << i for i in range(N))


def settled(log, hold=0.3, end=None):
    """[(tick_time, count)]: counts that stay unchanged for >= hold seconds (ripple glitches ignored);
    tick_time is when the lights first started to change towards that count."""
    ser = []
    for t, o in log:
        v = count(o)
        if not ser or ser[-1][1] != v:
            ser.append((t, v))
    out, start = [], 0
    for k, (t, v) in enumerate(ser):
        nxt = ser[k + 1][0] if k + 1 < len(ser) else end
        if (nxt is None or nxt - t >= hold) and (not out or out[-1][1] != v):
            out.append((ser[start][0] if out else t, v))
            start = k + 1
    return out


def run(path, seed, gd, a_windows, T, init_at=0.5):
    parts = rtg_sim.load(path)
    sim = rtg_sim.Sim(parts, *rtg_sim.structural_checks(parts))
    ts = rtg_sim.TimedSim(sim, seed, gd)
    ts.schedule(init_at, "INIT", 1)
    ts.schedule(init_at + 0.3, "INIT", 0)
    for t0, t1 in a_windows:
        ts.schedule(t0, "A", 1)
        ts.schedule(t1, "A", 0)
    ts.run_until(T)
    return ts


def check(path, seed, gd, a_windows, T, verbose=False):
    ts = run(path, seed, gd, a_windows, T)
    log = ts.log
    errs = []
    first = min(a for a, _ in a_windows) - 0.1
    pre = [o for t, o in log if t <= first]
    base = count(pre[-1])
    if base != 0:
        errs.append(f"after INIT the count is {base}, not 0")
    log2 = [(first, pre[-1])] + [(t, o) for t, o in log if t > first]
    st = settled(log2, HOLD, T)
    if not st or st[0][0] > first:
        st = [(first, base)] + st
    # counts only go +1 (mod 16), never skip
    for (t0, v0), (t1, v1) in zip(st, st[1:]):
        if (v1 - v0) % 16 != 1:
            errs.append(f"jump {v0}->{v1} at t={t1:.2f}")
    # CLK rising edges while A is on (A must be on for the whole 0.25 s before / after) must count
    clk = [t for (t, o), (_, p) in zip(ts.log[1:], ts.log[:-1]) if o["CLK"] and not p["CLK"]]
    changes = [t for t, v in st[1:]]
    on = lambda t: any(a <= t for a, b in a_windows if a <= t <= b)
    for t in changes:
        # a count change needs A on within the 0.5 s before it
        if not any(a <= t and t - 0.5 <= b for a, b in a_windows):
            errs.append(f"count changed at t={t:.2f} while A was off")
    for tc in clk:
        if tc < 1.0 or tc > T - 1.5:
            continue
        for a, b in a_windows:
            if a + 0.1 <= tc <= b - 0.2 and not any(abs(t - tc) < 0.7 for t in changes):
                errs.append(f"clock edge at t={tc:.2f} with A on did not count")
    # spacing while A stays on
    for a, b in a_windows:
        ch = [t for t in changes if a + 0.6 < t < b]
        for x, y in zip(ch, ch[1:]):
            if not 0.9 <= y - x <= 1.0 + 6 * gd[1] + 0.15:
                errs.append(f"spacing {y - x:.2f}s at t={y:.2f}")
    if verbose:
        print("  settled:", [(round(t, 2), v) for t, v in st][:40])
    return errs, st


def verify(path):
    ok = True
    # 1. long steady run, wraps 15 -> 0 and holds when A goes off
    errs, st = check(path, 1, (0.01, 0.06), [(2.0, 21.5)], 30, verbose=True)
    seq = [v for _, v in st]
    print("steady run: count sequence", seq)
    want = list(range(0, len(seq)))
    wrapped = any(b == 0 and a == 15 for a, b in zip(seq, seq[1:]))
    print(f"  wraps 15->0: {wrapped}; final hold after A off: {seq[-1]} (stays)")
    ok &= not errs and wrapped
    for e in errs[:5]:
        print("  FAIL", e)
    # 2. many random seeds / delay ranges / random A toggling
    bad = 0
    runs = 0
    for seed in range(60):
        rng = random.Random(seed)
        gd = rng.choice([(0.005, 0.03), (0.01, 0.06), (0.02, 0.1)])
        t, wins = 2.0, []
        while t < 55:
            t0 = t + rng.uniform(0.6, 4)
            t1 = t0 + rng.choice([rng.uniform(0.6, 1.5), rng.uniform(0.8, 6)])
            wins.append((t0, t1))
            t = t1
        errs, _ = check(path, seed, gd, wins, 62)
        runs += 1
        if errs:
            bad += 1
            if bad <= 3:
                print(f"  seed {seed} gd={gd}: {errs[:3]}")
    print(f"{runs} random runs (random A toggling, gate delays up to 0.1 s): {'PASS' if not bad else f'{bad} FAIL'}")
    return ok and not bad


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "switch"
    path = sys.argv[2] if len(sys.argv) > 2 else f"counter_{kind}_code.txt"
    result = build(MAKERS[kind](), Layout(optimise=True))
    open(path, "w").write(result.code + "\n")
    print(result.report())
    print(f"written to {path}\n")
    parts = rtg_sim.load(path)
    rtg_sim.structural_checks(parts)
    print("structure OK (ONE rigid assembly, one ball per cup, gate inputs driven)")
    sys.exit(0 if verify(path) else 1)


if __name__ == "__main__":
    main()
