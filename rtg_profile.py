#!/usr/bin/env python3
"""Where do the parts and the signal changes go?   python3 rtg_profile.py code.txt [--tau 0.05] [--run RUN] [--keys W,S,T,G]

Runs the timed simulator on a creation code and reports
  * gate output changes per tick while idle (run off) and while the machine runs with random key presses,
  * the same weighted by how many gates each change drives (a rough measure of what the game has to process),
  * the gates that change most often.
Use it before and after every optimisation; the part count alone does not show activity."""
import sys, random
from collections import Counter
from rtg_sim import load, structural_checks, Sim, TimedSim


def activity(path, tau=0.05, run=None, keys=(), ticks=30, gate_delay=(0.0005, 0.002), seed=3, press=0.3, verbose=True):
    parts = load(path)
    sim = Sim(parts, *structural_checks(parts))
    res = {}
    for mode in ("idle", "running"):
        ts = TimedSim(sim, 1, gate_delay)
        cnt, start = Counter(), [1e9]
        orig = ts._push

        def push(t, kind, key, val=None, orig=orig, cnt=cnt, start=start):
            if kind == "gate" and t > start[0]:
                cnt[key] += 1
            return orig(t, kind, key, val)
        ts._push = push
        ts.schedule(10 * tau, "INIT", 1)
        ts.schedule(16 * tau, "INIT", 0)
        for k in list(keys) + ([run] if run else []):
            ts.schedule(0, k, 0)
        ts.run_until(40 * tau)
        if mode == "running" and run:
            ts.schedule(41 * tau, run, 1)
        ts.run_until(60 * tau)
        start[0] = ts.t
        rng = random.Random(seed)
        for i in range(ticks * 4):                       # a tick = 4 tau
            if mode == "running" and i % 4 == 1:
                for k in keys:
                    if rng.random() < press:
                        ts.schedule(ts.t + 1e-6, k, rng.random() < 0.5)
            ts.run_until(ts.t + tau)
        readers = ts.readers
        changes = sum(cnt.values()) / ticks
        weighted = sum(c * (1 + len(readers.get(("g", g), ()))) for g, c in cnt.items()) / ticks
        res[mode] = (changes, weighted, len(cnt), cnt.most_common(8))
        if verbose:
            print(f"{mode:<8} gate changes/tick {changes:7.1f}   fan-out weighted {weighted:8.1f}   "
                  f"gates that ever change {len(cnt)} of {len(sim.gates)}")
    return res


if __name__ == "__main__":
    a = sys.argv
    opt = lambda n, d=None: a[a.index(n) + 1] if n in a else d
    activity(a[1], float(opt("--tau", 0.05)), opt("--run"), [k for k in opt("--keys", "").split(",") if k])
