#!/usr/bin/env python3
"""Check script for pong.py: builds the game, then runs the timed simulation of the FINAL creation code
(random gate delays, random key presses, quick taps) and compares every frame with a python reference model.
Not needed to build the game; only to trust it.   Usage: python3 pong_check.py [outfile] [-v]"""
import sys
from rtg_circuit import Layout
from pong import game, W, H, PAD, MID_X, MID_Y


class Ref:
    """Reference model of one tick (python), same rules."""
    def __init__(self):
        self.bx, self.by, self.pl, self.pr = MID_X - 1, MID_Y, (H - PAD) // 2, (H - PAD) // 2
        self.hr, self.vu, self.vf, self.vh, self.ph = 1, 1, 0, 1, 0

    def step(self, k, run):
        if not run:
            return
        def pad(p, up, dn):
            if up and not dn and p < H - PAD: return p + 1
            if dn and not up and p >= 1: return p - 1
            return p
        pl, pr = pad(self.pl, k["W"], k["S"]), pad(self.pr, k["T"], k["G"])
        hr, vu, vf, vh, ph, bx, by = self.hr, self.vu, self.vf, self.vh, self.ph, self.bx, self.by
        mv = vf or (vh and ph)
        wall = mv and ((vu and by == H - 1) or (not vu and by == 0))
        vu1 = vu ^ int(bool(wall))
        ny = by + ((1 if vu1 else -1) if mv else 0)
        at_r, at_l = hr and bx == W - 2, (not hr) and bx == 1
        hr2, vf2, vh2, vu2, ph2 = hr, vf, vh, vu1, 1 - ph
        miss = None
        if at_r or at_l:
            d = ny - (self.pr if at_r else self.pl)
            ph2 = 0
            if -1 <= d <= 3:
                hr2 = 1 - hr
                vf2, vh2, vu2 = int(d in (-1, 3)), int(d in (0, 2)), int(d in (2, 3))
                if d == 1 and not vf and not vh:                   # arrived flat -> not flat again
                    vh2, vu2 = 1, int(ny < MID_Y)
            else:
                miss = "r" if at_r else "l"
        nx = bx + (1 if hr2 else -1)
        if miss:
            nx, ny = (MID_X if miss == "l" else MID_X - 1), MID_Y
            hr2, vf2, vh2, vu2 = (1 if miss == "r" else 0), 0, 0, 0
        self.bx, self.by, self.hr, self.vu, self.vf, self.vh, self.ph = nx, ny, int(hr2), int(vu2), int(vf2), int(vh2), int(ph2)
        self.pl, self.pr = pl, pr

    def frame(self):
        f = set()
        for y in range(H):
            if self.pl <= y < self.pl + PAD: f.add((0, y))
            if self.pr <= y < self.pr + PAD: f.add((W - 1, y))
        f.add((self.bx, self.by))
        return f


def show(frame):
    return "\n".join("".join("#" if (x, y) in frame else "." for x in range(W)) for y in reversed(range(H)))


def verify(path, manifest, tau, cycles=60, seed=1, gd=(0.005, 0.03), verbose=False):
    """Timed simulation of the final code: random gate delays, random key presses, compared tick by tick
    with the reference model.  Returns the number of mismatching frames."""
    import random
    from rtg_sim import load, structural_checks, Sim, TimedSim
    parts = load(path)
    sim = Sim(parts, *structural_checks(parts), manifest=manifest)
    ts = TimedSim(sim, seed, gd)
    rng = random.Random(seed)
    keys = ["W", "S", "T", "G", "RUN"]
    cur = {k: 0 for k in keys}
    ts.schedule(10 * tau, "INIT", 1)
    ts.schedule(16 * tau, "INIT", 0)
    for k in keys:
        ts.schedule(0.0, k, cur[k])
    ref, bad, t = Ref(), 0, 0.0
    prev_clk, c, sample_at, change_at = 0, -1, None, None
    last_edge = 0.0
    K, E, pending, release = [], [], {k: 0 for k in keys}, []        # levels / presses seen at each tick's pulse
    while c < cycles:
        t += tau / 10
        ts.run_until(t)
        o = ts.outputs()
        if t > 30 * tau and t - last_edge > 12 * tau:
            print(f"CLOCK STOPPED at t={t:.1f} (cycle {c})")
            return 10 ** 6
        if o["CLK"] and not prev_clk:          # a clock cycle starts
            last_edge = t
            c += 1
            K.append(dict(cur))
            E.append(dict(pending))
            pending = {k: 0 for k in keys}
            sample_at, change_at = t + 2.9 * tau, t + 2.0 * tau
        prev_clk = o["CLK"]
        for rel in [r for r in release if r[0] <= t]:                  # end of a tap
            release.remove(rel)
            cur[rel[1]] = 0
            ts.schedule(t, rel[1], 0)
        if change_at is not None and t >= change_at:      # key actions for the next tick (after this tick's pulse)
            change_at = None
            for k in keys:
                r = rng.random()
                if k == "RUN":
                    if r < 0.12:
                        cur[k] ^= 1
                    if c < 3:
                        cur[k] = 0
                    elif c < 5:
                        cur[k] = 1
                    ts.schedule(t, k, cur[k])
                elif cur[k] == 0 and r < 0.25:                 # a quick tap: down and up again within the tick
                    cur[k] = 1
                    pending[k] = 1
                    ts.schedule(t, k, 1)
                    release.append((t + tau, k))
                elif r < 0.5:
                    if cur[k] == 0:
                        pending[k] = 1
                    cur[k] ^= 1
                    ts.schedule(t, k, cur[k])
        if sample_at is not None and t >= sample_at:      # screen is steady: compare with the model
            sample_at = None
            if c >= 1:
                want = {k: K[c - 1][k] or E[c - 1][k] for k in "WSTG"}
                ref.step(want, K[c - 1]["RUN"])
            got = {(x, y) for x in range(W) for y in range(H) if o.get(f"SCR_{x}_{y}")}
            if got != ref.frame():
                bad += 1
                if bad <= 3:
                    print(f"MISMATCH in cycle {c} (t={t:.2f}), keys {K[c - 1] if c else '-'} presses {E[c - 1] if c else '-'}")
                    print("got:\n" + show(got) + "\nexpected:\n" + show(ref.frame()))
            if verbose:
                print(f"--- cycle {c} keys {K[c - 1] if c else '-'}\n{show(got)}")
    return bad


def main():
    from rtg_project import run
    path = next((a for a in sys.argv[1:] if not a.startswith("-")), "pong_code.txt")
    m = game()
    res = run(m, path)
    gmax = 2 * m.tau / (m.depth() + 6) / 1.2          # slowest gate this clock setting tolerates
    seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 1
    bad = 0
    for seed in range(1, seeds + 1):                  # --seeds N: N random seeds x 3 gate-delay ranges
        for gd in ([(gmax / 5, gmax)] if seeds == 1 else [(gmax / 5, gmax), (gmax * 0.9, gmax), (gmax / 10, gmax / 5)]):
            print(f"simulating seed {seed} with gate delays {gd[0] * 1000:.1f}-{gd[1] * 1000:.1f} ms")
            bad += verify(path, res.manifest, m.tau, gd=gd, cycles=150 if seeds == 1 else 120, seed=seed,
                          verbose="-v" in sys.argv)
    print("timed simulation vs reference model:", "PASS" if not bad else f"{bad} mismatching frames")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
