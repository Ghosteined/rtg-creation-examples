"""rtg_sim - independent checker / simulator for Road to Gramby creation codes.

Reads ONLY the base64 creation code (it does not import rtg_api or rtg_circuit),
so it checks the final artefact rather than the generator's intentions.

Rules applied (from the game's behaviour as described by the author):
  * wire-likes (Wire, Splitter_N, Switch, Button, Light): one part = one net.  It is
    active when it is an ON Switch / pressed Button, or when any of its BALLS is
    attached to a cup carrying an active signal.  Its own cups never feed it.
    Exception: a splitter's base (handle) cup (id 4) never sends a signal.
  * a cup carries a signal if it belongs to an active wire-like, or is the output
    cup (id 1) of a gate whose output is on.
  * gate inputs read the wire-like whose ball sits in the input cup (empty = 0).
  * every signal starts off; a NOT gate stays off until its input has been on once.

Structural checks: valid attachment ids, one ball per cup, every gate input driven,
nothing on a splitter base cup, one connected piece, and ONE RIGID ASSEMBLY (every
side of every wire-like is tied to the frame by an attachment path, because the
ropes between the sides are short).

Use:  verify("code.txt", func, init=("INIT",))   where func(inputs: dict) -> dict of outputs
"""
import base64
import itertools
import json
import random
import sys

SPL = {  # splitter variants: (ball ids, cup ids)
    "Splitter_1": ({3, 5}, {4, 6}),
    "Splitter_2": ({3, 5, 9}, {4, 6, 10}),
    "Splitter_3": ({3, 5, 7, 9}, {4, 6, 8, 10}),
    "Splitter_4": ({1, 3, 5, 7, 9}, {2, 4, 6, 8, 10}),
}
VALID = {  # part -> (ball ids, cup ids), taken from the game's item file
    "Wire": ({1, 3}, {2, 4}),
    "Switch": ({1, 3}, {2}),
    "Button": ({1, 3}, {2}),
    "Light": ({1}, set()),
    "Gate-AND": ({4}, {1, 2, 3}),
    "Gate-OR": ({4}, {1, 2, 3}),
    "Gate-NOT": ({4}, {1, 2}),
    "InputSensor": ({2}, {1}),
    "Connector": ({5}, {1, 2, 3, 4, 6}),
    "Base": ({3}, {1, 2, 4, 5, 6}),
    "ShortStick": ({2}, {1}),
    "Delayer": ({2}, {1}),
    **SPL,
}
WIRELIKE = {"Wire", "Switch", "Button", "Light", *SPL}
SOURCE = {"Switch", "Button"}
GATES = {"Gate-AND": (2, 3), "Gate-OR": (2, 3), "Gate-NOT": (2,)}
MUST_DRIVE = {"Gate-AND": (2, 3), "Gate-OR": (2,), "Gate-NOT": (2,)}   # an OR may leave its 2nd input open

# attachment id -> rigid body, for parts whose sides are only linked by a rope
SIDES = {
    "Wire": {1: "A", 2: "A", 3: "B", 4: "B"},
    "Switch": {1: "O", 2: "O", 3: "H"},
    "Button": {1: "O", 2: "O", 3: "H"},
}
for _n in SPL:
    SIDES[_n] = {3: "H", 4: "H", 5: "1", 6: "1", 9: "2", 10: "2", 7: "3", 8: "3", 1: "4", 2: "4"}


def load(path):
    data = json.loads(base64.b64decode(open(path).read().strip()))
    return [(e[0], e[1], e[2] if len(e) > 2 and isinstance(e[2], dict) else {}) for e in data]


def _body(parts, i, att):
    name = parts[i][0]
    return (i, SIDES[name][att]) if name in SIDES else (i, "-")


def structural_checks(parts):
    n = len(parts)
    ball_to_cup, cup_to_ball = {}, {}
    for i, (name, conns, _) in enumerate(parts):
        assert name in VALID, f"unknown part {name}"
        for b, c, t in conns:
            j = t - 1
            assert 0 <= j < n, "bad part index"
            b, c = int(b), int(c)
            assert b in VALID[name][0], f"{name} has no ball {b}"
            assert c in VALID[parts[j][0]][1], f"{parts[j][0]} has no cup {c}"
            assert (i, b) not in ball_to_cup, f"ball used twice: {name}#{i}:{b}"
            assert (j, c) not in cup_to_ball, f"cup used twice: {parts[j][0]}#{j}:{c}"
            ball_to_cup[(i, b)] = (j, c)
            cup_to_ball[(j, c)] = (i, b)
    for i, (name, _, _) in enumerate(parts):
        for c in MUST_DRIVE.get(name, ()):
            src = cup_to_ball.get((i, c))
            assert src and parts[src[0]][0] in WIRELIKE, f"gate #{i} input {c} not driven by a wire-like"
        if name == "Delayer":
            g = ball_to_cup.get((i, 2))
            assert g and parts[g[0]][0] in GATES and g[1] == 1, "a Delayer must sit on a gate output cup"
        if name == "Switch" and (i, 1) in ball_to_cup:
            j, c = ball_to_cup[(i, 1)]
            if parts[j][0] in GATES and c == 1:             # toggle switch: ball on a gate output
                assert (j, 1) in cup_to_ball and cup_to_ball[(j, 1)] == (i, 1)
        if name == "Light":
            assert (i, 1) in ball_to_cup, "light not plugged in"
        if name in SPL:
            assert (i, 4) not in cup_to_ball, "something is plugged into a splitter base cup"
    # rigid assembly: bodies are single parts, or the separate sides of rope-linked wire-likes
    bodies = set()
    for i, (name, _, _) in enumerate(parts):
        if name in SIDES:
            ids = VALID[name][0] | VALID[name][1]          # only the sides this variant really has
            bodies |= {(i, SIDES[name][a]) for a in ids}
        else:
            bodies.add((i, "-"))
    parent = {b: b for b in bodies}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, b), (j, c) in ball_to_cup.items():
        parent[find(_body(parts, i, b))] = find(_body(parts, j, c))
    roots = {find(b) for b in bodies}
    assert len(roots) == 1, f"{len(roots)} separate rigid assemblies (loose rope ends)"
    return ball_to_cup, cup_to_ball


class Sim:
    def __init__(self, parts, ball_to_cup, cup_to_ball, manifest=None):
        self.parts, self.ball_to_cup, self.cup_to_ball = parts, ball_to_cup, cup_to_ball
        inv = {v: k for k, v in (manifest or {}).items()}
        allsw = [i for i, p in enumerate(parts) if p[0] in SOURCE]
        # a Switch whose ball sits on a gate OUTPUT cup is a memory (toggle) switch: it flips on every
        # rising edge of that gate and its net then carries only its own state
        self.toggle_of = {}
        for i in allsw:
            tgt = ball_to_cup.get((i, 1))
            if parts[i][0] == "Switch" and tgt and parts[tgt[0]][0] in GATES and tgt[1] == 1:
                self.toggle_of[i] = tgt[0]
        self.sources = [i for i in allsw if i not in self.toggle_of]          # labelled inputs
        self.delayers = {i: ball_to_cup[(i, 2)][0] for i, p in enumerate(parts) if p[0] == "Delayer"}
        self.delay_value = {i: float(parts[i][2].get("Delay", 2.0)) for i in self.delayers}
        self.label = {i: parts[ball_to_cup[(i, 3)][0]][2]["ActivationKey"] for i in self.sources}
        self.lights = {}
        for i, p in enumerate(parts):
            if p[0] == "Light":
                wire, _ = ball_to_cup[(i, 1)]
                sensor, _ = ball_to_cup[(wire, 3)]
                if parts[sensor][0] == "InputSensor":
                    self.lights[parts[sensor][2]["ActivationKey"]] = i
                else:                                    # screen pixel: named by the build manifest
                    self.lights[inv.get(i + 1, f"light{i + 1}")] = i
        self.gates = [i for i, p in enumerate(parts) if p[0] in GATES]
        self.memo = {}
        self.gate_in = {g: [self._sources(cup_to_ball[(g, c)][0]) if (g, c) in cup_to_ball else set()
                            for c in GATES[parts[g][0]]] for g in self.gates}
        self.light_src = {k: self._sources(i) for k, i in self.lights.items()}
        self.reset()

    def _sources(self, i, stack=()):
        """('sw', i) / ('g', i) elements that can activate wire-like part i."""
        if i in self.memo:
            return self.memo[i]
        assert i not in stack, "feedback loop in the wire network"
        out = set()
        if self.parts[i][0] in SOURCE:
            out.add(("sw", i))
        for (pi, b), (j, c) in self.ball_to_cup.items():
            if pi != i or i in self.toggle_of:
                continue
            tp = self.parts[j][0]
            if tp in SPL and c == 4:
                continue                                  # splitter base cup: no signal
            if tp in WIRELIKE:
                out |= self._sources(j, stack + (i,))
            elif tp in GATES and c == 1:
                out.add(("g", j))
            elif tp == "Delayer" and c == 1:
                out.add(("d", j))
        self.memo[i] = out
        return out

    def reset(self):
        self.sw = {i: False for i in self.sources + list(self.toggle_of)}
        self.dout = {i: False for i in self.delayers}
        self.out = {g: False for g in self.gates}
        self.inited = {g: False for g in self.gates}

    def _read(self, srcs):
        return any(self.sw[i] if k == "sw" else self.out[i] if k == "g" else self.dout[i] for k, i in srcs)

    def update_gate(self, g):
        kind = self.parts[g][0]
        ins = [self._read(s) for s in self.gate_in[g]]
        if kind == "Gate-AND":
            new = all(ins)
        elif kind == "Gate-OR":
            new = any(ins)
        else:
            if ins[0]:
                self.inited[g] = True
            new = (not ins[0]) if self.inited[g] else False
        changed, self.out[g] = new != self.out[g], new
        return changed

    def settle(self, rng):
        assert not self.delayers and not self.toggle_of, "use TimedSim for circuits with memory / delays"
        while True:
            order = self.gates[:]
            rng.shuffle(order)
            if not any([self.update_gate(g) for g in order]):
                return

    def idx(self, label):
        return next(i for i in self.sources if self.label[i] == label)

    def set(self, **vals):
        for k, v in vals.items():
            self.sw[self.idx(k)] = bool(v)

    def outputs(self):
        return {k: int(self._read(s)) for k, s in self.light_src.items()}

    def pulse(self, labels, rng):
        self.set(**{k: 1 for k in labels})
        self.settle(rng)
        self.set(**{k: 0 for k in labels})
        self.settle(rng)


class TimedSim:
    """Event-driven simulation with real time.  Every gate output change takes a random time in
    gate_delay (transport delay, so glitches are NOT filtered - pessimistic), toggle switches flip a
    random time after the rising edge on their gate, a Delayer copies its gate exactly `Delay` seconds
    later.  Input switches are driven by schedule()."""

    def __init__(self, sim, seed=0, gate_delay=(0.01, 0.06)):
        import heapq
        self.hq = heapq
        self.s, self.rng, self.gd = sim, random.Random(seed), gate_delay
        sim.reset()
        self.t, self.n, self.q = 0.0, 0, []
        self.sched = {g: False for g in sim.gates}
        self.last_t = {g: 0.0 for g in sim.gates}
        self.readers = {}
        for g, lists in sim.gate_in.items():
            for srcs in lists:
                for src in srcs:
                    self.readers.setdefault(src, set()).add(g)
        self.dl_of = {}
        for d, g in sim.delayers.items():
            self.dl_of.setdefault(g, []).append(d)
        self.tg_of = {}
        for sw, g in sim.toggle_of.items():
            self.tg_of.setdefault(g, []).append(sw)
        self.watch = set().union(*sim.light_src.values()) if sim.light_src else set()
        self.prev = sim.outputs()
        self.log = [(0.0, dict(self.prev))]
        for g in sim.gates:
            self._eval(g)

    def _push(self, t, kind, key, val=None):
        self.n += 1
        self.hq.heappush(self.q, (t, self.n, kind, key, val))

    def _target(self, g):
        s = self.s
        kind = s.parts[g][0]
        ins = [s._read(x) for x in s.gate_in[g]]
        if kind == "Gate-AND":
            return all(ins)
        if kind == "Gate-OR":
            return any(ins)
        if ins[0]:
            s.inited[g] = True
        return (not ins[0]) if s.inited[g] else False

    def _eval(self, g):
        v = self._target(g)
        if v != self.sched[g]:
            self.sched[g] = v
            t = max(self.t + self.rng.uniform(*self.gd), self.last_t[g] + 1e-6)
            self.last_t[g] = t
            self._push(t, "gate", g, v)

    def schedule(self, t, label, value):
        self._push(t, "input", self.s.idx(label), bool(value))

    def _notify(self, src):
        for g in self.readers.get(src, ()):
            self._eval(g)

    def run_until(self, T):
        s = self.s
        budget = 3_000_000
        while self.q and self.q[0][0] <= T:
            budget -= 1
            if budget < 0:
                raise RuntimeError(f"event storm (a feedback loop is oscillating) near t={self.t:.2f}")
            t, _, kind, key, val = self.hq.heappop(self.q)
            self.t = t
            if kind == "input":
                if s.sw[key] != val:
                    s.sw[key] = val
                    self._notify(("sw", key))
            elif kind == "gate":
                s.out[key] = val
                self._notify(("g", key))
                for d in self.dl_of.get(key, ()):
                    self._push(t + s.delay_value[d], "delay", d, val)
                if val:
                    for sw in self.tg_of.get(key, ()):
                        self._push(t + self.rng.uniform(*self.gd), "flip", sw)
            elif kind == "delay":
                s.dout[key] = val
                self._notify(("d", key))
            elif kind == "flip":
                s.sw[key] = not s.sw[key]
                self._notify(("sw", key))
            src = ("g" if kind == "gate" else "d" if kind == "delay" else "sw", key)
            if src not in self.watch:
                continue
            cur = s.outputs()
            if cur != self.prev:
                self.prev = cur
                self.log.append((t, dict(cur)))
        self.t = T

    def outputs(self):
        return self.s.outputs()


def verify(path, func, init=("INIT",), seed=1234, max_exhaustive=14):
    """Check structure, then compare the circuit with func(inputs)->outputs.  Returns True if all pass."""
    parts = load(path)
    b2c, c2b = structural_checks(parts)
    sim = Sim(parts, b2c, c2b)
    rng = random.Random(seed)
    data_in = sorted(sim.label[i] for i in sim.sources if sim.label[i] not in init)
    print(f"structure OK: {len(parts)} parts, ONE rigid assembly, one ball per cup, all gate inputs driven, "
          "nothing on a splitter base cup")
    print("inputs :", data_in, "+ init", list(init))
    print("outputs:", sorted(sim.lights))
    ok = True

    def expect(vals):
        return {k: int(v) for k, v in func(vals).items()}

    def check(vals):
        return sim.outputs() == expect(vals)

    # without pressing INIT the circuit is not usable (informational)
    sim.reset()
    vals = {k: rng.randrange(2) for k in data_in}
    sim.set(**vals)
    sim.settle(rng)
    print(f"before INIT: {'already right (lucky pattern)' if check(vals) else 'outputs wrong, as expected'}")

    # 1. exhaustive truth table after one INIT press, random order
    sim.reset()
    sim.pulse(init, rng)
    combos = list(itertools.product((0, 1), repeat=len(data_in)))
    if len(data_in) > max_exhaustive:
        combos = rng.sample(combos, 1 << max_exhaustive)
    rng.shuffle(combos)
    bad = 0
    for combo in combos:
        vals = dict(zip(data_in, combo))
        sim.set(**vals)
        sim.settle(rng)
        if not check(vals):
            bad += 1
            if bad <= 5:
                print("  MISMATCH", vals, sim.outputs(), expect(vals))
    print(f"all {len(combos)} input combinations: {'PASS' if not bad else f'{bad} FAIL'}")
    ok &= not bad

    # 2. asynchronous stress: inputs flip (and INIT bounces) while gates are mid-update
    bad = 0
    ids = [sim.idx(k) for k in data_in]
    init_ids = [sim.idx(k) for k in init]
    for _ in range(300):
        for _ in range(rng.randint(20, 400)):
            r = rng.random()
            if r < 0.2:
                i = rng.choice(ids)
                sim.sw[i] = not sim.sw[i]
            elif r < 0.25 and init_ids:
                i = rng.choice(init_ids)
                sim.sw[i] = not sim.sw[i]
            else:
                sim.update_gate(rng.choice(sim.gates))
        for i in init_ids:
            sim.sw[i] = False
        sim.settle(rng)
        bad += not check({sim.label[i]: int(sim.sw[i]) for i in ids})
    print(f"300 asynchronous stress runs: {'PASS' if not bad else f'{bad} FAIL'}")
    ok &= not bad

    # 3. from random switch positions, press INIT once, then use it
    bad = 0
    for _ in range(100):
        sim.reset()
        for i in ids:
            sim.sw[i] = rng.random() < 0.5
        sim.settle(rng)
        sim.pulse(init, rng)
        vals = {k: rng.randrange(2) for k in data_in}
        sim.set(**vals)
        sim.settle(rng)
        bad += not check(vals)
    print(f"100 runs of 'press INIT once' from random switch positions: {'PASS' if not bad else f'{bad} FAIL'}")
    ok &= not bad
    return ok


if __name__ == "__main__":
    structural_checks(load(sys.argv[1]))
    print("structure OK")
