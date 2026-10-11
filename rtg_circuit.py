"""rtg_circuit - build Road to Gramby logic circuits from a netlist.

Typical use (see calculator_4bit.py and counter_4bit.py):

    c = Circuit()
    init = c.button("INIT")
    dr = Dual(c, init)                      # dual-rail helpers (combinational logic)
    seq = Seq(c, init, tau=0.25)            # memory + clock helpers (sequential logic)
    ...
    c.output("R0", some_gate)
    result = build(c, Layout(input_order=[...]))
    print(result.code)

Rules the netlist builder enforces (they come from how the game works):
  * Gates are AND / OR / NOT.  Every signal starts off, and a NOT gate only works
    after its input has been on once, so a NOT may only sit behind
    OR(something, INIT-button).  Pressing INIT once initialises all of them.
    Everything else is AND / OR (correct from the all-off state).
  * A cup takes one ball, so fan-out uses Wire / Splitter_N parts.  A splitter's
    base (handle) cup never carries a signal, so it is never used.
  * Memory is a loop through gates: a latch Q = OR(set, AND(Q, NOT(reset))).
    The reset side goes through NOT(OR(reset, INIT)), so pressing INIT also
    clears every latch.  Latches are the "switches" of the API (start off,
    switch on when set, off when reset).
  * Faster memory: c.toggle(trigger) is a real Switch part that flips on every rising edge of its
    trigger (starts off).  A ripple counter is just a chain of them.
  * Time comes from the Delayer part (its "Delay" property).  A Delayer is always
    plugged straight onto a dedicated one-input OR buffer (c.delay does this).

Physical layout produced by build():
  * Base A + a tower of Connectors (chained through top cups) carrying every
    gate on the connectors' side faces.
  * Base B + a second tower carrying the labelled InputSensors: inputs (with
    their Switch / Button on top) on one face, outputs (Wire + Light on top) on
    another face.
  * The two bases are linked (Base ball -> ShortStick -> Base cup), so the whole
    circuit is ONE rigid assembly and the wire ropes never pull on anything.
  * Gates are placed on the tower by a small optimiser so connected gates sit
    close together (ropes have a limited length).
"""
import math
import random
from collections import Counter, namedtuple
from dataclasses import dataclass
from typing import List, Optional

from rtg_api import RtgPart, Structure

# ----------------------------------------------------------------------------
# Netlist
# ----------------------------------------------------------------------------
SOURCES = ("switch", "button")
GATE_PART = {"AND": "Gate-AND", "OR": "Gate-OR", "NOT": "Gate-NOT"}
GATE_IN_CUPS = (2, 3)     # InputA, InputB cups of a gate; its output cup is 1
DELAY_PROPERTY = "Delay"  # name of the Delayer's delay property (seconds); the item's default is 2


class Node:
    __slots__ = ("id", "kind", "ins", "label", "value")

    def __init__(self, id, kind, ins=(), label=None, value=None):
        self.id, self.kind, self.ins, self.label, self.value = id, kind, tuple(ins), label, value


class Circuit:
    """A netlist of sources (switch / button), gates, delayers and labelled outputs."""

    def __init__(self):
        self.nodes: List[Node] = []
        self._cache = {}
        self.outputs = []          # (label, node)   (screen pixels are outputs too)
        self.screens = {}          # name -> (width, height)
        self.pixel_info = {}       # pixel label -> (screen name, x, y)
        self.latches = set()       # ids of latch output gates (memory loops), for depth analysis

    def _add(self, kind, ins=(), label=None, value=None):
        n = Node(len(self.nodes), kind, ins, label, value)
        self.nodes.append(n)
        return n

    def switch(self, label):
        return self._add("switch", label=label)

    def button(self, label, init=False):
        """A momentary button.  init=True marks THE initialisation button (the one NOT gates sit behind)."""
        return self._add("button", label=label, value="init" if init else None)

    def _gate(self, kind, a, b):
        assert a is not b, "a gate must not read the same signal twice"
        key = (kind,) + tuple(sorted((a.id, b.id)))   # identical gates are shared
        if key not in self._cache:
            self._cache[key] = self._add(kind, (a, b))
        return self._cache[key]

    def AND(self, a, b):
        return self._gate("AND", a, b)

    def OR(self, a, b):
        return self._gate("OR", a, b)

    def NOT(self, a):
        assert a.kind == "OR" and any(i.kind == "button" and i.value == "init" for i in a.ins), \
            "a NOT must sit behind OR(x, INIT button) so it can be initialised"
        key = ("NOT", a.id)
        if key not in self._cache:
            self._cache[key] = self._add("NOT", (a,))
        return self._cache[key]

    def buf(self, x):
        """A dedicated one-input OR (second input left empty): a private driver for one load."""
        return self._add("OR", (x,))

    def delay(self, x, seconds):
        """x delayed by `seconds` (Delayer part on its own buffer gate).  Both edges are delayed."""
        return self._add("delay", (self.buf(x),), value=seconds)

    def toggle(self, trigger):
        """Memory element made of the game's own Switch part: it starts off and FLIPS every time a
        signal arrives on it (rising edge of `trigger`).  The returned node is its state (a wire-like
        source, like an input switch).  The trigger goes through a private buffer gate whose output cup
        drives the switch's ball directly."""
        return self._add("toggle", (self.buf(trigger),))

    def output(self, label, node):
        assert node.kind in GATE_PART, "an output must be driven by a gate"
        self.outputs.append((label, node))

    def screen(self, name, width, height, pixel):
        """A width x height screen of lights.  pixel(x, y) -> gate node (lit when true) or None for
        a pixel that is never lit.  x grows to the viewer's right, y grows upwards.  Pixel labels are
        f"{name}_{x}_{y}" (used by build().manifest)."""
        assert not self.screens, "one screen per build"
        self.screens[name] = (width, height)
        for y in range(height):
            for x in range(width):
                n = pixel(x, y)
                if n is None:
                    continue
                lbl = f"{name}_{x}_{y}"
                self.pixel_info[lbl] = (name, x, y)
                self.outputs.append((lbl, n))

    def live(self, outputs=None) -> List[Node]:
        """Nodes that feed an output (dead logic is dropped).  Cycles (memory loops) are fine."""
        seen, stack = set(), [n for _, n in (self.outputs if outputs is None else outputs)]
        while stack:
            n = stack.pop()
            if n.id not in seen:
                seen.add(n.id)
                stack.extend(n.ins)
        return [n for n in self.nodes if n.id in seen]


# ----------------------------------------------------------------------------
# Dual-rail helpers: a value is D(t, f) = (gate that is on when true, gate that is on when false)
# ----------------------------------------------------------------------------
D = namedtuple("D", "t f")


class Dual:
    def __init__(self, circuit: Circuit, init=None):
        self.c, self.init = circuit, init

    def input(self, source: Node) -> D:
        """Switch -> dual-rail value; its complement goes through OR(.,INIT) + NOT."""
        assert self.init is not None, "Dual needs an INIT button for its NOT gates"
        return D(source, self.c.NOT(self.c.OR(source, self.init)))

    def inputs(self, prefix, n) -> List[D]:
        return [self.input(self.c.switch(f"{prefix}{i}")) for i in range(n)]

    def complement(self, node: Node) -> D:
        """Any signal -> dual-rail value (adds one OR + one NOT)."""
        return D(node, self.c.NOT(self.c.OR(node, self.init)))

    def and_(self, a, b):
        return D(self.c.AND(a.t, b.t), self.c.OR(a.f, b.f))

    def or_(self, a, b):
        return D(self.c.OR(a.t, b.t), self.c.AND(a.f, b.f))

    def not_(self, a):                                  # free: swap the rails
        return D(a.f, a.t)

    def xor(self, a, b):
        c = self.c
        return D(c.OR(c.AND(a.t, b.f), c.AND(a.f, b.t)),
                 c.OR(c.AND(a.t, b.t), c.AND(a.f, b.f)))

    def mux(self, sel, if0, if1):
        return self.or_(self.and_(sel, if1), self.and_(self.not_(sel), if0))

    def half_adder(self, a, b):
        return self.xor(a, b), self.and_(a, b)

    def full_adder(self, a, b, cin):
        """Returns (sum, carry_out), both dual-rail (unused rails are dropped).
        Uses  not(maj(a,b,c)) = a'b' + (a xor b)c'  to reuse gates."""
        c = self.c
        p1, p2 = c.AND(a.t, b.f), c.AND(a.f, b.t)
        p3, p4 = c.AND(a.t, b.t), c.AND(a.f, b.f)
        x = D(c.OR(p1, p2), c.OR(p3, p4))                       # a xor b
        q1, q2 = c.AND(x.t, cin.f), c.AND(x.f, cin.t)
        r1, r2 = c.AND(x.t, cin.t), c.AND(x.f, cin.f)
        s = D(c.OR(q1, q2), c.OR(r1, r2))                       # x xor cin
        cout = D(c.OR(p3, r1), c.OR(p4, q1))                    # maj, not(maj)
        return s, cout


# ----------------------------------------------------------------------------
# Sequential helpers: memory loops and a clock
# ----------------------------------------------------------------------------
Clock = namedtuple("Clock", "q d1 master slave")


class Seq:
    """Memory (latches, master/slave registers) and a free-running clock.

    tau is THE time unit: every Delayer uses it, so if the game ignored the Delay
    property the circuit would only run slower (all Delayers default to 2 s)."""

    def __init__(self, circuit: Circuit, init: Node, tau: float = 0.25):
        self.c, self.init, self.tau = circuit, init, tau

    # -- memory ---------------------------------------------------------------
    def latch_open(self) -> Node:
        """The output gate of a latch whose inputs are connected later (loops need this)."""
        return self.c._add("OR", ())

    def latch_close(self, q: Node, set_: Node, reset: Node) -> Node:
        """q = OR(set, AND(q, NOT(OR(reset, INIT)))): set wins, reset clears, otherwise holds.
        Starts off; pressing INIT clears it."""
        c = self.c
        q.ins = (set_, c.AND(q, c.NOT(c.OR(reset, self.init))))
        c.latches.add(q.id)
        return q

    def latch(self, set_: Node, reset: Node) -> Node:
        return self.latch_close(self.latch_open(), set_, reset)

    def ninit(self) -> Node:
        """Off until INIT has been pressed and released, then on (shared by every gated latch)."""
        if not hasattr(self, "_ninit"):
            self._ninit = self.c.NOT(self.c._add("OR", (self.init,)))
        return self._ninit

    def dlatch(self, q: Node, d: Node, en: Node) -> Node:
        """Gated D latch on the output gate `q` (from latch_open): follows `d` while `en` is high, holds
        otherwise; INIT clears it.  Q = OR(AND(d, en), AND(Q, nr)) with nr = AND(OR(d, NOT en), ninit):
        the hold term stays on whenever d is 1, so closing the window cannot glitch the latch, and only the
        true rail of `d` is needed (5 gates instead of 6 + a false rail)."""
        c = self.c
        nen = c.NOT(c.OR(en, self.init))
        nr = c.AND(c.OR(d, nen), self.ninit())
        q.ins = (c.AND(d, en), c.AND(q, nr))
        c.latches.add(q.id)
        return q

    # -- clock ---------------------------------------------------------------
    def clock(self) -> Clock:
        """Free-running clock, period 4*tau: a NOT + two Delayers in a ring.
        q rises at 0, d1 (= q delayed tau) at tau.  From these:
          slave  = AND(q, d1)               high from tau to 2tau   (slave latches open)
          master = AND(NOT q, NOT d1)       high from 3tau to 4tau  (master latches open)
        The two windows never overlap and are separated by gaps of tau, which is
        much longer than a gate delay.  Both start after INIT is pressed."""
        c, init, tau = self.c, self.init, self.tau
        x = c._add("OR", (init,))                 # becomes OR(d2, INIT)
        q = c.NOT(x)
        d1 = c.delay(q, tau)
        d2 = c.delay(d1, tau)
        x.ins = (d2, init)
        qn = c.NOT(c.OR(q, init))
        d1n = c.NOT(c.OR(d1, init))
        return Clock(q, d1, c.AND(qn, d1n), c.AND(q, d1))

    def oscillator(self, half_period: float = 0.5) -> Node:
        """Square wave: NOT + one Delayer in a ring.  Starts when INIT is released; the period is
        2*half_period plus the (small) gate delays of the ring."""
        c, init = self.c, self.init
        x = c._add("OR", (init,))                 # becomes OR(delayed q, INIT)
        q = c.NOT(x)
        x.ins = (c.delay(q, half_period), init)
        return q

    # -- flip-flop ------------------------------------------------------------
    def master_latch(self, d: D, enable: Node) -> D:
        """Gated D latch (transparent while `enable` is high); returns its dual-rail output."""
        c = self.c
        m = self.latch(c.AND(d.t, enable), c.AND(d.f, enable))
        return D(m, c.NOT(c.OR(m, self.init)))

    def slave_close(self, slave_q: Node, m: D, enable: Node) -> Node:
        """Connect the slave latch whose output gate was created earlier with latch_open()."""
        c = self.c
        return self.latch_close(slave_q, c.AND(m.t, enable), c.AND(m.f, enable))


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
CONNECTOR = 0.4                                  # connector cube size
FACE_DIR = {1: (1, 0, 0), 2: (-1, 0, 0), 3: (0, 0, -1), 6: (0, 0, 1)}   # connector side cups
BASE_TOP = 0.25                                   # Base top cup height
STICK = 1.0                                       # ShortStick length (lifts the towers off the floor)


@dataclass
class Layout:
    gate_faces: tuple = (1, 2, 3, 6)   # connector faces that carry gates
    in_face: int = 1                   # IO tower face for inputs
    out_face: int = 2                  # IO tower face for outputs
    in_pitch: int = 2                  # connectors between input sensors (sensor is wider than a connector)
    out_pitch: int = 3                 # ... between output sensors (lights are 0.96 wide)
    input_order: Optional[list] = None     # labels bottom -> top (default: creation order)
    output_order: Optional[list] = None
    link: str = "stick"                # bases linked by a "stick" (like the reference build) or "direct"
    rope_length: float = 10.0          # rope length, only used for the informational report: inside ONE rigid
                                       # structure a rope that is too short pulls on nothing
    reach_margin: float = 1.0          # warn when estimated rope > rope_length - margin
    gate_reach: float = 0.9            # how far a gate's cups stick out of its connector face
    io_reach: float = 1.0
    pixel_pitch: int = 2               # connectors between screen pixels: 2 x 0.4 = 0.8 < light width 0.96, so no gap
    pixel_reach: float = 0.6
    viewer: tuple = (0, 0, 1)          # the screen faces this way (world direction towards the player)
    lift: bool = True                  # start each tower with a ShortStick (gates hang below their connector)
    gate_connector_props: dict = None  # properties set on every connector of the GATE tower (default: OrientationZ = 180)
    optimise: bool = True
    effort: int = 300                 # annealing moves per gate
    slack: float = 1.1                # free gate slots (tower height factor) so the optimiser can move gates
    seed: int = 1


Site = namedtuple("Site", "att pos")      # an attachment (_Attach) and its estimated 3D position


def _dist(a, b):
    return math.dist(a, b)


class _Tower:
    def __init__(self, new, base, n, origin_xz, lift=True, props=None):
        self.base, self.origin, self.conns = base, origin_xz, []
        self.props = props or {}
        self.lift = STICK if lift else 0.0
        self._cup = base.get_cup(5)
        if lift:
            st = new("ShortStick")
            st.get_ball(2).attach(self._cup)
            self._cup = st.get_cup(1)
        self.grow(new, n)

    def grow(self, new, n):
        """Make the tower n connectors tall."""
        while len(self.conns) < n:
            con = new("Connector")
            for k, v in self.props.items():
                con.set_property(k, v)
            # towers only use the TOP cup (4): a connector faces the way it is plugged in
            con.get_ball(5).attach(self._cup if not self.conns else self.conns[-1].get_cup(4))
            self.conns.append(con)

    def pos(self, k, face=None, reach=0.0):
        y = BASE_TOP + self.lift + CONNECTOR / 2 + CONNECTOR * k
        dx, _, dz = FACE_DIR[face] if face else (0, 0, 0)
        out = CONNECTOR / 2 + reach if face else 0.0
        return (self.origin[0] + dx * out, y, self.origin[1] + dz * out)


def _column_index(n_items, pitch, height, i):
    span = pitch * (n_items - 1) + 1
    return (height - span) // 2 + pitch * i


def _build_screen(new, tower, k0, w, h, L):
    """Grow the screen from tower connector k0: a spine of connectors to each side (chained through
    their top cups, so they run straight out), and from every third spine connector a column (again a
    straight chain) whose side faces towards the viewer carry the pixels.  Orientation comes from the
    real attachment frames (rtg_geom).  Returns {(x, y): Site}."""
    import numpy as np
    from rtg_geom import Conn
    root = Conn(p=np.array(tower.pos(k0)))
    view = np.array(L.viewer, float)
    left = w // 2
    sites, n_columns = {}, {-1: left, 1: w - left}
    for side, face in ((-1, 2), (1, 1)):
        nc = n_columns[side]
        if not nc:
            continue
        spine, sgeo = [], []
        pp = L.pixel_pitch
        for i in range(pp * (nc - 1) + 1):
            con = new("Connector")
            if i == 0:
                con.get_ball(5).attach(tower.conns[k0].get_cup(face))
                sgeo.append(root.child(face))
            else:
                con.get_ball(5).attach(spine[-1].get_cup(4))
                sgeo.append(sgeo[-1].child(4))
            spine.append(con)
        up = sgeo[0].face_with_normal((0, 1, 0))
        assert up, "no spine face points up"
        for j in range(nc):
            xs = (left - 1 - j) if side == -1 else (left + j)
            si = pp * j
            geo, prev = sgeo[si].child(up), None
            for c in range(L.pixel_pitch * (h - 1) + 2):
                con = new("Connector")
                con.get_ball(5).attach(spine[si].get_cup(up) if prev is None else prev.get_cup(4))
                if c:
                    geo = geo.child(4)
                prev = con
                if c % L.pixel_pitch == 1 % L.pixel_pitch and (c - 1) // L.pixel_pitch < h and c >= 1:
                    f = geo.face_with_normal(view)
                    assert f, "no column face points at the viewer"
                    pos = geo.cup_pos(f) + view * L.pixel_reach
                    sites[(xs, (c - 1) // L.pixel_pitch)] = Site(con.get_cup(f), tuple(float(v) for v in pos))
    return sites


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Gate placement (keeps connected gates close together)
# ----------------------------------------------------------------------------
def _place(gates, edges, fixed, slot_pos, layout):
    """Assign each gate node to a slot index.  edges: (key, key) pairs, keys are
    node ids or ('out', label); fixed: key -> position for the IO sites."""
    lim = layout.rope_length - layout.reach_margin - 0.5

    def ecost(d):
        return d * d + (60.0 * (d - lim) ** 2 if d > lim else 0.0)

    ids = [g.id for g in gates]
    nbrs = {i: [] for i in ids}
    for a, b in edges:
        if a in nbrs:
            nbrs[a].append(b)
        if b in nbrs:
            nbrs[b].append(a)
    # 1. barycentre relaxation on height, then sort into slots
    ys = [p[1] for p in fixed.values()]
    y = {i: sum(ys) / len(ys) for i in ids}
    for _ in range(300):
        for i in ids:
            vals = [fixed[k][1] if k in fixed else y[k] for k in nbrs[i]]
            if vals:
                y[i] = sum(vals) / len(vals)
    order = sorted(ids, key=lambda i: y[i])
    slot_of = {i: s for s, i in enumerate(order)}          # slots are listed bottom -> top
    occupant = {s: i for i, s in slot_of.items()}
    if not layout.optimise:
        return slot_of
    # 2. simulated annealing with swap / move
    inc = {i: [] for i in ids}
    for e, (a, b) in enumerate(edges):
        for k in (a, b):
            if k in inc:
                inc[k].append(e)

    def P(k):
        return fixed[k] if k in fixed else slot_pos[slot_of[k]]

    def cost(es):
        return sum(ecost(_dist(P(edges[e][0]), P(edges[e][1]))) for e in es)

    rng = random.Random(layout.seed)
    n_slots = len(slot_pos)
    iters = max(60000, int(layout.effort * len(ids)))
    nb_ids = {i: [k for k in nbrs[i] if k not in fixed] for i in ids}
    fx_y = {i: [fixed[k] for k in nbrs[i] if k in fixed] for i in ids}
    per_level = max(1, len(layout.gate_faces))
    T0, T1 = 3.0, 0.003
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / iters)
        i = rng.choice(ids)
        s0 = slot_of[i]
        r = rng.random()
        if r < 0.6 and (nb_ids[i] or fx_y[i]):                  # guided: near a connected gate / IO site
            if nb_ids[i] and (not fx_y[i] or rng.random() < 0.8):
                tgt = slot_of[rng.choice(nb_ids[i])]
            else:
                py = rng.choice(fx_y[i])[1]
                tgt = min(max(int((py - slot_pos[0][1]) / CONNECTOR) * per_level, 0), n_slots - 1)
            s = min(max(tgt + rng.randint(-3 * per_level, 3 * per_level), 0), n_slots - 1)
        elif r < 0.85:                                           # local shuffle
            s = min(max(s0 + rng.randint(-8 * per_level, 8 * per_level), 0), n_slots - 1)
        else:
            s = rng.randrange(n_slots)
        if s == s0:
            continue
        j = occupant.get(s)
        es = set(inc[i]) | (set(inc[j]) if j is not None else set())
        before = cost(es)
        slot_of[i] = s
        if j is not None:
            slot_of[j] = s0
        delta = cost(es) - before
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            occupant[s] = i
            if j is not None:
                occupant[s0] = j
            else:
                occupant.pop(s0, None)
        else:
            slot_of[i] = s0
            if j is not None:
                slot_of[j] = s
    return slot_of


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------
@dataclass
class BuildResult:
    struct: Structure
    code: str
    parts: Counter
    ropes: list            # (description, estimated length)
    layout: Layout
    manifest: dict = None  # label -> 1-based part index of the Light (pixels and outputs), for the simulator

    def report(self):
        lim = self.layout.rope_length - self.layout.reach_margin
        lens = [l for _, l in self.ropes]
        over = [(d, l) for d, l in self.ropes if l > lim]
        lines = [f"parts: {sum(self.parts.values())}  {dict(self.parts)}",
                 f"code length: {len(self.code)} chars",
                 f"estimated rope lengths: max {max(lens):.1f}, mean {sum(lens) / len(lens):.1f} "
                 f"(rope {self.layout.rope_length:g}, warn above {lim:g}); "
                 f"{len(over)} of {len(lens)} above the warning level"]
        for d, l in sorted(self.ropes, key=lambda r: -r[1])[:5]:
            lines.append(f"   longest: {l:.1f}  {d}")
        return "\n".join(lines)


SIDE_BALLS = (5, 9, 7, 1)      # Splitter_k sides in order: Output1, Output2, Output3, Output4
SIDE_CUPS = (6, 10, 8, 2)


def _driver_outputs(circuit: Circuit):
    """Outputs whose gate also feeds other logic (or another output) get a private buffer gate,
    so the indicator wire has a driver cup of its own."""
    cons = Counter(s.id for n in circuit.live() for s in n.ins)
    seen, outs = set(), []
    for lbl, n in circuit.outputs:
        if cons[n.id] or n.id in seen:
            n = circuit.buf(n)
        seen.add(n.id)
        outs.append((lbl, n))
    return outs


def build(circuit: Circuit, layout: Optional[Layout] = None) -> BuildResult:
    L = layout or Layout()
    outs = _driver_outputs(circuit)
    pix = set(circuit.pixel_info)
    io_outs = [(l, n) for l, n in outs if l not in pix]
    live = circuit.live(outs)
    gates = [n for n in live if n.kind in GATE_PART]
    delays = [n for n in live if n.kind == "delay"]
    toggles = [n for n in live if n.kind == "toggle"]
    sources = [n for n in live if n.kind in SOURCES]
    cons = Counter(s.id for n in live for s in n.ins)
    alias = {}                                   # delay / toggle node -> the buffer gate it sits on
    for d in delays:
        b = d.ins[0]
        assert b.kind == "OR" and len(b.ins) == 1 and cons[b.id] == 1 and all(b is not n for _, n in outs), \
            "a Delayer needs its own private buffer gate"
        alias[d.id] = b.id
    tog_buf = {}
    for t in toggles:
        b = t.ins[0]
        assert b.kind == "OR" and len(b.ins) == 1 and cons[b.id] == 1 and all(b is not n for _, n in outs), \
            "a toggle switch needs its own private trigger buffer gate"
        tog_buf[t.id] = alias[t.id] = b.id
    in_labels = L.input_order or [n.label for n in sources]
    out_labels = L.output_order or [lbl for lbl, _ in io_outs]
    assert sorted(in_labels) == sorted(n.label for n in sources), "input_order must list every input"
    assert sorted(out_labels) == sorted(lbl for lbl, _ in io_outs), "output_order must list every output"

    n_faces = len(L.gate_faces)
    sensor_h = max(L.in_pitch * (len(in_labels) - 1) + 1, L.out_pitch * (len(out_labels) - 1) + 1, 1)
    gate_h = max(math.ceil(L.slack * (len(gates) + len(toggles)) / n_faces), 1)   # slots offered to the optimiser
    io_h = sensor_h                                                                  # IO tower: just the sensors
    screen_k0 = None
    if circuit.screens:
        (sw, sh), = circuit.screens.values()
        screen_k0 = sensor_h + 2                      # the screen's spine hangs just above the sensors
        io_h = screen_k0 + 1                          # ... and the columns grow from the spine, not from the tower
    sensor_span = sensor_h
    link_dist = 5.0 if L.link == "stick" else 4.0
    struct = Structure()

    def new(name):
        p = RtgPart(name)
        struct.add(p)
        return p

    base_g, base_io = new("Base"), new("Base")
    if L.link == "stick":
        stick = new("ShortStick")
        base_g.get_ball(3).attach(stick.get_cup(1))
        stick.get_ball(2).attach(base_io.get_cup(2))
    else:
        base_g.get_ball(3).attach(base_io.get_cup(2))
    gprops = {"OrientationZ": 180} if L.gate_connector_props is None else L.gate_connector_props
    gate_tower = _Tower(new, base_g, 0, (0.0, 0.0), L.lift, gprops)     # connectors are added once the gates are placed
    io_tower = _Tower(new, base_io, io_h, (0.0, link_dist), L.lift)

    # ---- IO sites (position + cup on the IO tower) ------------------------
    io_site, io_pos = {}, {}
    for face, labels, pitch in ((L.in_face, in_labels, L.in_pitch), (L.out_face, out_labels, L.out_pitch)):
        for i, lbl in enumerate(labels):
            k = _column_index(len(labels), pitch, sensor_span, i)
            io_site[lbl] = io_tower.conns[k].get_cup(face)
            io_pos[lbl] = io_tower.pos(k, face, L.io_reach)
    pixel_site = {}
    if screen_k0 is not None:
        scr = _build_screen(new, io_tower, screen_k0, sw, sh, L)
        for lbl, (_, x, y) in circuit.pixel_info.items():
            pixel_site[lbl] = scr[(x, y)]
            io_pos[lbl] = scr[(x, y)].pos

    # ---- gate placement ----------------------------------------------------
    slots = [(k, f) for k in range(gate_h) for f in L.gate_faces]
    slot_pos = [gate_tower.pos(k, f, L.gate_reach) for k, f in slots]
    fixed = {n.id: io_pos[n.label] for n in sources}
    edges = [(alias.get(s.id, s.id), g.id) for g in gates for s in g.ins]
    edges += [(tog_buf[t.id], t.id) for t in toggles]
    for lbl, n in outs:
        fixed[("out", lbl)] = io_pos[lbl]
        edges.append((n.id, ("out", lbl)))
    slot_of = _place(gates + toggles, edges, fixed, slot_pos, L)

    # ---- parts ---------------------------------------------------------------
    used_levels = {slots[s][0] for s in slot_of.values()}
    gate_tower.grow(new, max(used_levels) + 1 if used_levels else 1)    # the tower is exactly as tall as needed
    ropes = []
    gate_part, gate_site_pos = {}, {}
    for g in gates:
        p = new(GATE_PART[g.kind])
        s = slot_of[g.id]
        k, f = slots[s]
        p.get_ball(4).attach(gate_tower.conns[k].get_cup(f))     # handle ball -> connector face
        gate_part[g.id], gate_site_pos[g.id] = p, slot_pos[s]
    delay_part = {}
    for d in delays:                                             # Delayer sits right on its buffer gate
        b = d.ins[0]
        p = new("Delayer")
        p.set_property(DELAY_PROPERTY, d.value)
        p.get_ball(2).attach(gate_part[b.id].get_cup(1))
        delay_part[d.id] = p
        gate_site_pos[d.id] = gate_site_pos[b.id]

    def rope(desc, a, b):
        ropes.append((desc, _dist(a, b) + 0.0))

    tog_part = {}
    for t in toggles:                          # real Switch: handle on a connector face, ball on its trigger gate
        b = tog_buf[t.id]
        p = new("Switch")
        k, f = slots[slot_of[t.id]]
        p.get_ball(3).attach(gate_tower.conns[k].get_cup(f))
        p.get_ball(1).attach(gate_part[b].get_cup(1))
        rope(f"toggle{t.id} handle to trigger", slot_pos[slot_of[t.id]], gate_site_pos[b])
        gate_site_pos[t.id] = gate_site_pos[b]  # its output cup hangs at the trigger gate
        tog_part[t.id] = p

    def deliver(up: Site, load: Site, label):
        w = new("Wire")
        w.get_ball(1).attach(up.att)
        w.get_ball(3).attach(load.att)
        rope(f"{label} wire", up.pos, load.pos)

    def splitter(up: Site, loads: List[Site], carriers, label):
        """Splitter on `up`; loads[i] plugged into side i; carriers = {side index: [loads hanging on
        that side's cup]} (every side is plugged, no loose rope ends)."""
        sp = new(f"Splitter_{len(loads)}")
        sp.get_ball(3).attach(up.att)                              # handle ball; its cup 4 stays unused
        for i, ld in enumerate(loads):
            sp.get_ball(SIDE_BALLS[i]).attach(ld.att)
            rope(f"{label} splitter side {i + 1}", up.pos, ld.pos)
            if carriers.get(i):
                chain(Site(sp.get_cup(SIDE_CUPS[i]), ld.pos), carriers[i], label)

    def chain(up: Site, loads: List[Site], label):
        """Drive every load from `up` (a cup carrying the signal): a short wire, or a chain of splitters
        walking along the tower (each hop only reaches a few studs, ropes are limited)."""
        if len(loads) == 1:
            deliver(up, loads[0], label)
            return
        if len(loads) <= 4:
            splitter(up, sorted(loads, key=lambda s: s.pos[1]), {}, label)
            return
        ups = sorted([s for s in loads if s.pos[1] >= up.pos[1]], key=lambda s: s.pos[1])
        downs = sorted([s for s in loads if s.pos[1] < up.pos[1]], key=lambda s: -s.pos[1])
        if ups and downs:                                          # root splitter: one carrier each way
            splitter(up, [ups[0], downs[0]], {0: ups[1:], 1: downs[1:]}, label)
        else:
            seq = ups or downs
            splitter(up, seq[:4], {3: seq[4:]}, label)

    loads = {n.id: [] for n in live}
    for g in gates:
        for cup, src in zip(GATE_IN_CUPS, g.ins):
            loads[src.id].append(Site(gate_part[g.id].get_cup(cup), gate_site_pos[g.id]))

    out_label = {n.id: lbl for lbl, n in outs}
    delay_buf = set(alias.values())          # buffers that drive only a Delayer / a toggle switch
    manifest = {}
    light_of = {}
    for n in live:
        if n.kind in SOURCES:
            part = new("Switch" if n.kind == "switch" else "Button")
            sensor = new("InputSensor")
            sensor.set_property("ActivationKey", n.label)            # printed label
            sensor.get_ball(2).attach(io_site[n.label])
            part.get_ball(3).attach(sensor.get_cup(1))               # sits on the sensor
            ld = loads[n.id]
            if len(ld) == 1:
                host = ld[0]
            else:                                                    # side ball goes to the load nearest the rest
                cx = [sum(s.pos[i] for s in ld) / len(ld) for i in range(3)]
                host = min(ld, key=lambda s: _dist(s.pos, cx))
            part.get_ball(1).attach(host.att)
            rope(f"{n.label} {n.kind}", io_pos[n.label], host.pos)
            rest = [s for s in ld if s is not host]
            if rest:
                chain(Site(part.get_cup(2), host.pos), rest, n.label)
        elif n.kind == "delay":
            assert loads[n.id], "a Delayer must drive something"
            chain(Site(delay_part[n.id].get_cup(1), gate_site_pos[n.id]), loads[n.id], f"delay{n.id}")
        elif n.kind == "toggle":
            assert loads[n.id], "a toggle switch must drive something"
            chain(Site(tog_part[n.id].get_cup(2), gate_site_pos[n.id]), loads[n.id], f"toggle{n.id}")
        elif n.id in out_label:
            lbl = out_label[n.id]
            w, light = new("Wire"), new("Light")
            src = Site(gate_part[n.id].get_cup(1), gate_site_pos[n.id])
            w.get_ball(1).attach(src.att)                            # driven by the gate output
            if lbl in pixel_site:                                    # screen pixel: wire end on the grid
                w.get_ball(3).attach(pixel_site[lbl].att)
            else:                                                    # labelled output: wire end on a sensor
                sensor = new("InputSensor")
                sensor.set_property("ActivationKey", lbl)
                sensor.get_ball(2).attach(io_site[lbl])
                w.get_ball(3).attach(sensor.get_cup(1))
            light.get_ball(1).attach(w.get_cup(4))                  # light on that wire end
            light_of[lbl] = light
            rope(f"{lbl} output wire", src.pos, io_pos[lbl])
            assert not loads[n.id], "an output gate cannot also feed other gates"
        elif n.id in delay_buf:
            assert not loads[n.id]                                   # drives only its Delayer / toggle
        else:
            assert loads[n.id], f"gate {n.id} drives nothing"
            chain(Site(gate_part[n.id].get_cup(1), gate_site_pos[n.id]), loads[n.id], f"gate{n.id}")

    order = {id(p): i + 1 for i, p in enumerate(struct.get_all_parts())}
    manifest = {lbl: order[id(p)] for lbl, p in light_of.items()}
    parts = Counter(p._name for p in struct.get_all_parts())
    return BuildResult(struct, struct.code, parts, ropes, L, manifest)
