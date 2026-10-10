"""rtg_logic - describe a circuit with memory, a screen and key inputs WITHOUT touching parts.

    m = Machine(tau=0.4)                    # owns the INIT button, the clock and the circuit
    up, down = m.key("W"), m.key("S")       # held-at-this-tick values (safe to read anywhere)
    x = m.reg(3, init=3)                    # 3-bit register, starts at 3
    L = m.L                                 # logic helpers: and_/or_/not_/xor/mux, words, compare
    x.set(L.mux_w(up, x.q, L.inc(x.q)), when=m.key("RUN"))      # next value (when = enable)
    m.screen("SCR", 8, 6, lambda px, py: L.eq(x.q, px))        # pixel (px,py) lit when True
    m.output("MSB", x.q[2])                 # a labelled light
    result = m.build(Layout())              # -> code, report, manifest

Values are dual-rail pairs D(t, f) (a gate that is on when true, one when false), so NOT is free.
Constants are folded away (L.ONE / L.ZERO / L.const).  A word is a python list of D, LSB first.

How time works (see Machine): every register is a master/slave pair of gate latches clocked by a
Delayer ring (period 4*tau).  Everything between two registers is plain AND/OR logic that has to
settle within about 1.5*tau: pick tau from  m.depth() * (slowest gate delay).  Keys go through a
Switch that flips on a rising edge, so an asynchronous key press can never upset a latch.
"""
from rtg_circuit import Circuit, D, Seq, Node, build


class Logic:
    def __init__(self, circuit, init):
        self.c, self.init = circuit, init
        self.ONE, self.ZERO = D(True, False), D(False, True)

    # ---- rails (None-free constant folding: True / False are constants) ----------------------
    def A(self, x, y):
        if x is False or y is False:
            return False
        if x is True:
            return y
        if y is True or x is y:
            return x
        return self.c.AND(x, y)

    def O(self, x, y):
        if x is True or y is True:
            return True
        if x is False:
            return y
        if y is False or x is y:
            return x
        return self.c.OR(x, y)

    # ---- bits ----------------------------------------------------------------------------------
    def const(self, v):
        return self.ONE if v else self.ZERO

    def not_(self, a):
        return D(a.f, a.t)

    def and_(self, a, b):
        return D(self.A(a.t, b.t), self.O(a.f, b.f))

    def or_(self, a, b):
        return D(self.O(a.t, b.t), self.A(a.f, b.f))

    def xor(self, a, b):
        return D(self.O(self.A(a.t, b.f), self.A(a.f, b.t)), self.O(self.A(a.t, b.t), self.A(a.f, b.f)))

    def mux(self, sel, if0, if1):
        """sel ? if1 : if0 (bits)"""
        return D(self.O(self.A(sel.t, if1.t), self.A(sel.f, if0.t)),
                 self.O(self.A(sel.t, if1.f), self.A(sel.f, if0.f)))

    def _tree(self, bits, op, unit):
        bits = list(bits)
        if not bits:
            return unit
        while len(bits) > 1:                       # balanced tree: depth log2(n) instead of n
            bits = [op(bits[i], bits[i + 1]) if i + 1 < len(bits) else bits[i] for i in range(0, len(bits), 2)]
        return bits[0]

    def all_(self, bits):
        return self._tree(bits, self.and_, self.ONE)

    def any_(self, bits):
        return self._tree(bits, self.or_, self.ZERO)

    # ---- words (lists of bits, LSB first) -------------------------------------------------------
    def word(self, value, width):
        return [self.const((value >> i) & 1) for i in range(width)]

    def mux_w(self, sel, if0, if1):
        return [self.mux(sel, a, b) for a, b in zip(if0, if1)]

    def eq(self, w, k):
        """word == constant k"""
        return self.all_([b if (k >> i) & 1 else self.not_(b) for i, b in enumerate(w)])

    def eq_w(self, a, b):
        return self.all_([self.not_(self.xor(x, y)) for x, y in zip(a, b)])

    def ge(self, w, k):
        """word >= constant k"""
        r = self.ONE
        for i, b in enumerate(w):
            r = self.and_(b, r) if (k >> i) & 1 else self.or_(b, r)
        return r

    def lt(self, w, k):
        return self.not_(self.ge(w, k))

    def inc(self, w):
        out, carry = [], self.ONE
        for b in w:
            out.append(self.xor(b, carry))
            carry = self.and_(b, carry)
        return out

    def dec(self, w):
        out, borrow = [], self.ONE
        for b in w:
            out.append(self.xor(b, borrow))
            borrow = self.and_(self.not_(b), borrow)
        return out

    def add_w(self, a, b):
        out, carry = [], self.ZERO
        for x, y in zip(a, b):
            p = self.xor(x, y)
            out.append(self.xor(p, carry))
            carry = self.or_(self.and_(x, y), self.and_(p, carry))
        return out


class Reg:
    """A register (list of bits).  q = current value (word); set(next, when=enable) = next value."""

    def __init__(self, m, width, init=0, free=False):
        self.m, self.width, self.init, self.free = m, width, init, free
        self._slaves = [m.seq.latch_open() for _ in range(width)]
        self._inv = [(init >> i) & 1 for i in range(width)]        # a bit that starts at 1 is stored inverted
        c, ini = m.c, m.init
        self.q = []
        for s, inv in zip(self._slaves, self._inv):
            d = D(s, c.NOT(c.OR(s, ini)))
            self.q.append(m.L.not_(d) if inv else d)
        self._done = False

    def set(self, nxt, when=None):
        m, L, c = self.m, self.m.L, self.m.c
        assert not self._done, "register set twice"
        assert len(nxt) == self.width
        if when is not None:                       # (prefer Machine.run_when: it costs no logic depth)
            nxt = L.mux_w(when, self.q, nxt)
        self._done = True
        slave_window = m.clk.slave if (self.free or m._slave_en is None) else m._slave_en
        for i, (s, inv) in enumerate(zip(self._slaves, self._inv)):
            d = L.not_(nxt[i]) if inv else nxt[i]
            assert isinstance(d.t, Node) and isinstance(d.f, Node), "register input is constant"
            mq = m.seq.latch(c.AND(d.t, m.clk.master), c.AND(d.f, m.clk.master))      # master: open in window 3
            M = D(mq, c.NOT(c.OR(mq, m.init)))
            m.seq.latch_close(s, c.AND(M.t, slave_window), c.AND(M.f, slave_window))    # slave: open in window 1


class Machine:
    def __init__(self, tau=0.25, pulse=None):
        self.c = Circuit()
        self.init = self.c.button("INIT", init=True)
        self.L = Logic(self.c, self.init)
        self.seq = Seq(self.c, self.init, tau)
        self.tau, self.pulse_w = tau, (pulse if pulse is not None else tau / 3)
        self.clk = self.seq.clock()
        self.regs, self._pulse, self._slave_en = [], None, None

    def run_when(self, bit):
        """The game state only advances in ticks where `bit` is true (pause otherwise).  Done by gating the
        slave window with a master-latched copy of the bit - zero extra logic depth.  Call before the
        .set() of the registers it should gate; registers made with free=True are not gated."""
        assert self._slave_en is None, "run_when called twice"
        c = self.c
        mq = self.seq.latch(c.AND(bit.t, self.clk.master), c.AND(bit.f, self.clk.master))
        self._slave_en = c.AND(mq, self.clk.slave)

    def reg(self, width, init=0, free=False):
        r = Reg(self, width, init, free)
        self.regs.append(r)
        return r

    def key(self, label, tap=True):
        """A key (Button on a labelled sensor) as 'wants to move at this clock tick': true when the key is
        held down at the tick, or (tap=True) when it was pressed at any moment since the previous tick,
        so even a quick tap is never lost.  The result is stable for the whole tick."""
        c, L = self.c, self.L
        if self._pulse is None:
            # a short pulse shortly after each cycle starts (inside the slave window, far from the master
            # window that closes at the cycle start): keys can only change state there, never during a latch
            # window edge
            d1 = self.clk.d1
            self._pulse = c.AND(d1, c.NOT(c.OR(c.delay(d1, self.pulse_w), self.init)))
        btn = c.button(label)

        def parity_change(toggle):
            """bit that is true in the tick in which `toggle` flipped (it only ever flips near the pulse)"""
            td = D(toggle, c.NOT(c.OR(toggle, self.init)))
            prev = self.reg(1, free=True)
            prev.set([td])
            return L.xor(td, prev.q[0])

        held = c.toggle(c.AND(btn, self._pulse))              # flips once per tick in which the key is down
        want = parity_change(held)
        if tap:
            press = c.toggle(btn)                              # flips at every press, at any moment
            seen_buf = c._add("OR", ())
            seen = c._add("toggle", (seen_buf,))               # copies `press` at the pulse (so it only flips there)
            pd, sd = D(press, c.NOT(c.OR(press, self.init))), D(seen, c.NOT(c.OR(seen, self.init)))
            seen_buf.ins = (c.AND(self._pulse, L.xor(pd, sd).t),)
            want = L.or_(want, parity_change(seen))
        return want

    def output(self, label, bit):
        assert isinstance(bit.t, Node), "output is constant"
        self.c.output(label, bit.t)

    def screen(self, name, width, height, pixel):
        def node(x, y):
            b = pixel(x, y)
            if b is None or b.t is False:
                return None
            assert isinstance(b.t, Node), "a pixel cannot be always on"
            return b.t
        self.c.screen(name, width, height, node)

    def depth(self):
        """Longest chain of gates between memory/inputs and any gate (the settling time is about
        depth x gate delay; keep it under ~1.5 x tau)."""
        c = self.c
        memo = {}

        def d(n):
            if n.id in memo:
                return memo[n.id]
            if n.kind in ("switch", "button", "toggle", "delay") or n.id in c.latches:
                memo[n.id] = 0
                return 0
            memo[n.id] = 0
            memo[n.id] = 1 + max([d(i) for i in n.ins] or [0])
            return memo[n.id]

        return max(d(n) for n in c.live())

    def suggest_tau(self, gate_delay):
        """Smallest clock unit that is safe if the slowest gate needs `gate_delay` seconds: the logic must
        settle inside two clock units (a tick lasts 4 x tau).  Measured margin included."""
        return round(0.6 * (self.depth() + 6) * gate_delay, 3)

    def finish(self):
        assert all(r._done for r in self.regs), "a register has no next value (call .set)"

    def build(self, layout=None):
        self.finish()
        self.c.output("CLK", self.clk.q)
        return build(self.c, layout)
