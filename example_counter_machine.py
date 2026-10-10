"""Smallest complete example: a 4-bit counter that counts once per tick while key A is held."""
from rtg_circuit import Layout
from rtg_logic import Machine

m = Machine(tau=0.25)                 # a tick lasts 4*tau = 1 s
L = m.L
m.run_when(m.key("A", tap=False))     # the registers only advance in ticks where A is down
n = m.reg(4)                          # 4-bit register, starts at 0
n.set(L.inc(n.q))                     # next value = value + 1
for i in range(4):
    m.output(f"Q{i}", n.q[i])         # labelled lights Q0..Q3
res = m.build(Layout())               # res.code = the creation code, res.report(), res.manifest
if __name__ == "__main__":
    open("example_counter_code.txt", "w").write(res.code + "\n")
    print(res.report().splitlines()[0])
    # quick check with the timed simulator
    from rtg_sim import load, structural_checks, Sim, TimedSim
    parts = load("example_counter_code.txt")
    sim = Sim(parts, *structural_checks(parts), manifest=res.manifest)
    ts = TimedSim(sim, 1, (0.002, 0.01))
    ts.schedule(0.5, "INIT", 1); ts.schedule(0.8, "INIT", 0); ts.schedule(1.0, "A", 1)
    seen = []
    for k in range(1, 120):
        ts.run_until(k * 0.25)
        o = ts.outputs(); v = sum(o[f"Q{i}"] << i for i in range(4))
        if not seen or seen[-1] != v: seen.append(v)
    print("counts seen:", seen)
