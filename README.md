# Road to Gramby circuit toolchain

Build working circuits (calculators, counters, games with a light screen) for *Road to Gramby*
and get a creation code you can paste into the game.

Depends on [rtg_api_py](https://github.com/Ghosteined/rtg_api_py) (creation-code format) and numpy.

```
pip install -r requirements.txt
python3 pong.py                # writes pong_code.txt (paste into the game)
python3 pong_check.py          # builds + timed simulation vs. a python reference model
python3 example_counter_machine.py
```

## Layers (a project script only uses the top one)

| file | role |
|---|---|
| `pong.py`, `example_counter_machine.py` | **project scripts**: only define inputs, memory, rules, screen |
| `rtg_project.py` | `run(machine, "out.txt")`: build, structure check, write code |
| `rtg_logic.py` | `Machine`, `Reg`, `Logic` (`m.key`, `m.reg`, `L.inc`, `L.mux_w`, `m.screen`...) |
| `rtg_circuit.py` | gates -> dual-rail circuit, latches/clock, tower layout, placement, screen geometry, creation code |
| `rtg_geom.py` | exact connector orientation maths |
| `rtg_sim.py` | independent simulator that reads only the final creation code (structure checks, timed sim) |
| `pong_check.py` | reference model + randomised timed verification of Pong |
| `calculator_4bit.py`, `counter_4bit.py` | older gate-level examples (use `rtg_circuit.Circuit` directly) |

## Game rules the toolchain assumes
- Wire-likes (wire, button, switch, splitters, light) get signal only from ball attachments; gates read their input cups when a wire-like is attached and output through their output cup. A splitter base never sends.
- Every signal starts off, so each NOT sits behind an OR with the INIT button.
- Labelled input = Switch/Button on an InputSensor (label in `ActivationKey`); labelled output = Wire end on an InputSensor with a Light.
- A Switch flips on each received signal. Everything is one rigid structure (two connector towers joined through Bases), so rope length does not matter.

## Minimal project
```python
from rtg_logic import Machine
from rtg_project import run
m = Machine(tau=0.25); L = m.L           # one tick = 4*tau
m.run_when(m.key("A", tap=False))
n = m.reg(4); n.set(L.inc(n.q))
for i in range(4): m.output(f"Q{i}", n.q[i])
run(m, "counter_code.txt")
```

## Known limits
5 ticks/s (Pong) holds only if real gate delay is about 2 ms or less; use `m.suggest_tau(delay)` otherwise.
Tower placement and `OrientationZ = 180` on gate-tower connectors cannot be tested outside the game.
