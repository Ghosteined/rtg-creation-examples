#!/usr/bin/env python3
"""4-bit adder/subtractor for Road to Gramby.

Inputs  (switches): A0-A3, B0-B3, SUB   + INIT (button; press once after spawning)
Outputs (lights)  : R0-R3 (result, R0 = least significant bit)
                    R4    (add: carry out / result >= 16;
                           sub: borrow, i.e. A < B, the result is negative)

SUB = 0 : R = A + B
SUB = 1 : R = A - B  (two's complement in R0-R3, R4 = 1 when negative)

Usage:  python3 calculator_4bit.py [output_file]
        (needs rtg_api + rbxm_parser importable; builds, prints a report, verifies)
"""
import sys

from rtg_circuit import Circuit, Dual, Layout, build
import rtg_sim

N = 4


def make_circuit():
    c = Circuit()
    init = c.button("INIT", init=True)
    dr = Dual(c, init)
    A, B = dr.inputs("A", N), dr.inputs("B", N)
    sub = dr.input(c.switch("SUB"))

    carry = sub                                   # A - B = A + ~B + 1
    for i in range(N):
        s, carry = dr.full_adder(A[i], dr.xor(B[i], sub), carry)
        c.output(f"R{i}", s.t)
    c.output(f"R{N}", dr.xor(carry, sub).t)       # carry (add) / borrow (sub)
    return c


# inputs bottom -> top on the IO tower; SUB and INIT sit in the middle because
# they feed every bit slice (keeps their ropes short)
LAYOUT = Layout(input_order=["A0", "B0", "A1", "B1", "SUB", "INIT", "A2", "B2", "A3", "B3"])


def reference(v):
    a = sum(v[f"A{i}"] << i for i in range(N))
    b = sum(v[f"B{i}"] << i for i in range(N))
    d = a - b if v["SUB"] else a + b
    out = {f"R{i}": (d >> i) & 1 for i in range(N)}
    out[f"R{N}"] = int(d < 0) if v["SUB"] else int(d >= 1 << N)
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "calculator_code.txt"
    result = build(make_circuit(), LAYOUT)
    open(path, "w").write(result.code + "\n")
    print(result.report())
    print(f"written to {path}\n")
    ok = rtg_sim.verify(path, reference, init=("INIT",))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
