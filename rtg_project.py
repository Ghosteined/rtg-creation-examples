"""Tiny runner for a project script:  run(machine, "out.txt")  builds, checks the structure, writes the
creation code and prints a report.  A project script then only has to *define* the machine."""
import sys
from rtg_circuit import Layout


def run(machine, outfile, layout=None):
    res = machine.build(layout or Layout())
    open(outfile, "w").write(res.code + "\n")
    print(f"logic depth {machine.depth()} gates, tau {machine.tau} -> one tick = {4 * machine.tau:g} s")
    print(res.report())
    from rtg_sim import load, structural_checks
    structural_checks(load(outfile))
    print(f"written to {outfile}; structure OK (one rigid assembly, one ball per cup, gate inputs driven)")
    return res
