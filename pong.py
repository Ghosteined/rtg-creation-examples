#!/usr/bin/env python3
"""PONG  -  this file only DEFINES what is built: inputs, memory, rules, screen.
Everything else (gates, towers, wires, clock, placement) is done by rtg_logic / rtg_circuit.
To build something else (a counter, a snake...), edit the five sections inside game().

    python3 pong.py            -> writes pong_code.txt (the creation code)
    python3 pong_check.py      -> same, plus a timed simulation checked against a python model
"""
from rtg_logic import Machine
from rtg_project import run

# ---------------------------------------------------------------- SETTINGS
W, H = 16, 10                  # screen size (lights)
PAD = 3                        # paddle height
FPS = 5                        # game ticks per second
TAU = 1 / (4 * FPS)            # clock unit: one tick = 4 tau (slow gates? use m.suggest_tau(delay))
MID_X, MID_Y = W // 2, H // 2
PAD_BITS = (H - PAD).bit_length()     # bits needed for a paddle row


def game(tau=TAU):
    m = Machine(tau)
    L = m.L
    ONE, ZERO, NOT, AND, OR, MUX, XOR = L.ONE, L.ZERO, L.not_, L.and_, L.or_, L.mux, L.xor

    # ------------------------------------------------------------ 1. INPUTS (keyboard keys = Buttons, other controls = Switches)
    left_up, left_down = m.key("W"), m.key("S")
    right_up, right_down = m.key("T"), m.key("G")
    run_game = m.switch("RUN")                   # not a keyboard key -> a Switch (keeps its state)
    m.run_when(run_game)                         # memory only changes while RUN is on

    # ------------------------------------------------------------ 2. MEMORY (registers, with start value)
    ball_x, ball_y = m.reg(4, init=MID_X - 1), m.reg(4, init=MID_Y)
    left_pad = m.reg(PAD_BITS, init=(H - PAD) // 2)     # row of the paddle's bottom cell
    right_pad = m.reg(PAD_BITS, init=(H - PAD) // 2)
    # hidden direction of the ball: it always moves one column per tick, and vertically
    #   full slope (1 row / tick), half slope (1 row / 2 ticks, `phase` says which tick) or flat.
    h_right, v_up, v_full, v_half, phase = (m.reg(1, init=i) for i in (1, 1, 0, 1, 0))
    HR, VU, VF, VH, PH = h_right.q[0], v_up.q[0], v_full.q[0], v_half.q[0], phase.q[0]

    m.mark('memory+inputs')
    # ------------------------------------------------------------ 3. DECODED POSITIONS (one-hot lines)
    is_x = [L.eq(ball_x.q, x) for x in range(W)]
    is_y = [L.eq(ball_y.q, y) for y in range(H)]
    left_at = [L.eq(left_pad.q, k) for k in range(H - PAD + 1)]
    right_at = [L.eq(right_pad.q, k) for k in range(H - PAD + 1)]

    def line(lines, i):                          # lines[i], or false outside the board
        return lines[i] if 0 <= i < len(lines) else ZERO

    m.mark('decoders')
    # ------------------------------------------------------------ 4. RULES (next value of every register)
    # 4a. paddles: one step per tick, never past the edge
    def move_paddle(pad, up, down, at):
        go_up = AND(AND(up, NOT(down)), NOT(at[H - PAD]))
        go_down = AND(AND(down, NOT(up)), NOT(at[0]))
        pad.set(L.step(pad.q, OR(go_up, go_down), go_up))

    move_paddle(left_pad, left_up, left_down, left_at)
    move_paddle(right_pad, right_up, right_down, right_at)

    # 4b. vertical move this tick, walls: touching the top / bottom reverses the vertical direction
    moves_v = OR(VF, AND(VH, PH))
    wall = AND(moves_v, OR(AND(VU, is_y[H - 1]), AND(NOT(VU), is_y[0])))
    v_up_now = XOR(VU, wall)
    new_y = L.step(ball_y.q, moves_v, v_up_now)
    enters_row = [MUX(moves_v, line(is_y, r), MUX(v_up_now, line(is_y, r + 1), line(is_y, r - 1)))
                  for r in range(H)]             # enters_row[r]: ball will be in row r after this tick

    # 4c. paddle columns: where does the ball land relative to the paddle?
    at_right_col = AND(HR, is_x[W - 2])
    at_left_col = AND(NOT(HR), is_x[1])
    at_paddle = OR(at_right_col, at_left_col)    # (the two columns are mutually exclusive)
    pad_at = [MUX(at_right_col, line(left_at, i), line(right_at, i))
              for i in range(H - PAD + 1)]       # one-hot row of the paddle that is being reached

    def lands(k):                                # ball lands k cells above the paddle's bottom
        return AND(at_paddle, L.any_([AND(enters_row[r], line(pad_at, r - k)) for r in range(H)]))

    # 4d. return angle by landing cell (the paddle plus one cell above and below it, 5 cells -> 5 angles):
    #       k=3 up, full slope   k=2 up, half   k=1 flat*   k=0 down, half   k=-1 down, full slope
    #       (* flat only if the ball came in sloped)
    up_full, up_half, flat, down_half, down_full = lands(3), lands(2), lands(1), lands(0), lands(-1)
    # a ball that arrives flat can not leave flat: a flat return becomes a half slope towards the centre row
    in_top_half = L.any_(enters_row[MID_Y:])
    from_flat = AND(flat, NOT(OR(VF, VH)))
    hit = L.any_([up_full, up_half, flat, down_half, down_full])
    miss = AND(at_paddle, NOT(hit))
    miss_right, miss_left = AND(miss, at_right_col), AND(miss, at_left_col)
    h_right_bounce = MUX(hit, HR, NOT(HR))
    new_x = L.step(ball_x.q, ONE, h_right_bounce)

    # 4e. next state: a hit sets the new angle (and restarts the half-slope phase), a miss serves the
    #     ball flat from the middle towards the player who missed
    ball_x.set(L.override(new_x, [(miss_left, MID_X), (miss_right, MID_X - 1)]))
    ball_y.set(L.override(new_y, [(miss, MID_Y)]))
    h_right.set(L.override([h_right_bounce], [(miss_right, 1), (miss_left, 0)]))
    v_full.set([MUX(at_paddle, VF, OR(up_full, down_full))])
    v_half.set([MUX(at_paddle, VH, L.any_([up_half, down_half, from_flat]))])
    v_up.set([MUX(at_paddle, v_up_now, L.any_([up_full, up_half, AND(from_flat, NOT(in_top_half))]))])
    phase.set([AND(NOT(PH), NOT(at_paddle))])

    m.mark('rules')
    # ------------------------------------------------------------ 5. SCREEN (which lights are on)
    def pixel(x, y):
        if x == 0:
            return L.any_([line(left_at, y - k) for k in range(PAD)])
        if x == W - 1:
            return L.any_([line(right_at, y - k) for k in range(PAD)])
        return AND(is_x[x], is_y[y])

    m.screen("SCR", W, H, pixel)
    return m


if __name__ == "__main__":
    run(game(), "pong_code.txt")
