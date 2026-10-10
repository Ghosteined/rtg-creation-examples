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


def game(tau=TAU):
    m = Machine(tau)
    L = m.L
    ONE, ZERO, NOT, AND, OR, MUX, XOR = L.ONE, L.ZERO, L.not_, L.and_, L.or_, L.mux, L.xor

    # ------------------------------------------------------------ 1. INPUTS (button + label)
    left_up, left_down = m.key("W"), m.key("S")
    right_up, right_down = m.key("T"), m.key("G")
    run_game = m.key("RUN", tap=False)           # a held key = a level, not a tap
    m.run_when(run_game)                         # memory only changes while RUN is on

    # ------------------------------------------------------------ 2. MEMORY (registers, with start value)
    ball_x, ball_y = m.reg(4, init=MID_X - 1), m.reg(4, init=MID_Y)
    left_pad = m.reg(4, init=(H - PAD) // 2)     # row of the paddle's bottom cell
    right_pad = m.reg(4, init=(H - PAD) // 2)
    # hidden direction of the ball (8 directions): moves horizontally? to the right? / vertically? up?
    h_moves, h_right, v_moves, v_up = (m.reg(1, init=1) for _ in range(4))
    HM, HR, VM, VU = h_moves.q[0], h_right.q[0], v_moves.q[0], v_up.q[0]

    # ------------------------------------------------------------ 3. DECODED POSITIONS (one-hot lines)
    is_x = [L.eq(ball_x.q, x) for x in range(W)]
    is_y = [L.eq(ball_y.q, y) for y in range(H)]
    left_at = [L.eq(left_pad.q, k) for k in range(H - PAD + 1)]
    right_at = [L.eq(right_pad.q, k) for k in range(H - PAD + 1)]

    def line(lines, i):                          # lines[i], or false outside the board
        return lines[i] if 0 <= i < len(lines) else ZERO

    # ------------------------------------------------------------ 4. RULES (next value of every register)
    # 4a. paddles: one step per tick, never past the edge
    def move_paddle(pad, up, down, at):
        go_up = AND(AND(up, NOT(down)), NOT(at[H - PAD]))
        go_down = AND(AND(down, NOT(up)), NOT(at[0]))
        pad.set(L.mux_w(go_up, L.mux_w(go_down, pad.q, L.dec(pad.q)), L.inc(pad.q)))

    move_paddle(left_pad, left_up, left_down, left_at)
    move_paddle(right_pad, right_up, right_down, right_at)

    # 4b. walls: a vertical ball touching top/bottom turns diagonal (away from the nearer side)
    wall = AND(VM, OR(AND(VU, is_y[H - 1]), AND(NOT(VU), is_y[0])))
    v_up_now = XOR(VU, wall)
    h_moves_now = OR(HM, wall)
    h_right_now = MUX(AND(wall, NOT(HM)), HR, NOT(L.ge(ball_x.q, MID_X)))
    new_y = L.mux_w(VM, ball_y.q, L.mux_w(v_up_now, L.dec(ball_y.q), L.inc(ball_y.q)))
    enters_row = [MUX(VM, line(is_y, r), MUX(v_up_now, line(is_y, r + 1), line(is_y, r - 1)))
                  for r in range(H)]             # enters_row[r]: ball will be in row r after this tick

    # 4c. paddle columns: where does the ball land relative to the paddle?
    at_right_col = AND(AND(h_moves_now, h_right_now), is_x[W - 2])
    at_left_col = AND(AND(h_moves_now, NOT(h_right_now)), is_x[1])

    def lands_on(pad_at, k):                     # ball lands k cells above the paddle's bottom
        return L.any_([AND(enters_row[r], line(pad_at, r - k)) for r in range(H)])

    def lands(k):
        return OR(AND(at_right_col, lands_on(right_at, k)), AND(at_left_col, lands_on(left_at, k)))

    top, middle, bottom = lands(2), lands(1), lands(0)
    corner_above, corner_below = lands(3), lands(-1)
    hit = L.any_([top, middle, bottom])
    corner = OR(corner_above, corner_below)
    at_paddle = OR(at_right_col, at_left_col)
    miss = AND(at_paddle, AND(NOT(hit), NOT(corner)))
    miss_right, miss_left = AND(miss, at_right_col), AND(miss, at_left_col)

    # 4d. bounce: top -> up, middle -> straight, bottom -> down, corner -> vertical along the screen
    h_right_bounce = MUX(hit, h_right_now, NOT(h_right_now))
    h_moves_bounce = AND(h_moves_now, NOT(corner))
    v_moves_bounce = MUX(hit, MUX(corner, VM, ONE), NOT(middle))
    v_up_bounce = MUX(hit, MUX(corner, v_up_now, corner_above), MUX(middle, top, v_up_now))
    new_x = L.mux_w(h_moves_bounce, ball_x.q, L.mux_w(h_right_bounce, L.dec(ball_x.q), L.inc(ball_x.q)))

    # 4e. a miss serves the ball from the middle towards the player who missed
    ball_x.set(L.mux_w(miss, new_x, L.mux_w(miss_left, L.word(MID_X - 1, 4), L.word(MID_X, 4))))
    ball_y.set(L.mux_w(miss, new_y, L.word(MID_Y, 4)))
    h_moves.set([MUX(miss, h_moves_bounce, ONE)])
    h_right.set([MUX(miss, h_right_bounce, miss_right)])
    v_moves.set([MUX(miss, v_moves_bounce, ZERO)])
    v_up.set([MUX(miss, v_up_bounce, ONE)])

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
