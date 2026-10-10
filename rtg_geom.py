"""rtg_geom - exact placement maths for Connector trees, from the item file's attachment frames.
A ball plugged into a cup puts the ball frame on the cup frame, so a child's rotation is
R_cup * R_ball^-1.  Used to find which connector face points where once connectors are rotated."""
import numpy as np
from rtg_api.rtg_object import root, get_id

_C = {}
for _ch in root.find_first_child("Connector").get_descendants():
    if _ch.class_name == "Attachment":
        _cf = _ch.get_property("CFrame")
        _C[(_ch.get_property("Name")[0], int(get_id(_ch)))] = (
            np.array([_cf.position.x, _cf.position.y, _cf.position.z]), np.array(_cf.rotation, float))
_BP, _BR = _C[("B", 5)]
CUPS = (1, 2, 3, 4, 6)


class Conn:
    """A connector with a world rotation R and position p."""
    def __init__(self, R=None, p=None):
        self.R = np.eye(3) if R is None else R
        self.p = np.zeros(3) if p is None else p

    def cup_pos(self, k):
        return self.R @ _C[("C", k)][0] + self.p

    def cup_normal(self, k):
        return np.rint(self.R @ (_C[("C", k)][1] @ np.array([0, 0, -1.0])))

    def child(self, k):
        """The connector plugged (by its ball) into this connector's cup k."""
        R = self.R @ _C[("C", k)][1] @ _BR.T
        wc = self.cup_pos(k)
        return Conn(R, wc - R @ _BP)

    def face_with_normal(self, n):
        for k in CUPS:
            if np.allclose(self.cup_normal(k), n):
                return k
        return None
