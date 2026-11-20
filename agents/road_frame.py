"""
Road coordinates along a centreline: distance s along the road and offset d across it (d > 0 is LEFT of the travel direction).
Everything is in the SUMO net frame (x east, y north, heading east 0 / north 90). A straight road is the polyline (0,0)-(length,0).
"""
import math

LANE_WIDTH = 3.5


def lane_offset(lanes, k, width=LANE_WIDTH):
    """Offset of lane k (0 = right-most) of a carriageway with `lanes` lanes from the road centre line (negative = right)."""
    return -(lanes - 0.5 - k) * width


class RoadFrame:
    def __init__(self, points):
        pts = [(float(x), float(y)) for x, y in points]
        if len(pts) < 2:
            raise ValueError("a centreline needs at least 2 points")
        self.pts = pts
        self.cum = [0.0]
        for a, b in zip(pts, pts[1:]):
            self.cum.append(self.cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        if self.cum[-1] <= 0.0:
            raise ValueError("the centreline has no length")
        self.length = self.cum[-1]

    @classmethod
    def straight(cls, length=300.0):
        return cls([(0.0, 0.0), (float(length), 0.0)])

    def _seg(self, s):
        s = max(0.0, min(self.length, s))
        lo, hi = 0, len(self.cum) - 2
        while lo < hi:                                  # last segment whose start is <= s
            mid = (lo + hi + 1) // 2
            if self.cum[mid] <= s:
                lo = mid
            else:
                hi = mid - 1
        return lo, s

    def _dir(self, i):
        a, b = self.pts[i], self.pts[i + 1]
        n = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
        return (b[0] - a[0]) / n, (b[1] - a[1]) / n

    def to_xy(self, s, d=0.0):
        i, s = self._seg(s)
        a = self.pts[i]
        ux, uy = self._dir(i)
        t = s - self.cum[i]
        return a[0] + ux * t - uy * d, a[1] + uy * t + ux * d

    def heading(self, s):
        i, _ = self._seg(s)
        ux, uy = self._dir(i)
        return math.degrees(math.atan2(uy, ux))

    def to_sd(self, x, y):
        best = None
        for i in range(len(self.pts) - 1):
            a = self.pts[i]
            ux, uy = self._dir(i)
            seg_len = self.cum[i + 1] - self.cum[i]
            t = max(0.0, min(seg_len, (x - a[0]) * ux + (y - a[1]) * uy))
            px, py = a[0] + ux * t, a[1] + uy * t
            dist = math.hypot(x - px, y - py)
            if best is None or dist < best[0]:
                d = ux * (y - a[1]) - uy * (x - a[0])
                best = (dist, self.cum[i] + t, d)
        return best[1], best[2]

    def points_xy(self, step=2.0):
        n = max(int(self.length / step), 1)
        return [self.to_xy(self.length * i / n, 0.0) for i in range(n + 1)]
