"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations

import random

class UBR:

    def __init__(self, x, y, s0):
        self.x = x
        self.y = y
        self.initial_spread = s0

    @property
    def lower_bound(self) -> float:
        return min(self.x, self.y)

    @property
    def upper_bound(self) -> float:
        return max(self.x, self.y)

    @property
    def spread(self) -> float:
        return float(self.upper_bound) - float(self.lower_bound)

    @classmethod
    def copy(cls, old) -> UBR:
        return cls(old.x, old.y, old.initial_spread)

    def does_intersect_with(self, other) -> bool:
        return self.upper_bound >= other.lower_bound and other.upper_bound >= self.lower_bound

    def widen_with_ubr(self, other):
        if not self.subsumes(other):
            min_lower_bound = min(self.lower_bound, other.lower_bound)
            max_upper_bound = max(self.upper_bound, other.upper_bound)
            self.x = min_lower_bound
            self.y = max_upper_bound

    def widen_with_spread(self):
        growth = random.uniform(0.,max(self.spread, self.initial_spread))
        amount_x_y = random.random()
        self.x -= growth * amount_x_y
        self.y += growth * (1. - amount_x_y)

    def subsumes(self, other: UBR) -> bool:
        return self.lower_bound <= other.lower_bound and \
            other.upper_bound <= self.upper_bound

    def __contains__(self, x):
        return self.lower_bound <= x and x <= self.upper_bound

    def __eq__(self, o) -> bool:
        if not isinstance(o, UBR):
            return False
        return self.lower_bound == o.lower_bound and \
            self.upper_bound == o.upper_bound

    def __hash__(self):
        return hash((self.lower_bound, self.upper_bound))

    def __str__(self):
        return "[{:.2f}; {:.2f}]".format(self.lower_bound, self.upper_bound)
