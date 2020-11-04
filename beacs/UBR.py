"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations

class UBR:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def lower_bound(self) -> float:
        return min(self.x, self.y)

    @property
    def upper_bound(self) -> float:
        return max(self.x, self.y)

    @property
    def spread(self) -> float:
        return float(self.upper_bound) - float(self.lower_bound)

    def subsumes(self, other: UBR) -> bool:
        return self.lower_bound <= other.lower_bound and \
            other.upper_bound <= self.upper_bound

    def __contains__(self, x):
        return self.lower_bound <= x <= self.upper_bound

    def __eq__(self, o) -> bool:
        if not isinstance(o, UBR):
            return False
        return self.lower_bound == o.lower_bound and \
            self.upper_bound == o.upper_bound

    def __hash__(self):
        return hash((self.lower_bound, self.upper_bound))

    def __str__(self):
        return '[' + self.lower_bound + ';' + self.upper_bound + ']'
