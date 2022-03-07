"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations

from beacs import Perception
from beacs.agents import AbstractPerception

class Effect(AbstractPerception):
    """
    Anticipates the effects that the classifier 'believes'
    to be caused by the specified action.
    """

    def __init__(self, observation, wildcard='#'):
        super().__init__(observation, wildcard)


    @property
    def specify_change(self) -> bool:
        """
        Checks whether there is any attribute in the effect part that
        is not "pass-through" - so predicts a change.

        Returns
        -------
        bool
            True if the effect part predicts a change, False otherwise
        """
        return any(True for e in self if e != self.wildcard)


    def is_specializable(
            self,
            p0: Perception,
            p1: Perception
        ) -> bool:
        """
        Determines if the effect part can be modified to anticipate
        changes from `p0` to `p1` correctly by only specializing attributes.

        Parameters
        ----------
        p0: Perception
            Previous perception
        p1: Perception
            Current perception

        Returns
        -------
        bool
            True if specializable
        """
        for p0i, p1i, ei in zip(p0, p1, self):
            if ei != self.wildcard:
                if ei != p1i or p0i == p1i:
                    return False
        return True


    def does_anticipate_correctly(
            self,
            p0: Perception,
            p1: Perception
        ) -> bool:
        """
        Determines if the effect anticipates correctly changes from `p0` to `p1`.

        Parameters
        ----------
        p0: Perception
            Previous perception
        p1: Perception
            Current perception

        Returns
        -------
        bool
            True the anticipation is correct
        """
        def _item_anticipate_change(
                item,
                p0_item,
                p1_item,
                wildcard
            ) -> bool:
            if item == wildcard:
                if p0_item != p1_item: return False
            else:
                if p0_item == p1_item: return False
                if item != p1_item: return False
            # All checks passed
            return True
        return all(_item_anticipate_change(self[idx], p0[idx], p1[idx], self.wildcard) for idx in range(len(p0)))