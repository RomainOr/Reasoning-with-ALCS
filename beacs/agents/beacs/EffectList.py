"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations

from typing import Optional

from beacs import Perception
from beacs.agents import AbstractPerception
from beacs.agents.beacs import Configuration, Effect

class EffectList():
    """
    List of anticipations
    """

    def __init__(self, effect: Optional[Effect] = None, length: Optional[int] = None, wildcard='#'):
        if effect:
            self.effect_list = [effect]
            self.effect_detailled_counter = [1]
        else:
            self.effect_list = []
            self.effect_detailled_counter = []
        self.enhanced_trace_ga = [True] * length
        self.wildcard = wildcard


    def __eq__(self, other):
        return set(other.effect_list) == set(self.effect_list)


    def __ne__(self, other):
        return not self.__eq__(other)


    def __getitem__(self, i):
        return self.effect_list[i]


    def __len__(self) -> int:
        return len(self.effect_list)


    def __str__(self):
        return "("+", ".join("{}:{}".format(str(effect), counter) for effect, counter in zip(self.effect_list, self.effect_detailled_counter)) + ")"


    def enhance(
            self,
            other: EffectList,
            length: int
        ):
        """
        Creates a new enhanced effectlist by merging two effect lists.

        Parameters
        ----------
        self: EffectList
            First effect list
        other: EffectList
            Second effect list
        length: int
            Classifier effect length
        """
        for oi, oeffect in enumerate(other):
            if oeffect not in self:
                effect_to_append = Effect.empty(length)
                for i in range(length):
                    effect_to_append[i] = oeffect[i]
                self.effect_list.append(effect_to_append)
                self.effect_detailled_counter.append(other.effect_detailled_counter[oi])
            else:
                ei = self.effect_list.index(oeffect)
                self.effect_detailled_counter[ei] += other.effect_detailled_counter[oi]
        self.update_enhanced_trace_ga(length)


    def update_enhanced_trace_ga(
            self,
            length: int
        ):
        for idx in range(length):
            symbols = []
            for effect in self:
                if effect[idx] not in symbols:
                    symbols.append(effect[idx])
            self.enhanced_trace_ga[idx] = (self.wildcard not in symbols) or (len(symbols)==1)


    @property
    def specify_change(self) -> bool:
        """
        Checks whether there is any attribute in the effect part that
        is not "pass-through" - so predicts a change.

        Returns
        -------
        bool
            True if the effect list predicts a change
        """
        index = self.effect_detailled_counter.index(max(self.effect_detailled_counter))
        return self[index].specify_change


    def is_enhanced(self) -> bool:
        """
        Determines if the effect list is enhanced.

        Returns
        -------
        bool
            True if self is enhanced
        """
        return len(self) > 1


    def is_specializable(
            self,
            p0: Perception,
            p1: Perception
        ) -> bool:
        """
        Determines if the effect part can be modified to
        correctly anticipate changes from `p0` to `p1`.
        No need to check for enhanced effect : see the same
        function in classifier.py

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
        return self.is_enhanced() or self[0].is_specializable(p0, p1)


    def does_anticipate_correctly(
            self,
            p0: Perception,
            p1: Perception,
            update_counters: bool = True
        ) -> bool:
        """
        Determines if the effect list anticipates correctly changes from `p0` to `p1`.

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
        for idx, effect in enumerate(self):
            if effect.does_anticipate_correctly(p0, p1):
                if self.is_enhanced() and update_counters:
                    self.effect_detailled_counter[idx] += 1
                return True
        return False


    def subsumes(
            self,
            other: EffectList
        ) -> bool:
        """
        Determines if the effect list subsumes another effect list.

        Parameters
        ----------
        other: EffectList
            Other EffectList

        Returns
        -------
        bool
            True if self subsumes other
        """
        return set(other.effect_list) <= set(self.effect_list)


    def getEffectAttribute(
            self,
            perception,
            index: int
        ) -> tuple:
        """
        Computes from raw observations the probability to get each effect attribute
        for a position in the anticipation.

        Parameters
        ----------
        perception
            Related anticipation
        index: int
            Position in the anticipation

        Returns
        -------
        tuple
            The respective probabilities
        """
        total_counter = float(sum(self.effect_detailled_counter))
        result = {}
        for idx, effect in enumerate(self):
            if effect[index] == effect.wildcard:
                result[int(perception[index])] = result.get(int(perception[index]), 0) + self.effect_detailled_counter[idx] / total_counter
            else:
                result[int(effect[index])] = result.get(int(effect[index]), 0) + self.effect_detailled_counter[idx] / total_counter
        return result


    def sum_effect_counter(self) -> float:
        """
        Computes the sum of all observations in the effect list

        Returns
        -------
        float
            The respective probabilities
        """
        return float(sum(self.effect_detailled_counter))
