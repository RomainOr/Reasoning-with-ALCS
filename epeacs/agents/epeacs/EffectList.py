"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations

from typing import Optional

from epeacs import Perception
from epeacs.agents import AbstractPerception
from epeacs.agents.epeacs import Configuration, Effect

class EffectList():
    """
    List of anticipations
    """

    def __init__(self, effect: Optional[Effect] = None, wildcard='#'):
        if effect:
            self.effect_list = [effect]
            self.effect_detailled_counter = [1]
        else:
            self.effect_list = []
            self.effect_detailled_counter = []
        self.wildcard = wildcard


    def __eq__(self, other):
        return set(other.effect_list) == set(self.effect_list)


    def __ne__(self, other):
        return not self.__eq__(other)


    def __getitem__(self, i):
        return self.effect_list[i]


    def __len__(self) -> int:
        return len(self.effect_list)
            

    @property
    def specify_change(self) -> bool:
        """
        Checks whether there is any attribute in the effect part that
        is not "pass-through" - so predicts a change.

        Returns
        -------
        bool
            True if the effect list predicts a change, False otherwise
        """
        if len(self.effect_list) > 1:
            return True
        else:
            return self.effect_list[0].specify_change


    def is_specializable(self, p0: Perception, p1: Perception) -> bool:
        """
        Determines if the effect part can be modified to anticipate
        changes from `p0` to `p1` correctly by only specializing attributes.
        Parameters
        ----------
        p0: Perception
            previous perception
        p1: Perception
            current perception
        Returns
        -------
        bool
            True if specializable, false otherwise
        """
        if len(self.effect_list) > 1:
            return True
        else:
            return self.effect_list[0].is_specializable(p0, p1)


    def anticipates_correctly(self, p0: Perception, p1: Perception) -> bool:
        for effect in self.effect_list:
            if effect.anticipates_correctly(p0, p1):
                return True
        return False


    def subsumes(self, other) -> bool:
        return set(other.effect_list) <= set(self.effect_list)


    def is_enhanced(self) -> bool:
        return len(self.effect_list) > 1


    def update_enhanced_effect_probs(self, p1: Perception):
        for idx, effect in enumerate(self.effect_list):
            if effect.does_match(p1):
                self.effect_detailled_counter[idx] += 1
                break


    @classmethod
    def enhanced_effect(
            cls, 
            effectlist1,
            effectlist2,
            perception: AbstractPerception = None
        ):
        """
        Create a new enhanced effect part.
        """
        result = cls()
        result.effect_list.extend(effectlist1.effect_list)
        result.effect_detailled_counter.extend(effectlist1.effect_detailled_counter)
        for idxe, e in enumerate(effectlist2):
            is_updated = False
            for idxr, r in enumerate(result.effect_list):
                if e == r:
                    result.effect_detailled_counter[idxr] += effectlist2.effect_detailled_counter[idxe]
                    is_updated = True
                    break
            if not is_updated:
                result.effect_list.append(e)
                result.effect_detailled_counter.append(effectlist2.effect_detailled_counter[idxe])
        return result


    def __str__(self):
        return "("+", ".join("{}:#{}".format(str(effect), counter) for effect, counter in zip(self.effect_list, self.effect_detailled_counter)) + ")"


    def getEffectAttribute(self, perception, index):
        total_counter = float(sum(self.effect_detailled_counter))
        result = {}
        for idx, effect in enumerate(self.effect_list):
            if effect[index] == effect.wildcard:
                result[int(perception[index])] = result.get(int(perception[index]), 0) + self.effect_detailled_counter[idx] /total_counter
            else:
                result[int(effect[index])] = result.get(int(effect[index]), 0) + self.effect_detailled_counter[idx] /total_counter
        return result, result