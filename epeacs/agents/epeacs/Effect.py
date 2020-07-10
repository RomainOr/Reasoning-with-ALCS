"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations

from epeacs import Perception
from epeacs.agents import AbstractPerception
from epeacs.agents.epeacs.ProbabilityEnhancedAttribute import ProbabilityEnhancedAttribute

class Effect(AbstractPerception):
    """
    Anticipates the effects that the classifier 'believes'
    to be caused by the specified action.
    """

    def __init__(self, observation, wildcard='#', oktypes=(str, dict)):
        # Convert dict to ProbabilityEnhancedAttribute
        if not all( isinstance(attr, ProbabilityEnhancedAttribute) for attr in observation):
            observation = (ProbabilityEnhancedAttribute(attr)
                           if isinstance(attr, dict)
                           else attr
                           for attr in observation)
        super().__init__(observation, wildcard, oktypes)
        self.detailled_counter = {}


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
        if self.is_enhanced():
            return True
        else:
            return any(True for e in self if e != self.wildcard)


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
        if self.is_enhanced():
            return True
        for p0i, p1i, ei in zip(p0, p1, self):
            if ei != self.wildcard:
                if ei != p1i or p0i == p1i:
                    return False
        return True


    def anticipates_correctly(self, p0: Perception, p1: Perception) -> bool:
        def item_anticipate_change(item, p0_item, p1_item, wildcard) -> bool:
            if not isinstance(item, ProbabilityEnhancedAttribute):
                if item == wildcard:
                    if p0_item != p1_item: return False
                else:
                    if p0_item == p1_item: return False
                    if item != p1_item: return False        
            else:
                if not item.does_contain(p1_item):
                    return False
            # All checks passed
            return True
        return all(item_anticipate_change(self[idx], p0[idx], p1[idx], self.wildcard) for idx in range(len(p0)))


    def subsumes(self, other: Effect, self_condition, other_condition) -> bool:
        for idx, (si, oi) in enumerate(zip(self, other)):
            if isinstance(si, ProbabilityEnhancedAttribute):
                if isinstance(oi, ProbabilityEnhancedAttribute):
                    if not si.subsumes(oi): return False
                else:
                    if oi == other.wildcard:
                        if not si.does_contain(other_condition[idx]): return False
                    else:
                        if not si.does_contain(oi): return False
            else:
                if isinstance(oi, ProbabilityEnhancedAttribute):
                    return False
                else:
                    if si != oi: return False
        return True


    def clean(self):
        for idx, ei in enumerate(self):
            if isinstance(ei, ProbabilityEnhancedAttribute):
                if len(ei) == 1:
                    self[idx] = next(iter(ei))


    def is_enhanced(self) -> bool:
        """
        Checks whether any element of the Effect is Probability-Enhanced.
        str elements of the Effect are not Enhanced,
        ProbabilityEnhancedAttribute elements are Enhanced.

        Returns
        -------
        bool
            True if there is a Probability-Enhanced Effect, False otherwise
        """
        return any(isinstance(elem, ProbabilityEnhancedAttribute)
                   for elem in self)


    def update_enhanced_effect_probs(
            self, 
            perception: Perception, 
            update_rate: float
        ):
        for i, elem in enumerate(self):
            if isinstance(elem, ProbabilityEnhancedAttribute):
                elem.make_compact()
                effect_symbol = perception[i]
                elem.increase_probability(effect_symbol, update_rate)
        self.detailled_counter[perception] = self.detailled_counter.get(perception, 0) + 1


    @classmethod
    def enhanced_effect(
            cls, 
            effect1, 
            exp1,
            effect2,
            exp2,
            perception: AbstractPerception = None
        ):
        """
        Create a new enhanced effect part.
        """
        result = cls(observation=effect1)
        wildcard = effect1.wildcard
        for i, attr2 in enumerate(effect2):
            attr1 = effect1[i]
            if attr1 == wildcard and attr2 == wildcard: continue
            if attr1 == wildcard: attr1 = perception[i]
            if attr2 == wildcard: attr2 = perception[i]
            result[i] = ProbabilityEnhancedAttribute.merged_attributes(attr1, exp1, attr2, exp2)
        return result


    def __str__(self):
        return ''.join(str(attr) for attr in self)

    
    def print_detailled_counter(self):
        return ", ".join("{}:#{}".format("".join(dse_key), dse_value) for dse_key, dse_value in self.detailled_counter.items())

    def getEffectAttribute(self, index):
        for i, attr in enumerate(self):
            if i == index:
                if isinstance(attr, ProbabilityEnhancedAttribute):
                    return {int(k):v for k,v in attr.items()}, {int(k): v / total for total in (sum(attr.test.values()),) for k, v in attr.test.items()}
                else:
                    if attr == self.wildcard:
                        return attr, attr
                    else:
                        return {int(attr):1.0}, {int(attr):1.0}
            else:
                continue