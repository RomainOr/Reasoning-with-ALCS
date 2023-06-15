"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations

from agents.common.Perception import Perception
from agents.common.classifier_components.AbstractPerception import AbstractPerception
from agents.common.classifier_components.Effect import Effect

from agents.pepacs.classifier_components.ProbabilityEnhancedAttribute import ProbabilityEnhancedAttribute


class PEPEffect(Effect):
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
            return super().specify_change


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
        return super().is_specializable(p0, p1)


    def does_anticipate_correctly(self, p0: Perception, p1: Perception) -> bool:
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
        return all(item_anticipate_change(eitem, p0[idx], p1[idx], self.wildcard) for idx, eitem in enumerate(self))


    def subsumes(self, other: Effect) -> bool:
        for si, oi in zip(self, other):
            if isinstance(si, ProbabilityEnhancedAttribute):
                if isinstance(oi, ProbabilityEnhancedAttribute):
                    if not si.subsumes(oi): return False
                else:
                    if not si.does_contain(oi): return False
            else:
                if isinstance(oi, ProbabilityEnhancedAttribute):
                    return False
                else:
                    if si != oi: return False
        return True
    
    
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
        if isinstance(self[index], ProbabilityEnhancedAttribute):
            return {int(k):v for k,v in self[index].items()}
        else:
            return super().getEffectAttribute(perception, index)


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
        return any(isinstance(elem, ProbabilityEnhancedAttribute) for elem in self)


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


    @classmethod
    def enhanced_effect(
            cls, 
            effect1, 
            effect2,
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
            result[i] = ProbabilityEnhancedAttribute.merged_attributes(attr1, attr2)
        return result
    
    
    