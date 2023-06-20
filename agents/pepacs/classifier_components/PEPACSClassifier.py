"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations
from typing import Optional, Union, List

from agents.common import Perception
from agents.common.classifier_components.Condition import Condition
from agents.common.classifier_components.BaseClassifier import BaseClassifier

from agents.pepacs.PEPACSConfiguration import PEPACSConfiguration
from agents.pepacs.classifier_components.PEPEffect import PEPEffect
from agents.pepacs.classifier_components.ProbabilityEnhancedAttribute import ProbabilityEnhancedAttribute


class PEPACSClassifier(BaseClassifier):

    def __init__(
            self,
            condition: Union[Condition, str, None] = None,
            action: Optional[int] = None,
            effect: Union[PEPEffect, str, None] = None,
            quality: float=0.5,
            reward: float=0.5,
            immediate_reward: float=0.0,
            numerosity: int=1,
            experience: int=1,
            talp: int=0,
            tga: int=0,
            tav: float=0.0,
            cfg: Optional[PEPACSConfiguration] = None
        ) -> None:
        super().__init__(
            condition=condition,
            action=action,
            effect=effect,
            quality=quality,
            reward=reward,
            immediate_reward=immediate_reward,
            numerosity=numerosity,
            experience=experience,
            talp=talp,
            tga=tga,
            tav=tav,
            cfg=cfg
        )
        def _build_perception_string(
                cls,
                initial,
                length=self.cfg.classifier_length,
                wildcard=self.cfg.classifier_wildcard
            ):
            if initial:
                return cls(initial, wildcard=wildcard)
            return cls.empty(wildcard=wildcard, length=length)
        self.effect = _build_perception_string(PEPEffect, effect)


    def copy(
            self, 
            time: int, 
            perception: Perception = None
        ) -> PEPACSClassifier:
        """
        Copies old classifier with given time (tga, talp).
        Old tav gets replaced with new value.
        New classifier also has no mark.

        Parameters
        ----------
            time: int
            perception: Perception = None

        Returns
        -------
        PEPACSClassifier
        """
        new_cls = PEPACSClassifier(
            condition=Condition(self.condition, self.cfg.classifier_wildcard),
            action=self.action,
            effect=PEPEffect(self.effect, self.cfg.classifier_wildcard),
            quality=self.q,
            reward=self.r,
            immediate_reward=self.ir,
            cfg=self.cfg,
            tga=time,
            talp=time,
            tav=self.tav
        )
        if perception and new_cls.is_enhanced():
            for idx, ei in enumerate(new_cls.effect):
                if isinstance(ei, ProbabilityEnhancedAttribute):
                    new_cls.effect[idx] = perception[idx]
        return new_cls


    def is_enhanced(self) -> bool:
        """
        Checks whether the classifier is enhanced.

        Returns
        -------
        bool
        """
        return self.effect.is_enhanced()


    def specialize(
            self,
            previous_situation: Perception,
            situation: Perception
        ) -> None:
        """
        Specializes the effect part where necessary to correctly anticipate
        the changes from p0 to p1.

        Parameters
        ----------
            previous_situation: Perception
            situation: Perception
        """
        for idx, _ in enumerate(situation):
            if self.effect[idx] != self.cfg.classifier_wildcard:
                # If we have a specialized attribute don't change it.
                continue
            if previous_situation[idx] != situation[idx]:
                if self.effect[idx] == self.cfg.classifier_wildcard:
                    self.effect[idx] = situation[idx]
                elif self.ee:
                    if not isinstance(self.effect[idx], ProbabilityEnhancedAttribute):
                        self.effect[idx] = ProbabilityEnhancedAttribute(self.effect[idx])
                    self.effect[idx].insert_symbol(situation[idx])
                self.condition[idx] = previous_situation[idx]


    def merge_with(
            self,
            other_classifier: PEPACSClassifier,
            perception: Perception,
            time: int
        ) -> PEPACSClassifier:
        """
        Merges two classifiers in an enhanced one.

        Parameters
        ----------
            other_classifier: PEPACSClassifier
            perception: Perception
            time: int

        Returns
        -------
        PEPACSClassifier
        """
        result = PEPACSClassifier(
            action = self.action,
            quality = max((self.q + other_classifier.q) / 2.0, 0.5),
            reward = (self.r + other_classifier.r) / 2.0,
            talp = time,
            tga = time,
            cfg = self.cfg
        )
        result.condition = Condition(self.condition)
        result.condition.specialize_with_condition(other_classifier.condition)
        result.effect = PEPEffect.enhanced_effect(
            self.effect, 
            other_classifier.effect,
            perception)
        return result