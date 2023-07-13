"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations
from typing import Optional, Union, List

from agents.common import Perception
from agents.common.classifier_components.Condition import Condition
from agents.common.classifier_components.Effect import Effect
from agents.common.classifier_components.BaseClassifier import BaseClassifier

from agents.beacs.BEACSConfiguration import BEACSConfiguration
from agents.beacs.classifier_components.EffectList import EffectList


class BEACSClassifier(BaseClassifier):
    """
    Represents a BEACS classifier
    """

    def __init__(
            self,
            condition: Union[Condition, str, None] = None,
            action: Optional[int] = None,
            behavioral_sequence: Optional[List[int]] = None,
            effect: Optional[Effect] = None,
            quality: float=0.5,
            reward: float=0.,
            reward_bis: float=0.,
            immediate_reward: float=0.0,
            numerosity: int=1,
            experience: int=1,
            talp: int=0,
            tga: int=0,
            tav: float=0.0,
            cfg: Optional[BEACSConfiguration] = None,
            tbseq: int=0,
            aliased_state: Optional[Perception] = None,
            pai_state: Optional[Perception] = None,
        ) -> None:
        super().__init__(
            condition=condition,
            action=action,
            behavioral_sequence=behavioral_sequence,
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
        self.effect = EffectList(_build_perception_string(Effect, effect), self.cfg.classifier_length, self.cfg.classifier_wildcard)
        self.r_bis = reward_bis
        self.tbseq = tbseq
        if aliased_state:
            self.aliased_state = aliased_state
        else:
            self.aliased_state = Perception.empty()
        if pai_state:
            self.pai_state = pai_state
        else:
            self.pai_state = Perception.empty()
        self.err = 0.


    def __repr__(self) -> str:
        return f"C:{self.condition} A:{self.action} {str(self.behavioral_sequence)} E:{str(self.effect)}\n" \
            f"q: {self.q:<6.4} r: {self.r:<6.4} r_bis: {self.r_bis:<6.4} ir: {self.ir:<6.4} f: {self.fitness:<6.4} err: {self.err:<6.4}\n" \
            f"exp: {self.exp:<5} num: {self.num} ee: {self.ee}\n" \
            f"Mark: {str(self.mark)} Can_be_generalized: {str(self.effect.enhanced_trace_ga)} Aliased_state: {'.'.join(str(attr) for attr in self.aliased_state)} PAI_state: {'.'.join(str(attr) for attr in self.pai_state)}\n" \
            f"tga: {self.tga:<5} tbseq: {self.tbseq:<5} talp: {self.talp:<5} tav: {self.tav:<6.4} \n" \


    def copy(
            self: BEACSClassifier,
            time: int,
            perception: Perception = None
        ) -> BEACSClassifier:
        """
        Copies old classifier with given time.
        Old tav gets replaced with new value.
        New classifier also has no mark.

        Parameters
        ----------
            time: int

        Returns
        -------
        BEACSClassifier
            New copied classifier - Hard copy
        """
        new_cls = BEACSClassifier(
            condition=Condition(self.condition, self.cfg.classifier_wildcard),
            action=self.action,
            behavioral_sequence=self.behavioral_sequence,
            quality=self.q,
            reward=self.r,
            reward_bis=self.r_bis,
            immediate_reward=self.ir,
            cfg=self.cfg,
            tga=time,
            tbseq=time,
            talp=time,
            tav=self.tav,
            aliased_state=self.aliased_state,
            pai_state=self.pai_state
        )
        new_cls.effect.effect_list = []
        for oeffect in self.effect:
            effect_to_append = Effect.empty(new_cls.cfg.classifier_length)
            for i in range(new_cls.cfg.classifier_length):
                effect_to_append[i] = oeffect[i]
            new_cls.effect.effect_list.append(effect_to_append)
        new_cls.effect.effect_detailled_counter = self.effect.effect_detailled_counter[:]
        new_cls.effect.enhanced_trace_ga = self.effect.enhanced_trace_ga[:]
        new_cls.effect.update_enhanced_trace_ga(new_cls.cfg.classifier_length)
        return new_cls


    def copy_time_num_exp_from_other_cl(
            self,
            other: BEACSClassifier
        ) -> None:
        """
        Copies internal parameters (tga, tbseq, talp, exp, num)
        from other classifier

        Parameters
        ----------
            other: BEACSClassifier
        """
        super().copy_time_num_exp_from_other_cl(other)
        self.tbseq = other.tbseq


    @property
    def fitness(self) -> float:
        """
        Computes the fitness of the classifier.

        Returns
        -------
        float
            Fitness value
        """
        epsilon = 1e-6
        max_r = max(self.r, self.r_bis)
        min_r = min(self.r, self.r_bis)
        diff = max_r - min_r + epsilon
        quality = self.q
        prediction_reward = max_r
        if self.behavioral_sequence:
            prediction_reward = max_r - diff * len(self.behavioral_sequence) / self.cfg.bs_max
        if prediction_reward < 0:
            quality = 1. / quality
        return quality * prediction_reward


    def is_enhanced(self) -> bool:
        """
        Checks whether the classifier is enhanced by EP.

        Returns
        -------
        bool
        """
        return self.effect.is_enhanced()


    def does_anticipate_correctly(
            self,
            previous_situation: Perception,
            situation: Perception,
            update_counter: bool = True
        ) -> bool:
        """
        Checks anticipation. While the pass-through symbols in the effect part
        of a classifier directly anticipate that these attributes stay the same
        after the execution of an action, the specified attributes anticipate
        a change to the specified value. Thus, if the perceived value did not
        change to the anticipated but actually stayed at the value, the
        classifier anticipates incorrectly.
        update_counter is used to indicate if the counters related to the EP have
        to be updated.

        Parameters
        ----------
            previous_situation: Perception
            situation: Perception
            update_counter: bool = True

        Returns
        -------
        bool
        """
        return self.effect.does_anticipate_correctly(previous_situation, situation, update_counter)


    def specialize(
            self,
            previous_situation: Perception,
            situation: Perception
        ) -> None:
        """
        Specializes the classifier parts where necessary to correctly anticipate
        the changes from previous_situation to situation.
        Only occurs when a new classifier is produced from scratch or by copy.

        Parameters
        ----------
            previous_situation: Perception
            situation: Perception
        """
        length = self.cfg.classifier_length
        wildcard = self.cfg.classifier_wildcard
        if not self.is_enhanced():
            for idx in range(length):
                if previous_situation[idx] != situation[idx] and self.effect[0][idx] == wildcard:
                    self.effect[0][idx] = situation[idx]
                    self.condition[idx] = previous_situation[idx]
        else:
            if self.aliased_state != previous_situation:
                for idx in range(length):
                    if self.aliased_state[idx] != previous_situation[idx]:
                        self.condition[idx] = self.aliased_state[idx]
                        self.effect.enhanced_trace_ga[idx] = False
            else:
                new_effect_index = len(self.effect)
                self.effect.effect_list.append(Effect.empty(length, wildcard))
                self.effect.effect_detailled_counter.append(1)
                for idx in range(length):
                    if previous_situation[idx] != situation[idx]:
                        self.effect[new_effect_index][idx] = situation[idx]
                        self.condition[idx] = previous_situation[idx]
                self.effect.update_enhanced_trace_ga(length)


    def average_fitnesses_from_other_cl(
            self,
            other: BEACSClassifier
        ) -> None:
        """
        Average fitnesses from other BEACSClassifier cl (q, r and r_bis).
        Carefull, both fitnesses are modified.

        Parameters
        ----------
            other: BEACSClassifier
        """
        super().average_fitnesses_from_other_cl(other)
        self.r_bis = other.r_bis = (self.r_bis + other.r_bis) / 2.0


    def weighted_average_rewards_from_other_cl(
            self,
            other: BEACSClassifier
        ) -> None:
        """
        Average fitnesses depending on experience of both classifiers (q, r and r_bis).
        Carefull, both fitnesses are modified.

        Parameters
        ----------
            other: BEACSClassifier
        """
        super().weighted_average_rewards_from_other_cl(other)
        self.r_bis = (self.exp * self.r_bis + other.exp * other.r_bis) / (self.exp + other.exp)


    def merge_with(
            self,
            other_classifier: BEACSClassifier,
            aliased_state: Perception,
            time: int
        ) -> BEACSClassifier:
        """
        Merges two classifier in an enhanced one.

        Parameters
        ----------
            other_classifier: BEACSClassifier
            aliased_state: Perception
            time: int

        Returns
        -------
        BEACSClassifier
        """
        result = self.copy(time=time)
        result.q = max((self.q + other_classifier.q) / 2.0, 0.5)
        result.r = (self.r + other_classifier.r) / 2.0
        result.r_bis = (self.r_bis + other_classifier.r_bis) / 2.0
        result.condition.specialize_with_condition(other_classifier.condition)
        result.effect.enhance(other_classifier.effect, self.cfg.classifier_length)
        result.aliased_state = aliased_state
        return result