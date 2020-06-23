"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations

import random
from typing import Optional, Union, Callable, List

from bacs import Perception
from bacs.agents.bacs import Configuration, Condition, Effect, PMark


class Classifier:

    __slots__ = ['condition', 'action','behavioral_sequence' ,'effect', 'mark', 'q', 'r',
                 'ir', 'num', 'exp', 'talp', 'tga', 'tav', 'cfg', 'ee']

    def __init__(
            self,
            condition: Union[Condition, str, None] = None,
            action: Optional[int] = None,
            behavioral_sequence: Optional[List[int]] = None,
            effect: Union[Effect, str, None] = None,
            quality: float=0.5,
            reward: float=0.5,
            immediate_reward: float=0.0,
            numerosity: int=1,
            experience: int=1,
            talp: int=0,
            tga: int=0,
            tav: float=0.0,
            cfg: Optional[Configuration] = None
        ) -> None:
        if cfg is None:
            raise TypeError("Configuration should be passed to Classifier")
        self.cfg = cfg

        def build_perception_string(
                cls,
                initial,
                length=self.cfg.classifier_length,
                wildcard=self.cfg.classifier_wildcard
            ):
            if initial:
                return cls(initial, wildcard=wildcard)
            return cls.empty(wildcard=wildcard, length=length)

        self.condition = build_perception_string(Condition, condition)
        self.action = action
        self.behavioral_sequence = behavioral_sequence
        self.effect = build_perception_string(Effect, effect)
        self.mark = PMark(cfg=self.cfg)
        self.q = quality
        self.r = reward
        self.ir = immediate_reward
        self.num = numerosity
        self.exp = experience
        self.talp = talp
        self.tga = tga
        self.tav = tav
        self.ee = False


    def __eq__(self, other):
        if self.condition == other.condition and \
                self.action == other.action and \
                self.behavioral_sequence == other.behavioral_sequence and \
                self.effect == other.effect:
            return True

        return False


    def __ne__(self, other):
        return not self.__eq__(other)


    def __hash__(self):
        return hash((str(self.condition), self.action, str(self.effect)))


    def __repr__(self):
        return f"{self.condition} " \
               f"{self.action} " \
               f"{str(self.behavioral_sequence)} " \
               f"{str(self.effect):16} " \
               f"{'(' + str(self.mark) + ')':21} \n" \
               f"q: {self.q:<5.3} " \
               f"r: {self.r:<6.4} ir: {self.ir:<6.4} f: {self.fitness:<6.4} " \
               f"exp: {self.exp:<3} tga: {self.tga:<5} talp: {self.talp:<5} " \
               f"tav: {self.tav:<6.3} num: {self.num} ee: {self.ee}"


    @classmethod
    def copy_from(cls, old_cls: Classifier, p: Perception, time: int):
        """
        Copies old classifier with given time (tga, talp).
        Old tav gets replaced with new value.
        New classifier also has no mark.

        Parameters
        ----------
        old_cls: Classifier
            classifier to copy from
        time: int
            time of creation / current epoch

        Returns
        -------
        Classifier
            copied classifier
        """
        new_cls = cls(
            condition=Condition(old_cls.condition, old_cls.cfg.classifier_wildcard),
            action=old_cls.action,
            behavioral_sequence=old_cls.behavioral_sequence,
            effect=Effect(old_cls.effect, old_cls.cfg.classifier_wildcard),
            quality=old_cls.q,
            reward=old_cls.r,
            immediate_reward=old_cls.ir,
            cfg=old_cls.cfg,
            tga=time,
            talp=time,
            tav=old_cls.tav
        )
        return new_cls


    @property
    def fitness(self):
        return self.q * self.r


    @property
    def specified_unchanging_attributes(self) -> List[int]:
        """
        Determines the number of specified unchanging attributes in
        the classifier. An unchanging attribute is one that is anticipated
        not to change in the effect part.

        Returns
        -------
        List[int]
            list specified unchanging attributes indices
        """
        indices = []
        for idx, (cpi, epi) in enumerate(zip(self.condition, self.effect)):
            if cpi != self.cfg.classifier_wildcard and epi == self.cfg.classifier_wildcard:
                indices.append(idx)
        return indices


    @property
    def specificity(self):
        return self.condition.specificity / len(self.condition)


    def does_anticipate_change(self) -> bool:
        """
        Checks whether any change in environment is anticipated

        Returns
        -------
        bool
            true if the effect part contains any specified attributes
        """
        return self.effect.specify_change


    def is_reliable(self) -> bool:
        return self.q > self.cfg.theta_r


    def is_inadequate(self) -> bool:
        return self.q < self.cfg.theta_i


    def increase_experience(self) -> int:
        self.exp += 1
        return self.exp


    def increase_quality(self) -> float:
        self.q += self.cfg.beta * (1 - self.q)
        return self.q


    def decrease_quality(self) -> float:
        self.q -= self.cfg.beta * self.q
        return self.q


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
                self.condition[idx] = previous_situation[idx]


    def predicts_successfully(
            self,
            p0: Perception,
            action: int,
            p1: Perception
        ) -> bool:
        """
        Check if classifier matches previous situation `p0`,
        has action `action` and predicts the effect `p1`

        Parameters
        ----------
        p0: Perception
            previous situation
        action: int
            action
        p1: Perception
            anticipated situation after execution action

        Returns
        -------
        bool
            True if classifier makes successful predictions, False otherwise
        """
        if self.condition.does_match(p0):
            if self.action == action:
                if self.does_anticipate_correctly(p0, p1):
                    return True
        return False


    def does_anticipate_correctly(
            self,
            previous_situation: Perception,
            situation: Perception
        ) -> bool:
        """
        Checks anticipation. While the pass-through symbols in the effect part
        of a classifier directly anticipate that these attributes stay the same
        after the execution of an action, the specified attributes anticipate
        a change to the specified value. Thus, if the perceived value did not
        change to the anticipated but actually stayed at the value, the
        classifier anticipates incorrectly.

        Parameters
        ----------
        previous_situation: Perception
            Previous situation
        situation: Perception
            Current situation

        Returns
        -------
        bool
            True if classifier's effect pat anticipates correctly,
            False otherwise
        """
        return self.effect.anticipates_correctly(previous_situation, situation)


    def set_mark(self, perception: Perception) -> None:
        """
        Specializes the mark in all attributes

        Parameters
        ----------
        perception: Perception
            current situation
        """
        self.ee = self.mark.set_mark(perception, self.ee)


    def set_alp_timestamp(self, time: int) -> None:
        """
        Sets the ALP time stamp and the application average parameter.

        Parameters
        ----------
        time: int
            current step
        """
        if 1. / self.exp > self.cfg.beta:
            self.tav = (self.tav * self.exp + (time - self.talp)) / (
                self.exp + 1)
        else:
            self.tav += self.cfg.beta * ((time - self.talp) - self.tav)
        self.talp = time


    def is_more_general(self, other: Classifier) -> bool:
        """
        Checks if the classifiers condition is formally
        more general than `other`s.

        Parameters
        ----------
        other: Classifier
            other classifier to compare

        Returns
        -------
        bool
            True if classifier is more general than other
        """
        return self.condition.specificity <= other.condition.specificity


    def generalize_unchanging_condition_attribute(
                self,
                randomfunc: Callable=random.choice
            ) -> bool:
        """
        Generalizes one randomly unchanging attribute in the condition.
        An unchanging attribute is one that is anticipated not to change
        in the effect part.

        Parameters
        ----------
        randomfunc: Callable
            function returning attribute index to generalize

        Returns
        -------
        bool
            True if attribute was generalized, False otherwise
        """
        if len(self.specified_unchanging_attributes) > 0:
            ridx = randomfunc(self.specified_unchanging_attributes)
            self.condition.generalize(ridx)
            return True
        return False


    def is_marked(self):
        return self.mark.is_marked()


    def does_match(self, situation: Perception) -> bool:
        """
        Returns if the classifier matches the situation.
        Parameters
        -------
        situation

        Returns
        -------
        bool
        """
        return self.condition.does_match(situation)
