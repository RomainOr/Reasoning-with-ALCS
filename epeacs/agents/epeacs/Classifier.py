"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations

import random
from typing import Optional, Union, Callable, List

from epeacs import Perception
from epeacs.agents.epeacs import Configuration, Condition, EffectList, Effect, PMark


class Classifier:

    __slots__ = ['condition', 'action', 'behavioral_sequence', 'effect', 'mark', 'q', 'ra', 'rb',
                 'ir', 'num', 'exp', 'talp', 'tga', 'tbseq', 'tav', 'cfg', 'ee', 'pai_state']

    # In paper it's advised to set experience and reward of newly generated
    # classifier to 0. However in original code these values are initialized
    # with defaults 1 and 0.5 correspondingly.
    def __init__(
            self,
            condition: Union[Condition, str, None] = None,
            action: Optional[int] = None,
            behavioral_sequence: Optional[List[int]] = None,
            effect: Optional[EffectList] = None,
            quality: float=0.5,
            rewarda: float=0.,
            rewardb: float=0.,
            immediate_reward: float=0.0,
            numerosity: int=1,
            experience: int=1,
            talp: int=0,
            tga: int=0,
            tbseq: int=0,
            tav: float=0.0,
            pai_state: Optional[Perception] = None,
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
        self.effect = EffectList(build_perception_string(Effect, effect), self.cfg.classifier_wildcard)
        self.mark = PMark(cfg=self.cfg)
        self.q = quality
        self.ra = rewarda
        self.rb = rewardb
        self.ir = immediate_reward
        self.num = numerosity
        self.exp = experience
        self.talp = talp
        self.tga = tga
        self.tbseq = tbseq
        self.tav = tav
        self.ee = False
        if pai_state:
            self.pai_state = pai_state
        else:
            self.pai_state = Perception.empty()


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
               f"{str(self.effect)} " \
               f"{'(' + str(self.mark) + ')':21} \n" \
               f"q: {self.q:<5.3} ra: {self.ra:<6.4} rb: {self.rb:<6.4} ir: {self.ir:<6.4} f: {self.fitness:<6.4} \n" \
               f"exp: {self.exp:<3} tga: {self.tga:<5} tbseq: {self.tbseq:<5} talp: {self.talp:<5} " \
               f"tav: {self.tav:<6.3} num: {self.num} ee: {self.ee}"


    @classmethod
    def copy_from(cls, old_cls: Classifier, p: Perception, time: int):
        """
        Copies old classifier with given time.
        Old tav gets replaced with new value.
        New classifier also has no mark.

        Parameters
        ----------
        old_cls: Classifier
            classifier to copy from
        time: int
            time of creation / current epoch
        p: Perception

        Returns
        -------
        Classifier
            copied classifier
        """
        new_cls = cls(
            condition=Condition(old_cls.condition, old_cls.cfg.classifier_wildcard),
            action=old_cls.action,
            behavioral_sequence=old_cls.behavioral_sequence,
            quality=old_cls.q,
            rewarda=old_cls.ra,
            rewardb=old_cls.rb,
            immediate_reward=old_cls.ir,
            cfg=old_cls.cfg,
            tga=time,
            tbseq=time,
            talp=time,
            tav=old_cls.tav,
            pai_state=old_cls.pai_state
        )
        if old_cls.is_enhanced():
            for idx in range(len(new_cls.effect[0])):
                change_anticipated = False
                for effect in old_cls.effect.effect_list:
                    if effect[idx] != effect.wildcard:
                        change_anticipated = True
                        break
                if change_anticipated and p[idx] != new_cls.condition[idx]:
                    new_cls.effect[0][idx] = p[idx]
        else:
            for idx in range(len(new_cls.effect[0])):
                new_cls.effect[0][idx] = old_cls.effect[0][idx]
        return new_cls


    @property
    def fitness(self):
        max_r = max(self.ra, self.rb)
        min_r = min(self.ra, self.rb)
        diff = max_r - min_r
        if self.behavioral_sequence:
            return self.q * (max_r - diff * len(self.behavioral_sequence) / self.cfg.bs_max)
        return self.q * max_r


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
        for idx, ci in enumerate(self.condition):
            if self.effect.is_enhanced():
                to_append = False
                for effect in self.effect.effect_list:
                    if ci != self.condition.wildcard and effect[idx] == ci:
                        to_append = True
                        break
                if to_append:
                    indices.append(idx)
            else:
                if ci != self.condition.wildcard and self.effect[0][idx] == self.effect.wildcard:
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


    def is_enhanced(self):
        return self.effect.is_enhanced()


    def is_reliable(self) -> bool:
        return self.q > self.cfg.theta_r


    def is_inadequate(self) -> bool:
        return self.q < self.cfg.theta_i


    def increase_experience(self) -> int:
        self.exp += 1
        return self.exp


    def increase_quality(self) -> float:
        self.q += self.cfg.beta_alp * (1 - self.q)
        return self.q


    def decrease_quality(self) -> float:
        self.q -= self.cfg.beta_alp * self.q
        return self.q


    def specialize(
            self,
            previous_situation: Perception,
            situation: Perception
        ) -> None:
        """
        Specializes the effect part where necessary to correctly anticipate
        the changes from p0 to p1.
        Only occurs when a new classifier is produced from scratch or by copy.

        Parameters
        ----------
        previous_situation: Perception
        situation: Perception
        """
        for idx, _ in enumerate(situation):
            if previous_situation[idx] != situation[idx] and self.effect[0][idx] == self.cfg.classifier_wildcard:
                self.effect[0][idx] = situation[idx]
                self.condition[idx] = previous_situation[idx]


    def predicts_successfully(
            self,
            p0: Perception,
            action: int,
            p1: Perception
        ) -> bool:
        """
        Check if classifier matches previous situation `p0`,
        has action `action` and predicts the effect `p1`.

        Usefull to compute knowlegde metric

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
        if self.exp < 1. / self.cfg.beta_alp:
            self.tav += (time - self.talp - self.tav) / self.exp
        else:
            self.tav += self.cfg.beta_alp * (time - self.talp - self.tav)
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


    def merge_with(self, other_classifier, perception, time):
        result = Classifier(
            action = self.action,
            behavioral_sequence=self.behavioral_sequence,
            quality = max((self.q + other_classifier.q) / 2.0, 0.5),
            rewarda = (self.ra + other_classifier.ra) / 2.0,
            rewardb = (self.rb + other_classifier.rb) / 2.0,
            talp = time,
            tga = time,
            tbseq = time,
            pai_state = self.pai_state,
            cfg = self.cfg
        )
        result.condition = Condition(self.condition)
        result.condition.specialize_with_condition(other_classifier.condition)
        result.effect = EffectList.enhanced_effect(
            self.effect, 
            other_classifier.effect,
            perception)
        return result