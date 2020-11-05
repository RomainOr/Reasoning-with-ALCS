"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations

import random
from typing import Optional, Union, Callable, List

from beacs import Perception, UBR
from beacs.agents.beacs import Configuration, Condition, Anticipation, Effect, PMark


class Classifier:

    __slots__ = ['condition', 'action', 'behavioral_sequence', 'anticipation', 'mark', 'q', 'ra', 'rb',
                 'ir', 'num', 'exp', 'talp', 'tga', 'tbseq', 'tav', 'cfg', 'ee', 'pai_state', 'err']

    # In paper it's advised to set experience and reward of newly generated
    # classifier to 0. However in original code these values are initialized
    # with defaults 1 and 0.5 correspondingly.
    def __init__(
            self,
            condition: Union[Condition, str, None] = None,
            action: Optional[int] = None,
            behavioral_sequence: Optional[List[int]] = None,
            anticipation: Optional[Anticipation] = None,
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

        def _build_perception_string(
                cls,
                initial,
                length=self.cfg.classifier_length,
                wildcard=self.cfg.classifier_wildcard
            ):
            if initial:
                return cls(initial, wildcard=wildcard)
            return cls.empty(wildcard=wildcard, length=length)

        self.condition = _build_perception_string(Condition, condition)
        self.action = action
        self.behavioral_sequence = behavioral_sequence
        self.anticipation = Anticipation(_build_perception_string(Effect, anticipation), self.cfg.classifier_wildcard)
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
        self.err = 0.


    def __eq__(self, other):
        if self.condition == other.condition and \
                self.action == other.action and \
                self.behavioral_sequence == other.behavioral_sequence and \
                self.anticipation == other.anticipation:
            return True
        return False


    def __ne__(self, other):
        return not self.__eq__(other)


    def __hash__(self):
        return hash((str(self.condition), self.action, str(self.anticipation)))


    def __repr__(self):
        return f"{self.condition} {self.action} {str(self.behavioral_sequence)} {str(self.anticipation)} ({str(self.mark)})\n" \
            f"q: {self.q:<6.4} ra: {self.ra:<6.4} rb: {self.rb:<6.4} ir: {self.ir:<6.4} f: {self.fitness:<6.4} err: {self.err:<6.4}\n" \
            f"exp: {self.exp:<5} num: {self.num} ee: {self.ee} PAI_state: {', '.join(str(attr) for attr in self.pai_state)}\n" \
            f"tga: {self.tga:<5} tbseq: {self.tbseq:<5} talp: {self.talp:<5} tav: {self.tav:<6.4} \n" \


    @classmethod
    def copy_from(
            cls,
            old_cls: Classifier,
            p0: Perception,
            p1: Perception,
            time: int
        ) -> Classifier:
        """
        Copies old classifier with given time.
        Old tav gets replaced with new value.
        New classifier also has no mark.

        Parameters
        ----------
        old_cls: Classifier
            Classifier to copy from
        p: Perception
            Current perception to refine effect component
        time: int
            Current epoch

        Returns
        -------
        Classifier
            New copied classifier - Hard copy
        """
        new_cls = cls(
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
        # Condition copied
        for idx in range(len(new_cls.condition)):
            if isinstance(old_cls.condition[idx], UBR):
                new_cls.condition[idx] = UBR.copy(old_cls.condition[idx])
            else:
                new_cls.condition[idx] = old_cls.condition[idx]
        # Effect contrained reconstruction
        if old_cls.is_enhanced():
            for idx in range(len(new_cls.anticipation[0])):
                change_anticipated = False
                for effect in old_cls.anticipation.effect_list:
                    if effect[idx] != effect.wildcard:
                        change_anticipated = True
                        break
                if change_anticipated:
                    if new_cls.condition[idx] == new_cls.condition.wildcard or p1[idx] not in new_cls.condition[idx]:
                        new_cls.anticipation[0][idx] = UBR(p1[idx] - new_cls.cfg.spread/2., p1[idx] + new_cls.cfg.spread/2.)
        else:
            for idx in range(len(new_cls.anticipation[0])):
                if isinstance(old_cls.anticipation[0][idx], UBR):
                    new_cls.anticipation[0][idx] = UBR.copy(old_cls.anticipation[0][idx])
                else:
                    new_cls.anticipation[0][idx] = old_cls.anticipation[0][idx]
        return new_cls


    @property
    def fitness(self) -> float:
        """
        Computes the fitness of the classifier.

        Returns
        -------
        Float
            Fitness value
        """
        epsilon = 1e-6
        max_r = max(self.ra, self.rb)
        min_r = min(self.ra, self.rb)
        diff = max_r - min_r + epsilon
        if self.behavioral_sequence:
            return self.q * (max_r - diff * len(self.behavioral_sequence) / self.cfg.bs_max)
        return self.q * max_r


    @property
    def specified_unchanging_attributes(self) -> List[int]:
        """
        Determines the number of non-# symbols in cl.C

        Returns
        -------
        List[int]
            List specified specified attributes indices
        """
        indices = []
        for idx, ci in enumerate(self.condition):
                if ci != self.condition.wildcard:
                    indices.append(idx)
        return indices


    @property
    def specificity(self) -> float:
        """
        Computes the specificity of the classifier.

        Returns
        -------
        Float
            Specificity value
        """
        return self.condition.specificity


    def is_enhanced(self) -> bool:
        """
        Checks whether the classifier is enhanced.

        Returns
        -------
        bool
            True if the classifier is enhanced
        """
        return self.anticipation.is_enhanced()


    def is_reliable(self) -> bool:
        """
        Checks whether the classifier is reliable.

        Returns
        -------
        bool
            True if the classifier is reliable
        """
        return self.q > self.cfg.theta_r


    def is_inadequate(self) -> bool:
        """
        Checks whether the classifier is inadequate.

        Returns
        -------
        bool
            True if the classifier is inadequate
        """
        return self.q < self.cfg.theta_i


    def is_marked(self):
        """
        Checks whether the classifier is marked.

        Returns
        -------
        bool
            True if the classifier is marked
        """
        return self.mark.is_marked()


    def is_more_general(
            self,
            other: Classifier
        ) -> bool:
        """
        Checks if the classifiers condition is formally
        more general than `other`s.

        Parameters
        ----------
        other: Classifier
            Cther classifier to compare

        Returns
        -------
        bool
            True if classifier is more general than other
        """
        return self.condition.specificity <= other.condition.specificity


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
        return self.anticipation.is_specializable(p0, p1)


    def does_anticipate_change(self) -> bool:
        """
        Checks whether any change in environment is anticipated.

        Returns
        -------
        bool
            True if the effect part contains any specified attributes
        """
        return self.anticipation.specify_change


    def does_match(
            self,
            other: Union[Perception, Condition]
        ) -> bool:
        """
        Returns if the classifier matches the situation.

        Parameters
        -------
        situation

        Returns
        -------
        bool
        """
        return self.condition.does_match(other)


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
            Perception related to a state preceding the action
        situation: Perception
            Perception related to a state following the action

        Returns
        -------
        bool
            True if classifier's effect pat anticipates correctly
        """
        return self.anticipation.does_anticipate_correctly(previous_situation, situation)


    def does_predict_successfully(
            self,
            p0: Perception,
            action: int,
            p1: Perception
        ) -> bool:
        """
        Checks if classifier matches previous situation `p0`,
        has action `action` and predicts the effect `p1`.

        Usefull to compute knowlegde metric

        Parameters
        ----------
        p0: Perception
            Previous situation
        action: int
            Action
        p1: Perception
            Anticipated situation

        Returns
        -------
        bool
            True if classifier makes successful predictions
        """
        if self.condition.does_match(p0):
            if self.action == action:
                if self.does_anticipate_correctly(p0, p1):
                    return True
        return False


    def increase_experience(self):
        """
        Increases the experience of a classifier.
        """
        self.exp += 1


    def increase_quality(self):
        """
        Increases the quality of a classifier.
        """
        self.q += self.cfg.beta_alp * (1 - self.q)


    def decrease_quality(self):
        """
        Decreases the quality of a classifier.
        """
        self.q -= self.cfg.beta_alp * self.q


    def set_mark(
            self,
            perception: Perception
        ) -> None:
        """
        Specializes the mark in all attributes.

        Parameters
        ----------
        perception: Perception
            Current situation
        """
        self.ee = self.mark.set_mark(perception, self.ee)


    def set_alp_timestamp(
            self,
            time: int
        ) -> None:
        """
        Sets the ALP time stamp and the application average parameter.

        Parameters
        ----------
        time: int
            Current step
        """
        if self.exp < 1. / self.cfg.beta_alp:
            self.tav += (time - self.talp - self.tav) / self.exp
        else:
            self.tav += self.cfg.beta_alp * (time - self.talp - self.tav)
        self.talp = time


    def update_anticipation_counter(
            self,
            p0: Perception,
            p1: Perception
        ) -> None:
        """
        Updates the counter of respective effect when it correctly anticipates.

        Parameters
        ----------
        p0: Perception
            Previous perception
        p1: Perception
            Current perception
        """
        self.anticipation.update_anticipation_counter(p0,p1)


    def specialize(
            self,
            previous_situation: Perception,
            situation: Perception
        ) -> None:
        """
        Specializes the effect part where necessary to correctly anticipate
        the changes from previous_situation to situation.
        Only occurs when a new classifier is produced from scratch or by copy.

        Parameters
        ----------
        previous_situation: Perception
            Perception related to a state preceding the action
        situation: Perception
            Perception related to a state following the action
        """
        for idx, _ in enumerate(situation):
            if previous_situation[idx] != situation[idx] and self.anticipation[0][idx] == self.cfg.classifier_wildcard:
                self.anticipation[0][idx] = UBR(situation[idx] - self.cfg.spread/2., situation[idx] + self.cfg.spread/2.)
                self.condition[idx] = UBR(previous_situation[idx] - self.cfg.spread/2., previous_situation[idx] + self.cfg.spread/2.)


    def specialize_with_condition(
            self,
            other: Condition
        ) -> None:
        """
        Specializes the condition with another one.

        Parameters
        ----------
        other: Condition
            Condition object
        """
        self.condition.specialize_with_condition(other)


    def generalize_condition_attribute(self, idx):
        """
        Generalizes one attribute in the condition at idx.
        """
        self.condition.generalize(idx)


    def generalize_unchanging_condition_attribute(self):
        """
        Generalizes one randomly unchanging attribute in the condition.
        """
        ridx = random.choice(self.specified_unchanging_attributes)
        self.condition.generalize(ridx)


    def merge_with(
            self,
            other_classifier,
            time
        ) -> Classifier:
        """
        Merges two classifier in an enhanced one.

        Parameters
        ----------
        other_classifier: Classifier
            Classifier to merge with the self one
        time: int
            Current epoch

        Returns
        -------
        Classifier
            New enhanced classifier
        """
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
        result.specialize_with_condition(other_classifier.condition)
        result.anticipation = Anticipation.enhanced(
            self.anticipation, 
            other_classifier.anticipation)
        return result