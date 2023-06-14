"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations
from typing import Optional, Union, List

from agents.common.Perception import Perception
from agents.common.BaseConfiguration import BaseConfiguration
from agents.common.classifier_components.Condition import Condition
from agents.common.classifier_components.PMark import PMark
from agents.common.classifier_components.Effect import Effect


class BaseClassifier:

    __slots__ = ['condition', 'action', 'behavioral_sequence', 'effect', 'mark', 'q', 'r',
        'ir', 'num', 'exp', 'talp', 'tga', 'tav', 'cfg', 'ee']

    def __init__(
            self,
            condition: Union[Condition, str, None] = None,
            action: Optional[int] = None,
            behavioral_sequence: Optional[List[int]] = None,
            effect: Optional[Effect] = None,
            quality: float=0.5,
            reward: float=0.,
            immediate_reward: float=0.0,
            numerosity: int=1,
            experience: int=1,
            talp: int=0,
            tga: int=0,
            tav: float=0.0,
            cfg: Optional[BaseConfiguration] = None
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
        self.effect = _build_perception_string(Effect, effect)
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
        return hash((str(self.condition), self.action, str(self.behavioral_sequence), str(self.effect)))


    def __repr__(self):
        return f"C:{self.condition} A:{self.action} {str(self.behavioral_sequence)} E:{str(self.effect)}\n" \
            f"q: {self.q:<6.4} r: {self.r:<6.4} ir: {self.ir:<6.4} f: {self.fitness:<6.4}\n" \
            f"exp: {self.exp:<5} num: {self.num} ee: {self.ee}\n" \
            f"Mark: {str(self.mark)} tga: {self.tga:<5} talp: {self.talp:<5} tav: {self.tav:<6.4} \n"


    @classmethod
    def copy_from(
            cls,
            old_cls: BaseClassifier,
            time: int
        ) -> BaseClassifier:
        """
        Copies old classifier with given time.
        Old tav gets replaced with new value.
        New classifier also has no mark.

        Parameters
        ----------
        old_cls: Classifier
            Classifier to copy from
        time: int
            Current epoch

        Returns
        -------
        Classifier
            New copied classifier - Hard copy
        """
        new_cls = cls(
            condition=Condition(old_cls.condition, old_cls.cfg.classifier_wildcard),
            action=old_cls.action,
            behavioral_sequence=old_cls.behavioral_sequence,
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
    def fitness(self) -> float:
        """
        Computes the fitness of the classifier.

        Returns
        -------
        Float
            Fitness value
        """
        return self.q * self.r


    @property
    def specificity(self) -> float:
        """
        Computes the specificity of the classifier.

        Returns
        -------
        Float
            Specificity value
        """
        return self.condition.specificity / len(self.condition)


    def is_experienced(self) -> bool:
        """
        Checks whether the classifier is enough experienced.

        Returns
        -------
        bool
            True if the classifier is enough experienced
        """
        return self.exp > self.cfg.theta_exp


    def is_inadequate(self) -> bool:
        """
        Checks whether the classifier is inadequate.

        Returns
        -------
        bool
            True if the classifier is inadequate
        """
        return self.q < self.cfg.theta_i


    def is_reliable(self) -> bool:
        """
        Checks whether the classifier is reliable.

        Returns
        -------
        bool
            True if the classifier is reliable
        """
        return self.q > self.cfg.theta_r


    def is_marked(self) -> bool:
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
            other: BaseClassifier
        ) -> bool:
        """
        Checks if the classifiers condition is formally
        more general than `other`s.

        Parameters
        ----------
        other: Classifier
            Other classifier to compare

        Returns
        -------
        bool
            True if classifier is more general than other
        """
        return self.condition.specificity <= other.condition.specificity


    def is_hard_subsumer_criteria_satisfied(self, other) -> bool:
        """
        Determines whether the classifier satisfies the hard subsumer criteria.

        Parameters
        ----------
        other: Classifier
            Other classifier to compare

        Returns
        -------
        bool
            True if the classifier satisfies the subsumer criteria.
        """
        if self.is_reliable() and self.is_experienced():
            if not self.is_marked():
                return True
            if self.is_marked() and other.is_marked() and self.mark == other.mark:
                return True
        return False


    def is_soft_subsumer_criteria_satisfied(self, other) -> bool:
        """
        Determines whether the classifier satisfies the soft subsumer criteria.

        Parameters
        ----------
        other: Classifier
            Other classifier to compare

        Returns
        -------
        bool
            True if the classifier satisfies the subsumer criteria.
        """
        if self.is_reliable() or (self.q > other.q):
            if not self.is_marked():
                return True
            if self.is_marked() and other.is_marked() and self.mark == other.mark:
                return True
        return False


    def is_specializable(
            self,
            previous_situation: Perception,
            situation: Perception
        ) -> bool:
        """
        Determines if both effect and condition can be modified to
        correctly anticipate changes from `p0` to `p1`.

        Parameters
        ----------
        previous_situation: Perception
            Previous perception
        situation: Perception
            Current perception

        Returns
        -------
        bool
            True if specializable
        """
        return self.effect.is_specializable(previous_situation, situation)


    def does_anticipate_change(self) -> bool:
        """
        Checks whether any change in environment is anticipated.

        Returns
        -------
        bool
            True if the effect part contains any specified attributes
        """
        return self.effect.specify_change


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
        return self.effect.does_anticipate_correctly(previous_situation, situation)


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
        if self.does_match(p0):
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
            Perception related to a state preceding the action
        situation: Perception
            Perception related to a state following the action
        """
        for idx in range(self.cfg.classifier_length):
            if previous_situation[idx] != situation[idx] and self.effect[idx] == self.cfg.classifier_wildcard:
                self.effect[idx] = situation[idx]
                self.condition[idx] = previous_situation[idx]


    def average_fitnesses_from_other_cl(self, other):
        self.q = other.q = (self.q + other.q) / 2.0
        self.r = other.r = (self.r + other.r) / 2.0


    def weighted_average_rewards_from_other_cl(self, other):
        self.r = (self.exp * self.r + other.exp * other.r) / (self.exp + other.exp)


    def generalize_specific_attribute_randomly(self):
        """
        Generalizes one randomly attribute in the condition.
        """
        self.condition.generalize_specific_attribute_randomly()


    def subsumes(self, other) -> bool:

        """
        Check if one classifier can subsume another one.
        Can be used anywhere and anytime...

        Parameters
        ----------
        self: Classifier
        other: Classifier

        Returns
        -------
        bool
            True if self sulbsumes other
        """
        if self.condition.subsumes(other.condition) and \
                self.action == other.action and \
                self.behavioral_sequence == other.behavioral_sequence and \
                self.effect.subsumes(other.effect) and \
                self.is_soft_subsumer_criteria_satisfied(other):
            return True
        return False
