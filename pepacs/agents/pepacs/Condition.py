"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations

import random
from typing import Callable, Union

from pepacs import Perception
from pepacs.agents import AbstractPerception


class Condition(AbstractPerception):
    """
    Specifies the set of situations (perceptions) in which the classifier
    can be applied.
    """


    @property
    def specificity(self) -> int:
        """
        Returns
        -------
        int
            Number of not generic (wildcards) attributes
        """
        return sum(1 for attr in self if attr != self.wildcard)


    @property
    def wildcard_count(self) -> int:
        """
        Returns
        -------
        int
            Number of generic (wildcards) attributes
        """
        return sum(1 for attr in self if attr == self.wildcard)


    def specialize_with_condition(self, other: Condition) -> None:
        for idx, new_el in enumerate(other):
            if new_el != self.wildcard:
                self[idx] = new_el


    def generalize(self, position=None):
        self[position] = self.wildcard


    def generalize_specific_attribute_randomly(
            self, 
            func: Callable = random.choice
        ) -> None:
        """
        Generalizes one randomly selected specified attribute.

        Parameters
        ----------
        func: Callable
            Function for choosing which ID to generalize from the list of
            available ones
        """
        specific_ids = [ci for ci, c in enumerate(self) if c != self.wildcard]
        if len(specific_ids) > 0:
            ridx = func(specific_ids)
            self.generalize(ridx)


    def does_match(
            self,
            other: Union[Perception, Condition]
        ) -> bool:
        """
        Checks if condition match other list such as perception or another
        condition.

        Parameters
        ----------
        other: Union[Perception, Condition]
            Perception or condition object

        Returns
        -------
        bool
            True if condition match given list, False otherwise
        """
        for ci, oi in zip(self, other):
            if ci != self.wildcard and oi != self.wildcard and ci != oi:
                return False
        return True


    def subsumes(
        self,
        other: Condition
        ) -> bool:
        """
        Determines if the condition subsumes another condition.

        Parameters
        ----------
        other: Condition
            Other condition

        Returns
        -------
        bool
            True if self subsumes other
        """
        for ci, oi in zip(self, other):
            if ci != self.wildcard and oi != self.wildcard and ci != oi:
                return False
            if ci != self.wildcard and oi == self.wildcard:
                return False
        return True

