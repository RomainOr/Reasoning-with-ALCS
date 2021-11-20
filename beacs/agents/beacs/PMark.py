import random
from typing import List

from beacs import Perception, TypedList
from . import Configuration, Condition


class PMark(TypedList):

    def __init__(self, cfg: Configuration) -> None:
        self.cfg = cfg
        initial: List = [set() for _ in range(self.cfg.classifier_length)]
        super().__init__((set,), *initial)


    def is_marked(self) -> bool:
        """
        Returns
        -------
        bool
            If mark is specified at any attribute
        """
        return any(len(attrib) != 0 for attrib in self)


    def set_mark(
            self,
            perception: Perception,
            is_ee: bool
        ) -> bool:
        """
        Specializes the mark in all attributes

        Parameters
        ----------
        perception: Perception
            Current situation
        is_ee: bool
            Indicates if the classifier is enhanceable
        """
        set_ee = is_ee
        for idx, item in enumerate(perception):
            if item not in self[idx]:
                self[idx].add(perception[idx])
                set_ee = False
        return set_ee


    def corresponds_to(
            self,
            perception: Perception
        ) -> bool:
        """
        Indicates if the mark corresponds to the perception

        Parameters
        ----------
        perception: Perception
            Current situation

        Returns
        ----------
        bool
            True if they correspond
        """
        if not self.one_situation_in_mark():
            return False
        for idx, item in enumerate(perception):
            if item not in self[idx]:
                return False
        return True


    def get_differences(
            self,
            p0: Perception
        ) -> Condition:
        """
        Determines the strongest differences in between the mark
        and current perception.

        Parameters
        ----------
        p0: Perception
            Current situation

        Returns
        ----------
        Condition that specifies all the differences.
        """
        diff = Condition.empty(
            wildcard=self.cfg.classifier_wildcard,
            length=self.cfg.classifier_length
        )
        # Count difference types
        nr1, nr2 = 0, 0
        for idx, item in enumerate(self):
            if len(item) > 0 and p0[idx] not in item:
                nr1 += 1
            elif len(item) > 1:
                nr2 += 1
        if nr1 > 0:
            possible_idx = [pi for pi, p in enumerate(p0) if
                            p not in self[pi] and len(self[pi]) > 0]
            rand_idx = random.choice(possible_idx)
            diff[rand_idx] = p0[rand_idx]
        elif nr2 > 0:
            for idx, item in enumerate(self):
                if len(item) > 1:
                    diff[idx] = p0[idx]
        return diff


    def one_situation_in_mark(self) -> bool:
        """
        Check if the mark correspond to only one situation or more..
        Help to detect aliased states in POMDPs
        
        Returns
        -------
        bool
        """
        if not self.is_marked():
            return False
        for _, item in enumerate(self):
            if len(item) > 1:
                return False
        return True


    def __repr__(self):
        def compact_set_str(s):
            if len(s) == 0:
                return self.cfg.classifier_wildcard
            elif len(s) == 1:
                return next(iter(s))  # the only element in set
            else:
                return '{' + ' '.join(x for x in s) + '}'

        if self.is_marked():
            return ''.join(compact_set_str(x) for x in self)
        else:
            return 'empty'
