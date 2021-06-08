import collections.abc
import itertools

from . import check_types


class Perception(collections.abc.Sequence):
    """
    Represents current state of the environment at given time instance.
    By default each environment attribute is represented as `str` type.
    """

    __slots__ = ['_items', 'oktypes']

    def __init__(self, observation, oktypes=(str,)):
        self._items = list()

        for el in observation:
            check_types(oktypes, el)

        self._items.extend(list(observation))

    @classmethod
    def empty(cls):
        return cls([], oktypes=(None,))

    def is_empty(self):
        return len(self._items) == 0

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self):
        return ''.join(map(str, self))

    def __eq__(self, other):
        if len(self) != len(other):
            return False 
        for si, oi in zip(self, other):
            if si != oi:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)
