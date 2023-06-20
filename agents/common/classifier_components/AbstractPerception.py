from __future__ import annotations
from copy import copy

class AbstractPerception:

    def __init__(
            self,
            observation,
            wildcard='#'
        ) -> None:
        self._items = tuple(observation)
        self.wildcard = wildcard

    @classmethod
    def empty(
            cls, 
            length: int,
            wildcard='#'
        ) -> AbstractPerception:
        """
        Creates an AbstractPerception composed from wildcard symbols.
        Note that in case that wildcard is an object is get's copied
        not to be accessed by reference.

        Parameters
        ----------
            length: int
            wildcard='#'

        Returns
        -------
        AbstractPerception
        """
        ps_str = [copy(wildcard) for _ in range(length)]
        return cls(ps_str)

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value) -> None:
        lst = list(self._items)
        lst[index] = value
        self._items = tuple(lst)

    def __eq__(self, other) -> bool:
        return self._items == other._items

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return ''.join(map(str, self._items))

    def __str__(self) -> str:
        return ''.join(str(attr) for attr in self)