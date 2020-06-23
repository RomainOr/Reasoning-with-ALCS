from copy import copy

class AbstractPerception:

    def __init__(self, observation, wildcard='#', oktypes=(str, dict)):
        obs = tuple(observation)
        self.oktypes = oktypes
        assert type(wildcard) in self.oktypes
        assert all(isinstance(o, self.oktypes) for o in obs)
        self._items = obs
        self.wildcard = wildcard


    @classmethod
    def empty(cls, length: int, wildcard='#', oktypes=(str, dict)):
        """
        Creates an AbstractPerception composed from wildcard symbols.
        Note that in case that wildcard is an object is get's copied
        not to be accessed by reference.

        Parameters
        ----------
        length: int
            length of perception string
        wildcard: Any
            wildcard symbol
        oktypes: (str, dict)
            tuple of allowed classes to represent perception string

        Returns
        -------
        AbstractPerception
            generic AbstractPerception
        """
        ps_str = [copy(wildcard) for _ in range(length)]
        return cls(ps_str)


    def subsumes(self, other) -> bool:
        """
        Checks if given perception string subsumes other one.
        Parameters
        ----------
        other: AbstractPerception

        Returns
        -------
        bool
            True if `other` is subsumed by `self`, False otherwise
        """
        raise NotImplementedError()


    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        assert isinstance(value, self.oktypes)
        lst = list(self._items)
        lst[index] = value

        self._items = tuple(lst)

    def __eq__(self, other):
        return self._items == other._items

    def __hash__(self):
        return hash(self._items)

    def __repr__(self):
        return ''.join(map(str, self._items))