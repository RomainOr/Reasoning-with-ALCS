import collections.abc


class TypedList(collections.abc.MutableSequence):

    __slots__ = ['_items', 'oktypes']

    def __init__(self, oktypes, *args):
        self._items = list()
        self.oktypes = oktypes

        for el in args:
            if not isinstance(el, oktypes):
                raise TypeError(f"Wrong element type: object {el}, type {type(el)}")

        self._items.extend(list(args))

    def insert(self, index: int, el) -> None:
        if not isinstance(el, self.oktypes):
            raise TypeError(f"Wrong element type: object {el}, type {type(el)}")
        self._items.insert(index, el)

    def safe_remove(self, o) -> None:
        try:
            self.remove(o)
        except ValueError:
            pass

    def sort(self, *args, **kwargs) -> None:
        self._items.sort(*args, **kwargs)

    def __repr__(self):
        return f"{len(self._items)} items"

    def __setitem__(self, i, el):
        if not isinstance(el, self.oktypes):
            raise TypeError(f"Wrong element type: object {el}, type {type(el)}")
        self._items[i] = el

    def __delitem__(self, i):
        del self._items[i]

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self):
        return hash((self.oktypes, self._items))

    def __eq__(self, o) -> bool:
        return self.oktypes == o.oktypes \
            and self._items == o._items
