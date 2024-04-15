"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations


class ProbabilityEnhancedAttribute(dict):
    """
    A dict of symbols with the related occurence probabilities.
    """

    def __init__(self, attr):
        super().__init__()
        if isinstance(attr, str):
            self[attr] = 1.0
        if isinstance(attr, dict):
            for symbol in attr:
                self[symbol] = attr[symbol]
        self.adjust_probabilities()


    @classmethod
    def merged_attributes(
            cls,
            attr1: str,
            attr2: str
        ) -> ProbabilityEnhancedAttribute:
        """
        Create a new enhanced effect part.

        Parameters
        ----------
            attr1: str
            attr2: str

        Returns
        ----------
        ProbabilityEnhancedAttribute
        """
        result = ProbabilityEnhancedAttribute(attr1)
        result.insert(attr2)
        return result


    def copy(self) -> ProbabilityEnhancedAttribute:
        """
        Copy a PEP.

        Returns
        ----------
        ProbabilityEnhancedAttribute
        """
        return ProbabilityEnhancedAttribute(self)


    def does_contain(
            self,
            symbol: str
        ) -> bool:
        """
        Checks whether the specified symbol occurs in the attribute.

        Parameters
        ----------
            symbol: str

        Returns
        ----------
        bool
        """
        return self.get(symbol, 0.0) != 0.0


    def subsumes(
            self,
            other: ProbabilityEnhancedAttribute
        ) -> bool:
        """
        Check if one Pep subsumes another one.

        Parameters
        ----------
            other: ProbabilityEnhancedAttribute

        Returns
        ----------
        bool
        """
        return self.keys() >= other.keys()


    def make_compact(self) -> None:
        """
        Delete symbol in PEP if their probability is zero.
        """
        for symbol, prob in list(self.items()):
            if prob == 0.0:
                del self[symbol]


    def adjust_probabilities(
            self,
            prev_sum: float = None
        ) -> None:
        """
        Adjust the probabilities to sum to one.

        Parameters
        ----------
            prev_sum: float
        """
        if prev_sum is None:
            prev_sum = sum(self.get(symbol, 0.0) for symbol in self)
        for symbol in self:
            self[symbol] /= prev_sum


    def increase_probability(
            self,
            effect_symbol: str,
            update_rate: float
        ) -> None:
        """
        Increase the probability related to effect_symbol with update_rate.

        Parameters
        ----------
            effect_symbol: str
            update_rate: float
        """
        update_delta = update_rate * (1.0 - self[effect_symbol])
        self[effect_symbol] += update_delta
        self.adjust_probabilities(1.0 + update_delta)


    def symbols_specified(self) -> set:
        """
        Return set of symbols whose probability is not zero.

        Returns
        ----------
        set
        """
        return {k for k, v in self.items() if v > 0.0}


    def insert_symbol(
            self,
            symbol: str
        ) -> None:
        """
        Insert or update PEP from symbol.

        Parameters
        ----------
            symbol: str
        """
        self[symbol] = self.get(symbol, 0.0) + 1.0 / len(self)
        self.adjust_probabilities()


    def insert_attribute(
            self,
            o: ProbabilityEnhancedAttribute
        ) -> None:
        """
        Insert or update symbol of PEP from PEP.

        Parameters
        ----------
            o: ProbabilityEnhancedAttribute
        """
        for symbol in self.symbols_specified().union(o.symbols_specified()):
            self[symbol] = self.get(symbol, 0.0) + o.get(symbol, 0.0)
        self.adjust_probabilities()


    def insert(
            self, 
            symbol_or_attr: str | ProbabilityEnhancedAttribute
        ) -> None:
        """
        Insert or update symbol of PEP from PEP or symbol.

        Parameters
        ----------
            symbol_or_attr: str | ProbabilityEnhancedAttribute
        """
        if isinstance(symbol_or_attr, ProbabilityEnhancedAttribute):
            self.insert_attribute(symbol_or_attr)
        else:
            self.insert_symbol(symbol_or_attr)


    def sorted_items(self) -> list[tuple]:
        """
        Return the pep by probability descending order.

        Returns
        ----------
        list[tuple]
        """
        return sorted(self.items(), key=lambda x: x[1], reverse=True)


    def is_similar(
            self,
            other: str | ProbabilityEnhancedAttribute
        ) -> bool:
        """
        Determines if the two lists specify the same characters.
        Order and probabilities are not considered.

        Parameters
        ----------
            other: str | ProbabilityEnhancedAttribute

        Returns
        ----------
        bool
        """
        if isinstance(other, ProbabilityEnhancedAttribute):
            return self.symbols_specified() == other.symbols_specified()
        else:
            return self.symbols_specified() == {other}


    def __eq__(self, other) -> bool:
        return self.is_similar(other)


    def __str__(self):
        if len(self) == 1:
            return str(next(iter(self)))
        return "{" + ", ".join( "{}:{:.2f}%".format(sym[0], sym[1] * 100) for sym in self.sorted_items()) + "}"