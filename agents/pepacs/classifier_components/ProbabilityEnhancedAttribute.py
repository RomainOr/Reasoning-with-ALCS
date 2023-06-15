"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

class ProbabilityEnhancedAttribute(dict):

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
            attr1,
            attr2
        ):
        """
        Create a new enhanced effect part.
        """
        result = ProbabilityEnhancedAttribute(attr1)
        result.insert(attr2)
        return result


    def copy(self):
        return ProbabilityEnhancedAttribute(self)


    def does_contain(self, symbol):
        """
        Checks whether the specified symbol occurs in the attribute.
        """
        return self.get(symbol, 0.0) != 0.0


    def subsumes(self, other):
        """
        Check if one Pep subsumes another one
        """
        return self.keys() >= other.keys()


    def make_compact(self):
        for symbol, prob in list(self.items()):
            if prob == 0.0:
                del self[symbol]


    def adjust_probabilities(self, prev_sum=None):
        """
        Adjust the probabilities to sum to one
        """
        if prev_sum is None:
            prev_sum = sum(self.get(symbol, 0.0) for symbol in self)
        for symbol in self:
            self[symbol] /= prev_sum


    def increase_probability(self, effect_symbol, update_rate):
        update_delta = update_rate * (1.0 - self[effect_symbol])
        self[effect_symbol] += update_delta
        self.adjust_probabilities(1.0 + update_delta)


    def symbols_specified(self):
        return {k for k, v in self.items() if v > 0.0}


    def insert_symbol(self, symbol):
        self[symbol] = self.get(symbol, 0.0) + 1.0 / len(self)
        self.adjust_probabilities()


    def insert_attribute(self, o):
        for symbol in self.symbols_specified().union(o.symbols_specified()):
            self[symbol] = self.get(symbol, 0.0) + o.get(symbol, 0.0)
        self.adjust_probabilities()


    def insert(self, symbol_or_attr):
        if isinstance(symbol_or_attr, ProbabilityEnhancedAttribute):
            self.insert_attribute(symbol_or_attr)
        else:
            self.insert_symbol(symbol_or_attr)


    def sorted_items(self):
        return sorted(self.items(), key=lambda x: x[1], reverse=True)


    def is_similar(self, other):
        """
        Determines if the two lists specify the same characters.
        Order and probabilities are not considered.
        """
        if isinstance(other, ProbabilityEnhancedAttribute):
            return self.symbols_specified() == other.symbols_specified()
        else:
            return self.symbols_specified() == {other}


    def __eq__(self, other):
        return self.is_similar(other)


    def __str__(self):
        if len(self) == 1:
            return str(next(iter(self)))
        return "{" + ", ".join( "{}:{:.2f}%".format(sym[0], sym[1] * 100) for sym in self.sorted_items()) + "}"