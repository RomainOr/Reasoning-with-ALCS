"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from agents.common.RandomNumberGenerator import RandomNumberGenerator


def mutation_enhanced_trace(
        cl1,
        cl2,
        mu: float
    ) -> None:
    """
    Executes a particular mutation depending on the classifiers

    Parameters
    ----------
    cl1
        First classifier
    cl2
        Second classifier
    mu
        Mutation rate
    """
    for idx in range(len(cl1.condition)):
        #
        if cl1.condition[idx] == cl1.cfg.classifier_wildcard and \
            cl2.condition[idx] == cl2.cfg.classifier_wildcard:
            continue
        #
        if cl1.condition[idx] != cl1.cfg.classifier_wildcard and \
            cl2.condition[idx] == cl2.cfg.classifier_wildcard:
            if RandomNumberGenerator.random() < mu and cl1.effect.enhanced_trace_ga[idx]:
                cl1.condition.generalize(idx)
            continue
        #
        if cl1.condition[idx] == cl1.cfg.classifier_wildcard and \
            cl2.condition[idx] != cl2.cfg.classifier_wildcard:
            if RandomNumberGenerator.random() < mu and cl2.effect.enhanced_trace_ga[idx]:
                cl2.condition.generalize(idx)
            continue
        #
        if cl1.condition[idx] != cl1.cfg.classifier_wildcard and \
            cl1.behavioral_sequence is None and cl1.effect.enhanced_trace_ga[idx] and \
                RandomNumberGenerator.random() < mu:
            cl1.condition.generalize(idx)
        if cl2.condition[idx] != cl2.cfg.classifier_wildcard and \
            cl2.behavioral_sequence is None and cl2.effect.enhanced_trace_ga[idx] and \
                RandomNumberGenerator.random() < mu:
            cl2.condition.generalize(idx)