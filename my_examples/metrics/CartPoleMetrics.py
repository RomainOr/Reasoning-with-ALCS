"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""


def population_metrics(
        population,
        environment
    ):
    metrics = {
        'population': 0,
        'numerosity': 0,
        'reliable': 0,
    }
    for cl in population:
        metrics['population'] += 1
        metrics['numerosity'] += cl.num
        if cl.is_reliable():
            metrics['reliable'] += 1
    return metrics


def _cartpole_metrics(
        pop,
        env
    ) -> dict:
    metrics = {}
    # Add basic population metrics
    metrics.update(population_metrics(pop, env))
    return metrics


def _mean_reliable_classifier_specificity(
        pop,
        env
    ) -> int:
    # TODO : Return values for legacy cl, behavioral cl, enhanced cl, behavioral enhanced cl and all ?
    reliable_classifiers = [cl for cl in pop if cl.is_reliable()]
    if len(reliable_classifiers) > 0:
        return float(sum(cl.specificity for cl in reliable_classifiers)) / len(reliable_classifiers)
    else:
        return 1.

def _check_cartpole_solved_requirement(trials):
    average_scores=[]
    solved = -1
    for i in range(len(trials)-100):
        check_solved = trials[i:i+100]
        average = float(sum(check_solved) / 100)
        average_scores.append(average)
        if average >= 475.0 and solved == -1:
            solved = 100+1+i
    return average_scores, solved