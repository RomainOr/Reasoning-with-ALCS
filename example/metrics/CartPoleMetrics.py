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