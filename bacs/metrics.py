"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""


def basic_metrics(trial: int, steps: int, reward: int):
    return {
        'trial': trial,
        'steps_in_trial': steps,
        'reward': reward
    }


def population_metrics(population, environment):
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


def _maze_metrics(pop, env):

    def _maze_knowledge(population, environment) -> float:
        transitions = environment.env.get_all_possible_transitions()
        # Take into consideration only reliable classifiers
        reliable_classifiers = [cl for cl in population if cl.is_reliable() and cl.does_anticipate_change()]
        # Count how many transitions are anticipated correctly
        nr_correct = 0
        # For all possible destinations from each path cell
        for start, action, end in transitions:
            p0 = environment.env.maze.perception(*start)
            p1 = environment.env.maze.perception(*end)
            if any([True for cl in reliable_classifiers
                    if cl.predicts_successfully(p0, action, p1)]):
                nr_correct += 1
        return nr_correct / len(transitions) * 100.0

    metrics = {
        'knowledge': _maze_knowledge(pop, env)
    }

    # Add basic population metrics
    metrics.update(population_metrics(pop, env))

    return metrics

def _does_pees_match_non_aliased_states(pop, env) -> int:
    counter = 0
    non_aliased_perceptions = env.env.get_all_non_aliased_states()
    enhanced_classifiers = [cl for cl in pop if cl.is_reliable() and cl.is_enhanced()]
    for percept in non_aliased_perceptions:
        for cl in enhanced_classifiers:
            if cl.does_match(percept):
                counter += 1
    return counter

def _mean_reliable_classifier_specificity(pop, env) -> int:
    reliable_classifiers = [cl for cl in pop if cl.is_reliable()]
    return float(sum(cl.specificity for cl in reliable_classifiers)) / len(reliable_classifiers)
