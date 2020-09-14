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


def _how_many_peps_match_non_aliased_states(pop, env) -> int:
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
    if len(reliable_classifiers) > 0:
        return float(sum(cl.specificity for cl in reliable_classifiers)) / len(reliable_classifiers)
    else:
        return 1.


def _when_full_knowledge_is_achieved(metrics):
    first_trial_when_full_knowledge = -1
    stable_trial_when_full_knowledge = -1
    last_trial_when_full_knowledge = -1
    for trial in metrics:
        if first_trial_when_full_knowledge == -1 and trial['knowledge'] == 100:
            first_trial_when_full_knowledge = trial['trial']
        if stable_trial_when_full_knowledge == -1 and trial['knowledge'] == 100:
            stable_trial_when_full_knowledge = trial['trial']
        if stable_trial_when_full_knowledge != -1 and trial['knowledge'] != 100:
            stable_trial_when_full_knowledge = -1
        if trial['knowledge'] == 100:
            last_trial_when_full_knowledge = trial['trial']
    return first_trial_when_full_knowledge, stable_trial_when_full_knowledge, last_trial_when_full_knowledge


def _state_of_population(metrics, trial, step):
    return metrics[trial//step - 1]


def _enhanced_effect_error(population, environment, classifier_length, random_attribute_length) -> float:
    theoritical_probabilities = environment.env.get_theoritical_probabilities()
    # Take into consideration only reliable classifiers
    reliable_classifiers = [cl for cl in population if cl.is_reliable() and cl.behavioral_sequence is None]
    # Accumulation of difference in probabilities
    error_old_pep = 0.
    error_new_pep = 0.
    # For all possible destinations from each path cell
    for perception, action_and_probabiltiies in theoritical_probabilities.items():
        for action, probabilities_and_states in action_and_probabiltiies.items():
            classifiers = [cl for cl in reliable_classifiers if cl.condition.does_match(perception) and cl.action ==  action]
            if len(classifiers) > 0:
                most_experienced_classifier = max(classifiers, key=lambda cl: cl.exp * cl.num)
            else:
                most_experienced_classifier = max([cl for cl in population if cl.condition.does_match(perception) and cl.action ==  action and cl.behavioral_sequence is None], key=lambda cl: cl.exp * cl.num)
            prob = probabilities_and_states['probabilities']
            for direction in prob:
                old_effect_attribute, new_effect_attribute = most_experienced_classifier.effect.getEffectAttribute(direction)
                theoritical_prob_of_attribute = prob[direction]
                if old_effect_attribute == '#':
                    if most_experienced_classifier.condition[direction] == '#':
                        old_effect_attribute = {int(perception[direction]):1.0}
                        new_effect_attribute = {int(perception[direction]):1.0}
                    else:
                        old_effect_attribute = {int(most_experienced_classifier.condition[direction]):1.0}
                        new_effect_attribute = {int(most_experienced_classifier.condition[direction]):1.0}
                for key in theoritical_prob_of_attribute:
                    error_old_pep += abs(theoritical_prob_of_attribute[key] - old_effect_attribute.get(key, 0.0))
                    error_new_pep += abs(theoritical_prob_of_attribute[key] - new_effect_attribute.get(key, 0.0))
            for ra in range(classifier_length-random_attribute_length, classifier_length):
                old_effect_attribute, new_effect_attribute = most_experienced_classifier.effect.getEffectAttribute(ra)
                theoritical_prob_of_attribute = {1:0.5, 0:0.5}
                if old_effect_attribute == '#':
                    if most_experienced_classifier.condition[ra] == '#':
                        old_effect_attribute = {0:1.0, 1:1.0}
                        new_effect_attribute = {0:1.0, 1:1.0}
                    else:
                        old_effect_attribute = {int(most_experienced_classifier.condition[ra]):1.0}
                        new_effect_attribute = {int(most_experienced_classifier.condition[ra]):1.0}
                for key in theoritical_prob_of_attribute:
                    error_old_pep += abs(theoritical_prob_of_attribute[key] - old_effect_attribute.get(key, 0.0))
                    error_new_pep += abs(theoritical_prob_of_attribute[key] - new_effect_attribute.get(key, 0.0))
    return error_old_pep * 100 / (len(theoritical_probabilities)*8*classifier_length), error_new_pep * 100 / (len(theoritical_probabilities)*8*classifier_length)