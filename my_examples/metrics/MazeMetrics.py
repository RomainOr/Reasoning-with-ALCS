"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import statistics


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


def _maze_metrics(
        pop,
        env
    ) -> dict:

    def _maze_knowledge(
            population,
            environment
        ) -> float:
        transitions = environment.env.get_all_possible_transitions()
        env_trans = []
        for start, action, end in transitions:
            p0 = environment.env.maze.perception(*start)
            p1 = environment.env.maze.perception(*end)
            env_trans.append((p0, action, p1))
        # Take into consideration only reliable classifiers
        reliable_classifiers = [cl for cl in population if cl.is_reliable() and cl.behavioral_sequence is None]
        # Count how many transitions are anticipated correctly
        nr_correct = 0
        # For all possible destinations from each path cell
        for p0, action, p1 in set(env_trans):
            if any(True for cl in reliable_classifiers
                    if cl.does_predict_successfully(p0, action, p1)):
                nr_correct += 1
        return nr_correct / len(set(env_trans)) * 100.0

    metrics = {
        'knowledge': _maze_knowledge(pop, env)
    }
    # Add basic population metrics
    metrics.update(population_metrics(pop, env))
    return metrics


def _how_many_eps_match_non_aliased_states(
        pop,
        env
    ) -> int:
    counter = 0
    non_aliased_perceptions = env.env.get_all_non_aliased_states()
    enhanced_classifiers = [cl for cl in pop if cl.is_reliable() and cl.is_enhanced()]
    for percept in non_aliased_perceptions:
        for cl in enhanced_classifiers:
            if cl.does_match(percept):
                counter += 1
    return counter


def _mean_reliable_classifier_specificity(
        pop,
        env
    ) -> int:
    mean_reliable_classifier_specificity = 1.
    mean_reliable_non_behavioral_classifier_specificity = 1.
    mean_reliable_behavioral_classifier_specificity = 1.
    reliable_classifiers = [cl for cl in pop if cl.is_reliable()]
    if len(reliable_classifiers) > 0:
        mean_reliable_classifier_specificity = float(sum(cl.specificity for cl in reliable_classifiers)) / len(reliable_classifiers)
        non_behavioral_cl = [cl for cl in reliable_classifiers if not cl.behavioral_sequence]
        if len(non_behavioral_cl) > 0:
            mean_reliable_non_behavioral_classifier_specificity = float(sum(cl.specificity for cl in non_behavioral_cl)) / len(non_behavioral_cl)
        behavioral_cl = [cl for cl in reliable_classifiers if cl.behavioral_sequence]
        if len(behavioral_cl) > 0:
            mean_reliable_behavioral_classifier_specificity = float(sum(cl.specificity for cl in behavioral_cl)) / len(behavioral_cl)
    return mean_reliable_classifier_specificity, mean_reliable_non_behavioral_classifier_specificity, mean_reliable_behavioral_classifier_specificity


def _when_full_knowledge_is_achieved(metrics) -> tuple:
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


def _enhanced_effect_error(
        population,
        environment,
        classifier_length,
        random_attribute_length
    ) -> float:
    theoritical_probabilities = environment.env.get_theoritical_probabilities()
    # Accumulation of difference in probabilities
    error_pep = 0.
    # For all possible destinations from each path cell
    for perception, action_and_probabiltiies in theoritical_probabilities.items():
        for action, probabilities_and_states in action_and_probabiltiies.items():
            # Try to find a suitable one, even if it is unreliable
            unreliable_classifiers = [cl for cl in population if cl.does_match(perception) and cl.action ==  action and cl.behavioral_sequence is None]
            if len(unreliable_classifiers) > 0:
                # Try to promote a reliable classifier
                reliable_classifiers = [cl for cl in unreliable_classifiers if cl.is_reliable()]
                if len(reliable_classifiers) > 0:
                    most_experienced_classifier = max(reliable_classifiers, key=lambda cl: cl.exp * pow(cl.q, 3))
                else:
                    most_experienced_classifier = max(unreliable_classifiers, key=lambda cl: cl.exp * pow(cl.q, 3))
            # If there are no matching classifiers, a none case as defaut case
            else:
                most_experienced_classifier = None
            prob = probabilities_and_states['probabilities']
            # If the system succeed to find a classifier, error is computed through the probabilities differences
            if most_experienced_classifier:
                for direction in prob:
                    # First, get effect attribute
                    effect_attribute = most_experienced_classifier.effect.getEffectAttribute(perception, direction)
                    theoritical_prob_of_attribute = prob[direction]
                    # Second error computation
                    for key in theoritical_prob_of_attribute:
                        error_pep += abs(theoritical_prob_of_attribute[key] - effect_attribute.get(key, 0.0))
                for ra in range(classifier_length-random_attribute_length, classifier_length):
                    # First, get effect attribute
                    effect_attribute = most_experienced_classifier.effect.getEffectAttribute(perception,ra)
                    # We consider here the probabilities are defined as 50% to get 1 and 50% to get 0. Could be automated and linked to environmental properties
                    theoritical_prob_of_attribute = {1:0.5, 0:0.5}
                    # Second error computation
                    for key in theoritical_prob_of_attribute:
                        error_pep += abs(theoritical_prob_of_attribute[key] - effect_attribute.get(key, 0.0))
            # None case as default case to increase error
            else:
                for direction in prob:
                    theoritical_prob_of_attribute = prob[direction]
                    for key in theoritical_prob_of_attribute:
                        error_pep += abs(theoritical_prob_of_attribute[key])
                for ra in range(classifier_length-random_attribute_length, classifier_length):
                    theoritical_prob_of_attribute = {1:0.5, 0:0.5}
                    for key in theoritical_prob_of_attribute:
                        error_pep += abs(theoritical_prob_of_attribute[key])
    return error_pep * 100 / (len(theoritical_probabilities)*8*classifier_length)

def compute_mean_and_stdev_for_one_env(env_name, results):
    
    knowledge_list = []
    population_list = []
    numerosity_list = []
    reliable_list = []
    mean_reliable_classifier_specificity_list = []
    mean_reliable_bs_classifier_specificity_list = []
    mean_reliable_no_bs_classifier_specificity_list = []
    ep_error_list = []
    eps_match_non_aliased_states_list = []
    
    full_knowledge_first_trial_list = []
    full_knowledge_stable_trial_list = []
    full_knowledge_last_trial_list = []
    
    avg_exploit_no_rl_list = []
    avg_exploit_rl_start_list = []
    avg_exploit_rl_list = []
    
    memory_of_pai_states_list = []
    
    explore_time_list = []
    cracs_time_list = []
    time_list = []

    for res in results:
        if res['maze'] == env_name:
            
            knowledge_list.append(res['knowledge'])
            population_list.append(res['population'])
            numerosity_list.append(res['numerosity'])
            reliable_list.append(res['reliable'])
            mean_reliable_classifier_specificity_list.append(res['mean_reliable_classifier_specificity'])
            mean_reliable_bs_classifier_specificity_list.append(res['mean_reliable_bs_classifier_specificity'])
            mean_reliable_no_bs_classifier_specificity_list.append(res['mean_reliable_no_bs_classifier_specificity'])
            ep_error_list.append(res['ep_error'])
            eps_match_non_aliased_states_list.append(res['eps_match_non_aliased_states'])
            
            full_knowledge_first_trial_list.append(res['full_knowledge_first_trial'])
            full_knowledge_stable_trial_list.append(res['full_knowledge_stable_trial'])
            full_knowledge_last_trial_list.append(res['full_knowledge_last_trial'])
            
            avg_exploit_no_rl_list.append(res['avg_exploit_no_rl'])
            avg_exploit_rl_start_list.append(res['avg_exploit_rl_start'])
            avg_exploit_rl_list.append(res['avg_exploit_rl'])
    
            explore_time_list.append(res['explore_time'])
            cracs_time_list.append(res['cracs_time'])
            time_list.append(res['time'])
            
            memory_of_pai_states_list.append(res['memory_of_pai_states'])
    
    memory_of_pai_states_dict = {}
    for pai_states_list in memory_of_pai_states_list:
        for pai_state in pai_states_list:
            pai_state = "".join(pai_state)
            if pai_state in memory_of_pai_states_dict:
                memory_of_pai_states_dict[pai_state] += 1
            else:
                memory_of_pai_states_dict[pai_state] = 1
    
    # Compute the means and standard deviations
    
    avg_knowledge = statistics.mean(knowledge_list)
    std_knowledge = statistics.stdev(knowledge_list)
    avg_population = statistics.mean(population_list)
    std_population = statistics.stdev(population_list)
    avg_numerosity = statistics.mean(numerosity_list)
    std_numerosity = statistics.stdev(numerosity_list)
    avg_reliable = statistics.mean(reliable_list)
    std_reliable = statistics.stdev(reliable_list)
    avg_mean_reliable_classifier_specificity = statistics.mean(mean_reliable_classifier_specificity_list)
    std_mean_reliable_classifier_specificity = statistics.stdev(mean_reliable_classifier_specificity_list)
    avg_mean_reliable_bs_classifier_specificity = statistics.mean(mean_reliable_bs_classifier_specificity_list)
    std_mean_reliable_bs_classifier_specificity = statistics.stdev(mean_reliable_bs_classifier_specificity_list)
    avg_mean_reliable_no_bs_classifier_specificity = statistics.mean(mean_reliable_no_bs_classifier_specificity_list)
    std_mean_reliable_no_bs_classifier_specificity = statistics.stdev(mean_reliable_no_bs_classifier_specificity_list)
    avg_ep_error_list = statistics.mean(ep_error_list)
    std_ep_error_list = statistics.stdev(ep_error_list)
    avg_eps_match_non_aliased_states_list = statistics.mean(eps_match_non_aliased_states_list)
    std_eps_match_non_aliased_states_list = statistics.stdev(eps_match_non_aliased_states_list)
    
    avg_full_knowledge_first_trial_list = statistics.mean(full_knowledge_first_trial_list)
    std_full_knowledge_first_trial_list = statistics.stdev(full_knowledge_first_trial_list)
    avg_full_knowledge_stable_trial_list = statistics.mean(full_knowledge_stable_trial_list)
    std_full_knowledge_stable_trial_list = statistics.stdev(full_knowledge_stable_trial_list)
    avg_full_knowledge_last_trial_list = statistics.mean(full_knowledge_last_trial_list)
    std_full_knowledge_last_trial_list = statistics.stdev(full_knowledge_last_trial_list)
    
    avg_exploit_no_rl = statistics.mean(avg_exploit_no_rl_list)
    std_exploit_no_rl = statistics.stdev(avg_exploit_no_rl_list)
    avg_exploit_rl_start = statistics.mean(avg_exploit_rl_start_list)
    std_exploit_rl_start = statistics.stdev(avg_exploit_rl_start_list)
    avg_exploit_rl = statistics.mean(avg_exploit_rl_list)
    std_exploit_rl = statistics.stdev(avg_exploit_rl_list)
    
    avg_explore_time = statistics.mean(explore_time_list)
    std_explore_time = statistics.stdev(explore_time_list)
    avg_cracs_time = statistics.mean(cracs_time_list)
    std_cracs_time = statistics.stdev(cracs_time_list)
    avg_time = statistics.mean(time_list)
    std_time = statistics.stdev(time_list)
    
    dic = {
        'maze'             : env_name,
        
        'avg_knowledge'    : avg_knowledge,
        'std_knowledge'    : std_knowledge,
        'avg_population'   : avg_population,
        'std_population'   : std_population,
        'avg_numerosity'   : avg_numerosity,
        'std_numerosity'   : std_numerosity,
        'avg_reliable'     : avg_reliable,
        'std_reliable'     : std_reliable,
        'avg_mean_reliable_classifier_specificity' : avg_mean_reliable_classifier_specificity,
        'std_mean_reliable_classifier_specificity' : std_mean_reliable_classifier_specificity,
        'avg_mean_reliable_bs_classifier_specificity' : avg_mean_reliable_bs_classifier_specificity,
        'std_mean_reliable_bs_classifier_specificity' : std_mean_reliable_bs_classifier_specificity,
        'avg_mean_reliable_no_bs_classifier_specificity' : avg_mean_reliable_no_bs_classifier_specificity,
        'std_mean_reliable_no_bs_classifier_specificity' : std_mean_reliable_no_bs_classifier_specificity,
        'avg_ep_error_list' : avg_ep_error_list,
        'std_ep_error_list' : std_ep_error_list,
        'avg_eps_match_non_aliased_states_list' : avg_eps_match_non_aliased_states_list,
        'std_eps_match_non_aliased_states_list' : std_eps_match_non_aliased_states_list,
        
        'avg_full_knowledge_first_trial_list'  : avg_full_knowledge_first_trial_list,
        'std_full_knowledge_first_trial_list'  : std_full_knowledge_first_trial_list,
        'avg_full_knowledge_stable_trial_list' : avg_full_knowledge_stable_trial_list,
        'std_full_knowledge_stable_trial_list' : std_full_knowledge_stable_trial_list,
        'avg_full_knowledge_last_trial_list'   : avg_full_knowledge_last_trial_list,
        'std_full_knowledge_last_trial_list'   : std_full_knowledge_last_trial_list,
        
        'avg_exploit_no_rl'   : avg_exploit_no_rl,
        'std_exploit_no_rl'   : std_exploit_no_rl,
        'avg_exploit_rl_start': avg_exploit_rl_start,
        'std_exploit_rl_start': std_exploit_rl_start,
        'avg_exploit_rl'      : avg_exploit_rl,
        'std_exploit_rl'      : std_exploit_rl,
    
        'avg_explore_time' : avg_explore_time,
        'std_explore_time' : std_explore_time,
        'avg_cracs_time' : avg_cracs_time,
        'std_cracs_time' : std_cracs_time,
        'avg_time'     : avg_time,
        'std_time'     : std_time,
        
        'knowledge_list'  : knowledge_list,
        'population_list' : population_list,
        'numerosity_list' : numerosity_list,
        'reliable_list'   : reliable_list,
        'mean_reliable_classifier_specificity_list' : mean_reliable_classifier_specificity_list,
        'mean_reliable_bs_classifier_specificity_list' : mean_reliable_bs_classifier_specificity_list,
        'mean_reliable_no_bs_classifier_specificity_list' : mean_reliable_no_bs_classifier_specificity_list,
        'ep_error_list' : ep_error_list,
        'eps_match_non_aliased_states_list' : eps_match_non_aliased_states_list,
        
        'full_knowledge_first_trial_list'  : full_knowledge_first_trial_list,
        'full_knowledge_stable_trial_list' : full_knowledge_stable_trial_list,
        'full_knowledge_last_trial_list'   : full_knowledge_last_trial_list,
        
        'avg_exploit_no_rl_list'    : avg_exploit_no_rl_list,
        'avg_exploit_rl_start_list' : avg_exploit_rl_start_list,
        'avg_exploit_rl_list'       : avg_exploit_rl_list,
        
        'explore_time_list'  : explore_time_list,
        'cracs_time_list' : cracs_time_list,
        'time_list'   : time_list,
        
        'memory_of_pai_states_dict' : memory_of_pai_states_dict
    }
    
    return dic