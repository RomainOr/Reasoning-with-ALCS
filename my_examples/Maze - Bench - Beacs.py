#
# 
#     This Source Code Form is subject to the terms of the Mozilla Public
#     License, v. 2.0. If a copy of the MPL was not distributed with this
#     file, You can obtain one at http://mozilla.org/MPL/2.0/.
# 

import os
import sys

os.environ["RAY_DEDUP_LOGS"] = "0"

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)    

import gymnasium as gym
import gymnasium_mazes

import json
from my_examples.metrics.MazeMetrics import compute_mean_and_stdev_for_one_env

import ray

#Environmental Set Up
CLASSIFIER_LENGTH = 8
NUMBER_OF_POSSIBLE_ACTIONS = 8
SLIPPERY_PROB = 0.25

#Exploration Set Up
NUMBER_OF_EXPLORE_TRIALS = 5000
METRICS_TRIAL_FREQUENCY_EXPLORE = 100
EPSILON = 0.8
BETA_ALP = 0.05

#Exploitation Set Up
NUMBER_OF_EXPLOIT_TRIALS_NO_RL = 500
BETA_EXPLOIT_NO_RL = 0.05
NUMBER_OF_EXPLOIT_TRIALS_RL_START = 500
BETA_EXPLOIT_RL_START = 0.05
NUMBER_OF_EXPLOIT_TRIALS_RL = 500
BETA_EXPLOIT_RL = 0.05

#RL Set Up
GAMMA = 0.95
BETA_RL = 0.05

#GA Set Up
CROSSOVER = 0.8
MUTATION = 0.3

#BEACS Set Up
ENABLE_EP = True
LENGTH_OF_BEHAVIORAL_SEQUENCES = 2

#Parallelization and Iterations for Stats
NUMBER_OF_ITERATIONS_TO_BENCH = 10
JSON_RESULTS_FILENAME = "test.json"


# Function to get benchmark value on one gym environment :

runtime_env= {"working_dir": "..", 'excludes': ['.git']}
ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

@ray.remote
def bench_on_maze(env):

    import time
    from agents.beacs import BEACS, BEACSConfiguration
    from my_examples.metrics.MazeMetrics import \
        _maze_metrics, \
        _how_many_eps_match_non_aliased_states, \
        _mean_reliable_classifier_specificity, \
        _when_full_knowledge_is_achieved, \
        _enhanced_effect_error
    import gymnasium as gym
    import gymnasium_mazes
    
    cfg_explore = BEACSConfiguration(
        classifier_length=CLASSIFIER_LENGTH,
        number_of_possible_actions=NUMBER_OF_POSSIBLE_ACTIONS,
        user_metrics_collector_fcn=_maze_metrics,
        metrics_trial_frequency=METRICS_TRIAL_FREQUENCY_EXPLORE,
        do_ep=ENABLE_EP,
        beta_alp=BETA_ALP,
        beta_rl=BETA_RL,
        gamma=GAMMA,
        epsilon=EPSILON,
        u_max=CLASSIFIER_LENGTH,
        mu=MUTATION,
        chi=CROSSOVER,
        bs_max=LENGTH_OF_BEHAVIORAL_SEQUENCES
    )

    cfg_exploit_no_rl = BEACSConfiguration(
        classifier_length=CLASSIFIER_LENGTH,
        number_of_possible_actions=NUMBER_OF_POSSIBLE_ACTIONS,
        user_metrics_collector_fcn=_maze_metrics,
        metrics_trial_frequency=1,
        beta_rl=BETA_EXPLOIT_NO_RL,
        gamma=GAMMA,
        epsilon=0.2
    )

    cfg_exploit_rl_start = BEACSConfiguration(
        classifier_length=CLASSIFIER_LENGTH,
        number_of_possible_actions=NUMBER_OF_POSSIBLE_ACTIONS,
        user_metrics_collector_fcn=_maze_metrics,
        metrics_trial_frequency=1,
        beta_rl=BETA_EXPLOIT_RL_START,
        gamma=GAMMA,
        epsilon=0.0
    )

    cfg_exploit_rl = BEACSConfiguration(
        classifier_length=CLASSIFIER_LENGTH,
        number_of_possible_actions=NUMBER_OF_POSSIBLE_ACTIONS,
        user_metrics_collector_fcn=_maze_metrics,
        metrics_trial_frequency=1,
        beta_rl=BETA_EXPLOIT_RL,
        gamma=GAMMA,
        epsilon=0.0,
    )
        
    # Initialize environment
    maze = gym.make(env, slippery_prob=SLIPPERY_PROB)

    # Reset it, by putting an agent into random position
    maze.reset()

    # Training of BEACS - Exploration
    explore_start_time = time.process_time()
    agent_explore = BEACS(cfg_explore)
    population_explore, metrics_explore = agent_explore.explore(maze, NUMBER_OF_EXPLORE_TRIALS)
    explore_end_time = time.process_time()
    
    # Applying CRACS
    cracs_start_time = time.process_time()
    agent_explore.apply_CRACS()
    cracs_end_time = time.process_time()
    population_explore = agent_explore.get_population()
    
    eps_match_non_aliased_states = _how_many_eps_match_non_aliased_states(population_explore, maze)
    ep_error = _enhanced_effect_error(population_explore, maze, CLASSIFIER_LENGTH)
    mean_reliable_classifier_specificity, mean_reliable_no_bs_classifier_specificity, mean_reliable_bs_classifier_specificity = _mean_reliable_classifier_specificity(population_explore, maze)
    maze_metrics = _maze_metrics(population_explore, maze)
    
    first_trial, stable_trial, last_trial = _when_full_knowledge_is_achieved(metrics_explore)

    ### Using BEACS - Compressed population
    
    start_time = time.process_time()
    
    # Using BEACS - Exploitation - No RL module
    agent_exploit_no_rl = BEACS(cfg_exploit_no_rl, population_explore)
    population_exploit_no_rl, metrics_exploit_no_rl = agent_exploit_no_rl.exploit(maze, NUMBER_OF_EXPLOIT_TRIALS_NO_RL)

    # Using BEACS - Exploitation - Starting using RL module
    agent_exploit_rl_start = BEACS(cfg_exploit_rl_start, population_exploit_no_rl)
    population_exploit_rl_start, metrics_exploit_rl_start = agent_exploit_rl_start.exploit(maze, NUMBER_OF_EXPLOIT_TRIALS_RL_START)

    # Using BEACS - Exploitation - Using RL module
    agent_exploit_rl = BEACS(cfg_exploit_rl, population_exploit_rl_start)
    population_exploit_rl, metrics_exploit_rl = agent_exploit_rl.exploit(maze, NUMBER_OF_EXPLOIT_TRIALS_RL)

    end_time = time.process_time()
    
    # Get average 'steps to exit' in all exploitation modes
    avg_step_exploit_no_rl = 0
    for trial in metrics_exploit_no_rl:
        avg_step_exploit_no_rl += trial['steps_in_trial']
    avg_step_exploit_no_rl /= NUMBER_OF_EXPLOIT_TRIALS_NO_RL
    avg_step_exploit_rl_start = 0
    for trial in metrics_exploit_rl_start:
        avg_step_exploit_rl_start += trial['steps_in_trial']
    avg_step_exploit_rl_start /= NUMBER_OF_EXPLOIT_TRIALS_RL_START
    avg_step_exploit_rl = 0
    for trial in metrics_exploit_rl:
        avg_step_exploit_rl += trial['steps_in_trial']
    avg_step_exploit_rl /= NUMBER_OF_EXPLOIT_TRIALS_RL
    
    result = {
        'maze' : env,
        
        'knowledge' : maze_metrics['knowledge'],
        'population' : maze_metrics['population'],
        'numerosity' : maze_metrics['numerosity'],
        'reliable' : maze_metrics['reliable'],
        'mean_reliable_classifier_specificity' : mean_reliable_classifier_specificity,
        'mean_reliable_bs_classifier_specificity' : mean_reliable_bs_classifier_specificity,
        'mean_reliable_no_bs_classifier_specificity' : mean_reliable_no_bs_classifier_specificity,
        'ep_error': ep_error,
        'eps_match_non_aliased_states': eps_match_non_aliased_states,
        
        'full_knowledge_first_trial' : first_trial,
        'full_knowledge_stable_trial' : stable_trial,
        'full_knowledge_last_trial' : last_trial,
        
        'avg_exploit_no_rl' : avg_step_exploit_no_rl,
        'avg_exploit_rl_start' : avg_step_exploit_rl_start,
        'avg_exploit_rl' : avg_step_exploit_rl,
        
        'memory_of_pai_states' : agent_explore.get_pai_states_memory(), 
        
        'explore_time' : explore_end_time - explore_start_time,
        'cracs_time' : cracs_end_time - cracs_start_time,
        'time' : (end_time - start_time)
    }
    
    print(result)
    
    return result


filter_envs_typeIII = lambda env: "Maze10-" in env or "MazeE1" in env \
    or "MazeE2" in env or "Woods10" in env

filter_envs_typeII = lambda env: "MazeF4" in env or "Maze7" in env \
    or "MiyazakiB" in env

filter_envs_typeI = lambda env: "MazeB" in env or "MazeD" in env \
    or "Littman" in env or "MiyazakiA" in env \
    or "Cassandra" in env

filter_envs_na = lambda env: "MazeF1" in env or "MazeF2" in env \
    or "MazeF3" in env or "Woods14" in env \
    or "Maze4" in env or "Maze5" in env \
    or "MazeA" in env

all_envs = [env for env in gym.envs.registry]

# Set up the list of environments to bench : 
maze_envs = []
maze_envs_name = []
for env in all_envs:
    if "MazeF3" in env:
    #if filter_envs_typeIII(env) or filter_envs_typeII(env) or filter_envs_typeI(env) or filter_envs_na(env):
        maze_envs_name.append(env)
        for i in range(NUMBER_OF_ITERATIONS_TO_BENCH):
            maze_envs.append(env)



futures = [bench_on_maze.remote(env) for env in maze_envs]
results = ray.get(futures)
results = [compute_mean_and_stdev_for_one_env(env_name, results) for env_name in maze_envs_name]

jsonString = json.dumps(results)
jsonFile = open(JSON_RESULTS_FILENAME, "w")
jsonFile.write(jsonString)
jsonFile.close()
