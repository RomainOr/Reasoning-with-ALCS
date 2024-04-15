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


def _mountaincar_metrics(
        pop,
        env
    ) -> dict:
    metrics = {}
    # Add basic population metrics
    metrics.update(population_metrics(pop, env))
    return metrics


def get_score_exploit(
        metrics_exploit, 
        NUMBER_OF_EXPLOIT_TRIALS
    ):
    trials=[]

    avg_step_exploit = 0
    for trial in metrics_exploit:
        trials.append(trial['reward'])
        avg_step_exploit += trial['reward']
    avg_step_exploit /= NUMBER_OF_EXPLOIT_TRIALS
    
    avg_step_exploit_last_100 = 0
    for trial in metrics_exploit[-100:]:
        trials.append(trial['reward'])
        avg_step_exploit_last_100 += trial['reward']
    avg_step_exploit_last_100 /= 100

    # https://github.com/openai/gym/wiki/Leaderboard#mountaincar-v0
    # MountainCar-v0 defines "solving" as getting average reward of -110.0 over 100 consecutive trials.
    average_scores=[]
    solved = False
    solved_averaged = 0.
    for i in range(len(trials)-99):
        check_solved = trials[i:i+100]
        average = float(sum(check_solved) / 100)
        average_scores.append(average)
        if average >= -110.0 and not solved:
            solved = True

    return avg_step_exploit, avg_step_exploit_last_100, max(average_scores), solved, average_scores