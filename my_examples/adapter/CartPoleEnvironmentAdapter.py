import math

from agents.common.EnvironmentAdapter import EnvironmentAdapter

#Could fail but should be sufficient with 1, 1, 4, 3 buckets.
#Best solution achieved with population that fully converged towards 12 classifiers with EP.
#Otherwise, use 1, 1, 6, 3.

class CartPoleEnvironmentAdapter(EnvironmentAdapter):

    def __init__(
            self, 
            buckets=(1, 1, 6, 3,)
        ) -> None:
        super().__init__()
        self.buckets = buckets

    def to_genotype(self, env, phenotype):
        """
        Converts environment representation of a state to LCS
        representation.
        """
        def _discretize(env, obs):
            upper_bounds = [env.env.observation_space.high[0], 0.5, env.env.observation_space.high[2], math.radians(50)]
            lower_bounds = [env.env.observation_space.low[0], -0.5, env.env.observation_space.low[2], -math.radians(50)]
            ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
            new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
            new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
            new_obs = [str(new_obs[i]) for i in range(len(obs))]
            return tuple(new_obs)
        return _discretize(env, phenotype)