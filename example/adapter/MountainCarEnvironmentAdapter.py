import numpy as np

from beacs.agents.EnvironmentAdapter import EnvironmentAdapter


buckets=(12, 20)

class MountainCarEnvironmentAdapter(EnvironmentAdapter):

    pos_space = np.linspace(-1.2, 0.6, buckets[0])
    vel_space = np.linspace(-0.07, 0.07, buckets[1])

    @staticmethod
    def to_genotype(env, phenotype):
        """
        Converts environment representation of a state to LCS
        representation.
        """
        def _discretize(env, obs):
            pos, vel =  obs
            pos_bin = int(np.digitize(pos, MountainCarEnvironmentAdapter.pos_space))
            vel_bin = int(np.digitize(vel, MountainCarEnvironmentAdapter.vel_space))
            return (str(pos_bin), str(vel_bin))
            
        return _discretize(env, phenotype)