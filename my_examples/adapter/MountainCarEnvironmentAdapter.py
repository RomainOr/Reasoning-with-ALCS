import numpy as np

from agents.common.EnvironmentAdapter import EnvironmentAdapter


class MountainCarEnvironmentAdapter(EnvironmentAdapter):

    def __init__(
            self, 
            pos_bucket = 5, 
            vel_bucket = 4
        ) -> None:
        super().__init__()
        self.pos_space = np.linspace(-1.2, 0.6, num=pos_bucket, endpoint=False)
        self.vel_space = np.linspace(-0.07, 0.07, num=vel_bucket, endpoint=False)

    def to_genotype(self, env, phenotype):
        """
        Converts environment representation of a state to LCS
        representation.
        """
        def _discretize(env, obs):
            pos, vel =  obs
            pos_bin = int(np.digitize(pos, self.pos_space, right = False))
            if pos_bin == 0:
                pos_bin = 1
            vel_bin = int(np.digitize(vel, self.vel_space, right = False))
            return (str(pos_bin), str(vel_bin))
            
        return _discretize(env, phenotype)