class EnvironmentAdapter:
    """
    Sometimes the observation returned by the OpenAI Gym environment
    does not suit the observation representation as used by the LCS.
    In that case there's a need for converter functions that map the LCS
    representation to the environment representation and vice versa.

    The same goes for action representation.

    This class realizes this conversion.

    This implementation works for environments which provide LCS-compatible
    state and action representations (no conversion needed).

    Subclass this class to provide an adapter for a specific environment.
    """

    @staticmethod
    def to_lcs_action(env, env_action):
        """
        Converts environment representation of an action to LCS
        representation.
        """
        return env_action

    @staticmethod
    def to_genotype(env, phenotype):
        """
        Converts environment representation of a state to LCS
        representation.
        """
        return phenotype

    @staticmethod
    def to_env_action(env, lcs_action):
        """
        Converts LCS representation of an action to environment
        representation.
        """
        return lcs_action

    @staticmethod
    def to_phenotype(env, genotype):
        """
        Converts LCS representation of a state to environment
        representation.
        """
        return genotype
