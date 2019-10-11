from typing import List

from bacs import Perception
from bacs.agents.bacs import ClassifiersList
from bacs.agents.bacs.components.goal_sequence_searcher_bacs \
    import GoalSequenceSearcher


def suitable_cl_exists(classifiers: ClassifiersList,
                       p0: Perception,
                       action: int,
                       p1: Perception) -> bool:
    """

    Parameters
    ----------
    classifiers
    p0: Perception
        previous situation
    action: int
    p1: Perception
        situation

    Returns
    -------
    bool
        True if there is a reliable classifier in this list
        that matches previous_situation, specifies action,
        and predicts situation.
        False otherwise.
    """
    # TODO updatewith behavioral sequences
    def _ok(cl):
        return cl.is_reliable() \
            and cl.does_match(p0) \
            and cl.action == action \
            and cl.does_anticipate_correctly(p0, p1)

    return any([cl for cl in classifiers if _ok(cl)])


def search_goal_sequence(classifiers: ClassifiersList,
                         p0: Perception,
                         p1: Perception) -> List:
    """
    Searches a path from start to goal using a bidirectional method in the
    environmental model (i.e. the list of reliable classifiers).

    Parameters
    ----------
    classifiers: ClassifiersList
    p0: Perception
        start state
    p1: Perception
        destination state

    Returns
    -------
    list
        sequence of actions
    """
    reliable = [cl for cl in classifiers if cl.is_reliable()]
    gs = GoalSequenceSearcher()

    return gs.search_goal_sequence(ClassifiersList(*reliable), p0, p1)
