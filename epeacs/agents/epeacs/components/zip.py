"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""


from epeacs.agents.epeacs.components.subsumption import does_subsume


def should_apply(
        pop_set, 
        time: int, 
        theta_zip: int
    ) -> bool:
    """
    Checks the average last ZIP application to determine if zipping
    should be applied.
    If no classifier is in the current set, no zipping is applied!

    Parameters
    ----------
    pop_set
        population of classifiers (with `num` and `tzip` properties)
    time: int
        current epoch
    theta_zip: int
        The ZIP application threshold (θzip ∈ N) controls the ZIP frequency. A ZIP
        is applied in an population set if the average delay of the last ZIP
        application of the classifiers in the set is greater than θzip.

    Returns
    -------
    bool
        True if zipping should be applied, False otherwise
    """
    if pop_set is None:
        return False
    
    if len(pop_set) < 2:
        return False

    overall_time = sum(cl.tzip * cl.num for cl in pop_set)
    overall_num = sum(cl.num for cl in pop_set)

    if overall_num == 0:
        return False

    if time - (overall_time / overall_num) > theta_zip:
        return True

    return False


def set_timestamps(pop_set, epoch: int) -> None:
    """
    Sets the GA time stamps to the current time to control
    the GA application frequency.
    Each classifier `tga` property in population is updated with current
    epoch

    Parameters
    ----------
    pop_set
        population of classifiers
    epoch: int
        current epoch
    """
    for cl in pop_set:
        cl.tzip = epoch

def zip_set(pop_set, theta_exp):
    to_delete = []
    for idx1 in range(len(pop_set)-1):
        cl = pop_set[idx1]
        if cl in to_delete:
            continue
        for idx2 in range(idx1+1, len(pop_set)):
            other_cl = pop_set[idx2]
            if other_cl in to_delete:
                continue
            if cl.condition.subsumes(other_cl.condition) and does_subsume(cl, other_cl, theta_exp):
                to_delete.append(other_cl)
    for cl_to_delete in to_delete:
        pop_set.safe_remove(cl_to_delete)