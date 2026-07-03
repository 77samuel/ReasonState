"""
statistics.py — Validated statistical functions for Paper #13 / ReasonState.

blocked_permanova() implements a restricted-permutation PERMANOVA test that
shuffles group labels only WITHIN blocks (e.g. environment), preserving
block structure while testing for an association between a multivariate
distance matrix and a grouping variable.

This function must be validated (see validate_blocked_permanova.py) before
use in any confirmatory analysis (Step 5A). Do not modify this file after
Step 5A has been run without a formal governance reopening.
"""

import numpy as np
from skbio.stats.distance import permanova


def blocked_permanova(dm, groups, blocks, permutations=9999, seed=42):
    """Blocked-permutation PERMANOVA.

    Tests association between a distance matrix `dm` and a categorical
    `groups` variable, restricting the permutation null distribution to
    shuffles that occur only WITHIN each level of `blocks` (e.g. only
    within-environment shuffles of module labels), never across blocks.

    This is the "restricted permutation" design frozen in Blueprint v2,
    Step 5A: it tests whether the group association holds beyond what
    could be explained by the block structure alone, while using the
    block structure itself (rather than removing it) to build a valid
    null distribution.

    Parameters
    ----------
    dm : skbio.DistanceMatrix
    groups : array-like, the grouping variable to test (e.g. module)
    blocks : array-like, the blocking factor restricting permutations
             (e.g. environment)
    permutations : int, number of permutations for the null distribution
    seed : int, random seed (recorded for reproducibility)

    Performance note: this implementation calls permanova(..., permutations=0)
    once per permutation rather than computing the pseudo-F directly from
    sums of squares. This was checked against the actual dataset scale
    (N~182, 6 predictors): 9999 permutations completes in ~3.4 seconds,
    which is not a practical bottleneck. A hand-rolled direct pseudo-F
    computation was considered but rejected, since it would introduce a
    second unvalidated implementation of the same statistic in the name of
    an optimization with no measurable benefit at this dataset's scale.

    Returns
    -------
    observed_stat : float, the observed pseudo-F statistic
    p_value : float, blocked-permutation p-value
    perm_distribution : np.ndarray, the permutation null distribution
    """
    rng = np.random.default_rng(seed)
    groups = np.array(groups)
    blocks = np.array(blocks)

    obs = permanova(dm, groups, permutations=0)
    observed_stat = obs["test statistic"]

    perm_stats = np.zeros(permutations)
    unique_blocks = np.unique(blocks)
    for p in range(permutations):
        permuted = groups.copy()
        for block in unique_blocks:
            mask = blocks == block
            idx = np.where(mask)[0]
            permuted[idx] = rng.permutation(groups[mask])
        res = permanova(dm, permuted, permutations=0)
        perm_stats[p] = res["test statistic"]

    p_value = (np.sum(perm_stats >= observed_stat) + 1) / (permutations + 1)
    return observed_stat, p_value, perm_stats
