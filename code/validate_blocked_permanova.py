"""
validate_blocked_permanova.py — Three-test validation suite for the custom
blocked_permanova() function before it is used in Step 5A.

Test 1: Type I error under a true null (random predictors, random labels)
        should be ~5% at alpha=0.05.
Test 2: Power under a strong artificial signal should be very high
        (p << 0.05 reliably).
Test 3: Behavior under a known blocked structure -- compare unrestricted
        vs blocked permutation p-values when the grouping variable is
        partially confounded with the blocking factor, to confirm blocking
        behaves as expected (more conservative / different null than
        unrestricted permutation in the presence of block-group
        association).
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix, permanova

from statistics import blocked_permanova

np.random.seed(0)


def make_distance_matrix(X):
    dist = squareform(pdist(X, metric="euclidean"))
    ids = [f"t{i}" for i in range(len(X))]
    return DistanceMatrix(dist, ids=ids)


# ---------------------------------------------------------------------------
# TEST 1: Type I error rate under the null (no true signal)
# ---------------------------------------------------------------------------
def test1_type_i_error(n_trials=200, n_obs=150, n_groups=6, n_blocks=3,
                        n_predictors=6, permutations=999):
    """Repeat many times: random predictors, random group labels (no true
    association), random block assignment. Across n_trials runs, the
    proportion of p < 0.05 results should be close to 0.05."""
    rejections = 0
    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        X = rng.normal(size=(n_obs, n_predictors))
        groups = rng.choice([f"g{i}" for i in range(n_groups)], size=n_obs)
        blocks = rng.choice([f"b{i}" for i in range(n_blocks)], size=n_obs)
        dm = make_distance_matrix(X)
        _, p, _ = blocked_permanova(dm, groups, blocks,
                                     permutations=permutations, seed=trial)
        if p < 0.05:
            rejections += 1
    rate = rejections / n_trials
    return rate


# ---------------------------------------------------------------------------
# TEST 2: Power under a strong artificial signal
# ---------------------------------------------------------------------------
def test2_power(n_obs=150, n_groups=6, n_blocks=3, n_predictors=6,
                 permutations=999, effect_size=3.0, seed=42):
    """Inject a strong group effect: each group gets a distinct mean offset
    in predictor space. The blocked test should detect this with very
    small p, despite blocks being randomly assigned (no block confound)."""
    rng = np.random.default_rng(seed)
    group_labels = [f"g{i}" for i in range(n_groups)]
    groups = rng.choice(group_labels, size=n_obs)
    blocks = rng.choice([f"b{i}" for i in range(n_blocks)], size=n_obs)

    group_offsets = {g: rng.normal(scale=effect_size, size=n_predictors)
                      for g in group_labels}
    X = np.array([rng.normal(size=n_predictors) + group_offsets[g]
                  for g in groups])

    dm = make_distance_matrix(X)
    observed_stat, p, _ = blocked_permanova(dm, groups, blocks,
                                             permutations=permutations, seed=seed)
    return observed_stat, p


# ---------------------------------------------------------------------------
# TEST 3: Blocked vs unrestricted permutation under a block-group confound
# ---------------------------------------------------------------------------
def test3_blocking_behavior(n_obs=150, n_groups=4, n_blocks=3,
                             n_predictors=6, permutations=999, seed=42):
    """Construct a scenario where the blocking factor is itself associated
    with the predictors (a "block effect"), but the GROUP variable has NO
    true effect beyond what block explains. An unrestricted permutation
    test (ignoring blocks) risks falsely detecting a group effect, because
    shuffling freely can accidentally preserve some block-correlated
    structure by chance, or more importantly, the raw distances already
    contain block-driven variance that unrestricted permutation doesn't
    isolate. The blocked test, which only permutes within blocks, should
    not be fooled by the block effect."""
    rng = np.random.default_rng(seed)

    blocks = np.array([f"b{i}" for i in range(n_blocks) for _ in range(n_obs // n_blocks)])
    rng.shuffle(blocks)

    # Block effect: each block has a distinct mean offset (this is the
    # confound -- e.g. like environment affecting predictor scale).
    block_offsets = {b: rng.normal(scale=4.0, size=n_predictors) for b in sorted(set(blocks))}

    # Group assigned WITHOUT regard to block (no true group signal),
    # but because blocks differ in size/composition this can create
    # spurious unrestricted-permutation associations if blocking isn't honored.
    group_labels = [f"g{i}" for i in range(n_groups)]
    # Deliberately make group correlated with block to stress-test:
    # each block disproportionately contains certain groups (~70% one group).
    # Use a deterministic, seed-controlled block->group assignment (NOT
    # Python's built-in hash(), which is randomized per-process via
    # PYTHONHASHSEED and would silently make this test non-reproducible
    # across runs -- this was an actual bug found during validation).
    unique_blocks_list = sorted(set(blocks))
    block_to_preferred_group = {
        b: group_labels[i % n_groups] for i, b in enumerate(unique_blocks_list)
    }
    groups = []
    for b in blocks:
        if rng.random() < 0.7:
            groups.append(block_to_preferred_group[b])
        else:
            groups.append(rng.choice(group_labels))
    groups = np.array(groups)

    X = np.array([rng.normal(size=n_predictors) + block_offsets[b] for b in blocks])
    dm = make_distance_matrix(X)

    # Unrestricted (standard) permanova -- ignores blocks entirely
    unrestricted = permanova(dm, groups, permutations=permutations)

    # Blocked permanova -- respects block structure
    obs_blocked, p_blocked, _ = blocked_permanova(dm, groups, blocks,
                                                   permutations=permutations, seed=seed)

    return {
        "unrestricted_p": unrestricted["p-value"],
        "unrestricted_stat": unrestricted["test statistic"],
        "blocked_p": p_blocked,
        "blocked_stat": obs_blocked,
    }


if __name__ == "__main__":
    print("=== TEST 1: Type I error rate (target ~0.05) ===")
    rate = test1_type_i_error(n_trials=200, permutations=499)
    print(f"Observed rejection rate across 200 null trials: {rate:.4f}")
    print(f"Expected: ~0.05. Acceptable range (binomial CI, n=200): ~[0.02, 0.09]")
    print()

    print("=== TEST 2: Power under strong artificial signal ===")
    stat, p = test2_power(permutations=999)
    print(f"Observed pseudo-F: {stat:.4f}, p-value: {p:.4f}")
    print(f"Expected: p << 0.05")
    print()

    print("=== TEST 3: Blocked vs unrestricted under block-group confound ===")
    results = test3_blocking_behavior(permutations=999)
    print(f"Unrestricted PERMANOVA: stat={results['unrestricted_stat']:.4f}, p={results['unrestricted_p']:.4f}")
    print(f"Blocked PERMANOVA:      stat={results['blocked_stat']:.4f}, p={results['blocked_p']:.4f}")
    print()
    print("Interpretation: under this deliberate block-group confound, the")
    print("unrestricted test treats confound-driven structure as group signal.")
    print("The blocked test, using the correct restricted null, should show")
    print("attenuated evidence (typically a higher p-value, by roughly an")
    print("order of magnitude or more) relative to the unrestricted test --")
    print("it need not necessarily cross the alpha=0.05 threshold in every")
    print("simulated draw to demonstrate correct behavior; the key signature")
    print("is substantial attenuation, not guaranteed non-significance.")
