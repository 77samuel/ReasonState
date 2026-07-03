"""
aggregator.py — Computes the 12 frozen predictors per trajectory.

Implements Step 3's construct/predictor table exactly:
  Reasoning Expression:        mean_think_length, var_think_length
  Memory Expression:           mean_memory_length, memory_proportion, memory_variability
  Reflection Expression:       mean_reflection_length, reflection_proportion, reflection_variability
  Planning Expression (excl.): mean_plan_length, plan_proportion, plan_variability
  Reasoning-Action Balance:    reasoning_to_action_ratio

Extraction/aggregation rules (frozen):
  - A predictor's per-step inputs come only from steps where that tag is
    present; absent-tag steps are excluded from that predictor's
    aggregation, never imputed as zero or treated as missing data needing
    interpolation.
  - GAIA's terminal step uses <answer> in place of <action>; for the
    reasoning-to-action ratio, <answer> tokens are treated as the
    action-equivalent denominator contribution for that step.
  - Proportions are computed as: tag_tokens / (think+memory+reflection+plan
    tokens), per trajectory, using only steps where the numerator tag is
    present (denominator always uses all reasoning-tag tokens across the
    whole trajectory, per the frozen definition in Step 3).
  - No normalization (e.g. z-scoring) happens here. Within-environment
    standardization is a Step 5A/5B/5C analysis-stage operation, not an
    extraction-stage operation, per the frozen separation of concerns.
"""

import statistics
from dataclasses import dataclass

from tokenizer import count_tokens_DIAGNOSTIC_ONLY as count_tokens
# ^ SWAP POINT: replace the above import with
#   `from tokenizer import count_tokens`
# when running on a host where tiktoken's o200k_base encoding is available
# (e.g. Kaggle). No other line in this file should need to change.


@dataclass
class TrajectoryPredictors:
    """Blueprint v2: Reasoning Expression (<think>-based) and Reasoning-Action
    Balance constructs removed per the Step 3 reopening (construct-validity
    failure -- <think> presence is model-determined: 0%/0%/40%/100% across
    the four model identities in this dataset). Confirmatory set is now
    Memory Expression + Reflection Expression (6 predictors). Planning
    Expression (3 predictors) remains exploratory-only."""
    trajectory_id: str
    environment: str
    model: str
    n_steps_total: int
    n_steps_with_memory: int
    n_steps_with_reflection: int
    n_steps_with_plan: int
    mean_memory_length: float | None
    memory_proportion: float | None
    memory_variability: float | None
    mean_reflection_length: float | None
    reflection_proportion: float | None
    reflection_variability: float | None
    mean_plan_length: float | None
    plan_proportion: float | None
    plan_variability: float | None
    extraction_warnings: list


def _lengths(steps, attr_name):
    """Token-length list for a given tag attribute, over steps where present."""
    out = []
    for s in steps:
        val = getattr(s, attr_name)
        if val is not None:
            out.append(count_tokens(val))
    return out


def compute_predictors(trajectory) -> TrajectoryPredictors:
    """Compute the 9 frozen predictors (Blueprint v2) for one parsed
    TrajectoryExtraction. <think> and the reasoning-to-action ratio are
    no longer computed -- see module docstring and Step 3 reopening
    record in the blueprint."""
    steps = trajectory.steps
    warnings_list = []

    memory_lens = _lengths(steps, "memory")
    reflection_lens = _lengths(steps, "reflection")
    plan_lens = _lengths(steps, "plan")

    total_reasoning_tokens = sum(memory_lens) + sum(reflection_lens) + sum(plan_lens)

    def mean_or_none(lst):
        return statistics.mean(lst) if lst else None

    def var_or_none(lst):
        return statistics.variance(lst) if len(lst) >= 2 else None

    def proportion_or_none(component_sum, denom):
        if denom == 0:
            return None
        return component_sum / denom

    mean_memory = mean_or_none(memory_lens)
    memory_var = var_or_none(memory_lens)
    memory_prop = proportion_or_none(sum(memory_lens), total_reasoning_tokens)

    mean_reflection = mean_or_none(reflection_lens)
    reflection_var = var_or_none(reflection_lens)
    reflection_prop = proportion_or_none(sum(reflection_lens), total_reasoning_tokens)

    mean_plan = mean_or_none(plan_lens)
    plan_var = var_or_none(plan_lens)
    plan_prop = proportion_or_none(sum(plan_lens), total_reasoning_tokens)

    if not memory_lens:
        warnings_list.append("NO_MEMORY_TAGS_FOUND")
    if not reflection_lens:
        warnings_list.append("NO_REFLECTION_TAGS_FOUND")
    if not plan_lens:
        warnings_list.append("NO_PLAN_TAGS_FOUND")

    return TrajectoryPredictors(
        trajectory_id=trajectory.trajectory_id,
        environment=trajectory.environment,
        model=trajectory.metadata.get("model", "UNKNOWN"),
        n_steps_total=len(steps),
        n_steps_with_memory=len(memory_lens),
        n_steps_with_reflection=len(reflection_lens),
        n_steps_with_plan=len(plan_lens),
        mean_memory_length=mean_memory,
        memory_proportion=memory_prop,
        memory_variability=memory_var,
        mean_reflection_length=mean_reflection,
        reflection_proportion=reflection_prop,
        reflection_variability=reflection_var,
        mean_plan_length=mean_plan,
        plan_proportion=plan_prop,
        plan_variability=plan_var,
        extraction_warnings=warnings_list,
    )
