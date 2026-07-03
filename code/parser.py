"""
parser.py — Reasoning-State Expression tag extraction.

Implements the frozen Step 3 extraction rules:
  - Parses <think>, <memory>, <reflection>, <plan>, <action> (and GAIA's
    terminal <answer>) from each assistant message in a trajectory.
  - A tag's absence at a given step means that predictor simply does not
    aggregate a value for that step (no imputation, no zero-filling).
  - This module performs NO tokenization. Token counting is isolated in
    tokenizer.py so that the execution environment can be swapped (e.g.
    sandbox prototyping -> Kaggle) without touching extraction logic.

This module is environment-agnostic: ALFWorld, GAIA, and WebShop all use
the same tag vocabulary, as verified by direct inspection during Stage 1
(two samples per environment, all consistent).
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field

TAG_NAMES = ["think", "memory", "reflection", "plan", "action", "answer"]
TAG_PATTERN = {tag: re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL) for tag in TAG_NAMES}


@dataclass
class StepExtraction:
    """Extracted tag content for a single assistant message (one step)."""
    message_index: int
    think: str | None = None
    memory: str | None = None
    reflection: str | None = None
    plan: str | None = None
    action: str | None = None
    answer: str | None = None


@dataclass
class TrajectoryExtraction:
    """Full extraction result for one trajectory file."""
    trajectory_id: str
    environment: str
    file_path: str
    metadata: dict
    steps: list = field(default_factory=list)  # list[StepExtraction]
    parse_errors: list = field(default_factory=list)


def extract_tags_from_content(content: str) -> dict:
    """Extract each known tag's inner text from a single message's content.
    Returns a dict {tag_name: text_or_None}. Absence is represented as None,
    never as an empty string, so downstream aggregation can distinguish
    'tag present but empty' from 'tag not present at all'."""
    result = {}
    for tag in TAG_NAMES:
        match = TAG_PATTERN[tag].search(content)
        result[tag] = match.group(1).strip() if match else None
    return result


def parse_trajectory_file(path: Path, environment: str) -> TrajectoryExtraction:
    """Parse a single trajectory JSON file into a TrajectoryExtraction."""
    trajectory_id = path.stem  # filename without .json extension
    errors = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        errors.append(f"JSON_LOAD_FAILURE: {e}")
        return TrajectoryExtraction(
            trajectory_id=trajectory_id,
            environment=environment,
            file_path=str(path),
            metadata={},
            steps=[],
            parse_errors=errors,
        )

    metadata = data.get("metadata", {})
    if "metadata" not in data:
        errors.append("MISSING_METADATA_FIELD")

    messages = data.get("messages", [])
    if "messages" not in data:
        errors.append("MISSING_MESSAGES_FIELD")

    steps = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            errors.append(f"NON_STRING_CONTENT_AT_INDEX_{i}")
            continue
        tags = extract_tags_from_content(content)
        steps.append(StepExtraction(
            message_index=i,
            think=tags["think"],
            memory=tags["memory"],
            reflection=tags["reflection"],
            plan=tags["plan"],
            action=tags["action"],
            answer=tags["answer"],
        ))

    return TrajectoryExtraction(
        trajectory_id=trajectory_id,
        environment=environment,
        file_path=str(path),
        metadata=metadata,
        steps=steps,
        parse_errors=errors,
    )


def parse_all_trajectories(data_dir: Path) -> list:
    """Parse every trajectory across ALFWorld, GAIA, WebShop subfolders."""
    results = []
    for env_name in ["ALFWorld", "GAIA", "WebShop"]:
        env_dir = data_dir / env_name
        json_files = sorted(env_dir.glob("*.json"))
        for jf in json_files:
            results.append(parse_trajectory_file(jf, env_name))
    return results


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    trajectories = parse_all_trajectories(data_dir)
    print(f"Parsed {len(trajectories)} trajectories")
    error_count = sum(1 for t in trajectories if t.parse_errors)
    print(f"Trajectories with parse errors: {error_count}")
    for t in trajectories:
        if t.parse_errors:
            print(f"  {t.trajectory_id}: {t.parse_errors}")
