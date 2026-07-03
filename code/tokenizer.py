"""
tokenizer.py — Isolated token-counting module.

Frozen Step 3 decision: tiktoken, o200k_base encoding, irrespective of
the model that originally generated the trajectory.

This module is intentionally the ONLY place token counting happens, so
that switching execution environments (e.g. this sandbox, where the
o200k_base BPE file cannot be downloaded because
openaipublic.blob.core.windows.net is not network-allowlisted, -> Kaggle,
where it should download normally) touches exactly one function and
nothing else in the pipeline.

STATUS IN THIS ENVIRONMENT: o200k_base failed to load (HTTP 403,
host_not_allowed on the encoding download). Per the Document First,
Decide Second governance rule, this is documented here rather than
silently substituted. A diagnostic fallback (count_tokens_DIAGNOSTIC_ONLY)
is provided so the rest of the pipeline (parser, aggregator, predictor
table shape, Step 8 gate logic) can be built and verified end-to-end in
this sandbox. Diagnostic-mode output is NOT the frozen predictor table
and must never be used for Step 5 statistical analysis. The final,
publication-grade predictor_table.csv must be regenerated on a host
where tiktoken's o200k_base encoding can be downloaded (e.g. Kaggle).
"""

import warnings

_TIKTOKEN_AVAILABLE = False
_encoding = None

try:
    import tiktoken
    _encoding = tiktoken.get_encoding("o200k_base")
    _TIKTOKEN_AVAILABLE = True
except Exception as e:
    warnings.warn(
        f"tiktoken o200k_base unavailable in this environment ({e}). "
        f"Falling back to DIAGNOSTIC-ONLY token counting. "
        f"Do NOT use this output for the frozen Step 5 statistical analysis."
    )


def count_tokens(text: str) -> int:
    """Frozen token-counting function: tiktoken, o200k_base encoding.
    Raises RuntimeError if the real encoding is unavailable, to prevent
    silent use of a substitute tokenizer in any analysis-facing code path."""
    if not _TIKTOKEN_AVAILABLE:
        raise RuntimeError(
            "tiktoken o200k_base is not available in this environment. "
            "Use count_tokens_DIAGNOSTIC_ONLY() for pipeline verification, "
            "or run this module on a host where the encoding can download."
        )
    if text is None:
        return 0
    return len(_encoding.encode(text))


def count_tokens_DIAGNOSTIC_ONLY(text: str) -> int:
    """NON-FROZEN diagnostic fallback for pipeline verification only.
    Uses a simple whitespace-based approximation. Output from this
    function must never be used to populate the final predictor_table.csv
    that feeds Step 5 analyses -- it exists solely so parser.py and
    aggregator.py can be built and tested end-to-end while the real
    tokenizer is blocked in this sandbox."""
    if text is None:
        return 0
    return len(text.split())


def get_active_tokenizer_status() -> str:
    """Returns a human-readable status string for logging in Step 8 output."""
    if _TIKTOKEN_AVAILABLE:
        return "FROZEN: tiktoken o200k_base (active)"
    else:
        return "DIAGNOSTIC ONLY: tiktoken o200k_base unavailable, whitespace-split fallback in use"


if __name__ == "__main__":
    print(get_active_tokenizer_status())
    sample = "This is a short reasoning trace about finding an apple."
    print(f"Sample text: {sample!r}")
    try:
        print(f"  tiktoken o200k_base tokens: {count_tokens(sample)}")
    except RuntimeError as e:
        print(f"  tiktoken o200k_base: UNAVAILABLE ({e})")
    print(f"  diagnostic whitespace tokens: {count_tokens_DIAGNOSTIC_ONLY(sample)}")
