"""Helpers for dataset bookkeeping."""


def image_label_pair_count(items: list[tuple[str, int]]) -> int:
    """Return the number of labeled image items."""

    return len(items)
