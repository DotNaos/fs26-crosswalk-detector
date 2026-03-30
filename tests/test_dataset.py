from crosswalk_detector import image_label_pair_count


def test_image_label_pair_count_returns_number_of_samples() -> None:
    samples = [("tile-001.png", 1), ("tile-002.png", 0)]

    assert image_label_pair_count(samples) == 2
