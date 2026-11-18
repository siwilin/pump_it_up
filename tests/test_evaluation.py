import numpy as np

from src.evaluation import business_cost, confusion_matrix_frame


def test_business_cost_penalizes_missed_repairs() -> None:
    target = np.array(["functional needs repair", "functional"])
    missed = np.array(["functional", "functional"])
    other_error = np.array(["functional needs repair", "non functional"])

    assert business_cost(target, missed) > business_cost(target, other_error)


def test_confusion_matrix_frame_preserves_labels() -> None:
    labels = ["functional", "non functional"]
    frame = confusion_matrix_frame(np.array([[2, 1], [0, 3]]), labels)

    assert list(frame.index) == labels
    assert list(frame.columns) == labels
    assert frame.loc["functional", "non functional"] == 1
