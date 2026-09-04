import json
import os
from dataclasses import asdict, is_dataclass


RESULTS_DIR = "results"


def _make_serializable(value):
    """
    Convert common Python/NumPy values into JSON-compatible values.
    """

    if is_dataclass(value):
        return _make_serializable(asdict(value))

    if hasattr(value, "tolist"):
        return value.tolist()

    if isinstance(value, dict):
        return {
            key: _make_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _make_serializable(item)
            for item in value
        ]

    return value


def save_result(sample_id, result):
    """
    Save a result as JSON for one sample.
    """

    sample_dir = os.path.join(
        RESULTS_DIR,
        sample_id
    )

    os.makedirs(
        sample_dir,
        exist_ok=True
    )

    result_path = os.path.join(
        sample_dir,
        "result.json"
    )

    serializable_result = _make_serializable(result)

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            serializable_result,
            file,
            indent=4
        )

    return result_path


def load_result(sample_id):
    """
    Load a previously saved result.
    """

    result_path = os.path.join(
        RESULTS_DIR,
        sample_id,
        "result.json"
    )

    if not os.path.exists(result_path):
        raise FileNotFoundError(
            f"Result not found for sample: {sample_id}"
        )

    with open(
        result_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


if __name__ == "__main__":

    test_result = {
        "sample_id": "train_000",
        "decision": "HIGH_CONFIDENCE_CHANGE",
        "final_score": 0.855,
        "evidence": [
            {
                "source": "temporal_model",
                "score": 0.90
            },
            {
                "source": "spatial_model",
                "score": 0.80
            },
            {
                "source": "optical_sar_model",
                "score": 0.85
            }
        ]
    }

    saved_path = save_result(
        "train_000",
        test_result
    )

    loaded_result = load_result(
        "train_000"
    )

    print("RESULT STORAGE")
    print("--------------")

    print("Saved to:", saved_path)

    print()
    print("Loaded result:")
    print(json.dumps(
        loaded_result,
        indent=4
    ))

    print()
    print("Result storage: PASSED")