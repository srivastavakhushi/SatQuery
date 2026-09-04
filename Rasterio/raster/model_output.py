from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ModelOutput:
    """
    Standardized output received from an AI model.
    """

    model_name: str
    task: str

    prediction: np.ndarray

    confidence: Optional[float] = None

    metadata: Optional[dict] = None


def create_model_output(
    model_name,
    task,
    prediction,
    confidence=None,
    metadata=None
):
    """
    Create a standardized ModelOutput object.
    """

    if not isinstance(prediction, np.ndarray):
        raise TypeError(
            "Prediction must be a NumPy array."
        )

    if confidence is not None:

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "Confidence must be between 0 and 1."
            )

    return ModelOutput(
        model_name=model_name,
        task=task,
        prediction=prediction,
        confidence=confidence,
        metadata=metadata
    )


if __name__ == "__main__":

    # Simulated model prediction.
    # This is NOT an AI model.
    prediction = np.zeros(
        (256, 256),
        dtype=np.uint8
    )

    output = create_model_output(
        model_name="change_detection_model",
        task="bi_temporal_change_detection",
        prediction=prediction,
        confidence=0.92,
        metadata={
            "sample_id": "train_000"
        }
    )

    print("MODEL OUTPUT")
    print("------------")

    print("Model:", output.model_name)
    print("Task:", output.task)
    print("Prediction shape:", output.prediction.shape)
    print("Prediction dtype:", output.prediction.dtype)
    print("Confidence:", output.confidence)
    print("Metadata:", output.metadata)

    print()
    print("Model output interface: PASSED")