from dataclasses import dataclass
from typing import Optional
from raster.model_output import create_model_output
import numpy as np

@dataclass
class Evidence:
    """
    Represents one piece of evidence produced by
    an AI model or analysis component.
    """

    source: str
    score: float
    description: str = ""


@dataclass
class FusionResult:
    """
    Final result produced by the evidence fusion engine.
    """

    final_score: float
    decision: str
    evidence: list


def validate_evidence(evidence):
    """
    Validate an evidence item.
    """

    if not 0.0 <= evidence.score <= 1.0:
        raise ValueError(
            f"Evidence score must be between 0 and 1: "
            f"{evidence.score}"
        )


def evidence_from_model_output(model_output):
    """
    Convert a standardized ModelOutput into Evidence.

    The model output confidence becomes the evidence score.
    """

    if model_output.confidence is None:
        raise ValueError(
            f"Model output from '{model_output.model_name}' "
            "does not contain a confidence score."
        )

    return Evidence(
        source=model_output.model_name,
        score=model_output.confidence,
        description=f"Task: {model_output.task}"
    )


def fuse_evidence(evidence_items, weights=None):
    """
    Combine multiple evidence scores using a weighted average.

    Parameters
    ----------
    evidence_items : list[Evidence]
        Evidence from different sources.

    weights : Optional[dict]
        Optional source-specific weights.

    Returns
    -------
    FusionResult
    """

    if not evidence_items:
        raise ValueError(
            "At least one evidence item is required."
        )

    for evidence in evidence_items:
        validate_evidence(evidence)

    if weights is None:
        weights = {
            evidence.source: 1.0
            for evidence in evidence_items
        }

    weighted_sum = 0.0
    total_weight = 0.0

    for evidence in evidence_items:

        weight = weights.get(
            evidence.source,
            1.0
        )

        weighted_sum += (
            evidence.score * weight
        )

        total_weight += weight

    final_score = weighted_sum / total_weight

    if final_score >= 0.7:
        decision = "HIGH_CONFIDENCE_CHANGE"

    elif final_score >= 0.4:
        decision = "POSSIBLE_CHANGE"

    else:
        decision = "LOW_CONFIDENCE_CHANGE"

    return FusionResult(
        final_score=final_score,
        decision=decision,
        evidence=evidence_items
    )


if __name__ == "__main__":

    # Simulated outputs from other AI teams.
    # These are NOT AI models.
    temporal_output = create_model_output(
        model_name="temporal_model",
        task="bi_temporal_change_detection",
        prediction=np.zeros((256, 256), dtype=np.uint8),
        confidence=0.90
    )

    spatial_output = create_model_output(
        model_name="spatial_model",
        task="spatial_change_analysis",
        prediction=np.zeros((256, 256), dtype=np.uint8),
        confidence=0.80
    )

    optical_sar_output = create_model_output(
        model_name="optical_sar_model",
        task="optical_sar_analysis",
        prediction=np.zeros((256, 256), dtype=np.uint8),
        confidence=0.85
    )

    # Convert ModelOutput → Evidence
    evidence = [
        evidence_from_model_output(temporal_output),
        evidence_from_model_output(spatial_output),
        evidence_from_model_output(optical_sar_output)
    ]

    weights = {
        "temporal_model": 0.4,
        "spatial_model": 0.3,
        "optical_sar_model": 0.3
    }

    result = fuse_evidence(
        evidence,
        weights
    )

    print("MODEL OUTPUT → EVIDENCE → FUSION")
    print("--------------------------------")

    print("Final score:", result.final_score)
    print("Decision:", result.decision)

    print()
    print("Evidence:")

    for item in result.evidence:
        print(
            f"  {item.source}: "
            f"{item.score} - "
            f"{item.description}"
        )

    print()
    print("ModelOutput → Evidence conversion: PASSED")
    print("Evidence fusion: PASSED")