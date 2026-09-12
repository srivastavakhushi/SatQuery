import json
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors


RESULTS_DIR = "results"
REPORTS_DIR = "reports_output"


def load_result(sample_id):
    """
    Load a stored analysis result.
    """

    result_path = os.path.join(
        RESULTS_DIR,
        sample_id,
        "result.json"
    )

    if not os.path.exists(result_path):
        raise FileNotFoundError(
            f"Result not found: {result_path}"
        )

    with open(
        result_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def generate_report(sample_id):
    """
    Generate a PDF analysis report from a stored result.

    The report contains information only about the model
    that produced the current analysis result.
    """

    # ---------------------------------------------------------
    # Load result
    # ---------------------------------------------------------

    result = load_result(sample_id)

    # Get the model and task used for this analysis.
    model_name = result.get(
        "model_name",
        "Unknown"
    )

    task = result.get(
        "task",
        "Unknown"
    )

    os.makedirs(
        REPORTS_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        REPORTS_DIR,
        f"{sample_id}_report.pdf"
    )

    # ---------------------------------------------------------
    # Create PDF document
    # ---------------------------------------------------------

    document = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()

    story = []

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Satellite Analysis Report",
            styles["Title"]
        )
    )

    story.append(
        Spacer(
            1,
            8 * mm
        )
    )

    # ---------------------------------------------------------
    # Basic Information
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            f"<b>Sample ID:</b> {result.get('sample_id', sample_id)}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Model:</b> {model_name}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Task:</b> {task}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Generated:</b> "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(
            1,
            6 * mm
        )
    )

    # ---------------------------------------------------------
    # Final Analysis
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Final Analysis",
            styles["Heading2"]
        )
    )

    decision = result.get(
        "decision",
        "Not available"
    )

    final_score = result.get(
        "final_score",
        None
    )

    if final_score is not None:
        try:
            score_text = f"{float(final_score):.3f}"
        except (TypeError, ValueError):
            score_text = str(final_score)
    else:
        score_text = "Not available"

    decision_data = [
        ["Decision", decision],
        ["Confidence / Score", score_text],
    ]

    decision_table = Table(
        decision_data,
        colWidths=[
            55 * mm,
            100 * mm
        ]
    )

    decision_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.black
            ),
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            ),
        ])
    )

    story.append(
        decision_table
    )

    story.append(
        Spacer(
            1,
            8 * mm
        )
    )

    # ---------------------------------------------------------
    # Evidence Section
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Evidence Summary",
            styles["Heading2"]
        )
    )

    evidence_table_data = [
        [
            "Evidence Source",
            "Score"
        ]
    ]

    evidence_list = result.get(
        "evidence",
        []
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Only include evidence generated by the current model.
    # ---------------------------------------------------------

    for evidence in evidence_list:

        evidence_source = evidence.get(
            "source",
            "Unknown"
        )

        # If model_name exists, only include evidence
        # belonging to this model.
        if (
            model_name != "Unknown"
            and evidence_source != model_name
        ):
            continue

        evidence_score = evidence.get(
            "score",
            None
        )

        if evidence_score is not None:
            try:
                score_text = f"{float(evidence_score):.3f}"
            except (TypeError, ValueError):
                score_text = str(evidence_score)
        else:
            score_text = "N/A"

        evidence_table_data.append([
            evidence_source,
            score_text
        ])

    # If there is no matching evidence
    if len(evidence_table_data) == 1:

        evidence_table_data.append([
            model_name,
            "No evidence score available"
        ])

    evidence_table = Table(
        evidence_table_data,
        colWidths=[
            100 * mm,
            55 * mm
        ]
    )

    evidence_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.black
            ),
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "ALIGN",
                (1, 1),
                (1, -1),
                "CENTER"
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            ),
        ])
    )

    story.append(
        evidence_table
    )

    story.append(
        Spacer(
            1,
            8 * mm
        )
    )

    # ---------------------------------------------------------
    # Report Summary
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Report Summary",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            f"The analysis was generated using the "
            f"<b>{model_name}</b> model for the "
            f"<b>{task}</b> task.",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm
        )
    )

    story.append(
        Paragraph(
            "This report is generated automatically from the "
            "stored analysis result.",
            styles["BodyText"]
        )
    )

    # ---------------------------------------------------------
    # Build PDF
    # ---------------------------------------------------------

    document.build(
        story
    )

    return output_path


# -------------------------------------------------------------
# Test / Standalone Execution
# -------------------------------------------------------------

if __name__ == "__main__":

    sample_id = "train_000"

    report_path = generate_report(
        sample_id
    )

    print("PDF REPORT")
    print("----------")
    print("Sample:", sample_id)
    print("Report generated:", report_path)
    print()
    print("PDF report generation: PASSED")