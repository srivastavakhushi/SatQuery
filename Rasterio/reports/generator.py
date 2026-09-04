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
    """

    result = load_result(sample_id)

    os.makedirs(
        REPORTS_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        REPORTS_DIR,
        f"{sample_id}_report.pdf"
    )

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

    # Title
    story.append(
        Paragraph(
            "Satellite Analysis Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 8 * mm))

    # Basic information
    story.append(
        Paragraph(
            f"<b>Sample ID:</b> {result['sample_id']}",
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

    story.append(Spacer(1, 6 * mm))

    # Final decision
    story.append(
        Paragraph(
            "Final Analysis",
            styles["Heading2"]
        )
    )

    decision_data = [
        ["Decision", result["decision"]],
        ["Fusion Confidence", f"{result['final_score']:.3f}"],
    ]

    decision_table = Table(
        decision_data,
        colWidths=[55 * mm, 100 * mm]
    )

    decision_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(decision_table)

    story.append(Spacer(1, 8 * mm))

    # Evidence section
    story.append(
        Paragraph(
            "Evidence Summary",
            styles["Heading2"]
        )
    )

    evidence_table_data = [
        ["Evidence Source", "Score"]
    ]

    for evidence in result["evidence"]:
        evidence_table_data.append([
            evidence["source"],
            f"{evidence['score']:.3f}"
        ])

    evidence_table = Table(
        evidence_table_data,
        colWidths=[100 * mm, 55 * mm]
    )

    evidence_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 1), (1, -1), "CENTER"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(evidence_table)

    story.append(Spacer(1, 8 * mm))

    # Explanation
    story.append(
        Paragraph(
            "Report Summary",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            "The final assessment was produced by combining "
            "evidence from multiple analysis sources through "
            "the Multi-Evidence Fusion Engine.",
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 5 * mm))

    story.append(
        Paragraph(
            "This report is generated automatically from the "
            "stored analysis result.",
            styles["BodyText"]
        )
    )

    document.build(story)

    return output_path


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