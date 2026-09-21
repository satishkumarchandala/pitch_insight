from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


OUTPUT_FILE = Path(__file__).with_name("Pitch_Insight_Interview_Cheat_Sheet.pdf")


def build_pdf(output_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="Pitch Insight Interview Cheat Sheet",
        author="GitHub Copilot",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0f172a"),
            fontSize=21,
            leading=24,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubTitle",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569"),
            fontSize=9.5,
            leading=12,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HeadingMini",
            parent=styles["Heading3"],
            textColor=colors.HexColor("#0f172a"),
            fontSize=12,
            leading=14,
            spaceBefore=5,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyMini",
            parent=styles["BodyText"],
            fontSize=9,
            leading=11.5,
            textColor=colors.HexColor("#1f2937"),
            spaceAfter=2,
        )
    )

    story = []
    story.append(Paragraph("Pitch Insight Interview Cheat Sheet", styles["TitleCenter"]))
    story.append(Paragraph("Use this for quick revision before the interview.", styles["SubTitle"]))
    story.append(Spacer(1, 3))

    sections = [
        (
            "30-Second Pitch",
            "Pitch Insight is a cricket pitch analysis platform. It uses computer vision and machine learning to detect the pitch, extract surface features, classify pitch type, and combine that with weather and domain rules to recommend match strategy. It also includes auth, subscriptions, analysis history, and a Pro chat assistant.",
        ),
        (
            "Architecture",
            "Frontend: React + Vite. Backend: FastAPI with modular routers. Database: MongoDB. ML: YOLO pitch detection, OpenCV feature extraction, and ONNX-based classification. External integrations: Weather API, Gemini, and Razorpay.",
        ),
        (
            "Core Flow",
            "Upload image -> detect pitch region -> extract features -> classify pitch type -> apply feature and weather adjustments -> generate strategy -> return results to the UI.",
        ),
        (
            "Most Likely Questions",
            "Why FastAPI? Strong validation, docs, and router structure. Why ONNX? Faster and lightweight inference. What is the difference between quick and complete analysis? Quick is fast classification; complete adds richer features and weather. How is Pro access handled? JWT auth plus Razorpay payment verification. Where is history stored? MongoDB analysis history collection.",
        ),
        (
            "Strong Closing Line",
            "The project solves a real cricket decision problem by converting pitch images into actionable, explainable insights for match preparation.",
        ),
    ]

    for heading, body in sections:
        story.append(Paragraph(heading, styles["HeadingMini"]))
        story.append(Paragraph(body, styles["BodyMini"]))

    def draw_page(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.setLineWidth(0.6)
        canvas.line(document.leftMargin, A4[1] - 11 * mm, A4[0] - document.rightMargin, A4[1] - 11 * mm)
        canvas.line(document.leftMargin, 11 * mm, A4[0] - document.rightMargin, 11 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(document.leftMargin, 8 * mm, "Pitch Insight interview cheat sheet")
        canvas.drawRightString(A4[0] - document.rightMargin, 8 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)


if __name__ == "__main__":
    build_pdf(OUTPUT_FILE)
    print(f"Created {OUTPUT_FILE}")