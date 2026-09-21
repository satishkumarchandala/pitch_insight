from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


OUTPUT_FILE = Path(__file__).with_name("Pitch_Insight_Interview_Preparation.pdf")


def build_pdf(output_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Pitch Insight Interview Preparation",
        author="GitHub Copilot",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0f172a"),
            fontSize=22,
            leading=26,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubtitleCenter",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#334155"),
            fontSize=10,
            leading=13,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#0f172a"),
            fontSize=14,
            leading=18,
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#1f2937"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="QuestionStyle",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=12.5,
            textColor=colors.HexColor("#111827"),
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AnswerStyle",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12.2,
            textColor=colors.HexColor("#374151"),
            leftIndent=6,
            spaceAfter=4,
        )
    )

    story = []
    story.append(Paragraph("Pitch Insight Interview Preparation", styles["TitleCenter"]))
    story.append(
        Paragraph(
            "A concise, interview-ready guide to explain the project, architecture, machine learning flow, APIs, and trade-offs.",
            styles["SubtitleCenter"],
        )
    )
    story.append(Spacer(1, 4))

    section_blocks = [
        (
            "Project Snapshot",
            [
                "Pitch Insight is a cricket pitch analysis platform built with a React/Vite frontend, FastAPI backend, and MongoDB persistence.",
                "The core value is turning an uploaded pitch image into a prediction, feature breakdown, weather-aware strategy, and optional Pro-only deep analysis.",
                "The system combines YOLO-based pitch detection, OpenCV feature extraction, and ONNX-based pitch classification.",
            ],
        ),
        (
            "Architecture Summary",
            [
                "Frontend: React 18 + Vite with upload flow, results visualization, auth, pricing, history, and chat widget.",
                "Backend: FastAPI modular routers for analysis, auth, chat, subscription, weather, and health.",
                "Storage: MongoDB collections for users and analysis history.",
                "External services: Weather API, Google Gemini, Razorpay.",
            ],
        ),
        (
            "How It Works",
            [
                "Image upload enters /api/analyze or /api/quick-analyze.",
                "YOLO detects the pitch region and returns bounding-box coordinates.",
                "OpenCV extracts grass coverage, cracks, moisture, color, texture, and brightness signals.",
                "ResNet18/ONNX classifies the pitch into batting, bowling, spin, or seam friendly.",
                "Rule-based adjustments combine ML output with detected surface features and, when requested, weather data.",
                "The backend returns final class, confidence, probable adjustments, and match strategy recommendations.",
            ],
        ),
        (
            "Key Features",
            [
                "Quick analysis for faster classification.",
                "Complete analysis for richer feature extraction and weather-aware strategy.",
                "JWT authentication for user accounts.",
                "Razorpay subscription flow for Pro access.",
                "Analysis history and saved reports.",
                "Gemini-powered cricket chat for Pro users.",
            ],
        ),
    ]

    for heading, items in section_blocks:
        story.append(Paragraph(heading, styles["SectionHeading"]))
        for item in items:
            story.append(Paragraph(f"• {item}", styles["BodySmall"]))
        story.append(Spacer(1, 4))

    story.append(Paragraph("Likely Interview Questions With Answer Starters", styles["SectionHeading"]))
    interview_qas = [
        (
            "1. What problem does Pitch Insight solve?",
            "It reduces manual guesswork in pitch assessment by using AI to analyze a cricket pitch image, predict pitch behavior, and suggest match strategy.",
        ),
        (
            "2. What is the end-to-end flow?",
            "Upload image -> pitch detection -> feature extraction -> classification -> optional weather adjustment -> strategy generation -> response to the frontend.",
        ),
        (
            "3. Why did you use FastAPI on the backend?",
            "FastAPI gives strong request validation, automatic docs, async support, and a clean router-based structure for separate modules.",
        ),
        (
            "4. Why use ONNX for the ML pipeline?",
            "ONNX makes inference lightweight, cross-platform, and suitable for faster deployment compared with keeping the whole pipeline in PyTorch.",
        ),
        (
            "5. What makes the complete analysis different from quick analysis?",
            "Quick analysis returns a fast classification. Complete analysis adds richer feature extraction, weather context, and strategy generation.",
        ),
        (
            "6. How is authentication handled?",
            "Users sign up and log in through JWT-based auth. Passwords are hashed, and tokens are stored on the frontend for protected requests.",
        ),
        (
            "7. How does the subscription model work?",
            "Users can upgrade through Razorpay. After successful payment verification, the backend marks the user as Pro and unlocks premium features.",
        ),
        (
            "8. How is weather used in the product?",
            "Weather data is used to modify pitch interpretation and generate better toss, bowling, and batting recommendations.",
        ),
        (
            "9. How is the analysis history stored?",
            "Saved analyses are written to MongoDB, allowing users to revisit earlier results and reuse them in chat or profile views.",
        ),
        (
            "10. What would you improve next?",
            "Likely next steps are stronger model evaluation, better caching or queueing for expensive inference, richer analytics, and more robust chat history persistence.",
        ),
    ]

    for question, answer in interview_qas:
        story.append(Paragraph(question, styles["QuestionStyle"]))
        story.append(Paragraph(answer, styles["AnswerStyle"]))

    story.append(Spacer(1, 6))
    story.append(Paragraph("Short explanation you can say in the interview:", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "Pitch Insight is a cricket pitch intelligence platform. It uses computer vision and machine learning to detect the pitch, extract surface features, classify the pitch type, and combine that with weather and domain rules to recommend match strategy. The product also includes login, subscriptions, analysis history, and a cricket-focused AI chat experience for premium users.",
            styles["BodySmall"],
        )
    )

    def draw_page(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.setLineWidth(0.6)
        canvas.line(document.leftMargin, A4[1] - 12 * mm, A4[0] - document.rightMargin, A4[1] - 12 * mm)
        canvas.line(document.leftMargin, 12 * mm, A4[0] - document.rightMargin, 12 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(document.leftMargin, 9 * mm, "Pitch Insight interview prep")
        canvas.drawRightString(A4[0] - document.rightMargin, 9 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)


if __name__ == "__main__":
    build_pdf(OUTPUT_FILE)
    print(f"Created {OUTPUT_FILE}")