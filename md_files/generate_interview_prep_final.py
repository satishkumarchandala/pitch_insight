#!/usr/bin/env python3
"""
Comprehensive Interview Preparation PDF Generator for Pitch Insight Project
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import ListFlowable, ListItem
import datetime

# ─── Color Palette ────────────────────────────────────────────────────────────
DARK_BG      = colors.HexColor('#0f172a')
PRIMARY      = colors.HexColor('#10b981')
SECONDARY    = colors.HexColor('#3b82f6')
ACCENT_ORANGE= colors.HexColor('#f59e0b')
ACCENT_RED   = colors.HexColor('#ef4444')
WHITE        = colors.white
LIGHT_GRAY   = colors.HexColor('#f1f5f9')
MID_GRAY     = colors.HexColor('#64748b')
CARD_BG      = colors.HexColor('#1e293b')
DARK_GREEN   = colors.HexColor('#065f46')
LIGHT_GREEN  = colors.HexColor('#d1fae5')
DARK_BLUE    = colors.HexColor('#1e3a5f')
LIGHT_BLUE   = colors.HexColor('#dbeafe')
LIGHT_ORANGE = colors.HexColor('#fef3c7')
LIGHT_RED    = colors.HexColor('#fee2e2')


def build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles['cover_title'] = ParagraphStyle(
        'cover_title', fontName='Helvetica-Bold', fontSize=28,
        textColor=WHITE, alignment=TA_CENTER, spaceAfter=6,
        leading=34
    )
    styles['cover_subtitle'] = ParagraphStyle(
        'cover_subtitle', fontName='Helvetica', fontSize=14,
        textColor=PRIMARY, alignment=TA_CENTER, spaceAfter=4
    )
    styles['cover_meta'] = ParagraphStyle(
        'cover_meta', fontName='Helvetica', fontSize=10,
        textColor=LIGHT_GRAY, alignment=TA_CENTER, spaceAfter=2
    )

    styles['chapter'] = ParagraphStyle(
        'chapter', fontName='Helvetica-Bold', fontSize=16,
        textColor=WHITE, spaceBefore=18, spaceAfter=8,
        borderPad=6, leading=20
    )
    styles['section'] = ParagraphStyle(
        'section', fontName='Helvetica-Bold', fontSize=12,
        textColor=PRIMARY, spaceBefore=10, spaceAfter=4
    )
    styles['question'] = ParagraphStyle(
        'question', fontName='Helvetica-Bold', fontSize=10,
        textColor=DARK_BG, spaceBefore=8, spaceAfter=3, leading=14
    )
    styles['answer'] = ParagraphStyle(
        'answer', fontName='Helvetica', fontSize=9,
        textColor=colors.HexColor('#1e293b'), spaceAfter=4,
        leading=14, leftIndent=10
    )
    styles['bullet'] = ParagraphStyle(
        'bullet', fontName='Helvetica', fontSize=9,
        textColor=colors.HexColor('#1e293b'), leftIndent=20,
        spaceAfter=2, leading=13, bulletIndent=10
    )
    styles['code'] = ParagraphStyle(
        'code', fontName='Courier', fontSize=8,
        textColor=colors.HexColor('#1e293b'), leftIndent=12,
        spaceAfter=4, leading=12, backColor=LIGHT_GRAY,
        borderPad=4
    )
    styles['tip_text'] = ParagraphStyle(
        'tip_text', fontName='Helvetica', fontSize=8.5,
        textColor=colors.HexColor('#064e3b'), leading=13,
        leftIndent=4
    )
    styles['warning_text'] = ParagraphStyle(
        'warning_text', fontName='Helvetica', fontSize=8.5,
        textColor=colors.HexColor('#7c2d12'), leading=13,
        leftIndent=4
    )
    styles['toc_item'] = ParagraphStyle(
        'toc_item', fontName='Helvetica', fontSize=10,
        textColor=DARK_BG, spaceAfter=3, leading=14
    )
    styles['normal'] = ParagraphStyle(
        'normal_custom', fontName='Helvetica', fontSize=9,
        textColor=colors.HexColor('#1e293b'), leading=14,
        spaceAfter=3
    )
    styles['bold_label'] = ParagraphStyle(
        'bold_label', fontName='Helvetica-Bold', fontSize=9,
        textColor=DARK_BG, leading=13
    )

    return styles


def qa_card(q_num, question, answer_lines, styles, bg_color=LIGHT_BLUE, q_color=DARK_BLUE):
    """Returns a styled Q&A block as a KeepTogether flowable."""
    items = []

    q_text = f"<b>Q{q_num}. {question}</b>"
    q_para = Paragraph(q_text, ParagraphStyle(
        'q_inner', fontName='Helvetica-Bold', fontSize=9.5,
        textColor=q_color, leading=14
    ))

    answer_paragraphs = []
    for line in answer_lines:
        if line.startswith('•'):
            answer_paragraphs.append(Paragraph(
                line, ParagraphStyle('a_bullet', fontName='Helvetica', fontSize=9,
                    textColor=colors.HexColor('#1e293b'), leftIndent=12, leading=13)
            ))
        elif line.startswith('→'):
            answer_paragraphs.append(Paragraph(
                f"<i>{line}</i>", ParagraphStyle('a_arrow', fontName='Helvetica-Oblique', fontSize=8.5,
                    textColor=MID_GRAY, leftIndent=16, leading=12)
            ))
        else:
            answer_paragraphs.append(Paragraph(
                line, ParagraphStyle('a_text', fontName='Helvetica', fontSize=9,
                    textColor=colors.HexColor('#1e293b'), leading=13, leftIndent=6)
            ))

    inner_table_data = [[q_para], *[[ap] for ap in answer_paragraphs]]
    inner_table = Table(inner_table_data, colWidths=[6.5*inch])
    inner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), bg_color),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8faff')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#c7d2fe')),
        ('LINEBELOW', (0,0), (0,0), 0.5, colors.HexColor('#93c5fd')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f0f9ff')]),
    ]))
    return KeepTogether([inner_table, Spacer(1, 6)])


def section_header(title, styles):
    """Returns a visually distinct section header."""
    data = [[Paragraph(f"  🏏  {title}", ParagraphStyle(
        'sec_hdr', fontName='Helvetica-Bold', fontSize=12,
        textColor=WHITE, leading=16
    ))]]
    t = Table(data, colWidths=[6.8*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), DARK_BG),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('LINEBELOW', (0,0), (-1,-1), 2, PRIMARY),
    ]))
    return KeepTogether([Spacer(1, 10), t, Spacer(1, 6)])


def tip_box(text, styles, box_type='tip'):
    bg  = LIGHT_GREEN if box_type == 'tip' else LIGHT_ORANGE
    tc  = colors.HexColor('#064e3b') if box_type == 'tip' else colors.HexColor('#78350f')
    icon = "💡 TIP:" if box_type == 'tip' else "⚠️  NOTE:"
    data = [[Paragraph(f"<b>{icon}</b> {text}", ParagraphStyle(
        'tip_inner', fontName='Helvetica', fontSize=8.5,
        textColor=tc, leading=13
    ))]]
    t = Table(data, colWidths=[6.8*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('BOX', (0,0), (-1,-1), 0.5, tc),
        ('ROUNDRECT', (0,0), (-1,-1), 0.5, tc),
    ]))
    return KeepTogether([t, Spacer(1, 5)])


def metrics_table(data_rows, col_widths, styles):
    t = Table(data_rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('TEXTCOLOR', (0,1), (-1,-1), DARK_BG),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
        ('BOX', (0,0), (-1,-1), 0.5, MID_GRAY),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.HexColor('#cbd5e1')),
    ]))
    return KeepTogether([t, Spacer(1, 6)])


def generate_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        title="Pitch Insight - Interview Preparation Guide",
        author="Pitch Insight"
    )

    styles = build_styles()
    story = []

    # ─── COVER PAGE ────────────────────────────────────────────────────────────
    cover_data = [[
        Paragraph("🏏 PITCH INSIGHT", styles['cover_title']),
    ]]
    cover_table = Table(cover_data, colWidths=[6.8*inch])
    cover_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), DARK_BG),
        ('TOPPADDING', (0,0), (-1,-1), 30),
        ('BOTTOMPADDING', (0,0), (-1,-1), 20),
        ('LEFTPADDING', (0,0), (-1,-1), 20),
        ('RIGHTPADDING', (0,0), (-1,-1), 20),
        ('LINEBELOW', (0,0), (-1,-1), 3, PRIMARY),
    ]))
    story.append(cover_table)

    sub_data = [[
        Paragraph("Interview Preparation Guide", styles['cover_subtitle']),
    ],[
        Paragraph("AI-Powered Cricket Pitch Analyzer", styles['cover_meta']),
    ],[
        Paragraph("Full-Stack • Deep Learning • Computer Vision • RESTful API", styles['cover_meta']),
    ]]
    sub_table = Table(sub_data, colWidths=[6.8*inch])
    sub_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), CARD_BG),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 20),
        ('RIGHTPADDING', (0,0), (-1,-1), 20),
        ('LINEBELOW', (0,-1), (-1,-1), 1, PRIMARY),
    ]))
    story.append(sub_table)
    story.append(Spacer(1, 16))

    # Quick Stats Banner
    stats_data = [
        [Paragraph("<b>91.6%</b>\nAccuracy", ParagraphStyle('stat', fontName='Helvetica-Bold',
            fontSize=12, textColor=PRIMARY, alignment=TA_CENTER, leading=16)),
         Paragraph("<b>2,585</b>\nTraining Images", ParagraphStyle('stat2', fontName='Helvetica-Bold',
            fontSize=12, textColor=SECONDARY, alignment=TA_CENTER, leading=16)),
         Paragraph("<b>4</b>\nPitch Classes", ParagraphStyle('stat3', fontName='Helvetica-Bold',
            fontSize=12, textColor=ACCENT_ORANGE, alignment=TA_CENTER, leading=16)),
         Paragraph("<b>6</b>\nCV Features", ParagraphStyle('stat4', fontName='Helvetica-Bold',
            fontSize=12, textColor=colors.HexColor('#a855f7'), alignment=TA_CENTER, leading=16)),
         Paragraph("<b>2-4s</b>\nProcess Time", ParagraphStyle('stat5', fontName='Helvetica-Bold',
            fontSize=12, textColor=ACCENT_RED, alignment=TA_CENTER, leading=16)),
        ]
    ]
    stats_table = Table(stats_data, colWidths=[1.36*inch]*5)
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), DARK_BG),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#334155')),
        ('BOX', (0,0), (-1,-1), 1, PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 14))

    # Tech Stack Banner
    tech_data = [
        [Paragraph("BACKEND", ParagraphStyle('tech_hdr', fontName='Helvetica-Bold', fontSize=8,
            textColor=WHITE, alignment=TA_CENTER)),
         Paragraph("FRONTEND", ParagraphStyle('tech_hdr2', fontName='Helvetica-Bold', fontSize=8,
            textColor=WHITE, alignment=TA_CENTER)),
         Paragraph("ML / AI", ParagraphStyle('tech_hdr3', fontName='Helvetica-Bold', fontSize=8,
            textColor=WHITE, alignment=TA_CENTER)),
         Paragraph("DATABASE", ParagraphStyle('tech_hdr4', fontName='Helvetica-Bold', fontSize=8,
            textColor=WHITE, alignment=TA_CENTER)),
        ],
        [Paragraph("FastAPI · Python\nUvicorn · Pydantic\nbcrypt · JWT", ParagraphStyle('tech_val',
            fontName='Helvetica', fontSize=8, textColor=LIGHT_GRAY, alignment=TA_CENTER, leading=12)),
         Paragraph("React 18 · Vite\nAxios · Lucide\nCSS3 Animations", ParagraphStyle('tech_val2',
            fontName='Helvetica', fontSize=8, textColor=LIGHT_GRAY, alignment=TA_CENTER, leading=12)),
         Paragraph("YOLOv8 · ResNet18\nOpenCV · PyTorch\nONNX Runtime", ParagraphStyle('tech_val3',
            fontName='Helvetica', fontSize=8, textColor=LIGHT_GRAY, alignment=TA_CENTER, leading=12)),
         Paragraph("MongoDB Atlas\nOpenWeatherMap\nRazorpay", ParagraphStyle('tech_val4',
            fontName='Helvetica', fontSize=8, textColor=LIGHT_GRAY, alignment=TA_CENTER, leading=12)),
        ]
    ]
    tech_table = Table(tech_data, colWidths=[1.7*inch]*4)
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor('#064e3b')),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor('#1e3a5f')),
        ('BACKGROUND', (2,0), (2,0), colors.HexColor('#713f12')),
        ('BACKGROUND', (3,0), (3,0), colors.HexColor('#4c1d95')),
        ('BACKGROUND', (0,1), (-1,-1), CARD_BG),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('BOX', (0,0), (-1,-1), 1, MID_GRAY),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#334155')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(tech_table)
    story.append(Spacer(1, 16))

    # Date & prep note
    date_str = datetime.datetime.now().strftime("%B %d, %Y")
    story.append(Paragraph(
        f"Prepared on: {date_str}  |  Covers: Technical, ML/AI, System Design, Behavioral Questions",
        ParagraphStyle('footer_note', fontName='Helvetica', fontSize=8,
            textColor=MID_GRAY, alignment=TA_CENTER)
    ))
    story.append(PageBreak())

    # ─── TABLE OF CONTENTS ─────────────────────────────────────────────────────
    story.append(section_header("TABLE OF CONTENTS", styles))
    toc_items = [
        ("1", "Project Overview & Problem Statement", "Questions 1-6"),
        ("2", "System Architecture & Design", "Questions 7-12"),
        ("3", "Machine Learning Pipeline", "Questions 13-22"),
        ("4", "Computer Vision & Feature Extraction", "Questions 23-30"),
        ("5", "Backend — FastAPI & Python", "Questions 31-40"),
        ("6", "Frontend — React & Vite", "Questions 41-48"),
        ("7", "Database & Authentication", "Questions 49-54"),
        ("8", "API Design & Integration", "Questions 55-60"),
        ("9", "Deployment & DevOps", "Questions 61-65"),
        ("10", "Challenges, Results & Future Work", "Questions 66-72"),
        ("11", "Behavioral & HR Questions", "Questions 73-80"),
        ("12", "Quick Reference Cheat Sheet", "—"),
    ]
    toc_data = [["#", "Section", "Questions"]]
    for num, title, qrange in toc_items:
        toc_data.append([num, title, qrange])
    toc_table = Table(toc_data, colWidths=[0.4*inch, 4.8*inch, 1.6*inch])
    toc_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), DARK_BG),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('TEXTCOLOR', (0,1), (-1,-1), DARK_BG),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
        ('BOX', (0,0), (-1,-1), 0.5, MID_GRAY),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (0,-1), 'CENTER'),
        ('ALIGN', (2,0), (2,-1), 'CENTER'),
    ]))
    story.append(toc_table)
    story.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 1 — PROJECT OVERVIEW
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(section_header("1. PROJECT OVERVIEW & PROBLEM STATEMENT", styles))

    story.append(qa_card(1,
        "What is Pitch Insight? Explain it in one sentence.",
        ["Pitch Insight is a full-stack AI web application that analyzes cricket pitch images using "
         "deep learning (YOLOv8 + ResNet18) and computer vision (OpenCV) to classify pitch conditions "
         "and generate real-time match strategies with optional weather integration.",
         "→ Key phrase to remember: 'AI-powered cricket pitch analyzer that classifies conditions and generates match strategies.'"],
        styles, LIGHT_BLUE, DARK_BLUE
    ))

    story.append(qa_card(2,
        "What problem does Pitch Insight solve?",
        ["Before a cricket match, captains and coaches rely on visual inspection and intuition to assess "
         "pitch conditions — which is subjective and error-prone.",
         "• Pitch Insight makes pitch analysis objective and data-driven.",
         "• It provides quantifiable metrics: grass %, crack count, moisture score, brightness.",
         "• It integrates live weather to give context-aware match strategies.",
         "• Use case: Pre-match analysis for toss decisions, team selection, batting/bowling order."],
        styles, LIGHT_GREEN, DARK_GREEN
    ))

    story.append(qa_card(3,
        "What are the 4 pitch classes your model predicts?",
        ["• <b>Batting Friendly</b> — Even bounce, minimal seam/spin movement. Great for big scores.",
         "• <b>Bowling Friendly</b> — Grass cover aids swing and seam. Bowlers dominate.",
         "• <b>Spin Friendly</b> — Cracks and dryness aid spin bowlers. Turn and variable bounce.",
         "• <b>Seam Friendly</b> — Hard, dry pitch aids lateral seam movement off the pitch.",
         "→ Memory tip: Batting = balance, Bowling = grass/swing, Spin = cracks/dry, Seam = dry/hard."],
        styles, LIGHT_ORANGE, colors.HexColor('#78350f')
    ))

    story.append(qa_card(4,
        "Who are the target users of this application?",
        ["• Cricket team coaches & analysts — for pre-match planning.",
         "• IPL/international team management — toss and team selection decisions.",
         "• Cricket broadcast media — real-time pitch analysis for commentary.",
         "• Fantasy cricket platforms — data-driven player selection advice.",
         "• Cricket academies — educational pitch condition understanding."],
        styles
    ))

    story.append(qa_card(5,
        "What is the overall accuracy of your model, and is it production-ready?",
        ["• <b>Test Accuracy: 91.6%</b> | <b>Validation Accuracy: 91.84%</b>",
         "• Trained on 2,585 labeled cricket pitch images (70/20/10 split).",
         "• Trained for 30 epochs on Kaggle (GPU T4), took ~20 minutes.",
         "• The gap between validation and test accuracy is minimal (<0.3%), showing good generalization.",
         "• Yes, it is production-ready: ONNX models are deployed for fast cross-platform inference."],
        styles, LIGHT_GREEN, DARK_GREEN
    ))

    story.append(qa_card(6,
        "What are the key features of the application?",
        ["• AI pitch classification with 91.6% accuracy",
         "• Real-time pitch region detection using YOLOv8",
         "• 6-feature computer vision analysis (grass, cracks, moisture, color, texture, brightness)",
         "• Weather integration via OpenWeatherMap API",
         "• Cricket domain rule engine with 6 expert-defined rules",
         "• Match strategy generator (toss, batting, bowling, team composition)",
         "• User authentication with JWT + bcrypt",
         "• Subscription management with Razorpay payment gateway",
         "• AI cricket chatbot powered by Google Gemini",
         "• Analysis history saved to MongoDB"],
        styles
    ))

    story.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 2 — SYSTEM ARCHITECTURE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(section_header("2. SYSTEM ARCHITECTURE & DESIGN", styles))

    story.append(qa_card(7,
        "Describe the high-level architecture of Pitch Insight.",
        ["The application follows a 3-tier client-server architecture:",
         "• <b>Presentation Layer</b>: React 18 + Vite frontend (port 3000)",
         "• <b>Application Layer</b>: FastAPI + Uvicorn backend (port 8000)",
         "• <b>Data/AI Layer</b>: ONNX ML models + MongoDB + External APIs",
         "Communication: REST API with JSON over HTTP/HTTPS.",
         "Frontend sends image as multipart/form-data → Backend processes through ML pipeline → Returns JSON."],
        styles, LIGHT_BLUE, DARK_BLUE
    ))

    story.append(qa_card(8,
        "Why did you choose FastAPI over Flask or Django?",
        ["• <b>Async Support</b>: FastAPI supports async/await natively, crucial for I/O-bound tasks (weather API calls).",
         "• <b>Automatic Docs</b>: Swagger UI at /docs and ReDoc at /redoc auto-generated from code.",
         "• <b>Pydantic Validation</b>: Strong type checking on request/response models, reduces bugs.",
         "• <b>Performance</b>: FastAPI is one of the fastest Python frameworks (comparable to Node.js).",
         "• <b>Modern Python</b>: Uses Python 3.10+ features, type hints, dependency injection.",
         "→ Flask: simpler but no automatic validation. Django: too heavy for API-only use."],
        styles
    ))

    story.append(qa_card(9,
        "Why did you use ONNX format for model deployment instead of PyTorch .pth?",
        ["• <b>Speed</b>: ONNX Runtime is typically 2-3x faster than PyTorch for inference.",
         "• <b>No framework dependency</b>: ONNX runs without PyTorch installed, smaller container.",
         "• <b>Cross-platform</b>: Same ONNX model runs on CPU, GPU, mobile, edge devices.",
         "• <b>Memory efficiency</b>: Lower RAM usage — important for free-tier cloud deployments.",
         "• Sizes: PyTorch .pth = 11 MB, ONNX = 44 MB (larger but faster at runtime).",
         "→ In production, ONNX classifier + ONNX YOLO run on CPU without PyTorch overhead."],
        styles, LIGHT_GREEN, DARK_GREEN
    ))

    story.append(qa_card(10,
        "How does the data flow from user uploading an image to seeing results?",
        ["1. User uploads pitch image (JPEG/PNG, max 5MB) via drag-drop or file picker.",
         "2. React creates FormData with image + optional city/lat/long parameters.",
         "3. Axios sends POST /api/analyze (multipart/form-data) to FastAPI backend.",
         "4. FastAPI validates file, saves to temp file, extracts form params.",
         "5. YOLO model detects pitch region → crops to bounding box.",
         "6. OpenCV extracts 6 features from cropped pitch image.",
         "7. ResNet18 classifies cropped image → 4 class probabilities.",
         "8. Rule engine applies 6 cricket domain rules → adjusts probabilities.",
         "9. (Optional) OpenWeatherMap API called if weather enabled.",
         "10. Strategy generator creates toss, batting, bowling recommendations.",
         "11. JSON response sent back → React renders results with animations."],
        styles, LIGHT_ORANGE, colors.HexColor('#78350f')
    ))

    story.append(qa_card(11,
        "What design patterns did you use in the backend?",
        ["• <b>Singleton Pattern</b>: MongoDB connection reused across requests.",
         "• <b>Factory Pattern</b>: CompletePitchPipeline creates and manages model instances.",
         "• <b>Dependency Injection</b>: FastAPI's Depends() for auth, DB access.",
         "• <b>Router Pattern</b>: Modular routers (auth, analysis, chat, subscription, weather).",
         "• <b>Lifespan Events</b>: Graceful startup (load models) and shutdown (close DB).",
         "• <b>Repository Pattern</b>: database.py abstracts MongoDB operations."],
        styles
    ))

    story.append(qa_card(12,
        "How does the application handle errors?",
        ["• <b>HTTPException</b>: FastAPI's built-in exception with status codes (400, 401, 500, 503).",
         "• <b>Try-Catch blocks</b>: All ML inference wrapped in try-except.",
         "• <b>Temp file cleanup</b>: Finally blocks ensure temp images are deleted even on error.",
         "• <b>Axios interceptors</b>: Frontend automatically redirects to login on 401.",
         "• <b>User-friendly messages</b>: Generic error messages shown to users, details logged.",
         "• Common errors: 400 (not an image), 500 (ML failure), 503 (weather API down)."],
        styles, LIGHT_BLUE, DARK_BLUE
    ))

    story.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 3 — MACHINE LEARNING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(section_header("3. MACHINE LEARNING PIPELINE", styles))

    story.append(qa_card(13,
        "What models does Pitch Insight use and what does each do?",
        ["<b>Model 1 — YOLOv8 (pitch_yolov8_best.onnx)</b>",
         "• Task: Object detection — finds the pitch region in a full cricket ground image.",
         "• Output: Bounding box [x1, y1, x2, y2] with confidence score.",
         "• Architecture: Single-stage detector (one forward pass), 640×640 input.",
         "<b>Model 2 — ResNet18 (pitch_classifier.onnx)</b>",
         "• Task: Image classification — classifies the cropped pitch into 4 types.",
         "• Output: Softmax probabilities for 4 classes.",
         "• Architecture: 18-layer residual network, fine-tuned on cricket pitch dataset.",
         "• Input: 224×224 RGB, ImageNet normalized."],
        styles, LIGHT_BLUE, DARK_BLUE
    ))

    story.append(qa_card(14,
        "Why did you choose ResNet18 over other architectures?",
        ["• <b>Right size</b>: ResNet18 has 11M parameters — efficient for this task.",
         "• <b>Residual connections</b>: Solve vanishing gradient problem, enable deeper training.",
         "• <b>Transfer learning</b>: Pretrained on ImageNet gives excellent visual feature extraction.",
         "• <b>Proven accuracy</b>: Achieves 91.6% on our dataset — sufficient for production.",
         "• <b>Speed</b>: Inference in ~0.5-0.8s on CPU — acceptable latency.",
         "→ Alternatives considered: ResNet50 (more params, slower), EfficientNet (better but complex), MobileNet (faster but less accurate)."],
        styles
    ))

    story.append(qa_card(15,
        "Explain the ResNet18 training configuration.",
        ["• <b>Dataset</b>: 2,585 cricket pitch images across 4 classes.",
         "• <b>Split</b>: 70% train (1,808), 20% validation (515), 10% test (262).",
         "• <b>Optimizer</b>: Adam with lr=0.001.",
         "• <b>Loss</b>: CrossEntropyLoss (standard for multi-class classification).",
         "• <b>Batch size</b>: 32, <b>Epochs</b>: 30.",
         "• <b>Scheduler</b>: ReduceLROnPlateau (patience=3, factor=0.5) — reduces LR when validation loss plateaus.",
         "• <b>Data Augmentation</b>: Random horizontal flip, rotation ±10°, color jitter.",
         "• <b>Platform</b>: Kaggle GPU T4, training time ~20 minutes."],
        styles, LIGHT_GREEN, DARK_GREEN
    ))

    story.append(qa_card(16,
        "What is transfer learning and how did you apply it?",
        ["Transfer learning means taking a model pretrained on a large dataset (ImageNet — 1M+ images, 1000 classes) "
         "and fine-tuning it on your smaller domain-specific dataset.",
         "How we applied it:",
         "• Loaded ResNet18 pretrained weights from torchvision.",
         "• Replaced the final fully-connected layer: 1000 neurons → 4 neurons (our 4 pitch classes).",
         "• Trained all layers end-to-end (fine-tuning, not feature extraction).",
         "• Benefit: The early layers already detect edges, textures, colors — perfect for pitch features.",
         "→ Without transfer learning, would need ~10x more data to achieve same accuracy."],
        styles, LIGHT_ORANGE, colors.HexColor('#78350f')
    ))

    story.append(qa_card(17,
        "What is YOLOv8 and why is it used for pitch detection?",
        ["YOLO = 'You Only Look Once' — a real-time single-stage object detector.",
         "• <b>Single pass</b>: Unlike two-stage detectors (R-CNN), YOLO processes the whole image once.",
         "• <b>Speed</b>: Detects pitch in 0.1–0.2 seconds.",
         "• <b>Accuracy</b>: Confidence threshold of 0.5 ensures reliable detections.",
         "• <b>Custom trained</b>: Fine-tuned on cricket pitch images with bounding box annotations.",
         "Why needed: A full cricket ground image contains stands, players, grass — we need to isolate just the pitch "
         "before classification to avoid noise.",
         "• Output: [x1, y1, x2, y2] bounding box used to crop the pitch region."],
        styles
    ))

    # Model Performance Table
    story.append(Paragraph("Model Performance by Class:", styles['section']))
    perf_data = [
        ["Class", "Precision", "Recall", "F1-Score", "Accuracy", "Support"],
        ["Batting Friendly", "1.00", "0.97", "0.98", "96.77%", "62"],
        ["Bowling Friendly", "0.82", "1.00", "0.90", "100.00%", "84"],
        ["Seam Friendly", "0.96", "0.93", "0.94", "92.73%", "55"],
        ["Spin Friendly", "0.96", "0.74", "0.83", "73.77%", "61"],
        ["OVERALL", "—", "—", "—", "91.60%", "262"],
    ]
    story.append(metrics_table(perf_data,
        [1.5*inch, 0.9*inch, 0.8*inch, 0.9*inch, 0.9*inch, 0.8*inch], styles))

    story.append(qa_card(18,
        "Spin-friendly accuracy is only 73.77%. Why? How would you fix it?",
        ["<b>Root Cause</b>: Spin-friendly pitches visually resemble seam-friendly pitches "
         "(both appear dry/brown). The model confuses them.",
         "<b>Current mitigations</b>:",
         "• Rule engine adds +20% to spin probability when High cracks detected.",
         "• +15% when moisture is Very Dry AND cracks are High.",
         "<b>Future fixes</b>:",
         "• Collect more spin-friendly images (currently 602, lowest along with batting).",
         "• Add crack density as an additional input feature to the classifier.",
         "• Train a binary 'spin vs seam' classifier as a second stage.",
         "• Use class-weighted loss: give spin class higher weight during training."],
        styles, LIGHT_ORANGE, colors.HexColor('#78350f')
    ))

    story.append(qa_card(19,
        "Explain softmax and how it's used in your classification.",
        ["Softmax converts raw model outputs (logits) into probabilities that sum to 1.",
         "Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)",
         "Example output from ResNet18:",
         "• Logits: [1.2, 0.3, 0.8, 3.1]",
         "• After softmax: [batting=5.2%, bowling=8.1%, seam=4.9%, spin=81.8%]",
         "• The model returns spin_friendly with 81.8% confidence.",
         "• Then the rule engine adjusts: if cracks are present → spin += 2% → final 83.7%."],
        styles
    ))

    story.append(qa_card(20,
        "What data augmentation did you use and why?",
        ["• <b>Random Horizontal Flip</b>: Pitch looks same from both ends → doubles effective data.",
         "• <b>Random Rotation ±10°</b>: Camera angle varies in real photos.",
         "• <b>Color Jitter</b> (brightness=0.2, contrast=0.2): Lighting varies by time of day/weather.",
         "• <b>Normalization</b>: ImageNet mean [0.485, 0.456, 0.406] / std [0.229, 0.224, 0.225].",
         "→ Augmentation prevents overfitting by making the model learn features, not memorize images.",
         "→ We did NOT use vertical flip (pitches have a clear orientation)."],
        styles, LIGHT_GREEN, DARK_GREEN
    ))

    story.append(qa_card(21,
        "What is the cricket domain rule engine and why is it needed?",
        ["Pure ML sometimes misclassifies when lighting or angle is unusual. The rule engine adds "
         "cricket expertise as a post-processing correction.",
         "6 Rules (applied after ML inference):",
         "• <b>Rule 1</b>: Grass > 60% → Bowling friendly += 15%  (swing bowling conditions)",
         "• <b>Rule 2</b>: High cracks → Spin += 20%; Medium cracks → Spin += 12%",
         "• <b>Rule 3</b>: Very Dry + Grass < 20% → Seam += 15%",
         "• <b>Rule 4</b>: No/Low cracks + Grass 20-50% → Batting += 10%",
         "• <b>Rule 5</b>: Very Dry + High cracks → Spin += 15%  (extreme spin)",
         "• <b>Rule 6</b>: Wet/Damp + Grass > 50% → Bowling += 12%  (humid + grass = swing)",
         "After rules: re-normalize all 4 probabilities to sum to 100%."],
        styles, LIGHT_BLUE, DARK_BLUE
    ))

    story.append(qa_card(22,
        "How do you prevent the model from overfitting?",
        ["• <b>Train/Val/Test split</b>: 70/20/10 — separate test set untouched until final evaluation.",
         "• <b>Data augmentation</b>: Random flips, rotations, color jitter.",
         "• <b>ReduceLROnPlateau</b>: Reduces learning rate when validation loss stops improving.",
         "• <b>Early stopping logic</b>: Model saved at best validation accuracy (not last epoch).",
         "• <b>Transfer learning</b>: Pretrained weights provide strong regularization via feature reuse.",
         "• Evidence of no overfitting: Val accuracy (91.84%) ≈ Test accuracy (91.60%)."],
        styles
    ))

    story.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 4 — COMPUTER VISION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(section_header("4. COMPUTER VISION & FEATURE EXTRACTION (OpenCV)", styles))

    story.append(qa_card(23,
        "What are the 6 features extracted by OpenCV and what cricket insight does each provide?",
        ["<b>1. Grass Coverage (HSV filtering)</b>",
         "   • How: Convert BGR→HSV, mask green pixels (H:25-85, S:40-255, V:40-255).",
         "   • Output: % of pitch that is green. Levels: High/Medium/Low/Minimal.",
         "   • Cricket: More grass = more swing and seam movement for bowlers.",
         "<b>2. Crack Detection (Canny edges)</b>",
         "   • How: Grayscale → Gaussian blur → Canny(50,150) → find contours.",
         "   • Filter: length >30px, aspect ratio >3:1. Output: crack count, severity.",
         "   • Cricket: Cracks = spin turn and variable bounce.",
         "<b>3. Moisture Level (Grayscale intensity)</b>",
         "   • How: HSV V-channel mean. Lower brightness = more moisture (0-100 score).",
         "   • Levels: Wet/Damp/Normal/Dry/Very Dry.",
         "   • Cricket: Wet pitch = slow, low bounce. Dry = turn, crack, bounce.",
         "<b>4. Color Profile (K-means clustering, k=3)</b>",
         "   • Types: Brown/Red/Green/Mixed. Indicates pitch preparation type.",
         "   • Cricket: Red/Brown = baked clay (India). Green = grass cover (England/NZ).",
         "<b>5. Texture Analysis (Laplacian variance)</b>",
         "   • How: Laplacian filter measures edge sharpness = roughness.",
         "   • Types: Very Rough/Rough/Moderate/Smooth.",
         "   • Cricket: Rough = unpredictable bounce. Smooth = even pace.",
         "<b>6. Brightness (Mean grayscale value)</b>",
         "   • 5 levels: Very Dark to Very Bright (0-255 range).",
         "   • Cricket: Brightness correlates with pitch hardness and moisture."],
        styles, LIGHT_ORANGE, colors.HexColor('#78350f')
    ))

    story.append(qa_card(24,
        "What is HSV color space and why use it over RGB for grass detection?",
        ["HSV = Hue, Saturation, Value. RGB = Red, Green, Blue.",
         "• <b>Hue</b> encodes the color type (0-180 in OpenCV). Green is H: 25-85.",
         "• <b>Saturation</b> encodes color intensity.",
         "• <b>Value</b> encodes brightness.",
         "Why HSV over RGB:",
         "• In RGB, green looks different under bright sun vs shade (all 3 channels change).",
         "• In HSV, the Hue stays constant even as brightness changes.",
         "• Simple thresholding: cv2.inRange(hsv, lower_green, upper_green) → binary mask.",
         "• Result: Robust grass detection across different lighting conditions.",
         "→ Code: hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)"],
        styles
    ))

    story.append(qa_card(25,
        "Explain Canny edge detection and how it finds cracks.",
        ["Canny is a multi-step edge detection algorithm:",
         "1. <b>Grayscale</b>: Convert color image to single-channel.",
         "2. <b>Gaussian Blur</b>: 5×5 kernel — removes noise that would create false edges.",
         "3. <b>Gradient calculation</b>: Sobel filters find intensity changes (edges).",
         "4. <b>Non-maximum suppression</b>: Thins edges to 1-pixel width.",
         "5. <b>Double threshold</b>: Strong edges (>150), weak edges (50-150), no edges (<50).",
         "6. <b>Edge tracking</b>: Weak edges kept if connected to strong edges.",
         "Crack filtering: We keep contours where length>30px AND aspect ratio>3:1 (elongated = crack).",
         "→ Short, round blobs are ignored. Long, thin shapes = cracks."],
        styles, LIGHT_BLUE, DARK_BLUE
    ))

    story.append(qa_card(26,
        "What is K-means clustering and how is it used for color analysis?",
        ["K-means clustering partitions pixels into K groups by minimizing within-cluster variance.",
         "Application in Pitch Insight:",
         "• K=3: Find the 3 dominant colors in the pitch image.",
         "• Each pixel is assigned to nearest color centroid.",
         "• Iteration: centroids updated until convergence.",
         "• Result: Top 3 dominant colors + their percentages.",
         "Then classify: If dominant color is green → 'Green pitch'. If brown/red → 'Brown/Red pitch'.",
         "Cricket use: Green pitch = England-style, Brown = Indian subcontinent, Red = worn clay.",
         "→ cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)"],
        styles
    ))

    story.append(qa_card(27,
        "How does the Laplacian variance measure texture/roughness?",
        ["The Laplacian is a second-order derivative filter that highlights rapid intensity changes.",
         "• Formula: L = d²I/dx² + d²I/dy² (sum of second derivatives).",
         "• High Laplacian variance = lots of edges = rough texture.",
         "• Low Laplacian variance = smooth surface = fewer edges.",
         "• We compute: var = cv2.Laplacian(gray_image, cv2.CV_64F).var()",
         "Thresholds used:",
         "• variance > 500 → Very Rough",
         "• variance > 200 → Rough",
         "• variance > 100 → Moderate",
         "• variance ≤ 100 → Smooth",
         "Cricket: Rough pitch = ball grips more, variable bounce (helps spinners and medium-pacers)."],
        styles
    ))

    story.append(qa_card(28,
        "Why do you crop the pitch region before classification?",
        ["Without YOLO cropping, the ResNet18 would see the full cricket ground image:",
         "• Includes: spectator stands, outfield grass, advertising boards, umpires.",
         "• These are noise — not relevant to pitch condition.",
         "• The pitch occupies roughly 10-20% of the full image.",
         "Benefits of cropping:",
         "• ResNet18 focuses only on relevant pitch pixels.",
         "• Eliminates false signals from surrounding environment.",
         "• Improves accuracy significantly.",
         "• YOLO confidence threshold of 0.5 ensures only reliable detections are used.",
         "Fallback: If YOLO doesn't detect a pitch (confidence < 0.5), the full image is used."],
        styles, LIGHT_GREEN, DARK_GREEN
    ))

    story.append(qa_card(29,
        "How does moisture analysis work technically?",
        ["Moisture analysis uses the HSV V-channel (Value = brightness):",
         "• Wet surfaces appear darker (absorb more light).",
         "• Dry surfaces appear brighter (reflect more light).",
         "Steps:",
         "1. Convert image to HSV.",
         "2. Extract V-channel (index 2).",
         "3. Calculate mean brightness: mean_v = np.mean(v_channel).",
         "4. Moisture score = 100 - (mean_v / 255 * 100).",
         "5. Levels: Wet (score>60), Damp (30-60), Normal (20-30), Dry (10-20), Very Dry (<10).",
         "Cricket: Damp pitch = ball moves slower, batsmen comfortable. Very Dry = fast-turning."],
        styles
    ))

    story.append(qa_card(30,
        "How does weather impact the pitch analysis?",
        ["Weather data (from OpenWeatherMap API) adds context to pitch analysis:",
         "• <b>High Humidity (>70%)</b>: Ball swings more in air. Impact: Favors bowling.",
         "• <b>Recent Rainfall</b>: Wet outfield, slower pitch. Impact: Reduces seam movement.",
         "• <b>High Temperature (>35°C)</b>: Pitch dries faster. Impact: More spin later.",
         "• <b>Strong Wind</b>: Ball movement in the air. Impact: Helps medium-fast bowlers.",
         "Weather data: temperature, humidity, wind_speed, rainfall, conditions (string).",
         "Severity: low/medium/high based on combination of factors.",
         "→ The strategy generator combines pitch type + weather to give integrated recommendations."],
        styles, LIGHT_ORANGE, colors.HexColor('#78350f')
    ))

    story.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 5 — BACKEND
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(section_header("5. BACKEND — FastAPI & PYTHON", styles))

    story.append(qa_card(31,
        "Explain the backend project structure and the role of each file.",
        ["• <b>app.py</b>: FastAPI initialization, middleware, lifespan events, router registration.",
         "• <b>config.py</b>: Environment variables loaded from .env file (API keys, DB URL, secrets).",
         "• <b>database.py</b>: MongoDB connection management (singleton pattern).",
         "• <b>models.py</b>: Pydantic data models (User, Analysis, Subscription).",
         "• <b>schemas.py</b>: API request/response schemas for validation.",
         "• <b>auth.py</b>: JWT token creation/verification, bcrypt password hashing.",
         "• <b>complete_pipeline_onnx.py</b>: ONNX ML inference pipeline (YOLO + ResNet18).",
         "• <b>pitch_analyzer.py</b>: OpenCV feature extraction (PitchAnalyzer class).",
         "• <b>weather_forecast_analyzer.py</b>: OpenWeatherMap integration.",
         "• <b>razorpay_handler.py</b>: Payment order creation and verification.",
         "• <b>routes/</b>: Separate router files — auth, analysis, chat, subscription, weather, health."],
        styles, LIGHT_BLUE, DARK_BLUE
    ))

    story.append(qa_card(32,
        "How does JWT authentication work in your application?",
        ["JWT = JSON Web Token. Used for stateless authentication.",
         "Flow:",
         "1. User registers → password hashed with bcrypt → stored in MongoDB.",
         "2. User logs in → bcrypt verifies password → JWT token created with user_id + expiry.",
         "3. Token returned to frontend → stored in localStorage.",
         "4. Every subsequent request: Axios adds 'Authorization: Bearer <token>' header.",
         "5. FastAPI's get_current_user() dependency decodes token → fetches user from DB.",
         "6. If token expired/invalid → 401 Unauthorized returned.",
         "Token: Header.Payload.Signature (base64 encoded, signed with SECRET_KEY).",
         "Expiry: 7 days (configurable)."],
        styles
    ))

    story.append(qa_card(33,
        "What is bcrypt and why is it used for password storage?",
        ["bcrypt is a password hashing function specifically designed for security:",
         "• <b>Slow by design</b>: Takes 100ms+ to hash — makes brute force attacks impractical.",
         "• <b>Salt included</b>: Each hash includes a random salt — no two hashes are same even for same password.",
         "• <b>Cost factor</b>: Can increase difficulty as hardware gets faster.",
         "• <b>One-way</b>: Cannot reverse the hash to get the original password.",
         "Usage: pwd_context = CryptContext(schemes=['bcrypt'])",
         "→ pwd_context.hash(password) to hash.",
         "→ pwd_context.verify(plain_password, hashed) to verify.",
         "Never store plain text passwords. bcrypt is the industry standard."],
        styles, LIGHT_GREEN, DARK_GREEN
    ))

    story.append(qa_card(34,
        "Explain CORS and how it's configured in your application.",
        ["CORS = Cross-Origin Resource Sharing. Browser security policy preventing requests between different domains.",
         "Problem: React app (localhost:3000) calls FastAPI (localhost:8000) → different ports = different 'origin'.",
         "Solution: FastAPI CORS middleware tells browser it's OK.",
         "Configuration:",
         "• allow_origins: ['http://localhost:3000', 'https://yourdomain.com']",
         "• allow_credentials: True (needed for cookies/auth headers)",
         "• allow_methods: ['*'] (GET, POST, PUT, DELETE)",
         "• allow_headers: ['*'] (Content-Type, Authorization)",
         "In production: Replace '*' with specific domains for security."],
        styles
    ))

    story.append(qa_card(35,
        "How does file upload work with multipart/form-data?",
        ["FastAPI handles file uploads via UploadFile type:",
         "• Frontend creates FormData: formData.append('image', file)",
         "• Content-Type: multipart/form-data (set automatically by browser)",
         "• Backend: @app.post('/api/analyze') with image: UploadFile = File(...)",
         "Processing steps:",
         "1. Validate file type: image.content_type.startswith('image/')",
         "2. Save to temp file: with tempfile.NamedTemporaryFile() as tmp: ...",
         "3. Write bytes: tmp.write(await image.read())",
         "4. Pass temp path to ML pipeline.",
         "5. Delete temp file in finally block.",
         "→ python-multipart package required for FastAPI to handle multipart."],
        styles, LIGHT_ORANGE, colors.HexColor('#78350f')
    ))

    story.append(qa_card(36,
        "What is Pydantic and how does it help in FastAPI?",
        ["Pydantic is a Python data validation library used heavily by FastAPI.",
         "Uses Python type annotations to validate data automatically.",
         "Benefits:",
         "• <b>Automatic validation</b>: Wrong type → 422 Unprocessable Entity response.",
         "• <b>Serialization</b>: Python objects → JSON automatically.",
         "• <b>Documentation</b>: FastAPI reads Pydantic models to generate Swagger docs.",
         "• <b>IDE support</b>: Type hints enable autocomplete.",
         "Example: class AnalysisResult(BaseModel): prediction: str; confidence: float",
         "→ FastAPI uses this for both input (request body) and output (response)."],
        styles
    ))

    story.append(qa_card(37,
        "How do you manage environment variables and secrets?",
        ["• All secrets stored in .env file (never committed to Git — listed in .gitignore).",
         "• .env.example provided with placeholder values for documentation.",
         "• config.py reads with os.getenv() and python-dotenv's load_dotenv().",
         "Key variables: MONGODB_URL, SECRET_KEY, WEATHER_API_KEY, GEMINI_API_KEY,",
         "               RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET.",
         "In production (Render/Railway): Variables set via platform dashboard, not .env file.",
         "→ Never hardcode API keys in source code. Use environment variables always."],
        styles, LIGHT_BLUE, DARK_BLUE
    ))

    story.append(qa_card(38,
        "Explain the subscription/payment flow with Razorpay.",
        ["Razorpay is a payment gateway popular in India.",
         "Flow:",
         "1. User clicks 'Subscribe' → Frontend calls POST /api/subscription/create-order.",
         "2. Backend calls Razorpay API → creates order with amount, currency.",
         "3. Razorpay order_id returned to frontend.",
         "4. Frontend opens Razorpay checkout modal with order_id.",
         "5. User pays → Razorpay calls webhook / returns payment_id to frontend.",
         "6. Frontend sends payment_id + signature to POST /api/subscription/verify-payment.",
         "7. Backend verifies signature using HMAC-SHA256 with Razorpay secret.",
         "8. If valid: Update user's subscription in MongoDB → return success.",
         "Plans: Monthly (₹199/30 days), Yearly (₹1999/365 days)."],
        styles
    ))

    story.append(qa_card(39,
        "What is the Gemini AI chatbot and how does it work?",
        ["The chatbot uses Google Gemini API for conversational AI about cricket.",
         "• User asks questions like: 'What strategy should I use on this pitch?'",
         "• Frontend sends message + current analysis_id to POST /api/chat.",
         "• Backend fetches the analysis result from MongoDB.",
         "• Constructs a context prompt: 'The pitch is spin_friendly with 83% confidence...'",
         "• Sends to Gemini API with cricket-specific system prompt.",
         "• Gemini returns contextual advice.",
         "• Response streamed back to frontend ChatWidget component.",
         "Use case: Natural language Q&A about the specific pitch that was analyzed."],
        styles, LIGHT_GREEN, DARK_GREEN
    ))

    story.append(qa_card(40,
        "How is analysis history stored and retrieved?",
        ["Each analysis is saved to MongoDB 'analyses' collection (for logged-in users).",
         "Document structure: {user_id, analysis_id, pitch_type, confidence, features, weather, strategy, timestamp}",
         "• GET /api/analysis/history → returns all analyses for current user.",
         "• GET /api/analysis/{analysis_id} → returns specific analysis.",
         "Frontend HistorySection component displays past analyses in a timeline.",
         "Anonymous users: Analysis not saved (guest mode).",
         "Premium users: Unlimited history. Free users: Last 10 analyses."],
        styles
    ))

    story.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 6 — FRONTEND
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(section_header("6. FRONTEND — REACT & VITE", styles))

    story.append(qa_card(41,
        "Why did you choose React + Vite over other frameworks?",
        ["<b>React</b>:",
         "• Component-based architecture: reusable, maintainable UI.",
         "• Virtual DOM: efficient re-rendering only changed components.",
         "• Hooks: useState, useEffect for clean state management without classes.",
         "• Huge ecosystem: Lucide icons, Axios, extensive community support.",
         "<b>Vite</b>:",
         "• Lightning-fast HMR (Hot Module Replacement): instant dev feedback.",
         "• Native ES modules: no bundling during development.",
         "• Faster build than Create React App / Webpack.",
         "• Built-in env variable handling (VITE_API_URL)."],
        styles, LIGHT_BLUE, DARK_BLUE
    ))

    story.append(qa_card(42,
        "Explain the React component architecture of the application.",
        ["App Component (root) manages global state:",
         "• States: currentPage, user, token, theme.",
         "• Renders: Header + current page + Footer.",
         "Components:",
         "• <b>Header.jsx</b>: Navigation, auth buttons, theme toggle.",
         "• <b>UploadSection.jsx</b>: Drag-drop upload, weather toggle, analyze button.",
         "• <b>ResultsSection.jsx</b>: Pitch type card, probabilities chart, features grid, strategy.",
         "• <b>Auth.jsx</b>: Login/Signup modal with form validation.",
         "• <b>ChatWidget.jsx</b>: Floating chat interface for Gemini AI.",
         "• <b>PaymentModal.jsx</b>: Razorpay subscription UI.",
         "• <b>HistorySection.jsx</b>: Past analysis timeline.",
         "• <b>Footer.jsx</b>: Links to GitHub, API docs."],
        styles
    ))

    story.append(qa_card(43,
        "How does the image upload and preview work?",
        ["• File input (hidden): <input type='file' accept='image/*' />",
         "• Drag-drop area: onClick triggers fileInput.click().",
         "• File validation: Checks file.type.startsWith('image/') and file.size < 5MB.",
         "• Preview: URL.createObjectURL(file) creates a temporary local URL.",
         "• img src={preview} displays the image instantly (no server round-trip).",
         "• FormData: formData.append('image', file) prepares for API upload.",
         "→ URL.createObjectURL() is revoked after component unmounts to free memory."],
        styles, LIGHT_GREEN, DARK_GREEN
    ))

    story.append(qa_card(44,
        "How does Axios work and what are interceptors?",
        ["Axios is a promise-based HTTP client for JavaScript.",
         "• Creates API calls: axios.post('/api/analyze', formData, config)",
         "• Handles JSON serialization/deserialization automatically.",
         "<b>Interceptors</b>: Middleware that runs before/after every request.",
         "Request Interceptor: Reads JWT token from localStorage → adds to Authorization header.",
         "  config.headers.Authorization = `Bearer ${token}`",
         "Response Interceptor: If 401 Unauthorized → clear localStorage → redirect to login.",
         "→ Interceptors prevent writing auth header logic in every API call — DRY principle."],
        styles
    ))

    story.append(qa_card(45,
        "Explain the state management approach in your React app.",
        ["Using React's built-in state management (no Redux needed for this scale):",
         "• <b>useState</b>: Component-level state (result, loading, error, user, token).",
         "• <b>useEffect</b>: Side effects — load token from localStorage on mount.",
         "• <b>Props drilling</b>: User and token passed from App → child components.",
         "• <b>localStorage</b>: Persist auth state across page refreshes.",
         "State flow: User logs in → setUser(data) + setToken(token) → localStorage.setItem()",
         "On refresh: useEffect reads localStorage → restores auth state.",
         "When to use Redux: Only needed when 5+ components share the same state — not needed here."],
        styles, LIGHT_ORANGE, colors.HexColor('#78350f')
    ))

    story.append(qa_card(46,
        "How did you implement the dark theme and animations?",
        ["<b>Dark Theme</b>:",
         "• CSS variables (custom properties) in index.css:",
         "  --primary: #10b981, --background: #0f172a, --surface: #1e293b",
         "• Glassmorphism: backdrop-filter: blur(10px) on card components.",
         "<b>Animations</b>:",
         "• fadeIn: opacity 0→1 + translateY(20px→0)",
         "• slideInLeft/Right: translateX(-50px→0) / (50px→0)",
         "• pulse: scale(1)→scale(1.05)→scale(1) (heartbeat on loading icons)",
         "• spin: 360° rotation for loading spinners",
         "• Staggered delays: animation-delay: 0.1s, 0.2s, 0.3s for sequential appearance.",
         "→ All animations use CSS @keyframes, no JavaScript animation libraries."],
        styles
    ))

    story.append(qa_card(47,
        "How is the application made responsive for mobile?",
        ["Mobile-first CSS approach with media query breakpoints:",
         "• Desktop: 1024px+ — multi-column grid layout.",
         "• Tablet: 768px-1023px — 2-column grid.",
         "• Mobile: <768px — single column stack.",
         "Techniques:",
         "• CSS Grid with grid-template-columns: repeat(auto-fit, minmax(250px, 1fr))",
         "• Flexbox for row/column switching at breakpoints.",
         "• Reduced padding/font sizes on mobile.",
         "• Touch-friendly buttons (min-height: 44px for tap targets).",
         "• Drag-drop area works with touch events on mobile."],
        styles
    ))

    story.append(qa_card(48,
        "How is the environment/API URL configured for different environments?",
        ["Vite uses environment variable files:",
         "• Development: VITE_API_URL=http://localhost:8000 (in .env.local)",
         "• Production: VITE_API_URL=https://pitch-insight-api.onrender.com",
         "Access in code: import.meta.env.VITE_API_URL",
         "Usage in api.js:",
         "  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'",
         "On Vercel: Add VITE_API_URL as an environment variable in dashboard.",
         "→ This allows same codebase to work in development and production."],
        styles, LIGHT_BLUE, DARK_BLUE
    ))

    story.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 7 — DATABASE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(section_header("7. DATABASE & DATA MANAGEMENT", styles))

    story.append(qa_card(49,
        "Why MongoDB over a relational database like PostgreSQL?",
        ["• <b>Schema flexibility</b>: Analysis results have varying fields (some have weather, some don't).",
         "• <b>JSON-native</b>: Analysis results are already JSON — no ORM mapping needed.",
         "• <b>Easy scaling</b>: MongoDB Atlas auto-scales with usage.",
         "• <b>No migrations</b>: Add new fields without altering schema.",
         "• <b>Good for document storage</b>: Each analysis is a self-contained document.",
         "When PostgreSQL would be better: Complex relationships, transactions, strict schema needed.",
         "MongoDB Atlas: Managed cloud MongoDB — free tier available, automatic backups."],
        styles
    ))

    story.append(qa_card(50,
        "What are the MongoDB collections and their structure?",
        ["<b>users collection</b>:",
         "  { _id, email, hashed_password, name, created_at, subscription_status, subscription_expiry }",
         "<b>analyses collection</b>:",
         "  { _id, user_id, analysis_id (string), pitch_type, confidence, features{}, weather{}, strategy{}, timestamp }",
         "<b>Indexing</b>:",
         "  • users: email (unique index) for fast lookup during login.",
         "  • analyses: user_id (index) for fast history retrieval.",
         "  • analyses: analysis_id (unique index) for lookup by ID."],
        styles, LIGHT_GREEN, DARK_GREEN
    ))

    story.append(qa_card(51,
        "Is there a database for the ML models? How are they stored?",
        ["ML models are stored as files on the local filesystem / server:",
         "• pitch_yolov8_best.onnx (12 MB) — YOLO detector",
         "• pitch_classifier.onnx (44 MB) — ResNet18 classifier",
         "Models are loaded into memory once at application startup (lifespan event).",
         "No database needed for models — they are static files that don't change at runtime.",
         "In production (Render): Models are part of the Docker image / repository.",
         "→ Alternative: Store on S3/GCS and download on startup — better for large models."],
        styles
    ))

    story.append(qa_card(52,
        "How are temporary image files managed to avoid disk space issues?",
        ["• Python's tempfile.NamedTemporaryFile() creates a temp file with auto-cleanup.",
         "• Files saved in /tmp directory (Linux/Mac) or OS temp folder (Windows).",
         "• try-finally pattern ensures deletion even if ML pipeline throws an exception:",
         "  try: result = pipeline.analyze(tmp.name)",
         "  finally: os.unlink(tmp.name)  # Always runs",
         "• The stateless design means no permanent image storage — only metadata in MongoDB.",
         "→ For a production system, could add S3 storage if users want to re-analyze images."],
        styles, LIGHT_ORANGE, colors.HexColor('#78350f')
    ))

    story.append(qa_card(53,
        "Explain the singleton database connection pattern.",
        ["A singleton ensures only one MongoDB connection is created and reused:",
         "  mongodb_client: Optional[MongoClient] = None",
         "  def get_database():",
         "      global mongodb_client",
         "      if mongodb_client is None:",
         "          mongodb_client = MongoClient(MONGODB_URL)",
         "      return mongodb_client[DATABASE_NAME]",
         "Benefits:",
         "• Avoids creating new connections for every request (expensive operation).",
         "• Connection pooling: MongoClient maintains a pool of reusable connections.",
         "• Graceful shutdown: close_database_connection() called in lifespan shutdown."],
        styles
    ))

    story.append(qa_card(54,
        "How does subscription status affect the user experience?",
        ["Free users:",
         "• Can analyze 3 pitches per day.",
         "• Last 10 analyses in history.",
         "• Basic strategy recommendations.",
         "Premium users (after Razorpay payment):",
         "• Unlimited analyses.",
         "• Full analysis history.",
         "• Advanced strategy with detailed weather impact.",
         "• AI chatbot access.",
         "Implementation:",
         "• subscription_status field in users collection: 'free' / 'premium'.",
         "• subscription_expiry: datetime — backend checks if subscription is still active.",
         "• Frontend shows UpgradePrompt component when free limit reached."],
        styles, LIGHT_BLUE, DARK_BLUE
    ))

    story.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 8 — API DESIGN
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(section_header("8. API DESIGN & INTEGRATION", styles))

    story.append(Paragraph("Key API Endpoints:", styles['section']))
    api_data = [
        ["Method", "Endpoint", "Auth", "Description"],
        ["GET", "/api/health", "None", "Health check — is server running?"],
        ["GET", "/api/classes", "None", "List pitch classes with descriptions"],
        ["POST", "/api/quick-analyze", "Optional", "Fast classification (0.8-1.5s)"],
        ["POST", "/api/analyze", "Optional", "Full analysis with features + weather (2-4s)"],
        ["GET", "/api/weather", "None", "Fetch weather by lat/lon or city"],
        ["POST", "/api/auth/signup", "None", "Register new user"],
        ["POST", "/api/auth/login", "None", "Login → returns JWT token"],
        ["GET", "/api/auth/me", "JWT", "Get current user profile"],
        ["GET", "/api/analysis/history", "JWT", "User's past analyses"],
        ["POST", "/api/chat", "JWT", "Send message to Gemini chatbot"],
        ["POST", "/api/subscription/create-order", "JWT", "Create Razorpay order"],
        ["POST", "/api/subscription/verify-payment", "JWT", "Verify and activate subscription"],
    ]
    story.append(metrics_table(api_data,
        [0.7*inch, 2.0*inch, 0.7*inch, 3.4*inch], styles))

    story.append(qa_card(55,
        "What is the difference between quick-analyze and complete analyze?",
        ["<b>POST /api/quick-analyze</b>:",
         "• Only YOLO detection + ResNet18 classification.",
         "• No OpenCV features, no weather, no strategy.",
         "• Response time: 0.8-1.5 seconds.",
         "• Use case: When you just need the pitch type fast.",
         "<b>POST /api/analyze</b>:",
         "• Full pipeline: YOLO + OpenCV features + ResNet18 + rules + weather + strategy.",
         "• Response time: 2-4 seconds (4+ with weather API).",
         "• Returns comprehensive JSON with all features and match strategy.",
         "• Use case: Pre-match full analysis."],
        styles
    ))

    story.append(qa_card(56,
        "How is the API documented? How can someone test it?",
        ["FastAPI auto-generates interactive API documentation:",
         "• <b>Swagger UI</b>: http://localhost:8000/docs — interactive, try endpoints directly.",
         "• <b>ReDoc</b>: http://localhost:8000/redoc — clean, readable documentation.",
         "Both are generated from Pydantic models and function type annotations.",
         "Testing methods:",
         "• Swagger UI: Upload image file, test all endpoints in browser.",
         "• curl: curl -X POST http://localhost:8000/api/quick-analyze -F 'image=@pitch.jpg'",
         "• Python requests library: programmatic testing.",
         "• test_api.py: Our own test script for API validation."],
        styles, LIGHT_GREEN, DARK_GREEN
    ))

    story.append(qa_card(57,
        "What HTTP status codes does your API use?",
        ["• <b>200 OK</b>: Successful request.",
         "• <b>400 Bad Request</b>: Invalid input — file is not an image, missing required field.",
         "• <b>401 Unauthorized</b>: Missing or invalid JWT token.",
         "• <b>403 Forbidden</b>: Valid token but no permission (e.g., free user accessing premium).",
         "• <b>404 Not Found</b>: Analysis ID not found in database.",
         "• <b>422 Unprocessable Entity</b>: Pydantic validation failed (wrong data type).",
         "• <b>500 Internal Server Error</b>: ML pipeline failure, unexpected exception.",
         "• <b>503 Service Unavailable</b>: Weather API is down, cannot fetch weather data."],
        styles
    ))

    story.append(qa_card(58,
        "How does the OpenWeatherMap API integration work?",
        ["OpenWeatherMap provides free weather data via REST API.",
         "Function: get_weather_data(latitude, longitude, city)",
         "API call: GET https://api.openweathermap.org/data/2.5/weather",
         "  ?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric",
         "Response parsed into WeatherData model:",
         "  { temperature, humidity, wind_speed, rainfall, conditions, location }",
         "Fallback: If lat/lon not provided → use city name parameter instead.",
         "Error handling: If API fails → return None, analysis continues without weather.",
         "→ Weather is always optional — the pitch analysis works without it."],
        styles, LIGHT_ORANGE, colors.HexColor('#78350f')
    ))

    story.append(qa_card(59,
        "What is the analysis_id format and how is it generated?",
        ["analysis_id is a unique string identifier for each analysis.",
         "Format: 'PITCH_YYYYMMDD_HHMMSS'",
         "Example: 'PITCH_20251217_143052'",
         "Generated with: datetime.now().strftime('PITCH_%Y%m%d_%H%M%S')",
         "Use cases:",
         "• Retrieve specific analysis later.",
         "• Reference in chatbot context.",
         "• Share analysis link with team.",
         "In MongoDB: Stored as string, indexed for fast lookup."],
        styles
    ))

    story.append(qa_card(60,
        "How is processing time measured and why is it important?",
        ["Processing time is measured per request:",
         "  start_time = time.time()",
         "  # ... all ML processing ...",
         "  processing_time = time.time() - start_time",
         "Typical breakdown:",
         "• YOLO detection: 0.1-0.2s",
         "• OpenCV features: 0.3-0.5s",
         "• ResNet18 classification: 0.5-0.8s",
         "• Rule engine: <0.1s",
         "• Weather API: 0.5-1.0s (if enabled)",
         "• Total: 0.8-1.5s (quick) or 2-4s (full with weather)",
         "Why it matters: User experience — >4s feels slow. Monitoring API performance."],
        styles
    ))

    story.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 9 — DEPLOYMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(section_header("9. DEPLOYMENT & DEVOPS", styles))

    story.append(qa_card(61,
        "How would you deploy this application to production?",
        ["<b>Backend → Render (or Railway)</b>:",
         "• render.yaml defines service type, build command, start command.",
         "• Build: pip install -r requirements.txt",
         "• Start: uvicorn app:app --host 0.0.0.0 --port $PORT",
         "• Environment vars set in Render dashboard.",
         "• URL: https://pitch-insight-api.onrender.com",
         "<b>Frontend → Vercel (or Netlify)</b>:",
         "• Build command: npm run build → output: dist/ folder.",
         "• Vercel auto-detects Vite projects.",
         "• Set VITE_API_URL to backend URL.",
         "• URL: https://pitch-insight.vercel.app",
         "<b>Database → MongoDB Atlas</b>:",
         "• Cloud-managed MongoDB. Free tier: 512MB.",
         "• Connection string in MONGODB_URL env var."],
        styles, LIGHT_BLUE, DARK_BLUE
    ))

    story.append(qa_card(62,
        "How would you containerize this application with Docker?",
        ["<b>Backend Dockerfile</b>:",
         "  FROM python:3.12-slim",
         "  WORKDIR /app",
         "  COPY requirements.txt .",
         "  RUN pip install --no-cache-dir -r requirements.txt",
         "  COPY . .",
         "  EXPOSE 8000",
         "  CMD ['uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8000']",
         "<b>Frontend Dockerfile (multi-stage)</b>:",
         "  Stage 1 (build): node:18-alpine → npm run build → dist/",
         "  Stage 2 (serve): nginx:alpine → serve dist/ on port 80.",
         "Benefits: Consistent environment, easy scaling, works anywhere Docker runs."],
        styles
    ))

    story.append(qa_card(63,
        "What are the render.yaml and Procfile files for?",
        ["<b>render.yaml</b>: Infrastructure-as-code for Render platform.",
         "  Defines: service name, runtime (python), build command, start command, env vars.",
         "  Render reads this file to auto-configure the deployment.",
         "<b>Procfile</b>: Used by Heroku and similar platforms.",
         "  Content: web: uvicorn app:app --host 0.0.0.0 --port $PORT",
         "  Defines: process type (web) and the command to start it.",
         "  $PORT is set by the platform automatically.",
         "Both files enable one-click deployment on respective platforms."],
        styles
    ))

    story.append(qa_card(64,
        "What are the production considerations for a real deployment?",
        ["Security:",
         "• HTTPS only (SSL/TLS via platform or nginx).",
         "• Rate limiting (e.g., 10 requests/minute per IP).",
         "• Input validation + file size limits (5MB).",
         "• SECRET_KEY must be a strong random string.",
         "Performance:",
         "• Multiple Uvicorn workers: gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app",
         "• CDN for frontend static files.",
         "• Model caching: load once at startup, reuse.",
         "Monitoring:",
         "• Sentry for error tracking.",
         "• Health check endpoint (/api/health) for uptime monitoring.",
         "• Render/Railway provide basic metrics dashboard."],
        styles, LIGHT_GREEN, DARK_GREEN
    ))

    story.append(qa_card(65,
        "How would you scale this application if traffic increases 10x?",
        ["<b>Horizontal scaling</b>:",
         "• Multiple API server instances behind a load balancer.",
         "• Stateless design (no server-side sessions) → easy to scale horizontally.",
         "• MongoDB Atlas scales automatically with usage.",
         "<b>Vertical scaling</b>:",
         "• GPU inference: Replace CPUExecutionProvider with CUDAExecutionProvider.",
         "• GPU reduces ResNet18 inference from 0.5s to ~0.05s.",
         "<b>Caching</b>:",
         "• Redis cache for identical image analysis (hash image → cache result).",
         "• CDN for frontend assets.",
         "<b>Async processing</b>:",
         "• For very high load: queue analysis jobs in Celery + Redis.",
         "• Return job_id immediately, poll for results."],
        styles
    ))

    story.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 10 — CHALLENGES & FUTURE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(section_header("10. CHALLENGES, RESULTS & FUTURE WORK", styles))

    story.append(qa_card(66,
        "What was the biggest technical challenge you faced?",
        ["<b>Challenge 1: Memory optimization for ONNX models on free-tier servers</b>",
         "• Problem: Render free tier has 512MB RAM. YOLOv8 + ResNet18 together used ~800MB.",
         "• Solution: ONNX models use less RAM than PyTorch. Lazy loading. CPU-only inference.",
         "• Created MEMORY_OPTIMIZATION.md with detailed memory profiling.",
         "<b>Challenge 2: Spin vs Seam misclassification</b>",
         "• Problem: Both look similar (dry, brown). Model confused them at 73% accuracy for spin.",
         "• Solution: Added rule-based adjustments using crack detection as a differentiator.",
         "<b>Challenge 3: CORS in development</b>",
         "• Problem: React (3000) calling FastAPI (8000) → CORS blocked.",
         "• Solution: Configured CORSMiddleware with correct origins."],
        styles, LIGHT_ORANGE, colors.HexColor('#78350f')
    ))

    story.append(qa_card(67,
        "What would you do differently if you were to rebuild this project?",
        ["• Use GraphQL instead of REST for flexible data fetching.",
         "• Use React Query or SWR for server state management and caching.",
         "• Implement proper CI/CD pipeline from day one (GitHub Actions).",
         "• Use Docker Compose for local development (no manual setup).",
         "• Add comprehensive unit tests (pytest for backend, Jest for frontend).",
         "• Use a proper logging library (structlog or loguru) instead of print().",
         "• Train with more data — 2,585 is relatively small for computer vision.",
         "• Use a more modern model: EfficientNetV2 or Vision Transformer (ViT)."],
        styles
    ))

    story.append(qa_card(68,
        "What are the planned future enhancements?",
        ["<b>ML Improvements</b>:",
         "• Multi-model ensemble for higher accuracy.",
         "• Time-series model for pitch deterioration prediction (Day 1 vs Day 5).",
         "• Venue-specific models (different pitches behave differently at each ground).",
         "<b>Features</b>:",
         "• Video analysis: Analyze pitch condition from match footage.",
         "• Player performance predictions based on pitch type.",
         "• Historical data integration (ESPNcricinfo API).",
         "• Mobile app (React Native).",
         "<b>Technical</b>:",
         "• WebSocket for real-time updates.",
         "• Kubernetes for auto-scaling.",
         "• GraphQL API.",
         "• Automated testing with pytest + GitHub Actions CI/CD."],
        styles
    ))

    story.append(qa_card(69,
        "What are the limitations of your current approach?",
        ["• <b>Image quality dependent</b>: Poor lighting, blurry images reduce accuracy.",
         "• <b>Static analysis</b>: Can't track pitch deterioration over match overs.",
         "• <b>Training data bias</b>: Most data from professional matches — amateur pitches may differ.",
         "• <b>Weather is supplementary</b>: Weather doesn't directly change ML prediction, only strategy.",
         "• <b>No ground truth validation</b>: We don't know actual match outcomes to validate strategy.",
         "• <b>Single image</b>: Taking one photo may not capture the whole pitch condition.",
         "• <b>Spin/Seam confusion</b>: 73% accuracy for spin class needs improvement."],
        styles, LIGHT_BLUE, DARK_BLUE
    ))

    story.append(qa_card(70,
        "How would you evaluate if the system is actually useful in practice?",
        ["• <b>Collect feedback</b>: Ask coaches/analysts if recommendations matched actual pitch behavior.",
         "• <b>Compare to expert analysis</b>: Compare our predictions to official pitch curators' reports.",
         "• <b>Match outcome correlation</b>: Did teams that followed our toss advice win more?",
         "• <b>A/B testing</b>: Compare strategy outcomes for teams using vs not using the tool.",
         "• <b>Track accuracy over time</b>: Log predictions vs actual pitch behavior after each match.",
         "→ Currently, the 91.6% accuracy is on a labeled test set, not real-world outcome validation."],
        styles
    ))

    story.append(qa_card(71,
        "Can your model handle night match conditions or different pitch types globally?",
        ["Night matches (floodlights):",
         "• Floodlights change color temperature → may affect HSV grass detection.",
         "• The ML model may not handle artificial lighting well (not in training data).",
         "• Mitigation: Add artificial lighting images to training set.",
         "Global pitch types:",
         "• England (green, grassy) vs India (dry, spin) vs Australia (hard, bouncy).",
         "• Our model is trained on diverse images but may favor more common conditions.",
         "• Venue-specific models would handle this better.",
         "• Current approach: rule engine adjustments help somewhat.",
         "→ This is a valid limitation to acknowledge in interviews."],
        styles
    ))

    story.append(qa_card(72,
        "How do you measure the model's confidence and when is it reliable?",
        ["• Confidence = the softmax probability of the predicted class.",
         "• High confidence (>80%): Model is very sure. E.g., spin_friendly at 99.93%.",
         "• Medium confidence (60-80%): Reasonable prediction, check features for context.",
         "• Low confidence (<60%): Model is uncertain — features and rule engine are more important.",
         "Best practices:",
         "• Set a confidence threshold of 80% for 'high confidence' label.",
         "• If confidence is low, show all 4 probabilities prominently.",
         "• After rule adjustment, confidence may increase or decrease.",
         "• Perfect pitch images (clear, well-lit, close-up) → higher confidence."],
        styles, LIGHT_GREEN, DARK_GREEN
    ))

    story.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 11 — BEHAVIORAL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(section_header("11. BEHAVIORAL & HR QUESTIONS", styles))

    story.append(qa_card(73,
        "Why did you build this project?",
        ["• Cricket is deeply personal — I grew up watching matches and understanding pitch conditions.",
         "• I wanted to apply AI/ML to a real-world domain problem I genuinely care about.",
         "• The project challenged me to integrate multiple technologies: deep learning, computer vision, "
         "REST APIs, React, and payment gateways.",
         "• It's a complete full-stack product, not just a model — it demonstrates end-to-end software development.",
         "• I also wanted to solve the subjectivity problem in pitch analysis — make it objective and data-driven."],
        styles, LIGHT_BLUE, DARK_BLUE
    ))

    story.append(qa_card(74,
        "What did you learn from building this project?",
        ["Technical learnings:",
         "• ONNX model deployment: How to convert PyTorch → ONNX and optimize for production.",
         "• FastAPI's dependency injection and lifespan management.",
         "• OpenCV image processing techniques (HSV, Canny, Laplacian, K-means).",
         "• JWT authentication end-to-end implementation.",
         "• Payment gateway integration (Razorpay webhook verification).",
         "Soft learnings:",
         "• Breaking complex systems into manageable modules.",
         "• Importance of documentation — TECHNICAL_DOCUMENTATION.md helped me debug myself.",
         "• How small model improvements (rule engine) can significantly boost user trust."],
        styles
    ))

    story.append(qa_card(75,
        "How did you handle a situation where something didn't work as expected?",
        ["The ONNX conversion issue:",
         "• Original plan: Use PyTorch .pth model directly.",
         "• Problem: PyTorch + CUDA = 2GB+ dependencies, too heavy for Render free tier.",
         "• Action: Researched ONNX Runtime as a lighter alternative.",
         "• Implementation: Converted models, rewrote inference code with onnxruntime.InferenceSession.",
         "• Result: Memory reduced by ~60%, deployment succeeded on free tier.",
         "Lesson: Always profile memory/performance before choosing a technology for production."],
        styles, LIGHT_GREEN, DARK_GREEN
    ))

    story.append(qa_card(76,
        "How do you stay updated with the latest in ML/AI?",
        ["• Follow Hugging Face, Papers With Code for latest research.",
         "• Practice on Kaggle competitions.",
         "• YouTube: Andrej Karpathy, Sentdex, Two Minute Papers.",
         "• GitHub: Follow repositories of PyTorch, Ultralytics (YOLO), FastAPI.",
         "• Build projects: Best way to learn is to implement new concepts.",
         "→ For this project: Learned YOLOv8 from Ultralytics docs, ResNet from PyTorch tutorials."],
        styles
    ))

    story.append(qa_card(77,
        "How would you explain this project to a non-technical person?",
        ["'Imagine you're a cricket captain and you need to decide whether to bat or bowl after winning the toss. "
         "Traditionally, you'd look at the pitch and use experience to guess its nature.",
         "Pitch Insight is like a smart assistant — you take a photo of the pitch and it analyzes it using AI. "
         "It tells you: this pitch will spin a lot, so bat first and score big before the spinners dominate. "
         "It also checks the weather and says: humidity is high, which helps swing bowlers.",
         "Think of it as having a cricket expert in your pocket — one that uses data and AI instead of gut feeling.'"],
        styles, LIGHT_ORANGE, colors.HexColor('#78350f')
    ))

    story.append(qa_card(78,
        "What would you prioritize if you had 2 more weeks to work on this?",
        ["Priority 1: Improve spin-friendly accuracy (currently 73.77%)",
         "  • Collect 500+ more spin-friendly training images.",
         "  • Re-train with class-weighted loss.",
         "Priority 2: Add comprehensive unit tests",
         "  • pytest for backend API endpoints and ML pipeline.",
         "  • Jest + React Testing Library for React components.",
         "Priority 3: CI/CD pipeline",
         "  • GitHub Actions: auto-test on PR, auto-deploy on merge to main.",
         "Priority 4: Video analysis",
         "  • Analyze pitch from 5-second video instead of single photo.",
         "  • Average predictions across frames for more robust classification."],
        styles
    ))

    story.append(qa_card(79,
        "What is your biggest takeaway from this project for your career?",
        ["• Full-stack AI products require skills beyond just training models.",
         "• Production ML is very different from notebook ML.",
         "• System design matters: a well-designed architecture saved hours of debugging.",
         "• Domain knowledge amplifies ML — the rule engine doubled user trust.",
         "• Documentation is code: good docs made the project maintainable.",
         "• Solved a real problem I care about — passion drives quality."],
        styles
    ))

    story.append(qa_card(80,
        "Do you have any questions for us? (What to ask the interviewer)",
        ["• 'What ML/AI tech stack does your team primarily use?'",
         "• 'How does your team handle model versioning and deployment?'",
         "• 'What's the ratio of research/experimentation vs production work?'",
         "• 'How does the team stay updated with fast-moving ML developments?'",
         "• 'What would a typical project look like for someone in this role?'",
         "→ Always ask a question — it shows curiosity and preparation."],
        styles, LIGHT_GREEN, DARK_GREEN
    ))

    story.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 12 — CHEAT SHEET
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(section_header("12. QUICK REFERENCE CHEAT SHEET", styles))

    # Key Numbers
    story.append(Paragraph("📊 Key Numbers to Remember:", styles['section']))
    numbers_data = [
        ["Metric", "Value"],
        ["Test Accuracy", "91.60%"],
        ["Validation Accuracy", "91.84%"],
        ["Training Images", "2,585 images"],
        ["Pitch Classes", "4 (batting/bowling/seam/spin)"],
        ["Training Epochs", "30"],
        ["Batch Size", "32"],
        ["Learning Rate", "0.001 (Adam)"],
        ["ResNet18 Parameters", "11,178,564"],
        ["Model Size (.pth)", "11 MB"],
        ["ONNX Model Size", "44 MB"],
        ["Processing Time (quick)", "0.8-1.5 seconds"],
        ["Processing Time (full)", "2-4 seconds"],
        ["YOLO Confidence Threshold", "0.5"],
        ["OpenCV Features Count", "6"],
        ["Domain Rules Count", "6"],
        ["ResNet18 Input Size", "224 × 224 RGB"],
        ["YOLO Input Size", "640 × 640 RGB"],
    ]
    story.append(metrics_table(numbers_data, [3.4*inch, 3.4*inch], styles))

    # Key Technologies
    story.append(Paragraph("🛠️ Technology Mapping:", styles['section']))
    tech_map_data = [
        ["What?", "Technology Used", "Why?"],
        ["Pitch Detection", "YOLOv8 (ONNX)", "Real-time single-stage object detection"],
        ["Pitch Classification", "ResNet18 (ONNX)", "Transfer learning, 91.6% accuracy"],
        ["Feature Extraction", "OpenCV 4.12", "Computer vision, efficient CPU processing"],
        ["API Framework", "FastAPI + Uvicorn", "Async, auto-docs, Pydantic validation"],
        ["Frontend", "React 18 + Vite", "Component-based, fast HMR, modern tooling"],
        ["HTTP Client", "Axios", "Interceptors, promise-based, easy error handling"],
        ["Database", "MongoDB Atlas", "Flexible schema, JSON-native, cloud-managed"],
        ["Authentication", "JWT + bcrypt", "Stateless auth, secure password hashing"],
        ["Payment", "Razorpay", "India-focused, easy integration, webhook support"],
        ["AI Chatbot", "Google Gemini", "Free tier, strong cricket knowledge"],
        ["Weather", "OpenWeatherMap", "Free tier, real-time data, lat/lon support"],
        ["Deployment (BE)", "Render / Railway", "Free tier, auto-deploy from GitHub"],
        ["Deployment (FE)", "Vercel / Netlify", "Free tier, Vite-optimized, CDN"],
    ]
    story.append(metrics_table(tech_map_data, [1.5*inch, 1.6*inch, 3.7*inch], styles))

    # OpenCV Feature Summary
    story.append(Paragraph("🔍 OpenCV Features Summary:", styles['section']))
    cv_data = [
        ["Feature", "Method", "Output", "Cricket Impact"],
        ["Grass Coverage", "HSV filtering\n(H: 25-85)", "%, Level (High/Med/Low)", "Grass = swing/seam"],
        ["Crack Detection", "Canny edges +\ncontours", "Count, Severity (H/M/L)", "Cracks = spin turn"],
        ["Moisture", "HSV V-channel\nmean brightness", "Score 0-100, Level", "Wet = slow, Dry = spin"],
        ["Color Profile", "K-means\nclustering (k=3)", "Type (Brown/Red/Green)", "Prep type indicator"],
        ["Texture", "Laplacian\nvariance", "Roughness level", "Rough = variable bounce"],
        ["Brightness", "Grayscale mean\n(0-255)", "5 brightness levels", "Hardness indicator"],
    ]
    story.append(metrics_table(cv_data, [1.2*inch, 1.3*inch, 1.5*inch, 2.8*inch], styles))

    # Rule Engine Summary
    story.append(Paragraph("⚙️ Cricket Rule Engine Summary:", styles['section']))
    rule_data = [
        ["#", "Condition", "Action", "Reason"],
        ["R1", "Grass > 60%", "Bowling += 15%", "Heavy grass aids swing"],
        ["R2", "Cracks = High", "Spin += 20%", "Deep cracks grip for spinners"],
        ["R2b", "Cracks = Medium", "Spin += 12%", "Moderate cracks help spin"],
        ["R3", "Very Dry + Grass < 20%", "Seam += 15%", "Dry pitch + seam movement"],
        ["R4", "Low cracks + Grass 20-50%", "Batting += 10%", "Balanced pitch"],
        ["R5", "Very Dry + High Cracks", "Spin += 15%", "Extreme turning conditions"],
        ["R6", "Wet/Damp + Grass > 50%", "Bowling += 12%", "Humid + grass = swing"],
    ]
    story.append(metrics_table(rule_data, [0.4*inch, 1.8*inch, 1.4*inch, 3.2*inch], styles))

    story.append(Spacer(1, 10))
    story.append(tip_box(
        "INTERVIEW STRATEGY: Use the STAR method for behavioral questions (Situation, Task, Action, Result). "
        "For technical questions, always explain WHY you made choices, not just WHAT you built. "
        "Acknowledge limitations honestly — it shows maturity.", styles, 'tip'
    ))
    story.append(tip_box(
        "CONFIDENCE BUILDER: You built a full-stack AI application with 91.6% accuracy using YOLOv8, ResNet18, "
        "FastAPI, React, MongoDB, Razorpay, and Gemini AI. This demonstrates breadth and depth across the entire "
        "software development stack. Be proud of what you built!", styles, 'note'
    ))

    # Footer
    story.append(Spacer(1, 12))
    footer_data = [[
        Paragraph(
            f"🏏 Pitch Insight — Interview Preparation Guide | Generated: {date_str} | "
            "AI-Powered Cricket Pitch Analyzer | Built with FastAPI + React + YOLOv8 + ResNet18",
            ParagraphStyle('final_footer', fontName='Helvetica', fontSize=7.5,
                textColor=WHITE, alignment=TA_CENTER)
        )
    ]]
    footer_table = Table(footer_data, colWidths=[6.8*inch])
    footer_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), DARK_BG),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LINEABOVE', (0,0), (-1,-1), 1, PRIMARY),
    ]))
    story.append(footer_table)

    doc.build(story)
    print(f"[OK] PDF generated successfully: {output_path}")


if __name__ == '__main__':
    output = r'e:\pitch_insight\Pitch_Insight_Complete_Interview_Guide.pdf'
    generate_pdf(output)
