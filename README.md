# AI-Powered Handwritten Exam Answer Evaluation System

> **Production-Minded AI Prototype for Handwritten Exam Evaluation**  
> *Developed for CNV AI Labs Technical Assignment*

---

## 1. Project Overview

The **AI-Powered Handwritten Exam Answer Evaluation System** is an end-to-end Python backend application designed to automate the transcription, structuring, semantic evaluation, and quality assurance of handwritten student exam sheets. 

### Core Objective
Evaluating handwritten exam scripts manually is labor-intensive, error-prone, and inconsistent. This system automates the evaluation pipeline across four mandatory stages:
1. **Extraction (OCR)**: Ingests scanned or photographed exam sheets (PDF or multi-format images) and transcribes handwritten answers.
2. **Question-Wise Structuring**: Identifies question boundaries (`Q1`, `1(a)`, `Ans 2`), handles answers continuing across multiple pages, isolates diagram markers, and flags unmapped text as `UNSTRUCTURED`.
3. **Semantic Rubric Evaluation**: Grades conceptual understanding against an authoritative marking rubric using semantic alignment, avoiding rigid keyword matching and penalizing contradictions.
4. **Confidence Estimation & Human Triage**: Computes a multi-signal heuristic confidence rating and automatically routes ambiguous, unmapped, or low-scoring answers for human review.

### Key Features & Technology Stack
- **Backend Architecture**: Python 3.10+, FastAPI, Pydantic v2 data validation, Uvicorn ASGI server.
- **Computer Vision & Ingestion**: PyMuPDF (fitz) for 200 DPI PDF rendering, OpenCV (cv2) for adaptive contrast (CLAHE), bilateral denoising, deskewing, and projection-profile line segmentation.
- **Handwriting Recognition**:
  - **Primary Engine**: OpenAI Vision (`gpt-4o`) for end-to-end full-page handwriting transcription with `[unclear]` uncertainty tagging.
  - **Local Fallback Engine**: Microsoft TrOCR (`microsoft/trocr-base-handwritten`) via Hugging Face Transformers and PyTorch for 100% offline, zero-dependency local recognition.
- **Semantic Evaluation**:
  - **OpenAI Evaluator**: Contextual rubric-based grading via `gpt-4o` with strict Pydantic JSON schema enforcement.
  - **Local Transformer Evaluator**: Dense sentence embedding alignment using `sentence-transformers/all-MiniLM-L6-v2` with cosine similarity, dual-reference context alignment, and contradiction detection.
- **Output Artifacts**: Valid JSON conforming to required assignment schemas and flattened CSV exports.
- **Testing & Verification**: 96 automated unit and regression tests written in Pytest.

### Source Code Repository Reference
- **Local Workspace**: `answer-sheet-evaluator`
- **Default Branch**: `main`
- **Repository Setup**: Initialized with Git, protected with `.gitignore` to prevent credential leaks, and staged for remote synchronization via `git push -u origin main`.

---

## 2. Technical Approach and Future Improvements

### Pipeline Architecture

```
INPUT DOCUMENT (PDF / JPG / PNG)
               │
               ▼
┌───────────────────────────────┐
│     Document Ingestion        │  • PyMuPDF (200 DPI) & PIL image loader
│     (document_loader.py)      │  • Dimension & format validation
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│      Image Preprocessing      │  • Bilateral filtering & CLAHE contrast
│     (image_processing.py)     │  • HoughLinesP deskewing (±15°)
└──────────────┬────────────────┘  • Preserves continuous ink tones
               │
               ▼
┌───────────────────────────────┐
│    Handwriting Recognition    │  • Primary: OpenAI Vision (gpt-4o)
│  (handwriting_recognition.py) │  • Fallback: Line-based Microsoft TrOCR
└──────────────┬────────────────┘  • Emits [unclear] tags & OCR quality
               │
               ▼
┌───────────────────────────────┐
│     Question Segmentation     │  • Regex patterns: Q1, 1(a), 1., Ans 2
│       (segmentation.py)       │  • Multi-page continuation tracking
└──────────────┬────────────────┘  • Diagram marker extraction ([diagram])
               │
               ▼
┌───────────────────────────────┐
│   Rubric Semantic Evaluation  │  • Evaluates meaning, not exact keywords
│    (semantic_evaluation.py)   │  • Detects contradictions & missing concepts
│       (llm_provider.py)       │  • Clamps marks: score <= max_marks
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│      Confidence Engine        │  • 5 weighted signals (OCR, mapping, coverage,
│     (confidence_engine.py)    │    consistency, substance)
└──────────────┬────────────────┘  • Strict human-review guardrails
               │
               ▼
┌───────────────────────────────┐
│       Output Generation       │  • Schema-compliant JSON output
│     (output_generator.py)     │  • Flattened tabular CSV export
└───────────────────────────────┘
```

### Technical Approach Details
1. **Conservative Preprocessing**: Binarization (Otsu thresholding) often destroys faint pencil or ink strokes. The pipeline applies contrast-limited adaptive histogram equalization (CLAHE) and bilateral filtering to retain grayscale stroke dynamics before OCR.
2. **Hybrid Recognition Strategy**: Full-page vision transcription via `gpt-4o` eliminates line-cropping errors and understands handwritten paragraph structure. If the API is unreachable or quota is exhausted (HTTP 429), the system automatically routes images to local line-sliced TrOCR.
3. **Meaning-Centric Grading**: Answers are evaluated against criterion descriptions and expected contexts. Semantic similarity thresholds, evidence quoting, and negation heuristics prevent superficial keyword matching from awarding unwarranted marks.
4. **Heuristic Confidence & Safety Guards**: The confidence score is explicitly documented as a heuristic metric rather than a calibrated probability. Mandatory human review flags are triggered whenever answers are `UNSTRUCTURED`, rubrics are missing, OCR is poor, handwriting is illegible, or evaluations fail.

### Future Improvements Note
1. **End-to-End Multimodal Vision Fine-Tuning**: Fine-tune domain-specific vision-language models (e.g., ColPali, Florence-2, or Qwen2-VL) directly on dense handwritten examination sheets to recognize cursive handwriting, mathematical notations, and chemical equations without line-cropping.
2. **Formula and LaTeX Recognition**: Integrate specialized symbol-detection models (such as Nougat or Mathpix API) to transcribe complex fractional, integral, and matrix calculations directly into LaTeX representations for symbolic mathematical equivalence testing.
3. **Computer Vision Diagram Grading**: Implement a dual-branch visual-semantic pipeline that segments diagram bounding boxes and evaluates structural correctness, labeled axes, and flowchart topologies against reference geometry using visual similarity graphs.
4. **Multilingual and Vernacular Support**: Expand recognition and rubric evaluation to support non-English scripts (e.g., Devanagari, Arabic, Tamil) using multilingual transformer checkpoints and localized answer keys.
5. **Calibrated Bayesian Confidence Scoring**: Replace linear heuristic weighting with Platt scaling, temperature scaling, or conformal prediction to output mathematically calibrated posterior probabilities for human review routing.
6. **Active Human-in-the-Loop Triage UI**: Develop a dedicated teacher dashboard with side-by-side visual stroke overlays, diff inspectors, and one-click mark overrides that continuously collect correction feedback.
7. **Distributed Async Processing and Horizontal Scalability**: Decouple document ingestion from compute-intensive OCR using Celery/RabbitMQ or Google Cloud Tasks, backed by Redis caching and S3/GCS blob storage.
8. **Real-Time Telemetry and Drift Monitoring**: Implement Prometheus, Grafana, and OpenTelemetry instrumentation to monitor per-page OCR latency, model drift, token usage, and rubric alignment divergence in production.

---

## 3. Input Document Handling

### Accepted File Formats
The system accepts scanned or photographed exam documents via `POST /evaluate`:
- **PDF Documents** (`.pdf`): Multi-page PDF files rendered at 200 DPI using PyMuPDF to produce high-resolution page buffers.
- **Image Files** (`.jpg`, `.jpeg`, `.png`): Single-page photographs or flatbed scans with dimension validation ($100\text{px} \le \text{dimension} \le 4000\text{px}$).

### Reference Sample Files
Pre-generated sample test files are located in `data/samples/`:
- `data/samples/sample_exam.pdf`: A 3-page synthetic examination document covering biology, the water cycle, and plant cell structure.
- `data/samples/sample_page1.png`: Page 1 containing Questions Q1 and Q2.
- `data/samples/sample_page2.png`: Page 2 containing the continuation of Q2, and Question Q3 with a `[diagram]` marker.
- `data/samples/sample_page3.png`: Page 3 containing Question Q4.

> **Note on Test Data**: Synthetic typed-handwriting simulation images are provided in `data/samples/` for automated regression testing. When testing handwriting recognition, real photographed or scanned handwritten pages should be uploaded.

### Uploading Documents via Swagger UI
1. Start the API server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
2. Navigate to the interactive Swagger UI at:  
   **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
3. Expand the `POST /evaluate` endpoint and click **Try it out**.
4. Click **Choose File** and select an exam file (e.g., `data/samples/sample_exam.pdf` or an actual handwritten image).
5. Specify `student_id` (e.g., `S001`).
6. Optionally provide a custom `answer_key_path` (defaults to `config/answer_key.json`).
7. Click **Execute** to view the live evaluation output and confidence assessment.

---

## 4. Output Document

### Actual Test Execution Output (`evaluation_id: e44637a1`)

The following is an authentic test execution record produced during initial pipeline regression testing on an unconstrained test image.

> **Important**: This output is explicitly presented to document actual OCR and segmentation limitations encountered during early testing, **not as a demonstration of successful grading**.

```json
{
  "student_id": "S001",
  "evaluation": [
    {
      "question_no": "UNSTRUCTURED",
      "extracted_answer": "a member of the South Pacific Railroad that was the first ever ever ever ever ever ever ever ever ever ever\nDownload as PDFPrintable version\n2 Basic # and General #",
      "score": 0.0,
      "max_marks": 0.0,
      "matched_concepts": [],
      "missing_concepts": [],
      "evidence": [
        "No rubric entry found for this question."
      ],
      "criterion_results": [],
      "confidence": "medium",
      "confidence_score": 0.533,
      "reason": "Partial confidence: Answer not mapped to a specific question number; No criterion-level results available.",
      "human_review_required": false,
      "source_pages": [
        1
      ]
    }
  ],
  "total_score": 0.0,
  "maximum_score": 0.0,
  "review_required": false,
  "metadata": {
    "evaluation_id": "e44637a1",
    "document": "C:\\Users\\saiso\\AppData\\Local\\Temp\\tmpjz07h3z2\\thumb_1200_1698.png",
    "pages": 1,
    "provider": "see individual evaluations"
  }
}
```

### Analysis of Test Response `e44637a1`
1. **OCR Hallucination**: TrOCR was passed an unconstrained document slice without distinct line crops. Its RoBERTa language decoder hallucinated repetitive pretraining text (*"South Pacific Railroad... ever ever ever"*).
2. **Unstructured Classification**: Because the recognized text contained no identifiable question pattern (`Q1`, `1.`), the segmentation engine correctly classified the text as `UNSTRUCTURED`.
3. **Rubric Isolation**: Since no rubric existed for `UNSTRUCTURED`, the system awarded `0.0` marks out of `0.0` max marks and refrained from fabricating scores.
4. **Confidence Hardening**: In this initial run, the score heuristic was calculated as `0.533` (medium confidence) due to image contrast signals. This identified a critical edge case, leading to the implementation of **strict confidence guardrails** in `confidence_engine.py`: any `UNSTRUCTURED` answer, missing rubric, or poor OCR now strictly sets `confidence: "low"` and `human_review_required: true`.

### JSON Output Fields Reference
| Field | Type | Description |
|---|---|---|
| `student_id` | `string` | Unique identifier for the student |
| `question_no` | `string` | Identified question number (`Q1`, `1(a)`) or `UNSTRUCTURED` |
| `extracted_answer` | `string` | Verbatim transcribed student text (never rewritten) |
| `score` | `float` | Total marks awarded (guaranteed $\le \text{max\_marks}$) |
| `max_marks` | `float` | Maximum marks possible for the question |
| `matched_concepts` | `list[str]` | Rubric criteria confirmed present and correct |
| `missing_concepts` | `list[str]` | Rubric criteria absent, incorrect, or contradicted |
| `evidence` | `list[str]` | Exact quotes from student answer supporting the score |
| `criterion_results` | `list[dict]` | Per-criterion breakdown (`present`, `correct`, `marks_awarded`) |
| `confidence` | `string` | Heuristic uncertainty level (`high`, `medium`, `low`) |
| `confidence_score` | `float` | Composite signal score ($0.0 \le s \le 1.0$) |
| `reason` | `string` | Concise explanation of confidence determination |
| `human_review_required` | `bool` | Mandatory flag routing ambiguous answers to human reviewers |
| `source_pages` | `list[int]` | 1-indexed source document pages for traceability |
| `total_score` | `float` | Sum of all marks awarded across questions |
| `maximum_score` | `float` | Sum of maximum possible marks across questions |
| `review_required` | `bool` | Global document-level flag indicating if any question requires review |
| `metadata` | `dict` | Evaluation ID, page count, and OCR/evaluation engine attribution |

### CSV Output Format
Every evaluation automatically generates a companion flattened CSV file in `output/`:
- **File Reference**: `output/evaluation_S001.csv`
- **Headers**:
  `student_id, question_no, extracted_answer, score, max_marks, matched_concepts, missing_concepts, evidence, confidence, confidence_score, reason, human_review_required, source_pages`

---

## 5. Setup and Execution

### Prerequisites
- Python 3.10 or 3.11
- Operating System: Windows, macOS, or Linux
- ~1.5 GB disk space if running local TrOCR fallback models on first download

### Step-by-Step Installation

```bash
# 1. Clone repository and navigate to root
cd answer-sheet-evaluator

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env      # Windows
# cp .env.example .env      # Linux/macOS
```

### Environment Configuration (`.env`)
Configure settings in `.env` (never commit this file to Git):

```ini
# General Settings
DEBUG=false
LOG_LEVEL=INFO

# OpenAI API Configuration (Vision OCR & Semantic Scoring)
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_VISION_MODEL=gpt-4o
OPENAI_EVAL_MODEL=gpt-4o

# Primary Engines & Fallbacks
OCR_PROVIDER=openai       # "openai" with automatic TrOCR fallback
LLM_PROVIDER=openai       # "openai" with automatic Sentence-Transformer fallback

# Local Fallback Models (100% Offline)
TROCR_MODEL_NAME=microsoft/trocr-base-handwritten
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2

# Hardware Acceleration
DEVICE=auto               # "auto", "cuda", or "cpu"

# Image Preprocessing Toggles
PREPROCESS_GRAYSCALE=true
PREPROCESS_DENOISE=true
PREPROCESS_CONTRAST=true
PREPROCESS_THRESHOLD=false # Kept false to preserve faint handwriting strokes
PREPROCESS_DESKEW=true

# Confidence Thresholds
CONFIDENCE_HIGH_THRESHOLD=0.75
CONFIDENCE_LOW_THRESHOLD=0.45

# API Server
API_HOST=0.0.0.0
API_PORT=8000
```

> **Security Warning**: Never hardcode API keys or commit `.env` files to source control. The repository `.gitignore` is configured to exclude all `.env` files, virtual environments, and evaluation outputs.

### Starting the Application Server
Run the FastAPI application via Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
# or
python main.py
```

### Accessing API Documentation
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc UI**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Running Automated Test Suite
Run the 96 unit, regression, and integration tests:

```bash
python -m pytest tests/ -v --tb=short
```

---

## 6. Verified Limitations

Based on empirical testing and pipeline execution, the following limitations are documented:

1. **TrOCR Single-Line Limitation**: Microsoft TrOCR was trained on single-line crops. When applied to multi-line unconstrained handwritten exam paragraphs on CPU, it suffers from severe inference latency (~2–5 seconds per line) and autoregressive language model hallucinations. OpenAI Vision (`gpt-4o`) resolves this when active.
2. **Regex-Based Question Segmentation**: Answer segmentation relies on standard question identifiers (`Q1`, `1.`, `1(a)`). If a student omits question numbers or writes answers in free-flowing paragraphs, the text cannot be mapped to a rubric and is classified as `UNSTRUCTURED` for human review.
3. **Diagrammatic Visual Comprehension**: The system detects textual diagram indicators (`[diagram]`, `[figure]`) and preserves the corresponding image page, but does not perform computer vision grading of circuit diagrams, geometry sketches, or anatomical drawings.
4. **API Quota & External Dependency**: When using OpenAI models, if an account runs out of credits (`HTTP 429 insufficient_quota`), the system automatically fails over to the local TrOCR and Sentence-Transformer engines.
5. **In-Memory Result Persistence**: The current prototype stores evaluation responses in memory for `GET /evaluation/{id}` lookups, which reset upon server restart. Production deployments should attach a PostgreSQL/Redis persistence layer.
