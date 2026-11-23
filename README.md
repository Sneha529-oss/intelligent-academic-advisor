# Intelligent Academic Advisor 🎓

An intelligent, explainable web application designed for engineering students and academic mentors. It provides automated academic performance analytics, skill gap diagnosis, curriculum course recommendations, personalized career roadmaps, a Retrieval-Augmented Generation (RAG) academic chatbot for syllabus documents, and a dedicated **NLP Architecture & Evaluation** laboratory.

Built specifically as a B.Tech Computer Science & Engineering (AIML) capstone project.

---

## 1. Project Overview & Features

1. **Academic Dashboard:** Real-time metrics computed directly from institutional datasets — total enrolled students, course catalog volume, aggregate CGPA, average attendance %, placement rates, branch breakdown, and performance distributions.
2. **Student Directory & Profiling:** Comprehensive student dossiers covering academic marks, attendance, coding track (DSA problems solved, competitive programming rating, GitHub/projects), and career goals. Allows instant creation of new student profiles stored safely in MongoDB.
3. **Academic Analytics:** Deep-dive visual analytics including CGPA histograms, attendance distributions, attendance vs CGPA correlation study, subject-wise score averages across departments, and placement outcomes by branch.
4. **Competency Readiness & Skill Gap Analysis:** Automated diagnosis matching a student's technical skills against industry benchmarks for their target career role (e.g., Data Scientist, Software Engineer, Cloud Architect), identifying subject strengths, improvement areas, and skills not listed in their profile.
5. **Course Recommendation Engine:** Explainable TF-IDF Vector Space matching with cosine similarity and career bonus weighting across the 200-course catalog.
6. **3-Stage Career Roadmap:** Actionable, phased timeline (Foundation ➔ Skill Development ➔ Practical Application) with real courses mapped from the catalog and milestone project ideas.
7. **Academic AI Chatbot (RAG):** Document-grounded Q&A assistant for syllabi, academic regulations, and study materials. Uses PyMuPDF text extraction, chunking, TF-IDF vector retrieval, and Google Gemini generation with citations and regional language support (English, Hindi, Marathi).
8. **NLP Architecture & Evaluation Laboratory (`/nlp-architecture`):** Dedicated interactive showcase featuring:
   - 6-Stage End-to-End NLP Pipeline Stepper & Concept Alignment Matrix.
   - Live interactive text preprocessing (Lowercasing, punctuation removal, tokenization, stopword removal, rule-based lemmatization, and TF-IDF vector weights).
   - Supervised Sentiment Classification benchmark (MultinomialNB, LogisticRegression, LinearSVC) evaluated on stratified test split with a 3x3 Confusion Matrix, Information Extraction, and Sarcasm Diagnostics.
   - Unsupervised Curriculum Topic Modeling (Latent Dirichlet Allocation extracting Top 6 Academic Topics).
   - Multilingual NLP Engine supporting English, Hindi (हिन्दी), and Marathi (मराठी) detection, translation, and entity extraction.
   - Generative AI / RAG Prompt Engineering Inspector exposing system prompts, retrieved chunks, and model responses.

---

## 2. End-to-End NLP Pipeline Workflow

The Intelligent Academic Advisor processes unstructured and semi-structured academic texts through a 6-stage NLP pipeline:

```text
[ Raw Academic Sources ]
  │  ├── Syllabi & Regulations (PDF)
  │  ├── Student Profiles & Competencies (CSV / NoSQL)
  │  ├── Course Catalog Descriptions & Skills (CSV / NoSQL)
  │  └── Student Feedback Reviews (Text)
  ▼
[ Stage 1: Data Ingestion & Extraction ] ➔ PyMuPDF (fitz) text parsing & CSV ETL
  ▼
[ Stage 2: Text Preprocessing & Vectorization ]
  │  ├── Case Normalization & Punctuation Stripping
  │  ├── Lexical Whitespace Tokenization
  │  ├── Stopword Filtering (NLTK/sklearn stopword sets)
  │  ├── Rule-based Morphological Lemmatization & Stemming
  │  └── TF-IDF Sparse Matrix Generation (Sublinear TF, N-grams 1-2)
  ▼
[ Stage 3: Semantic Matching & Information Extraction ]
  │  ├── Cosine Similarity Distance Computation
  │  └── Academic Aspect & Entity Extraction (Skills, Roles, Topics)
  ▼
[ Stage 4: Supervised Sentiment Classification & Evaluation ]
  │  ├── 240 Curated Student Feedback Reviews (Stratified 80/20 Split)
  │  ├── Multinomial Naive Bayes vs Logistic Regression vs Linear SVC
  │  └── Sklearn Metrics: Accuracy, Weighted F1, Per-Class F1, Confusion Matrix
  ▼
[ Stage 5: Topic Modeling & Multilingual NLP ]
  │  ├── Unsupervised Latent Dirichlet Allocation (Top 6 Curriculum Themes)
  │  └── Devanagari Script Analysis & English-Hindi-Marathi Translation
  ▼
[ Stage 6: Retrieval-Augmented Generation (RAG) & Prompt Engineering ]
  │  ├── Vector Similarity Chunk Search (Top-k context passages)
  │  ├── Context Injection into System Prompt Template
  │  └── Google Gemini 2.5/1.5 Flash Synthesis with Source Citations
  ▼
[ Explainable Student UI / Dashboards ]
```

---

## 3. NLP Concepts Used & Project Alignment Matrix

| NLP Technique | Underlying Algorithm / Library | Project Module | Academic Implementation & Purpose |
| :--- | :--- | :--- | :--- |
| **TF-IDF Vectorization** | `TfidfVectorizer` (Scikit-learn, Sublinear TF, N-grams 1-2) | Course Recommendations & RAG Retrieval | Weights distinctive technical keywords higher than generic vocabulary, converting courses and syllabi into sparse vectors. |
| **Cosine Similarity** | Normalized Dot Product (Scikit-learn) | Recommendation Engine & Document Search | Computes angle between student interest vectors and course vectors to score and rank top recommendations. |
| **Supervised Sentiment Classification** | `MultinomialNB`, `LogisticRegression`, `LinearSVC` | Sentiment Evaluation Lab (`/nlp-architecture`) | Benchmarks three classifiers on 240 student feedback reviews to automatically monitor course and teaching quality. |
| **Information Extraction** | Regular Expression Pattern Matching & Noun Chunking | Skill Gap & Feedback Aspect Extraction | Identifies specific engineering skills, career benchmarks, and feedback aspects (Pedagogy, Labs, Syllabus, Grading). |
| **Unsupervised Topic Modeling** | `LatentDirichletAllocation` (LDA, Scikit-learn) | Curriculum Explorer (`/nlp-architecture`) | Uncovers 6 latent academic subject clusters across 200 courses without manual labeling. |
| **Multilingual NLP & Translation** | Unicode Devanagari Analysis + Google Gemini Multilingual API | Multilingual Advisor (`/api/nlp-lab/multilingual`, `/chatbot`) | Detects language script and translates academic guidance across English, Hindi (हिन्दी), and Marathi (मराठी). |
| **Retrieval-Augmented Generation (RAG)** | Chunk Retrieval + Gemini LLM Grounding | Academic AI Chatbot (`/chatbot`) | Grounds responses in verified university regulations and syllabus PDFs to eliminate hallucinations. |

---

## 4. Dataset Information & Preprocessing Details

The system relies on curated academic datasets:

1. **`data/student_profiles.csv` (1,200 records, 33 columns):**
   - Demographics, engineering branch (CSE, AIML, IT, DS, ECE, MECH), college tier, CGPA, attendance %, subject marks (Mathematics, Core Branch, Programming, DSA, Communication), DSA problems solved, projects, internships, certifications, target role, and technical skills.
2. **`data/courses.csv` (200 records, 10 columns):**
   - Course ID, title, description, category, skills, prerequisites, difficulty level, and syllabus links spanning programming, DSA, machine learning, cloud computing, cybersecurity, and full-stack web engineering.
3. **Curated Feedback Evaluation Dataset (240 labeled samples):**
   - Realistic student course reviews balanced across 3 classes (80 Positive, 80 Neutral, 80 Negative).
   - Preprocessed with `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)`, capturing sentiment-critical unigrams and bigrams like "not clear", "very helpful", and "average pace".
4. **Academic Documents (`documents/`):**
   - Official institutional syllabus and examination regulations in PDF format, parsed with PyMuPDF into ~500-character overlapping chunks.

---

## 5. Sentiment Classifier Benchmark Results

Evaluated on a stratified 80/20 train-test split (192 training reviews, 48 test reviews with 16 samples per class):

| Classifier | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Per-Class F1 (Pos / Neu / Neg) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (`LogisticRegression`, C=2.0)** | **100.0%** | **1.00** | **1.00** | **1.00** | **1.00 / 1.00 / 1.00** |
| **Linear Support Vector Machine (`LinearSVC`, C=1.0)** | **100.0%** | **1.00** | **1.00** | **1.00** | **1.00 / 1.00 / 1.00** |
| **Multinomial Naive Bayes (`MultinomialNB`, $\alpha$=0.3)** | **97.9%** | **0.98** | **0.98** | **0.98** | **1.00 / 0.97 / 0.97** |

### Confusion Matrix (Best Model — Logistic Regression / LinearSVC):
```text
                  Predicted Pos   Predicted Neu   Predicted Neg
Actual Positive:       16               0               0
Actual Neutral:         0              16               0
Actual Negative:        0               0              16
```

### Confusion Matrix (MultinomialNB, $\alpha$=0.3):
```text
                  Predicted Pos   Predicted Neu   Predicted Neg
Actual Positive:       16               0               0
Actual Neutral:         0              16               0
Actual Negative:        0               1              15
```

### Error & Sarcasm Diagnostics:
The system includes diagnostic heuristics for linguistic challenges:
- **Sarcasm / Irony:** Flags positive polarity words ("loved", "great job", "brilliant") when appearing in frustrating contexts (e.g. debugging syntax errors all night).
- **Contrastive Conjunctions:** Detects sentence polarity shifts via "but", "however", and "although".
- **Double Negation:** Normalizes negated phrases (e.g. "not bad at all", "hardly disappointed").

---

## 6. Technology Stack & Live Deployment Architecture

```text
               +-------------------------------------------+
               |             Client Browser                |
               |      (React 19 + TypeScript + Vite)       |
               +-------------------------------------------+
                                     │
                             Axios (/api/*)
                                     ▼
               +-------------------------------------------+
               |              FastAPI Backend              |
               |       (Python 3.12 Asynchronous ASGI)     |
               +-------------------------------------------+
                     │               │               │
       PyMongo / Atlas               │          Google Gemini API
             ▼                       ▼               ▼
      +--------------+      +-----------------+  +-----------------+
      | MongoDB      |      | Scikit-learn    |  | Gemini 2.5 /    |
      | Atlas Cloud  |      | TF-IDF / LDA /  |  | 1.5 Flash       |
      | Database     |      | PyMuPDF Storage |  | Generative LLM  |
      +--------------+      +-----------------+  +-----------------+
```

- **Frontend:** React 19, TypeScript, Vite 5, Tailwind CSS, Lucide React, Recharts, MarkdownRenderer.
- **Backend:** FastAPI, Uvicorn, Scikit-learn 1.3+, PyMuPDF (fitz), Pandas, NumPy, Pydantic, python-dotenv.
- **Database:** MongoDB Atlas cloud connection (with auto-seeding on startup if collections have fewer than 10 documents).
- **Deployment Compatibility:** Fully compatible with Render deployment configurations (Render Web Service using `uvicorn main:app --host 0.0.0.0 --port $PORT` and static frontend hosting).

---

## 7. Project Folder Structure

```text
intelligent-academic-advisor/
├── backend/
│   ├── routes/
│   │   ├── dashboard.py         # Dashboard statistics & distributions
│   │   ├── students.py          # Student CRUD operations
│   │   ├── analytics.py         # Academic analytics aggregations
│   │   ├── recommendations.py   # Course recommendation endpoints
│   │   ├── skill_gap.py         # Skill gap diagnosis
│   │   ├── roadmap.py           # 3-stage roadmap generation
│   │   ├── chatbot.py           # PDF upload, RAG Q&A, and regional language chat
│   │   └── nlp_features.py      # Preprocessing, Sentiment, Topics, Multilingual, & Inspector
│   ├── services/
│   │   ├── nlp_service.py                    # Preprocessing, lemmatization, TF-IDF weights
│   │   ├── sentiment_evaluation_service.py   # 120-review dataset, 3 classifiers, metrics
│   │   ├── topic_modeling_service.py         # Latent Dirichlet Allocation (Top 6 topics)
│   │   ├── multilingual_service.py           # En/Hi/Mr detection, translation, entities
│   │   ├── rag_service.py                    # PDF chunking, TF-IDF retrieval, Gemini prompt
│   │   ├── recommendation_engine.py          # TF-IDF cosine similarity matcher
│   │   ├── skill_gap_service.py              # Career benchmark mapper
│   │   └── roadmap_service.py                # Phased timeline generator
│   ├── database.py              # MongoDB connection manager
│   ├── import_data.py           # Idempotent CSV-to-MongoDB import script
│   ├── main.py                  # FastAPI application entry point
│   └── requirements.txt         # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   └── client.ts        # Axios client instance (baseURL: '/api')
│   │   ├── components/
│   │   │   ├── Layout.tsx       # Sidebar & navigation frame (with NLP Architecture link)
│   │   │   ├── MarkdownRenderer.tsx # Clean formatting for headers, bold, and citations
│   │   │   ├── LoadingSpinner.tsx
│   │   │   └── ErrorMessage.tsx
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx              # Page 1: Dashboard
│   │   │   ├── StudentProfile.tsx         # Page 2: Student Profile & Directory
│   │   │   ├── Analytics.tsx              # Page 3: Academic Analytics
│   │   │   ├── SkillGap.tsx               # Page 4: Skill Gap Analysis
│   │   │   ├── CourseRecommendations.tsx  # Page 5: Course Recommendations
│   │   │   ├── CareerRoadmap.tsx          # Page 6: Career Roadmap
│   │   │   ├── Chatbot.tsx                # Page 7: Academic AI Chatbot (with En/Hi/Mr)
│   │   │   ├── NlpArchitecture.tsx        # Page 8: Dedicated NLP Architecture & Evaluation
│   │   │   └── About.tsx                  # Page 9: About & Capstone Overview
│   │   ├── App.tsx              # Route configuration
│   │   ├── main.tsx             # React entry point
│   │   └── index.css            # Tailwind styling
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── data/
│   ├── student_profiles.csv     # 1,200 student records
│   └── courses.csv              # 200 course catalog records
├── documents/
│   └── sample_academic_regulations_syllabus.pdf  # Sample syllabus for RAG
├── .env.example
├── .env
├── .gitignore
└── README.md
```

---

## 8. Setup & Running Instructions (Windows PowerShell)

### Prerequisites
- **Python 3.10+** (Tested on Python 3.12 / 3.13)
- **Node.js 18+** (Tested on Node.js v20+)
- **MongoDB** running locally or a MongoDB Atlas connection string in `.env`

---

### Step 1: Configure Environment Variables

In `backend/` (or project root), customize `.env`:

```env
MONGODB_URI=mongodb+srv://<user>:<password>@nlpproject.wbhehla.mongodb.net/
DATABASE_NAME=intelligent_academic_advisor
LLM_API_KEY=your_gemini_api_key_here
LLM_PROVIDER=gemini
```

> **Note:** The entire system operates in offline mode without an API key, automatically falling back to verified document passage extraction, rule-based lemmatization, scikit-learn models, and academic dictionaries. Setting `LLM_API_KEY` enables generative Gemini LLM syntheses and live translations.

---

### Step 2: Set Up & Start Backend

Open a PowerShell terminal:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The backend starts at: **`http://localhost:8000`**
- Swagger API documentation: **`http://localhost:8000/docs`**
- API health check: **`http://localhost:8000/api/health`**

---

### Step 3: Set Up & Start Frontend

Open a second PowerShell terminal:

```powershell
cd frontend
npm install
npm run dev
```

The frontend dashboard opens at: **`http://localhost:5173`**

---

## 9. Demonstration & Viva Walkthrough

To present during evaluation or viva examination:

1. **End-to-End NLP Pipeline (`/nlp-architecture`):**
   - Click each of the 6 stages in the interactive stepper to explain how data transforms from raw PDFs into TF-IDF vectors, topic clusters, and generative prompts.
   - Walk through the Concept Alignment Matrix to show academic rigor and clear methodology.
2. **Interactive Text Preprocessing (`/nlp-architecture`):**
   - Enter any sentence or click quick presets to show the sequential transformations: Raw Text ➔ Cleaned ➔ Tokens ➔ Stopword Filtering ➔ Rule-Based Lemmatization ➔ Top TF-IDF Vector Weights.
3. **Sentiment Classification & Model Evaluation (`/nlp-architecture`):**
   - Review the Model Comparison Table comparing MultinomialNB, Logistic Regression, and LinearSVC on the stratified 80/20 test split.
   - Point out the 3x3 Confusion Matrix highlighting true vs predicted classes.
   - Test custom feedback reviews to display real-time predictions from all 3 models, Information Extraction of academic aspects, and the Sarcasm/Error analysis card.
4. **Curriculum Topic Modeling & Multilingual NLP (`/nlp-architecture`):**
   - Showcase the 6 academic topics extracted via LDA from the course catalog, showing weighted terms and top matching courses.
   - Test the Multilingual translator by typing queries in Marathi, Hindi, or English to see real-time script detection, translation, and entity extraction.
5. **Generative AI & Prompt Engineering Inspector (`/nlp-architecture`):**
   - Inspect the exact constructed RAG prompt with injected context chunks, cosine match scores, and synthesized Gemini response.
6. **Academic AI Chatbot (`/chatbot`):**
   - Select English, Hindi (हिन्दी), or Marathi (मराठी) from the language selector in the chat header.
   - Click "Load Sample Syllabus" to index the regulations PDF, then ask questions to demonstrate citation-backed, markdown-formatted answers without raw asterisks.

---

## 10. License & Academic Disclaimer

Developed as part of the B.Tech Computer Science and Engineering (AIML) capstone curriculum. Datasets and seed catalogs are curated for educational evaluation and system benchmarking.
