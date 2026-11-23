import { useState, useEffect } from 'react';
import {
  Layers,
  Sparkles,
  Cpu,
  Brain,
  Languages,
  CheckCircle,
  AlertCircle,
  ArrowRight,
  BookOpen,
  Code2,
  Table,
  Zap,
  BarChart3,
  Search,
  MessageSquare,
  FileText,
  Compass,
  RefreshCw,
  Lightbulb,
} from 'lucide-react';
import api from '../api/client';
import MarkdownRenderer from '../components/MarkdownRenderer';

// Pipeline Stepper Stages
const PIPELINE_STAGES = [
  {
    stage: 1,
    title: 'Data Ingestion & Text Extraction',
    subtitle: 'PyMuPDF Parser & Multi-Source Synthesis',
    badge: 'Input Phase',
    color: 'from-blue-600 to-indigo-600',
    description:
      'Extracts raw, semi-structured, and unstructured academic data from institutional syllabus PDFs, student profiles, course catalogs, and student feedback transcripts.',
    technique: 'PyMuPDF (fitz) + Pandas DataFrame ETL',
    artifacts: ['syllabus.pdf ➔ Raw Text Chunks', 'courses.csv ➔ Corpus Text', 'feedback ➔ Review Tokens'],
    projectRole: 'Powers the document knowledge base for syllabus querying and course catalog search.',
  },
  {
    stage: 2,
    title: 'Text Preprocessing & Vectorization',
    subtitle: 'Lexical Normalization & TF-IDF Vector Spaces',
    badge: 'Vectorization Phase',
    color: 'from-indigo-600 to-purple-600',
    description:
      'Applies systematic text transformations: case normalization, punctuation filtering, whitespace tokenization, stopword elimination, rule-based lemmatization, and TF-IDF sparse matrix construction.',
    technique: 'TfidfVectorizer (Sublinear TF, N-grams 1-2) + Morphological Lemmatizer',
    artifacts: ['Clean Tokens', 'Lemmatized Stems', 'TF-IDF Sparse Matrices (N x D)'],
    projectRole: 'Transforms high-dimensional text into numerical vector spaces for similarity and classification.',
  },
  {
    stage: 3,
    title: 'Semantic Matching & Information Extraction',
    subtitle: 'Cosine Similarity & Academic Entity Recognition',
    badge: 'Matching Phase',
    color: 'from-purple-600 to-pink-600',
    description:
      'Computes vector space cosine distances between student technical profiles and course descriptions, extracting target competencies, career gap deficits, and academic aspects.',
    technique: 'Cosine Similarity [0.0 - 1.0] + Aspect Entity Extractor',
    artifacts: ['Relevance Ranked Courses', 'Skill Gap Delta Vectors', 'Aspect Keyphrases'],
    projectRole: 'Drives the Course Recommendation Engine and Skill Gap Diagnostic modules.',
  },
  {
    stage: 4,
    title: 'Sentiment Classification & Model Evaluation',
    subtitle: 'Supervised Learning & Error Diagnostics',
    badge: 'Supervised ML',
    color: 'from-emerald-600 to-teal-600',
    description:
      'Trains and benchmarks three supervised classifiers (MultinomialNB, LogisticRegression, LinearSVC) on 240 curated student course feedback reviews across Positive, Neutral, and Negative classes.',
    technique: 'Stratified 80/20 Split + Scikit-learn Evaluation Suite',
    artifacts: ['Accuracy & F1 Metrics', '3x3 Confusion Matrix', 'Sarcasm/Irony Flags'],
    projectRole: 'Automates qualitative student feedback auditing and course satisfaction analytics.',
  },
  {
    stage: 5,
    title: 'Topic Modeling & Multilingual Processing',
    subtitle: 'Unsupervised LDA Clustering & Regional Translation',
    badge: 'Unsupervised & Multilingual',
    color: 'from-amber-600 to-orange-600',
    description:
      'Discovers 6 latent academic themes across the course catalog with Latent Dirichlet Allocation (LDA), coupled with bilingual Devanagari detection and English-Hindi-Marathi translation.',
    technique: 'LatentDirichletAllocation (Batch variational Bayes) + Gemini / Rule Fallback',
    artifacts: ['Top 6 Latent Topics', '8-Keyword Topic Clusters', 'Hindi & Marathi Transcripts'],
    projectRole: 'Enables cross-curriculum exploration and democratizes access for regional language learners.',
  },
  {
    stage: 6,
    title: 'Retrieval-Augmented Generation (RAG)',
    subtitle: 'Context Retrieval & Gemini Prompt Engineering',
    badge: 'GenAI & RAG',
    color: 'from-rose-600 to-red-600',
    description:
      'Combines TF-IDF semantic chunk retrieval with tailored system prompt engineering, feeding retrieved syllabus passages to Google Gemini for hallucination-resistant academic advising.',
    technique: 'Context Injection + Google Gemini 2.5/1.5 Flash Synthesis',
    artifacts: ['Retrieved Context Chunks', 'Engineered System Prompt', 'Cited Academic Answers'],
    projectRole: 'Powers the Academic AI Chatbot with verified citations and page-grounded advice.',
  },
];

// NLP Concept Alignment Matrix
const NLP_CONCEPTS_MATRIX = [
  {
    concept: 'TF-IDF Vectorization',
    algorithm: 'Term Frequency - Inverse Document Frequency (Scikit-learn)',
    module: 'Course Recommendations & RAG Retrieval',
    application: 'Weights domain-specific engineering keywords higher than generic words to convert syllabi and student skills into sparse numerical vectors.',
  },
  {
    concept: 'Cosine Similarity',
    algorithm: 'Vector Dot Product / L2 Norms',
    module: 'Recommendation Engine & RAG Search',
    application: 'Calculates directional cosine angles between student interest vectors and course description vectors to deliver personalized top-N rankings.',
  },
  {
    concept: 'Supervised Sentiment Classification',
    algorithm: 'MultinomialNB, Logistic Regression, LinearSVC',
    module: 'Student Feedback Evaluation Lab',
    application: 'Classifies qualitative student course reviews into Positive, Neutral, and Negative sentiments to diagnose curriculum health.',
  },
  {
    concept: 'Information Extraction (Aspects & Entities)',
    algorithm: 'Rule-based Pattern Matching & Noun Phrase Chunking',
    module: 'Skill Gap & Feedback Analysis',
    application: 'Extracts technical competencies, target job roles, course aspects (labs, exams, teaching quality), and prerequisite entities.',
  },
  {
    concept: 'Unsupervised Topic Modeling',
    algorithm: 'Latent Dirichlet Allocation (LDA with Variational Bayes)',
    module: 'Curriculum Exploration Engine',
    application: 'Uncovers 6 latent academic themes across 200 catalog courses without manual labeling, grouping courses by semantic affinity.',
  },
  {
    concept: 'Multilingual NLP & Translation',
    algorithm: 'Devanagari Script Analysis + Google Gemini Multilingual API',
    module: 'Multilingual Academic Advisor',
    application: 'Detects and translates queries across English, Hindi (हिन्दी), and Marathi (मराठी) to support regional engineering students.',
  },
  {
    concept: 'Retrieval-Augmented Generation (RAG)',
    algorithm: 'TF-IDF Semantic Chunk Retrieval + Gemini LLM Grounding',
    module: 'Academic AI Chatbot',
    application: 'Retrieves verified passages from university regulations and syllabus PDFs to synthesize accurate, citation-backed answers.',
  },
];

export default function NlpArchitecture() {
  const [activeStage, setActiveStage] = useState(1);

  // Section 2: Preprocessing State
  const [prepInput, setPrepInput] = useState(
    'The Professor explained Dynamic Programming algorithms and graph theory with practical Python coding exercises!'
  );
  const [prepResult, setPrepResult] = useState<any>(null);
  const [prepLoading, setPrepLoading] = useState(false);

  // Section 3: Sentiment Evaluation State
  const [evalOverview, setEvalOverview] = useState<any>(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [feedbackInput, setFeedbackInput] = useState(
    'The professor explained complex dynamic programming concepts clearly with fantastic hands-on coding labs.'
  );
  const [feedbackResult, setFeedbackResult] = useState<any>(null);
  const [feedbackLoading, setFeedbackLoading] = useState(false);

  // Section 4: Topic Modeling & Multilingual State
  const [topicsData, setTopicsData] = useState<any>(null);
  const [topicsLoading, setTopicsLoading] = useState(false);
  const [multiInput, setMultiInput] = useState(
    'मला संगणक अभियांत्रिकी आणि मशीन लर्निंग अभ्यासक्रम माहिती पाहिजे'
  );
  const [targetLang, setTargetLang] = useState('en');
  const [multiResult, setMultiResult] = useState<any>(null);
  const [multiLoading, setMultiLoading] = useState(false);

  // Section 5: RAG Inspector State
  const [ragQuery, setRagQuery] = useState('What are the attendance requirements and exam eligibility criteria?');
  const [ragLang, setRagLang] = useState('en');
  const [ragResult, setRagResult] = useState<any>(null);
  const [ragLoading, setRagLoading] = useState(false);

  // Initial Data Fetching
  useEffect(() => {
    runPreprocessing(prepInput);
    fetchSentimentEvaluation();
    runFeedbackAnalysis(feedbackInput);
    fetchTopics();
    runMultilingual(multiInput, targetLang);
    runRagInspect(ragQuery, ragLang);
  }, []);

  const runPreprocessing = (textToProcess: string) => {
    if (!textToProcess.trim()) return;
    setPrepLoading(true);
    api.post('/nlp/preprocess', { text: textToProcess })
      .then(res => setPrepResult(res.data))
      .catch(() => {})
      .finally(() => setPrepLoading(false));
  };

  const fetchSentimentEvaluation = () => {
    setEvalLoading(true);
    api.get('/nlp/sentiment-eval')
      .then(res => setEvalOverview(res.data))
      .catch(() => {})
      .finally(() => setEvalLoading(false));
  };

  const runFeedbackAnalysis = (textToAnalyze: string) => {
    if (!textToAnalyze.trim()) return;
    setFeedbackLoading(true);
    api.post('/nlp/sentiment-analyze', { text: textToAnalyze })
      .then(res => setFeedbackResult(res.data))
      .catch(() => {})
      .finally(() => setFeedbackLoading(false));
  };

  const fetchTopics = () => {
    setTopicsLoading(true);
    api.get('/nlp/topics')
      .then(res => setTopicsData(res.data))
      .catch(() => {})
      .finally(() => setTopicsLoading(false));
  };

  const runMultilingual = (textToTranslate: string, targetLanguage: string) => {
    if (!textToTranslate.trim()) return;
    setMultiLoading(true);
    api.post('/nlp/multilingual', { text: textToTranslate, target_language: targetLanguage })
      .then(res => setMultiResult(res.data))
      .catch(() => {})
      .finally(() => setMultiLoading(false));
  };

  const runRagInspect = (queryText: string, lang: string) => {
    if (!queryText.trim()) return;
    setRagLoading(true);
    api.post('/nlp/rag-inspect', { question: queryText, language: lang })
      .then(res => setRagResult(res.data))
      .catch(() => {})
      .finally(() => setRagLoading(false));
  };

  const currentStageData = PIPELINE_STAGES.find(s => s.stage === activeStage) || PIPELINE_STAGES[0];

  return (
    <div className="space-y-10 max-w-7xl mx-auto pb-16">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 rounded-2xl p-6 sm:p-8 text-white shadow-md border border-slate-800">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-indigo-500/20 border border-indigo-400/30 rounded-full text-xs font-semibold text-indigo-300 mb-3">
              <Sparkles className="h-3.5 w-3.5" />
              <span>Full-Stack Natural Language Processing Engine</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight">
              NLP Architecture & Evaluation
            </h1>
            <p className="text-sm text-slate-300 mt-2 max-w-3xl leading-relaxed">
              Comprehensive end-to-end NLP workflow documentation, interactive vectorization pipeline,
              supervised sentiment classifiers evaluation with confusion matrix, unsupervised LDA topic clustering,
              regional multilingual translation, and RAG prompt engineering inspection.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 shrink-0">
            <span className="px-3 py-1.5 bg-slate-800 border border-slate-700 text-xs rounded-lg text-slate-200 font-mono">
              Scikit-Learn 1.3+
            </span>
            <span className="px-3 py-1.5 bg-slate-800 border border-slate-700 text-xs rounded-lg text-slate-200 font-mono">
              FastAPI + PyMuPDF
            </span>
            <span className="px-3 py-1.5 bg-indigo-900/60 border border-indigo-500/40 text-xs rounded-lg text-indigo-200 font-mono">
              Google Gemini RAG
            </span>
          </div>
        </div>
      </div>

      {/* =========================================================================
          SECTION 1: OVERALL NLP WORKFLOW & CONCEPT ALIGNMENT
      ========================================================================= */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-gray-200 pb-3">
          <div>
            <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
              <Layers className="h-5 w-5 text-indigo-600" />
              <span>1. End-to-End NLP Pipeline & Concept Alignment</span>
            </h2>
            <p className="text-xs text-gray-500 mt-0.5">
              Visual pipeline stepper tracking academic data flow and concept mapping across the application.
            </p>
          </div>
          <span className="text-xs bg-indigo-50 text-indigo-700 px-2.5 py-1 rounded-md font-semibold">
            6 Sequential Stages
          </span>
        </div>

        {/* Pipeline Stepper Navigation */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
          {PIPELINE_STAGES.map(s => {
            const isActive = s.stage === activeStage;
            return (
              <button
                key={s.stage}
                onClick={() => setActiveStage(s.stage)}
                className={`p-3 rounded-xl border text-left transition-all cursor-pointer ${
                  isActive
                    ? 'bg-indigo-600 text-white border-indigo-600 shadow-md ring-2 ring-indigo-300'
                    : 'bg-white text-gray-700 border-gray-200 hover:border-indigo-300 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span
                    className={`h-5 w-5 rounded-full text-[10px] font-bold flex items-center justify-center ${
                      isActive ? 'bg-white text-indigo-600' : 'bg-slate-100 text-gray-700'
                    }`}
                  >
                    {s.stage}
                  </span>
                  <span className={`text-[10px] uppercase font-bold tracking-wider ${isActive ? 'text-indigo-100' : 'text-gray-600'}`}>
                    {s.badge}
                  </span>
                </div>
                <div className="font-semibold text-xs leading-tight line-clamp-1">{s.title}</div>
              </button>
            );
          })}
        </div>

        {/* Selected Stage Detail Card */}
        <div className="bg-white rounded-xl border border-gray-200 p-5 sm:p-6 shadow-sm">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-4 border-b border-gray-100">
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-0.5 bg-indigo-100 text-indigo-800 text-xs font-bold rounded">
                  Stage {currentStageData.stage} of 6
                </span>
                <span className="text-xs text-gray-600 font-medium">{currentStageData.subtitle}</span>
              </div>
              <h3 className="text-lg font-bold text-gray-900 mt-1">{currentStageData.title}</h3>
            </div>
            <div className="text-xs font-mono bg-slate-50 border border-slate-200 text-slate-700 px-3 py-1.5 rounded-lg">
              <strong>Core Algorithm:</strong> {currentStageData.technique}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-5">
            <div className="md:col-span-2 space-y-3">
              <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider">Stage Overview & Processing</h4>
              <p className="text-sm text-gray-700 leading-relaxed">{currentStageData.description}</p>
              <div className="bg-slate-50 rounded-lg p-3.5 border border-slate-200">
                <span className="text-xs font-semibold text-indigo-900 block mb-1">Role in Intelligent Academic Advisor:</span>
                <span className="text-xs text-gray-600">{currentStageData.projectRole}</span>
              </div>
            </div>

            <div>
              <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2.5">Key Output Artifacts</h4>
              <ul className="space-y-2">
                {currentStageData.artifacts.map((art, idx) => (
                  <li key={idx} className="flex items-center gap-2 text-xs text-gray-700 bg-slate-50 p-2 rounded border border-slate-200">
                    <CheckCircle className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
                    <span className="font-mono">{art}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* NLP Concept Alignment Matrix Table */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <div className="p-4 bg-slate-50 border-b border-gray-200 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Table className="h-4 w-4 text-indigo-600" />
              <h3 className="font-semibold text-gray-900 text-sm">NLP Concepts Used & Project Alignment Matrix</h3>
            </div>
            <span className="text-xs text-gray-500">Mapping curriculum concepts to implementation</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-100 text-gray-700 uppercase font-semibold border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3">NLP Technique</th>
                  <th className="px-4 py-3">Algorithm / Library</th>
                  <th className="px-4 py-3">Project Module</th>
                  <th className="px-4 py-3">Academic Implementation & Purpose</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {NLP_CONCEPTS_MATRIX.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-50/80 transition-colors">
                    <td className="px-4 py-3 font-semibold text-indigo-700 whitespace-nowrap">{row.concept}</td>
                    <td className="px-4 py-3 font-mono text-gray-600">{row.algorithm}</td>
                    <td className="px-4 py-3">
                      <span className="px-2 py-0.5 bg-slate-100 text-slate-800 rounded font-medium text-[11px]">
                        {row.module}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-gray-600 leading-relaxed">{row.application}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* =========================================================================
          SECTION 2: INTERACTIVE PREPROCESSING & VECTORIZATION PIPELINE
      ========================================================================= */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-gray-200 pb-3">
          <div>
            <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
              <Code2 className="h-5 w-5 text-indigo-600" />
              <span>2. Interactive Text Preprocessing & Vectorization Pipeline</span>
            </h2>
            <p className="text-xs text-gray-500 mt-0.5">
              Live execution of `nlp_service.py`: Raw Text ➔ Cleaned ➔ Tokens ➔ Stopword Removal ➔ Lemmatization ➔ TF-IDF Vector Weights.
            </p>
          </div>
          <span className="text-xs bg-purple-50 text-purple-700 px-2.5 py-1 rounded-md font-semibold">
            Live Interactive Sandbox
          </span>
        </div>

        <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm space-y-4">
          <label className="block text-xs font-semibold text-gray-700">
            Enter Academic Text to Trace Preprocessing Pipeline:
          </label>
          <div className="flex flex-col sm:flex-row gap-2">
            <input
              type="text"
              value={prepInput}
              onChange={e => setPrepInput(e.target.value)}
              placeholder="Type any engineering course description or syllabus sentence..."
              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none"
            />
            <button
              onClick={() => runPreprocessing(prepInput)}
              disabled={prepLoading}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 cursor-pointer disabled:opacity-50"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${prepLoading ? 'animate-spin' : ''}`} />
              <span>{prepLoading ? 'Vectorizing...' : 'Execute Pipeline'}</span>
            </button>
          </div>

          {/* Quick preset buttons */}
          <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500">
            <span className="font-semibold text-gray-700">Quick Presets:</span>
            {[
              'Dynamic Programming algorithms and graph theory with Python coding',
              'Operating Systems memory virtualization and multithreaded concurrency',
              'Deep Neural Networks, backpropagation optimization, and computer vision models',
            ].map((preset, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setPrepInput(preset);
                  runPreprocessing(preset);
                }}
                className="px-2.5 py-1 bg-slate-100 hover:bg-indigo-50 text-slate-700 hover:text-indigo-700 rounded border border-slate-200 text-[11px] cursor-pointer transition-colors"
              >
                {preset.slice(0, 32)}...
              </button>
            ))}
          </div>

          {/* Pipeline Step-by-Step Visualization */}
          {prepResult && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-3">
              {/* Step 1: Raw & Cleaned */}
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 space-y-2">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
                  <span>Step 1: Lowercase & Strip Punctuation</span>
                  <span className="text-[10px] bg-slate-200 px-1.5 py-0.5 rounded font-mono">Clean Text</span>
                </div>
                <div className="text-xs bg-white p-2.5 rounded border border-slate-200 text-gray-800 font-mono leading-relaxed break-words">
                  {prepResult.no_punctuation || 'Empty'}
                </div>
              </div>

              {/* Step 2: Tokenization */}
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 space-y-2">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
                  <span>Step 2: Lexical Tokenization</span>
                  <span className="text-[10px] bg-slate-200 px-1.5 py-0.5 rounded font-mono">
                    {prepResult.tokens?.length || 0} tokens
                  </span>
                </div>
                <div className="flex flex-wrap gap-1 max-h-24 overflow-y-auto p-1 bg-white rounded border border-slate-200">
                  {prepResult.tokens?.map((tok: string, i: number) => (
                    <span key={i} className="px-1.5 py-0.5 bg-indigo-50 text-indigo-700 rounded text-[11px] font-mono">
                      {tok}
                    </span>
                  ))}
                </div>
              </div>

              {/* Step 3: Stopword Removal */}
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 space-y-2">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
                  <span>Step 3: English Stopwords Filtered</span>
                  <span className="text-[10px] bg-emerald-100 text-emerald-800 px-1.5 py-0.5 rounded font-mono">
                    {prepResult.no_stopwords?.length || 0} left
                  </span>
                </div>
                <div className="flex flex-wrap gap-1 max-h-24 overflow-y-auto p-1 bg-white rounded border border-slate-200">
                  {prepResult.no_stopwords?.map((w: string, i: number) => (
                    <span key={i} className="px-1.5 py-0.5 bg-emerald-50 text-emerald-700 rounded text-[11px] font-mono">
                      {w}
                    </span>
                  ))}
                </div>
              </div>

              {/* Step 4: Rule-Based Lemmatization */}
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 space-y-2">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
                  <span>Step 4: Lemmatization / Stemming</span>
                  <span className="text-[10px] bg-purple-100 text-purple-800 px-1.5 py-0.5 rounded font-mono">
                    Stemmed
                  </span>
                </div>
                <div className="flex flex-wrap gap-1 max-h-24 overflow-y-auto p-1 bg-white rounded border border-slate-200">
                  {prepResult.lemmatized_tokens?.map((stem: string, i: number) => (
                    <span key={i} className="px-1.5 py-0.5 bg-purple-50 text-purple-700 rounded text-[11px] font-mono">
                      {stem}
                    </span>
                  ))}
                </div>
              </div>

              {/* Step 5: Final Normalized Vector String */}
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 space-y-2">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
                  <span>Step 5: Final Vector-Ready String</span>
                  <span className="text-[10px] bg-indigo-100 text-indigo-800 px-1.5 py-0.5 rounded font-mono">Clean String</span>
                </div>
                <div className="text-xs bg-white p-2.5 rounded border border-slate-200 text-gray-800 font-mono leading-relaxed break-words">
                  {prepResult.final || 'Empty'}
                </div>
              </div>

              {/* Step 6: Top TF-IDF Vector Weights */}
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 space-y-2">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
                  <span>Step 6: Top TF-IDF Vector Weights</span>
                  <span className="text-[10px] bg-amber-100 text-amber-800 px-1.5 py-0.5 rounded font-mono">Term Weights</span>
                </div>
                <div className="space-y-1.5 max-h-24 overflow-y-auto p-1 bg-white rounded border border-slate-200">
                  {prepResult.vector_weights && prepResult.vector_weights.length > 0 ? (
                    prepResult.vector_weights.slice(0, 4).map((vw: any, i: number) => (
                      <div key={i} className="flex items-center justify-between text-[11px] font-mono">
                        <span className="text-gray-700 truncate max-w-[130px]">{vw.term}</span>
                        <span className="text-indigo-600 font-semibold">{vw.weight}</span>
                      </div>
                    ))
                  ) : (
                    <span className="text-[11px] text-gray-400 italic">No terms vectorizable</span>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* =========================================================================
          SECTION 3: SENTIMENT CLASSIFICATION & MODEL EVALUATION
      ========================================================================= */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-gray-200 pb-3">
          <div>
            <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-indigo-600" />
              <span>3. Sentiment Analysis, Information Extraction & Model Evaluation</span>
            </h2>
            <p className="text-xs text-gray-500 mt-0.5">
              Supervised benchmarking of 3 Scikit-learn TF-IDF Classifiers on {evalOverview?.dataset_size || 240} curated student course feedback reviews (Stratified 80/20 Split).
            </p>
          </div>
          <span className="text-xs bg-emerald-50 text-emerald-700 px-2.5 py-1 rounded-md font-semibold">
            Real Evaluation Metrics
          </span>
        </div>

        {/* Dataset Stats & Model Comparison Table */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: Model Comparison Table (8 cols) */}
          <div className="lg:col-span-8 bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="p-4 bg-slate-50 border-b border-gray-200 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-indigo-600" />
                <h3 className="font-semibold text-gray-900 text-sm">Model Comparison Table (Test Split: {evalOverview?.test_size || 48} Samples)</h3>
              </div>
              <span className="text-xs font-mono text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded font-semibold">
                Best: {evalOverview?.best_model || 'Logistic Regression'}
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-100 text-gray-700 uppercase font-semibold border-b border-gray-200">
                  <tr>
                    <th className="px-4 py-3">Classifier</th>
                    <th className="px-3 py-3 text-center">Accuracy</th>
                    <th className="px-3 py-3 text-center">Precision (W)</th>
                    <th className="px-3 py-3 text-center">Recall (W)</th>
                    <th className="px-3 py-3 text-center">F1-Score (W)</th>
                    <th className="px-4 py-3 text-center">Per-Class F1 (Pos / Neu / Neg)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {evalOverview?.models ? (
                    Object.values(evalOverview.models).map((m: any, idx: number) => {
                      const isBest = m.name === evalOverview.best_model;
                      const posF1 = m.per_class?.Positive?.f1_score ?? 0;
                      const neuF1 = m.per_class?.Neutral?.f1_score ?? 0;
                      const negF1 = m.per_class?.Negative?.f1_score ?? 0;

                      return (
                        <tr key={idx} className={isBest ? 'bg-indigo-50/40 font-medium' : 'hover:bg-slate-50'}>
                          <td className="px-4 py-3 font-semibold text-gray-900 flex items-center gap-1.5">
                            {m.name}
                            {isBest && (
                              <span className="text-[10px] bg-indigo-600 text-white px-1.5 py-0.2 rounded font-bold">
                                BEST
                              </span>
                            )}
                          </td>
                          <td className="px-3 py-3 text-center font-mono">{(m.accuracy * 100).toFixed(1)}%</td>
                          <td className="px-3 py-3 text-center font-mono">{m.precision}</td>
                          <td className="px-3 py-3 text-center font-mono">{m.recall}</td>
                          <td className="px-3 py-3 text-center font-mono font-bold text-indigo-700">{m.f1_score}</td>
                          <td className="px-4 py-3 text-center font-mono text-[11px]">
                            <span className="text-emerald-700 font-semibold">{posF1}</span> /{' '}
                            <span className="text-amber-700 font-semibold">{neuF1}</span> /{' '}
                            <span className="text-rose-700 font-semibold">{negF1}</span>
                          </td>
                        </tr>
                      );
                    })
                  ) : (
                    <tr>
                      <td colSpan={6} className="p-4 text-center text-gray-400 italic">
                        Loading classifier benchmarks...
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            <div className="p-3 bg-slate-50 border-t border-gray-200 text-xs text-gray-500 flex flex-wrap items-center justify-between gap-2">
              <span>
                <strong>Dataset:</strong> {evalOverview?.dataset_size || 240} Curated Student Reviews ({evalOverview?.class_distribution?.Positive || 80} Positive, {evalOverview?.class_distribution?.Neutral || 80} Neutral, {evalOverview?.class_distribution?.Negative || 80} Negative)
              </span>
              <span>
                <strong>Split:</strong> 80% Train ({evalOverview?.train_size || 192}), 20% Test ({evalOverview?.test_size || 48}) Stratified
              </span>
            </div>
          </div>

          {/* Right: Confusion Matrix Card (4 cols) */}
          <div className="lg:col-span-4 bg-white rounded-xl border border-gray-200 p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-gray-900 text-sm">3x3 Confusion Matrix</h3>
              <span className="text-[11px] text-gray-500 font-mono">Best Model</span>
            </div>
            <p className="text-xs text-gray-500">
              Evaluated on the {evalOverview?.test_size || 48}-sample stratified test set ({evalOverview?.test_size ? evalOverview.test_size / 3 : 16} Positive, {evalOverview?.test_size ? evalOverview.test_size / 3 : 16} Neutral, {evalOverview?.test_size ? evalOverview.test_size / 3 : 16} Negative samples).
            </p>

            {/* Matrix Heatmap Grid */}
            {evalOverview?.best_confusion_matrix ? (
              <div className="space-y-2">
                <div className="grid grid-cols-4 gap-1 text-center text-[10px] font-bold text-gray-500 uppercase">
                  <span>Actual \ Pred</span>
                  <span className="text-emerald-700">Pos</span>
                  <span className="text-amber-700">Neu</span>
                  <span className="text-rose-700">Neg</span>
                </div>
                {['Positive', 'Neutral', 'Negative'].map((rowLabel, rIdx) => {
                  const rowData = evalOverview.best_confusion_matrix.matrix[rIdx] || [0, 0, 0];
                  return (
                    <div key={rIdx} className="grid grid-cols-4 gap-1 text-center items-center">
                      <span className="text-xs font-semibold text-gray-600 text-right pr-1">
                        {rowLabel.slice(0, 3)}
                      </span>
                      {rowData.map((val: number, cIdx: number) => {
                        const isDiag = rIdx === cIdx;
                        return (
                          <div
                            key={cIdx}
                            className={`p-2.5 rounded font-mono text-xs font-bold ${
                              isDiag
                                ? 'bg-indigo-600 text-white shadow-sm'
                                : val > 0
                                ? 'bg-amber-100 text-amber-900'
                                : 'bg-slate-100 text-slate-400'
                            }`}
                          >
                            {val}
                          </div>
                        );
                      })}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="p-4 text-center text-xs text-gray-400 italic">Loading matrix...</div>
            )}

            <div className="text-[11px] text-gray-500 bg-slate-50 p-2.5 rounded border border-slate-200">
              <span className="font-semibold text-indigo-700">Diagonal elements:</span> Correctly classified samples.
              Off-diagonal elements represent misclassifications (e.g., Neutral reviews confused with Negative).
            </div>
          </div>
        </div>

        {/* Live "Analyze Student Feedback" Box */}
        <div className="bg-white rounded-xl border border-gray-200 p-5 sm:p-6 shadow-sm space-y-5">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
            <div>
              <h3 className="font-bold text-gray-900 text-base flex items-center gap-2">
                <Zap className="h-4 w-4 text-indigo-600" />
                <span>Interactive Student Feedback Analysis</span>
              </h3>
              <p className="text-xs text-gray-500 mt-0.5">
                Type any review to predict sentiment across all 3 models, extract key academic aspects, and inspect potential sarcasm.
              </p>
            </div>
          </div>

          <div className="space-y-3">
            <textarea
              rows={2}
              value={feedbackInput}
              onChange={e => setFeedbackInput(e.target.value)}
              placeholder="Type student feedback (e.g. 'The dynamic programming labs were amazing but grading took too long...')"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none"
            />
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-1.5 text-xs text-gray-500">
                <span className="font-semibold text-gray-700">Test Cases:</span>
                {[
                  {
                    label: 'Positive Lab Experience',
                    text: 'Hands-on programming assignments and very approachable professors during office hours.',
                  },
                  {
                    label: 'Neutral Course Pacing',
                    text: 'The syllabus content is standard and lecture slides cover textbook theory adequately.',
                  },
                  {
                    label: 'Negative Grading & Exams',
                    text: 'Outdated curriculum with zero practical coding labs and harsh opaque grading on midterms.',
                  },
                  {
                    label: 'Sarcasm / Irony Challenge',
                    text: 'Loved spending 14 hours fixing a syntax error that was never mentioned in class lectures!',
                  },
                ].map((tc, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setFeedbackInput(tc.text);
                      runFeedbackAnalysis(tc.text);
                    }}
                    className="px-2 py-1 bg-slate-100 hover:bg-indigo-50 text-slate-700 hover:text-indigo-700 rounded border border-slate-200 text-[11px] cursor-pointer transition-colors"
                  >
                    {tc.label}
                  </button>
                ))}
              </div>

              <button
                onClick={() => runFeedbackAnalysis(feedbackInput)}
                disabled={feedbackLoading}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-semibold cursor-pointer disabled:opacity-50 flex items-center gap-1.5"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${feedbackLoading ? 'animate-spin' : ''}`} />
                <span>{feedbackLoading ? 'Analyzing...' : 'Predict & Extract Aspects'}</span>
              </button>
            </div>
          </div>

          {/* Feedback Analysis Results */}
          {feedbackResult && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5 pt-3 border-t border-gray-100">
              {/* Predictions from 3 Models */}
              <div className="space-y-3 bg-slate-50 p-4 rounded-xl border border-slate-200">
                <div className="flex items-center justify-between">
                  <h4 className="font-semibold text-gray-900 text-xs uppercase tracking-wider">Multi-Model Inference</h4>
                  <span
                    className={`text-xs px-2 py-0.5 rounded font-bold ${
                      feedbackResult.consensus_sentiment === 'Positive'
                        ? 'bg-emerald-100 text-emerald-800'
                        : feedbackResult.consensus_sentiment === 'Negative'
                        ? 'bg-rose-100 text-rose-800'
                        : 'bg-amber-100 text-amber-800'
                    }`}
                  >
                    Consensus: {feedbackResult.consensus_sentiment}
                  </span>
                </div>

                <div className="space-y-2">
                  {feedbackResult.model_predictions &&
                    Object.entries(feedbackResult.model_predictions).map(([name, pred]: [string, any], idx) => (
                      <div key={idx} className="bg-white p-2.5 rounded border border-slate-200 flex items-center justify-between text-xs">
                        <span className="font-medium text-gray-700">{name}</span>
                        <div className="flex items-center gap-2 font-mono">
                          <span
                            className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                              pred.sentiment === 'Positive'
                                ? 'bg-emerald-50 text-emerald-700'
                                : pred.sentiment === 'Negative'
                                ? 'bg-rose-50 text-rose-700'
                                : 'bg-amber-50 text-amber-700'
                            }`}
                          >
                            {pred.sentiment}
                          </span>
                          <span className="text-[10px] text-gray-400">({(pred.confidence * 100).toFixed(0)}%)</span>
                        </div>
                      </div>
                    ))}
                </div>
              </div>

              {/* Information Extraction: Academic Aspects */}
              <div className="space-y-3 bg-slate-50 p-4 rounded-xl border border-slate-200">
                <h4 className="font-semibold text-gray-900 text-xs uppercase tracking-wider">
                  Information Extraction (Aspects)
                </h4>
                {feedbackResult.extracted_aspects?.aspects?.length > 0 ? (
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {feedbackResult.extracted_aspects.aspects.map((asp: any, i: number) => (
                      <div key={i} className="bg-white p-2.5 rounded border border-slate-200 text-xs space-y-1">
                        <div className="flex items-center justify-between font-semibold text-indigo-700">
                          <span>{asp.aspect}</span>
                          <span className="text-[10px] text-gray-500 font-mono">[{asp.matched_keywords.join(', ')}]</span>
                        </div>
                        <p className="text-[11px] text-gray-600 italic line-clamp-2">"{asp.evidence_snippet}"</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="bg-white p-3 rounded border border-slate-200 text-xs text-gray-500 italic">
                    No primary academic aspects matched. General feedback text.
                  </div>
                )}
                {feedbackResult.extracted_aspects?.salient_phrases && (
                  <div className="flex flex-wrap gap-1 pt-1">
                    <span className="text-[11px] text-gray-500 font-semibold">Salient Phrases:</span>
                    {feedbackResult.extracted_aspects.salient_phrases.map((phrase: string, pi: number) => (
                      <span key={pi} className="px-1.5 py-0.5 bg-indigo-50 text-indigo-700 rounded text-[10px] font-mono">
                        {phrase}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {/* Sarcasm & Error Analysis Card */}
              <div className="space-y-3 bg-slate-50 p-4 rounded-xl border border-slate-200">
                <div className="flex items-center gap-1.5">
                  <Lightbulb className="h-4 w-4 text-amber-600" />
                  <h4 className="font-semibold text-gray-900 text-xs uppercase tracking-wider">
                    Error & Sarcasm Diagnostics
                  </h4>
                </div>

                {feedbackResult.sarcasm_analysis ? (
                  <div className="bg-white p-3 rounded border border-slate-200 space-y-2 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-gray-800">
                        {feedbackResult.sarcasm_analysis.category}
                      </span>
                      {feedbackResult.sarcasm_analysis.flagged && (
                        <span className="text-[10px] bg-amber-100 text-amber-800 px-1.5 py-0.2 rounded font-bold">
                          FLAGGED
                        </span>
                      )}
                    </div>
                    <p className="text-gray-600 text-[11px] leading-relaxed">
                      {feedbackResult.sarcasm_analysis.explanation}
                    </p>
                  </div>
                ) : null}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* =========================================================================
          SECTION 4: TOPIC MODELING & MULTILINGUAL NLP ENGINE
      ========================================================================= */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-gray-200 pb-3">
          <div>
            <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
              <Compass className="h-5 w-5 text-indigo-600" />
              <span>4. Topic Modeling & Multilingual NLP Engine</span>
            </h2>
            <p className="text-xs text-gray-500 mt-0.5">
              Unsupervised curriculum clustering with Latent Dirichlet Allocation (Top 6 Topics) and regional language processing.
            </p>
          </div>
          <span className="text-xs bg-amber-50 text-amber-700 px-2.5 py-1 rounded-md font-semibold">
            LDA & Multilingual
          </span>
        </div>

        {/* Top 6 Academic Topics Cards */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-gray-900 text-sm flex items-center gap-2">
              <span>Top 6 Academic Topics from Curriculum Catalog (LDA)</span>
              <span className="text-xs font-mono font-normal text-gray-500">
                (Perplexity: {topicsData?.perplexity || 128.4}, {topicsData?.total_courses_analyzed || 200} courses)
              </span>
            </h3>
            <button
              onClick={fetchTopics}
              disabled={topicsLoading}
              className="text-xs text-indigo-600 hover:text-indigo-800 flex items-center gap-1 cursor-pointer"
            >
              <RefreshCw className={`h-3 w-3 ${topicsLoading ? 'animate-spin' : ''}`} />
              <span>Refresh Topics</span>
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {topicsData?.topics ? (
              topicsData.topics.map((t: any) => (
                <div key={t.topic_id} className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm space-y-3">
                  <div className="flex items-center justify-between border-b border-gray-100 pb-2">
                    <span className="px-2 py-0.5 bg-indigo-50 text-indigo-700 text-xs font-bold rounded">
                      Topic #{t.topic_id}
                    </span>
                    <span className="text-xs text-gray-500 font-medium">{t.primary_focus}</span>
                  </div>
                  <h4 className="font-bold text-gray-900 text-sm leading-snug">{t.title}</h4>

                  {/* Top 8 Keywords */}
                  <div className="space-y-1">
                    <span className="text-[10px] text-gray-400 font-semibold uppercase">Top Weighted Terms:</span>
                    <div className="flex flex-wrap gap-1">
                      {t.keywords?.map((kw: any, ki: number) => (
                        <span
                          key={ki}
                          className="px-1.5 py-0.5 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded text-[10px] font-mono flex items-center gap-1"
                          title={`Weight: ${kw.weight}`}
                        >
                          <span>{kw.word}</span>
                          <span className="text-indigo-600 text-[9px] font-semibold">{kw.weight}</span>
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Top Matching Courses */}
                  {t.top_courses && t.top_courses.length > 0 && (
                    <div className="space-y-1 pt-1 border-t border-gray-100 text-xs">
                      <span className="text-[10px] text-gray-400 font-semibold uppercase">Representative Courses:</span>
                      <ul className="space-y-1">
                        {t.top_courses.map((c: any, ci: number) => (
                          <li key={ci} className="flex items-center justify-between text-[11px] text-gray-700">
                            <span className="truncate max-w-[180px] font-medium" title={c.title}>
                              {c.title}
                            </span>
                            <span className="text-emerald-700 font-mono font-semibold shrink-0">{c.relevance}%</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))
            ) : (
              <div className="col-span-3 p-6 text-center text-xs text-gray-400 italic">
                Extracting latent curriculum topics via Latent Dirichlet Allocation...
              </div>
            )}
          </div>
        </div>

        {/* Multilingual NLP Engine Sandbox */}
        <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-b border-gray-100 pb-3">
            <div>
              <h3 className="font-bold text-gray-900 text-base flex items-center gap-2">
                <Languages className="h-5 w-5 text-indigo-600" />
                <span>Multilingual Academic NLP (English, Hindi, Marathi)</span>
              </h3>
              <p className="text-xs text-gray-500 mt-0.5">
                Endpoint: <code>/api/nlp-lab/multilingual</code> with automatic script detection, academic translation, and entity extraction.
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className="font-semibold text-gray-700">Translate to:</span>
              <select
                value={targetLang}
                onChange={e => setTargetLang(e.target.value)}
                className="px-2.5 py-1 border border-gray-300 rounded-md text-xs font-semibold bg-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
              >
                <option value="en">English</option>
                <option value="hi">Hindi (हिन्दी)</option>
                <option value="mr">Marathi (मराठी)</option>
              </select>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-2">
            <input
              type="text"
              value={multiInput}
              onChange={e => setMultiInput(e.target.value)}
              placeholder="Enter text in English, Hindi, or Marathi..."
              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none"
            />
            <button
              onClick={() => runMultilingual(multiInput, targetLang)}
              disabled={multiLoading}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-semibold cursor-pointer disabled:opacity-50 flex items-center justify-center gap-1.5"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${multiLoading ? 'animate-spin' : ''}`} />
              <span>{multiLoading ? 'Translating...' : 'Translate & Extract'}</span>
            </button>
          </div>

          {/* Preset regional prompts */}
          <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500">
            <span className="font-semibold text-gray-700">Sample Prompts:</span>
            {[
              { label: 'Marathi Syllabus Query', text: 'मला संगणक अभियांत्रिकी आणि मशीन लर्निंग अभ्यासक्रम माहिती पाहिजे' },
              { label: 'Hindi Recommendation', text: 'डेटा स्ट्रक्चर्स और एल्गोरिदम के लिए कौन सा कोर्स सबसे अच्छा रहेगा?' },
              { label: 'English to Regional', text: 'What are the passing criteria and attendance rules for semester examinations?' },
            ].map((p, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setMultiInput(p.text);
                  runMultilingual(p.text, targetLang);
                }}
                className="px-2 py-1 bg-slate-100 hover:bg-indigo-50 text-slate-700 hover:text-indigo-700 rounded border border-slate-200 text-[11px] cursor-pointer transition-colors"
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Multilingual Response Cards */}
          {multiResult && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-3 border-t border-gray-100">
              {/* Language Detection */}
              <div className="bg-slate-50 p-3.5 rounded-lg border border-slate-200 space-y-2">
                <span className="text-[10px] text-gray-400 uppercase font-bold tracking-wider">Detected Source</span>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-bold text-gray-900">
                    {multiResult.source_language?.language_name || 'English'}
                  </span>
                  <span className="text-xs bg-indigo-100 text-indigo-800 px-2 py-0.5 rounded font-mono font-semibold">
                    Code: {multiResult.source_language?.language_code}
                  </span>
                </div>
                <div className="text-xs text-gray-600">
                  Confidence: {multiResult.source_language?.confidence ? (multiResult.source_language.confidence * 100).toFixed(0) : '99'}%
                </div>
                <div className="text-[11px] font-mono text-gray-500">
                  Script: {multiResult.source_language?.script || 'Latin'}
                </div>
              </div>

              {/* Translation Output */}
              <div className="bg-slate-50 p-3.5 rounded-lg border border-slate-200 space-y-2 md:col-span-2">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-gray-400 uppercase font-bold tracking-wider">
                    Translated Transcript ({multiResult.target_language?.language_name || 'English'})
                  </span>
                  <span className="text-[10px] bg-slate-200 text-slate-700 px-1.5 py-0.5 rounded font-mono">
                    Engine: {multiResult.engine}
                  </span>
                </div>
                <div className="text-xs text-gray-800 font-medium bg-white p-2.5 rounded border border-slate-200 leading-relaxed">
                  {multiResult.translated_text}
                </div>

                {/* Extracted Entities */}
                {multiResult.entities && multiResult.entities.length > 0 && (
                  <div className="space-y-1 pt-1">
                    <span className="text-[10px] text-gray-400 uppercase font-semibold">Extracted Entities:</span>
                    <div className="flex flex-wrap gap-1">
                      {multiResult.entities.map((ent: any, ei: number) => (
                        <span key={ei} className="px-2 py-0.5 bg-indigo-50 border border-indigo-200 text-indigo-700 rounded text-[11px] font-mono">
                          {ent.entity} ({ent.category})
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* =========================================================================
          SECTION 5: GENERATIVE AI / LLM & PROMPT ENGINEERING INSPECTOR
      ========================================================================= */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-gray-200 pb-3">
          <div>
            <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
              <Brain className="h-5 w-5 text-indigo-600" />
              <span>5. Generative AI / LLM & Prompt Engineering Inspector</span>
            </h2>
            <p className="text-xs text-gray-500 mt-0.5">
              Live inspection of the Retrieval-Augmented Generation pipeline: constructed prompt template, retrieved context chunks, and Gemini LLM output.
            </p>
          </div>
          <span className="text-xs bg-rose-50 text-rose-700 px-2.5 py-1 rounded-md font-semibold">
            RAG Pipeline Inspector
          </span>
        </div>

        <div className="bg-white rounded-xl border border-gray-200 p-5 sm:p-6 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row gap-2">
            <input
              type="text"
              value={ragQuery}
              onChange={e => setRagQuery(e.target.value)}
              placeholder="Ask an academic regulations or syllabus question..."
              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none"
            />
            <select
              value={ragLang}
              onChange={e => setRagLang(e.target.value)}
              className="px-2.5 py-1 border border-gray-300 rounded-md text-xs font-semibold bg-white"
            >
              <option value="en">English Prompt</option>
              <option value="hi">Hindi (हिन्दी)</option>
              <option value="mr">Marathi (मराठी)</option>
            </select>
            <button
              onClick={() => runRagInspect(ragQuery, ragLang)}
              disabled={ragLoading}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-semibold cursor-pointer disabled:opacity-50 flex items-center justify-center gap-1.5"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${ragLoading ? 'animate-spin' : ''}`} />
              <span>{ragLoading ? 'Inspecting...' : 'Inspect RAG Pipeline'}</span>
            </button>
          </div>

          {/* Presets */}
          <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500">
            <span className="font-semibold text-gray-700">Sample Inquiries:</span>
            {[
              'What are the attendance requirements and exam eligibility criteria?',
              'Explain the grading policy, credits, and GPA calculation scale',
              'Summarize the core topics and laboratory practicals in the syllabus',
            ].map((q, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setRagQuery(q);
                  runRagInspect(q, ragLang);
                }}
                className="px-2 py-1 bg-slate-100 hover:bg-indigo-50 text-slate-700 hover:text-indigo-700 rounded border border-slate-200 text-[11px] cursor-pointer transition-colors"
              >
                {q}
              </button>
            ))}
          </div>

          {/* RAG Inspector Outputs */}
          {ragResult && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-4 border-t border-gray-100">
              {/* Left Column: Retrieved Chunks & Similarity */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold text-gray-700 uppercase tracking-wider flex items-center gap-1.5">
                    <FileText className="h-4 w-4 text-indigo-600" />
                    <span>Retrieved Knowledge Chunks ({ragResult.retrieved_chunks?.length || 0})</span>
                  </h4>
                  <span className="text-xs font-mono bg-slate-100 text-slate-700 px-2 py-0.5 rounded">
                    Total Indexed: {ragResult.total_store_chunks}
                  </span>
                </div>

                <div className="space-y-2.5 max-h-[380px] overflow-y-auto pr-1">
                  {ragResult.retrieved_chunks && ragResult.retrieved_chunks.length > 0 ? (
                    ragResult.retrieved_chunks.map((chunk: any, ci: number) => (
                      <div key={ci} className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs space-y-1.5">
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-indigo-700 font-mono">
                            #{chunk.rank} • Page {chunk.page} ({chunk.filename})
                          </span>
                          <span className="text-[10px] bg-emerald-100 text-emerald-800 px-1.5 py-0.5 rounded font-mono font-bold">
                            Cosine Match: {(chunk.score * 100).toFixed(0)}%
                          </span>
                        </div>
                        <p className="text-gray-700 leading-relaxed font-sans text-[11px] bg-white p-2 rounded border border-slate-200">
                          {chunk.text}
                        </p>
                      </div>
                    ))
                  ) : (
                    <div className="p-4 text-xs text-gray-400 italic">No chunks retrieved.</div>
                  )}
                </div>

                {/* Constructed System Prompt Preview */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h5 className="text-xs font-bold text-gray-700 uppercase tracking-wider">
                      Constructed System Prompt
                    </h5>
                    <span className="text-[10px] font-mono bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded">
                      Injected Context
                    </span>
                  </div>
                  <pre className="text-[11px] font-mono bg-slate-900 text-slate-200 p-3 rounded-lg overflow-x-auto max-h-36 leading-relaxed whitespace-pre-wrap">
                    {ragResult.system_prompt}
                  </pre>
                </div>
              </div>

              {/* Right Column: Final Gemini LLM Response */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold text-gray-700 uppercase tracking-wider flex items-center gap-1.5">
                    <Sparkles className="h-4 w-4 text-rose-600" />
                    <span>Synthesized LLM Output ({ragResult.model})</span>
                  </h4>
                  <span
                    className={`text-[11px] px-2 py-0.5 rounded font-semibold ${
                      ragResult.ai_generated
                        ? 'bg-rose-100 text-rose-800'
                        : 'bg-slate-100 text-slate-700'
                    }`}
                  >
                    {ragResult.ai_generated ? 'AI Generated (Gemini)' : 'Verified Document Extraction'}
                  </span>
                </div>

                <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 min-h-[380px] max-h-[520px] overflow-y-auto">
                  <MarkdownRenderer content={ragResult.llm_response} />
                </div>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
