# InsureMatch — AI Insurance Recommendation System

A production-ready **Retrieval-Augmented Generation (RAG)** system that recommends the best insurance policy for a user based on their personal, financial, health, and lifestyle profile. Built with **FastAPI**, **LangChain**, **Groq LLM**, **ChromaDB**, and **XGBoost + SHAP** for explainability — paired with a **Next.js** frontend for a complete end-to-end experience.

**Client:** Union Assurance — a major insurance company in Sri Lanka.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Full RAG Flow](#full-rag-flow)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Frontend](#frontend)
- [API Endpoints](#api-endpoints)
  - [Ingest Documents](#1-ingest-documents)
  - [Get Recommendation](#2-get-recommendation)
  - [Explain a Policy](#3-explain-a-policy)
  - [Chat](#4-chat)
  - [List Policies](#5-list-policies)
- [User Profile Schema](#user-profile-schema)
- [Sample API Calls](#sample-api-calls)
- [SHAP Explainability](#shap-explainability)
- [Chunking Strategy](#chunking-strategy)
- [Environment Variables](#environment-variables)

---

## Overview

This system answers the question:

> *"Given a user's age, income, health conditions, employment, and insurance goals — which insurance policy is the best fit, and why?"*

The system:
1. Ingests insurance policy documents (PDF, DOCX, TXT) into a **ChromaDB** vector database.
2. Accepts a detailed **user profile** via the frontend form (or API directly).
3. Scores all available policies using a **XGBoost model** trained on domain rules.
4. Explains *why* each policy scored the way it did using **SHAP** (feature importance).
5. Retrieves relevant policy clauses via **semantic search** (RAG).
6. Synthesizes everything into a **natural language recommendation** using **Groq LLM** (LLaMA 3.3 70B).
7. Provides a **conversational chat interface** for follow-up insurance questions.

### End-to-End User Flow

```
Landing Page → 6-Step Profile Form → AI Recommendation → SHAP Explanations → Follow-up Chat
     /              /profile           POST /api/recommend      /results            /chat
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INSUREMATCH SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      FRONTEND (Next.js 16.2)                         │  │
│  │                                                                       │  │
│  │  Landing Page → 6-Step Form Wizard → Results + SHAP → Chat           │  │
│  │                                                                       │  │
│  │  Tech: TypeScript, Tailwind v4, shadcn/ui, Zustand, React Query      │  │
│  │  Pattern: Atomic Design (atoms → molecules → organisms → templates)  │  │
│  └──────────────────────────────┬────────────────────────────────────────┘  │
│                                 │ Axios (HTTP)                              │
│                                 ▼                                           │
│  ┌──────────────┐    ┌───────────────────────────────────────┐             │
│  │  INGESTION   │    │           RECOMMENDATION ENGINE       │             │
│  │  PIPELINE    │    │                                       │             │
│  │              │    │  User Profile                         │             │
│  │  PDF/DOCX    │    │       │                               │             │
│  │  TXT Docs    │    │       ▼                               │             │
│  │      │       │    │  Feature Extraction (21 features)     │             │
│  │      ▼       │    │       │                               │             │
│  │  Document    │    │       ▼                               │             │
│  │  Loader      │    │  XGBoost Scorer ──► Score [0-1]       │             │
│  │      │       │    │  (per policy)                         │             │
│  │      ▼       │    │       │                               │             │
│  │  Recursive   │    │       ▼                               │             │
│  │  Semantic    │    │  SHAP TreeExplainer                   │             │
│  │  Chunker     │    │  (feature importance)                 │             │
│  │      │       │    │       │                               │             │
│  │      ▼       │    │       ▼                               │             │
│  │  HuggingFace │    │  Top-K Ranked Policies + Explanations │             │
│  │  Embeddings  │    │       │                               │             │
│  │      │       │    │       ▼                               │             │
│  │      ▼       │    │  ChromaDB Semantic Retrieval (MMR)    │             │
│  │  ChromaDB ◄──┘    │  (relevant policy clauses)            │             │
│  │  (VectorDB)  │────►       │                               │             │
│  │              │    │       ▼                               │             │
│  │  Policy      │    │  Groq LLM (LLaMA 3.3 70B)            │             │
│  │  Registry    │    │  (narrative generation)               │             │
│  │  (JSON)      │    │       │                               │             │
│  └──────────────┘    │       ▼                               │             │
│                      │  RecommendationResponse               │             │
│                      │  + session_id (for chat)              │             │
│                      └───────────────────────────────────────┘             │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │                     CHAT ENGINE                          │              │
│  │  session_id → Conversation History + ChromaDB Retrieval  │              │
│  │              → Groq LLM Response                         │              │
│  └──────────────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Full RAG Flow

### Phase 1 — Document Ingestion

```
Insurance Policy Document (PDF / DOCX / TXT)
           │
           ▼
    Document Loader
    (PyPDF / Docx2txt / TextLoader)
           │
           ▼
    Recursive Semantic Chunker
    ┌─────────────────────────────────┐
    │  Splits on:                     │
    │  1. Multi-blank lines (major)   │
    │  2. Paragraph breaks            │
    │  3. Single newlines             │
    │  4. Sentence boundaries         │
    │  chunk_size=1000, overlap=200   │
    └─────────────────────────────────┘
           │
           ▼
    Section Detection
    (benefits / exclusions / premium /
     eligibility / claims / riders)
           │
           ▼
    Metadata Tagging per chunk:
    { policy_name, policy_type, company,
      section, page_num, chunk_index,
      min_age, max_age, premium_level,
      covers_health, covers_life, ... }
           │
           ├──► HuggingFace Embeddings
           │    (all-MiniLM-L6-v2)
           │           │
           │           ▼
           │    ChromaDB (persistent)
           │
           └──► Policy Registry (JSON)
                (structured features for scorer)
```

### Phase 2 — Recommendation

```
POST /api/recommend
{ user_profile: { personal, occupation, goals, health, lifestyle } }
           │
           ▼
  User Feature Extraction (21 features)
  ┌─────────────────────────────────────┐
  │  age_norm, gender, is_married,      │
  │  num_dependents, employment_type,   │
  │  hazardous_level, income_norm,      │
  │  goal_* (5 binary flags),           │
  │  has_chronic, has_cardiovascular,   │
  │  has_cancer, has_respiratory,       │
  │  has_neurological, bmi_norm,        │
  │  is_smoker, is_alcohol              │
  └─────────────────────────────────────┘
           │
           ▼
  For each policy in registry:
  ┌─────────────────────────────────────┐
  │  Policy Feature Extraction          │
  │  (10 features: type flags, age      │
  │   limits, premium level, coverage)  │
  └─────────────────────────────────────┘
           │
           ▼
  XGBoost Scorer
  (31 combined features → suitability score [0,1])
           │
           ▼
  Ranked Policy List (all policies scored)
           │
           ▼
  Top-K Policies Selected
           │
           ├──► SHAP TreeExplainer
           │    (per-feature impact values)
           │           │
           │           ▼
           │    Human-readable explanations:
           │    + Your age: +0.25 impact
           │    + Goal: Affordable coverage: +0.20 impact
           │    - Contract employment: -0.05 impact
           │
           ├──► ChromaDB MMR Retrieval
           │    (retrieve diverse relevant chunks
           │     for top policy)
           │
           ▼
  Groq LLM (LLaMA 3.3 70B)
  Prompt = User Profile + SHAP Summary + RAG Context
           │
           ▼
  RecommendationResponse {
    ranked_policies,      ← all top-K with scores
    top_recommendation,   ← best policy name
    explanations,         ← SHAP factors per policy
    rag_narrative,        ← LLM-written recommendation
    session_id            ← use for /api/chat
  }
```

### Phase 3 — Chat

```
POST /api/chat
{ session_id, message, user_profile? }
           │
           ▼
  Load session history (last 6 turns)
           │
           ▼
  ChromaDB Semantic Retrieval
  (query = message + user context)
           │
           ▼
  Groq LLM (LLaMA 3.3 70B)
  Prompt = RAG Context + Chat History + Question
           │
           ▼
  ChatResponse {
    session_id,
    response,
    sources    ← which policies were referenced
  }
```

---

## Tech Stack

### Backend

| Component | Technology |
|---|---|
| Web Framework | FastAPI + Uvicorn |
| LLM | Groq — LLaMA 3.3 70B Versatile |
| RAG Orchestration | LangChain 1.2.x |
| Vector Database | ChromaDB (local, persistent) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| ML Scoring | XGBoost (trained on synthetic domain data) |
| Explainability | SHAP `TreeExplainer` |
| Document Loaders | PyPDF, Docx2txt, TextLoader |
| Data Validation | Pydantic v2 |

### Frontend

| Component | Technology |
|---|---|
| Framework | Next.js 16.2 (App Router, TypeScript) |
| UI Components | shadcn/ui (base-ui primitives) |
| Styling | Tailwind CSS v4 |
| State Management | Zustand (sessionStorage persist) |
| Server State | TanStack React Query |
| Form Handling | React Hook Form + Zod validation |
| HTTP Client | Axios |
| Component Pattern | Atomic Design |

---

## Project Structure

```
insurance_recommendation_rag/
│
├── InsureMatch-backend/                    # ── FastAPI Backend ──
│   ├── app/
│   │   ├── main.py                         # FastAPI app, startup, CORS, routers
│   │   ├── config.py                       # Settings via pydantic-settings + .env
│   │   │
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── ingest.py               # POST /api/ingest, DELETE /api/ingest/{name}
│   │   │       ├── recommend.py            # POST /api/recommend
│   │   │       ├── explain.py              # POST /api/explain
│   │   │       ├── chat.py                 # POST /api/chat, /chat/new, DELETE /chat/{id}
│   │   │       └── policies.py             # GET  /api/policies, /registry, /{name}
│   │   │
│   │   ├── core/
│   │   │   ├── ingestion/
│   │   │   │   ├── loader.py               # PDF / DOCX / TXT document loaders
│   │   │   │   └── chunker.py              # Recursive semantic chunker + section detector
│   │   │   │
│   │   │   ├── vectorstore/
│   │   │   │   └── chroma_store.py         # ChromaDB wrapper + policy registry I/O
│   │   │   │
│   │   │   ├── rag/
│   │   │   │   ├── prompts.py              # All LangChain prompt templates
│   │   │   │   └── chain.py               # RAG chain, chat chain, metadata extraction
│   │   │   │
│   │   │   ├── recommendation/
│   │   │   │   ├── scorer.py               # XGBoost model, feature extraction, training
│   │   │   │   ├── explainer.py            # SHAP TreeExplainer + human-readable output
│   │   │   │   └── ranker.py               # Full pipeline orchestrator
│   │   │   │
│   │   │   └── llm/
│   │   │       └── groq_llm.py             # ChatGroq setup (standard + creative)
│   │   │
│   │   ├── models/
│   │   │   └── schemas.py                  # All Pydantic models (source of truth for types)
│   │   │
│   │   └── utils/
│   │       └── helpers.py                  # Profile-to-text, BMI category helpers
│   │
│   ├── data/raw/                           # Drop your insurance documents here
│   ├── vectordb/                           # ChromaDB persistent storage (auto-created)
│   ├── saved_models/
│   │   ├── xgb_scorer.pkl                  # Trained XGBoost model (auto-created)
│   │   └── policy_registry.json            # Structured policy metadata (auto-created)
│   │
│   ├── .env                                # Your environment variables
│   ├── .env.example                        # Environment variable template
│   ├── requirements.txt                    # Python dependencies
│   └── README.md                           # This file
│
├── InsureMatch-frontend/                   # ── Next.js Frontend ──
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx                  # Root layout + providers
│   │   │   ├── page.tsx                    # Landing page
│   │   │   ├── not-found.tsx               # 404 page
│   │   │   ├── profile/page.tsx            # 6-step form wizard
│   │   │   ├── results/page.tsx            # Recommendation results
│   │   │   ├── chat/page.tsx               # Follow-up chat
│   │   │   └── policies/page.tsx           # Browse indexed policies
│   │   │
│   │   ├── components/
│   │   │   ├── ui/                         # shadcn/ui generated components
│   │   │   ├── atoms/                      # Spinner, ProgressBar, ScoreBadge, ImpactIndicator
│   │   │   ├── molecules/                  # FormField, SelectField, CheckboxField, StepIndicator,
│   │   │   │                               # PolicyCard, SHAPFactorRow, ChatBubble, SourceChip
│   │   │   ├── organisms/                  # Navbar, Footer, StepperHeader, 5 form steps,
│   │   │   │                               # ReviewSummary, PolicyResultCard, NarrativePanel,
│   │   │   │                               # ChatWindow, PolicyListTable
│   │   │   └── templates/                  # FormWizardTemplate, ResultsTemplate,
│   │   │                                   # ChatTemplate, PoliciesTemplate
│   │   │
│   │   ├── lib/
│   │   │   ├── api/                        # Axios client + endpoint constants
│   │   │   ├── hooks/                      # TanStack Query hooks (useRecommendation, useChat, etc.)
│   │   │   ├── store/                      # Zustand store (useProfileStore)
│   │   │   ├── types/                      # TypeScript types mirroring backend schemas
│   │   │   ├── validators/                 # Zod schemas per form step
│   │   │   └── utils/                      # Formatters + constants
│   │   │
│   │   └── providers/
│   │       └── QueryProvider.tsx            # TanStack QueryClientProvider
│   │
│   ├── .env.local                          # NEXT_PUBLIC_API_URL=http://localhost:8000
│   ├── components.json                     # shadcn/ui config
│   ├── package.json                        # Node dependencies
│   └── tsconfig.json                       # TypeScript config
│
└── CLAUDE.md                               # Root-level Claude Code guidance
```

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- Node.js 18+ and npm
- A [Groq API key](https://console.groq.com) (free tier available)

### Backend Setup

#### Step 1 — Navigate to the backend

```bash
cd InsureMatch-backend
```

#### Step 2 — Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

#### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the HuggingFace embedding model (~90MB). This is cached locally after the first download.

#### Step 4 — Configure environment variables

Copy `.env.example` to `.env` and set your Groq API key:

```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

CHROMA_PERSIST_DIR=./vectordb
CHROMA_COLLECTION_NAME=insurance_policies

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

CHUNK_SIZE=1000
CHUNK_OVERLAP=200

RETRIEVAL_K=8
RETRIEVAL_FETCH_K=20

MODEL_SAVE_DIR=./saved_models
DATA_DIR=./data/raw
```

### Frontend Setup

#### Step 1 — Navigate to the frontend

```bash
cd InsureMatch-frontend
```

#### Step 2 — Install dependencies

```bash
npm install
```

#### Step 3 — Configure environment

The `.env.local` file should contain:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Running the Project

### Start the backend

```bash
cd InsureMatch-backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On startup the server will:
1. Create required directories (`vectordb/`, `saved_models/`, `data/raw/`)
2. Train the XGBoost scorer on synthetic data (first run only, ~10 seconds)
3. Save the model to `saved_models/xgb_scorer.pkl`

### Start the frontend

```bash
cd InsureMatch-frontend
npm run dev
```

The frontend runs at `http://localhost:3000` and connects to the backend at `http://localhost:8000`.

### Access the interactive API docs

```
http://localhost:8000/docs      # Swagger UI
http://localhost:8000/redoc     # ReDoc
```

### Health check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "policies_indexed": 0,
  "policies_in_registry": 0
}
```

---

## Frontend

The frontend is a **Next.js 16.2** application with an orange-themed professional UI (Union Assurance brand).

### Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Landing | Hero section with CTA, "How It Works" feature cards |
| `/profile` | Profile Form | 6-step wizard: Personal → Occupation → Goals → Health → Lifestyle → Review |
| `/results` | Results | Ranked policy cards with SHAP explanations + AI narrative panel |
| `/chat` | Chat | Follow-up Q&A with the AI about recommendations |
| `/policies` | Policies | Table of all indexed policies from the backend |

### Multi-Step Form Design

The profile form collects all fields required by `POST /api/recommend` across 6 validated steps:

| Step | Label | Key Fields |
|------|-------|------------|
| 1 | Personal Info | age, gender, marital_status, nationality, country, district, city, num_dependents |
| 2 | Occupation & Income | occupation, employment_type, hazardous_level, monthly_income_lkr, has_existing_insurance |
| 3 | Insurance Goals | primary_goal, secondary_goal, travel_history, dual_citizenship, regulatory_flags |
| 4 | Health Info | 10 boolean health condition checkboxes |
| 5 | Lifestyle | bmi, is_smoker, is_alcohol_consumer |
| 6 | Review & Submit | Read-only summary → Submit triggers `POST /api/recommend` |

### State Management

- **Zustand store** (`useProfileStore`) holds form data across steps, persisted to `sessionStorage`
- **TanStack React Query** manages API calls (`useRecommendation`, `useChat`, `useExplain`, `usePolicies`)
- **Zod schemas** validate each form step before advancing

### Theme

Orange professional theme matching Union Assurance brand:
- **Primary**: `#F97316` (orange-500) — buttons, stepper, accents
- **Secondary**: `#1E293B` (slate-800) — headings, navbar
- **Background**: White, **Cards**: `#F8FAFC` (slate-50)
- Stepper: orange active, gray pending, green completed

---

## API Endpoints

### 1. Ingest Documents

**`POST /api/ingest`**

Upload an insurance policy document. The system will automatically extract policy metadata using the LLM if not provided manually.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | PDF, DOCX, or TXT document |
| `policy_name` | string | Optional | Override extracted policy name |
| `policy_type` | string | Optional | `term_life` \| `whole_life` \| `endowment` \| `health` \| `critical_illness` \| `accident` |
| `company` | string | Optional | Insurance company name |
| `min_age` | integer | Optional | Minimum entry age |
| `max_age` | integer | Optional | Maximum entry age |
| `premium_level` | integer | Optional | `0`=low, `1`=medium, `2`=high |
| `covers_health` | boolean | Optional | Includes health coverage |
| `covers_life` | boolean | Optional | Includes life coverage |
| `covers_accident` | boolean | Optional | Includes accident coverage |
| `is_entry_level` | boolean | Optional | Is a basic/affordable product |

**Response:**
```json
{
  "message": "Successfully indexed 'AIA Term Life Basic'",
  "policy_name": "AIA Term Life Basic",
  "chunks_indexed": 47,
  "policy_metadata": {
    "policy_name": "AIA Term Life Basic",
    "policy_type": "term_life",
    "company": "AIA Sri Lanka",
    "min_age": 18,
    "max_age": 65,
    "premium_level": 0,
    "covers_health": false,
    "covers_life": true,
    "covers_accident": false,
    "is_entry_level": true
  }
}
```

**Delete a policy:**

```
DELETE /api/ingest/{policy_name}
```

---

### 2. Get Recommendation

**`POST /api/recommend`**

Submit a complete user profile and receive ranked insurance recommendations with SHAP explanations and a natural language narrative.

**Request Body:**
```json
{
  "user_profile": {
    "personal": {
      "age": 20,
      "gender": "male",
      "marital_status": "single",
      "nationality": "Sri Lankan",
      "country": "Sri Lanka",
      "district": "Gampaha",
      "city": "Negombo",
      "num_dependents": 0
    },
    "occupation": {
      "occupation": "Graphic Designer",
      "employment_type": "contract",
      "designation": "Junior Designer",
      "hazardous_level": "low",
      "hazardous_activities": null,
      "monthly_income_lkr": 140000,
      "has_existing_insurance": false,
      "current_insurance_status": "none",
      "employer_insurance_scheme": null
    },
    "goals": {
      "primary_goal": "cheap_and_quick",
      "secondary_goal": "none",
      "travel_history_high_risk": false,
      "dual_citizenship": false,
      "tax_regulatory_flags": false,
      "insurance_history_issues": false
    },
    "health": {
      "has_chronic_disease": false,
      "has_cardiovascular": false,
      "has_cancer": false,
      "has_respiratory": false,
      "has_neurological": false,
      "has_gastrointestinal": false,
      "has_musculoskeletal": false,
      "has_infectious_sexual": false,
      "recent_treatment_surgery": false,
      "covid_related": false
    },
    "lifestyle": {
      "bmi": 22.5,
      "is_smoker": false,
      "is_alcohol_consumer": false
    }
  },
  "top_k": 3
}
```

**Response:**
```json
{
  "ranked_policies": [
    {
      "policy_name": "AIA Term Life Basic",
      "policy_type": "term_life",
      "company": "AIA Sri Lanka",
      "suitability_score": 0.8734,
      "rank": 1
    },
    {
      "policy_name": "Personal Accident Shield",
      "policy_type": "accident",
      "company": "Ceylinco Life",
      "suitability_score": 0.7421,
      "rank": 2
    },
    {
      "policy_name": "Comprehensive Health Plan",
      "policy_type": "health",
      "company": "Union Assurance",
      "suitability_score": 0.6103,
      "rank": 3
    }
  ],
  "top_recommendation": "AIA Term Life Basic",
  "explanations": [
    {
      "policy_name": "AIA Term Life Basic",
      "suitability_score": 0.8734,
      "positive_factors": [
        {
          "feature": "Goal: Affordable / quick coverage",
          "impact_score": 0.2514,
          "direction": "positive",
          "reason": "This policy aligns perfectly with your goal for affordable, accessible coverage."
        },
        {
          "feature": "Your age",
          "impact_score": 0.1983,
          "direction": "positive",
          "reason": "Your age falls within the ideal range for this policy."
        },
        {
          "feature": "Non-smoker status",
          "impact_score": 0.0872,
          "direction": "positive",
          "reason": "Non-smoker status qualifies you for standard (lower) premium rates."
        }
      ],
      "negative_factors": [
        {
          "feature": "Permanent employment status",
          "impact_score": 0.0521,
          "direction": "negative",
          "reason": "Contract/freelance status may require more flexible premium terms."
        }
      ],
      "shap_summary": "Key factors driving this recommendation:\n\nPositive influences:\n  + Goal: Affordable / quick coverage: +0.251 impact\n  + Your age: +0.198 impact\n  + Non-smoker status: +0.087 impact\n\nNegative influences:\n  - Permanent employment status: -0.052 impact"
    }
  ],
  "rag_narrative": "Based on your profile as a 20-year-old graphic designer in Negombo...",
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```

**Available goal values:**

| Value | Meaning |
|---|---|
| `cheap_and_quick` | Wants affordable, accessible coverage |
| `protection` | Life protection for family |
| `savings_and_investment` | Wants savings component |
| `health_coverage` | Priority is medical coverage |
| `retirement` | Planning for retirement |
| `none` | No specific goal |

**Available employment types:** `permanent`, `contract`, `freelance`, `self_employed`, `unemployed`, `retired`

**Available hazardous levels:** `none`, `low`, `medium`, `high`

---

### 3. Explain a Policy

**`POST /api/explain`**

Get a deep SHAP explanation for why a specific policy is (or is not) suitable for a given user. Useful for comparing any two policies or understanding edge cases.

**Request Body:**
```json
{
  "user_profile": { ... },
  "policy_name": "Whole Life Protection"
}
```

**Response:** Same structure as the `explanations` array item from `/api/recommend`, but with up to 7 factors per direction for deeper analysis.

---

### 4. Chat

**`POST /api/chat`**

Ask questions about insurance policies in a conversational style. The chatbot retrieves relevant policy content from ChromaDB and responds using the Groq LLM.

**Request Body:**
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "message": "What are the exclusions for the AIA Term Life policy?",
  "user_profile": null
}
```

> Use the `session_id` from `/api/recommend` to continue the same conversation context. Or use `POST /api/chat/new` to start a fresh session.

**Response:**
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "response": "Based on the AIA Term Life policy documents, the main exclusions include...",
  "sources": ["AIA Term Life Basic", "AIA Term Life Premium"]
}
```

**Other chat endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat/new` | Generate a new session ID |
| `DELETE` | `/api/chat/{session_id}` | Clear conversation history |

**Example questions to ask the chatbot:**
- *"What is the waiting period for critical illness coverage?"*
- *"How do I submit a claim for this policy?"*
- *"Can I add a rider to my term life plan?"*
- *"What happens if I miss a premium payment?"*
- *"Is COVID-19 covered under the health plan?"*

---

### 5. List Policies

**`GET /api/policies`**

List all insurance policies currently indexed in ChromaDB.

**Response:**
```json
[
  {
    "policy_name": "AIA Term Life Basic",
    "policy_type": "term_life",
    "company": "AIA Sri Lanka",
    "source_file": "aia_term_life.pdf",
    "chunk_count": 47
  }
]
```

**Other policy endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/policies/registry` | Full registry with all metadata (used by scorer) |
| `GET` | `/api/policies/{policy_name}` | Metadata for one specific policy |

---

## User Profile Schema

All fields accepted by `POST /api/recommend`:

### Personal Information

| Field | Type | Constraints | Example |
|---|---|---|---|
| `age` | integer | 18–70 | `20` |
| `gender` | enum | `male`, `female`, `other` | `"male"` |
| `marital_status` | enum | `single`, `married`, `divorced`, `widowed` | `"single"` |
| `nationality` | string | — | `"Sri Lankan"` |
| `country` | string | — | `"Sri Lanka"` |
| `district` | string | optional | `"Gampaha"` |
| `city` | string | optional | `"Negombo"` |
| `num_dependents` | integer | 0–20 | `0` |

### Occupation & Financial Details

| Field | Type | Constraints | Example |
|---|---|---|---|
| `occupation` | string | — | `"Graphic Designer"` |
| `employment_type` | enum | `permanent`, `contract`, `freelance`, `self_employed`, `unemployed`, `retired` | `"contract"` |
| `designation` | string | optional | `"Junior Designer"` |
| `hazardous_level` | enum | `none`, `low`, `medium`, `high` | `"low"` |
| `hazardous_activities` | string | optional | `null` |
| `monthly_income_lkr` | float | > 0 | `140000` |
| `has_existing_insurance` | boolean | — | `false` |
| `current_insurance_status` | enum | `none`, `has_insurance` | `"none"` |
| `employer_insurance_scheme` | string | optional | `null` |

### Insurance Goals & Preferences

| Field | Type | Example |
|---|---|---|
| `primary_goal` | enum | `"cheap_and_quick"` |
| `secondary_goal` | enum / null | `"none"` |
| `travel_history_high_risk` | boolean | `false` |
| `dual_citizenship` | boolean | `false` |
| `tax_regulatory_flags` | boolean | `false` |
| `insurance_history_issues` | boolean | `false` |

### Health Information

| Field | Type | Default |
|---|---|---|
| `has_chronic_disease` | boolean | `false` |
| `has_cardiovascular` | boolean | `false` |
| `has_cancer` | boolean | `false` |
| `has_respiratory` | boolean | `false` |
| `has_neurological` | boolean | `false` |
| `has_gastrointestinal` | boolean | `false` |
| `has_musculoskeletal` | boolean | `false` |
| `has_infectious_sexual` | boolean | `false` |
| `recent_treatment_surgery` | boolean | `false` |
| `covid_related` | boolean | `false` |

### Lifestyle Information

| Field | Type | Constraints | Example |
|---|---|---|---|
| `bmi` | float | > 0, ≤ 60 | `22.5` |
| `is_smoker` | boolean | — | `false` |
| `is_alcohol_consumer` | boolean | — | `false` |

---

## Sample API Calls

### Upload a policy document (curl)

```bash
curl -X POST "http://localhost:8000/api/ingest" \
  -F "file=@./data/raw/aia_term_life.pdf" \
  -F "policy_type=term_life" \
  -F "company=AIA Sri Lanka" \
  -F "premium_level=0" \
  -F "covers_life=true" \
  -F "is_entry_level=true"
```

### Upload with auto-extracted metadata

```bash
curl -X POST "http://localhost:8000/api/ingest" \
  -F "file=@./data/raw/union_health_plan.pdf"
```

### Get a recommendation (curl)

```bash
curl -X POST "http://localhost:8000/api/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_profile": {
      "personal": {
        "age": 20, "gender": "male", "marital_status": "single",
        "num_dependents": 0, "district": "Gampaha", "city": "Negombo"
      },
      "occupation": {
        "occupation": "Graphic Designer", "employment_type": "contract",
        "designation": "Junior Designer", "hazardous_level": "low",
        "monthly_income_lkr": 140000, "has_existing_insurance": false,
        "current_insurance_status": "none"
      },
      "goals": {
        "primary_goal": "cheap_and_quick", "secondary_goal": "none",
        "travel_history_high_risk": false, "dual_citizenship": false,
        "tax_regulatory_flags": false, "insurance_history_issues": false
      },
      "health": {
        "has_chronic_disease": false, "has_cardiovascular": false,
        "has_cancer": false, "has_respiratory": false,
        "has_neurological": false, "has_gastrointestinal": false,
        "has_musculoskeletal": false, "has_infectious_sexual": false,
        "recent_treatment_surgery": false, "covid_related": false
      },
      "lifestyle": { "bmi": 22.5, "is_smoker": false, "is_alcohol_consumer": false }
    },
    "top_k": 3
  }'
```

### Chat follow-up (curl)

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "message": "What are the key exclusions I should be aware of?"
  }'
```

### Explain a specific policy (curl)

```bash
curl -X POST "http://localhost:8000/api/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "user_profile": { ... },
    "policy_name": "Whole Life Protection"
  }'
```

---

## SHAP Explainability

The system uses **SHAP (SHapley Additive exPlanations)** with `TreeExplainer` on the XGBoost scoring model to explain every recommendation.

### How it works

1. The XGBoost model is trained on **8,000 synthetic user-policy pairs** using domain-encoded insurance rules.
2. At inference time, SHAP decomposes the model's prediction into per-feature contributions.
3. Each feature's SHAP value represents how much it **increased or decreased** the suitability score.
4. The top positive and negative factors are surfaced with human-readable descriptions.

### Feature categories explained to the user

**User profile factors** (what about you affected the recommendation):
- Age, gender, marital status, dependents
- Employment type, hazardous level, income
- Primary and secondary insurance goals
- Health conditions (chronic, cardiovascular, cancer, respiratory, neurological)
- BMI, smoking status, alcohol consumption

**Policy characteristic factors** (what about the policy matched your profile):
- Policy type (term life, whole life, endowment, health, critical illness, accident)
- Age limits, premium level, health coverage inclusion

### Sample SHAP output

```
Why "AIA Term Life Basic" was recommended for you:

Positive factors:
  + Goal: Affordable / quick coverage       +0.251 impact
    → This policy aligns perfectly with your goal for affordable coverage.

  + Your age (20 years)                     +0.198 impact
    → Your age falls within the ideal range for this policy.

  + Non-smoker status                       +0.087 impact
    → Non-smoker status qualifies you for standard (lower) premium rates.

  + BMI in normal range (22.5)              +0.063 impact
    → Your BMI is within a healthy range, reducing premium loading.

Negative factors:
  - Contract employment status              -0.052 impact
    → Contract/freelance status may require more flexible premium terms.
```

---

## Chunking Strategy

The system uses **Recursive Semantic Chunking** — a hierarchical splitting strategy designed for structured legal/insurance documents:

```
Priority of split boundaries:
  1. Multi-blank lines    (\n\n\n)   → Major section breaks
  2. Double newlines      (\n\n)     → Paragraph breaks
  3. Single newlines      (\n)       → Line breaks
  4. Sentence boundaries  (". ")     → Sentence splits
  5. Clause boundaries    (", ")     → Sub-sentence splits
  6. Word boundaries      (" ")      → Word splits
  7. Character level      ("")       → Fallback

Parameters:
  chunk_size    = 1000 characters
  chunk_overlap = 200 characters
```

Each chunk is tagged with a **section label** detected by keyword matching:

| Section | Detected by keywords |
|---|---|
| `benefits` | "benefit", "coverage", "covers", "insured amount" |
| `exclusions` | "exclusion", "not covered", "does not cover" |
| `premium` | "premium", "payment", "contribution" |
| `eligibility` | "eligibility", "age limit", "entry age" |
| `riders` | "rider", "additional benefit", "optional" |
| `claims` | "claim", "procedure", "how to claim" |
| `definitions` | "definition", "means", "shall mean" |
| `general` | everything else |

---

## Environment Variables

### Backend (`InsureMatch-backend/.env`)

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key from console.groq.com |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model ID |
| `CHROMA_PERSIST_DIR` | `./vectordb` | Where ChromaDB stores data |
| `CHROMA_COLLECTION_NAME` | `insurance_policies` | ChromaDB collection name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `RETRIEVAL_K` | `8` | Number of chunks to return per query |
| `RETRIEVAL_FETCH_K` | `20` | Fetch pool size for MMR reranking |
| `MODEL_SAVE_DIR` | `./saved_models` | Where XGBoost model is saved |
| `DATA_DIR` | `./data/raw` | Raw document directory |

### Frontend (`InsureMatch-frontend/.env.local`)

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API base URL |

---

## Notes

- **First run:** XGBoost training takes ~10–20 seconds. The model is saved and reused on subsequent runs.
- **Embeddings:** Downloaded once on first use (~90MB). Cached by HuggingFace locally.
- **Chat sessions:** Stored in memory. Sessions reset on server restart. For production, replace with Redis.
- **Policy metadata:** If you don't provide metadata when uploading, the LLM will extract it automatically from the document content. Providing it manually is faster and more accurate.
- **GROQ rate limits:** The free Groq tier has generous rate limits for LLaMA 3.3 70B. No issues expected for development use.
- **Type source of truth:** Backend Pydantic models in `app/models/schemas.py` are the source of truth. Frontend TypeScript types in `src/lib/types/api.ts` must mirror these exactly.
- **CORS:** The backend allows all origins (`*`) for development. Restrict this in production.
