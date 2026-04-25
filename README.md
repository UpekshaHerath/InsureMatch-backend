# InsureMatch — AI Insurance Recommendation System

A production-ready **Retrieval-Augmented Generation (RAG)** platform that recommends the best insurance policy for a user — and the best **add-on riders** to close their coverage gaps — based on their personal, financial, health, and lifestyle profile.

Built with **FastAPI**, **LangChain**, **Groq LLM**, **ChromaDB**, **XGBoost + SHAP**, and **Supabase** (auth + persistence) — paired with a **Next.js 16.2** frontend for a complete end-to-end experience.

**Client:** Union Assurance — a major insurance company in Sri Lanka.

---

## Table of Contents

- [Overview](#overview)
- [Business Value & Problem Statement](#business-value--problem-statement)
- [System Architecture](#system-architecture)
- [Full Pipeline Flow](#full-pipeline-flow)
- [AI / ML Capabilities Deep Dive](#ai--ml-capabilities-deep-dive)
- [Authentication & Authorization](#authentication--authorization)
- [Rider Recommendation System](#rider-recommendation-system)
- [Persistence Layer (Supabase)](#persistence-layer-supabase)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Frontend](#frontend)
- [API Endpoints](#api-endpoints)
- [User Profile Schema](#user-profile-schema)
- [Sample API Calls](#sample-api-calls)
- [SHAP Explainability](#shap-explainability)
- [Chunking Strategy](#chunking-strategy)
- [Environment Variables](#environment-variables)
- [Notes & Production Considerations](#notes--production-considerations)

---

## Overview

InsureMatch answers two questions at once:

> *"Given a user's age, income, health conditions, employment, and insurance goals — which insurance policy is the best fit, and why?"*
>
> *"And which optional riders should they add to close the gaps the base policy leaves open?"*

The system:
1. Ingests insurance **policy documents** (PDF, DOCX, TXT) into a **ChromaDB** vector database, with structured metadata stored in a **policy registry**.
2. Ingests a separate **riders bundle document** — the LLM extracts each rider, its category, age limits, applicable policies, and target user profile.
3. Authenticates users through **Supabase** (email/password + Google OAuth) with a two-tier role model (`admin`, `client`).
4. Accepts a detailed **user profile** via the frontend form (or API directly).
5. Scores all available policies using an **XGBoost regressor** trained on synthetic data that encodes insurance domain rules.
6. Explains *why* each policy scored the way it did using **SHAP TreeExplainer** (per-feature attribution, mapped to human-readable factors).
7. **Ranks riders** for the top policies using a deterministic **rule-based gap-closer scorer** — no synthetic training data needed because the rules *are* the explanation.
8. Retrieves relevant policy clauses via **MMR semantic search** (RAG).
9. Synthesizes everything into a **natural language recommendation** using **Groq LLM** (LLaMA 3.3 70B Versatile).
10. Persists the user's profile, every recommendation, and the full chat history in **Supabase** so users can revisit past advice on the `/account` page.
11. Provides a **scope-locked conversational chat** for follow-up questions, with retrieval grounded in the same vector database.

### End-to-End User Flow

```
Sign up / Sign in → Landing → 6-Step Profile Form → AI Recommendation → SHAP Factors + Rider Suggestions → Follow-up Chat → Account history
   /login /signup     /         /profile             POST /api/recommend     /results                     /chat            /account
```

---

## Business Value & Problem Statement

Insurance products are dense, lifetime-binding, and full of hidden incompatibilities (age cut-offs, health loadings, hazardous-occupation exclusions, regional regulatory flags). A typical Sri Lankan client comparing 8–15 products faces:

- **Catalog overload** — they cannot read every booklet.
- **Opaque suitability** — even reading them, they cannot tell which one *fits them best*.
- **Gap blindness** — they don't know what their chosen base policy *doesn't* cover, so they buy the policy and discover the gap when a claim is denied.
- **Trust deficit** — agents are commissioned, so recommendations look biased.

InsureMatch solves all four:

| Problem | InsureMatch's answer |
|---|---|
| Catalog overload | Ranks policies automatically using XGBoost suitability score in [0, 1]. |
| Opaque suitability | SHAP per-feature explanations — *exactly* which factors moved the score up or down. |
| Gap blindness | Rule-based rider scorer surfaces add-ons that close coverage gaps for **this specific user** with this specific base policy. |
| Trust deficit | Deterministic ML + transparent rules + sourced RAG narrative. Every claim cites the underlying policy clause. |

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                                INSUREMATCH SYSTEM                                  │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                        FRONTEND (Next.js 16.2 — App Router)                  │  │
│  │                                                                              │  │
│  │  /login • /signup • /auth/callback                                           │  │
│  │  / (Landing) → /profile (6-step wizard) → /results → /chat → /account        │  │
│  │  /admin/policies  /admin/riders  (admin role only)                           │  │
│  │                                                                              │  │
│  │  Tech: TypeScript, Tailwind v4, shadcn/ui, Zustand, TanStack Query,          │  │
│  │        React Hook Form + Zod, @supabase/ssr middleware                       │  │
│  └─────────────────┬───────────────────────────────────────────────────────────┘  │
│                    │ Axios + Bearer JWT (Supabase access token)                   │
│                    ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────────┐│
│  │                       BACKEND (FastAPI + Uvicorn)                            ││
│  │                                                                              ││
│  │  ┌────────────────────────┐   JWT (HS256/ES256/RS256) verified per request  ││
│  │  │  Auth Middleware       │ ◄─ Supabase JWKS / JWT secret                   ││
│  │  │  get_current_user      │                                                 ││
│  │  │  require_admin         │                                                 ││
│  │  └───────────┬────────────┘                                                 ││
│  │              │                                                              ││
│  │  ┌───────────▼──────────┐  ┌─────────────────────┐  ┌────────────────────┐ ││
│  │  │  INGESTION           │  │  RECOMMENDATION     │  │  CHAT (per-user    │ ││
│  │  │  (admin only)        │  │  ENGINE             │  │  scoped sessions)  │ ││
│  │  │                      │  │                     │  │                    │ ││
│  │  │  policies + riders   │  │  XGBoost scorer →   │  │  history loaded    │ ││
│  │  │  → loader            │  │  SHAP explainer →   │  │  from Supabase →   │ ││
│  │  │  → chunker           │  │  Rider gap-closer → │  │  MMR retrieval →   │ ││
│  │  │  → embeddings        │  │  RAG narrative      │  │  scope-locked LLM  │ ││
│  │  │  → ChromaDB +        │  │  (Groq LLM)         │  │  → persist turn    │ ││
│  │  │    registry JSONs    │  │                     │  │                    │ ││
│  │  └───────┬──────────────┘  └─────────┬───────────┘  └─────────┬──────────┘ ││
│  │          │                           │                        │            ││
│  └──────────┼───────────────────────────┼────────────────────────┼────────────┘│
│             ▼                           ▼                        ▼              │
│  ┌──────────────────┐      ┌──────────────────────┐    ┌────────────────────┐   │
│  │  ChromaDB        │      │  saved_models/       │    │  Supabase Postgres │   │
│  │  (persistent     │      │  ├ xgb_scorer.pkl    │    │  ├ profiles       │   │
│  │  vector store)   │      │  ├ policy_registry  │    │  ├ recommendations │   │
│  │                  │      │  └ rider_registry   │    │  ├ chat_sessions   │   │
│  │  policy + rider  │      │     (JSON)          │    │  └ chat_messages   │   │
│  │  chunks          │      │                     │    │                    │   │
│  │  + metadata      │      │                     │    │  + Auth (users,    │   │
│  │                  │      │                     │    │    JWT, Google OAuth) │
│  └──────────────────┘      └─────────────────────┘    └────────────────────┘   │
│                                                                                  │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Full Pipeline Flow

### Phase 1 — Policy Ingestion (admin)

```
Insurance Policy Document (PDF / DOCX / TXT)
           │
           ▼  POST /api/ingest  (Bearer admin JWT)
    Document Loader (PyPDF / Docx2txt / TextLoader)
           │
           ▼
    Recursive Semantic Chunker
    (insurance-aware separators, 1000 char / 200 overlap)
           │
           ▼
    Section Detection
    (benefits / exclusions / premium / eligibility / claims / riders / definitions)
           │
           ▼
    Metadata extraction
    ├─ form-supplied fields take precedence
    └─ otherwise → Groq LLM extracts via METADATA_EXTRACTION_PROMPT
           │
           ▼
    Per-chunk metadata: { policy_name, policy_type, company, section,
                          page_num, chunk_index, min_age, max_age,
                          premium_level, covers_*, is_entry_level }
           │
           ├──► HuggingFace embeddings (all-MiniLM-L6-v2)
           │           │
           │           ▼
           │    ChromaDB (persistent collection)
           │
           └──► saved_models/policy_registry.json
                (structured metadata read by the XGBoost scorer)
```

### Phase 2 — Rider Ingestion (admin)

```
Single Riders-Bundle Document
           │
           ▼  POST /api/riders   (Bearer admin JWT)
    Loader → full text
           │
           ▼
    Groq LLM (RIDERS_EXTRACTION_PROMPT)
    ┌──────────────────────────────────────────────────────┐
    │  Input: document excerpt + list of known policy names│
    │  Output: JSON array of riders, each with             │
    │    rider_name, rider_code, category,                 │
    │    min_age/max_age, premium_level,                   │
    │    applicable_policies (constrained to known list),  │
    │    target_goals, health_relevant, hazard_relevant,   │
    │    dependents_relevant                               │
    └──────────────────────────────────────────────────────┘
           │
           │ (replace=True ⇒ wipe previous catalog)
           ▼
    saved_models/rider_registry.json   ← structured rider catalog
           │
    ChromaDB ← chunks of the bundle with doc_type="rider"
              for retrieval inside chat
```

### Phase 3 — Authenticated Recommendation

```
POST /api/recommend                Bearer <Supabase access token>
{ user_profile, top_k }
           │
           ▼
  Auth: get_current_user → AuthUser{user_id, role}
           │
           ▼
  Load policy registry → ValueError(422) if empty
           │
           ▼
  Feature extraction
    ├─ extract_user_features      (21 normalized features)
    └─ extract_policy_features    (10 normalized features per policy)
           │
           ▼
  XGBoost regressor predicts suitability ∈ [0, 1] per policy
           │
           ▼
  Sort → take top_k
           │
           ▼
  SHAP TreeExplainer
    └─ map raw shap values → human-readable {feature, impact, direction, reason}
           │
           ▼
  Generate RAG narrative
    ├─ MMR retrieval from ChromaDB (k=8 from fetch_k=20)
    ├─ Compose prompt: profile + SHAP summary + RAG context
    └─ Groq LLM (creative temp=0.3)
           │
           ▼
  Rank riders for each top-K policy (rule-based gap-closer)
    └─ rank_riders_for_policy(profile, policy_meta, rider_registry)
           │
           ▼
  RecommendationResponse = {
     ranked_policies, top_recommendation,
     explanations (SHAP factors),
     rag_narrative,
     session_id (UUID, for chat),
     rider_suggestions: { policy_name → [RiderRecommendation, ...] }
  }
           │
           ▼
  Best-effort persistence (failures logged, do not break the response):
    ├─ upsert_profile        (profiles table)
    ├─ ensure_session        (chat_sessions table)
    └─ insert_recommendation (recommendations table)
```

### Phase 4 — Chat

```
POST /api/chat                    Bearer <user JWT>
{ session_id, message, user_profile?, recommendation_context? }
           │
           ▼
  ensure_session(user_id, session_id) — owner-scoped
  list_messages(session_id, user_id, limit=12)  ← persisted history
           │
           ▼
  retrieval_query = message [+ user profile context]
  ChromaDB MMR retrieval (k=6)
           │
           ▼
  Groq LLM with CHAT_PROMPT
    └─ STRICT scope: only insurance topics; off-topic ⇒ refusal sentence
           │
           ▼
  Persist turn:
    ├─ insert_message(role=human)
    ├─ insert_message(role=ai, sources=[...])
    └─ touch_session(last_active=now())
           │
           ▼
  ChatResponse { session_id, response, sources }
```

---

## AI / ML Capabilities Deep Dive

InsureMatch combines **four** distinct AI techniques. Each has a clearly defined role:

### 1. XGBoost Regressor — Policy Suitability Scoring
- **Why ML, not rules?** The interaction between age × goal × health × hazardous level × premium level is too high-dimensional for a clean if/else tree. A gradient-boosted regressor learns these interactions while staying explainable.
- **Inputs:** 31-dim feature vector = 21 user features + 10 policy features (all normalized to [0, 1]).
- **Training:** 8,000 synthetic `(user, policy) → score` pairs generated by `_domain_score()` — a deterministic function encoding insurance domain rules (age eligibility, goal alignment, condition × coverage matching, income vs premium, hazardous work). Adds Gaussian noise (σ=0.04) so the model generalises rather than memorises.
- **Hyperparameters:** `n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror"`.
- **Persistence:** trained once at first startup, saved to `saved_models/xgb_scorer.pkl`, reused on every subsequent boot.
- **Inference:** ~ms per policy — runs synchronously inside the request.

### 2. SHAP TreeExplainer — Per-Feature Attribution
- **Why SHAP, not feature importance?** Global feature importance tells you *the model* cares about age. SHAP tells you *for this user*, age contributed +0.198 to *this score*. Crucial for trust.
- **Output mapping:** Raw float SHAP values are mapped to `SHAPFactor` objects via `FEATURE_DESCRIPTIONS` and `REASON_TEMPLATES` in `explainer.py`, producing human-readable lines like *"Your age falls within the ideal range for this policy."*
- **Top-N split:** Top positive and top negative factors are surfaced separately (5 each in `/recommend`, 7 each in `/explain`).

### 3. RAG (ChromaDB + LangChain + Groq)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` via `langchain_huggingface` (CPU, normalized). Free, ~90 MB, downloaded once.
- **Vector store:** ChromaDB (persistent, local). One collection holds both policy chunks and rider-bundle chunks (`doc_type` metadata distinguishes them).
- **Retrieval mode:** `max_marginal_relevance_search` — diversifies the chunks returned so the LLM sees benefits *and* exclusions *and* premium info, not 8 paragraphs from the same section.
- **Generation:** `ChatGroq(model=llama-3.3-70b-versatile)` — temp=0.0 for deterministic metadata extraction, temp=0.3 for narrative.
- **Prompt library** (`app/core/rag/prompts.py`):
  - `RECOMMENDATION_PROMPT` — narrative for the top policy.
  - `CHAT_PROMPT` — scope-locked Q&A. Refuses anything not insurance-related; immune to "ignore previous instructions" injection attempts.
  - `METADATA_EXTRACTION_PROMPT` — JSON-only structured extraction of policy metadata.
  - `RIDERS_EXTRACTION_PROMPT` — JSON array extraction of all riders in a bundle, with applicability constrained to the existing policy registry.

### 4. Rule-Based Rider Gap-Closer
- **Why rules, not ML?** Riders have no historical click/conversion training data, and their value to a user is inherently rule-shaped: *"if user has dependents AND policy doesn't cover income loss, suggest income-protection rider."* Rules ARE the explanation — no SHAP layer needed.
- **Implementation:** `app/core/recommendation/rider_scorer.py`.
- **Hard filters** (return score = 0):
  - User age outside the rider's `[min_age, max_age]`.
  - Rider's `applicable_policies` list non-empty and does not contain the base policy.
- **Soft signals** (additive, then clamped to [0, 1]):
  - Category-specific gap detection (critical illness × health risk × policy doesn't cover health, etc.).
  - Hazard relevance × medium/high hazardous occupation.
  - Dependents relevance × `num_dependents > 0`.
  - Goal alignment (rider's `target_goals` includes user's `primary_goal`).
- **Output:** Top 3 riders per top-K policy, each with a score and a deduplicated list of natural-language reasons.

---

## Authentication & Authorization

InsureMatch uses **Supabase Auth** end-to-end. There is **no custom user database** — Supabase owns identity, the FastAPI backend owns business logic.

### Roles

| Role | Source | Frontend access | Backend access |
|---|---|---|---|
| `admin` | `app_metadata.role = "admin"` set in Supabase dashboard | All pages including `/admin/policies`, `/admin/riders` | Ingestion routes (policies, riders), policy admin endpoints |
| `client` (default) | New users default to `client` (also accepted: `authenticated`) | Profile, recommend, chat, account | Recommendation, explain, chat |
| anonymous | no token | `/`, `/login`, `/signup`, `/auth/callback` only | none — every API route 401s |

### Sign-in methods

- **Email + password** — `supabase.auth.signInWithPassword()` on `/login`.
- **Sign-up with email confirmation** — `supabase.auth.signUp()` on `/signup`, with `emailRedirectTo` pointing at `/auth/callback`.
- **Google OAuth** — `GoogleSignInButton` triggers Supabase OAuth; callback at `/auth/callback` exchanges the code for a session via `supabase.auth.exchangeCodeForSession`.

### Frontend auth wiring

- `@supabase/ssr` browser + server clients (`src/lib/supabase/{client,server}.ts`).
- `src/middleware.ts` — runs on every navigation:
  - Redirects unauthenticated users on protected routes to `/login?next=<path>`.
  - Redirects non-admin users away from `/admin/*` to `/`.
- `useAuth()` hook (`src/lib/hooks/useAuth.ts`) — exposes `user`, `loading`, `role`, `isAdmin`, `signOut`.
- `apiClient` axios interceptor (`src/lib/api/client.ts`) — attaches `Authorization: Bearer <access_token>` from the current Supabase session to every backend request.

### Backend JWT verification

Implemented in `app/core/auth/deps.py`:

- Reads `Authorization: Bearer <jwt>` via FastAPI's `HTTPBearer`.
- Inspects header `alg`:
  - `HS256` → verifies with `SUPABASE_JWT_SECRET` (legacy mode).
  - `ES256` / `RS256` → fetches signing key from `${SUPABASE_URL}/auth/v1/.well-known/jwks.json` (cached 1 hour via `PyJWKClient`).
- Validates `aud="authenticated"`, requires `exp` and `sub` claims.
- Pulls `role` from `app_metadata.role` (or top-level `role`, defaults to `"authenticated"`).
- Returns `AuthUser(user_id, email, role, raw_claims)`.

Two FastAPI dependencies expose this:

```python
get_current_user    # any signed-in user (raises 401 if missing/invalid)
require_admin       # role == "admin" (raises 403 otherwise)
```

### Route-level protection

| Endpoint | Protection |
|---|---|
| `POST /api/ingest`, `DELETE /api/ingest/{name}` | `require_admin` |
| `POST /api/riders`, `GET /api/riders`, `DELETE /api/riders[*]` | `require_admin` |
| `GET /api/policies`, `/api/policies/registry`, `/api/policies/{name}` | `require_admin` |
| `POST /api/recommend` | `get_current_user` |
| `POST /api/explain` | `get_current_user` |
| `POST /api/chat`, `POST /api/chat/new`, `DELETE /api/chat/{id}` | `get_current_user` |
| `GET /health`, `GET /` | public |

Every authenticated route uses the `user_id` from the verified JWT to scope DB rows it reads or writes — there is no way for one user to read another user's chat history or profile.

---

## Rider Recommendation System

Riders ("add-on coverage") are first-class citizens in InsureMatch.

### Why riders matter
A user matched to *Term Life Basic* still loses out if they have:
- a hazardous job (no accident cover) → suggest **Accidental Death Benefit**
- chronic conditions (no health cover) → suggest **Critical Illness Rider** + **Hospital Cash**
- dependents (income-loss exposure) → suggest **Waiver of Premium**, **Income Protection**

### Ingestion model
- A **single bundle document** is uploaded (because in practice insurers publish all riders in one PDF).
- The LLM extracts each rider via `RIDERS_EXTRACTION_PROMPT`, with applicability **constrained to the existing `policy_registry`**. Riders that name unknown policies are dropped from `applicable_policies` to avoid hallucinated mappings.
- The bundle's full text is also chunked and stored in ChromaDB with `doc_type="rider"` so the chat can answer rider-specific questions ("what's the waiver-of-premium waiting period?").
- Re-uploading replaces the previous catalog (`replace=True` default) — `clear_rider_registry()` + `delete_all_rider_chunks()`.

### Categories
Each rider is normalised to one of: `critical_illness`, `accidental_death`, `waiver_of_premium`, `hospital_cash`, `income_protection`, `permanent_disability`, `term_extension`, `other`.

### Scoring (rule-based gap-closer)
See [`rider_scorer.py`](app/core/recommendation/rider_scorer.py) for the full table. Examples:

| Category | Strong signal (+0.5+) | Soft signal (+0.1–0.3) |
|---|---|---|
| `critical_illness` | health risk AND base policy lacks `covers_health` | health risk only OR no health cover only |
| `accidental_death` | medium/high hazard AND no accident cover | hazard only OR no accident cover only |
| `permanent_disability` | medium/high hazard | dependents > 0 |
| `waiver_of_premium` | dependents > 0 | variable income (contract/freelance/self-employed) |
| `income_protection` | dependents > 0 | base policy lacks life cover |
| `hospital_cash` | health risk AND no health cover | no health cover only |
| `term_extension` | base policy is `term_life` | — |

Plus generic boosts for declared `health_relevant` / `hazard_relevant` / `dependents_relevant` flags and goal alignment.

### Output
`RiderRecommendation { rider_name, rider_code, category, description, premium_level, score, reasons[] }`.
The `reasons` array is what the frontend renders as bullet points under "Suggested extra riders for you".

---

## Persistence Layer (Supabase)

The backend talks to Supabase via the **REST API using the service-role key** (bypassing Row Level Security). Ownership is enforced in code by always filtering on the `user_id` from the verified JWT — see `app/core/db/supabase_client.py`.

### Tables

| Table | Columns (highlights) | Written by |
|---|---|---|
| `profiles` | `user_id` (PK), all fields from `UserProfile` flattened (age, gender, occupation, monthly_income_lkr, has_chronic_disease, …) | `upsert_profile()` on every `/api/recommend` |
| `recommendations` | `id`, `user_id`, `session_id`, `top_recommendation`, `ranked_policies` (JSON), `explanations` (JSON), `rag_narrative`, `rider_suggestions` (JSON), `created_at` | `insert_recommendation()` on every `/api/recommend` |
| `chat_sessions` | `id` (session UUID, PK), `user_id`, `title`, `created_at`, `last_active` | `ensure_session()` / `touch_session()` |
| `chat_messages` | `id`, `session_id`, `user_id`, `role` (`human`/`ai`), `content`, `sources` (string[]), `created_at` | `insert_message()` on every chat turn |

The `/account` page (`src/app/account/page.tsx`) reads from `recommendations` directly via the **anon key + RLS** (Supabase's row-level security ensures the user only sees their own rows).

### Failure semantics
All write paths in `recommend.py` are best-effort: they log the full traceback and continue, so a transient Supabase failure does **not** break the recommendation response. Read paths (chat history) silently fall back to an empty list and a warning log.

---

## Tech Stack

### Backend

| Component | Technology |
|---|---|
| Web framework | FastAPI 0.135 + Uvicorn 0.41 |
| LLM | Groq — LLaMA 3.3 70B Versatile via `langchain-groq` |
| RAG orchestration | LangChain 1.2.x (`langchain`, `langchain-core`, `langchain-classic`, `langchain-community`, `langchain-text-splitters`) |
| Vector database | ChromaDB 1.5 (local, persistent) via `langchain-chroma` |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| ML scoring | XGBoost 3.2 (synthetic-domain-trained) + joblib |
| Explainability | SHAP 0.51 `TreeExplainer` |
| Document loaders | PyPDF, Docx2txt, TextLoader |
| Auth | Supabase JWT verification via `PyJWT` + `PyJWKClient` |
| DB / persistence | Supabase REST (httpx async client) |
| Data validation | Pydantic v2 |

### Frontend

| Component | Technology |
|---|---|
| Framework | Next.js 16.2 (App Router, TypeScript, React 19) |
| UI components | shadcn/ui (base-ui primitives) + lucide-react icons |
| Styling | Tailwind CSS v4 |
| Auth | `@supabase/ssr` + `@supabase/supabase-js` |
| State management | Zustand 5 (sessionStorage persist) |
| Server state | TanStack React Query 5 |
| Form handling | React Hook Form + Zod |
| HTTP client | Axios (with Supabase JWT interceptor) |
| Component pattern | Atomic Design |

> **Frontend note:** Next.js 16.2 has breaking changes from earlier major versions (per `InsureMatch-frontend/AGENTS.md`). Consult `node_modules/next/dist/docs/` when touching framework-specific APIs.

---

## Project Structure

```
insurance_recommendation_rag/
│
├── InsureMatch-backend/                    # ── FastAPI Backend ──
│   ├── app/
│   │   ├── main.py                         # FastAPI app, lifespan, CORS, routers
│   │   ├── config.py                       # Settings via pydantic-settings + .env
│   │   │
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── ingest.py               # POST /api/ingest, DELETE /api/ingest/{name}   (admin)
│   │   │       ├── riders.py               # POST /api/riders, GET, DELETE                  (admin)
│   │   │       ├── recommend.py            # POST /api/recommend                            (auth)
│   │   │       ├── explain.py              # POST /api/explain                              (auth)
│   │   │       ├── chat.py                 # POST /api/chat, /chat/new, DELETE /chat/{id}   (auth)
│   │   │       └── policies.py             # GET  /api/policies, /registry, /{name}        (admin)
│   │   │
│   │   ├── core/
│   │   │   ├── auth/
│   │   │   │   └── deps.py                 # JWT verification (HS256 + JWKS for ES/RS),
│   │   │   │                               # get_current_user / require_admin
│   │   │   ├── db/
│   │   │   │   └── supabase_client.py      # Async Supabase REST wrapper (profiles,
│   │   │   │                               # recommendations, chat_sessions, chat_messages)
│   │   │   │
│   │   │   ├── ingestion/
│   │   │   │   ├── loader.py               # PDF / DOCX / TXT document loaders
│   │   │   │   └── chunker.py              # Recursive semantic chunker, section detector,
│   │   │   │                               # rider-document chunker (doc_type="rider")
│   │   │   │
│   │   │   ├── vectorstore/
│   │   │   │   └── chroma_store.py         # ChromaDB wrapper + policy registry I/O
│   │   │   │                               # + rider registry I/O + rider chunk cleanup
│   │   │   │
│   │   │   ├── rag/
│   │   │   │   ├── prompts.py              # RECOMMENDATION_PROMPT, CHAT_PROMPT,
│   │   │   │   │                           # METADATA_EXTRACTION_PROMPT, RIDERS_EXTRACTION_PROMPT
│   │   │   │   └── chain.py                # RAG narrative, chat (Supabase-backed history),
│   │   │   │                               # extract_policy_metadata_with_llm, extract_riders_with_llm
│   │   │   │
│   │   │   ├── recommendation/
│   │   │   │   ├── scorer.py               # XGBoost model, 31-feature extraction, training
│   │   │   │   ├── explainer.py            # SHAP TreeExplainer + human-readable factors
│   │   │   │   ├── rider_scorer.py         # Rule-based rider gap-closer
│   │   │   │   └── ranker.py               # Pipeline orchestrator (ML + SHAP + riders + RAG)
│   │   │   │
│   │   │   └── llm/
│   │   │       └── groq_llm.py             # ChatGroq factory (standard + creative)
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
│   │   ├── policy_registry.json            # Structured policy metadata (auto-created)
│   │   └── rider_registry.json             # Structured rider catalog (auto-created)
│   │
│   ├── .env                                # Your environment variables
│   ├── .env.example                        # Environment variable template
│   ├── requirements.txt                    # Pinned Python dependencies
│   ├── requirements-flexible.txt           # Loose constraints (preferred for dev)
│   └── README.md                           # This file
│
├── InsureMatch-frontend/                   # ── Next.js Frontend ──
│   ├── src/
│   │   ├── middleware.ts                   # Supabase SSR auth middleware (route gating)
│   │   ├── app/
│   │   │   ├── layout.tsx                  # Root layout + QueryProvider
│   │   │   ├── page.tsx                    # Landing
│   │   │   ├── not-found.tsx               # 404
│   │   │   ├── login/page.tsx              # Email + Google sign-in
│   │   │   ├── signup/page.tsx             # Email + Google sign-up
│   │   │   ├── auth/callback/route.ts      # OAuth code exchange
│   │   │   ├── account/page.tsx            # Authenticated user history
│   │   │   ├── profile/page.tsx            # 6-step recommendation form
│   │   │   ├── results/page.tsx            # Recommendation results
│   │   │   ├── chat/page.tsx               # Follow-up Q&A
│   │   │   └── admin/
│   │   │       ├── policies/page.tsx       # Browse + delete indexed policies (admin only)
│   │   │       └── riders/page.tsx         # Manage rider catalog (admin only)
│   │   │
│   │   ├── components/
│   │   │   ├── ui/                         # shadcn/ui generated components
│   │   │   ├── atoms/                      # Spinner, ProgressBar, ScoreBadge, ImpactIndicator
│   │   │   ├── molecules/                  # FormField, SelectField, CheckboxField, StepIndicator,
│   │   │   │                               # PolicyCard, SHAPFactorRow, ChatBubble, SourceChip,
│   │   │   │                               # GoogleSignInButton, UserMenu
│   │   │   ├── organisms/                  # Navbar (role-aware), Footer, StepperHeader,
│   │   │   │                               # 5 form steps, ReviewSummary, PolicyResultCard
│   │   │   │                               # (with rider suggestions), NarrativePanel, ChatWindow,
│   │   │   │                               # FloatingChat, PolicyListTable, PolicyUploadForm,
│   │   │   │                               # RiderListTable, RiderUploadForm
│   │   │   └── templates/                  # FormWizardTemplate, ResultsTemplate,
│   │   │                                   # ChatTemplate, PoliciesTemplate, RidersTemplate
│   │   │
│   │   ├── lib/
│   │   │   ├── api/                        # Axios client (Bearer JWT) + endpoint constants
│   │   │   ├── hooks/                      # useRecommendation, useChat, useExplain,
│   │   │   │                               # usePolicies, useRiders, useAuth
│   │   │   ├── store/                      # Zustand (useProfileStore)
│   │   │   ├── supabase/                   # Browser + server SSR clients
│   │   │   ├── types/                      # TypeScript types mirroring backend schemas
│   │   │   ├── validators/                 # Zod schemas per form step
│   │   │   └── utils/                      # Formatters + constants
│   │   │
│   │   └── providers/
│   │       └── QueryProvider.tsx           # TanStack QueryClientProvider
│   │
│   ├── .env.local                          # NEXT_PUBLIC_API_URL + Supabase keys
│   ├── components.json                     # shadcn/ui config
│   ├── package.json                        # Node dependencies
│   ├── tsconfig.json                       # TypeScript config
│   └── AGENTS.md                           # Frontend-specific guidance
│
└── CLAUDE.md                               # Root-level Claude Code guidance
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- A [Groq API key](https://console.groq.com)
- A [Supabase project](https://supabase.com) with:
  - Email + Google OAuth providers enabled
  - Tables `profiles`, `recommendations`, `chat_sessions`, `chat_messages` (see [Persistence Layer](#persistence-layer-supabase) for columns)
  - For admin users: set `app_metadata.role = "admin"` in the Supabase dashboard (Auth → Users → Edit user)

### Backend Setup

```bash
cd InsureMatch-backend

# Virtual environment
python -m venv venv
source venv/Scripts/activate     # bash on Windows
# venv\Scripts\Activate.ps1      # PowerShell
# venv\Scripts\activate.bat      # CMD

# Dependencies
pip install -r requirements.txt
# or for looser constraints during development:
pip install -r requirements-flexible.txt
```

> First run downloads the HuggingFace embedding model (~90 MB). Cached locally.

### Configure backend environment

Copy `.env.example` → `.env` and fill in:

```env
# Groq LLM
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Supabase
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_JWT_SECRET=your_jwt_secret

# ChromaDB
CHROMA_PERSIST_DIR=./vectordb
CHROMA_COLLECTION_NAME=insurance_policies

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Chunking + Retrieval
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=8
RETRIEVAL_FETCH_K=20

# Storage
MODEL_SAVE_DIR=./saved_models
DATA_DIR=./data/raw
```

The service-role key bypasses RLS — keep it secret, never expose it to the frontend.

### Frontend Setup

```bash
cd InsureMatch-frontend
npm install
```

Create `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=https://YOUR_PROJECT.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
```

The anon key is safe to expose — it's combined with Supabase RLS for protection.

---

## Running the Project

### Start the backend

```bash
cd InsureMatch-backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On startup the server:
1. Creates `vectordb/`, `saved_models/`, `data/raw/`.
2. Trains the XGBoost scorer on synthetic data (first run only, ~10 s) and saves it to `saved_models/xgb_scorer.pkl`.
3. Initialises the Supabase JWKS client (lazy — first request).

### Start the frontend

```bash
cd InsureMatch-frontend
npm run dev          # http://localhost:3000
```

### API docs

```
http://localhost:8000/docs      # Swagger UI
http://localhost:8000/redoc     # ReDoc
```

### Health check

```bash
curl http://localhost:8000/health
```

```json
{ "status": "healthy", "policies_indexed": 0, "policies_in_registry": 0 }
```

### Bootstrap order (fresh install)

1. Sign up at `/signup`. In the Supabase dashboard, set `app_metadata.role = "admin"` for your user.
2. Sign back in. Navigate to `/admin/policies` and upload several policy documents.
3. Navigate to `/admin/riders` and upload the riders bundle document.
4. Sign up a second (non-admin) user — or stay signed in as admin — and run a recommendation through `/profile`.

---

## Frontend

The frontend is a **Next.js 16.2** app with an orange-themed Union Assurance UI.

### Pages

| Route | Auth | Purpose |
|---|---|---|
| `/` | public | Landing — hero, "How it works", CTA |
| `/login`, `/signup` | public | Email + Google sign-in / sign-up |
| `/auth/callback` | public | OAuth code exchange |
| `/profile` | client | 6-step wizard (Personal → Occupation → Goals → Health → Lifestyle → Review) |
| `/results` | client | Ranked policies, SHAP factors, AI narrative, **rider suggestions per policy** |
| `/chat` | client | Scope-locked Q&A; per-user persistent history |
| `/account` | client | List of past recommendations from Supabase |
| `/admin/policies` | admin | Upload, list, delete policies |
| `/admin/riders` | admin | Upload riders bundle, list, delete |

### State management
- **Zustand store** (`useProfileStore`) holds form data across the 6 steps; persisted to `sessionStorage`.
- **TanStack React Query** manages server state (`useRecommendation`, `useChat`, `useExplain`, `usePolicies`, `useRiders`).
- **Zod schemas** validate each form step before advancing.
- **Supabase auth state** is held by `useAuth()` and listened to via `onAuthStateChange`.

### Theme
- **Primary:** `#F97316` (orange-500) — buttons, stepper, accents
- **Secondary:** `#1E293B` (slate-800) — headings, navbar
- **Background:** white, **Cards:** `#F8FAFC` (slate-50)
- Stepper: orange active, gray pending, green completed

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/ingest` | admin | Upload an insurance policy document |
| `DELETE` | `/api/ingest/{policy_name}` | admin | Delete a policy + its chunks |
| `POST` | `/api/riders` | admin | Upload the riders bundle (LLM extracts every rider) |
| `GET` | `/api/riders` | admin | List all riders in the registry |
| `DELETE` | `/api/riders/{rider_code}` | admin | Delete a single rider |
| `DELETE` | `/api/riders` | admin | Wipe the rider catalog |
| `POST` | `/api/recommend` | user | Submit profile, get ranked policies + SHAP + narrative + rider suggestions |
| `POST` | `/api/explain` | user | Deep SHAP explanation for one specific policy |
| `POST` | `/api/chat` | user | Conversational Q&A (Supabase-persisted history) |
| `POST` | `/api/chat/new` | user | Generate a fresh session ID |
| `DELETE` | `/api/chat/{session_id}` | user | Clear a session (owner-only) |
| `GET` | `/api/policies` | admin | List indexed policies (chunk counts) |
| `GET` | `/api/policies/registry` | admin | Full registry with all metadata |
| `GET` | `/api/policies/{policy_name}` | admin | Metadata for one policy |
| `GET` | `/health` | public | Health check |

All authenticated endpoints expect `Authorization: Bearer <Supabase access token>`.

### Detailed examples

#### `POST /api/ingest` (multipart/form-data)

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | PDF, DOCX, or TXT |
| `policy_name` | string | Optional | Override extracted policy name |
| `policy_type` | string | Optional | `term_life` \| `whole_life` \| `endowment` \| `health` \| `critical_illness` \| `accident` |
| `company` | string | Optional | Insurance company name |
| `min_age`, `max_age` | int | Optional | Entry-age limits |
| `premium_level` | int | Optional | `0`=low, `1`=medium, `2`=high |
| `covers_health`, `covers_life`, `covers_accident` | bool | Optional | Coverage flags |
| `is_entry_level` | bool | Optional | Basic/affordable product flag |

If `policy_name` or `policy_type` is omitted, the LLM auto-extracts them from the document content.

```json
{
  "message": "Successfully indexed 'AIA Term Life Basic'",
  "policy_name": "AIA Term Life Basic",
  "chunks_indexed": 47,
  "policy_metadata": { ... }
}
```

#### `POST /api/riders` (multipart/form-data)

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | A single document listing all riders |
| `replace` | bool | default `true` | Wipe previous rider catalog before ingesting |

Errors:
- `422` if no policies are indexed yet (riders must reference known policies).
- `422` if the LLM extracts zero riders.

```json
{
  "message": "Successfully indexed 5 riders.",
  "riders_extracted": 5,
  "chunks_indexed": 31,
  "riders": [
    {
      "rider_name": "Critical Illness Rider",
      "rider_code": "CI-RIDER",
      "category": "critical_illness",
      "company": "AIA Sri Lanka",
      "description": "Covers 30 specified critical illnesses with a lump-sum payout.",
      "min_age": 18, "max_age": 65, "premium_level": 1,
      "applicable_policies": ["AIA Term Life Basic", "AIA Whole Life"],
      "target_goals": ["health_coverage", "protection"],
      "health_relevant": true,
      "hazard_relevant": false,
      "dependents_relevant": false
    }
  ]
}
```

#### `POST /api/recommend`

Request:

```json
{
  "user_profile": {
    "personal":   { "age": 20, "gender": "male", "marital_status": "single", "num_dependents": 0, "district": "Gampaha", "city": "Negombo" },
    "occupation": { "occupation": "Graphic Designer", "employment_type": "contract", "designation": "Junior", "hazardous_level": "low", "monthly_income_lkr": 140000, "has_existing_insurance": false, "current_insurance_status": "none" },
    "goals":      { "primary_goal": "cheap_and_quick", "secondary_goal": "none", "travel_history_high_risk": false, "dual_citizenship": false, "tax_regulatory_flags": false, "insurance_history_issues": false },
    "health":     { "has_chronic_disease": false, "has_cardiovascular": false, "has_cancer": false, "has_respiratory": false, "has_neurological": false, "has_gastrointestinal": false, "has_musculoskeletal": false, "has_infectious_sexual": false, "recent_treatment_surgery": false, "covid_related": false },
    "lifestyle":  { "bmi": 22.5, "is_smoker": false, "is_alcohol_consumer": false }
  },
  "top_k": 3
}
```

Response (truncated):

```json
{
  "ranked_policies": [
    { "policy_name": "AIA Term Life Basic", "policy_type": "term_life", "company": "AIA Sri Lanka", "suitability_score": 0.8734, "rank": 1 },
    { "policy_name": "Personal Accident Shield", "policy_type": "accident", "company": "Ceylinco Life", "suitability_score": 0.7421, "rank": 2 }
  ],
  "top_recommendation": "AIA Term Life Basic",
  "explanations": [ { "policy_name": "...", "positive_factors": [...], "negative_factors": [...], "shap_summary": "..." } ],
  "rag_narrative": "Based on your profile as a 20-year-old graphic designer in Negombo...",
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "rider_suggestions": {
    "AIA Term Life Basic": [
      {
        "rider_name": "Accidental Death Benefit",
        "rider_code": "ADB",
        "category": "accidental_death",
        "description": "Pays an additional sum on accidental death.",
        "premium_level": 0,
        "score": 0.55,
        "reasons": [
          "Your occupation carries a low hazardous level and the base policy has no accident coverage.",
          "Aligned with your primary goal (cheap and quick)."
        ]
      }
    ]
  }
}
```

Side effects (best-effort):
- `profiles` row upserted (one per user).
- `chat_sessions` row created (`id = session_id`).
- `recommendations` row inserted (full response JSON).

#### `POST /api/explain`

Same shape as one entry of `explanations` but with up to 7 factors per direction.

#### `POST /api/chat`

```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "message": "What are the exclusions for the AIA Term Life policy?",
  "user_profile": null,
  "recommendation_context": null
}
```

Response:

```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "response": "Suicide within 12 months and undisclosed pre-existing conditions are excluded.",
  "sources": ["AIA Term Life Basic"]
}
```

If the question is off-topic the response is exactly:

> *"I'm sorry, I can only help with questions about your insurance recommendations and policies. I don't have knowledge about that topic."*

#### `GET /api/riders` (admin)

```json
[
  {
    "rider_name": "Waiver of Premium",
    "rider_code": "WOP",
    "category": "waiver_of_premium",
    "company": "AIA Sri Lanka",
    "description": "Waives future premiums on permanent disability.",
    "min_age": 18, "max_age": 60, "premium_level": 0,
    "applicable_policies": ["AIA Term Life Basic", "AIA Whole Life"],
    "target_goals": ["protection"],
    "health_relevant": false, "hazard_relevant": true, "dependents_relevant": true
  }
]
```

---

## User Profile Schema

All fields accepted by `POST /api/recommend`.

### Personal

| Field | Type | Constraints | Example |
|---|---|---|---|
| `age` | int | 18–70 | `20` |
| `gender` | enum | `male` \| `female` \| `other` | `"male"` |
| `marital_status` | enum | `single` \| `married` \| `divorced` \| `widowed` | `"single"` |
| `nationality` | string | default `"Sri Lankan"` | |
| `country` | string | default `"Sri Lanka"` | |
| `district`, `city` | string? | optional | |
| `num_dependents` | int | 0–20 | `0` |

### Occupation & Financial

| Field | Type | Constraints |
|---|---|---|
| `occupation` | string | — |
| `employment_type` | enum | `permanent` \| `contract` \| `freelance` \| `self_employed` \| `unemployed` \| `retired` |
| `designation` | string? | optional |
| `hazardous_level` | enum | `none` \| `low` \| `medium` \| `high` |
| `hazardous_activities` | string? | optional |
| `monthly_income_lkr` | float | > 0 |
| `has_existing_insurance` | bool | — |
| `current_insurance_status` | enum | `none` \| `has_insurance` |
| `employer_insurance_scheme` | string? | optional |

### Goals

| Field | Type |
|---|---|
| `primary_goal` | `cheap_and_quick` \| `protection` \| `savings_and_investment` \| `health_coverage` \| `retirement` \| `none` |
| `secondary_goal` | same enum, optional |
| `travel_history_high_risk`, `dual_citizenship`, `tax_regulatory_flags`, `insurance_history_issues` | bool |

### Health (10 booleans)

`has_chronic_disease`, `has_cardiovascular`, `has_cancer`, `has_respiratory`, `has_neurological`, `has_gastrointestinal`, `has_musculoskeletal`, `has_infectious_sexual`, `recent_treatment_surgery`, `covid_related`.

### Lifestyle

| Field | Type | Constraints |
|---|---|---|
| `bmi` | float | > 0, ≤ 60 |
| `is_smoker`, `is_alcohol_consumer` | bool | — |

---

## Sample API Calls

### Upload a policy (admin)

```bash
TOKEN="<admin Supabase access_token>"

curl -X POST "http://localhost:8000/api/ingest" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./data/raw/aia_term_life.pdf" \
  -F "policy_type=term_life" \
  -F "company=AIA Sri Lanka" \
  -F "premium_level=0" \
  -F "covers_life=true" \
  -F "is_entry_level=true"
```

### Upload riders bundle (admin)

```bash
curl -X POST "http://localhost:8000/api/riders" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./data/raw/all_riders.pdf"
```

### Get a recommendation (any signed-in user)

```bash
USER_TOKEN="<user Supabase access_token>"

curl -X POST "http://localhost:8000/api/recommend" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d @profile.json
```

### Chat follow-up

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "message": "What are the key exclusions I should be aware of?"
  }'
```

---

## SHAP Explainability

The system uses **SHAP TreeExplainer** on the XGBoost model to explain every recommendation.

### How it works
1. The XGBoost model is trained on **8,000 synthetic user-policy pairs** generated by `_domain_score()` — a deterministic function of insurance domain rules.
2. At inference time, SHAP decomposes the model's prediction into per-feature contributions.
3. Each feature's SHAP value is mapped through `FEATURE_DESCRIPTIONS` and `REASON_TEMPLATES` (see `app/core/recommendation/explainer.py`) into a human-readable `SHAPFactor`.
4. The top positive and negative factors are surfaced separately.

### Sample output

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

Negative factors:
  - Contract employment status              -0.052 impact
    → Contract/freelance status may require more flexible premium terms.
```

### Feature reference

**User features (21):** `age_norm`, `gender_male`, `is_married`, `num_dependents_norm`, `permanent_employment`, `hazardous_level_norm`, `income_norm`, `goal_cheap_quick`, `goal_protection`, `goal_savings`, `goal_health`, `goal_retirement`, `has_any_condition`, `has_chronic`, `has_cardiovascular`, `has_cancer`, `has_respiratory`, `has_neurological`, `bmi_norm`, `is_smoker`, `is_alcohol`.

**Policy features (10):** `pol_term_life`, `pol_whole_life`, `pol_endowment`, `pol_health`, `pol_critical_illness`, `pol_accident`, `pol_min_age_norm`, `pol_max_age_norm`, `pol_premium_level_norm`, `pol_covers_health`.

**Important:** Feature order is fixed in `scorer.py`. Reordering or adding features requires deleting `saved_models/xgb_scorer.pkl` to trigger retraining.

---

## Chunking Strategy

Recursive semantic chunking with insurance-aware boundaries:

```
Priority of split boundaries:
  1. Multi-blank lines    (\n\n\n)   → Major section breaks
  2. Double newlines      (\n\n)     → Paragraph breaks
  3. Single newlines      (\n)
  4. Sentence boundaries  (". ")
  5. Clause boundaries    (", ")
  6. Word boundaries      (" ")
  7. Character level      ("")       → Fallback

chunk_size    = 1000 characters
chunk_overlap = 200 characters
```

Each chunk is tagged with a `section` label by keyword matching:

| Section | Detected by |
|---|---|
| `benefits` | "benefit", "coverage", "covers", "insured amount" |
| `exclusions` | "exclusion", "not covered", "does not cover" |
| `premium` | "premium", "payment", "contribution" |
| `eligibility` | "eligibility", "age limit", "entry age" |
| `riders` | "rider", "additional benefit", "optional" |
| `claims` | "claim", "procedure", "how to claim" |
| `definitions` | "definition", "means", "shall mean" |
| `general` | everything else |

For the riders bundle, a separate chunker (`chunk_rider_document`) tags chunks with `doc_type="rider"`, `rider_codes` (CSV of all codes in the doc), and a sentinel `policy_name="__riders_bundle__"` so existing retrieval code that reads `policy_name` does not crash.

---

## Environment Variables

### Backend (`InsureMatch-backend/.env`)

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model ID |
| `SUPABASE_URL` | *(required)* | `https://YOUR_PROJECT.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | *(required)* | Service-role key (bypasses RLS) — **never expose** |
| `SUPABASE_JWT_SECRET` | *(required)* | HS256 verification secret (Supabase legacy) |
| `CHROMA_PERSIST_DIR` | `./vectordb` | ChromaDB storage |
| `CHROMA_COLLECTION_NAME` | `insurance_policies` | Collection name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding |
| `CHUNK_SIZE` | `1000` | Max chars per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `RETRIEVAL_K` | `8` | Chunks returned per query |
| `RETRIEVAL_FETCH_K` | `20` | MMR fetch pool size |
| `MODEL_SAVE_DIR` | `./saved_models` | XGBoost + registries directory |
| `DATA_DIR` | `./data/raw` | Raw documents directory |

### Frontend (`InsureMatch-frontend/.env.local`)

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend base URL |
| `NEXT_PUBLIC_SUPABASE_URL` | *(required)* | Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | *(required)* | Supabase anon key (safe to expose with RLS) |

---

## Notes & Production Considerations

- **First run:** XGBoost training takes ~10–20 s. Saved and reused on subsequent runs.
- **Embeddings:** Downloaded once on first use (~90 MB), cached by HuggingFace.
- **Chat sessions:** Persisted to Supabase — no server-side memory state, sessions survive backend restarts.
- **Policy / rider metadata:** If you don't supply metadata at upload time, the LLM auto-extracts it (slower and slightly less reliable than supplying it directly).
- **Riders are catalog-replace by default:** Re-uploading the bundle wipes the old catalog. To delete a single rider, use `DELETE /api/riders/{rider_code}`.
- **Groq rate limits:** Free tier is generous for LLaMA 3.3 70B in development. Production should plan for higher tiers and add request retries.
- **Service-role key:** `SUPABASE_SERVICE_ROLE_KEY` lives only on the backend. It bypasses RLS — never ship it to the browser. Ownership is enforced in `supabase_client.py` by always filtering on `user_id`.
- **Type source of truth:** `app/models/schemas.py` is canonical. `InsureMatch-frontend/src/lib/types/api.ts` mirrors it; keep them in sync.
- **CORS:** `main.py` restricts origins to `http://localhost:3000`, `http://127.0.0.1:3000`, and one LAN IP. Update for production deployments.
- **JWT algs:** `deps.py` supports HS256 (legacy `SUPABASE_JWT_SECRET`) and the new asymmetric ES256 / RS256 via JWKS — no code change needed when Supabase rotates keys.
- **Frontend framework:** Next.js 16.2 has breaking changes from earlier majors. Consult `node_modules/next/dist/docs/` before touching framework APIs (per `InsureMatch-frontend/AGENTS.md`).
- **Tests:** None yet. When adding them, use `pytest` for backend (`tests/`) and Vitest/Jest for frontend.
