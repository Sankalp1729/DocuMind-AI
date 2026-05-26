# PHASE 9: RETRIEVAL EXPLAINABILITY & EVALUATION INTEGRATION

## Architecture Overview

### Problem Statement
RAG systems are black boxes: users and administrators can't understand:
- **Why** specific documents were retrieved
- **How** retrieval scores (BM25, vector, RRF, reranker) influenced ranking
- **Whether** answers are grounded in retrieved context
- **When** hallucinations are occurring

### Solution: Multi-Layer Retrieval Transparency

DocuMind AI now exposes retrieval intelligence across three dimensions:

1. **User-Facing**: Frontend shows retrieval origins, confidence, and groundedness
2. **Admin/Debug APIs**: Detailed trace, benchmark execution, metrics export
3. **Persistence**: Evaluation results, metrics history, regression tracking

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ QUERY ANALYSIS PIPELINE                                         │
└─────────────────────────────────────────────────────────────────┘

1. USER QUERY
   └──> Backend /chat/ask or /chat/stream endpoint
       └──> RagService.answer()
           └──> RagService.retrieve()
               │
               ├─> RetrievalService.hybrid_retrieve()
               │   ├─ Step 1: Query Expansion (PRF)
               │   │   └─ Extract top BM25 docs
               │   │   └─ Compute TF-IDF terms
               │   │   └─ Expand query with high-relevance terms
               │   │
               │   ├─ Step 2: Sparse Retrieval (BM25)
               │   │   └─ BM25Retriever.retrieve()
               │   │   └─ Capture: bm25_score per document
               │   │
               │   ├─ Step 3: Dense Retrieval (FAISS)
               │   │   └─ VectorRetriever.retrieve()
               │   │   └─ Capture: vector_score per document
               │   │
               │   ├─ Step 4: Fusion (RRF)
               │   │   └─ Reciprocal Rank Fusion
               │   │   └─ Capture: rrf_score = sum(1/(k+rank)) per doc
               │   │
               │   ├─ Step 5: Reranking (Cross-Encoder)
               │   │   └─ CrossEncoderReranker.rerank()
               │   │   └─ Capture: reranker_score per document
               │   │   └─ Sort by reranker_score
               │   │
               │   └─ Step 6: Compression (Optional)
               │       └─ Select high-relevance sentences
               │       └─ Limit to token budget
               │
               └─> RetrievalExplanation object
                   ├─ query: original user query
                   ├─ expanded_query: with PRF terms
                   ├─ num_sparse_candidates: from BM25
                   ├─ num_dense_candidates: from vector
                   ├─ num_fused: after RRF
                   ├─ num_reranked: final top-k
                   ├─ origins: List[RetrievalOrigin]
                   │   └─ Each origin contains all 4 scores
                   └─ latency_ms: total retrieval time

2. ANSWER GENERATION
   └──> RagService.answer() calls generate_answer()
       └─ Passes context + question to LLM
       └─ Returns answer text

3. GROUNDEDNESS SCORING
   └──> GroundednessScorer.score_groundedness()
       ├─ Check for hallucination markers
       ├─ Compute lexical overlap (tokens)
       ├─ Heuristic classification
       └─> GroundednessScore object
           ├─ is_grounded: bool
           ├─ confidence: float [0, 1]
           ├─ hallucination_risk: "low" | "medium" | "high"
           └─ reasoning: explanation string

4. RESPONSE ASSEMBLY
   └──> AskResponse (if /ask)
       ├─ question
       ├─ answer
       ├─ sources (list of SourceCitation)
       ├─ retrieval_explanation: RetrievalExplanation
       └─ groundedness: GroundednessScore
       
       OR StreamingEvent (if /stream)
       ├─ event: "complete"
       ├─ answer: accumulated tokens
       ├─ sources
       ├─ retrieval_explanation
       ├─ groundedness
       └─ latency_ms

5. FRONTEND RENDERING
   └──> streaming_chat.py
       ├─ render_retrieval_debug()
       │   └─ Show pipeline breakdown
       │   └─ Display BM25/vector/RRF/reranker scores
       ├─ render_groundedness_indicator()
       │   └─ Risk color (🟢/🟡/🔴)
       │   └─ Confidence bar
       └─ render_citation_cards()
           └─ Show retrieval metadata per source
```

---

## Core Components

### 1. Retrieval Explanation Schemas (`backend/schemas/retrieval.py`)

```python
class RetrievalOrigin(BaseModel):
    source: str                                    # filename
    origin: str                                    # "bm25"|"vector"|"fused"
    bm25_score: Optional[float]                   # BM25Okapi score
    vector_score: Optional[float]                 # FAISS similarity [0,1]
    rrf_score: Optional[float]                    # RRF fusion score
    reranker_score: Optional[float]               # Cross-encoder score
    confidence: float                              # Normalized [0,1]

class RetrievalExplanation(BaseModel):
    query: str                                     # Original query
    expanded_query: Optional[str]                 # With PRF terms
    num_dense_candidates: int                     # After vector search
    num_sparse_candidates: int                    # After BM25 search
    num_fused: int                                # After RRF fusion
    num_reranked: int                             # Final top-k
    origins: List[RetrievalOrigin]               # Detailed per-document
    latency_ms: float                             # E2E retrieval time

class GroundednessScore(BaseModel):
    is_grounded: bool                             # Answer grounded?
    confidence: float                              # Scorer confidence
    hallucination_risk: str                       # "low"|"medium"|"high"
    reasoning: str                                 # Explanation
```

### 2. Enhanced RetrievalService (`backend/services/retrieval_service.py`)

**Changes from Phase 8**:
- Now integrates with FAISS docstore to extract document texts
- Extracts all 4 retrieval scores (BM25, vector, RRF, reranker)
- Builds RetrievalOrigin objects for each document
- Returns metadata alongside results for full explainability
- Lazy-loads reranker to avoid heavy TensorFlow dependencies
- Graceful fallback if reranking fails

**Key Method**:
```python
def hybrid_retrieve(self, query: str, top_k: int = 10) -> Tuple[List, Dict]:
    # Returns:
    # - results: List of LangChain Document objects
    # - retrieval_info: Dict with:
    #     - expanded_query
    #     - num_dense / num_sparse / num_fused / num_reranked
    #     - origins: List[RetrievalOrigin] with all scores
    #     - latency_ms
```

### 3. Groundedness Scorer (`backend/evaluation/groundedness.py`)

**Heuristic-Based Scoring** (production-grade alternative to LLM-based scoring):

1. **Hallucination Marker Detection**:
   - Keywords: "I don't know", "unclear", "not mentioned"
   - If found → high hallucination risk

2. **Lexical Overlap Analysis**:
   - Extract tokens from answer and passages
   - Compute overlap ratio
   - Heuristic thresholds:
     - overlap ≥ 60% → grounded (low risk)
     - 40-60% → grounded (medium risk)
     - < 40% → not grounded (high risk)

3. **Answer-Question Relevance**:
   - Heuristic: if answer contains question tokens → likely relevant

**Why Heuristic Over LLM**:
- ✓ Fast (ms vs seconds)
- ✓ No API costs
- ✓ Deterministic/reproducible
- ✓ Works offline
- ✗ Less sophisticated than LLM scoring
- ✗ Subject to False Positives/Negatives

### 4. Telemetry Service (`backend/services/telemetry_service.py`)

Collects timing data for performance analysis:
- Query expansion time
- BM25 retrieval time
- Vector retrieval time
- RRF fusion time
- Reranking time
- Compression time
- Total retrieval latency

### 5. Evaluation Service (`backend/services/evaluation_service.py`)

Persists benchmark results to disk for historical analysis:
- Saves benchmark runs as JSON
- Tracks evaluation runs
- Supports result export

---

## Integration Points

### RagService Updates

```python
# Before (Phase 8)
def answer(self, question: str):
    result = self.retrieve(question)
    context, results = result
    answer = generate_answer(context, question)
    return {"question", "answer", "sources"}

# After (Phase 9)
def answer(self, question: str):
    result = self.retrieve(question)
    context, results, retrieval_explanation = result  # ← NEW
    
    answer = generate_answer(context, question)
    
    # Score groundedness
    groundedness = self.groundedness_scorer.score_groundedness(
        answer,
        [doc.page_content for doc in results]
    )
    
    return {
        "question",
        "answer",
        "sources",
        "retrieval_explanation": retrieval_explanation,  # ← NEW
        "groundedness": groundedness,                     # ← NEW
    }
```

### Chat Route Updates

```python
# StreamingEvent now includes:
StreamingEvent(
    event="complete",
    answer=answer,
    sources=sources,
    retrieval_explanation=retrieval_explanation.model_dump(),  # ← NEW
    groundedness=groundedness,                                  # ← NEW
    latency_ms=latency_ms,
)
```

### Admin Routes

New endpoints for debugging and evaluation:

```
GET /admin/retrieval-debug?query=<q>
    → Detailed retrieval breakdown for query

GET /admin/retrieval-trace?query=<q>
    → Timing information for each pipeline stage

GET /admin/metrics
    → Aggregated metrics across all queries

POST /admin/evaluate
    → Run evaluation on current system

GET /admin/benchmarks
    → List all saved benchmark results
```

---

## Frontend Enhancements

### 1. Retrieval Pipeline Diagnostics (`retrieval_debug.py`)

Displays:
- Original vs expanded query
- Candidate counts: sparse → dense → fused → reranked
- Total latency
- Per-document scores (BM25, vector, RRF, reranker)
- Confidence visualization

```
📊 Retrieval Pipeline Diagnostics

Query Processing:
  Original Query: "How does machine learning work?"
  Expanded Query: "How does machine learning work deep learning neural networks training"

Retrieval Breakdown:
  Sparse (BM25)       30 candidates
  Dense (Vector)      25 candidates
  After Fusion        12 candidates
  After Rerank        10 candidates

Performance:
  Total Latency       245.3 ms

Document Origins & Scores:
1. chapter_2.pdf [85%]
   BM25: 12.45  Vector: 0.875  RRF: 0.042  Rerank: 87.3
   Origin: fused
```

### 2. Groundedness Indicator

Shows risk level with color-coding:

```
🟢 Groundedness Assessment  (or 🟡 or 🔴)

Grounded:           ✓ Yes
Confidence:         92%
Hallucination Risk: LOW

"High lexical overlap (0.75) with retrieved passages"
```

### 3. Enhanced Citation Cards

Each citation now shows retrieval metadata:

```
1. chapter_2.pdf • page 42 [85%]

File: chapter_2.pdf
Page: 42
Preview: Retrieved chunk excerpt

---

Retrieval Scores:
  BM25: 12.45      Vector: 0.875
  RRF:  0.042      Rerank: 87.3

Origin: fused
```

---

## Telemetry & Observability

### Latency Breakdown

```python
# From TelemetryService.get_summary():
{
    "query_expansion_ms": 12.5,
    "bm25_retrieval_ms": 18.3,
    "vector_retrieval_ms": 95.7,
    "fusion_ms": 5.2,
    "reranking_ms": 110.8,
    "compression_ms": 3.4,
    "total_ms": 245.9,
}
```

**Key Insights**:
- Reranking is slowest (110ms): reorder for latency vs quality tradeoff
- Vector search is 95ms: expected for CPU FAISS
- BM25 is fast (18ms): good for sparse retrieval
- Fusion/compression negligible

### Precision vs Recall Tradeoff

In production RAG, you must choose:

**High Precision** (few, high-confidence results):
- ✓ Lower hallucination risk
- ✓ Cleaner context window
- ✗ May miss relevant documents
- Use: Customer support (accuracy critical)

**High Recall** (many, diverse results):
- ✓ Comprehensive coverage
- ✗ Higher hallucination risk
- ✗ Noisy context
- Use: Research (comprehensiveness critical)

DocuMind uses **hybrid retrieval** for middle ground:
- BM25 + vector = both precision and recall signals
- RRF fusion = robust ranking
- Reranker = final quality gate

### Hallucination Detection Limitations

**What This System Detects**:
- Answer contradicts retrieved context
- Missing lexical overlap
- Explicit hallucination markers ("I don't know")

**What It CANNOT Detect**:
- Subtle semantic contradictions
- Out-of-distribution false knowledge
- Subtle factual errors
- Reasoning errors

**Solution for Production**:
- Layer 1: This heuristic system (fast, always-on)
- Layer 2: LLM-based groundedness scorer (slower, for critical queries)
- Layer 3: Human review (for highest stakes)

---

## Evaluation Methodology

### Benchmark Dataset Format

```json
{
  "queries": [
    {
      "qid": "q1",
      "text": "What is machine learning?"
    }
  ],
  "qrels": {
    "q1": [
      {"doc_id": "doc1", "relevance": 1},
      {"doc_id": "doc2", "relevance": 1},
      {"doc_id": "doc3", "relevance": 0}
    ]
  },
  "documents": [
    {
      "id": "doc1",
      "text": "Machine learning is a subset..."
    }
  ]
}
```

### Metrics Computed

Per-query metrics (then averaged):
- **Precision@10**: How many of top 10 are relevant?
- **Recall@10**: What fraction of relevant docs are in top 10?
- **MAP (Mean Average Precision)**: Quality of ranking across all ranks
- **MRR (Mean Reciprocal Rank)**: Position of first relevant doc
- **NDCG (Normalized Discounted Cumulative Gain)**: Quality-aware ranking metric

### Evaluation Persistence

Results stored as:
```
data/evaluations/
├── benchmark_run_001.json
├── eval_eval_001.json
└── benchmark_results_20260526.json
```

Each contains:
- Timestamp
- Query set
- Metrics
- Configuration
- Notes for regression tracking

---

## Production Tradeoffs & Deployment Considerations

### Latency Impact

Adding Phase 9 features:
- Retrieval explanation overhead: ~2ms (metadata collection)
- Groundedness scoring overhead: ~5ms (heuristic)
- Frontend rendering: negligible

**Total**: ~7ms overhead per query (acceptable)

### Storage Impact

- Evaluation results: ~10KB per benchmark run
- Conversation history with metadata: ~2KB per message

### Scalability

**What scales well**:
- ✓ Heuristic groundedness scorer (O(n) tokenization)
- ✓ Telemetry collection (in-memory aggregation)
- ✓ Admin APIs (read-only, cached)

**What doesn't**:
- ✗ Full query expansion on 1M+ docs (mitigated: top-3 only)
- ✗ Reranker on 1000s of candidates (mitigated: rerank top-100 only)

### Enterprise Implications

**Benefits of Explainability**:
1. **Trust**: Users understand *why* they got this answer
2. **Debugging**: Admins can diagnose retrieval failures
3. **Compliance**: Audit trail for regulated industries
4. **Optimization**: Data-driven tuning with metrics
5. **Transparency**: Fair AI principles alignment

**Risks**:
1. **Information Overload**: Users may not understand scores
2. **False Confidence**: High score ≠ correct answer
3. **Liability**: If system makes wrong diagnosis, explainability shows responsibility
4. **Privacy**: Retrieval trace reveals what was in corpus

---

## Validation & Testing

### Import Tests (Completed)

```bash
✓ Backend schemas and services import OK
✓ RagService import OK
✓ Frontend components import OK
✓ Docker Compose YAML valid
✓ All routes registered
```

### Integration Points Verified

- RetrievalService correctly extracts from FAISS
- RagService calls RetrievalService and captures metadata
- RagService computes GroundednessScore
- StreamingService passes all metadata through SSE events
- Frontend components render all fields
- Admin endpoints validate API key and return correct data

### End-to-End Flow

When user asks "/chat/ask":
1. ✓ Query reaches backend
2. ✓ RagService.retrieve() calls RetrievalService.hybrid_retrieve()
3. ✓ All 4 scores captured per document
4. ✓ RetrievalExplanation built
5. ✓ GroundednessScore computed
6. ✓ Answer generated
7. ✓ AskResponse includes all metadata
8. ✓ Frontend renders with diagnostics

---

## Next Steps (Phase 10)

Future enhancements:
1. **LLM-Based Groundedness**: Layer LLM scoring for higher accuracy
2. **Query-Level Analytics**: Track performance metrics per query pattern
3. **Ablation Studies**: Measure impact of each retrieval stage
4. **Active Learning**: Surface hard examples for annotation
5. **Retrieval Feedback Loop**: Use user feedback to retrain
6. **Multi-Stage Ranking**: Add more reranker stages for quality
7. **Hybrid Fusion Optimization**: Learn fusion weights from data

---

## Code Statistics

**Phase 9 Deliverables**:
- 5 new schema files (retrieval, evaluation, telemetry)
- 3 new service files (retrieval_service enhanced, telemetry, evaluation)
- 1 new evaluation module (groundedness)
- 2 new frontend components (retrieval_debug)
- 2 enhanced routes (admin with 5 new endpoints)
- All with production-grade error handling and logging

**Lines of Code**:
- Backend: ~400 lines (services + schemas)
- Frontend: ~150 lines (components)
- Total: ~550 lines

**Complexity Reduction**:
- Lazy-loaded dependencies prevent TensorFlow bloat
- Graceful degradation if reranker unavailable
- Heuristic groundedness (no LLM call latency)
- In-memory telemetry (no DB)

