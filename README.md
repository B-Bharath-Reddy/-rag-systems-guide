# RAG: A Practical Guide to Building, Evaluating, and Operating Retrieval-Augmented Generation Systems

This repository contains five module PDFs, supporting lab notebooks, and utility scripts about Retrieval-Augmented Generation (RAG). This README turns those materials into one practical reference so someone new to the topic can understand:

- what RAG is
- when to use it
- how to build it
- how to evaluate it
- which techniques matter in production
- what a mature, production-minded RAG stack looks like

The write-up is based primarily on the five learning modules in this repo, their extracted text notes, the accompanying hands-on labs, and the local helper code used to demonstrate retrieval, generation, evaluation, and observability.

It also aligns with a few official references that reflect common production patterns:

- OpenAI file search docs: <https://developers.openai.com/api/docs/guides/tools-file-search>
- OpenAI evals docs: <https://developers.openai.com/api/docs/guides/evals>
- Weaviate hybrid search docs: <https://docs.weaviate.io/weaviate/search/hybrid>
- Arize Phoenix docs: <https://arize.com/docs/phoenix>

For a shorter companion reference, see [docs/rag-techniques-guide.md](docs/rag-techniques-guide.md).

## Table of Contents

1. [What Is RAG?](#what-is-rag)
2. [When To Use RAG](#when-to-use-rag)
3. [Core RAG Architecture](#core-rag-architecture)
4. [How To Build a RAG System](#how-to-build-a-rag-system)
5. [Retrieval Foundations](#retrieval-foundations)
6. [Production Retrieval Techniques](#production-retrieval-techniques)
7. [LLMs, Prompting, and Grounding](#llms-prompting-and-grounding)
8. [Evaluation, Monitoring, and Observability](#evaluation-monitoring-and-observability)
9. [Cost, Latency, Security, and Multimodal RAG](#cost-latency-security-and-multimodal-rag)
10. [What Mature Teams Usually Do](#what-mature-teams-usually-do)
11. [Production Patterns Seen in Major Platforms](#production-patterns-seen-in-major-platforms)
12. [Production Examples and Flows](#production-examples-and-flows)
13. [GraphRAG](#graphrag)
14. [From RAG to Agents](#from-rag-to-agents)
15. [Key Takeaways](#key-takeaways)

## What Is RAG?

Retrieval-Augmented Generation (RAG) pairs a language model with a knowledge base so the model can answer with information that is:

- private
- recent
- domain-specific
- verifiable

A plain LLM answers from its training data and prompt only. A RAG system first retrieves relevant information from a data source, then injects that information into the prompt before generation.

In simple terms:

```text
Question -> Retrieve relevant knowledge -> Build grounded prompt -> Generate answer
```

That is why RAG is useful for systems such as:

- internal company assistants
- support chatbots
- codebase assistants
- legal or medical document Q&A
- search experiences with summarized answers
- PDF, slide, and multimodal knowledge assistants

## When To Use RAG

RAG is the right default when the model needs access to knowledge that cannot be trusted to live inside the model itself.

Use RAG when:

- the information changes frequently
- the information is private or proprietary
- the domain is narrow and specialized
- users need citations or source-backed answers
- the corpus is too large to paste into prompts manually
- you want to update knowledge without retraining the model

Do not reach for RAG first when:

- the task is pure transformation of user-provided text
- the answer is already stable and generic
- a deterministic database lookup is enough
- the real problem is behavior or style control rather than knowledge access

### RAG vs Fine-Tuning vs Plain Prompting

| Need | Best Fit |
| --- | --- |
| Inject new facts, recent facts, private facts | RAG |
| Change task behavior, tone, or domain style | Fine-tuning |
| Solve simple prompt-only tasks | Plain prompting |
| Need both current facts and domain behavior | Fine-tuning + RAG |

The course material makes an important distinction: fine-tuning is mainly for domain adaptation and task behavior, while RAG is mainly for knowledge injection.

## Core RAG Architecture

At a high level, a good RAG system has these stages:

```text
User Query
  -> Query Processing
  -> Retrieval
  -> Reranking
  -> Prompt Construction
  -> LLM Generation
  -> Citations / Structured Output
  -> Logging, Evaluation, Feedback
```

A slightly richer production view looks like this:

```text
Documents
  -> parsing / extraction
  -> chunking
  -> metadata enrichment
  -> embeddings
  -> vector + lexical indexes

User query
  -> route / classify / rewrite if needed
  -> hybrid search
  -> metadata filtering
  -> reranking
  -> grounded prompt
  -> LLM answer
  -> citations
  -> traces, evals, and feedback loop
```

### Main Components

#### 1. Knowledge base

The source of truth. It may contain:

- documents
- FAQs
- code
- tickets
- policies
- PDFs
- slides
- images
- tabular or relational data

#### 2. Retriever

The retriever is responsible for finding the best supporting context. In mature systems this is rarely just one search method. It is usually some combination of:

- metadata filtering
- BM25 or other lexical search
- embedding-based semantic search
- hybrid fusion
- reranking

#### 3. LLM

The language model turns the retrieved context into a response. Its job is not to know everything. Its job is to:

- read the context well
- ignore irrelevant noise
- answer clearly
- cite sources when required
- avoid inventing unsupported claims

#### 4. Evaluation and observability

This is where most toy RAG systems fail. Production RAG requires:

- retriever evaluation
- LLM evaluation
- system metrics
- traces
- experiments
- human feedback

## How To Build a RAG System

This is the practical build sequence implied across the PDFs and labs in this repo.

### 1. Define the use case and the success criteria

Before touching embeddings or vector databases, decide:

- who the users are
- what questions they ask
- what sources are trusted
- whether citations are required
- what latency is acceptable
- what failures are unacceptable

Example success criteria:

- answer relevance
- groundedness or faithfulness
- citation accuracy
- p95 latency
- cost per query
- deflection rate for support use cases

### 2. Collect and normalize the knowledge

Ingestion is not glamorous, but it determines the ceiling of the entire system.

Tasks usually include:

- extracting text from PDFs or slides
- parsing HTML, Markdown, JSON, or code
- preserving titles, authors, dates, file names, section headers, and permissions
- cleaning duplicated boilerplate
- attaching metadata needed for search and access control

This repo includes two extraction scripts:

- `extract_pdfs.py` using `PyPDF2`
- `extract_pdfs_pymupdf.py` using `PyMuPDF`

Those scripts generate the `RAG_M*_content.txt` files from the module PDFs.

### 3. Chunk the documents

Chunking is one of the highest-leverage decisions in RAG.

Why chunk at all:

- whole documents are too coarse for retrieval
- whole documents fill the LLM context window too quickly
- smaller chunks improve topical relevance

The course material covers these chunking strategies:

- fixed-size chunking
- fixed-size chunking with overlap
- recursive character splitting
- semantic chunking
- language-model-based chunking
- context-aware chunking

Good default:

- start with fixed-size or recursive chunking
- preserve metadata
- add overlap
- evaluate before moving to more complex methods

Useful rule of thumb:

- if chunks are too large, you lose precision
- if chunks are too small, you lose context
- overlap helps recover cut-off context

### 4. Create embeddings and indexes

Once chunks exist, turn them into vectors and store them in a retrieval system.

Typical assets created at this step:

- dense embeddings for semantic search
- lexical index for BM25 or keyword retrieval
- metadata fields for filtering
- vector index such as HNSW for ANN retrieval

### 5. Build retrieval

A practical retrieval pipeline usually looks like this:

```text
query
  -> optional query rewrite / classification
  -> metadata filters
  -> semantic search
  -> keyword search
  -> hybrid fusion
  -> rerank top candidates
  -> select final top_k chunks
```

The strongest systems usually do not rely on a single search method.

### 6. Build the augmented prompt

Prompt structure matters. The course materials repeatedly push toward prompt templates that contain:

- system instructions
- conversation history if needed
- retrieved documents
- source markers
- the current user question

A simple grounded template:

```text
System:
You are a domain assistant. Use only the retrieved documents.
If the answer is not supported, say you do not know.
Cite sources as [DOC n].

Retrieved Documents:
[DOC 1] ...
[DOC 2] ...

User Question:
...
```

### 7. Generate the answer

For factual RAG, generation settings should be conservative. The module on LLMs recommends lower randomness for code and factual work.

Good defaults for factual RAG:

- lower temperature
- controlled `top_p`
- short, focused responses unless detail is required
- explicit grounding instructions

### 8. Add citations and guardrails

Citations improve trust and debugging, but citations can also be hallucinated. So:

- ask for citations
- keep source markers in context
- evaluate citation quality
- do not assume citations are correct just because they look formal

### 9. Evaluate offline

Before production, build an evaluation set and measure both retrieval and generation quality.

### 10. Instrument and observe online

In production, capture:

- prompts
- rewritten queries
- retrieved chunks
- reranked order
- final prompt
- output
- latency and token usage
- user feedback

### 11. Iterate

The repo materials treat RAG as an optimization loop, not a one-time build:

```text
Observe traffic -> Evaluate failures -> Experiment -> Measure -> Keep or revert
```

## Retrieval Foundations

Module 2 and the corresponding labs cover the retrieval basics that every RAG builder should understand.

### Metadata filtering

Metadata filtering is not search by meaning. It is a way to narrow candidate documents using rigid criteria such as:

- date
- author
- region
- tenant
- subscription level
- document type

Strengths:

- simple
- fast
- easy to debug
- essential for permissions and narrowing search

Limitations:

- not a ranking system
- ignores document content
- not useful by itself

### Keyword search

Keyword search is still fundamental in modern RAG.

#### TF-IDF

TF-IDF scores terms by:

- how often they appear in a document
- how rare they are across the corpus

It is simple and interpretable, but BM25 is generally the stronger production default.

#### BM25

The course materials emphasize BM25 as the practical standard lexical algorithm in modern retrievers because it improves on TF-IDF with:

- term frequency saturation
- document length normalization

Why BM25 matters in RAG:

- catches exact terms, codes, names, and identifiers
- complements semantic retrieval
- is cheap and mature

### Semantic search

Semantic retrieval uses embeddings instead of term counts. Documents and queries are mapped to vectors, then compared by similarity.

Key distance functions covered in the course:

- cosine similarity
- Euclidean distance
- dot product

Why semantic search matters:

- handles paraphrases and synonyms
- retrieves by meaning, not exact wording

Where it struggles:

- may miss exact identifiers
- may overgeneralize semantically related but incorrect matches

### Hybrid search

Hybrid search combines lexical and semantic retrieval, often alongside metadata filtering.

This is one of the most important ideas in the entire course. High-performing retrievers usually balance:

- keyword sensitivity
- semantic flexibility
- metadata constraints

The course introduces rank fusion ideas such as Reciprocal Rank Fusion (RRF), while tools like Weaviate expose a weighting parameter for lexical vs vector balance. In Weaviate, the `alpha` parameter controls that balance:

- `alpha = 1.0` means pure vector search
- `alpha = 0.0` means pure keyword search

In practice:

- increase vector weight when paraphrase matching matters more
- increase lexical weight when exact terms matter more

### Retrieval metrics

To evaluate a retriever, you need:

- a query
- a ranked result list
- ground-truth relevance labels

Metrics covered in the course and labs:

#### Precision@K

How many returned documents are relevant?

```text
Precision@K = relevant retrieved in top K / K
```

#### Recall@K

How many relevant documents did the system recover?

```text
Recall@K = relevant retrieved in top K / total relevant
```

#### MAP@K

Mean Average Precision rewards systems that rank relevant results higher, not just somewhere in the top K.

#### MRR

Mean Reciprocal Rank focuses on how early the first relevant result appears.

These metrics matter because a weak retriever caps the quality of the whole system.

## Production Retrieval Techniques

Module 3 is where the course moves from theory to practical retrieval engineering.

### Approximate nearest neighbors (ANN)

Brute-force KNN becomes too expensive at scale because every query compares against every vector.

ANN solves that by trading a little exactness for a large speed gain.

Main idea:

- exact KNN is accurate but slow
- ANN is much faster and usually good enough

### HNSW

The course highlights HNSW (Hierarchical Navigable Small World) as a core indexing technique behind vector databases.

Why HNSW matters:

- much faster than brute-force search
- scales to very large corpora
- good quality-speed tradeoff

### Vector databases

A vector database is more than vector storage. It usually provides:

- vector indexing
- semantic search
- lexical or hybrid search
- filtering
- sometimes reranking and multimodal support

The labs in this repo use Weaviate for:

- collection creation
- vectorizer configuration
- semantic search
- BM25
- hybrid search
- filters
- reranking

### Chunking strategies in production

The course does not treat chunking as a single fixed recipe. It treats it as an experimental space.

#### Fixed-size chunking

Pros:

- simple
- fast
- easy to implement

Cons:

- ignores natural document boundaries

#### Overlapping chunking

Pros:

- reduces context loss at chunk boundaries
- often improves answer quality

Cons:

- increases storage and retrieval cost

#### Recursive splitting

Split on increasingly smaller separators while trying to respect structure such as:

- paragraphs
- section markers
- headers
- language-specific boundaries

#### Semantic chunking

Group sentences by meaning rather than by a raw size limit.

Pros:

- smarter boundaries
- often better precision and recall

Cons:

- repeated vector work
- more preprocessing cost

#### LLM-based chunking

Ask an LLM to separate concepts or topics.

Pros:

- can align better with meaning than fixed heuristics

Cons:

- extra cost
- more pipeline complexity

#### Context-aware chunking

Add short contextual summaries or labels to chunks so retrieval works better even when the chunk text alone is ambiguous.

This is one of the most practical "first improvements" after getting a baseline system working.

### Query parsing and query improvement

The course treats query processing as a real performance lever.

#### Query rewriting

Use an LLM to rewrite messy user prompts into better retrieval queries by:

- clarifying ambiguity
- removing irrelevant chatter
- adding synonyms
- using domain terminology

This is especially useful in domains like healthcare, support, or enterprise search where users ask vague questions.

#### Named Entity Recognition (NER)

NER lets the system extract key entities such as:

- people
- organizations
- dates
- locations
- titles

That can drive:

- targeted filters
- better routing
- entity-aware search

#### HyDE

HyDE stands for Hypothetical Document Embeddings.

Workflow:

1. Generate a hypothetical ideal answer document for the query.
2. Embed that hypothetical document.
3. Search using that embedding.

Why it helps:

- the retriever matches documents to a document-like representation
- sometimes improves recall for difficult or underspecified queries

Tradeoff:

- extra LLM latency and cost

### Bi-encoders, cross-encoders, and ColBERT

This distinction is important if you want better than baseline retrieval.

#### Bi-encoder

- embed query and documents separately
- supports precomputed document vectors
- fast and scalable
- usually the semantic retrieval default

#### Cross-encoder

- score query-document pairs jointly
- usually better relevance quality
- too slow for large-scale first-pass retrieval
- excellent for reranking a shortlist

#### ColBERT

- token-level vectors with late interaction
- closer to cross-encoder quality
- much heavier storage footprint than bi-encoders

### Reranking

Reranking is a classic mature-team move:

1. retrieve a broader candidate set quickly
2. rerank the shortlist with a stronger but slower model
3. send only the best few chunks to the LLM

Why reranking works:

- first-pass retrieval is fast but noisy
- reranking sharpens the final context
- the LLM sees fewer irrelevant chunks

In the course, reranking is positioned as a high-value upgrade after baseline retrieval is working.

## LLMs, Prompting, and Grounding

Module 4 shifts focus from retrieval to the LLM side of the system.

### Why grounding works

Transformers can integrate injected context deeply through attention. That is why retrieved documents can materially improve responses even without changing model weights.

But grounding is not automatic. Models can still:

- ignore context
- overgeneralize from it
- invent unsupported details

So the system must guide and evaluate the model.

### Sampling parameters

The module and notebooks cover the main generation controls:

- `temperature`
- `top_p`
- `top_k`
- repetition penalties

For factual RAG:

- use lower temperature
- keep diversity controlled
- avoid overly creative settings

For creative assistants:

- slightly more exploration can help

### Model selection

The course frames model choice around:

- quality
- cost
- latency
- context window
- training cutoff date

That is the right production mindset. There is no single best model for every RAG system.

### Prompt engineering for RAG

The prompt must make the model's job explicit.

Good RAG prompts usually tell the model to:

- use only retrieved information
- say when evidence is missing
- cite sources
- ignore irrelevant documents
- follow a strict output format when needed

The repo labs also show:

- conversation context management
- prompt templates
- structured outputs
- task classification via prompts

### In-context learning

The course covers one-shot and few-shot examples, especially for style and output structure.

Examples are useful when you need:

- consistent format
- customer support tone
- specific response templates
- better classification or extraction behavior

One practical extension in RAG is to retrieve good past examples and inject them alongside knowledge snippets.

### Reasoning models and chain-of-thought-style workflows

The course covers reasoning-oriented prompting and the tradeoffs involved.

Benefits:

- better decomposition
- better integration of retrieved context
- stronger complex reasoning

Costs:

- more latency
- more token usage
- more context pressure

The course also notes an important warning: many prompting tricks that work on older chat models do not transfer cleanly to reasoning models. Clear goals and strict formats usually work better.

### Context management

As conversations grow, context must be managed aggressively.

Good strategies from the module:

- prune old history
- include only chunks relevant to the latest turn
- drop unnecessary reasoning artifacts from history
- use long-context models only when truly needed

### Hallucinations

The course is explicit: RAG reduces hallucinations, but does not eliminate them.

Why hallucinations still happen:

- models predict likely text, not truth
- retrieved evidence may be incomplete
- prompts may be weak
- irrelevant chunks may be included

Ways to reduce hallucinations:

- strong grounding instructions
- better retrieval and reranking
- smaller, cleaner context
- citations
- faithfulness evaluation
- system prompts that require the model to admit uncertainty

The materials also mention self-consistency methods, but they are framed as expensive and unreliable as a primary solution.

### Citations

Citations are valuable for:

- user trust
- analyst review
- debugging
- evaluation

But they are not enough by themselves because models can hallucinate citations. Citation quality must be measured, not assumed.

### Agentic RAG

The later part of Module 4 describes agentic systems as flows of smaller specialized steps, not just one big prompt.

Examples include:

- router LLMs
- evaluators
- citation generators
- planners
- sequential workflows
- conditional workflows
- iterative workflows
- parallel workflows

This is useful, but the production lesson is clear:

- start simple
- add components only when they measurably help
- use smaller models for routing and evaluation
- reserve larger models for final answer generation

## Evaluation, Monitoring, and Observability

Modules 4 and 5 together make a strong point: if you cannot evaluate your RAG system, you do not really control it.

### Evaluation should happen at multiple levels

| Level | What to measure | Example metrics |
| --- | --- | --- |
| Retriever | are the right chunks being found? | Recall@K, Precision@K, MAP, MRR |
| LLM | is the response relevant and grounded? | Response relevancy, faithfulness, citation quality, noise sensitivity |
| System | is the whole pipeline useful and efficient? | latency, throughput, token usage, cost, thumbs up/down |
| Human | does the output actually help? | acceptance rate, manual review, curated benchmarks |

### Retriever evaluation

The notebooks show the mechanics of computing retrieval metrics over labeled data. This is essential whenever you change:

- chunking
- embedding model
- hybrid weights
- reranking
- query rewriting

### LLM evaluation

The course highlights several classes of LLM evaluation:

- automated benchmarks
- human-scored evaluation
- LLM-as-a-judge

For RAG-specific evaluation, it points to RAGAS-style metrics such as:

- response relevancy
- faithfulness
- citation quality
- noise sensitivity

Important nuance:

- response relevancy does not guarantee factual correctness
- faithfulness measures support from retrieved evidence
- citation quality matters independently from fluency

### Human feedback

Human evaluation is expensive, but it catches failures that automated checks miss.

Useful human signals:

- thumbs up / thumbs down
- textual feedback
- curated test set review
- escalation queues for bad answers

### Code-based evaluation

Some checks should be automated and cheap:

- latency
- throughput
- tokens per second
- JSON schema validity
- presence of citations
- empty-context handling

### Custom evaluation datasets

Module 5 strongly favors creating datasets from real traffic. That is the right move in production.

Store at least:

- prompt
- answer
- source chunks
- route taken
- model used
- user feedback

Then analyze:

- failure clusters
- topic-level quality
- component-level regressions

### Tracing and observability

The repo's Module 5 notebook uses OpenTelemetry and Phoenix to trace the path of a prompt through a RAG system.

That is a production-grade habit.

A good trace should show:

- original user query
- rewritten query if any
- retrieved chunks
- reranked order
- final prompt
- model response
- latency per stage
- errors and retries

Phoenix is especially useful because it gives a visual inspection layer for RAG pipelines and is built on top of OpenTelemetry and OpenInference instrumentation.

### Experiments and A/B tests

The course repeatedly pushes experimentation instead of intuition.

Typical experiments:

- prompt version A vs B
- reranker on vs off
- chunk size variant A vs B
- semantic only vs hybrid
- query rewriter on vs off

This is how good teams avoid cargo-cult architecture.

## Cost, Latency, Security, and Multimodal RAG

Module 5 moves from "can it answer?" to "can it survive production?"

### Latency

One of the clearest production lessons in the course:

- the transformer is usually the latency bottleneck
- retrieval is often relatively fast
- expensive extra components should justify themselves

Ways to reduce latency:

- use router models to skip unnecessary steps
- use smaller or quantized models
- reduce prompt size
- lower `top_k` where safe
- remove query rewriting or reranking if they do not help enough
- cache repeated or near-duplicate questions

### Caching

The course covers two useful caching patterns:

#### Direct caching

Return a cached answer for highly similar prompts.

#### Personalized caching

Use a smaller fast model to adapt a cached answer to the current user prompt.

This is especially useful for support-style assistants with repeated requests.

### Quantization

Quantization is presented as a practical way to cut memory and compute costs for both:

- LLMs
- embedding vectors

Important themes from the module:

- 8-bit quantization often preserves much of the value
- aggressive 1-bit approaches are faster but can hurt quality
- Matryoshka-style embeddings support flexible vector sizes

### Cost control

The course breaks cost into two main buckets:

- LLM inference cost
- vector database storage and query cost

Common levers:

- smaller models
- shorter prompts
- fewer retrieved chunks
- dedicated endpoints for predictable serving
- storage tiering for vectors and documents

### Security and privacy

RAG is often chosen because the knowledge itself is sensitive. That means security is not optional.

Core concerns from the course:

- knowledge base leakage
- tenant separation
- prompt-based data exfiltration
- local or on-prem deployment
- vector reconstruction risk

Practical controls:

- authenticate and authorize every request
- enforce tenant-aware retrieval boundaries
- do not rely only on prompt instructions for access control
- separate tenant indexes or strongly enforce metadata filters
- encrypt document text where possible
- understand that ANN search typically needs vectors in usable memory

### Multimodal RAG

The final module expands RAG beyond plain text.

Multimodal RAG is needed when the source material includes:

- images
- diagrams
- slides
- charts
- screenshots
- PDFs where layout matters

To do this well, both the retriever and the generator need multimodal capability.

The course describes:

- multimodal embedding models
- multimodal vector retrieval
- language-vision generation
- PDF patch retrieval similar in spirit to ColBERT-style late interaction

Why this matters:

- slides and PDFs are information-dense
- charts and captions often carry key meaning
- OCR-only pipelines can miss layout and visual relationships

Tradeoff:

- many more vectors
- more storage
- more pipeline complexity

## What Mature Teams Usually Do

If you want this README to reflect what strong engineering teams typically converge on, this is the practical stack implied by the repo materials and reinforced by official tooling docs.

### 1. They start with a measurable use case

They define:

- target users
- trusted sources
- unacceptable failure modes
- latency and cost budget
- evaluation set before scaling the system

### 2. They treat ingestion and metadata as first-class

They do not just embed raw files. They preserve:

- source
- section
- author
- time
- permissions
- tenant
- document type

### 3. They default to hybrid retrieval, not semantic-only retrieval

Why:

- lexical search catches exact terms
- vector search catches paraphrases
- metadata controls narrow the search space

### 4. They rerank shortlists

Fast first-pass retrieval plus stronger second-pass reranking is one of the cleanest quality improvements in production RAG.

### 5. They experiment with chunking instead of assuming one perfect size

They use simple chunking first, then test:

- overlap
- semantic chunking
- context-aware chunking
- domain-specific splitting

### 6. They use query rewriting and routing selectively

These components are useful, but only when measured against real evals. Mature systems do not add LLM steps just because they are fashionable.

### 7. They make grounding explicit

They require the model to:

- use retrieved evidence
- cite sources
- admit uncertainty
- ignore unsupported claims

### 8. They build offline evals and online feedback loops

They measure:

- retrieval quality
- grounded answer quality
- latency and cost
- user satisfaction

### 9. They trace the full pipeline

They know for each bad answer:

- what query was used
- what documents were retrieved
- how reranking changed the list
- what prompt the model actually saw

### 10. They optimize for quality, latency, and cost together

Not just one of them.

Common levers:

- caching
- quantization
- smaller specialist models
- removing low-value components
- prompt pruning

### 11. They design for security and tenancy early

This is essential for enterprise and internal data systems.

### 12. They go multimodal when the data demands it

Not every RAG system needs this. But PDF-heavy, slide-heavy, or visually structured knowledge systems often do.

## Production Patterns Seen in Major Platforms

If you look across official docs from OpenAI, Weaviate, cloud architecture guides, and observability platforms, a few production patterns repeat. That is a stronger signal than any one blog post.

### 1. Retrieval is rarely "vector search only"

Official platform docs repeatedly converge on hybrid retrieval:

- lexical or keyword retrieval for exact terms
- semantic retrieval for paraphrases and meaning
- reranking to clean up the shortlist
- metadata filters for scope, permissions, or tenant boundaries

Examples:

- OpenAI's file search docs describe a retrieval stack that rewrites queries, breaks complex questions into multiple searches, runs both keyword and semantic search, and reranks the results.
- Weaviate's hybrid search docs expose this directly through combined vector and BM25 retrieval with a tunable `alpha`.
- Azure and Google Cloud reference architectures both present retrieval as an orchestrated subsystem, not just a single embedding lookup.

### 2. Query preprocessing is a first-class step

Mature systems do not always send the raw user prompt directly to the retriever. Official docs and architectures commonly include:

- query rewriting
- decomposition of multi-part questions
- routing
- filtering
- tool selection

This matters because real prompts are messy, conversational, and often under-specified.

### 3. Reranking is treated as a quality multiplier

Major platform patterns consistently separate retrieval into:

- fast first pass
- stronger second pass

That is exactly the right engineering tradeoff because:

- the first pass optimizes recall and speed
- the second pass optimizes precision

### 4. Observability is part of the architecture, not an afterthought

Phoenix and OpenTelemetry-style tracing represent what good teams actually need in production:

- which query was sent
- which chunks came back
- how reranking changed the order
- what prompt the model saw
- where latency accumulated

If a team cannot inspect those steps, it cannot debug or improve the system reliably.

### 5. "Production-ready" means balancing quality, speed, cost, and security together

Cloud and platform docs consistently frame RAG as a tradeoff problem:

- quality vs latency
- recall vs prompt size
- cost vs model size
- openness vs data security

That is why serious systems almost always include:

- caching
- storage tiering
- quantization where appropriate
- access control
- tenant-aware retrieval boundaries

### Official references behind these production patterns

- OpenAI file search: <https://developers.openai.com/api/docs/guides/tools-file-search>
- OpenAI evals: <https://developers.openai.com/api/docs/guides/evals>
- Weaviate hybrid search: <https://docs.weaviate.io/weaviate/search/hybrid>
- Arize Phoenix: <https://arize.com/docs/phoenix>
- Azure RAG architecture guidance: <https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-application-design>
- Google Cloud RAG reference architectures: <https://cloud.google.com/architecture/gen-ai-rag-vertex-ai-vector-search>

## Production Examples and Flows

The sections above explain the ideas. This section shows what those ideas look like as systems.

### Example 1: Customer Support RAG

This is the cleanest first production use case.

#### Knowledge base

- FAQs
- help center articles
- refund policies
- account and billing docs
- curated high-quality support conversations

#### Query-time flow

```text
User asks question
  -> classify intent
  -> apply tenant or product filters
  -> rewrite query if needed
  -> run hybrid retrieval
  -> rerank top 20 down to top 5
  -> build grounded prompt
  -> answer with citations
  -> log feedback and trace
```

#### Example

User question:

```text
I forgot my password and I cannot get the reset email. What should I do?
```

Possible final context:

```text
[DOC 1] Password reset emails can take up to 10 minutes. Check spam and promotions folders.
[DOC 2] If no reset email arrives after 10 minutes, verify the account email and contact support.
[DOC 3] Enterprise tenants may require admin-managed password resets.
```

Possible grounded answer:

```text
Wait up to 10 minutes and check your spam or promotions folder first [DOC 1].
If the email still does not arrive, confirm that you are using the correct account email [DOC 2].
If your workspace is enterprise-managed, your admin may need to reset it for you [DOC 3].
```

#### What to evaluate

- Recall@K on support questions
- answer relevancy
- faithfulness
- citation accuracy
- resolution rate
- p95 latency

#### Why strong teams like this use case

- repeated prompts make caching effective
- answer quality is easy to evaluate
- citations are natural
- routing can grow later into agentic workflows

### Example 2: Codebase RAG

This is one of the most useful internal engineering systems.

#### Knowledge base

- source files
- docstrings
- README files
- ADRs
- API specs
- runbooks
- incident notes

#### Ingestion flow

```text
repo files
  -> split by file, class, function, module, and doc block
  -> preserve path, symbol, language, and repository metadata
  -> create lexical + vector indexes
```

#### Query-time flow

```text
Developer question
  -> detect whether question is code navigation, explanation, or change request
  -> retrieve by symbol names and semantic similarity
  -> rerank using path, symbol, and chunk text
  -> build answer with file and symbol references
  -> optionally hand off to an agent that edits code
```

#### Example

User question:

```text
Where is authentication handled and what should I change to add token refresh?
```

Retrieved context might include:

- auth middleware file
- token service implementation
- refresh endpoint
- API contract

A good response would:

- explain the current auth flow
- reference exact files and functions
- identify the refresh token lifecycle
- point out tests to update

#### What to evaluate

- file-level retrieval accuracy
- exact symbol retrieval
- answer usefulness to engineers
- citation or file-reference correctness
- time saved on onboarding or debugging

### Example 3: PDF and Slide Deck RAG

This is where a lot of teams move after text-only RAG works.

#### Knowledge base

- PDFs
- slides
- diagrams
- charts
- captions
- page metadata

#### Flow

```text
documents
  -> extract text
  -> keep page, section, title, and file metadata
  -> if layout matters, add multimodal or patch-based retrieval

user question
  -> retrieve relevant text chunks and pages
  -> optionally retrieve image or chart regions
  -> build grounded prompt
  -> answer with page references
```

#### Example

User question:

```text
What did the Q4 operations review say about shipping delays in Europe?
```

Strong answer behavior:

- cite the exact document and page
- summarize text and chart evidence together
- distinguish between stated facts and inferred trends

#### What to evaluate

- page retrieval quality
- citation by page or slide
- chart and caption coverage
- multimodal grounding quality

## GraphRAG

GraphRAG is a retrieval pattern that adds a graph layer on top of your knowledge base so the system can reason over entities, relationships, and multi-hop connections across documents.

Standard RAG is usually strongest when the answer lives in a few chunks that can be retrieved directly. GraphRAG becomes attractive when the answer depends on relationships spread across many sources.

### When GraphRAG helps

Use GraphRAG when questions look like this:

- "How are these teams, systems, and incidents connected?"
- "What changed across related policies over time?"
- "Which suppliers, products, and regions are affected by the same issue?"
- "Summarize the main themes across this whole body of research."

It is especially useful for:

- enterprise knowledge maps
- research and literature synthesis
- compliance and policy linkage
- incident and dependency analysis
- relationship-heavy document collections

### Core GraphRAG idea

Instead of storing only chunks and vectors, you also build a graph containing:

- entities
- relationships
- document references
- communities or clusters
- optional summaries at graph or community level

Typical flow:

```text
documents
  -> extract entities and relationships
  -> build graph nodes and edges
  -> link graph elements back to source documents
  -> create summaries or community views

query
  -> detect entities and intent
  -> retrieve graph neighborhoods or communities
  -> retrieve supporting source chunks
  -> synthesize answer with both relationship context and evidence
```

### GraphRAG vs standard RAG

| Pattern | Best for | Tradeoff |
| --- | --- | --- |
| Standard RAG | direct fact lookup, local context answers | weaker for cross-document relationship reasoning |
| GraphRAG | multi-hop reasoning, relationship discovery, corpus-level synthesis | more complex ingestion and maintenance |

### Practical recommendation

Do not start with GraphRAG unless the problem clearly needs it.

Best order:

1. Build standard production RAG first.
2. Measure where direct retrieval fails.
3. Add GraphRAG when the failures are caused by missing relationship structure rather than weak chunk retrieval.

### Example GraphRAG flow

User question:

```text
Which services were indirectly affected by the payments outage, and which teams owned those dependencies?
```

Strong GraphRAG behavior:

- identify the outage entity
- traverse related services and dependency edges
- collect team ownership links
- retrieve source incidents or runbooks for evidence
- generate a grounded answer with relationship-aware reasoning

## From RAG to Agents

RAG should usually come before agents, not after them.

The practical maturity path looks like this:

### Stage 1: Baseline RAG

```text
retrieve -> prompt -> answer
```

### Stage 2: Production RAG

```text
rewrite -> hybrid retrieve -> rerank -> grounded answer -> evaluate -> trace
```

### Stage 3: Agent-ready RAG

```text
route -> retrieve -> call tool if needed -> verify -> answer -> log
```

### Stage 4: Multi-agent workflow

```text
router agent
  -> retrieval agent
  -> SQL or API tool agent
  -> evaluator or verifier agent
  -> synthesizer agent
```

The reason this progression works is simple:

- agents without strong retrieval usually become expensive guessers
- good RAG gives agents a reliable evidence layer
- once retrieval and evals are stable, agents can safely take on planning and tool use

### Recommended order for teams building toward agents

1. Start with one high-value RAG use case.
2. Add hybrid retrieval and reranking.
3. Add offline evals and Phoenix tracing.
4. Add routing only when you have multiple paths worth choosing between.
5. Add tools and agent steps only where they clearly outperform a direct grounded answer.

### Scope note

This README covers mainstream production RAG, a high-level GraphRAG overview, and agent-ready RAG. It does not go deep into graph construction pipelines, fine-tuning workflows, or full multi-agent frameworks.

## Key Takeaways

RAG is not just "embed documents and call an LLM." The material in this repo shows that serious RAG systems are built from several interacting parts:

- clean ingestion
- thoughtful chunking
- hybrid retrieval
- reranking
- grounded prompt design
- LLM evaluation
- human feedback
- observability
- cost and latency optimization
- security controls
- multimodal support when the data requires it

The most important practical lesson across all five modules is this:

```text
The best RAG systems are measured systems.
```

They do not assume:

- that semantic search alone is enough
- that citations are automatically correct
- that more components always help
- that retrieval quality can be guessed
- that good answers in a notebook will hold up in production

They evaluate, trace, iterate, and simplify wherever possible.

This README should give readers the conceptual map they need before they start building real RAG systems.
