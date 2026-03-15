# RAG Techniques Quick Reference

This file is the short companion to the main [README](../README.md).

Use this when you want a compact operating guide for building a production-grade RAG system without rereading the full repository guide.

## Recommended Default Stack

If you are building a first serious RAG system, this is the strongest default:

```text
documents
  -> clean extraction
  -> chunking with overlap
  -> metadata enrichment
  -> lexical index + vector index

query
  -> optional rewrite
  -> metadata filters
  -> hybrid retrieval
  -> rerank shortlist
  -> grounded prompt
  -> answer with citations
  -> trace + evaluate
```

## Retrieval Choices

| Technique | Best for | Strength | Limitation |
| --- | --- | --- | --- |
| Metadata filtering | narrowing scope | fast and exact | not ranking |
| BM25 | exact words, ids, names | strong lexical precision | misses paraphrases |
| Semantic search | paraphrases and meaning | flexible retrieval | can miss exact tokens |
| Hybrid search | most production systems | best balance | needs tuning |
| Reranking | final shortlist cleanup | strong relevance gains | adds latency |

### Default recommendation

- Start with hybrid retrieval.
- Add metadata filters early.
- Rerank only the top candidate set, not the full corpus.

## Chunking Choices

| Technique | Use when | Why |
| --- | --- | --- |
| Fixed-size | baseline system | simplest and reliable |
| Fixed-size + overlap | most first versions | better boundary handling |
| Recursive splitting | docs have structure | respects sections and paragraphs |
| Semantic chunking | quality matters more than preprocessing cost | smarter topic boundaries |
| LLM chunking | concept grouping is critical | can align chunks to meaning |
| Context-aware chunking | chunk text is ambiguous alone | improves retrieval without query-time cost |

### Default recommendation

- Start with fixed-size plus overlap.
- Preserve source, section, page, and chunk index metadata.
- Evaluate before switching to semantic or LLM chunking.

## Query Improvement Techniques

### Query rewriting

Use when user prompts are:

- messy
- conversational
- ambiguous
- missing domain terminology

### Named entity extraction

Use when retrieval depends on:

- people
- dates
- organizations
- product names
- ticket ids

### HyDE

Use when:

- the query is underspecified
- baseline retrieval misses obviously relevant documents

Avoid as a default if:

- latency budget is tight
- rewrite plus hybrid retrieval is already good enough

## Reranking Rules

Use reranking when:

- top results are semantically close but not precise
- the answer quality depends heavily on the best 3-5 chunks
- hybrid retrieval returns decent recall but weak ordering

Good production pattern:

```text
retrieve top 20
  -> rerank
  -> keep top 3 to top 5
  -> send only those to the LLM
```

## Prompting Rules for Factual RAG

Always tell the model to:

- use only retrieved information
- say when evidence is missing
- cite sources
- ignore irrelevant retrieved documents

Simple template:

```text
System:
Use only the retrieved documents.
If the answer is not supported, say that clearly.
Cite sources as [DOC n].

Retrieved documents:
[DOC 1] ...
[DOC 2] ...

User question:
...
```

## Evaluation Map

| Layer | Measure |
| --- | --- |
| Retriever | Recall@K, Precision@K, MAP, MRR |
| LLM | response relevancy, faithfulness, citation quality |
| System | latency, throughput, token usage, cost |
| Users | thumbs up/down, task success, escalation rate |

### Minimum offline evaluation set

Include:

- common questions
- hard questions
- ambiguous questions
- adversarial questions
- questions where the answer should be "not enough information"

## Observability Checklist

For every request, log or trace:

- original query
- rewritten query if any
- applied filters
- retrieved chunks
- reranked order
- final prompt
- final answer
- model and parameters
- latency by stage
- user feedback

## Cost and Latency Checklist

### If latency is too high

- reduce `top_k`
- shrink prompts
- use smaller generation models
- cache repeated questions
- remove weak query-time components
- keep reranking only if it materially helps

### If cost is too high

- shorten responses
- reduce prompt size
- move to smaller or quantized models
- use cheaper retrieval storage tiers
- cache repeated answers

## Security Checklist

- enforce auth before retrieval
- scope retrieval by tenant or permission boundary
- do not rely on prompt instructions for access control
- protect chunk text and metadata
- understand vector reconstruction risk
- prefer local or controlled deployments for sensitive knowledge

## Production Build Order

Use this order unless you have a strong reason not to:

1. Build one baseline RAG pipeline.
2. Add chunk metadata and hybrid retrieval.
3. Add reranking.
4. Build offline evaluation.
5. Add tracing and observability.
6. Add caching and cost controls.
7. Add routing or tools only if the use case demands it.
8. Add agents after retrieval and evaluation are already stable.

## Common Mistakes

- using semantic search only
- retrieving too many chunks
- not preserving metadata
- adding agents before building evals
- trusting citations without checking them
- measuring only answer fluency instead of grounding
- shipping without traces

## Good First Use Cases

- support-document assistant
- FAQ bot
- policy assistant
- codebase assistant
- PDF and slide Q&A

## Not Covered Here

This quick reference does not go deep on:

- GraphRAG
- fine-tuning workflows
- full agent frameworks
- domain-specific compliance architectures

Those are better layered on after a standard production RAG stack is working well.
