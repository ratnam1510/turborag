# LLM Request Model

## Short Answer

TurboQuant-style compression by itself does not inherently reduce the number of LLM requests in a standard RAG pipeline.

It changes the retrieval layer, not the final generation call.

## If You Replace Normal Vector Search With TurboRAG Dense Retrieval

A typical standard RAG flow is:

1. Embed query.
2. Search vector store.
3. Fetch chunks.
4. Send context to the LLM.

If you switch only the retrieval engine to TurboRAG dense mode, the LLM request count usually stays the same:

- same embedding request pattern,
- same final generation request pattern,
- fewer memory and storage costs in the retrieval layer,
- potentially lower retrieval latency.

## Where TurboRAG Can Change LLM Usage

### Index Time

If you enable the graph layer described in the PDF, TurboRAG can require more LLM calls during indexing because it may use an LLM for:

- entity extraction,
- relationship extraction,
- community summarisation.

Those are build-time costs, not necessarily query-time costs.

### Query Time

In the intended architecture, query-time can still stay close to a normal RAG pattern:

- one embedding step,
- one retrieval step,
- one final LLM answer step.

If reranking uses a cross-encoder rather than an LLM, that does not add LLM requests.

## When Overall LLM Spend Can Go Down Indirectly

Even if request count stays the same, total spend can still go down if TurboRAG helps you:

- retrieve more accurate chunks,
- avoid retries and fallback prompts,
- send less irrelevant context,
- reduce the need for oversized top-k retrieval,
- get acceptable quality with smaller prompts.

So the honest answer is:

- direct LLM request count reduction: not guaranteed,
- index storage and retrieval efficiency improvement: yes,
- potential end-to-end LLM cost improvement through better retrieval quality: yes, indirectly.

## Practical Recommendation

For an existing RAG user, start with dense TurboRAG sidecar adoption first.

That gives you:

- the easiest migration,
- no database rewrite,
- no immediate increase in LLM indexing complexity,
- and a clean baseline to compare against your current stack.

After that, if you want multi-hop or graph reasoning, add the graph layer as a second phase and evaluate whether the build-time LLM cost is worth it for your workload.
