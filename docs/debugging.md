# Debugging log

Kept specific rather than generic — this is the part that actually shows the work, not a
"challenges faced" summary.

| Issue | Root cause | Fix |
|---|---|---|
| AraBERT load hung/crashed the kernel on Windows | `sentence-transformers`'s SentencePiece tokenizer deadlocks under Windows threading | Load via `transformers` `AutoTokenizer`/`AutoModel` directly, `TOKENIZERS_PARALLELISM=false` |
| Re-indexing silently didn't take effect, multi-hour debugging loop | Indexing and querying scripts used inconsistent relative Qdrant paths, resolving to different physical folders depending on working directory | Single absolute/dynamically-resolved path used everywhere |
| Filtered searches leaked wrong categories into results | Qdrant's top-level `query_filter` only applies after RRF fusion, not to each `Prefetch` stage | Pass the same filter into each `Prefetch` individually |
| Duplicate chunks stacking on every re-index | `uuid.uuid4()` per point meant re-running indexing without deleting the collection just added copies | Deterministic MD5 hash IDs — indexing is now idempotent |
| `uvicorn --reload` crashed with a file-lock error | Local-mode Qdrant only allows one process to hold the storage lock; `--reload` spawns two | Migrated to Qdrant-as-a-service (Docker) for local dev |
| Reranked results showed stale RRF scores instead of Cohere's actual relevance scores | Serialization reused the original point score instead of the reranker's returned score | Explicit score-override map keyed by point ID during serialization |
| Eval numbers changed unexplainably between two identical runs | Cohere trial tier's 10-calls/minute limit was silently failing mid-run and falling back to unranked order, corrupting both latency and downstream generation scores | Retry-with-backoff on rate limit, plus disclosed as a real constraint rather than hidden |
| Al Jazeera / Al Arabiya scraping failed with connection resets and 403s despite realistic headers | Cloudflare-class bot detection operating at the TLS fingerprint level, which `requests`/urllib3 can't spoof via headers alone | Diagnosed correctly rather than endlessly retried; documented as a disclosed limitation with a concrete next attempt (`curl_cffi`) instead of a silent workaround |