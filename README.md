# bedrock-serverless-rag

Multi-tenant RAG on AWS serverless, built the way it gets built in a customer account. It uses Bedrock (Claude for answers, Titan for embeddings), Lambda, S3, SQS, DynamoDB, OpenSearch Serverless, Cognito, API Gateway, Step Functions and Terraform.

The work here is mostly the plumbing around the model, where these systems usually fail in production:

- duplicate and out-of-order S3 events;
- a document's old version lingering in the index;
- one tenant's chunks showing up in another tenant's answer;
- planted instructions in documents;
- answers that cite nothing.

```
            ┌──────────── ingest ─────────────┐
 upload ──► S3 ──► SQS (+DLQ) ──► Lambda ──► Titan embeddings ──► OpenSearch Serverless
 <tenant>/<doc>          │                │                            (k-NN, tenant pre-filter)
                         │                └─► DynamoDB ledger (sha256, conditional write)
 Step Functions ─► list ─► Map ─► same ingest Lambda   (backfill / re-embed after model change)

            ┌──────────── query ──────────────┐
 client ─► API Gateway (Cognito JWT) ─► Lambda ─► embed ─► k-NN (tenant from token claims)
                                              ─► strip injected sentences ─► Claude on Bedrock
                                              ─► citation check / abstain ─► DynamoDB session (TTL)
                                              ─► CloudWatch EMF metrics (latency, cost, abstain)
```

## Design decisions

| problem | what this does |
|---|---|
| **Duplicate or replayed S3 events.** S3 delivers at least once. | A DynamoDB ledger stores `sha256` per `<tenant>#<doc>`. A conditional `PutItem` (`attribute_not_exists(pk) OR sha256 <> :h`) makes an unchanged re-delivery a no-op. If embedding fails after the claim, the claim is released, so the SQS retry re-indexes instead of being skipped as "unchanged". |
| **A new document version leaves stale chunks behind.** | All of a document's chunks are deleted before the new ones are written. A delete event removes both the chunks and the ledger row. |
| **One poison document blocks a batch.** | SQS sits in front of Lambda with `ReportBatchItemFailures`, so only the failing message is retried. After 3 tries it goes to a DLQ, which has an alarm. |
| **Tenant isolation.** | The tenant comes only from the verified Cognito JWT (`custom:tenant`, which is immutable and not writable by users), never from the request body. OpenSearch applies the tenant as an **efficient k-NN pre-filter**, so another tenant's chunk can never displace your top-k. Chunk IDs are tenant-qualified (see the bug below). |
| **Indirect prompt injection.** | Sentences matching instruction patterns are **stripped from retrieved chunks** before prompting and counted in a metric. The system prompt also tells the model that sources are untrusted data. |
| **Ungrounded answers.** | Every citation `[n]` must refer to a provided source. An answer that cites nothing, or cites a source that doesn't exist, becomes an abstention. A retrieval-score floor (`MIN_SCORE`) abstains before calling the model at all. |
| **Bulk re-embedding** (e.g. when changing the embedding model). | A Step Functions Map over a prefix listing reuses the same ingest Lambda, with jittered retries on throttling. `MaxConcurrency` and the function's reserved concurrency cap parallel Bedrock calls, so a backfill can't exhaust the account quota. |
| **Observability and cost.** | CloudWatch Embedded Metric Format gives metrics from one JSON log line per request, with no `PutMetricData` calls: retrieval and total latency, hits, stripped injections, abstentions and USD per request. Alarms cover the DLQ, the abstain rate (above 40% usually means a broken index) and Lambda errors. |
| **Least privilege.** | Each function has its own role. `bedrock:InvokeModel` is scoped to the embedding model and Claude ARNs. Writes go to only the tables each function needs. KMS encryption covers S3 and DynamoDB, S3 public access is blocked, and the Lambdas run on arm64 with X-Ray tracing. |

## Evaluation (offline harness)

`python -m srag.evaluate` drives the **real handlers** end to end, with moto for S3 and DynamoDB, over a synthetic corpus of 6 tenants × 12 topics × 3 seeds. The corpus is built so the failure modes are present:

- every topic exists in several tenants with *different* values;
- a third of each tenant's questions ask about a topic only *other* tenants have, where the correct answer is to abstain;
- about 15% of documents contain a planted "ignore all previous instructions…" sentence.

It runs on an offline lexical embedder and an extractive generator. The corpus is templated and easy, so absolute accuracy here is a **floor, not what Titan + Claude would score**. The point is the harness: swap `Deps` to Bedrock and the same script measures the real stack.

**v1: drop any chunk that contains an injection** (`results/eval_v1_chunk_drop.md`):

| min_score | answer accuracy | wrong answers | answered an unanswerable question | accuracy on injected docs |
|---|---|---|---|---|
| 0.0 | 84.0% | 13.9% | 93.1% | **0.0%** (n=23) |
| 0.2 | 83.3% | 0.0% | 0.0% | **0.0%** |

Dropping the whole chunk also threw away the fact next to the injection. Each short doc is a single chunk, so all 23 injected documents became unanswerable, and at a low threshold the model answered from the wrong document instead.

**v2: strip only the injected sentences** (`results/eval.md`):

| min_score | answer accuracy (answerable) | wrong answers | abstained (answerable) | abstained (unanswerable) | cross-tenant citations | accuracy on injected docs |
|---|---|---|---|---|---|---|
| 0.0 | 100.0% | 0.0% | 0.0% | 4.2% | 0 | 100.0% (n=23) |
| 0.1 | 100.0% | 0.0% | 0.0% | 72.2% | 0 | 100.0% |
| **0.2** | **99.3%** | 0.0% | 0.7% | **100.0%** | 0 | 100.0% |
| 0.3 | 71.5% | 0.0% | 28.5% | 100.0% | 0 | 47.8% |
| 0.5 | 5.6% | 0.0% | 94.4% | 100.0% | 0 | 0.0% |

- **The abstention threshold is the main quality knob, and it depends on the embedder.** With no threshold, 96% of questions about topics this tenant doesn't have get a confident answer from an unrelated document. At 0.2 every one abstains, at a cost of 0.7% of answerable questions. The threshold has to be re-calibrated whenever the embedding model changes; that is what the harness is for.
- **Zero cross-tenant citations** at every threshold. Pre-filtering means another tenant's chunks are never candidates.
- **CI gate.** The test job reruns the eval and fails the build if the 0.2 operating point drops below 95% accuracy or 95% correct abstention, or if any cross-tenant citation appears.

### A real bug the tests caught

Chunk IDs were originally `<doc>#<n>`. When two tenants each uploaded `policy.txt`, the second upload **overwrote the first tenant's chunks** in the index. That is silent data loss, and depending on the store's upsert semantics a cross-tenant mix-up. `test_tenant_isolation` failed on it; IDs are now `<tenant>/<doc>#<n>`, and there's a regression test.

## Run it

```
pip install -e ".[dev]"
pytest -q                          # 14 tests: handlers end-to-end on moto S3/DynamoDB
python -m srag.evaluate            # writes results/eval.md
```

Deploy:

```
make package                       # vendors arm64 wheels + src into build/lambda
cd infra && terraform init
terraform apply -var gen_model_id=<Claude model or inference-profile ID from your Bedrock console>
aws s3 cp policy.txt s3://$(terraform output -raw docs_bucket)/acme/policy.txt
```

The OpenSearch index (`opensearch_index_body()` in `components.py`, HNSW/faiss, inner product on normalised Titan vectors) is created once before the first ingest. CI runs `terraform fmt -check` and `terraform validate`. **Nothing here has been applied to a real AWS account**, and the Bedrock and OpenSearch adapters are exercised only through their offline twins and request-shape tests.

## Layout

```
src/srag/components.py   chunker, embedders (Titan / hashing), vector stores (OpenSearch / in-memory),
                         generators (Claude on Bedrock via the Anthropic SDK / extractive)
src/srag/handlers.py     ingest, reindex-list and query Lambda handlers, EMF metrics
src/srag/evaluate.py     offline multi-tenant eval harness
infra/                   Terraform: S3, SQS+DLQ, DynamoDB, AOSS, Lambda, Cognito, HTTP API, SFN, alarms
```
