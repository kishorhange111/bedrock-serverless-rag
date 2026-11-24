3 seeds × 6 tenants × 12 questions (216 queries per row); offline lexical embedder + extractive generator.

| min_score | answer accuracy (answerable) | wrong answers | abstained (answerable) | abstained (unanswerable) | cross-tenant leaks | cross-tenant citations | accuracy on docs with planted injection |
|---|---|---|---|---|---|---|---|
| 0.0 | 84.0% | 13.9% | 2.1% | 6.9% | 7 | 0 | 0.0% (n=23) |
| 0.1 | 84.0% | 2.1% | 13.9% | 72.2% | 3 | 0 | 0.0% (n=23) |
| 0.2 | 83.3% | 0.0% | 16.7% | 100.0% | 0 | 0 | 0.0% (n=23) |
| 0.3 | 63.9% | 0.0% | 36.1% | 100.0% | 0 | 0 | 0.0% (n=23) |
| 0.4 | 25.7% | 0.0% | 74.3% | 100.0% | 0 | 0 | 0.0% (n=23) |
| 0.5 | 5.6% | 0.0% | 94.4% | 100.0% | 0 | 0 | 0.0% (n=23) |
