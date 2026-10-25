# DADA2 — Agentic RAG over SEC filings

**Fill in both sections at the bottom before you submit.** Keep the deployed
app line exactly in this form; it is how your app gets found and graded.

## Getting started

There is no separate spec. The instructions are in the notebooks, and
`DADA2_00_Start_Here.ipynb` is the one to open first: it explains the
assignment, the setup, the grading, and what each of the four notebooks after
it asks for. Work through them in order.

Get the files the same way as every class day: pull, then work in a copy:

```bash
cd lecture-files
git pull
cd ..
cp -r lecture-files/DADA2 DADA2     # work in the copy, not in lecture-files/
cd DADA2
cp example.env .env                  # then fill in your keys
```

Then open `DADA2_00_Start_Here.ipynb` from your `DADA2` copy and run its
setup check.

Deploying is `DEPLOY.md`.

---

## Submission

Author: Donny Le

```
Deployed app: https://dada2-donny-le.onrender.com/
```

ADD YOUR NAME TO YOUR RENDER PROJECT SO THE LINK CAN BE PERSONALIZED.

PLEASE ENSURE YOUR URL WORKS BY TESTING IT IN INCOGNITO MODE.

### WHAT TO SUBMIT
Upload the entire project folder (DADA2) via GitHub integration on Canvas

### Write-up

**Conditions.** I used Gemini gemini-3.5-flash-lite for chat, Cloudflare clef-flash for decisions, and Cohere rerank-v3.5 for reranking. The embeddings were Qwen3 0.6B through Cloudflare, with a chunk size of 1,200 characters and an overlap of 150. The reranking pool held 10 passages, and the RRF constant was 60. I tested retrieval on six specific questions at k=5 and five comparative questions at k=10. The router comparison used all 24 questions while the agent comparison used eight specific and six comparative questions at k=5.

**The routers.** The decision router got 20 out of 24 questions right compared to 19 for the chat router. It was also faster, averaging 0.45 seconds per call compared to 0.69 seconds. I think this is a small advantage, especially since the accuracy difference comes down to one question. What makes the decision model useful to me is that it gives probabilities that the code can actually use. If it is unsure, the agent can fall back to another strategy. It also avoids spending Gemini calls just to choose a search strategy. I would still want more questions before saying it is consistently the better router.

**The agents.** The ladder agent had an accuracy of 0.86, compared to 0.71 for the diagnosis agent. The diagnosis agent used fewer chat calls, averaging 3.4 per question compared to 4.5, but it also used 2.6 decision calls. Their times were close, at 13.5 seconds for diagnosis and 14.0 for the ladder agent. I think this was the clearest tradeoff in the results: saving chat calls did not mean getting more questions right. The AMD dress-code example stood out because the diagnosis agent stopped after its first search with an absent verdict, while the ladder agent took three attempts and nine chat calls to reach a similar answer. That shows where the diagnosis agent saved work even though it was less accurate overall. The saved comparison only gives aggregate scores so I can't tell which question had the biggest accuracy disagreement.

**What did not help.** Adding multi-query retrieval after filtering did not help on my six specific questions. Recall@5 stayed at 1.00, while MRR dropped from 0.722 to 0.694. Adding reranking brought MRR back to 0.722, which was the same as filtering alone. Filtering by company and year had already found the relevant files. The extra versions of the query seem to have added more noise to the ranking without bringing in any additional relevant files. Since this also took another chat call, I would just use filtering for these questions. However, this does not mean reranking was useless overall: without filtering, it increased the baseline MRR from 0.325 to 0.583. Hybrid retrieval also did well on the specific questions, with an MRR of 0.783, so my results did not match every example in the starter.

**What I would ship.** Based on this run, I would ship the ladder agent first because I care more about getting the answer right than saving roughly one chat call per question, especially when the time difference was so small. Cost would still matter if a lot of people were using the app, since those extra chat calls would add up and use more of the available quota. The diagnosis agent used fewer chat calls, but it also made decision-model calls, so I would compare token usage and the cost of both models before deciding which is actually cheaper. For now, I think the higher accuracy is worth the extra chat usage. I still like the diagnosis agent's ability to stop on unsupported questions and plan missing searches, but I would want to understand its mistakes before choosing it over the ladder agent. I would also ask what people actually use the app for. On the five comparative questions, filtering lowered recall@10 from 0.83 to 0.67, so a strategy that works well for a named company and year may be a poor fit for broader questions. Before shipping, I would check more examples of each kind and review the answers and citations myself, since the accuracy scores also depend on a model judging the answers.
