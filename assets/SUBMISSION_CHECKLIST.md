# Final submission checklist

Use this with the root **`README.md`** presentation section.

## Done in repo (verify after `git pull`)

- [x] Public GitHub URL in README: https://github.com/samrakshanadhikari/-applied-ai-system-project
- [x] Comprehensive **`README.md`** (architecture Mermaid, setup, samples, reliability, reflection pointer, portfolio section)
- [x] **`model_card.md`** (biases, AI collaboration, testing — plus sections 10–13 for RAG)
- [x] **`reflection.md`** (Responsible AI Q&A for limitations, misuse, reliability surprises, collaboration)
- [x] **`assets/`** folder with demo PNGs copied from `images/` (`recommender-output.png`, `evaluation-part1.png`, `evaluation-part2.png`)
- [x] **`assets/MERMAID_ARCHITECTURE_FOR_EXPORT.mmd`** — source for PNG export
- [ ] **`assets/system-architecture.png`** — export from Mermaid Live (optional if README Mermaid alone is accepted; add if your grader requires a diagram image file)

## You must still do

1. **Export architecture PNG**  
   Open `MERMAID_ARCHITECTURE_FOR_EXPORT.mmd` → copy all → [mermaid.live](https://mermaid.live) → **Export PNG** → save as **`assets/system-architecture.png`** → commit.

2. **Optional extra screenshots** (nice for portfolio, not strictly required if Loom is complete):  
   `streamlit-demo-1.png`, `agentic-confidence.png` from live UI.

3. **Loom (required)** — 5–7 minutes, **Streamlit** focus:  
   - 2–3 different prompts with **clear on-screen outputs**  
   - Call out **RAG** (external song count / retrieval line) and **agentic** (confidence + follow-up)  
   - One **guardrail** (empty prompt warning, or describe **local-only** fallback)  
   - **No** install or repo tour  

   Paste the share URL into **`README.md`** where it says `REPLACE_ME`.

4. **Git**  
   Meaningful commits; **push** to `main` before the deadline.  
   If `images/` and `assets/` duplicate PNGs, you can delete `images/` later or keep both—rubric asks for **`assets/`**.

5. **5–7 min live presentation** (class slides)  
   Same story as Loom: demo + one lesson learned + limitations in one sentence.
