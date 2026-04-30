# Phase 4 Reflection

## Profile Comparisons

- Reflective Hip-Hop / Folk vs. High-Energy Pop: The reflective profile pushes thoughtful, lower-energy songs like `4 Your Eyez Only` and `Diamonds & Rust`, while the pop profile prefers bright, upbeat songs like `Sunrise City` and `Rooftop Lights`. That makes sense because the reflective profile asks for lower valence and more acousticness, while the pop profile rewards happy energy and danceability.
- Reflective Hip-Hop / Folk vs. Chill Lofi: Both profiles like calmer energy, but the reflective one leans toward lyric-heavy hip-hop and folk while the lofi profile favors background-focus songs. The acousticness and tempo targets overlap, but the genre and mood bonuses create clearly different top results.
- Reflective Hip-Hop / Folk vs. Deep Intense Rock: These outputs are almost opposites. The reflective profile likes softer and more serious songs, while the rock profile moves toward faster, louder, and more intense tracks like `Storm Runner`.
- High-Energy Pop vs. Chill Lofi: The pop profile rewards happy, high-tempo songs for movement and excitement, while the lofi profile shifts toward slow, mellow, and more acoustic tracks. This difference shows that the energy and tempo targets are doing a lot of useful work.
- High-Energy Pop vs. Deep Intense Rock: Both profiles like energy, so some songs stay competitive across both lists. The difference is that pop rewards happiness and polish, while rock rewards intensity and slightly darker emotion.
- Chill Lofi vs. Deep Intense Rock: This is the clearest contrast in the system. Lofi prefers soft, steady songs with low energy, while rock prefers fast and intense songs with much higher energy.
- Contradictory Edge Case vs. All Other Profiles: The edge-case profile exposes the system's weakness because it mixes signals that do not naturally fit together. `Spacewalk Thoughts` ranks first mostly because the genre match still matters a lot, even when the energy and mood targets point somewhere else.

## Experiment Note

- I halved the genre weight and doubled the energy weight for the High-Energy Pop profile. The same top songs stayed near the top, but high-energy songs that were not pop moved closer to the leaders. That made the recommendations different more than it made them better, which suggests the original scoring already fit that profile fairly well.

---

## Responsible AI reflection (final project)

These notes answer how I think about **limitations**, **misuse**, **reliability surprises**, and **AI collaboration** beyond “does it run.”

### What are the limitations or biases in your system?

The recommender is only as fair and diverse as its **data and labels**. The local CSV is tiny, so users repeatedly see the same artists and styles. **Genre and mood keywords** can overweight shallow matches: my edge-case profile still favored a genre hit even when energy and mood conflicted. **RAG over iTunes** inherits **catalog bias** (commercial/region availability) and **flattened metadata**—the API does not capture lyrics, trauma, or cultural context, so “emotional” prompts are approximated. **Embeddings** reflect general-language similarity; they can miss how a specific fan hears “struggle” or “lyrical.” **Intent rules** map words to a small mood/genre set, so outliers are misread. **Confidence scores** are **heuristics**, not calibrated probabilities—they signal “internal spread of scores,” not real-world accuracy.

### Could your AI be misused, and how would you prevent that?

**Misuse:** Someone could treat rankings as **authoritative wellness or identity advice** (“this playlist fixes how you feel”), use retrieved metadata to **spam or impersonate** artists, or **over-trust** the system for **copyright/commercial** decisions (what’s “safe” to use publicly). A bad actor could also try to **game** wording to force awkward artist/strict filters.

**Mitigation I implemented or would add:** Clear **non-production / educational** framing in the model card and README; **explanations** on every row so users see *why* something ranked; **empty-input** rejection and **safe fallbacks** when the API fails; **logging** for debugging without exposing secrets; **tests** so ranking rules don’t drift silently. In a product version I would add **Terms of use**, **rate limiting**, **audit logs**, stronger **diversity** constraints, and **human moderation** for sensitive prompts—none of that replaces disclosure that this is a **student simulation**.

### What surprised you while testing your AI’s reliability?

Two surprises stood out. First, many failures looked like “**the model doesn’t understand**” when the real issue was **data and gatekeeping**: e.g. **strict artist** mode with only one matching track in the pool—not embedding magic, just an empty shelf. Second, **follow-up reliability** was harder than retrieval: a **question** (“Do you only have…?”) was initially parsed like a **command**, or words like **“only”** triggered **strict** filters in ways users didn’t intend. Fixing that required **targeted tests** and logic changes, not more tunings of the embedding model. **SSL** issues on macOS also looked like random breakage until logs and cert paths made the root cause obvious.

### Describe your collaboration with AI during this project

Here is one **helpful** case and one **flawed** case.

- **Helpful:** An AI assistant suggested structuring **RAG** as distinct stages—**fetch, merge, shortlist, score**—and adding **pytest** with **mocked** HTTP for iTunes. That matched how I already thought about the pipeline but sped up scaffolding and edge-case tests (bad JSON, SSL) without hitting the live API every time.

- **Flawed:** Early suggestions sometimes pushed **heavy keyword hacks** or **hardcoded artist lists** for “better” demos. That conflicted with my goal of **artist names coming from the merged catalog** and led to brittle behavior. I kept the **data-driven artist index** and **narrow validation rules** instead, and wrote tests so follow-up and strict-mode behavior stayed aligned with real user language.

