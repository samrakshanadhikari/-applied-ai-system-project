# 🎵 Music Recommender Simulation

## Project Summary

This project builds a CLI-first music recommender that scores songs from a CSV catalog against different user taste profiles. It uses genre, mood, energy, tempo, valence, danceability, and acousticness to rank songs and explain why they matched.

The project now also includes a prompt-based RAG pipeline: users can describe what they feel like listening to in natural language, the app retrieves candidates from a public music source (iTunes Search API) plus the local CSV catalog, applies semantic retrieval with a pretrained embedding model, and then generates grounded top-k recommendations using the shared ranking engine.

---

## How The System Works

Real-world recommendation systems often combine large amounts of user behavior data, such as likes, skips, playlists, and listening history, with content information about the songs themselves. At scale, platforms like Spotify or YouTube may use collaborative filtering to learn from patterns across many users and content-based filtering to compare item attributes. My simulation focuses on the content-based side: it compares a user's preferred music traits to each song's attributes, gives higher scores to songs that are closer to the user's preferred vibe, and then ranks songs from best match to worst match.

Features used in this simulation:

- `Song` features: `genre`, `mood`, `energy`, `tempo_bpm`, `valence`, `danceability`, `acousticness`
- `UserProfile` features: `favorite_genre`, `secondary_genre`, `favorite_mood`, `target_energy`, `target_tempo_bpm`, `target_valence`, `target_danceability`, `target_acousticness`

The recommender now supports two integrated workflows:

- **Profile evaluation mode**: tests multiple profiles, including reflective hip-hop and folk, high-energy pop, chill lofi, deep intense rock, and one contradictory edge case.
- **Prompt-based RAG mode**: takes a free-text request (for example, "calm acoustic songs for study"), derives a profile, retrieves candidates from external + local sources, fuses semantic + lexical similarity, and ranks the top matches.
- **Optional agentic mode**: runs a plan -> retrieve -> rank -> self-check cycle and asks a follow-up question if confidence is low.

For each workflow, it calculates a score for candidate songs and returns the top `k` songs with the highest scores.

### Prompt-Based RAG Flow

```mermaid
flowchart LR
    A[User prompt] --> B[Parse intent into profile targets]
    B --> C[Retrieve songs from iTunes API and local CSV]
    C --> D[Semantic and lexical retrieval fusion]
    D --> E[Score and rank with shared recommender]
    E --> F[Self-check confidence and optional follow-up]
    F --> G[Return top-k songs with explanations]
```

### Data Flow

```mermaid
flowchart LR
    A[User Preferences] --> B[Load songs from songs.csv]
    B --> C[Loop through each song]
    A --> C
    C --> D[Check genre and mood matches]
    C --> E[Measure closeness for energy tempo valence danceability acousticness]
    D --> F[Combine weighted points into one total score]
    E --> F
    F --> G[Store song with score and explanation]
    G --> H[Sort all songs by score]
    H --> I[Return Top K recommendations]
```

### Algorithm Recipe

- `+2.0` points for a match with `favorite_genre`
- `+1.0` point for a match with `secondary_genre`
- `+1.5` points for a match with `favorite_mood`
- Up to `+2.0` points for how close the song's `energy` is to `target_energy`
- Up to `+1.5` points for how close `tempo_bpm` is to `target_tempo_bpm`
- Up to `+1.5` points for how close `valence` is to `target_valence`
- Up to `+1.0` point for how close `danceability` is to `target_danceability`
- Up to `+1.0` point for how close `acousticness` is to `target_acousticness`

For the numerical features, the scoring rule rewards closeness rather than simply giving more credit to larger values. A song gets more points when its feature value is nearer to the user's target value, and fewer points when it is farther away. After every song receives a total score, the system applies a ranking rule by sorting the catalog from highest score to lowest score and recommending the top results.

### Potential Biases

- This system may over-prioritize genre and miss songs from other genres that still match the user's mood and vibe.
- A fixed user profile can be too static, even though real musical taste changes across time, context, and emotion.
- The dataset is small, so the recommender can only choose from a narrow slice of music.
- The system does not understand lyrics, storytelling, or personal meaning, which matter a lot for songs like reflective hip-hop and folk.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the original profile demo:

```bash
python -m src.main
```

4. Run prompt-based RAG mode:

```bash
python -m src.main --prompt "I want calm acoustic songs for late-night focus"
```

5. Run prompt mode interactively:

```bash
python -m src.main --interactive
```

6. Run agentic mode with semantic retrieval:

```bash
python -m src.main --interactive --agentic
```

7. Local-only fallback (disable external retrieval):

```bash
python -m src.main --prompt "happy workout pop tracks" --no-external
```

8. Lexical-only retrieval fallback (disable embeddings):

```bash
python -m src.main --prompt "lyrical rap songs" --lexical-only
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

RAG pipeline tests live in `tests/test_rag_recommender.py`.

### Logging and Guardrails

- Runtime logs are written to `logs/app.log`.
- Empty prompt input is rejected with a clear error message.
- External retrieval failures are handled safely; the app falls back to local data instead of crashing.
- If semantic model loading fails, retrieval falls back to lexical matching.
- Agentic mode asks follow-up clarification when confidence is below threshold.

---

## Experiments You Tried

- Tested five profiles: Reflective Hip-Hop / Folk, High-Energy Pop, Chill Lofi, Deep Intense Rock, and a Contradictory Edge Case.
- Ran a weight-shift experiment where genre was cut in half and energy was doubled for the High-Energy Pop profile.
- Compared whether the results felt intuitive or if the scoring logic could be tricked by conflicting preferences.

---

## Limitations and Risks

- It only works on a tiny catalog, so some profiles will see the same songs repeatedly.
- Genre bonuses can overpower other signals, especially for edge-case users.
- It does not understand lyrics, storytelling, or personal associations with songs.

You will go deeper on this in your model card.

---

## Reflection

Read `model_card.md` and `reflection.md`:

[**Model Card**](model_card.md)
[**Reflection Notes**](reflection.md)

Building the recommender made it much easier to see how recommendation systems turn preference data into rankings. A few carefully chosen weights can make the results feel smart for normal users, but edge cases quickly show where the logic is brittle.

The most important lesson from Phase 4 was that explanation matters. When the CLI prints the score and the reasons, it becomes much easier to see why a song ranked highly and where the recommender may be biased or overly rigid.


---

## Example Terminal Output

This screenshot shows the recommender running in the terminal.

![Terminal output](images/recommender-output.png)

## Evaluation Screenshots

This first screenshot shows the recommender output through the Chill Lofi profile.

![Evaluation output part 1](images/evaluation-part1.png)

This second screenshot shows the Deep Intense Rock profile and the remaining evaluation output.

![Evaluation output part 2](images/evaluation-part2.png)

