# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**VibeRank CLI**

---

## 2. Intended Use

This recommender suggests songs from a small classroom catalog by comparing a user's taste profile to each song's features. It is meant for learning how recommendation systems work, not for real production music apps.

---

## 3. How the Model Works

The model gives points for matching the user's main genre, backup genre, and mood. It also gives similarity points when a song's energy, tempo, valence, danceability, and acousticness are close to the user's target values. After every song gets a total score, the system sorts the songs from highest score to lowest score and returns the top matches with short explanations.

---

## 4. Data

The dataset contains 17 songs from `data/songs.csv`. It includes pop, lofi, rock, ambient, jazz, synthwave, indie pop, hip-hop, folk, and indie folk, with moods like happy, chill, intense, reflective, anxious, defiant, yearning, heartbroken, sentimental, and tender. I expanded the starter file with more reflective hip-hop and folk songs so the catalog better matches the kind of music I wanted to test.

---

## 5. Strengths

The system works well when the user profile has a clear vibe. The reflective hip-hop and folk profile correctly put `4 Your Eyez Only`, `FEAR.`, and softer folk tracks near the top, while the chill lofi profile favored `Library Rain`, `Midnight Coding`, and `Spacewalk Thoughts`. The explanations also make the results easy to understand because I can see exactly which features contributed to a song's ranking.

---

## 6. Limitations and Bias

One weakness I found is that the system can over-reward genre even when the rest of the profile is contradictory. In my edge-case test, `Spacewalk Thoughts` ranked first mainly because it matched the ambient genre, even though the profile also asked for very high energy and a sad mood. The model also depends heavily on a small catalog, so some users may keep seeing the same songs repeatedly. Because the system only uses labels and numeric audio features, it ignores lyrics, storytelling, and cultural context, which matter a lot for songs like J. Cole, Kendrick Lamar, Bob Dylan, and Joan Baez.

---

## 7. Evaluation

I tested five profiles: Reflective Hip-Hop / Folk, High-Energy Pop, Chill Lofi, Deep Intense Rock, and a Contradictory Edge Case. Most of the results felt right: `4 Your Eyez Only` ranked first for the reflective profile, `Sunrise City` ranked first for the pop profile, `Library Rain` ranked first for chill lofi, and `Storm Runner` ranked first for intense rock. The most surprising result came from the contradictory profile, where `Spacewalk Thoughts` still ranked first even though its calm energy does not fully match the profile's high-energy target. I also ran an experiment where I halved the genre weight and doubled the energy weight for the High-Energy Pop profile. That change kept the same top songs but made raw energy matter more, which pushed songs like `Night Drive Loop` and `Storm Runner` closer to the top.

---

## 8. Future Work

I would improve the model by adding more songs and more kinds of genres so the output is less repetitive. I would also let users have multiple moods or shifting tastes over time instead of a single fixed profile. Another improvement would be to add lyric themes or diversity rules so the system does not always recommend the closest numerical match.

---

## 9. Personal Reflection

This project made recommendation systems feel much more understandable because I could see how a few weights and feature comparisons turn into a ranked list. What surprised me most was that a scoring rule can look reasonable on normal profiles but still behave strangely on edge cases. It also reminded me that human judgment still matters, because musical taste depends on memory, lyrics, and context in ways that a simple scoring formula cannot fully capture.

---

## 10. RAG, retrieval, and extended system (final project)

The final artifact adds **natural-language prompts**, **iTunes Search API** retrieval merged with the CSV, **sentence-transformer** embeddings (with lexical fallback), optional **strict artist** filtering, **follow-up** parsing, and a lightweight **agentic** self-check (**confidence score** + clarifying question). Songs are still scored with the **same explainable** feature engine; retrieval changes **which candidates** enter the shortlist.

---

## 11. AI collaboration (how I built this responsibly)

I used **AI coding assistants** to speed up boilerplate, sketch tests, and explore edge cases (for example SSL failures and Streamlit import paths). I treated suggestions as **drafts**: I still **read diffs**, ran **`pytest`**, and **reproduced bugs** in the real app—especially around “only artist” prompts and follow-up questions. Documentation here states **limits** explicitly so tooling does not blur accountability for what was verified versus assumed.

---

## 12. Biases and risks (including RAG)

Beyond the original catalog biases, **external retrieval** inherits **Apple/iTunes** catalog skew (commercial availability, region). **Embeddings** can favor phrasing similar to pretraining and may miss emotional nuance. **Intent rules** map words to moods/genres imperfectly. **Strict artist** mode can return few tracks if the merged pool is thin. **Logs and tests** catch many failures but do not fix representation gaps in metadata.

---

## 13. Testing results (summary)

**Automated:** `pytest` — **17** tests covering CSV scoring, prompt profiles, hybrid retrieval with a **stub** embedder, mocked iTunes payloads, artist prioritization, follow-up directives, question-style follow-ups, strict-artist enrichment, and agentic low-confidence behavior. Tests do **not** assert exact live iTunes titles. **Human:** manual Streamlit/CLI runs drove fixes later encoded as regression tests. **Confidence:** a **heuristic** self-score to surface uncertainty—not a calibrated accuracy metric.

---
