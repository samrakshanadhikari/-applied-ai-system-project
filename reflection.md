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
