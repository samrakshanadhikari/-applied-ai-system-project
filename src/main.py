"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 
    print(f"Loaded songs: {len(songs)}")

    # Step 2 user profile: a reflective, lower-energy listener profile
    user_prefs = {
        "favorite_genre": "hip-hop",
        "secondary_genre": "folk",
        "favorite_mood": "reflective",
        "target_energy": 0.40,
        "target_tempo_bpm": 85,
        "target_valence": 0.35,
        "target_danceability": 0.45,
        "target_acousticness": 0.70,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()


if __name__ == "__main__":
    main()
