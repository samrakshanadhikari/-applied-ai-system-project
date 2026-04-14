"""CLI runner for evaluating the music recommender with multiple profiles."""

from src.recommender import recommend_songs, load_songs


USER_PROFILES = {
    "Reflective Hip-Hop / Folk": {
        "favorite_genre": "hip-hop",
        "secondary_genre": "folk",
        "favorite_mood": "reflective",
        "target_energy": 0.40,
        "target_tempo_bpm": 85,
        "target_valence": 0.35,
        "target_danceability": 0.45,
        "target_acousticness": 0.70,
    },
    "High-Energy Pop": {
        "favorite_genre": "pop",
        "secondary_genre": "indie pop",
        "favorite_mood": "happy",
        "target_energy": 0.85,
        "target_tempo_bpm": 125,
        "target_valence": 0.82,
        "target_danceability": 0.82,
        "target_acousticness": 0.20,
    },
    "Chill Lofi": {
        "favorite_genre": "lofi",
        "secondary_genre": "ambient",
        "favorite_mood": "chill",
        "target_energy": 0.35,
        "target_tempo_bpm": 75,
        "target_valence": 0.60,
        "target_danceability": 0.55,
        "target_acousticness": 0.80,
    },
    "Deep Intense Rock": {
        "favorite_genre": "rock",
        "secondary_genre": "synthwave",
        "favorite_mood": "intense",
        "target_energy": 0.90,
        "target_tempo_bpm": 145,
        "target_valence": 0.45,
        "target_danceability": 0.62,
        "target_acousticness": 0.12,
    },
    "Contradictory Edge Case": {
        "favorite_genre": "ambient",
        "secondary_genre": "pop",
        "favorite_mood": "sad",
        "target_energy": 0.90,
        "target_tempo_bpm": 70,
        "target_valence": 0.15,
        "target_danceability": 0.35,
        "target_acousticness": 0.85,
    },
}

EXPERIMENTAL_WEIGHTS = {
    "favorite_genre": 1.0,
    "secondary_genre": 0.5,
    "target_energy": 4.0,
}


def print_recommendations(label: str, recommendations) -> None:
    """Print one profile's ranked recommendations in a readable block."""
    print(f"\n=== {label} ===")
    for index, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"{index}. {song['title']} by {song['artist']} | Score: {score:.2f}")
        print(f"   Because: {explanation}")


def main() -> None:
    """Run the CLI evaluation workflow for multiple recommendation profiles."""
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    for label, user_prefs in USER_PROFILES.items():
        recommendations = recommend_songs(user_prefs, songs, k=5)
        print_recommendations(label, recommendations)

    experimental_profile = USER_PROFILES["High-Energy Pop"]
    experimental_results = recommend_songs(
        experimental_profile,
        songs,
        k=5,
        weights=EXPERIMENTAL_WEIGHTS,
    )
    print_recommendations(
        "Experiment: High-Energy Pop with Genre Halved and Energy Doubled",
        experimental_results,
    )


if __name__ == "__main__":
    main()
