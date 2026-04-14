import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs ranked for the given user."""
        scored_songs = sorted(
            self.songs,
            key=lambda song: score_song_from_profile(
                {
                    "favorite_genre": user.favorite_genre,
                    "favorite_mood": user.favorite_mood,
                    "target_energy": user.target_energy,
                    "target_acousticness": 0.8 if user.likes_acoustic else 0.2,
                },
                {
                    "genre": song.genre,
                    "mood": song.mood,
                    "energy": song.energy,
                    "acousticness": song.acousticness,
                },
            )[0],
            reverse=True,
        )
        return scored_songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a short explanation for why a song matches the user."""
        _, reasons = score_song_from_profile(
            {
                "favorite_genre": user.favorite_genre,
                "favorite_mood": user.favorite_mood,
                "target_energy": user.target_energy,
                "target_acousticness": 0.8 if user.likes_acoustic else 0.2,
            },
            {
                "genre": song.genre,
                "mood": song.mood,
                "energy": song.energy,
                "acousticness": song.acousticness,
            },
        )
        return "; ".join(reasons)


def _closeness_score(song_value: float, target_value: float, weight: float) -> float:
    """Convert numeric similarity into weighted recommendation points."""
    similarity = max(0.0, 1.0 - abs(song_value - target_value))
    return similarity * weight


def _tempo_score(song_tempo: float, target_tempo: float, weight: float) -> float:
    """Score tempo closeness using a wider BPM distance scale."""
    similarity = max(0.0, 1.0 - abs(song_tempo - target_tempo) / 100.0)
    return similarity * weight


def score_song_from_profile(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against a profile and collect matching reasons."""
    score = 0.0
    reasons: List[str] = []

    favorite_genre = user_prefs.get("favorite_genre")
    secondary_genre = user_prefs.get("secondary_genre")
    favorite_mood = user_prefs.get("favorite_mood")

    if favorite_genre and song.get("genre") == favorite_genre:
        score += 2.0
        reasons.append("genre match (+2.0)")
    elif secondary_genre and song.get("genre") == secondary_genre:
        score += 1.0
        reasons.append("secondary genre match (+1.0)")

    if favorite_mood and song.get("mood") == favorite_mood:
        score += 1.5
        reasons.append("mood match (+1.5)")

    if "target_energy" in user_prefs and "energy" in song:
        energy_points = _closeness_score(song["energy"], user_prefs["target_energy"], 2.0)
        score += energy_points
        reasons.append(f"energy similarity (+{energy_points:.2f})")

    if "target_tempo_bpm" in user_prefs and "tempo_bpm" in song:
        tempo_points = _tempo_score(song["tempo_bpm"], user_prefs["target_tempo_bpm"], 1.5)
        score += tempo_points
        reasons.append(f"tempo similarity (+{tempo_points:.2f})")

    if "target_valence" in user_prefs and "valence" in song:
        valence_points = _closeness_score(song["valence"], user_prefs["target_valence"], 1.5)
        score += valence_points
        reasons.append(f"valence similarity (+{valence_points:.2f})")

    if "target_danceability" in user_prefs and "danceability" in song:
        danceability_points = _closeness_score(
            song["danceability"], user_prefs["target_danceability"], 1.0
        )
        score += danceability_points
        reasons.append(f"danceability similarity (+{danceability_points:.2f})")

    if "target_acousticness" in user_prefs and "acousticness" in song:
        acousticness_points = _closeness_score(
            song["acousticness"], user_prefs["target_acousticness"], 1.0
        )
        score += acousticness_points
        reasons.append(f"acousticness similarity (+{acousticness_points:.2f})")

    return score, reasons


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Return a song's score and explanation list for a user profile."""
    return score_song_from_profile(user_prefs, song)

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from CSV into dictionaries with numeric fields parsed."""
    songs: List[Dict] = []

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            songs.append(
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "artist": row["artist"],
                    "genre": row["genre"],
                    "mood": row["mood"],
                    "energy": float(row["energy"]),
                    "tempo_bpm": float(row["tempo_bpm"]),
                    "valence": float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                }
            )

    return songs

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Return the highest-scoring songs with scores and explanations."""
    scored_recommendations: List[Tuple[Dict, float, str]] = []

    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = "; ".join(reasons)
        scored_recommendations.append((song, score, explanation))

    scored_recommendations.sort(key=lambda item: item[1], reverse=True)
    return scored_recommendations[:k]
