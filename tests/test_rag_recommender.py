import json

from src.rag_recommender import (
    agentic_recommend_from_prompt,
    build_profile_from_prompt,
    fetch_public_songs,
    recommend_from_prompt,
    retrieve_candidates,
)


def test_build_profile_from_prompt_parses_user_intent():
    profile = build_profile_from_prompt("I want chill acoustic folk music for studying.")

    assert profile["favorite_genre"] == "folk"
    assert profile["favorite_mood"] == "chill"
    assert profile["target_energy"] < 0.5
    assert profile["target_acousticness"] > 0.7


def test_retrieve_candidates_prioritizes_prompt_overlap():
    songs = [
        {
            "id": 1,
            "title": "Calm Folk Evening",
            "artist": "Acoustic Poet",
            "genre": "folk",
            "mood": "chill",
            "energy": 0.30,
            "tempo_bpm": 78.0,
            "valence": 0.48,
            "danceability": 0.40,
            "acousticness": 0.88,
        },
        {
            "id": 2,
            "title": "Hard Gym Anthem",
            "artist": "Beat Rush",
            "genre": "rock",
            "mood": "intense",
            "energy": 0.95,
            "tempo_bpm": 150.0,
            "valence": 0.52,
            "danceability": 0.62,
            "acousticness": 0.10,
        },
    ]

    retrieved = retrieve_candidates("chill folk acoustic", songs, limit=2)
    assert retrieved[0]["title"] == "Calm Folk Evening"


def test_retrieve_candidates_can_use_semantic_embedder():
    songs = [
        {
            "id": 1,
            "title": "Acoustic Reflection",
            "artist": "Poet Tone",
            "genre": "folk",
            "mood": "reflective",
            "energy": 0.33,
            "tempo_bpm": 80.0,
            "valence": 0.42,
            "danceability": 0.40,
            "acousticness": 0.88,
        },
        {
            "id": 2,
            "title": "Club Voltage",
            "artist": "Night Grid",
            "genre": "electronic",
            "mood": "intense",
            "energy": 0.92,
            "tempo_bpm": 140.0,
            "valence": 0.65,
            "danceability": 0.85,
            "acousticness": 0.10,
        },
    ]

    class FakeEmbedder:
        def encode(self, inputs, normalize_embeddings=True):
            texts = inputs if isinstance(inputs, list) else [inputs]
            vectors = []
            for text in texts:
                lowered = text.lower()
                if "lyrical rap poetry" in lowered:
                    vectors.append([0.9, 0.1, 0.0])
                elif "acoustic reflection" in lowered:
                    vectors.append([0.8, 0.2, 0.1])
                else:
                    vectors.append([0.1, 0.9, 0.2])
            return vectors

    retrieved = retrieve_candidates(
        "lyrical rap poetry",
        songs,
        limit=2,
        prefer_semantic_retrieval=True,
        embedder=FakeEmbedder(),
    )
    assert retrieved[0]["title"] == "Acoustic Reflection"


def test_fetch_public_songs_handles_bad_payload():
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"{not-valid-json"

    result = fetch_public_songs("happy pop", urlopen_fn=lambda *_args, **_kwargs: FakeResponse())
    assert result == []


def test_fetch_public_songs_normalizes_public_results():
    payload = {
        "results": [
            {
                "trackName": "City Lights",
                "artistName": "Neon Shore",
                "primaryGenreName": "Pop",
            }
        ]
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    songs = fetch_public_songs("city pop", urlopen_fn=lambda *_args, **_kwargs: FakeResponse())
    assert len(songs) == 1
    assert songs[0]["title"] == "City Lights"
    assert songs[0]["genre"] == "pop"
    assert songs[0]["source"] == "itunes"


def test_recommend_from_prompt_local_mode_runs_end_to_end():
    local_songs = [
        {
            "id": 1,
            "title": "Night Study",
            "artist": "LoRoom",
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.34,
            "tempo_bpm": 74.0,
            "valence": 0.59,
            "danceability": 0.52,
            "acousticness": 0.82,
        },
        {
            "id": 2,
            "title": "Rooftop Run",
            "artist": "Max Pulse",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.88,
            "tempo_bpm": 128.0,
            "valence": 0.80,
            "danceability": 0.86,
            "acousticness": 0.12,
        },
    ]

    recommendations, diagnostics = recommend_from_prompt(
        "need calm acoustic music for focus",
        local_songs,
        k=1,
        use_external_retrieval=False,
    )

    assert len(recommendations) == 1
    assert diagnostics["external_catalog_size"] == 0
    assert diagnostics["retrieved_candidate_size"] >= 1
    assert recommendations[0][2]  # explanation string is non-empty


def test_agentic_workflow_emits_follow_up_for_low_confidence():
    local_songs = [
        {
            "id": 1,
            "title": "Unknown Tone",
            "artist": "Artist A",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.5,
            "tempo_bpm": 100.0,
            "valence": 0.5,
            "danceability": 0.5,
            "acousticness": 0.5,
        },
        {
            "id": 2,
            "title": "Random Echo",
            "artist": "Artist B",
            "genre": "rock",
            "mood": "intense",
            "energy": 0.5,
            "tempo_bpm": 100.0,
            "valence": 0.5,
            "danceability": 0.5,
            "acousticness": 0.5,
        },
    ]

    recommendations, diagnostics = agentic_recommend_from_prompt(
        "play something nice",
        local_songs,
        k=2,
        use_external_retrieval=False,
        confidence_threshold=0.95,
        prefer_semantic_retrieval=False,
    )

    assert len(recommendations) == 2
    assert diagnostics["confidence_score"] < diagnostics["confidence_threshold"]
    assert diagnostics["follow_up_question"] is not None
