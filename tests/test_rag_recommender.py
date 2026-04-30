import json

from src.rag_recommender import (
    agentic_recommend_from_prompt,
    build_profile_from_prompt,
    compose_follow_up_assistant_reply,
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


def test_build_profile_from_prompt_accepts_dynamic_artist_hint():
    profile = build_profile_from_prompt(
        "I want deep songs please.",
        artist_hint={"artist": "Kendrick Lamar", "genre": "hip-hop", "mood": "reflective"},
    )

    assert profile["favorite_genre"] == "hip-hop"
    assert profile["target_artist"] == "Kendrick Lamar"
    assert profile["favorite_mood"] == "reflective"


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


def test_recommend_from_prompt_prioritizes_requested_artist():
    local_songs = [
        {
            "id": 1,
            "title": "FEAR.",
            "artist": "Kendrick Lamar",
            "genre": "hip-hop",
            "mood": "anxious",
            "energy": 0.44,
            "tempo_bpm": 79.0,
            "valence": 0.30,
            "danceability": 0.58,
            "acousticness": 0.52,
        },
        {
            "id": 2,
            "title": "Sunrise City",
            "artist": "Neon Echo",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.82,
            "tempo_bpm": 118.0,
            "valence": 0.84,
            "danceability": 0.79,
            "acousticness": 0.18,
        },
    ]

    recommendations, diagnostics = recommend_from_prompt(
        "I want deep kendrick lamar songs please",
        local_songs,
        k=1,
        use_external_retrieval=False,
        prefer_semantic_retrieval=False,
    )

    assert recommendations[0][0]["artist"] == "Kendrick Lamar"
    assert "requested artist match" in recommendations[0][2]
    assert diagnostics["artist_match_hint"]["artist"] == "Kendrick Lamar"


def test_strict_artist_prompt_fetches_artist_catalog_boost(monkeypatch):
    import src.rag_recommender as rr

    def fake_fetch(query, limit=30, timeout_seconds=8, urlopen_fn=None):
        _ = limit, timeout_seconds, urlopen_fn
        if str(query).strip().lower() == "eminem":
            return [
                {
                    "id": 201,
                    "title": "Mockingbird",
                    "artist": "Eminem",
                    "genre": "hip-hop",
                    "mood": "sad",
                    "energy": 0.50,
                    "tempo_bpm": 90.0,
                    "valence": 0.30,
                    "danceability": 0.60,
                    "acousticness": 0.40,
                    "source": "itunes",
                },
                {
                    "id": 202,
                    "title": "Stan",
                    "artist": "Eminem",
                    "genre": "hip-hop",
                    "mood": "reflective",
                    "energy": 0.55,
                    "tempo_bpm": 88.0,
                    "valence": 0.25,
                    "danceability": 0.58,
                    "acousticness": 0.35,
                    "source": "itunes",
                },
            ]
        return []

    monkeypatch.setattr(rr, "fetch_public_songs", fake_fetch)

    local_songs = [
        {
            "id": 1,
            "title": "Never Love Again",
            "artist": "Eminem",
            "genre": "hip-hop",
            "mood": "heartbroken",
            "energy": 0.57,
            "tempo_bpm": 90.0,
            "valence": 0.18,
            "danceability": 0.61,
            "acousticness": 0.33,
        },
    ]

    recommendations, diagnostics = rr.recommend_from_prompt(
        "I want only Eminem emotional songs",
        local_songs,
        k=3,
        use_external_retrieval=True,
        prefer_semantic_retrieval=False,
    )

    assert diagnostics["external_primary_fetch_size"] == 0
    assert diagnostics["external_artist_boost_size"] == 2
    assert diagnostics["external_catalog_size"] == 2
    assert len(recommendations) == 3
    assert {rec[0]["artist"] for rec in recommendations} == {"Eminem"}
    assert {rec[0]["title"] for rec in recommendations} == {"Never Love Again", "Mockingbird", "Stan"}


def test_follow_up_directives_influence_ranking_and_explanations():
    local_songs = [
        {
            "id": 1,
            "title": "Intro",
            "artist": "J. Cole",
            "genre": "hip-hop",
            "mood": "reflective",
            "energy": 0.55,
            "tempo_bpm": 100.0,
            "valence": 0.45,
            "danceability": 0.58,
            "acousticness": 0.40,
        },
        {
            "id": 2,
            "title": "She's Mine",
            "artist": "J. Cole",
            "genre": "hip-hop",
            "mood": "reflective",
            "energy": 0.50,
            "tempo_bpm": 95.0,
            "valence": 0.55,
            "danceability": 0.54,
            "acousticness": 0.52,
        },
        {
            "id": 3,
            "title": "Middle Child",
            "artist": "J. Cole",
            "genre": "hip-hop",
            "mood": "intense",
            "energy": 0.75,
            "tempo_bpm": 120.0,
            "valence": 0.50,
            "danceability": 0.72,
            "acousticness": 0.20,
        },
    ]

    recommendations, diagnostics = recommend_from_prompt(
        "I need J cole love songs",
        local_songs,
        k=3,
        use_external_retrieval=False,
        prefer_semantic_retrieval=False,
        follow_up_text="I don't think Intro is his love song. Better songs might be She's Mine.",
    )

    assert recommendations[0][0]["title"] == "She's Mine"
    assert "follow-up preference" in recommendations[0][2]
    assert diagnostics["follow_up_directives"]["avoid_titles"] == ["intro"]
    assert "she s mine" in diagnostics["follow_up_directives"]["prefer_titles"]


def test_question_style_follow_up_does_not_add_forced_preference():
    local_songs = [
        {
            "id": 1,
            "title": "Never Love Again",
            "artist": "Eminem",
            "genre": "hip-hop",
            "mood": "heartbroken",
            "energy": 0.57,
            "tempo_bpm": 90.0,
            "valence": 0.18,
            "danceability": 0.61,
            "acousticness": 0.33,
        },
        {
            "id": 2,
            "title": "Mockingbird",
            "artist": "Eminem",
            "genre": "hip-hop",
            "mood": "reflective",
            "energy": 0.40,
            "tempo_bpm": 84.0,
            "valence": 0.32,
            "danceability": 0.57,
            "acousticness": 0.50,
        },
    ]

    recommendations, diagnostics = recommend_from_prompt(
        "I need eminem emotional songs to listen.",
        local_songs,
        k=2,
        use_external_retrieval=False,
        prefer_semantic_retrieval=False,
        follow_up_text="Do you only have Never love again song in iTunes API?",
    )

    assert len(recommendations) == 2
    assert diagnostics["follow_up_directives"]["prefer_titles"] == []
    assert diagnostics["follow_up_directives"]["avoid_titles"] == []
    assert diagnostics["follow_up_directives"]["prefer_terms"] == []
    assert diagnostics["follow_up_directives"]["treated_as_question_only"] is True
    assert diagnostics["derived_profile"].get("strict_artist_match", False) is False
    assert all("follow-up preference" not in entry[2] for entry in recommendations)


def test_compose_follow_up_assistant_reply_includes_actions_and_list_delta():
    reply = compose_follow_up_assistant_reply(
        "skip Intro, prefer She's Mine",
        {
            "avoid_titles": ["intro"],
            "prefer_titles": ["she s mine"],
            "prefer_terms": ["skip", "prefer", "she", "mine"],
            "treated_as_question_only": False,
        },
        top_titles_before=["Intro", "Middle Child"],
        top_titles_after=["She's Mine", "Middle Child"],
    )
    assert "intro" in reply.lower()
    assert "she s mine" in reply.lower() or "mine" in reply.lower()
    assert "intro" in reply or "Intro" in reply
    assert "went from" in reply.lower() or "top list" in reply.lower()


def test_compose_follow_up_assistant_reply_handles_question_mode():
    reply = compose_follow_up_assistant_reply(
        "Do you only have that one song?",
        {"avoid_titles": [], "prefer_titles": [], "prefer_terms": [], "treated_as_question_only": True},
        top_titles_before=["A"],
        top_titles_after=["A"],
    )
    assert "question" in reply.lower()


def test_profile_marks_strict_artist_when_explicitly_requested():
    profile = build_profile_from_prompt(
        "I want only Eminem songs please.",
        artist_hint={"artist": "Eminem", "genre": "hip-hop", "mood": "reflective"},
    )
    assert profile["strict_artist_match"] is True


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
