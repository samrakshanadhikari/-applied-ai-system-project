"""Prompt-driven RAG and agentic helpers for the music recommender."""

from __future__ import annotations

import json
import importlib
import logging
import math
import re
from typing import Any, Callable, Dict, List, Tuple
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from src.recommender import recommend_songs


LOGGER = logging.getLogger(__name__)
ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_EMBEDDER_CACHE: Dict[str, Any] = {}

GENRE_KEYWORDS = {
    "pop": "pop",
    "rock": "rock",
    "metal": "rock",
    "hiphop": "hip-hop",
    "hip": "hip-hop",
    "rap": "hip-hop",
    "lofi": "lofi",
    "ambient": "ambient",
    "folk": "folk",
    "indie": "indie pop",
    "jazz": "jazz",
    "electronic": "synthwave",
    "synthwave": "synthwave",
}

MOOD_KEYWORDS = {
    "happy": "happy",
    "uplifting": "happy",
    "chill": "chill",
    "calm": "chill",
    "focus": "chill",
    "study": "chill",
    "intense": "intense",
    "angry": "intense",
    "sad": "sad",
    "heartbroken": "sad",
    "reflective": "reflective",
    "melancholy": "reflective",
}

LOW_ENERGY_WORDS = {"calm", "slow", "soft", "chill", "quiet", "sleep"}
HIGH_ENERGY_WORDS = {"hype", "workout", "gym", "fast", "party", "energy", "energetic"}
ACOUSTIC_WORDS = {"acoustic", "unplugged", "raw"}
DANCE_WORDS = {"dance", "party", "club"}
ELECTRONIC_WORDS = {"electronic", "edm", "synth", "techno"}
LYRICAL_WORDS = {"lyrical", "poetic", "storytelling", "conscious", "genius", "bars", "lyrics"}

GENRE_DEFAULTS = {
    "pop": {"mood": "happy", "energy": 0.80, "tempo_bpm": 122, "valence": 0.80, "danceability": 0.82, "acousticness": 0.20},
    "rock": {"mood": "intense", "energy": 0.88, "tempo_bpm": 142, "valence": 0.45, "danceability": 0.55, "acousticness": 0.10},
    "hip-hop": {"mood": "reflective", "energy": 0.65, "tempo_bpm": 95, "valence": 0.50, "danceability": 0.75, "acousticness": 0.25},
    "lofi": {"mood": "chill", "energy": 0.35, "tempo_bpm": 76, "valence": 0.58, "danceability": 0.55, "acousticness": 0.78},
    "ambient": {"mood": "chill", "energy": 0.28, "tempo_bpm": 64, "valence": 0.60, "danceability": 0.40, "acousticness": 0.86},
    "folk": {"mood": "reflective", "energy": 0.45, "tempo_bpm": 88, "valence": 0.48, "danceability": 0.40, "acousticness": 0.82},
    "indie pop": {"mood": "happy", "energy": 0.72, "tempo_bpm": 112, "valence": 0.72, "danceability": 0.70, "acousticness": 0.32},
    "jazz": {"mood": "chill", "energy": 0.42, "tempo_bpm": 92, "valence": 0.62, "danceability": 0.52, "acousticness": 0.72},
    "synthwave": {"mood": "intense", "energy": 0.74, "tempo_bpm": 116, "valence": 0.60, "danceability": 0.66, "acousticness": 0.12},
}


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _derive_genres(tokens: List[str]) -> List[str]:
    genres: List[str] = []
    for token in tokens:
        genre = GENRE_KEYWORDS.get(token)
        if genre and genre not in genres:
            genres.append(genre)
    return genres


def _derive_mood(tokens: List[str]) -> str:
    for token in tokens:
        if token in MOOD_KEYWORDS:
            return MOOD_KEYWORDS[token]
    return "chill"


def build_profile_from_prompt(prompt: str) -> Dict[str, Any]:
    """Build user preference targets from natural-language prompt text."""
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        raise ValueError("Prompt must not be empty.")

    tokens = _tokenize(cleaned_prompt)
    genres = _derive_genres(tokens)
    mood = _derive_mood(tokens)

    has_low_energy = any(token in LOW_ENERGY_WORDS for token in tokens)
    has_high_energy = any(token in HIGH_ENERGY_WORDS for token in tokens)
    has_acoustic = any(token in ACOUSTIC_WORDS for token in tokens)
    has_dance = any(token in DANCE_WORDS for token in tokens)
    has_electronic = any(token in ELECTRONIC_WORDS for token in tokens)
    has_lyrical = any(token in LYRICAL_WORDS for token in tokens)

    target_energy = 0.55
    if has_high_energy and not has_low_energy:
        target_energy = 0.86
    elif has_low_energy and not has_high_energy:
        target_energy = 0.30

    target_tempo_bpm = 100
    if target_energy >= 0.8:
        target_tempo_bpm = 128
    elif target_energy <= 0.35:
        target_tempo_bpm = 74

    target_valence = {
        "happy": 0.82,
        "chill": 0.60,
        "intense": 0.45,
        "sad": 0.20,
        "reflective": 0.35,
    }.get(mood, 0.50)
    target_danceability = 0.78 if has_dance else (0.42 if has_low_energy else 0.60)
    target_acousticness = 0.80 if has_acoustic else (0.20 if has_electronic else 0.50)

    favorite_genre = genres[0] if genres else "pop"
    if has_lyrical and mood == "chill":
        mood = "reflective"
        target_valence = min(target_valence, 0.40)
        target_acousticness = max(target_acousticness, 0.60)
        target_danceability = min(target_danceability, 0.60)

    profile: Dict[str, Any] = {
        "favorite_genre": favorite_genre,
        "favorite_mood": mood,
        "target_energy": target_energy,
        "target_tempo_bpm": target_tempo_bpm,
        "target_valence": target_valence,
        "target_danceability": target_danceability,
        "target_acousticness": target_acousticness,
    }
    if len(genres) > 1:
        profile["secondary_genre"] = genres[1]
    return profile


def _build_search_query(prompt: str, profile: Dict[str, Any]) -> str:
    prompt_tokens = _tokenize(prompt)
    trimmed_prompt = " ".join(prompt_tokens[:8])
    return f"{trimmed_prompt} {profile['favorite_genre']} {profile['favorite_mood']}".strip()


def _song_to_document(song: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(song.get("title", "")),
            str(song.get("artist", "")),
            str(song.get("genre", "")),
            str(song.get("mood", "")),
        ]
    ).strip()


def _guess_features_from_genre(genre: str) -> Dict[str, Any]:
    lowered_genre = genre.lower()
    for known_genre, defaults in GENRE_DEFAULTS.items():
        if known_genre in lowered_genre:
            return defaults
    return {
        "mood": "chill",
        "energy": 0.55,
        "tempo_bpm": 100,
        "valence": 0.55,
        "danceability": 0.60,
        "acousticness": 0.50,
    }


def _normalize_itunes_song(track: Dict[str, Any], song_id: int) -> Dict[str, Any]:
    title = (track.get("trackName") or track.get("collectionName") or "Unknown Track").strip()
    artist = (track.get("artistName") or "Unknown Artist").strip()
    genre = (track.get("primaryGenreName") or "unknown").strip().lower()
    defaults = _guess_features_from_genre(genre)

    return {
        "id": song_id,
        "title": title,
        "artist": artist,
        "genre": genre,
        "mood": defaults["mood"],
        "energy": float(defaults["energy"]),
        "tempo_bpm": float(defaults["tempo_bpm"]),
        "valence": float(defaults["valence"]),
        "danceability": float(defaults["danceability"]),
        "acousticness": float(defaults["acousticness"]),
        "source": "itunes",
    }


def fetch_public_songs(
    query: str,
    limit: int = 30,
    timeout_seconds: int = 8,
    urlopen_fn: Callable[..., Any] = urlopen,
) -> List[Dict[str, Any]]:
    """Fetch songs from a public source (iTunes Search API)."""
    if not query.strip():
        return []

    params = {"term": query, "entity": "song", "limit": min(max(limit, 1), 200)}
    request_url = f"{ITUNES_SEARCH_URL}?{urlencode(params)}"

    try:
        with urlopen_fn(request_url, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        LOGGER.warning("External retrieval failed for query '%s': %s", query, exc)
        return []

    raw_results = payload.get("results", [])
    normalized_results: List[Dict[str, Any]] = []
    for index, track in enumerate(raw_results, start=1):
        if "trackName" not in track and "collectionName" not in track:
            continue
        normalized_results.append(_normalize_itunes_song(track, song_id=200_000 + index))
    return normalized_results


def merge_song_catalogs(local_songs: List[Dict[str, Any]], external_songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge song lists while deduplicating by title and artist."""
    merged = list(local_songs)
    seen_pairs = {(song.get("title", "").lower(), song.get("artist", "").lower()) for song in local_songs}

    for song in external_songs:
        key = (song.get("title", "").lower(), song.get("artist", "").lower())
        if key not in seen_pairs:
            merged.append(song)
            seen_pairs.add(key)
    return merged


def _vector_to_list(vector: Any) -> List[float]:
    if hasattr(vector, "tolist"):
        converted = vector.tolist()
        if isinstance(converted, list):
            return [float(item) for item in converted]
    if isinstance(vector, list):
        return [float(item) for item in vector]
    return [float(item) for item in vector]


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(left[index] * right[index] for index in range(len(left)))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any | None:
    if model_name in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[model_name]

    try:
        sentence_transformers_module = importlib.import_module("sentence_transformers")
        sentence_transformer_class = getattr(sentence_transformers_module, "SentenceTransformer")
    except (ImportError, AttributeError):
        LOGGER.info("sentence-transformers not installed; using lexical retrieval fallback.")
        return None

    try:
        embedder = sentence_transformer_class(model_name)
        _EMBEDDER_CACHE[model_name] = embedder
        return embedder
    except Exception as exc:  # pragma: no cover - depends on local environment/network
        LOGGER.warning("Embedding model '%s' unavailable: %s", model_name, exc)
        return None


def _semantic_similarity_scores(
    prompt: str,
    songs: List[Dict[str, Any]],
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedder: Any | None = None,
) -> Tuple[Dict[int, float], bool]:
    active_embedder = embedder or _get_embedder(embedding_model_name)
    if active_embedder is None:
        return {}, False

    try:
        prompt_vector_raw = active_embedder.encode([prompt], normalize_embeddings=True)[0]
        song_documents = [_song_to_document(song) for song in songs]
        song_vectors_raw = active_embedder.encode(song_documents, normalize_embeddings=True)
    except Exception as exc:  # pragma: no cover - model/runtime dependent
        LOGGER.warning("Semantic retrieval fallback due to embedding error: %s", exc)
        return {}, False

    prompt_vector = _vector_to_list(prompt_vector_raw)
    scores: Dict[int, float] = {}
    for index, song_vector_raw in enumerate(song_vectors_raw):
        song_vector = _vector_to_list(song_vector_raw)
        cosine = _cosine_similarity(prompt_vector, song_vector)
        # Scale cosine from [-1, 1] to [0, 1] for weighted fusion.
        scores[index] = max(0.0, min(1.0, (cosine + 1.0) / 2.0))
    return scores, True


def _lexical_similarity_scores(prompt: str, songs: List[Dict[str, Any]]) -> Dict[int, float]:
    prompt_tokens = set(_tokenize(prompt))
    if not prompt_tokens:
        return {}

    scores: Dict[int, float] = {}
    for index, song in enumerate(songs):
        searchable_text = _song_to_document(song).lower()
        song_tokens = set(_tokenize(searchable_text))
        overlap_ratio = len(prompt_tokens & song_tokens) / max(len(prompt_tokens), 1)
        phrase_bonus = 0.3 if any(token in searchable_text for token in prompt_tokens) else 0.0
        scores[index] = overlap_ratio + phrase_bonus
    return scores


def _rank_candidate_indexes(
    prompt: str,
    songs: List[Dict[str, Any]],
    limit: int = 30,
    prefer_semantic_retrieval: bool = True,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedder: Any | None = None,
) -> Tuple[List[int], Dict[str, Any]]:
    lexical_scores = _lexical_similarity_scores(prompt, songs)
    semantic_scores: Dict[int, float] = {}
    semantic_available = False
    if prefer_semantic_retrieval:
        semantic_scores, semantic_available = _semantic_similarity_scores(
            prompt,
            songs,
            embedding_model_name=embedding_model_name,
            embedder=embedder,
        )

    scored_indexes: List[Tuple[float, int]] = []
    for index in range(len(songs)):
        lexical_score = lexical_scores.get(index, 0.0)
        semantic_score = semantic_scores.get(index, 0.0)
        if semantic_available:
            combined_score = (0.75 * semantic_score) + (0.25 * lexical_score)
        else:
            combined_score = lexical_score
        if combined_score > 0.0:
            scored_indexes.append((combined_score, index))

    if not scored_indexes:
        top_indexes = list(range(min(limit, len(songs))))
    else:
        scored_indexes.sort(key=lambda item: item[0], reverse=True)
        top_indexes = [index for _, index in scored_indexes[:limit]]

    diagnostics = {
        "retrieval_strategy": "semantic+lexical" if semantic_available else "lexical",
        "semantic_model_used": semantic_available,
        "embedding_model_name": embedding_model_name if semantic_available else None,
    }
    return top_indexes, diagnostics


def retrieve_candidates(
    prompt: str,
    songs: List[Dict[str, Any]],
    limit: int = 30,
    prefer_semantic_retrieval: bool = True,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedder: Any | None = None,
) -> List[Dict[str, Any]]:
    """RAG retrieval stage: shortlist songs relevant to the prompt."""
    if not songs:
        return []

    ranked_indexes, _ = _rank_candidate_indexes(
        prompt,
        songs,
        limit=limit,
        prefer_semantic_retrieval=prefer_semantic_retrieval,
        embedding_model_name=embedding_model_name,
        embedder=embedder,
    )
    return [songs[index] for index in ranked_indexes]


def _run_retrieval_and_ranking(
    prompt: str,
    profile: Dict[str, Any],
    local_songs: List[Dict[str, Any]],
    k: int = 5,
    use_external_retrieval: bool = True,
    prefer_semantic_retrieval: bool = True,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedder: Any | None = None,
) -> Tuple[List[Tuple[Dict[str, Any], float, str]], Dict[str, Any]]:
    external_songs: List[Dict[str, Any]] = []
    if use_external_retrieval:
        query = _build_search_query(prompt, profile)
        external_songs = fetch_public_songs(query, limit=40)

    full_catalog = merge_song_catalogs(local_songs, external_songs)
    candidate_indexes, retrieval_diagnostics = _rank_candidate_indexes(
        prompt,
        full_catalog,
        limit=max(20, k * 4),
        prefer_semantic_retrieval=prefer_semantic_retrieval,
        embedding_model_name=embedding_model_name,
        embedder=embedder,
    )
    retrieved_candidates = [full_catalog[index] for index in candidate_indexes]
    recommendations = recommend_songs(profile, retrieved_candidates, k=k)

    diagnostics = {
        "local_catalog_size": len(local_songs),
        "external_catalog_size": len(external_songs),
        "retrieved_candidate_size": len(retrieved_candidates),
        **retrieval_diagnostics,
    }
    return recommendations, diagnostics


def recommend_from_prompt(
    prompt: str,
    local_songs: List[Dict[str, Any]],
    k: int = 5,
    use_external_retrieval: bool = True,
    prefer_semantic_retrieval: bool = True,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedder: Any | None = None,
) -> Tuple[List[Tuple[Dict[str, Any], float, str]], Dict[str, Any]]:
    """Run prompt -> retrieval -> ranking -> explanation pipeline."""
    profile = build_profile_from_prompt(prompt)
    recommendations, diagnostics = _run_retrieval_and_ranking(
        prompt,
        profile,
        local_songs,
        k=k,
        use_external_retrieval=use_external_retrieval,
        prefer_semantic_retrieval=prefer_semantic_retrieval,
        embedding_model_name=embedding_model_name,
        embedder=embedder,
    )
    diagnostics["derived_profile"] = profile
    return recommendations, diagnostics


def _confidence_score(recommendations: List[Tuple[Dict[str, Any], float, str]], diagnostics: Dict[str, Any]) -> float:
    if not recommendations:
        return 0.0

    top_score = recommendations[0][1]
    second_score = recommendations[1][1] if len(recommendations) > 1 else max(0.0, top_score - 1.0)
    score_gap = max(0.0, top_score - second_score)
    candidate_ratio = diagnostics["retrieved_candidate_size"] / max(
        diagnostics["local_catalog_size"] + diagnostics["external_catalog_size"], 1
    )

    confidence = (
        min(top_score / 10.0, 1.0) * 0.55
        + min(score_gap / 2.0, 1.0) * 0.30
        + min(candidate_ratio, 1.0) * 0.15
    )
    return round(min(max(confidence, 0.0), 1.0), 3)


def _generate_follow_up_question(prompt: str) -> str:
    tokens = set(_tokenize(prompt))
    if not any(token in MOOD_KEYWORDS for token in tokens):
        return "Should the vibe be more chill, intense, or reflective?"
    if not any(token in LOW_ENERGY_WORDS or token in HIGH_ENERGY_WORDS for token in tokens):
        return "Do you want low-energy songs or high-energy songs?"
    return "Do you prefer more acoustic tracks or more danceable tracks?"


def agentic_recommend_from_prompt(
    prompt: str,
    local_songs: List[Dict[str, Any]],
    k: int = 5,
    use_external_retrieval: bool = True,
    follow_up_answer: str | None = None,
    confidence_threshold: float = 0.70,
    prefer_semantic_retrieval: bool = True,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedder: Any | None = None,
) -> Tuple[List[Tuple[Dict[str, Any], float, str]], Dict[str, Any]]:
    """Run a simple plan->retrieve->rank->self-check agentic workflow."""
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        raise ValueError("Prompt must not be empty.")

    combined_prompt = cleaned_prompt
    if follow_up_answer:
        combined_prompt = f"{cleaned_prompt}. {follow_up_answer.strip()}"

    plan_notes: List[str] = [
        "Parse user intent into a recommendation profile.",
        "Retrieve candidate songs from local and optional external sources.",
        "Rank candidates and self-check recommendation confidence.",
    ]
    if follow_up_answer:
        plan_notes.append("Incorporate follow-up clarification from the user.")

    profile = build_profile_from_prompt(combined_prompt)
    recommendations, diagnostics = _run_retrieval_and_ranking(
        combined_prompt,
        profile,
        local_songs,
        k=k,
        use_external_retrieval=use_external_retrieval,
        prefer_semantic_retrieval=prefer_semantic_retrieval,
        embedding_model_name=embedding_model_name,
        embedder=embedder,
    )
    confidence = _confidence_score(recommendations, diagnostics)

    follow_up_question = None
    if confidence < confidence_threshold and not follow_up_answer:
        follow_up_question = _generate_follow_up_question(combined_prompt)

    diagnostics.update(
        {
            "derived_profile": profile,
            "agentic_plan": plan_notes,
            "confidence_score": confidence,
            "follow_up_question": follow_up_question,
            "confidence_threshold": confidence_threshold,
        }
    )
    return recommendations, diagnostics
