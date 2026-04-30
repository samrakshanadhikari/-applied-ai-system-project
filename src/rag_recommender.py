"""Prompt-driven RAG and agentic helpers for the music recommender."""

from __future__ import annotations

import json
import importlib
import logging
import math
import re
import ssl
from typing import Any, Callable, Dict, List, Tuple
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from src.recommender import score_song


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
STRICT_ARTIST_WORDS = {"only", "just", "strictly", "specifically"}
FOLLOW_UP_STOPWORDS = {
    "the",
    "this",
    "that",
    "with",
    "from",
    "your",
    "his",
    "her",
    "their",
    "song",
    "songs",
    "better",
    "might",
    "please",
    "want",
    "need",
    "think",
    "do",
    "you",
    "have",
    "itunes",
    "api",
    "only",
}

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


def _normalize_text(value: str) -> str:
    return " ".join(_tokenize(value))


def _most_frequent_value(values: List[str]) -> str | None:
    frequency: Dict[str, int] = {}
    for value in values:
        cleaned_value = str(value).strip().lower()
        if not cleaned_value:
            continue
        frequency[cleaned_value] = frequency.get(cleaned_value, 0) + 1
    if not frequency:
        return None
    return max(frequency.items(), key=lambda item: item[1])[0]


def _build_artist_index(songs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    artist_index: Dict[str, Dict[str, Any]] = {}
    for song in songs:
        artist_name = str(song.get("artist", "")).strip()
        if not artist_name:
            continue
        artist_key = _normalize_text(artist_name)
        if not artist_key:
            continue

        entry = artist_index.setdefault(
            artist_key,
            {
                "artist": artist_name,
                "genres": [],
                "moods": [],
                "count": 0,
            },
        )
        entry["count"] += 1
        if song.get("genre"):
            entry["genres"].append(str(song["genre"]))
        if song.get("mood"):
            entry["moods"].append(str(song["mood"]))
    return artist_index


def _resolve_artist_from_catalog(prompt: str, songs: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    artist_index = _build_artist_index(songs)
    if not artist_index:
        return None

    normalized_prompt = _normalize_text(prompt)
    prompt_tokens = set(normalized_prompt.split())
    if not prompt_tokens:
        return None

    best_match: Dict[str, Any] | None = None
    best_score = 0.0
    for artist_key, metadata in artist_index.items():
        artist_tokens = set(artist_key.split())
        if not artist_tokens:
            continue

        if artist_key in normalized_prompt:
            score = 1.0
        else:
            overlap = artist_tokens & prompt_tokens
            overlap_ratio = len(overlap) / len(artist_tokens)
            longest_overlap = max((len(token) for token in overlap), default=0)
            if overlap_ratio >= 0.6:
                score = overlap_ratio
            elif overlap_ratio > 0 and longest_overlap >= 5:
                score = 0.55
            else:
                score = 0.0

        if score > best_score:
            best_score = score
            best_match = metadata

    if best_match is None or best_score < 0.55:
        return None

    return {
        "artist": best_match["artist"],
        "genre": _most_frequent_value(best_match["genres"]),
        "mood": _most_frequent_value(best_match["moods"]),
        "match_score": round(best_score, 3),
        "catalog_song_count": best_match["count"],
    }


def build_profile_from_prompt(prompt: str, artist_hint: Dict[str, Any] | None = None) -> Dict[str, Any]:
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
    if artist_hint and artist_hint.get("genre") and not genres:
        favorite_genre = str(artist_hint["genre"])
    if artist_hint and artist_hint.get("mood") and mood == "chill":
        mood = str(artist_hint["mood"])
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
    if artist_hint and artist_hint.get("artist"):
        profile["target_artist"] = str(artist_hint["artist"])
        if _is_strict_artist_request(cleaned_prompt, str(artist_hint["artist"])):
            profile["strict_artist_match"] = True
    return profile


def _is_strict_artist_request(prompt: str, artist_name: str) -> bool:
    normalized_prompt = _normalize_text(prompt)
    normalized_artist = _normalize_text(artist_name)
    if not normalized_prompt or not normalized_artist:
        return False

    strict_cues_pattern = "|".join(sorted(STRICT_ARTIST_WORDS))
    before_pattern = rf"\b({strict_cues_pattern})\b(?: \w+){{0,3}} {re.escape(normalized_artist)}\b"
    after_pattern = rf"\b{re.escape(normalized_artist)}\b(?: \w+){{0,3}} ({strict_cues_pattern})\b"
    return bool(re.search(before_pattern, normalized_prompt) or re.search(after_pattern, normalized_prompt))


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


def _build_certifi_ssl_context() -> ssl.SSLContext | None:
    """Build SSL context from certifi bundle when system certs fail."""
    try:
        certifi_module = importlib.import_module("certifi")
        certifi_path = str(certifi_module.where())
        return ssl.create_default_context(cafile=certifi_path)
    except Exception:
        return None


def _load_remote_json_payload(
    request_url: str,
    timeout_seconds: int,
    urlopen_fn: Callable[..., Any],
) -> Dict[str, Any]:
    with urlopen_fn(request_url, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


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
        payload = _load_remote_json_payload(request_url, timeout_seconds, urlopen_fn)
    except URLError as exc:
        # Common Mac/Python environment issue: missing trusted CA roots.
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            ssl_context = _build_certifi_ssl_context()
            if ssl_context is not None:
                try:
                    with urlopen_fn(request_url, timeout=timeout_seconds, context=ssl_context) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    LOGGER.info("External retrieval succeeded after certifi SSL fallback.")
                except (URLError, TimeoutError, json.JSONDecodeError) as retry_exc:
                    LOGGER.warning("External retrieval failed for query '%s': %s", query, retry_exc)
                    return []
            else:
                LOGGER.warning("External retrieval failed for query '%s': %s", query, exc)
                return []
        else:
            LOGGER.warning("External retrieval failed for query '%s': %s", query, exc)
            return []
    except (TimeoutError, json.JSONDecodeError) as exc:
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
) -> Tuple[List[int], Dict[str, Any], Dict[int, float]]:
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

    score_lookup: Dict[int, float] = {index: score for score, index in scored_indexes}

    if not scored_indexes:
        top_indexes = list(range(min(limit, len(songs))))
        score_lookup = {index: 0.0 for index in top_indexes}
    else:
        scored_indexes.sort(key=lambda item: item[0], reverse=True)
        top_indexes = [index for _, index in scored_indexes[:limit]]
        score_lookup = {index: score_lookup[index] for index in top_indexes}

    diagnostics = {
        "retrieval_strategy": "semantic+lexical" if semantic_available else "lexical",
        "semantic_model_used": semantic_available,
        "embedding_model_name": embedding_model_name if semantic_available else None,
    }
    return top_indexes, diagnostics, score_lookup


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

    ranked_indexes, _, _ = _rank_candidate_indexes(
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
    local_songs: List[Dict[str, Any]],
    k: int = 5,
    use_external_retrieval: bool = True,
    prefer_semantic_retrieval: bool = True,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedder: Any | None = None,
    follow_up_text: str | None = None,
) -> Tuple[List[Tuple[Dict[str, Any], float, str]], Dict[str, Any]]:
    effective_prompt = prompt.strip()
    if follow_up_text and follow_up_text.strip():
        effective_prompt = f"{effective_prompt}. {follow_up_text.strip()}"

    base_profile = build_profile_from_prompt(effective_prompt)
    external_songs: List[Dict[str, Any]] = []
    artist_boost_songs: List[Dict[str, Any]] = []
    if use_external_retrieval:
        query = _build_search_query(effective_prompt, base_profile)
        external_songs = fetch_public_songs(query, limit=40)

    full_catalog = merge_song_catalogs(local_songs, external_songs)
    artist_hint = _resolve_artist_from_catalog(effective_prompt, full_catalog)
    if use_external_retrieval and artist_hint and str(artist_hint.get("artist", "")).strip():
        artist_boost_songs = fetch_public_songs(str(artist_hint["artist"]).strip(), limit=100)
        if artist_boost_songs:
            full_catalog = merge_song_catalogs(full_catalog, artist_boost_songs)
            artist_hint = _resolve_artist_from_catalog(effective_prompt, full_catalog)

    profile = build_profile_from_prompt(effective_prompt, artist_hint=artist_hint)

    follow_up_directives: Dict[str, List[str]] = {"avoid_titles": [], "prefer_titles": [], "prefer_terms": []}
    if follow_up_text and follow_up_text.strip():
        follow_up_directives = _extract_follow_up_directives(follow_up_text, full_catalog)
        if follow_up_directives["avoid_titles"]:
            profile["avoid_titles"] = follow_up_directives["avoid_titles"]
        if follow_up_directives["prefer_titles"]:
            profile["prefer_titles"] = follow_up_directives["prefer_titles"]
        if follow_up_directives["prefer_terms"]:
            profile["prefer_terms"] = follow_up_directives["prefer_terms"]

    retrieval_pool: List[Dict[str, Any]] = full_catalog
    target_for_strict = str(profile.get("target_artist", "")).strip().lower()
    if profile.get("strict_artist_match") and target_for_strict:
        artist_rows = [
            song for song in full_catalog if target_for_strict in str(song.get("artist", "")).lower()
        ]
        if artist_rows:
            retrieval_pool = artist_rows

    candidate_indexes, retrieval_diagnostics, retrieval_scores = _rank_candidate_indexes(
        effective_prompt,
        retrieval_pool,
        limit=max(20, k * 4),
        prefer_semantic_retrieval=prefer_semantic_retrieval,
        embedding_model_name=embedding_model_name,
        embedder=embedder,
    )
    retrieved_candidates: List[Dict[str, Any]] = []
    for index in candidate_indexes:
        song_copy = dict(retrieval_pool[index])
        song_copy["_retrieval_score"] = float(retrieval_scores.get(index, 0.0))
        retrieved_candidates.append(song_copy)
    recommendations = _recommend_songs_with_artist_priority(profile, retrieved_candidates, k=k)

    diagnostics = {
        "local_catalog_size": len(local_songs),
        "external_catalog_size": max(0, len(full_catalog) - len(local_songs)),
        "external_primary_fetch_size": len(external_songs),
        "external_artist_boost_size": len(artist_boost_songs),
        "retrieved_candidate_size": len(retrieved_candidates),
        "artist_match_hint": artist_hint,
        "follow_up_directives": follow_up_directives,
        **retrieval_diagnostics,
    }
    return recommendations, {**diagnostics, "derived_profile": profile}


def _extract_follow_up_directives(follow_up_text: str, songs: List[Dict[str, Any]]) -> Dict[str, Any]:
    normalized_follow_up = _normalize_text(follow_up_text)
    directives: Dict[str, Any] = {"avoid_titles": [], "prefer_titles": [], "prefer_terms": []}

    negative_cue_pattern = r"(don t|dont|not|avoid|exclude|skip)"
    positive_cue_pattern = r"(better|prefer|try|include|instead|replace|swap)"
    has_negative_cue = bool(re.search(negative_cue_pattern, normalized_follow_up))
    has_positive_cue = bool(re.search(positive_cue_pattern, normalized_follow_up))
    looks_like_question = "?" in follow_up_text and bool(
        re.search(r"^\s*(do|does|did|can|could|would|is|are|was|were)\b", follow_up_text.strip().lower())
    )

    # Questions like "Do you only have X?" should not mutate ranking.
    if looks_like_question and not has_negative_cue and not has_positive_cue:
        directives["treated_as_question_only"] = True
        return directives

    for song in songs:
        title = str(song.get("title", ""))
        normalized_title = _normalize_text(title)
        if not normalized_title or normalized_title not in normalized_follow_up:
            continue

        negative_pattern = rf"{negative_cue_pattern}(?: \w+){{0,8}} {re.escape(normalized_title)}"
        positive_pattern = rf"{positive_cue_pattern}(?: \w+){{0,8}} {re.escape(normalized_title)}"
        has_negative = bool(re.search(negative_pattern, normalized_follow_up))
        has_positive = bool(re.search(positive_pattern, normalized_follow_up))

        if has_negative and not has_positive:
            directives["avoid_titles"].append(normalized_title)
        elif has_positive:
            directives["prefer_titles"].append(normalized_title)

    directives["avoid_titles"] = sorted(set(directives["avoid_titles"]))
    directives["prefer_titles"] = sorted(set(directives["prefer_titles"]))

    terms = [
        token
        for token in _tokenize(follow_up_text)
        if len(token) >= 4 and token not in FOLLOW_UP_STOPWORDS
    ]
    directives["prefer_terms"] = sorted(set(terms))
    directives["treated_as_question_only"] = False
    return directives


def compose_follow_up_assistant_reply(
    follow_up_text: str,
    directives: Dict[str, Any],
    *,
    top_titles_before: List[str] | None = None,
    top_titles_after: List[str] | None = None,
) -> str:
    """Natural-language summary of how a user follow-up was interpreted (no external LLM)."""
    cleaned = follow_up_text.strip()
    if not cleaned:
        return ""

    parts: List[str] = []
    if directives.get("treated_as_question_only"):
        parts.append(
            "Thanks for writing in. I read that as a question, not a command, "
            "so I did not apply avoid or prefer rules automatically. "
            "Your words were still blended into retrieval, so ordering can shift a little."
        )
    else:
        fragments: List[str] = []
        avoid = directives.get("avoid_titles") or []
        prefer = directives.get("prefer_titles") or []
        terms: List[str] = list(directives.get("prefer_terms") or [])

        if avoid:
            fragments.append(f"I pulled back on tracks matching: {', '.join(avoid)}.")
        if prefer:
            fragments.append(f"I boosted songs closer to: {', '.join(prefer)}.")
        if terms:
            tail = terms[:8]
            extra = " …" if len(terms) > 8 else ""
            if avoid or prefer:
                fragments.append(f"I also weighted lyrics/metadata keywords: {', '.join(tail)}.{extra}")
            else:
                fragments.append(
                    f"I emphasized overlap with these themes in the catalog: {', '.join(tail)}.{extra}"
                )
        if fragments:
            parts.append(" ".join(fragments))
        else:
            parts.append(
                "I folded your follow-up into retrieval and scoring. "
                "I did not detect explicit avoid or prefer patterns, so any change is mostly from the extra context."
            )

    if top_titles_before is not None and top_titles_after is not None:
        before = [t for t in top_titles_before if t]
        after = [t for t in top_titles_after if t]
        if before == after and before:
            parts.append("Your top titles stayed the same this round.")
        elif before and after:
            parts.append(
                f"Top list went from {', '.join(before)} to {', '.join(after)}."
            )
        elif after and not before:
            parts.append(f"Here are the top picks now: {', '.join(after)}.")
    elif top_titles_after is not None:
        after = [t for t in top_titles_after if t]
        if after:
            parts.append(f"Updated top songs: {', '.join(after)}.")

    return " ".join(parts)


def _recommend_songs_with_artist_priority(
    user_prefs: Dict[str, Any],
    songs: List[Dict[str, Any]],
    k: int,
) -> List[Tuple[Dict[str, Any], float, str]]:
    target_artist = str(user_prefs.get("target_artist", "")).strip().lower()
    strict_artist_match = bool(user_prefs.get("strict_artist_match", False))
    avoid_titles = set(str(value) for value in user_prefs.get("avoid_titles", []))
    prefer_titles = set(str(value) for value in user_prefs.get("prefer_titles", []))
    prefer_terms = set(str(value) for value in user_prefs.get("prefer_terms", []))
    scored_recommendations: List[Tuple[Dict[str, Any], float, str]] = []

    for song in songs:
        score, reasons = score_song(user_prefs, song)

        retrieval_score = float(song.get("_retrieval_score", 0.0))
        retrieval_bonus = max(0.0, min(1.5, retrieval_score))
        if retrieval_bonus > 0:
            score += retrieval_bonus
            reasons.append(f"retrieval relevance (+{retrieval_bonus:.2f})")

        normalized_title = _normalize_text(str(song.get("title", "")))
        if normalized_title and normalized_title in avoid_titles:
            score -= 3.0
            reasons.append("follow-up exclusion (-3.0)")
        if normalized_title and normalized_title in prefer_titles:
            score += 2.5
            reasons.append("follow-up preference (+2.5)")
        if prefer_terms:
            document_tokens = set(_tokenize(_song_to_document(song)))
            matched_terms = document_tokens & prefer_terms
            if matched_terms:
                term_bonus = min(1.5, 0.35 * len(matched_terms))
                score += term_bonus
                reasons.append(f"follow-up keyword match (+{term_bonus:.2f})")

        if target_artist and target_artist in str(song.get("artist", "")).lower():
            score += 3.0
            reasons.append("requested artist match (+3.0)")
        scored_recommendations.append((song, score, "; ".join(reasons)))

    scored_recommendations.sort(key=lambda item: item[1], reverse=True)
    if target_artist and strict_artist_match:
        artist_only = [item for item in scored_recommendations if target_artist in str(item[0].get("artist", "")).lower()]
        if artist_only:
            return artist_only[:k]
    return scored_recommendations[:k]


def recommend_from_prompt(
    prompt: str,
    local_songs: List[Dict[str, Any]],
    k: int = 5,
    use_external_retrieval: bool = True,
    prefer_semantic_retrieval: bool = True,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedder: Any | None = None,
    follow_up_text: str | None = None,
) -> Tuple[List[Tuple[Dict[str, Any], float, str]], Dict[str, Any]]:
    """Run prompt -> retrieval -> ranking -> explanation pipeline."""
    recommendations, diagnostics = _run_retrieval_and_ranking(
        prompt,
        local_songs,
        k=k,
        use_external_retrieval=use_external_retrieval,
        prefer_semantic_retrieval=prefer_semantic_retrieval,
        embedding_model_name=embedding_model_name,
        embedder=embedder,
        follow_up_text=follow_up_text,
    )
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

    plan_notes: List[str] = [
        "Parse user intent into a recommendation profile.",
        "Retrieve candidate songs from local and optional external sources.",
        "Rank candidates and self-check recommendation confidence.",
    ]
    if follow_up_answer:
        plan_notes.append("Incorporate follow-up clarification from the user.")

    recommendations, diagnostics = _run_retrieval_and_ranking(
        cleaned_prompt,
        local_songs,
        k=k,
        use_external_retrieval=use_external_retrieval,
        prefer_semantic_retrieval=prefer_semantic_retrieval,
        embedding_model_name=embedding_model_name,
        embedder=embedder,
        follow_up_text=follow_up_answer,
    )
    confidence = _confidence_score(recommendations, diagnostics)

    follow_up_question = None
    if confidence < confidence_threshold and not follow_up_answer:
        follow_up_question = _generate_follow_up_question(cleaned_prompt)

    diagnostics.update(
        {
            "agentic_plan": plan_notes,
            "confidence_score": confidence,
            "follow_up_question": follow_up_question,
            "confidence_threshold": confidence_threshold,
        }
    )
    return recommendations, diagnostics
