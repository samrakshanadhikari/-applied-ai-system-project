"""CLI runner for profile and prompt-driven music recommendation workflows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

from src.rag_recommender import (
    DEFAULT_EMBEDDING_MODEL,
    recommend_from_prompt,
)
from src.recommender import load_songs, recommend_songs


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

LOGGER = logging.getLogger(__name__)


def print_recommendations(label: str, recommendations) -> None:
    """Print one profile's ranked recommendations in a readable block."""
    print(f"\n=== {label} ===")
    for index, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"{index}. {song['title']} by {song['artist']} | Score: {score:.2f}")
        print(f"   Because: {explanation}")


def setup_logging() -> None:
    """Configure file logging for reproducibility and debugging."""
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        filename="logs/app.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_profile_demo(songs) -> None:
    """Run the original profile-comparison workflow."""
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


def run_prompt_mode(
    prompt: str,
    songs,
    top_k: int,
    use_external_retrieval: bool,
    prefer_semantic_retrieval: bool = True,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Dict[str, Any]:
    """Run prompt-based recommendation with retrieval + ranking."""
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        raise ValueError("Prompt cannot be empty.")

    LOGGER.info(
        "Prompt mode start | prompt=%s | top_k=%d | external_retrieval=%s | semantic=%s",
        cleaned_prompt,
        top_k,
        use_external_retrieval,
        prefer_semantic_retrieval,
    )

    recommendations, diagnostics = recommend_from_prompt(
        cleaned_prompt,
        songs,
        k=top_k,
        use_external_retrieval=use_external_retrieval,
        prefer_semantic_retrieval=prefer_semantic_retrieval,
        embedding_model_name=embedding_model_name,
    )

    print("\n=== Prompt-Based Recommendation (RAG Pipeline) ===")
    print(f"Prompt: {cleaned_prompt}")
    print(f"Derived profile: {diagnostics['derived_profile']}")
    print(
        "Retrieval:"
        f" strategy={diagnostics['retrieval_strategy']},"
        f" semantic_model_used={diagnostics['semantic_model_used']},"
        f" embedding_model={diagnostics.get('embedding_model_name')}"
    )
    print(
        "Catalog stats:"
        f" local={diagnostics['local_catalog_size']},"
        f" external={diagnostics['external_catalog_size']},"
        f" retrieved={diagnostics['retrieved_candidate_size']}"
    )

    print_recommendations("Top matches", recommendations)
    LOGGER.info("Prompt mode completed successfully.")
    return diagnostics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for selecting workflow mode."""
    parser = argparse.ArgumentParser(description="Music recommender CLI")
    parser.add_argument(
        "--prompt",
        type=str,
        help="Natural language request, e.g. 'I want calm acoustic songs for studying'.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask for a prompt interactively in the terminal.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of recommendations to return (default: 5).",
    )
    parser.add_argument(
        "--no-external",
        action="store_true",
        help="Disable public API retrieval and use local catalog only.",
    )
    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Disable semantic embedding retrieval and use lexical retrieval only.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformers model name used for semantic retrieval.",
    )
    return parser.parse_args()


def main() -> None:
    """Run either profile demo mode or prompt-based RAG mode."""
    setup_logging()
    args = parse_args()

    try:
        songs = load_songs("data/songs.csv")
    except FileNotFoundError as exc:
        LOGGER.exception("Song catalog not found.")
        raise FileNotFoundError("Missing required file: data/songs.csv") from exc

    print(f"Loaded songs: {len(songs)}")

    if args.prompt or args.interactive:
        prompt = args.prompt or input("Describe what you want to listen to: ")
        try:
            diagnostics = run_prompt_mode(
                prompt=prompt,
                songs=songs,
                top_k=max(args.top_k, 1),
                use_external_retrieval=not args.no_external,
                prefer_semantic_retrieval=not args.lexical_only,
                embedding_model_name=args.embedding_model,
            )
            _ = diagnostics
        except ValueError as exc:
            LOGGER.warning("Guardrail triggered: %s", exc)
            print(f"Input error: {exc}")
        except Exception as exc:  # pragma: no cover - defensive guardrail
            LOGGER.exception("Unexpected prompt mode failure.")
            print(f"Something went wrong while generating recommendations: {exc}")
        return

    run_profile_demo(songs)


if __name__ == "__main__":
    main()
