"""Streamlit UI for the music recommender."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import streamlit as st

# Ensure `src` package imports work when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_recommender import (
    DEFAULT_EMBEDDING_MODEL,
    agentic_recommend_from_prompt,
    compose_follow_up_assistant_reply,
    recommend_from_prompt,
)
from src.recommender import load_songs

LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure file logging for UI mode."""
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        filename="logs/app.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@st.cache_data(show_spinner=False)
def load_catalog() -> List[Dict[str, Any]]:
    """Load song data from local CSV once per session."""
    return load_songs("data/songs.csv")


def render_recommendations(recommendations: List[Tuple[Dict[str, Any], float, str]]) -> None:
    """Render ranked recommendation cards."""
    st.subheader("Top Matches")
    for index, (song, score, explanation) in enumerate(recommendations, start=1):
        with st.container(border=True):
            st.markdown(f"**{index}. {song['title']}** by *{song['artist']}*")
            st.write(f"Score: `{score:.2f}`")
            st.caption(explanation or "No explanation available.")


def main() -> None:
    """Run the Streamlit app."""
    setup_logging()
    st.set_page_config(page_title="AI Music Recommender", page_icon="🎵", layout="wide")

    st.title("AI Music Recommender")
    st.write(
        "Describe what you feel like listening to, then get grounded recommendations "
        "from local songs plus optional public retrieval."
    )

    try:
        songs = load_catalog()
    except FileNotFoundError:
        st.error("Missing `data/songs.csv`. Please run the app from the project root.")
        return

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top K recommendations", min_value=1, max_value=10, value=5)
        use_external_retrieval = st.checkbox("Use external retrieval (iTunes API)", value=True)
        use_agentic_workflow = st.checkbox("Use agentic workflow", value=True)
        prefer_semantic_retrieval = st.checkbox("Use semantic retrieval embeddings", value=True)
        embedding_model_name = st.text_input("Embedding model", value=DEFAULT_EMBEDDING_MODEL)
        confidence_threshold = st.slider(
            "Agentic confidence threshold",
            min_value=0.50,
            max_value=0.95,
            value=0.70,
            step=0.05,
            disabled=not use_agentic_workflow,
        )
        st.caption(f"Loaded local songs: {len(songs)}")

    prompt = st.text_area(
        "Describe your vibe",
        placeholder=(
            "Example: I want poetic, reflective hip-hop with lyrical depth for a late-night walk."
        ),
        height=120,
    )

    follow_up_answer = st.text_input(
        "Optional follow-up clarification",
        placeholder="Example: Make it low-energy and more acoustic.",
    )

    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = ""
    if "last_recommendations" not in st.session_state:
        st.session_state.last_recommendations = []
    if "last_diagnostics" not in st.session_state:
        st.session_state.last_diagnostics = None
    if "previous_recommendations" not in st.session_state:
        st.session_state.previous_recommendations = []

    recommend_clicked_col, follow_up_clicked_col = st.columns(2)
    recommend_clicked = recommend_clicked_col.button("Recommend songs", type="primary")
    follow_up_clicked = follow_up_clicked_col.button(
        "Apply follow-up",
        disabled=not (st.session_state.last_prompt or prompt.strip()),
    )

    def run_and_store(active_prompt: str, active_follow_up: str | None) -> None:
        try:
            if use_agentic_workflow:
                recommendations, diagnostics = agentic_recommend_from_prompt(
                    prompt=active_prompt,
                    local_songs=songs,
                    k=top_k,
                    use_external_retrieval=use_external_retrieval,
                    follow_up_answer=active_follow_up,
                    confidence_threshold=confidence_threshold,
                    prefer_semantic_retrieval=prefer_semantic_retrieval,
                    embedding_model_name=embedding_model_name.strip() or DEFAULT_EMBEDDING_MODEL,
                )
            else:
                recommendations, diagnostics = recommend_from_prompt(
                    prompt=active_prompt,
                    local_songs=songs,
                    k=top_k,
                    use_external_retrieval=use_external_retrieval,
                    prefer_semantic_retrieval=prefer_semantic_retrieval,
                    embedding_model_name=embedding_model_name.strip() or DEFAULT_EMBEDDING_MODEL,
                    follow_up_text=active_follow_up,
                )
        except Exception as exc:  # pragma: no cover - UI safety guardrail
            LOGGER.exception("UI recommendation run failed.")
            st.error(f"Recommendation failed: {exc}")
            return

        st.session_state.previous_recommendations = st.session_state.last_recommendations
        st.session_state.last_prompt = active_prompt
        st.session_state.last_recommendations = recommendations
        st.session_state.last_diagnostics = diagnostics
        if active_follow_up:
            prev_rows = st.session_state.previous_recommendations
            diagnostics["follow_up_assistant_reply"] = compose_follow_up_assistant_reply(
                active_follow_up,
                diagnostics.get("follow_up_directives", {}),
                top_titles_before=[e[0].get("title", "") for e in prev_rows[:top_k]],
                top_titles_after=[e[0].get("title", "") for e in recommendations[:top_k]],
            )

    if recommend_clicked:
        if not prompt.strip():
            st.warning("Please enter a prompt first.")
        else:
            run_and_store(prompt.strip(), None)

    if follow_up_clicked:
        base_prompt = st.session_state.last_prompt or prompt.strip()
        if not base_prompt:
            st.warning("Please run a recommendation first.")
        elif not follow_up_answer.strip():
            st.warning("Please enter follow-up clarification before applying.")
        else:
            run_and_store(base_prompt, follow_up_answer.strip())

    recommendations = st.session_state.last_recommendations
    diagnostics = st.session_state.last_diagnostics
    if not diagnostics:
        return

    if diagnostics.get("follow_up_assistant_reply"):
        with st.chat_message("assistant"):
            st.write(diagnostics["follow_up_assistant_reply"])

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Local catalog", diagnostics["local_catalog_size"])
    col_b.metric("External songs", diagnostics["external_catalog_size"])
    col_c.metric("Retrieved candidates", diagnostics["retrieved_candidate_size"])

    st.markdown("### Derived Profile")
    st.json(diagnostics["derived_profile"])

    if diagnostics.get("artist_match_hint"):
        artist_hint = diagnostics["artist_match_hint"]
        st.write(
            "Detected artist intent: "
            f"`{artist_hint['artist']}` "
            f"(catalog matches: {artist_hint['catalog_song_count']})"
        )

    st.markdown("### Retrieval Diagnostics")
    st.write(
        f"Strategy: `{diagnostics['retrieval_strategy']}` | "
        f"Semantic model used: `{diagnostics['semantic_model_used']}` | "
        f"Embedding model: `{diagnostics.get('embedding_model_name')}`"
    )

    follow_up_directives = diagnostics.get("follow_up_directives", {})
    if any(follow_up_directives.get(key) for key in ["avoid_titles", "prefer_titles", "prefer_terms"]):
        st.markdown("### Follow-up Interpretation")
        if follow_up_directives.get("avoid_titles"):
            st.write(f"- Avoid titles: {', '.join(follow_up_directives['avoid_titles'])}")
        if follow_up_directives.get("prefer_titles"):
            st.write(f"- Preferred titles: {', '.join(follow_up_directives['prefer_titles'])}")
        if follow_up_directives.get("prefer_terms"):
            st.write(f"- Preferred keywords: {', '.join(follow_up_directives['prefer_terms'])}")

    if use_agentic_workflow and "confidence_score" in diagnostics:
        st.markdown("### Agentic Self-Check")
        st.write(
            f"Confidence: `{diagnostics['confidence_score']:.3f}` "
            f"(threshold: `{diagnostics['confidence_threshold']:.2f}`)"
        )
        for plan_step in diagnostics["agentic_plan"]:
            st.write(f"- {plan_step}")

        if diagnostics.get("follow_up_question") and not follow_up_answer:
            st.info(f"Agent follow-up: {diagnostics['follow_up_question']}")

    previous_recommendations = st.session_state.previous_recommendations or []
    if previous_recommendations:
        previous_top = [entry[0].get("title", "") for entry in previous_recommendations[:top_k]]
        current_top = [entry[0].get("title", "") for entry in recommendations[:top_k]]
        if previous_top != current_top:
            st.markdown("### What Changed After Follow-up")
            st.write(f"Before: {', '.join(previous_top)}")
            st.write(f"After: {', '.join(current_top)}")

    render_recommendations(recommendations)


if __name__ == "__main__":
    main()
