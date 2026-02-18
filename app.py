import random
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Flashcards", layout="wide")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent question/answer columns for mixed CSV formats."""
    if df.empty:
        return pd.DataFrame(columns=["question", "answer"])

    # If headers already look like question/answer, use them directly.
    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    q_key = next((k for k in lower_cols if k in {"question", "questions", "q"}), None)
    a_key = next((k for k in lower_cols if k in {"answer", "answers", "a"}), None)
    if q_key and a_key:
        out = df[[lower_cols[q_key], lower_cols[a_key]]].copy()
        out.columns = ["question", "answer"]
        return out

    # Fallback for headerless files: treat first two columns as question/answer.
    raw = df.copy()
    if raw.shape[1] < 2:
        return pd.DataFrame(columns=["question", "answer"])

    out = raw.iloc[:, :2].copy()
    out.columns = ["question", "answer"]

    # Drop a possible header row accidentally read as data.
    first_q = str(out.iloc[0, 0]).strip().lower()
    first_a = str(out.iloc[0, 1]).strip().lower()
    if first_q in {"question", "questions", "q"} and first_a in {"answer", "answers", "a"}:
        out = out.iloc[1:]

    return out


@st.cache_data(show_spinner=False)
def load_flashcards(folder: str = ".") -> pd.DataFrame:
    rows = []
    csv_files = sorted(Path(folder).glob("*.csv"))

    for csv_file in csv_files:
        module_name = csv_file.stem
        try:
            # Always parse headerless first so we never lose row 1 in no-header CSV files.
            data = pd.read_csv(csv_file, dtype=str, keep_default_na=False, header=None)
            normalized = _normalize_columns(data)

            normalized = normalized.dropna(how="all")
            normalized["question"] = normalized["question"].astype(str).str.strip()
            normalized["answer"] = normalized["answer"].astype(str).str.strip()
            normalized = normalized[
                (normalized["question"] != "") & (normalized["answer"] != "")
            ]

            for _, row in normalized.iterrows():
                rows.append(
                    {
                        "module": module_name,
                        "question": row["question"],
                        "answer": row["answer"],
                    }
                )
        except Exception as exc:
            st.warning(f"Skipped {csv_file.name}: {exc}")

    return pd.DataFrame(rows)


def ensure_state(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 70% 35%, #ddf8df 0%, #f4f6f8 45%, #eef0f3 100%);
        }
        .block-container {
            max-width: 1274px;
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }
        .flashcard-wrap {
            background: #2d2d30;
            border-radius: 46px;
            min-height: 400px;
            padding: 48px 50px 36px 50px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 18px 42px rgba(20, 20, 22, 0.22);
        }
        .flashcard-wrap .flashcard-question {
            color: #ffffff;
            font-size: clamp(1rem, 2vw, 2rem) !important;
            line-height: 1.06 !important;
            font-weight: 700 !important;
            margin: 0 !important;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .see-answer {
            color: #9fa4ae;
            font-size: 1.9rem;
            text-align: center;
            margin-top: 10px;
        }
        .answer-box {
            margin-top: 18px;
            border-top: 1px solid rgba(255, 255, 255, 0.18);
            padding-top: 16px;
            color: #d8f8df;
            font-size: 1.2rem;
            line-height: 1.5;
        }
        .score-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 18px;
            margin-top: 16px;
        }
        .score-left {
            color: #ef3f3f;
            font-size: 2rem;
            font-weight: 700;
            min-width: 24px;
            text-align: right;
        }
        .score-right {
            color: #0ea645;
            font-size: 2rem;
            font-weight: 700;
            min-width: 24px;
            text-align: left;
        }
        .score-spacer {
            color: #6d7480;
            font-size: 1.15rem;
        }
        .footer-row {
            display: flex;
            justify-content: space-between;
            margin-top: 0.65rem;
            color: #4b5462;
            font-size: 1rem;
            font-weight: 600;
        }
        div[data-testid="stButton"] > button {
            font-size: 1.2rem;
            font-weight: 600;
        }
        div[data-testid="stButton"] > button[kind="secondary"] {
            color: #274cff;
            border: 1px solid #7f94ff;
            background: rgba(255, 255, 255, 0.65);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_styles()
    st.title("Flashcards")

    flashcards = load_flashcards(".")

    if flashcards.empty:
        st.error("No valid flashcards found. Add CSV files with Question,Answer rows.")
        return

    modules = ["All modules"] + sorted(flashcards["module"].unique().tolist())
    selected_module = st.sidebar.selectbox("Choose module", modules)

    if selected_module == "All modules":
        view_df = flashcards.reset_index(drop=True)
    else:
        view_df = flashcards[flashcards["module"] == selected_module].reset_index(drop=True)

    ensure_state("card_index", 0)
    ensure_state("show_answer", False)
    ensure_state("missed_count", 0)
    ensure_state("got_count", 0)
    ensure_state("active_module", selected_module)

    total = len(view_df)
    if total == 0:
        st.warning("No flashcards in this module.")
        return

    if st.session_state.active_module != selected_module:
        st.session_state.active_module = selected_module
        st.session_state.card_index = 0
        st.session_state.show_answer = False
        st.session_state.missed_count = 0
        st.session_state.got_count = 0

    # Normalize old session values so attempts never exceed the number of cards.
    attempts_total = st.session_state.missed_count + st.session_state.got_count
    if attempts_total > total:
        st.session_state.got_count = min(st.session_state.got_count, total)
        st.session_state.missed_count = min(
            st.session_state.missed_count, total - st.session_state.got_count
        )

    # Keep index valid if module changed.
    st.session_state.card_index %= total

    attempts_count = st.session_state.missed_count + st.session_state.got_count
    review_complete = attempts_count >= total

    nav_left, card_center, nav_right = st.columns([1, 6, 1])

    with nav_left:
        st.markdown("<div style='height:140px'></div>", unsafe_allow_html=True)
        if st.button("←", key="prev_card", use_container_width=True):
            st.session_state.card_index = (st.session_state.card_index - 1) % total
            st.session_state.show_answer = False
            st.rerun()

    with nav_right:
        st.markdown("<div style='height:140px'></div>", unsafe_allow_html=True)
        if st.button("→", key="next_card", use_container_width=True):
            st.session_state.card_index = (st.session_state.card_index + 1) % total
            st.session_state.show_answer = False
            st.rerun()

    idx = st.session_state.card_index
    card = view_df.iloc[idx]

    with card_center:
        answer_html = ""
        if st.session_state.show_answer:
            answer_html = f"{card['answer']}"
        st.markdown(
            f"""
            <div class="flashcard-wrap">
                <p class="flashcard-question">{card['question']}</p>
             
                {answer_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Reveal answer", key="reveal_answer", use_container_width=True):
            st.session_state.show_answer = not st.session_state.show_answer
            st.rerun()

    score_left, score_mid1, score_mid2, score_right = st.columns([1, 2.2, 2.2, 1])
    with score_mid1:
        if st.button(
            "Missed it",
            key="missed_btn",
            use_container_width=True,
            disabled=review_complete,
        ):
            st.session_state.missed_count += 1
            st.session_state.card_index = random.randint(0, total - 1)
            st.session_state.show_answer = False
            st.rerun()
    with score_mid2:
        if st.button(
            "Got it",
            key="got_btn",
            use_container_width=True,
            disabled=review_complete,
        ):
            st.session_state.got_count += 1
            st.session_state.card_index = random.randint(0, total - 1)
            st.session_state.show_answer = False
            st.rerun()
    with score_left:
        st.markdown(
            f"<div style='margin-top:10px' class='score-left'>{st.session_state.missed_count}</div>",
            unsafe_allow_html=True,
        )
    with score_right:
        st.markdown(
            f"<div style='margin-top:10px' class='score-right'>{st.session_state.got_count}</div>",
            unsafe_allow_html=True,
        )

    attempts_count = st.session_state.missed_count + st.session_state.got_count
    reviewed_count = attempts_count
    progress = min(reviewed_count / total, 1.0)
    st.progress(progress)
    st.markdown(
        f"<div class='footer-row'><span>{selected_module}</span><span>{reviewed_count} / {total} cards reviewed</span></div>",
        unsafe_allow_html=True,
    )

    if review_complete:
        st.success("All cards reviewed for this module.")



if __name__ == "__main__":
    main()
