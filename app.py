"""
app.py
------
AI Sentiment Analysis - a Streamlit teaching app for NLP classes.

Run with:
    streamlit run app.py

Tabs:
    1. Analyze Text     - single sentence/paragraph analysis with a
                           word-by-word explanation of the score
    2. Batch (CSV)       - upload a CSV of text and get sentiment for
                           every row, downloadable as a new CSV
    3. Compare Methods   - run both analyzers on the same text side by
                           side to discuss where/why they disagree
    4. How It Works      - classroom reference material embedded in the
                           app itself (definitions, formulas, pipeline)
"""

import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sentiment_engine import (
    SimpleLexiconSentiment,
    VaderSentiment,
    tokenize,
)

st.set_page_config(
    page_title="AI Sentiment Analysis - NLP Teaching App",
    page_icon="💬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached analyzer construction (VADER needs a one-time lexicon download)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_simple_analyzer():
    return SimpleLexiconSentiment()


@st.cache_resource(show_spinner=False)
def load_vader_analyzer():
    return VaderSentiment()


def get_vader_safely():
    """Returns (analyzer_or_None, error_message_or_None)."""
    try:
        return load_vader_analyzer(), None
    except Exception as e:  # noqa: BLE001 - want to show any setup issue to the class
        return None, (
            "VADER could not be loaded. Make sure `nltk` is installed and "
            "you have an internet connection the first time you run this "
            f"app (it downloads a small lexicon file).\n\nDetails: {e}"
        )


# ---------------------------------------------------------------------------
# Small display helpers
# ---------------------------------------------------------------------------

LABEL_COLORS = {"Positive": "#2e7d32", "Negative": "#c62828", "Neutral": "#616161"}


def label_badge(label: str) -> str:
    color = LABEL_COLORS.get(label, "#616161")
    return f"<span style='background-color:{color};color:white;padding:4px 12px;border-radius:12px;font-weight:600;'>{label}</span>"


def render_score_bar(scores: dict, title: str):
    fig, ax = plt.subplots(figsize=(5, 2.2))
    labels = list(scores.keys())
    values = list(scores.values())
    colors = ["#c62828" if v < 0 else "#2e7d32" if v > 0 else "#9e9e9e" for v in values]
    ax.barh(labels, values, color=colors)
    ax.set_xlim(-1, 1)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_word_highlight(contributions):
    """Render the original tokens with color-coded backgrounds showing
    each word's contribution to the score -- a key teaching visual."""
    if not contributions:
        st.info("No sentiment-bearing words were found in the lexicon.")
        return

    html_parts = []
    for c in contributions:
        if c.final_score > 0:
            bg = "#c8e6c9"
        elif c.final_score < 0:
            bg = "#ffcdd2"
        else:
            bg = "#eeeeee"
        tag = " (negated)" if c.negated else ""
        html_parts.append(
            f"<span title='base={c.base_score}, scaled x{c.scaled_by}, "
            f"final={c.final_score:.2f}{tag}' "
            f"style='background-color:{bg}; padding:2px 6px; margin:2px; "
            f"border-radius:4px; display:inline-block; font-size:0.95rem;'>"
            f"{c.word}</span>"
        )
    st.markdown(" ".join(html_parts), unsafe_allow_html=True)
    st.caption("Hover over a highlighted word to see how its score was computed. Green = positive contribution, red = negative.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("💬 Sentiment Analysis")
st.sidebar.markdown(
    "A teaching tool for **Natural Language Processing**.\n\n"
    "Explore how rule-based sentiment analysis works, word by word, "
    "and compare a hand-built analyzer to an industry-standard one (VADER)."
)
st.sidebar.divider()
st.sidebar.markdown(
    "**Included analyzers**\n"
    "- `Simple Lexicon` — a small, from-scratch analyzer you can read "
    "line by line in `sentiment_engine.py`\n"
    "- `VADER` — a widely used lexicon + rule-based model (NLTK)"
)
st.sidebar.divider()
st.sidebar.caption("Built as a companion to the accompanying PDF teaching note.")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_analyze, tab_batch, tab_compare, tab_how = st.tabs(
    ["🔎 Analyze Text", "📄 Batch (CSV)", "⚖️ Compare Methods", "📘 How It Works"]
)

# --- Tab 1: Analyze Text ---------------------------------------------------
with tab_analyze:
    st.subheader("Analyze a sentence or paragraph")

    default_text = "I absolutely loved this movie, the acting was great! But the ending was a bit disappointing."
    text_input = st.text_area("Enter text to analyze:", value=default_text, height=120)

    method = st.radio(
        "Choose analyzer:",
        options=["Simple Lexicon (from scratch)", "VADER (NLTK)"],
        horizontal=True,
    )

    if st.button("Analyze", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text first.")
        elif method.startswith("Simple"):
            analyzer = load_simple_analyzer()
            result = analyzer.analyze(text_input)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**Sentiment:** {label_badge(result['label'])}", unsafe_allow_html=True)
                st.metric("Compound score", f"{result['compound']:.3f}", help="Range: -1 (very negative) to +1 (very positive)")
            with col2:
                render_score_bar({"compound": result["compound"]}, "Overall Score")

            st.markdown("#### Word-by-word breakdown")
            render_word_highlight(result["contributions"])

            with st.expander("See raw tokens"):
                st.write(result["tokens"])

        else:  # VADER
            analyzer, err = get_vader_safely()
            if err:
                st.error(err)
            else:
                result = analyzer.analyze(text_input)
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**Sentiment:** {label_badge(result['label'])}", unsafe_allow_html=True)
                    st.metric("Compound score", f"{result['compound']:.3f}")
                with col2:
                    render_score_bar(
                        {"negative": -result["neg"], "neutral": 0, "positive": result["pos"]},
                        "pos / neu / neg proportions"
                    )
                st.markdown("#### Raw VADER scores")
                st.json({k: result[k] for k in ("neg", "neu", "pos", "compound")})
                st.caption(
                    "VADER doesn't expose per-word contributions the way our simple "
                    "analyzer does — it's a good discussion point about transparency "
                    "vs. sophistication in NLP tools."
                )

# --- Tab 2: Batch CSV -------------------------------------------------------
with tab_batch:
    st.subheader("Analyze a batch of texts from a CSV file")
    st.markdown(
        "Upload a CSV with a column of text (e.g. product reviews, tweets, "
        "survey responses). Great for a class dataset exercise."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    batch_method = st.radio(
        "Analyzer for batch run:",
        options=["Simple Lexicon (from scratch)", "VADER (NLTK)"],
        horizontal=True,
        key="batch_method",
    )

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        if df is not None:
            st.write("Preview:")
            st.dataframe(df.head())

            text_col = st.selectbox("Which column contains the text?", options=df.columns)

            if st.button("Run batch analysis", type="primary"):
                if batch_method.startswith("Simple"):
                    analyzer = load_simple_analyzer()
                    labels, compounds = [], []
                    for t in df[text_col].astype(str):
                        r = analyzer.analyze(t)
                        labels.append(r["label"])
                        compounds.append(r["compound"])
                    df["sentiment_label"] = labels
                    df["compound_score"] = compounds
                else:
                    analyzer, err = get_vader_safely()
                    if err:
                        st.error(err)
                        analyzer = None
                    if analyzer:
                        labels, compounds = [], []
                        for t in df[text_col].astype(str):
                            r = analyzer.analyze(t)
                            labels.append(r["label"])
                            compounds.append(r["compound"])
                        df["sentiment_label"] = labels
                        df["compound_score"] = compounds

                if "sentiment_label" in df.columns:
                    st.success(f"Analyzed {len(df)} rows.")
                    st.dataframe(df)

                    counts = df["sentiment_label"].value_counts()
                    fig, ax = plt.subplots(figsize=(4, 3))
                    colors = [LABEL_COLORS.get(l, "#616161") for l in counts.index]
                    ax.bar(counts.index, counts.values, color=colors)
                    ax.set_ylabel("Count")
                    ax.set_title("Sentiment distribution")
                    st.pyplot(fig)
                    plt.close(fig)

                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "Download results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                    )

# --- Tab 3: Compare Methods -------------------------------------------------
with tab_compare:
    st.subheader("Compare both analyzers on the same text")
    st.markdown(
        "A good class discussion: find a sentence where the two analyzers "
        "**disagree**, and talk about why (sarcasm, domain-specific words, "
        "lexicon coverage, negation handling, etc.)."
    )

    compare_text = st.text_area(
        "Text to compare:",
        value="The service wasn't bad, but I wouldn't call it great either.",
        height=100,
        key="compare_text",
    )

    if st.button("Compare", type="primary"):
        simple_result = load_simple_analyzer().analyze(compare_text)
        vader_analyzer, err = get_vader_safely()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Simple Lexicon (from scratch)")
            st.markdown(label_badge(simple_result["label"]), unsafe_allow_html=True)
            st.metric("Compound", f"{simple_result['compound']:.3f}")
            render_word_highlight(simple_result["contributions"])

        with col2:
            st.markdown("##### VADER (NLTK)")
            if err:
                st.error(err)
            else:
                vresult = vader_analyzer.analyze(compare_text)
                st.markdown(label_badge(vresult["label"]), unsafe_allow_html=True)
                st.metric("Compound", f"{vresult['compound']:.3f}")
                st.json({k: vresult[k] for k in ("neg", "neu", "pos", "compound")})

        if not err and simple_result["label"] != vresult["label"]:
            st.info(
                f"🔍 The two methods disagree: Simple Lexicon says "
                f"**{simple_result['label']}**, VADER says **{vresult['label']}**. "
                "Ask students: which one matches human judgment here, and why?"
            )

# --- Tab 4: How It Works -----------------------------------------------------
with tab_how:
    st.subheader("How sentiment analysis works (classroom reference)")

    st.markdown("""
#### 1. The NLP pipeline used in this app
1. **Tokenization** — split raw text into words (tokens). See `tokenize()` in
   `sentiment_engine.py`.
2. **Lexicon lookup** — check each token against a dictionary of words with
   known positive/negative "valence" scores.
3. **Negation handling** — if a negation word (*not*, *never*, *n't*...)
   appears shortly before a sentiment word, its score is flipped.
4. **Intensifiers/diminishers** — words like *very* or *slightly* scale the
   sentiment word's score up or down.
5. **Aggregation & normalization** — all token scores are summed, then
   squashed into a fixed range (here, roughly -1 to +1) so scores are
   comparable across sentences of different lengths.

#### 2. Two approaches, one comparison
| | Simple Lexicon (this app) | VADER (NLTK) |
|---|---|---|
| Lexicon size | ~50 words (teaching-sized) | ~7,500 words |
| Handles negation | Yes (basic window) | Yes (more nuanced) |
| Handles punctuation/caps (e.g. "!!!", "GREAT") | No | Yes |
| Transparent per-word scores | Yes | No (aggregate only) |
| Built for | Learning the mechanics | Real-world short text (reviews, tweets) |

#### 3. Key vocabulary
- **Valence** — the intrinsic positive/negative "charge" of a word.
- **Compound score** — a single normalized number summarizing overall
  sentiment, typically in [-1, 1].
- **Lexicon-based approach** — sentiment is looked up from a dictionary of
  pre-scored words (fast, interpretable, but limited by dictionary coverage).
- **Machine-learning approach** (not shown in this app, worth discussing) —
  a model (e.g. logistic regression, LSTM, transformer) learns sentiment
  patterns from labeled training data instead of a fixed dictionary.

#### 4. Suggested classroom exercises
- Add 10 new words to `POSITIVE_WORDS`/`NEGATIVE_WORDS` and see how scores change.
- Feed the app a sarcastic sentence. Does it fail? Why is sarcasm hard for
  lexicon-based methods?
- Upload a small CSV of real product reviews and discuss the distribution
  of predicted sentiment vs. the actual star ratings.
- Extend `NEGATION_WORDS` or `INTENSIFIERS` and re-test the "Compare" tab.
""")

    st.info(
        "A full written version of this material, formatted for handing "
        "out to students, is available as the accompanying PDF teaching note."
    )
