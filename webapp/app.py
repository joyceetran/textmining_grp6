"""
MD&A Sentiment Dashboard
========================
Run: streamlit run webapp/app.py
Requires: webapp/final_df.parquet  (built by webapp/final_data.ipynb)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

WEBAPP_DIR = Path(__file__).parent

COMPANY_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#4c78a8", "#f58518",
]


def dn(c: str) -> str:
    """Display name: replace underscores with spaces."""
    return c.replace("_", " ")


def fmt_score(v):
    return "-" if pd.isna(v) else f"{v:+.2f}"


st.set_page_config(
    page_title="MD&A Sentiment",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    final_path = WEBAPP_DIR / "final_df.parquet"
    if not final_path.exists():
        st.error(
            "final_df.parquet not found. "
            "Run webapp/final_data.ipynb first to build it."
        )
        st.stop()

    df = pd.read_parquet(final_path)
    df = df.rename(columns={
        "company":      "company_name",
        "score":        "sentiment_score",
        "label":        "finbert_label",
        "sentence":     "text",
        "pos":          "pos_prob",
        "neg":          "neg_prob",
        "neu":          "neu_prob",
        "topic_weight": "topic_prob",
    })
    df["topic_label"] = df["topic_label"].fillna("Uncategorised")
    df["year"] = df["year"].astype(int)
    return df


merged = load_data()

# ── Derived universe ────────────────────────────────────────────────────────
all_cos    = sorted(merged["company_name"].dropna().unique().tolist())
avail_yrs  = sorted(merged["year"].dropna().unique().tolist())
all_topics = sorted(merged["topic_label"].dropna().unique().tolist())

co_colors = {
    c: COMPANY_PALETTE[i % len(COMPANY_PALETTE)] for i, c in enumerate(all_cos)
}

# ── Helper functions ────────────────────────────────────────────────────────
def _filter(df, company=None, year=None, quarter=None, topic=None):
    if company is not None:
        df = df[df["company_name"] == company]
    if year is not None:
        df = df[df["year"] == year]
    if quarter is not None:
        df = df[df["quarter"] == quarter]
    if topic is not None:
        df = df[df["topic_label"] == topic]
    return df


def co_score(company, year, quarter):
    sub = _filter(merged, company=company, year=year, quarter=quarter)
    return float(sub["sentiment_score"].mean()) if len(sub) else np.nan


def portfolio_score(weights, year, quarter):
    ws = wt = 0.0
    for c, w in weights.items():
        s = co_score(c, year, quarter)
        if not np.isnan(s):
            ws += w * s
            wt += w
    return ws / wt if wt else np.nan


def topic_scores(company, year, quarter):
    sub = _filter(merged, company=company, year=year, quarter=quarter)
    return sub.groupby("topic_label")["sentiment_score"].mean()


def portfolio_topic_scores(weights, year, quarter):
    rows = []
    for c, w in weights.items():
        for t, s in topic_scores(c, year, quarter).items():
            rows.append({"topic": t, "score": s, "w": w})
    if not rows:
        return pd.Series(dtype=float)
    tmp = pd.DataFrame(rows)
    return tmp.groupby("topic").apply(
        lambda g: np.average(g["score"], weights=g["w"])
    )


def ranked_company_scores(companies, year, quarter, topic=None):
    rows = []
    for c in companies:
        sub = _filter(merged, company=c, year=year, quarter=quarter, topic=topic)
        if len(sub):
            rows.append({"company": c, "display": dn(c), "score": float(sub["sentiment_score"].mean())})
    if not rows:
        return pd.DataFrame(columns=["company", "display", "score"])
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def apply_top_bottom(df, mode, n, score_col="score"):
    if not len(df):
        return df
    if mode == "Top":
        return df.nlargest(n, score_col)
    if mode == "Bottom":
        return df.nsmallest(n, score_col)
    return df


def topic_keyword_table(df, topics, top_n_words=10):
    rows = []
    for topic in topics:
        texts = df[df["topic_label"] == topic]["text"].dropna().astype(str)
        texts = texts[texts.str.strip().str.len() > 0]
        keywords = ""
        if len(texts):
            try:
                vec = CountVectorizer(stop_words="english", max_features=5000)
                x = vec.fit_transform(texts)
                if x.shape[1] > 0:
                    freqs = np.asarray(x.sum(axis=0)).ravel()
                    terms = np.array(vec.get_feature_names_out())
                    top_idx = np.argsort(freqs)[::-1][:top_n_words]
                    keywords = ", ".join(terms[top_idx])
            except ValueError:
                pass
        rows.append({"Topic": topic, "Top words": keywords})
    return pd.DataFrame(rows)


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("MD&A Sentiment")
    st.success(f"Real data loaded · {len(merged):,} sentences · {len(all_cos)} companies")

    selected = st.multiselect(
        "Companies",
        all_cos,
        default=all_cos[:4],
        format_func=dn,
    )
    if not selected:
        st.warning("Select at least one company.")
        st.stop()

    st.subheader("Weights")
    w_raw = {}
    if len(selected) == 1:
        c = selected[0]
        st.slider(dn(c), 0, 100, 100, 1, disabled=True, key=f"w_{c}_only")
        w_raw[c] = 100
    else:
        remaining = 100
        for i, c in enumerate(selected[:-1]):
            slots_after = len(selected) - i - 2
            default_val = max(0, min(remaining // (slots_after + 2), remaining))
            v = st.slider(dn(c), 0, remaining, int(default_val), 1, key=f"w_{c}")
            w_raw[c] = int(v)
            remaining -= int(v)
        last_c = selected[-1]
        st.slider(dn(last_c), 0, 100, int(remaining), 1, disabled=True, key=f"w_{last_c}_auto")
        w_raw[last_c] = int(remaining)

    st.caption(f"Weight total: {sum(w_raw.values())}%")
    weights = {c: v / 100 for c, v in w_raw.items()}

    st.subheader("Period")
    period_mode = st.selectbox("Scope", ["All-time", "Specific quarter"], index=0)
    if period_mode == "Specific quarter":
        sel_year = st.selectbox("Year", list(reversed(avail_yrs)))
        sel_q = st.selectbox("Quarter", ["Q4", "Q3", "Q2", "Q1"])
        period_label = f"{sel_year} {sel_q}"
    else:
        sel_year = None
        sel_q = None
        period_label = "All-time"

    sel_topic = st.selectbox("Topic filter", all_topics)


# ── Header ───────────────────────────────────────────────────────────────────
st.title("MD&A Sentiment Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Corpus EDA", "Topic Trends", "Peer Comparison", "Filing Drill-Down", "Topic Modeling"]
)


# ── Tab 1: Corpus EDA ────────────────────────────────────────────────────────
with tab1:
    st.subheader("Portfolio Overview")
    p_now   = portfolio_score(weights, sel_year, sel_q)
    p_prev  = portfolio_score(weights, sel_year - 1, sel_q) if sel_year else np.nan
    p_delta = p_now - p_prev if not (np.isnan(p_now) or np.isnan(p_prev)) else np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Portfolio score", fmt_score(p_now), None if np.isnan(p_delta) else fmt_score(p_delta))
    m2.metric("Companies selected", str(len(selected)))
    m3.metric("Period", period_label)
    m4.metric("Topic", dn(sel_topic))

    c1, c2 = st.columns(2)

    with c1:
        ctrl1, ctrl2 = st.columns([1.1, 0.9])
        t1_line_mode = ctrl1.selectbox("Line chart filter", ["All", "Top", "Bottom"], key="t1_line_mode")
        max_line_n   = max(1, min(10, len(selected)))
        t1_line_n    = ctrl2.slider("N companies", 1, max_line_n, min(5, max_line_n), key="t1_line_n") if max_line_n > 1 else 1

        ranked_all      = ranked_company_scores(selected, sel_year, sel_q)
        ranked_filtered = apply_top_bottom(ranked_all, t1_line_mode, t1_line_n)
        companies_to_plot = ranked_filtered["company"].tolist() if len(ranked_filtered) else selected

        periods = (
            merged[merged["company_name"].isin(companies_to_plot)]
            .groupby(["year", "quarter", "company_name"])["sentiment_score"]
            .mean()
            .reset_index()
        )
        if len(periods):
            periods["period_key"] = periods["year"] * 10 + periods["quarter"].map(
                {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
            )
            fig = go.Figure()
            for c in companies_to_plot:
                sub = periods[periods["company_name"] == c].sort_values("period_key")
                if not len(sub):
                    continue
                fig.add_trace(go.Scatter(
                    x=sub["year"].astype(str) + " " + sub["quarter"],
                    y=sub["sentiment_score"],
                    name=dn(c),
                    mode="lines+markers",
                    line=dict(color=co_colors.get(c, "#1f77b4"), width=2),
                    marker=dict(size=5),
                ))
            fig.update_layout(height=420, yaxis_title="Sentiment", xaxis_title="Period")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timeline data available.")

    with c2:
        ctrl3, ctrl4 = st.columns([1.1, 0.9])
        t1_topic_mode = ctrl3.selectbox("Topic table filter", ["All", "Top", "Bottom"], key="t1_topic_mode")
        max_topics    = max(1, min(10, len(all_topics)))
        t1_topic_n    = ctrl4.slider("N topics", 1, max_topics, min(5, max_topics), key="t1_topic_n") if max_topics > 1 else 1

        p_ts = portfolio_topic_scores(weights, sel_year, sel_q)
        if len(p_ts):
            show = p_ts.sort_values(ascending=False).reset_index()
            show.columns = ["Topic", "Score"]
            if t1_topic_mode == "Top":
                show = show.head(t1_topic_n)
            elif t1_topic_mode == "Bottom":
                show = show.tail(t1_topic_n).sort_values("Score")
            st.dataframe(show, use_container_width=True, height=420)
        else:
            st.info("No topic scores available for this period.")


# ── Tab 2: Topic Trends ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Topic Trends")
    t2_co   = st.selectbox("Company", selected, format_func=dn, key="t2_co")
    co_data = merged[merged["company_name"] == t2_co]

    sc_now  = co_score(t2_co, sel_year, sel_q)
    sc_prev = co_score(t2_co, sel_year - 1, sel_q) if sel_year else np.nan
    yoy     = sc_now - sc_prev if not (np.isnan(sc_now) or np.isnan(sc_prev)) else np.nan

    m1, m2, m3 = st.columns(3)
    m1.metric("Latest score", fmt_score(sc_now), None if np.isnan(yoy) else fmt_score(yoy))
    m2.metric("Topic", dn(sel_topic))
    m3.metric("Company", dn(t2_co))

    t2_trend_view = st.selectbox(
        "Trend chart filter", ["Both", "Topic only", "Overall only"], key="t2_trend_view"
    )

    def _period_df(df):
        if not len(df):
            return df
        df = df.copy()
        df["period_key"] = df["year"] * 10 + df["quarter"].map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4})
        df["period"]     = df["year"].astype(str) + " " + df["quarter"]
        return df.sort_values("period_key")

    trend   = _period_df(co_data[co_data["topic_label"] == sel_topic]
                         .groupby(["year", "quarter"])["sentiment_score"].mean().reset_index())
    overall = _period_df(co_data.groupby(["year", "quarter"])["sentiment_score"].mean().reset_index())

    fig_t = go.Figure()
    if len(overall) and t2_trend_view in ["Both", "Overall only"]:
        fig_t.add_trace(go.Scatter(
            x=overall["period"], y=overall["sentiment_score"],
            name="Overall avg", mode="lines",
            line=dict(color="#7f7f7f", width=1.8, dash="dot"),
        ))
    if len(trend) and t2_trend_view in ["Both", "Topic only"]:
        fig_t.add_trace(go.Scatter(
            x=trend["period"], y=trend["sentiment_score"],
            name=dn(sel_topic), mode="lines+markers",
            line=dict(color="#1f77b4", width=2.5), marker=dict(size=6),
        ))
    fig_t.update_layout(height=430, yaxis_title="Sentiment", xaxis_title="Period")
    st.plotly_chart(fig_t, use_container_width=True)

    dist = co_data.groupby(["year", "finbert_label"]).size().reset_index(name="n")
    if len(dist):
        dist["total"] = dist.groupby("year")["n"].transform("sum")
        dist["pct"]   = dist["n"] / dist["total"] * 100

        label_colors = {"positive": "#1f77b4", "neutral": "#7f7f7f", "negative": "#9467bd"}
        fig_bar = go.Figure()
        for lbl in ["positive", "neutral", "negative"]:
            sub = dist[dist["finbert_label"] == lbl]
            fig_bar.add_trace(go.Bar(
                x=sub["year"], y=sub["pct"],
                name=lbl.capitalize(), marker_color=label_colors[lbl],
            ))
        fig_bar.update_layout(barmode="stack", height=330, yaxis_title="% of sentences")
        st.plotly_chart(fig_bar, use_container_width=True)


# ── Tab 3: Peer Comparison ───────────────────────────────────────────────────
with tab3:
    st.subheader(f"Peer Comparison — {dn(sel_topic)}")

    pctrl1, pctrl2 = st.columns([1.1, 0.9])
    peer_line_mode  = pctrl1.selectbox("Line chart filter", ["All", "Top", "Bottom"], key="peer_line_mode")
    max_peer_n      = max(1, min(10, len(selected)))
    peer_line_n     = pctrl2.slider("N companies (peer line)", 1, max_peer_n, min(5, max_peer_n), key="peer_line_n") if max_peer_n > 1 else 1

    peer_rank     = ranked_company_scores(selected, sel_year, sel_q, topic=sel_topic)
    peer_line_df  = apply_top_bottom(peer_rank, peer_line_mode, peer_line_n)
    peer_companies = peer_line_df["company"].tolist() if len(peer_line_df) else selected

    fig_peer = go.Figure()
    for c in peer_companies:
        sub = (
            merged[(merged["company_name"] == c) & (merged["topic_label"] == sel_topic)]
            .groupby(["year", "quarter"])["sentiment_score"].mean().reset_index()
        )
        if not len(sub):
            continue
        sub["period_key"] = sub["year"] * 10 + sub["quarter"].map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4})
        sub["period"]     = sub["year"].astype(str) + " " + sub["quarter"]
        sub = sub.sort_values("period_key")
        fig_peer.add_trace(go.Scatter(
            x=sub["period"], y=sub["sentiment_score"],
            name=dn(c), mode="lines+markers",
            line=dict(color=co_colors.get(c, "#1f77b4"), width=2), marker=dict(size=5),
        ))
    fig_peer.update_layout(height=470, yaxis_title="Sentiment", xaxis_title="Period")
    st.plotly_chart(fig_peer, use_container_width=True)

    rctrl1, rctrl2 = st.columns([1.1, 0.9])
    peer_rank_mode = rctrl1.selectbox("Ranking filter", ["All", "Top", "Bottom"], key="peer_rank_mode")
    max_rank_n     = max(1, min(10, len(selected)))
    peer_rank_n    = rctrl2.slider("N companies (ranking)", 1, max_rank_n, min(5, max_rank_n), key="peer_rank_n") if max_rank_n > 1 else 1

    rank_df = ranked_company_scores(selected, sel_year, sel_q, topic=sel_topic)
    if len(rank_df):
        rank_df = apply_top_bottom(rank_df, peer_rank_mode, peer_rank_n).sort_values("score", ascending=False)
        fig_rank = go.Figure(go.Bar(
            x=rank_df["score"], y=rank_df["display"], orientation="h",
            marker_color=[COMPANY_PALETTE[i % len(COMPANY_PALETTE)] for i in range(len(rank_df))],
            text=[f"{v:+.2f}" for v in rank_df["score"]], textposition="outside",
        ))
        fig_rank.update_layout(height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_rank, use_container_width=True)

    yoy_rows = []
    for c in selected:
        now  = co_score(c, sel_year, sel_q)
        prev = co_score(c, sel_year - 1, sel_q) if sel_year else np.nan
        delta = now - prev if not (np.isnan(now) or np.isnan(prev)) else np.nan
        yoy_rows.append({"Company": dn(c), "YoY delta": delta})
    st.dataframe(pd.DataFrame(yoy_rows), use_container_width=True)


# ── Tab 4: Filing Drill-Down ─────────────────────────────────────────────────
with tab4:
    st.subheader("Filing Drill-Down")

    fc1, fc2, fc3 = st.columns(3)
    f_co = fc1.selectbox("Company", all_cos, format_func=dn, key="t4_co")
    f_yr = fc2.selectbox("Year", list(reversed(avail_yrs)), key="t4_yr")
    f_q  = fc3.selectbox("Quarter", ["Q4", "Q3", "Q2", "Q1"], key="t4_q")

    fd = merged[
        (merged["company_name"] == f_co) &
        (merged["year"] == f_yr) &
        (merged["quarter"] == f_q)
    ].copy()

    if not len(fd):
        st.info(f"No data for {dn(f_co)} {f_yr} {f_q}.")
    else:
        f_score = float(fd["sentiment_score"].mean())
        f_pos   = float(fd["pos_prob"].mean())
        f_neu   = float(fd["neu_prob"].mean())
        f_neg   = float(fd["neg_prob"].mean())

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Overall score", fmt_score(f_score))
        m2.metric("Positive",  f"{f_pos:.1%}")
        m3.metric("Neutral",   f"{f_neu:.1%}")
        m4.metric("Negative",  f"{f_neg:.1%}")
        m5.metric("Sentences", f"{len(fd):,}")

        topic_df = (
            fd.groupby("topic_label")["sentiment_score"]
            .mean().sort_values(ascending=False).reset_index()
        )
        topic_df.columns = ["Topic", "Score"]

        if len(topic_df):
            fctrl1, fctrl2 = st.columns([1.1, 0.9])
            t4_topic_mode = fctrl1.selectbox("Topic breakdown filter", ["All", "Top", "Bottom"], key="t4_topic_mode")
            max_t4        = max(1, min(10, len(topic_df)))
            t4_topic_n    = fctrl2.slider("N topics (filing)", 1, max_t4, min(5, max_t4), key="t4_topic_n") if max_t4 > 1 else 1
            if t4_topic_mode == "Top":
                topic_df = topic_df.head(t4_topic_n)
            elif t4_topic_mode == "Bottom":
                topic_df = topic_df.tail(t4_topic_n).sort_values("Score")

        left, right = st.columns([1, 1.4])
        with left:
            st.write("Topic breakdown")
            st.dataframe(topic_df, use_container_width=True, height=360)

        with right:
            n_show = st.slider("Sentences to show", 4, 20, 10, key="t4_n")
            n_pos, n_neg = n_show // 2, n_show - n_show // 2
            show_sents = pd.concat([
                fd.nlargest(n_pos, "sentiment_score"),
                fd.nsmallest(n_neg, "sentiment_score"),
            ]).sort_values("sentiment_score", ascending=False)
            out = show_sents[["topic_label", "sentiment_score", "text"]].copy()
            out.columns = ["Topic", "Score", "Sentence"]
            out["Score"] = out["Score"].map(lambda v: f"{v:+.2f}")
            st.dataframe(out, use_container_width=True, height=420)

        fig_hist = go.Figure(go.Histogram(x=fd["sentiment_score"], nbinsx=35, marker_color="#4c78a8"))
        fig_hist.update_layout(height=300, xaxis_title="Sentiment score", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)


# ── Tab 5: Topic Modeling ────────────────────────────────────────────────────
with tab5:
    st.subheader("Topic Modeling Outputs")

    tm_c1, tm_c2 = st.columns([1, 1])
    tm_topic_mode = tm_c1.selectbox("Topic list filter", ["All", "Top", "Bottom"], key="tm_topic_mode")
    max_tm        = max(1, min(15, len(all_topics)))
    tm_n_topics   = tm_c2.slider("N topics", 1, max_tm, min(8, max_tm), key="tm_n_topics") if max_tm > 1 else 1

    tm_df = merged[merged["company_name"].isin(selected)].copy()
    if sel_year is not None:
        tm_df = tm_df[tm_df["year"] == sel_year]
    if sel_q is not None:
        tm_df = tm_df[tm_df["quarter"] == sel_q]

    if not len(tm_df):
        st.info("No topic data available for the selected filters.")
    else:
        topic_stats = (
            tm_df.groupby("topic_label")
            .agg(
                docs=("sentence_id", "count"),
                avg_sentiment=("sentiment_score", "mean"),
                avg_topic_prob=("topic_prob", "mean"),
            )
            .reset_index()
            .rename(columns={"topic_label": "Topic"})
            .sort_values("docs", ascending=False)
        )

        if tm_topic_mode == "Top":
            topic_stats = topic_stats.head(tm_n_topics)
        elif tm_topic_mode == "Bottom":
            topic_stats = topic_stats.tail(tm_n_topics).sort_values("docs")

        if not len(topic_stats):
            st.info("No topics after filtering.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Topics shown",        f"{len(topic_stats)}")
            m2.metric("Sentences in scope",  f"{len(tm_df):,}")
            m3.metric("Companies in scope",  f"{tm_df['company_name'].nunique()}")

            fig_bubble = go.Figure(go.Scatter(
                x=topic_stats["docs"],
                y=topic_stats["avg_sentiment"],
                mode="markers+text",
                text=topic_stats["Topic"],
                textposition="top center",
                marker=dict(
                    size=np.clip(topic_stats["avg_topic_prob"].fillna(0.3) * 65, 10, 60),
                    color=topic_stats["avg_sentiment"],
                    colorscale="RdBu", reversescale=True, showscale=True,
                    colorbar=dict(title="Avg sentiment"),
                    line=dict(width=1, color="#333"), opacity=0.85,
                ),
                hovertemplate="Topic: %{text}<br>Docs: %{x}<br>Avg sentiment: %{y:.3f}<extra></extra>",
            ))
            fig_bubble.update_layout(
                height=460,
                xaxis_title="Topic prevalence (sentence count)",
                yaxis_title="Average sentiment",
            )
            st.plotly_chart(fig_bubble, use_container_width=True)

            left_tm, right_tm = st.columns([1.15, 1.25])

            with left_tm:
                st.write("Top words per topic")
                word_n = st.slider("Words per topic", 5, 20, 10, key="tm_word_n")
                kw_df  = topic_keyword_table(tm_df, topic_stats["Topic"].tolist(), top_n_words=word_n)
                kw_df  = kw_df.merge(
                    topic_stats[["Topic", "docs", "avg_sentiment"]], on="Topic", how="left"
                ).rename(columns={"docs": "Docs", "avg_sentiment": "Avg sentiment"})
                st.dataframe(kw_df, use_container_width=True, height=420)

            with right_tm:
                st.write("Company-topic distribution")
                max_dist = max(1, min(12, len(topic_stats)))
                top_topic_n = st.slider("Topics in distribution", 1, max_dist, min(6, max_dist), key="tm_dist_n") if max_dist > 1 else 1
                top_topics  = topic_stats["Topic"].head(top_topic_n).tolist()
                dist = (
                    tm_df[tm_df["topic_label"].isin(top_topics)]
                    .groupby(["company_name", "topic_label"])["sentence_id"]
                    .count().reset_index(name="n")
                )
                if len(dist):
                    dist["share"] = dist["n"] / dist.groupby("company_name")["n"].transform("sum") * 100
                    fig_dist = go.Figure()
                    for t in top_topics:
                        sub = dist[dist["topic_label"] == t]
                        fig_dist.add_trace(go.Bar(
                            x=[dn(c) for c in sub["company_name"]],
                            y=sub["share"], name=dn(t),
                        ))
                    fig_dist.update_layout(barmode="stack", height=420, yaxis_title="Topic share (%)")
                    st.plotly_chart(fig_dist, use_container_width=True)
                else:
                    st.info("No company-topic distribution for selected settings.")
