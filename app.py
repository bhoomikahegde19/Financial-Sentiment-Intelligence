"""Streamlit dashboard for Financial Sentiment Intelligence."""

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

from src.data_generator import generate_sentiment_dataset
from src.evaluation import run_out_of_time_evaluation
from src.explainability import explain_document, get_global_feature_importance
from src.models import SENTIMENT_LABELS, predict_sentiment_probabilities, sentiment_confidence

st.set_page_config(page_title="Financial Sentiment Intelligence", page_icon="S", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#17233b; --muted:#526079; --paper:#edf2f7; --surface:#ffffff; --cobalt:#2d5bff; --apricot:#ff8c69; --mint:#a7e8c5; --line:#cbd5e1; }
html, body, [class*="css"] { font-family:'Manrope', sans-serif; color:var(--ink); }
.stApp { background:var(--paper); }
.block-container { padding-top:2rem; max-width:1380px; }
[data-testid="stSidebar"] { background:#17233b; border-right:0; }
[data-testid="stSidebar"] * { color:#f5f8ff !important; }
[data-testid="stSidebar"] .stCaption { color:#aebbd2 !important; }
.hero { background:linear-gradient(118deg, #17233b 0%, #244a93 68%, #2d5bff 100%); color:#ffffff; padding:2.2rem 2.5rem; border-radius:18px 4px 18px 4px; margin-bottom:1.7rem; box-shadow:0 12px 28px rgba(31,55,100,.15); }
.hero h1 { color:#ffffff !important; font-size:3.15rem; line-height:1; margin:0 0 .8rem; letter-spacing:0; }
.hero p { color:#e8efff !important; max-width:710px; font-size:1.05rem; margin:0; }
.kicker { color:var(--cobalt); font:500 .75rem 'DM Mono', monospace; text-transform:uppercase; letter-spacing:.08em; }
.hero .kicker { color:var(--mint); }
.section { border-top:2px solid var(--ink); padding-top:.8rem; margin-top:1.5rem; }
h2, h3, h4, p, label, .stMarkdown, [data-testid="stMetricLabel"] { color:var(--ink) !important; }
h2 { font-weight:800; margin-top:.25rem; }
h3, h4 { font-weight:700; }
.metric-card { background:var(--surface); border:1px solid var(--line); border-top:4px solid var(--cobalt); padding:1rem 1.1rem; min-height:110px; box-shadow:0 5px 15px rgba(39,61,96,.06); }
.metric-card strong { display:block; font:500 .72rem 'DM Mono', monospace; text-transform:uppercase; color:var(--muted) !important; }
.metric-card b { display:block; color:var(--ink); font-size:2rem; margin-top:.35rem; }
.evidence { border-left:5px solid var(--apricot); background:var(--surface); padding:.85rem 1rem; margin:.45rem 0; box-shadow:0 3px 10px rgba(39,61,96,.06); }
.evidence b { color:var(--cobalt); }
.brief { background:#dce8ff; border:1px solid #b9cbf4; padding:1rem 1.1rem; border-radius:10px; color:var(--ink); }
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div { background:var(--surface); color:var(--ink); border-color:var(--line); }
.stTextArea textarea { line-height:1.55; }
.stButton button { background:var(--cobalt); color:#ffffff; border:0; border-radius:5px; }
.stWarning { background:#fff0d9; color:#6d3c00; }
.small { color:var(--muted); font-size:.9rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    data_path = os.path.join("data", "synthetic_financial_sentiment.csv")
    artifact_path = os.path.join("models", "sentiment_system.joblib")
    if not (os.path.exists(data_path) and os.path.exists(artifact_path)):
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        df = generate_sentiment_dataset()
        df.to_csv(data_path, index=False)
        result = run_out_of_time_evaluation(df, model_name="Logistic Regression")
        result["global_importance"] = get_global_feature_importance(result["fitted_model"], result["fitted_pipeline"].get_feature_names())
        joblib.dump(result, artifact_path)
    else:
        df = pd.read_csv(data_path)
        result = joblib.load(artifact_path)
    return df, result


def score_documents(df, system):
    probabilities = predict_sentiment_probabilities(system["fitted_model"], system["fitted_pipeline"].transform(df))
    predictions = probabilities.argmax(axis=1)
    output = df.copy()
    output["predicted_sentiment"] = [SENTIMENT_LABELS[index] for index in predictions]
    output["confidence"] = sentiment_confidence(probabilities)
    output["sentiment_index"] = (50 + (predictions - 1) * output["confidence"] * 50).round(1)
    return output


def metric_card(label, value):
    st.markdown(f'<div class="metric-card"><strong>{label}</strong><b>{value}</b></div>', unsafe_allow_html=True)


def source_reliability(source):
    return {"Regulatory Filing": "High", "Earnings Call": "High", "Analyst Note": "Medium", "Business News": "Medium"}.get(source, "Unrated")


def signal_brief(row, prediction, confidence):
    reliability = source_reliability(row.source)
    review = "human review recommended" if confidence < 0.75 or row.sarcasm_flag or row.misinformation_flag else "suitable for monitored use"
    return f"{row.company} is reading as {prediction.lower()} with {confidence:.0%} model confidence. The {row.source.lower()} source has {reliability.lower()} reliability; this signal is {review}."


def main():
    df, system = load_system()
    scored = score_documents(df, system)
    st.sidebar.markdown("### FINANCIAL SENTIMENT INTELLIGENCE")
    st.sidebar.caption("Financial Sentiment Intelligence / v3.0")
    page = st.sidebar.radio("Workspace", ["Pulse board", "Read a filing", "Issuer timeline", "Model card"])
    st.sidebar.markdown("---")
    years = st.sidebar.multiselect("Publication years", sorted(scored.year.unique(), reverse=True), default=sorted(scored.year.unique(), reverse=True))
    sources = st.sidebar.multiselect("Sources", sorted(scored.source.unique()), default=sorted(scored.source.unique()))
    view = scored[scored.year.isin(years) & scored.source.isin(sources)].copy()

    st.markdown('<div class="hero"><div class="kicker">Case study 39 / Financial Sentiment Intelligence</div><h1>What is the market hearing?</h1><p>Turn financial language into a reviewable signal. Trace the words, source reliability, publication timing, and uncertainty behind every classification.</p></div>', unsafe_allow_html=True)

    if page == "Pulse board":
        st.markdown('<div class="section"><span class="kicker">01 / signal map</span><h2>Pulse board</h2></div>', unsafe_allow_html=True)
        counts = view.predicted_sentiment.value_counts()
        cols = st.columns(4)
        for col, label, value in zip(cols, ["Documents", "Positive", "Neutral", "Negative"], [len(view), counts.get("Positive", 0), counts.get("Neutral", 0), counts.get("Negative", 0)]):
            with col: metric_card(label, f"{value:,}")
        left, right = st.columns([1.25, 1])
        with left:
            trend = view.groupby(["year", "predicted_sentiment"], as_index=False).size()
            fig = px.area(trend, x="year", y="size", color="predicted_sentiment", markers=True, color_discrete_map={"Positive":"#82b547", "Neutral":"#8a9890", "Negative":"#ee765f"}, labels={"size":"Documents", "year":"Publication year"})
            fig.update_layout(template="simple_white", height=380, legend_title_text="Predicted tone")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            source_view = view.groupby(["source", "predicted_sentiment"], as_index=False).size()
            fig = px.bar(source_view, x="source", y="size", color="predicted_sentiment", barmode="stack", color_discrete_map={"Positive":"#82b547", "Neutral":"#8a9890", "Negative":"#ee765f"}, labels={"size":"Documents"})
            fig.update_layout(template="simple_white", height=380, xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="section"><span class="kicker">attention queue</span><h3>Low-confidence signals</h3></div>', unsafe_allow_html=True)
        st.dataframe(view.sort_values("confidence").head(12)[["publication_date", "company", "source", "predicted_sentiment", "confidence", "text"]], use_container_width=True, hide_index=True)

    elif page == "Read a filing":
        st.markdown('<div class="section"><span class="kicker">02 / inspect one document</span><h2>Read a filing</h2></div>', unsafe_allow_html=True)
        selected = st.selectbox("Choose a corpus document", view.document_id.tolist())
        row = view[view.document_id == selected].iloc[0]
        text = st.text_area("Text under review", row.text, height=150)
        pipeline = system["fitted_pipeline"]
        input_row = row.copy(); input_row["text"] = text
        probabilities = predict_sentiment_probabilities(system["fitted_model"], pipeline.transform(pd.DataFrame([input_row]))) [0]
        prediction = SENTIMENT_LABELS[int(probabilities.argmax())]; confidence = float(probabilities.max())
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Predicted tone", prediction)
        with c2: metric_card("Confidence", f"{confidence:.0%}")
        with c3: metric_card("Source reliability", source_reliability(row.source))
        st.markdown(f'<div class="brief"><b>Signal brief</b><br>{signal_brief(row, prediction, confidence)}</div>', unsafe_allow_html=True)
        if row.sarcasm_flag: st.warning("Sarcasm cue detected. Polarity may be inverted by irony.")
        if row.misinformation_flag: st.warning("Verification warning. Treat this document as an unconfirmed claim until corroborated.")
        st.markdown("#### Evidence phrases")
        for item in explain_document(text, prediction, confidence): st.markdown(f'<div class="evidence"><b>{item["feature"]}</b> · {item["narrative"]}</div>', unsafe_allow_html=True)

    elif page == "Issuer timeline":
        st.markdown('<div class="section"><span class="kicker">03 / temporal alignment</span><h2>Issuer timeline</h2></div>', unsafe_allow_html=True)
        company = st.selectbox("Company", sorted(view.company.unique()))
        company_df = view[view.company == company].sort_values("publication_date")
        c1, c2 = st.columns([1.4, 1])
        with c1:
            fig = px.scatter(company_df, x="publication_date", y="sentiment_index", color="predicted_sentiment", size="confidence", hover_data=["source", "topic"], color_discrete_map={"Positive":"#82b547", "Neutral":"#8a9890", "Negative":"#ee765f"}, labels={"sentiment_index":"Sentiment index (-100 to +100)", "publication_date":"Publication date"})
            fig.update_layout(template="simple_white", height=400, yaxis_range=[-5, 105])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Latest signal")
            latest = company_df.iloc[-1]
            metric_card("Current tone", latest.predicted_sentiment)
            st.write(latest.text)
            st.caption(f"{latest.publication_date} · {latest.source} · {latest.topic}")
        st.dataframe(company_df[["publication_date", "source", "topic", "predicted_sentiment", "confidence", "sarcasm_flag", "misinformation_flag"]], use_container_width=True, hide_index=True)

    else:
        st.markdown('<div class="section"><span class="kicker">04 / trust & model risk</span><h2>Model card</h2></div>', unsafe_allow_html=True)
        train_metrics = system["train_metrics"]; test_metrics = system["oot_test_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        for col, label, value in zip([c1, c2, c3, c4], ["OOT accuracy", "OOT macro F1", "Train documents", "OOT documents"], [test_metrics["accuracy"], test_metrics["macro_f1"], system["n_train"], system["n_test"]]):
            with col: metric_card(label, value)
        st.markdown("#### What the evaluation protects against")
        st.markdown("**Temporal alignment:** training uses earlier publication years and evaluation uses later years. **Leakage control:** TF-IDF vocabulary and metadata encoders are fitted only on the training split. **Ambiguity:** low-confidence and sarcasm-flagged items enter human review. **Misinformation:** public claims are surfaced with a verification warning, never treated as fact.")
        st.markdown("#### Most influential signals")
        st.dataframe(system["global_importance"][["human_name", "importance_pct"]], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
