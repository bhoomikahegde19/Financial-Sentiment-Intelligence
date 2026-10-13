# Financial Sentiment Intelligence

An end-to-end NLP prototype for Case Study 39. It analyzes synthetic news, earnings-call text, regulatory filings, and analyst notes to estimate market sentiment and expose the evidence behind each prediction.

## What changed

- Timestamped financial text corpus with company, sector, source, topic, and sentiment labels.
- TF-IDF word and phrase features plus source/topic metadata.
- Positive, Neutral, and Negative multiclass models.
- Publication-date out-of-time evaluation with macro F1 and class-balanced reporting.
- Document-level evidence phrases, confidence scores, sarcasm cues, and misinformation warnings.
- Streamlit workspace with Market Pulse, Document Analyzer, Company Lens, and Evaluation & Governance views.

## Run it

```powershell
pip install -r requirements.txt
python train.py
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

Run the tests with:

```powershell
python -m pytest tests/test_pipeline.py -v
```

## Project layout

```text
app.py                         Streamlit sentiment workspace
train.py                       Corpus generation and model serialization
data/synthetic_financial_sentiment.csv
models/sentiment_system.joblib
src/data_generator.py          Timestamped financial text generator
src/features.py                Leakage-aware TF-IDF and metadata features
src/models.py                  Multiclass sentiment models
src/evaluation.py              Temporal evaluation and robustness output
src/explainability.py          Phrase evidence and global importance
tests/test_pipeline.py         Sentiment pipeline tests
```

## Responsible-use notes

This is a synthetic demonstration, not investment advice. Sentiment is a noisy proxy for market reaction. Publication dates are used for temporal separation, but real deployments also need duplicate detection, source credibility scoring, multilingual coverage, sarcasm benchmarks, human review, and strict verification of factual claims before downstream decisions.
