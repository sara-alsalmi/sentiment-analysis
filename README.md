# Movie Review Sentiment Analysis

This project addresses binary sentiment classification on the IMDB 50K movie reviews dataset. The
work is organized into three progressive versions, each building upon the previous: beginning with
classical TF-IDF features and a Linear SVM, advancing to a fine-tuned DistilBERT, and culminating
in a fine-tuned RoBERTa model.

The best-performing model is the RoBERTa variant (V3), which achieves an F1 score of 0.9581 and
reduces the error rate to 4.22%.

---

## Results at a Glance

| Version | Approach | Accuracy | F1 Score | Error Rate |
|---------|----------|----------|----------|------------|
| V1 | TF-IDF (bigrams) + Linear SVM | 0.9132 | 0.9135 | 8.68% |
| V2 | DistilBERT fine-tuned (3 epochs) | 0.9218 | 0.9224 | 7.82% |
| V3 | RoBERTa-base fine-tuned (4 epochs, max_len=512) | **0.9578** | **0.9581** | **4.22%** |

Each version represents a measurable improvement over the prior. Moving from the TF-IDF baseline to
RoBERTa more than halved the number of misclassified reviews on the 10,000-review test set.

![Model progression across V1, V2, and V3](images/08_version_comparison.png)

Accuracy, precision, recall, and F1 all climb steadily from V1 to V3.

---

## Dataset

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

- 50,000 reviews, split evenly into 25,000 positive and 25,000 negative
- A perfectly balanced binary task, so accuracy is a meaningful headline metric
- Two columns: `review` (raw text) and `sentiment` (`positive` / `negative`)

The dataset is exactly balanced, so no class rebalancing was required prior to training.

---

## Exploratory Data Analysis

Prior to modeling, an exploratory analysis was conducted to examine the structure of the reviews
and the linguistic features that distinguish the two classes.

**Review length.** Most reviews are between roughly 100 and 300 words, with a long tail of very
detailed ones. Positive and negative reviews have almost identical length distributions, so the
length of a review on its own doesn't tell you much about its sentiment.

**Most frequent words.** The top words in both classes are generic film vocabulary (`film`, `movie`,
`one`, `like`). The more interesting differences appear a bit further down the list: words like
`great`, `good`, and `story` rank high in positive reviews, while `bad` and `even` stand out in
negative ones. That's the kind of signal the models end up learning to weight.

For preprocessing, HTML tags are removed, text is lowercased, and stopwords are stripped; however,
negation words such as `not`, `never`, and `didn't` are deliberately retained. A default stopword
list would discard these terms, yet they are precisely the words that invert a review's sentiment.

---

## V1 — TF-IDF + Traditional Machine Learning

The first version is a full classical NLP pipeline: EDA, text cleaning, TF-IDF features, training
four models, evaluation, error analysis, and saving the final model.

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Linear SVM** | **0.9132** | **0.9102** | **0.9168** | **0.9135** |
| Logistic Regression | 0.9076 | 0.9010 | 0.9158 | 0.9084 |
| Multinomial Naive Bayes | 0.8835 | 0.8778 | 0.8910 | 0.8844 |
| Random Forest | 0.8697 | 0.8723 | 0.8662 | 0.8692 |

TF-IDF settings: bigrams, `max_features=50000`, `sublinear_tf=True`, `min_df=2`.

The two linear models clearly do best on the sparse, high-dimensional TF-IDF features. Random Forest
is both the weakest and the slowest to train here, which makes sense given how sparse the input is.

![V1 model comparison](images/04_v1_model_comparison.png)

Linear SVM was selected as the V1 model, as it outperforms Logistic Regression on every reported metric.

**Error analysis.** Of the 10,000 test reviews, the SVM misclassifies approximately 868, with errors
distributed near-evenly between false positives and false negatives, indicating no systematic bias
toward either class.

---

## V2 — DistilBERT Fine-Tuning

V2 replaces the hand-crafted TF-IDF features with a fine-tuned `distilbert-base-uncased` transformer
(66M parameters), allowing the model to learn contextual representations directly from raw text.

Training setup: 3 epochs, `lr=2e-5`, `max_length=256`, batch size 32 on a Tesla T4 GPU (~27 minutes).

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| DistilBERT | 0.9218 | 0.9155 | 0.9294 | 0.9224 |

![DistilBERT confusion matrix](images/06_v2_distilbert_confusion.png)

DistilBERT makes 782 errors on the test set, already a step up from the V1 baseline.

---

## V3 — RoBERTa Fine-Tuning

V3 upgrades to `roberta-base` (125M parameters) and uses the full 512-token context window, so it can
read entire long reviews instead of cutting them off early.

Training setup: 4 epochs, `lr=1e-5`, `max_length=512`, batch size 16 on a Tesla T4 GPU (~150 minutes).
The training loss converged to 0.258, compared with 0.411 in V2.

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **RoBERTa** | **0.9578** | **0.9524** | **0.9638** | **0.9581** |

![RoBERTa confusion matrix](images/07_v3_roberta_confusion.png)

RoBERTa produces only 422 errors on the 10,000-review test set, down from 782 in V2 and 868 in V1.
This constitutes the best-performing model in the project and is used for inference.

---

## Repository Structure

```
Sentiment Analysis/
├── images/                            # result plots used in this README
├── dataset/
│   └── IMDB Dataset.csv               # raw data (not pushed to GitHub)
├── models/
│   ├── v1/                            # Linear SVM + TF-IDF artifacts
│   ├── v2/                            # DistilBERT artifacts
│   └── v3/                            # RoBERTa artifacts (best)
├── sentiment_analysis.ipynb           # V1: TF-IDF + traditional ML
├── sentiment_analysis_v2.ipynb        # V2: DistilBERT fine-tuning
├── sentiment_analysis_v3.ipynb        # V3: RoBERTa fine-tuning
├── predict.py                         # CLI inference script (V3 RoBERTa)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt

# V1 only — download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"
```

The notebooks are designed to run end to end; V2 and V3 require a GPU to complete training within a reasonable timeframe.

---

## Inference

Inference is performed using the V3 RoBERTa model. After training `sentiment_analysis_v3.ipynb`,
extract the exported model into `models/v3/`, then run:

```bash
# Single review
python predict.py "This movie was absolutely fantastic."
# Sentiment  : Positive
# Confidence : 0.9970

# Batch — one review per line
python predict.py --file reviews.txt
```

---

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · NLTK · scikit-learn · Joblib · PyTorch · Hugging Face Transformers
