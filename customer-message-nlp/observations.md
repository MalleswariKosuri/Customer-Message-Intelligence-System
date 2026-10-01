# NLP Customer Message Classifier — Observations

## Project Files

- `nlp_classifier.py` — classifies customer-support messages.
- `rule_extractor.py` — extracts order IDs from messages.
- `support_messages.csv` — contains messages and their categories.

# **Step 0: Install and import the required packages**

Install the external packages once:

```bash
pip install pandas scikit-learn matplotlib
```

Import the required libraries:

```python
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

**Observation:** `pandas`, `scikit-learn`, and `matplotlib` must be installed using `pip`. The `re` module is included with Python and does not require installation.


## **Step 1: Load the dataset**

The CSV file is loaded using Pandas. The first few rows, total number of records, and category counts are displayed.

**Observation:** This helps confirm that the dataset was loaded correctly and shows whether the categories are balanced.

## **Step 2: Split the dataset**

The messages are divided into:

- **75% training data**
- **25% testing data**

`stratify=y` keeps a similar category distribution in both sets. `random_state=42` produces the same split whenever the program runs.

## **Step 3: Build the unigram model**

The first model uses:

```python
ngram_range=(1, 1)
```

A unigram represents one word, such as `payment`, `login`, or `refund`.

TF-IDF converts these words into numbers, and Logistic Regression learns how they relate to the categories.

## **Step 4: Evaluate the unigram model**

The model predicts categories for the test messages. `accuracy_score()` compares the predicted categories with the actual categories.

**Unigram accuracy:** Record the value produced when the program runs.

## **Step 5: Build the unigram and bigram model**

The second model uses:

```python
ngram_range=(1, 2)
```

It considers individual words and two-word phrases.

Examples:

- Unigram: `payment`
- Bigram: `payment failed`

**Observation:** Bigrams provide more context and may help the model understand phrases such as `reset password` or `money back`.

## **Step 6: Compare the models**

The two accuracy scores are displayed in a bar chart.

**Observation:** The taller bar represents the model with better test accuracy. Both models must use the same test data for a fair comparison.

| Model | Accuracy |
|---|---:|
| Unigram | Record after running |
| Unigram + bigram | Record after running |

## **Step 7: Test new messages**

Both models predict categories for new customer messages.

**Observation:** If the models return different categories, the bigram model may be using phrase-level information that the unigram model cannot capture.

## **Step 8: Inspect prediction confidence**

`predict_proba()` returns a score for every possible category.

```python
key=lambda item: item[1]
```

Each item contains:

```python
(category, probability)
```

Therefore, `item[1]` selects the probability. `reverse=True` sorts the categories from the highest score to the lowest.

**Observation:** The category with the highest probability becomes the prediction. A high score shows confidence, but it does not guarantee correctness.

## **Step 9: Test difficult messages**

The stress test includes:

- Messages containing multiple problems
- Sarcastic messages
- Telugu-English mixed messages
- Vague requests

**Observation:** TF-IDF may struggle with sarcasm, mixed languages, and messages that contain more than one issue.

## Rule-Based Order ID Extraction

`rule_extractor.py` uses:

```python
re.findall(r"ORD-\d+", message)
```

It extracts order IDs such as:

```text
ORD-58329
```

If no order ID is found, it returns:

```python
[]
```

For exactly five digits, a stricter pattern can be used:

```python
r"\bORD-\d{5}\b"
```

## Suggested Corrections

- Rename `new_messges` to `new_messages`.
- Change the repeated **Step 5** to **Step 6**.
- Import each library only once.
- Add `.venv/` to `.gitignore`.
- Record the actual accuracy values after running the program.

## Conclusion

This project demonstrates two NLP approaches:

- **Machine learning** classifies customer messages using TF-IDF and Logistic Regression.
- **Regular expressions** extract order IDs using a fixed pattern.

Unigrams identify important individual words, while bigrams provide additional context through two-word phrases. The better model should be selected using the actual test results.