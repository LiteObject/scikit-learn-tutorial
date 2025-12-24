# Scikit-Learn Tutorial: Learning ML Through Email Spam/Ham Classification

**Learn scikit-learn fundamentals using practical email classification scenarios**

This tutorial teaches you scikit-learn basics by working with realistic examples from email spam detection. Instead of boring datasets like iris flowers, you'll learn classification, regression, and anomaly detection by analyzing email patterns. The goal is to understand machine learning concepts - the spam/ham theme just makes it more interesting.

## What is scikit-learn?

Scikit-learn is basically a Python library that makes machine learning accessible. Think of it like having a Swiss Army knife for AI - it has tools for classification (sorting things into categories), regression (predicting numbers), anomaly detection (finding weird stuff), and data processing.

Here's how simple it is:

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create a model
model = RandomForestClassifier()

# Train it with your data (features must be numerical array)
model.fit(training_features, training_labels)

# Make predictions on new data
prediction = model.predict(new_features)
```

That's it. No PhD required.

In this tutorial, you'll learn scikit-learn by building models that analyze email text data. It's a much more engaging way to learn ML concepts than working with abstract datasets, and you'll see how the same techniques apply to any domain.

## Project files

```
├── README.md                    # You are here
├── docs/                        
│   ├── supervised-vs-unsupervised.md    # ML concepts explained
│   ├── classification-vs-regression.md  # Types of predictions
│   ├── feature-engineering.md           # Data prep basics
│   └── naive-bayes.md                   # Naive Bayes classifier explained
├── data_generator.py            # Creates synthetic email data for training
├── model_trainer.py             # Where the ML magic happens
├── model_evaluator.py           # Tests how good your models are
├── email_spam_ml.py             # Main app you'll actually run
├── test_components.py           # Quick tests to make sure stuff works
├── requirements.txt             # Python packages you need
└── setup scripts               # Makes installation easier
```

## What you'll learn

**Core scikit-learn concepts:**
- Classification (sorting emails into spam vs ham)
- Regression (predicting spam likelihood scores)
- Anomaly detection (finding unusual email patterns)
- Feature engineering (turning raw text into ML-ready format)
- Model training and evaluation
- Performance metrics and visualization

**Practical skills:**
- How to structure an ML project
- Data preprocessing and feature extraction
- Cross-validation and model comparison
- Interpreting results and debugging models

The spam/ham examples help make these concepts concrete, but the techniques work for any domain - network security, stock prediction, medical diagnosis, etc.

**New to machine learning?** Check out the guides in the `docs/` folder first. They explain the basics without the jargon:
- [Training Features and Labels](docs/training-features-and-labels.md) - understand the data format ML needs
- [Supervised vs Unsupervised Learning](docs/supervised-vs-unsupervised.md) 
- [Classification vs Regression](docs/classification-vs-regression.md)
- [Feature Engineering](docs/feature-engineering.md)
- [Naive Bayes Classifier](docs/naive-bayes.md) - a classic algorithm for text classification

## Prerequisites

- Basic Python (variables, functions, loops)
- Know what lists and dictionaries are
- That's it - no ML experience needed

## Getting started

**Easy way (recommended):**

Windows:
```cmd
git clone https://github.com/LiteObject/scikit-learn-tutorial.git
cd scikit-learn-tutorial
setup_windows.bat
```

Mac/Linux:
```bash
git clone https://github.com/LiteObject/scikit-learn-tutorial.git
cd scikit-learn-tutorial
chmod +x setup_unix.sh
./setup_unix.sh
```

**Manual way:**
```bash
python -m venv tutorial_env
source tutorial_env/bin/activate  # Windows: tutorial_env\Scripts\activate
pip install -r requirements.txt
python setup.py
```

## Running the tutorial

Basic version:
```bash
python email_spam_ml.py
```

Step-by-step interactive version (better for learning):
```bash
python email_spam_ml.py --mode interactive
```

Full version with charts and graphs:
```bash
python email_spam_ml.py --mode advanced --visualize
```

Test individual parts:
```bash
python test_components.py
```

## How it works

This tutorial walks you through a complete machine learning workflow using email data as examples:

**Step 1: Data preparation** (`data_generator.py`)
Learn feature engineering by converting email text into numerical features. For example, emails get transformed into feature vectors that capture spam/ham characteristics like word counts, spam word ratio, urgency indicators, etc.

**Step 2: Model training** (`model_trainer.py`)
Build and compare three different types of models:
- Random Forest Classifier (learn classification for spam vs ham)
- Random Forest Regressor (learn regression for spam scores)
- Isolation Forest (learn anomaly detection for unusual emails)

**Step 3: Evaluation** (`model_evaluator.py`)
Master model evaluation with confusion matrices, feature importance analysis, and learning curves. See what's working and what isn't.

**Step 4: Interactive testing** (`email_spam_ml.py`)
Test your trained models on new email examples and understand how they make decisions.

## Example output

When you run it, you'll see something like:

```
Analyzing email: "Congratulations! You won FREE cash!"

AI Analysis Results:
   Classification: Spam (92% confidence)
   Spam Score: 0.87 (HIGH SPAM)
   Anomaly Status: NORMAL
```

## Common issues

**"No module named sklearn"**
```bash
pip install scikit-learn
```

**Charts not showing**
```bash
pip install matplotlib seaborn
```

**Models performing poorly**
Try generating more training data by increasing `samples_per_class` in the data generator.

## What's next?

Once you understand these scikit-learn fundamentals, you can apply them anywhere:

**Try different domains:**
- Network security (device classification, intrusion detection)
- Financial data (stock prediction, fraud detection)
- Medical data (diagnosis, drug discovery)
- Image analysis (object recognition, medical imaging)

**Experiment with the code:**
- Add new email patterns to practice classification
- Create different features to see their impact
- Try other algorithms (SVM, Neural Networks, Naive Bayes)
- Modify the evaluation metrics

The patterns you learn here are universal - once you understand scikit-learn with email classification, you can tackle any machine learning problem.

## Background reading

**New to machine learning?**
- [Training Features and Labels](docs/training-features-and-labels.md) - understand the data format ML needs
- [Supervised vs Unsupervised Learning](docs/supervised-vs-unsupervised.md) - explains the basic types
- [Classification vs Regression](docs/classification-vs-regression.md) - when to use each approach
- [Feature Engineering](docs/feature-engineering.md) - how to prepare your data
- [Naive Bayes Classifier](docs/naive-bayes.md) - a classic text classification algorithm

**Want to learn more about scikit-learn?**
- [Scikit-learn Documentation](https://scikit-learn.org/stable/) - official documentation with tutorials and API reference

---

That's it. Have fun learning scikit-learn through practical examples that actually make sense.
