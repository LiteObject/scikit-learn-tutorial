# Scikit-Learn Tutorial: Email Spam Detection

Learn scikit-learn by building spam filters instead of analyzing iris flowers.

This project covers classification, regression, and anomaly detection using email examples. The concepts apply to any domain - spam detection just makes it easier to follow.

## What is scikit-learn?

A Python library for machine learning. Classification, regression, anomaly detection, data processing - it handles all of it.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(training_features, training_labels)
prediction = model.predict(new_features)
```

Three lines to train a model. No PhD required.

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

## What you'll build

Three models that analyze emails:
- **Spam Classifier** - is it spam or not?
- **Spam Score Predictor** - how spammy is it (0.0 to 1.0)?
- **Anomaly Detector** - does this email look weird?

Along the way you'll pick up feature engineering, model evaluation, and how to structure an ML project.

**New to ML?** Start with the [docs](docs/) - they explain the basics without jargon.

## Prerequisites

Basic Python (variables, functions, loops). No ML experience needed.

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

1. **Data prep** (`data_generator.py`) - turns email text into numbers the model can use
2. **Training** (`model_trainer.py`) - builds three different models
3. **Evaluation** (`model_evaluator.py`) - confusion matrices, feature importance, the usual
4. **Testing** (`email_spam_ml.py`) - try it on your own examples

## Example

```
> "Congratulations! You won FREE cash!"

Classification: Spam (92% confidence)
Spam Score: 0.87
```

## Troubleshooting

**"No module named sklearn"** - run `pip install scikit-learn`

**Charts not showing** - run `pip install matplotlib seaborn`

**Bad accuracy** - increase `samples_per_class` in data_generator.py

## Next steps

Once you get this, the same patterns work for fraud detection, medical diagnosis, stock prediction - whatever. Try swapping in different algorithms or tweaking the features.

## Docs

- [Training Features and Labels](docs/training-features-and-labels.md)
- [Supervised vs Unsupervised](docs/supervised-vs-unsupervised.md)
- [Classification vs Regression](docs/classification-vs-regression.md)
- [Feature Engineering](docs/feature-engineering.md)
- [Naive Bayes](docs/naive-bayes.md)
- [Scikit-learn official docs](https://scikit-learn.org/stable/)

---

Dig in and experiment. That's how you learn this stuff.
