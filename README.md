# Scikit-Learn Tutorial: Learning ML Through Network Security Examples

**Learn scikit-learn fundamentals using practical network security scenarios**

This tutorial teaches you scikit-learn basics by working with realistic examples from network security. Instead of boring datasets like iris flowers, you'll learn classification, regression, and anomaly detection by analyzing network devices and traffic patterns. The goal is to understand machine learning concepts - the network security theme just makes it more interesting.

## What is scikit-learn?

Scikit-learn is basically a Python library that makes machine learning accessible. Think of it like having a Swiss Army knife for AI - it has tools for classification (sorting things into categories), regression (predicting numbers), anomaly detection (finding weird stuff), and data processing.

Here's how simple it is:

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create a model
model = RandomForestClassifier()

# Train it with your data (features must be numerical array
model.fit(training_features, training_labels)

# Make predictions on new data
prediction = model.predict(new_features)
```

That's it. No PhD required.

In this tutorial, you'll learn scikit-learn by building models that analyze network data. It's a much more engaging way to learn ML concepts than working with abstract datasets, and you'll see how the same techniques apply to any domain.

## Project files

```
├── README.md                    # You are here
├── docs/                        
│   ├── supervised-vs-unsupervised.md    # ML concepts explained
│   ├── classification-vs-regression.md  # Types of predictions
│   └── feature-engineering.md           # Data prep basics
├── data_generator.py            # Creates fake network data for training
├── model_trainer.py             # Where the ML magic happens
├── model_evaluator.py           # Tests how good your models are
├── network_security_ml.py       # Main app you'll actually run
├── test_components.py           # Quick tests to make sure stuff works
├── requirements.txt             # Python packages you need
└── setup scripts               # Makes installation easier
```

## What you'll learn

**Core scikit-learn concepts:**
- Classification (sorting things into categories)
- Regression (predicting numerical values)
- Anomaly detection (finding unusual patterns)
- Feature engineering (turning raw data into ML-ready format)
- Model training and evaluation
- Performance metrics and visualization

**Practical skills:**
- How to structure an ML project
- Data preprocessing and feature extraction
- Cross-validation and model comparison
- Interpreting results and debugging models

The network security examples help make these concepts concrete, but the techniques work for any domain - spam detection, stock prediction, medical diagnosis, etc.

**New to machine learning?** Check out the guides in the `docs/` folder first. They explain the basics without the jargon:
- [Training Features and Labels](docs/training-features-and-labels.md) - understand the data format ML needs
- [Supervised vs Unsupervised Learning](docs/supervised-vs-unsupervised.md) 
- [Classification vs Regression](docs/classification-vs-regression.md)
- [Feature Engineering](docs/feature-engineering.md)

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
python network_security_ml.py
```

Step-by-step interactive version (better for learning):
```bash
python network_security_ml.py --mode interactive
```

Full version with charts and graphs:
```bash
python network_security_ml.py --mode advanced --visualize
```

Test individual parts:
```bash
python test_components.py
```

## How it works

This tutorial walks you through a complete machine learning workflow using network data as examples:

**Step 1: Data preparation** (`data_generator.py`)
Learn feature engineering by converting network port data into numerical features. For example, ports [22, 80, 443] become a feature vector that captures device characteristics.

**Step 2: Model training** (`model_trainer.py`)
Build and compare three different types of models:
- Random Forest Classifier (learn classification)
- Random Forest Regressor (learn regression)
- Isolation Forest (learn anomaly detection)

**Step 3: Evaluation** (`model_evaluator.py`)
Master model evaluation with confusion matrices, feature importance analysis, and learning curves. See what's working and what isn't.

**Step 4: Interactive testing** (`network_security_ml.py`)
Test your trained models on new examples and understand how they make decisions.

## Example output

When you run it, you'll see something like:

```
Analyzing device with ports: [22, 80, 443, 3389]

AI Analysis Results:
   Device Type: Linux Server (89% confidence)
   Risk Score: 0.42 (MEDIUM risk)
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
- Text analysis (spam detection, sentiment analysis)
- Financial data (stock prediction, fraud detection)
- Medical data (diagnosis, drug discovery)
- Image analysis (object recognition, medical imaging)

**Experiment with the code:**
- Add new device types to practice classification
- Create different features to see their impact
- Try other algorithms (SVM, Neural Networks, etc.)
- Modify the evaluation metrics

The patterns you learn here are universal - once you understand scikit-learn with network data, you can tackle any machine learning problem.

## Background reading

**New to machine learning?**
- [Training Features and Labels](docs/training-features-and-labels.md) - understand the data format ML needs
- [Supervised vs Unsupervised Learning](docs/supervised-vs-unsupervised.md) - explains the basic types
- [Classification vs Regression](docs/classification-vs-regression.md) - when to use each approach
- [Feature Engineering](docs/feature-engineering.md) - how to prepare your data

**Want to learn more about scikit-learn?**
- [Scikit-learn Documentation](https://scikit-learn.org/stable/) - official documentation with tutorials and API reference

---

That's it. Have fun learning scikit-learn through practical examples that actually make sense.
