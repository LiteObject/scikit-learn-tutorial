# Naive Bayes

**A surprisingly simple algorithm that works better than you'd expect**

## What is Naive Bayes?

Naive Bayes is a classification algorithm based on probability. It looks at past data and asks: "Given what I know about this item, what's the most likely category it belongs to?"

Think of it like a doctor making a diagnosis. The patient has certain symptoms, and based on experience with thousands of previous patients, the doctor estimates which illness is most likely.

## Why is it called "Naive"?

Here's the thing: Naive Bayes assumes that the features are independent of each other *once you know the class*. In real life, that's often not true, but the algorithm pretends it is anyway.

For example, if you're classifying emails as spam or not spam, Naive Bayes treats each word independently. It doesn't consider that "free" and "money" appearing together might be more suspicious than either word alone.

This assumption is technically wrong (hence "naive"), but somehow it still works remarkably well in practice. Sometimes good enough beats perfect.

## How does it actually work?

Let's break it down with a simple example.

Imagine you're trying to figure out if someone will buy ice cream based on two things:
- Weather (sunny or rainy)
- Day of week (weekend or weekday)

You've collected data from 100 customers:
- 60 bought ice cream
- 40 didn't buy

Among those who bought:
- 50 out of 60 visited on sunny days
- 40 out of 60 visited on weekends

Among those who didn't buy:
- 10 out of 40 visited on sunny days
- 15 out of 40 visited on weekends

Now a new customer walks in. It's sunny and it's a weekend. Will they buy?

Naive Bayes calculates:
- Probability of buying given sunny weekend
- Probability of not buying given sunny weekend

Then picks whichever is higher. That's the prediction.

## The spam filter example

This is where Naive Bayes really shines. Let's walk through how email spam detection works.

You have two categories: Spam and Not Spam (Ham).

From training data, you learn:
- Words like "free", "winner", "click" appear often in spam
- Words like "meeting", "project", "thanks" appear often in ham

When a new email arrives with the words "Congratulations! You've won free tickets!", Naive Bayes:

1. Looks up how often each word appears in spam vs ham
2. Combines those probabilities together (this is the "naive" part)
3. Compares the final spam score vs ham score
4. Picks the higher one

In real implementations, it usually does this using log-probabilities (adding logs instead of multiplying tiny numbers) to avoid numerical underflow.

```python
# Conceptually, it's doing something like this:
spam_score = P(spam) * P("congratulations"|spam) * P("free"|spam) * P("won"|spam)
ham_score = P(ham) * P("congratulations"|ham) * P("free"|ham) * P("won"|ham)

if spam_score > ham_score:
    prediction = "SPAM"
else:
    prediction = "HAM"
```

## Three flavors of Naive Bayes

Scikit-learn gives you three main options. Pick based on your data type.

### 1. Gaussian Naive Bayes
**Use when**: Your features are continuous numbers (like height, weight, temperature)

```python
from sklearn.naive_bayes import GaussianNB

# Good for: measurements, sensor data, anything with decimal values
model = GaussianNB()
```

### 2. Multinomial Naive Bayes
**Use when**: You're counting things (like word frequencies in text)

```python
from sklearn.naive_bayes import MultinomialNB

# Good for: text classification, document categorization
model = MultinomialNB()
```

### 3. Bernoulli Naive Bayes
**Use when**: Your features are yes/no, true/false, 0/1

```python
from sklearn.naive_bayes import BernoulliNB

# Good for: binary features, "does this word appear?" type problems
model = BernoulliNB()
```

## Complete example: Classifying network activity

Let's build something relevant to our network security project. We'll classify network connections as normal or suspicious.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Generate some sample network data
# Features: [packets_per_second, avg_packet_size, connection_duration, port_count]
np.random.seed(42)

# Normal traffic patterns
normal_traffic = np.random.randn(100, 4) * [10, 50, 30, 2] + [50, 500, 60, 5]

# Suspicious traffic patterns (different characteristics)
suspicious_traffic = np.random.randn(100, 4) * [20, 100, 10, 5] + [200, 100, 5, 20]

# Combine the data
X = np.vstack([normal_traffic, suspicious_traffic])
y = np.array([0] * 100 + [1] * 100)  # 0 = normal, 1 = suspicious

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Check how we did
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nDetailed breakdown:")
print(classification_report(y_test, predictions, target_names=['Normal', 'Suspicious']))
```

Output looks something like:
```
Accuracy: 95.00%

Detailed breakdown:
              precision    recall  f1-score   support

      Normal       0.94      0.97      0.95        30
  Suspicious       0.97      0.93      0.95        30

    accuracy                           0.95        60
   macro avg       0.95      0.95      0.95        60
weighted avg       0.95      0.95      0.95        60
```

## Text classification example

Here's a more classic use case: classifying short messages.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# Sample messages with labels
messages = [
    "Buy now! Limited time offer!",
    "Meeting at 3pm tomorrow",
    "You've won a prize! Click here!",
    "Can you review the project proposal?",
    "Free gift card waiting for you",
    "Let's grab coffee after work",
    "Urgent: Update your password now!",
    "Thanks for your help yesterday",
]

labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam

# Build a simple pipeline: convert text to numbers, then classify
spam_detector = Pipeline([
    ('vectorizer', CountVectorizer()),  # Turns text into word counts
    ('classifier', MultinomialNB())      # Our Naive Bayes model
])

# Train it
spam_detector.fit(messages, labels)

# Test on new messages
new_messages = [
    "Free money waiting for you!",
    "Are you coming to the meeting?",
    "Click here to claim your reward"
]

predictions = spam_detector.predict(new_messages)

for msg, pred in zip(new_messages, predictions):
    status = "SPAM" if pred == 1 else "Not spam"
    print(f"{status}: '{msg}'")
```

Output:
```
SPAM: 'Free money waiting for you!'
Not spam: 'Are you coming to the meeting?'
SPAM: 'Click here to claim your reward'
```

## When to use Naive Bayes

**Great for:**
- Text classification (spam, sentiment, topic categorization)
- When you have limited training data
- When you need fast predictions
- Real-time classification systems
- Multi-class problems with many categories
- As a baseline model to compare against fancier algorithms

**Not ideal for:**
- When features have strong dependencies on each other
- When you need highly accurate probability estimates
- Complex pattern recognition (images, audio)
- When you have lots of data and time (other algorithms might do better)

## Pros and cons

| Pros | Cons |
|------|------|
| Fast to train | Assumes features are independent (not always true) |
| Works with small datasets | Can't learn interactions between features |
| Easy to understand | Probability estimates are often poorly calibrated |
| Handles many features well | Can be outperformed by other algorithms |
| Good at multi-class problems | Sensitive to irrelevant features |
| No parameter tuning needed | |

## Common gotchas

**Zero probability problem**: If a word never appeared in spam during training, Naive Bayes can treat it like it can NEVER appear in spam. Scikit-learn handles this with smoothing for `MultinomialNB`/`BernoulliNB` (enabled by default via `alpha`).

For `GaussianNB`, the stability knob is different (`var_smoothing`), because it models each feature with a Gaussian distribution instead of counts.

**Feature scaling**: It depends on the Naive Bayes type. For `MultinomialNB` and `BernoulliNB`, you typically want non-negative count features or 0/1 features (so standard scaling is usually a bad fit). For `GaussianNB`, scaling isn't required, but changing the scale does change the learned means/variances.

**Class imbalance**: If 99% of your emails are spam, Naive Bayes might just predict "spam" for everything. Balance your training data or adjust class priors.

## Quick comparison with other classifiers

| Algorithm | Speed | Accuracy | Interpretability | Data needed |
|-----------|-------|----------|------------------|-------------|
| Naive Bayes | Very fast | Good | High | Small |
| Random Forest | Medium | Very good | Medium | Medium |
| Logistic Regression | Fast | Good | High | Medium |
| SVM | Slow | Very good | Low | Large |
| Neural Networks | Very slow | Excellent | Very low | Very large |

## Test your understanding

1. You're building a system to classify customer reviews as positive, negative, or neutral. Which Naive Bayes would you use?
   - Answer: MultinomialNB (you're counting words in text)

2. Your dataset has features like temperature, pressure, and humidity (all decimal numbers). Which version fits best?
   - Answer: GaussianNB (continuous numerical features)

3. You want to predict if a network port is open or closed based on binary flags. Best choice?
   - Answer: BernoulliNB (binary yes/no features)

## What's next?

Naive Bayes is a solid starting point for classification tasks, especially with text. It won't always beat more complex models, but it's fast, simple, and often good enough.

