# Scikit-Learn Learning Roadmap

A practical guide to becoming comfortable with scikit-learn, organized from foundational to advanced topics.

## Where You Are Now

Based on the spam detection example and network security project, you already understand:
- Building a model pipeline
- Feature extraction (CountVectorizer)
- Classification (MultinomialNB, RandomForestClassifier)
- Making predictions
- Basic evaluation (accuracy, classification report)

## The Roadmap

### Level 1: Evaluation Fundamentals

Before improving models, you need to measure them properly.

#### Train/Test Split
Separate your data so you can test on examples the model has never seen.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducible results
    stratify=y          # Balanced split across classes
)
```

**Already used in:** [model_trainer.py](../model_trainer.py) - `step_1_generate_and_explore_data()`

#### Evaluation Metrics
Different metrics tell you different things about your model.

| Metric | What it measures | Use when |
|--------|------------------|----------|
| Accuracy | Overall correct predictions | Classes are balanced |
| Precision | "Of predicted positives, how many were right?" | False positives are costly (spam filter) |
| Recall | "Of actual positives, how many did we catch?" | False negatives are costly (disease detection) |
| F1 Score | Balance of precision and recall | You need both |
| Confusion Matrix | Breakdown of all predictions | Understanding error patterns |

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Get all metrics at once
print(classification_report(y_test, predictions))
```

**Already used in:** [model_trainer.py](../model_trainer.py) - `step_2_build_device_classifier()`

---

### Level 2: Cross-Validation

A single train/test split can be misleading. Cross-validation gives a more reliable estimate.

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

**How it works:**
1. Split data into 5 parts (folds)
2. Train on 4 folds, test on 1
3. Repeat 5 times, each fold gets a turn as test set
4. Average the results

---

### Level 3: Feature Preprocessing

Many algorithms work better when features are on similar scales.

#### StandardScaler
Centers data around 0 with standard deviation of 1.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Learn + transform
X_test_scaled = scaler.transform(X_test)        # Only transform
```

**Already used in:** [model_trainer.py](../model_trainer.py) - feature scaling before training

#### MinMaxScaler
Scales features to a range (usually 0-1).

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

#### When to scale?
| Algorithm | Needs scaling? |
|-----------|---------------|
| Naive Bayes | Usually no |
| Random Forest | No |
| Logistic Regression | Yes |
| SVM | Yes |
| KNN | Yes |
| Neural Networks | Yes |

---

### Level 4: Other Classifiers

Try different algorithms and compare results.

#### Logistic Regression
Simple, fast, interpretable. Good baseline.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
```

#### Support Vector Machine (SVM)
Powerful for complex boundaries. Works well with scaling.

```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', probability=True)
model.fit(X_train_scaled, y_train)
```

#### K-Nearest Neighbors (KNN)
Simple concept: classify based on closest training examples.

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)
```

---

### Level 5: Regression

Predicting continuous values instead of categories.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, predictions):.3f}")
print(f"MSE: {mean_squared_error(y_test, predictions):.3f}")
print(f"R²:  {r2_score(y_test, predictions):.3f}")
```

**Already used in:** [model_trainer.py](../model_trainer.py) - `step_4_build_risk_predictor()`

---

### Level 6: Hyperparameter Tuning

Find the best settings for your model automatically.

#### GridSearchCV
Tries all combinations of parameters you specify.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1  # Use all CPU cores
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_
```

#### RandomizedSearchCV
Faster alternative when you have many parameters.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,  # Try 50 random combinations
    cv=5,
    random_state=42
)

random_search.fit(X_train, y_train)
```

---

### Level 7: Handling Categorical Data

Convert text categories to numbers.

#### LabelEncoder
For target variables (y).

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(['spam', 'ham', 'spam', 'ham'])
# Result: [1, 0, 1, 0]

# Convert back
original = le.inverse_transform([1, 0])
# Result: ['spam', 'ham']
```

#### OneHotEncoder
For features with no inherent order.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
# 'red', 'green', 'blue' becomes:
# [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
```

---

### Level 8: Model Persistence

Save and load trained models.

```python
import joblib

# Save
joblib.dump(model, 'my_model.joblib')
joblib.dump(scaler, 'my_scaler.joblib')

# Load
loaded_model = joblib.load('my_model.joblib')
loaded_scaler = joblib.load('my_scaler.joblib')

# Use
X_scaled = loaded_scaler.transform(new_data)
predictions = loaded_model.predict(X_scaled)
```

---

### Level 9: Pipelines

Chain preprocessing and model into one object.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Now just one call for everything
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

# Save the entire pipeline
joblib.dump(pipeline, 'complete_pipeline.joblib')
```

**Already used in:** [naive_bayes_spam_example.py](../naive_bayes_spam_example.py)

---

### Level 10: Unsupervised Learning

When you don't have labels.

#### K-Means Clustering
Group similar data points together.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
```

#### PCA (Dimensionality Reduction)
Reduce features while preserving information.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_reduced = pca.fit_transform(X)

print(f"Explained variance: {pca.explained_variance_ratio_}")
```

---

## Suggested Practice Order

1. **Week 1**: Cross-validation + more metrics
2. **Week 2**: Try LogisticRegression and SVM on your spam example
3. **Week 3**: Hyperparameter tuning with GridSearchCV
4. **Week 4**: Build a complete pipeline with persistence
5. **Week 5**: Explore clustering on unlabeled data

## Resources

- [Scikit-learn Official Documentation](https://scikit-learn.org/stable/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)

## What's in This Project

| File | Concepts demonstrated |
|------|----------------------|
| [model_trainer.py](../model_trainer.py) | RandomForest, IsolationForest, train/test split, scaling, evaluation |
| [naive_bayes_spam_example.py](../naive_bayes_spam_example.py) | Naive Bayes, Pipeline, CountVectorizer, text classification |
| [data_generator.py](../data_generator.py) | Feature engineering, synthetic data generation |
| [model_evaluator.py](../model_evaluator.py) | Visualization, confusion matrices, learning curves |
