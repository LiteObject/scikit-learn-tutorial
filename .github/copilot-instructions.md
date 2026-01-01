# GitHub Copilot Instructions for Scikit-Learn Tutorial

This repository is an educational project designed to teach **basic to advanced scikit-learn concepts**. It uses a cohesive email spam/ham classification scenario to demonstrate practical applications of machine learning, moving beyond simple classification to include regression, anomaly detection, and complex feature engineering.

## Project Goals and Educational Focus

- **Primary Goal**: Teach ML concepts (Classification, Regression, Anomaly Detection, Evaluation).
- **Secondary Goal**: Demonstrate a realistic ML pipeline structure.
- **Guideline**: When generating code, prioritize clarity and educational value. Add comments explaining *why* a specific scikit-learn class or method is used.

## Architecture and Core Components

The project is structured as a modular ML pipeline.

- **Entry Point**: `email_spam_ml.py`
  - Orchestrates the application flow.
  - Handles CLI arguments (`--mode`, `--samples`, `--visualize`).

- **Data Generation**: `data_generator.py` (`EmailDataGenerator`)
  - Simulates email content and performs **Feature Engineering**.
  - **Key Logic**: Converts raw email text into a 10-dimensional feature vector (word count, spam words, ham words, spam ratio, urgency indicators, etc.).
  - **Concepts**: Feature extraction, synthetic data generation, handling noise.

- **Model Training**: `model_trainer.py` (`EmailSpamMLTutorial`)
  - Manages the lifecycle of multiple model types:
    1. **Classification**: `RandomForestClassifier` (identifying spam vs ham).
    2. **Anomaly Detection**: `IsolationForest` (finding unusual email patterns).
    3. **Regression**: `RandomForestRegressor` (predicting continuous spam scores).
  - **Critical Pattern**: Always apply `self.feature_scaler` (StandardScaler) to features before passing them to models.

- **Evaluation**: `model_evaluator.py` (`ModelEvaluator`)
  - Handles visualization using `matplotlib` and `seaborn`.
  - **Concepts**: Confusion matrices, feature importance, learning curves, precision/recall.

## Workflows and Development

### Running the Application
- **Basic Mode**: `python email_spam_ml.py --mode basic`
- **Interactive Mode**: `python email_spam_ml.py --mode interactive` (Step-by-step execution)
- **Visualization**: Add `--visualize` to generate plots.

### Testing
- **Component Tests**: `python test_components.py`
  - Runs functional tests for data generation, training, and prediction.

## Coding Conventions

- **Code Quality**: Ensure generated code stays clean for **both Pylance and Pylint** (no errors/warnings). Follow PEP 8 standards.
- **Documentation Style**: Write in a natural, professional, human-like tone. Avoid robotic phrasing or repetitive AI-style transitions. **Do not use emojis** in generated documentation or comments.
- **Educational Comments**: Explain ML choices (e.g., "Using RandomForest because it handles non-linear relationships well").
- **Type Hinting**: Use comprehensive Python type hints (e.g., `List[int]`, `Tuple[float, int]`) to support static analysis.
- **Docstrings**: Maintain descriptive docstrings for all functions and classes.
- **Data Handling**:
  - Features should be `numpy` arrays or lists of floats.
  - **Scaling**: Never feed raw features to the models; use the shared `StandardScaler` instance.

## Common Patterns and Gotchas

- **Feature Extraction**:
  - Located in `data_generator.py` -> `extract_features`.
  - If you modify the feature set, you must update the `feature_names` list in `model_trainer.py`.

- **Model Persistence**:
  - Currently, models are trained in-memory.
  - If adding persistence, use `joblib`.

- **Reproducibility**:
  - `random_state=42` is used throughout to ensure consistent results for learners.
