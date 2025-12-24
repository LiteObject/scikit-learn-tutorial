"""
Machine learning model training and orchestration.

This module manages the complete ML pipeline for email spam/ham classification.
The EmailSpamMLTutorial class trains three different scikit-learn models:
- RandomForestClassifier for spam/ham classification
- IsolationForest for anomaly detection
- RandomForestRegressor for spam score prediction

All features are standardized using StandardScaler before training to ensure
consistent model performance.
"""

# pylint: disable=invalid-name
# Pylint warns about variables such as X and y that are standard in ML literature.

from typing import List, Optional, Tuple

import numpy as np
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_generator import EmailDataGenerator


class EmailSpamMLTutorial:
    """
    Complete ML pipeline for email spam/ham classification.

    This class orchestrates the full machine learning workflow:
    1. Generates synthetic email data
    2. Trains three different model types (classification, anomaly detection, regression)
    3. Evaluates performance using standard ML metrics

    Uses StandardScaler for feature normalization across all models.
    """

    def __init__(self) -> None:
        """
        Initialize the ML tutorial with data generator and model instances.

        Sets up:
        - Data generator for synthetic email data
        - Feature scaler (StandardScaler) for consistent preprocessing
        - Three model types: RandomForestClassifier, IsolationForest, RandomForestRegressor
        - Train/test split arrays for evaluation
        """
        self.data_generator = EmailDataGenerator()

        # Initialize models (we'll train these step by step)
        self.spam_classifier: Optional[RandomForestClassifier] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.spam_score_predictor: Optional[RandomForestRegressor] = None

        # Without scaling, features with larger ranges (like word count)
        # would dominate over smaller ones (like spam ratio). Many algorithms
        # perform better when features are on the same scale.
        self.feature_scaler = StandardScaler()

        # Store data for analysis
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_category_train: Optional[np.ndarray] = None
        self.y_category_test: Optional[np.ndarray] = None
        self.y_spam_score_train: Optional[np.ndarray] = None
        self.y_spam_score_test: Optional[np.ndarray] = None
        self.texts_train: Optional[List[str]] = None
        self.texts_test: Optional[List[str]] = None

    def step_1_generate_and_explore_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step 1: Generate data and explore it.

        Returns:
            Tuple containing:
            - X: Feature matrix of shape (n_samples, 10)
            - y_category: Category labels (0=ham, 1=spam)
            - y_spam_score: Spam score labels (0.0 to 1.0)
        """
        print("STEP 1: DATA GENERATION AND EXPLORATION")
        print("=" * 50)

        # Generate dataset
        X, y_category, y_spam_score, texts = self.data_generator.generate_dataset(
            samples_per_class=100
        )

        print("\nDataset Overview:")
        print(f"   Total samples: {len(X)}")
        print(f"   Features per sample: {X.shape[1]}")
        print(f"   Categories: {len(np.unique(y_category))} (Ham, Spam)")

        # Show feature names for understanding
        feature_names = [
            "Word Count",
            "Spam Words",
            "Ham Words",
            "Spam Ratio",
            "Has Urgency",
            "Has Money Words",
            "Has Exclamation",
            "Exclamation Count",
            "Avg Word Length",
            "Uppercase Ratio",
        ]

        print("\nFeatures we're using:")
        for i, name in enumerate(feature_names):
            print(f"   [{i}] {name}")

        # Show some actual data samples
        print("\nSample Data (first 3 emails):")
        for i in range(3):
            category_name = self.data_generator.get_category_name(y_category[i])
            print(f"   Sample {i+1}: {category_name}")
            print(f"      Text: {texts[i][:50]}...")
            print(f"      Features: {X[i]}")
            print(f"      Spam Score: {y_spam_score[i]:.3f}")

        # Split data for training and testing
        (
            X_train,
            X_test,
            y_category_train,
            y_category_test,
            y_spam_score_train,
            y_spam_score_test,
            texts_train,
            texts_test,
        ) = train_test_split(
            X,
            y_category,
            y_spam_score,
            texts,
            test_size=0.2,  # 20% for testing
            random_state=42,  # Reproducible results
            stratify=y_category,  # Ensure balanced split
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_category_train = y_category_train
        self.y_category_test = y_category_test
        self.y_spam_score_train = y_spam_score_train
        self.y_spam_score_test = y_spam_score_test
        self.texts_train = texts_train
        self.texts_test = texts_test

        print("\nData Split:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")

        return X, y_category, y_spam_score

    def step_2_build_spam_classifier(self) -> float:
        """Step 2: Build and train spam classification model.

        Returns:
            float: Classification accuracy score (0.0 to 1.0).
        """
        print("\nSTEP 2: SPAM CLASSIFICATION MODEL")
        print("=" * 50)

        print("Building Random Forest Classifier...")
        print("   Random Forest = Many decision trees voting together")

        # Create the model with specific parameters
        spam_classifier = RandomForestClassifier(
            n_estimators=100,  # Number of trees in the forest
            random_state=42,  # For reproducible results
            max_depth=10,  # Maximum depth of each tree
            min_samples_split=5,  # Minimum samples to split a node
            min_samples_leaf=2,  # Minimum samples in a leaf
        )
        self.spam_classifier = spam_classifier

        print(f"   Forest size: {spam_classifier.n_estimators} trees")  # type: ignore
        print(f"   Max depth: {spam_classifier.max_depth}")  # type: ignore

        # Scale features (normalize them)
        print("Scaling features...")
        X_train = self.X_train
        X_test = self.X_test
        y_category_train = self.y_category_train

        if X_train is None or X_test is None or y_category_train is None:
            raise ValueError("Data not generated. Run step 1 first.")

        # You fit_transform on training data but only transform on
        # test data - this prevents data leakage.
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)

        # Train the model
        print("Training the model...")
        spam_classifier.fit(X_train_scaled, y_category_train)
        print("Model training complete!")

        # Test the model
        print("\nTesting the model...")
        y_category_test = self.y_category_test
        if y_category_test is None:
            raise ValueError("Test data not available.")

        predictions = spam_classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_category_test, predictions)

        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

        # Show detailed results
        category_names = [self.data_generator.get_category_name(i) for i in range(2)]
        print("\nDetailed Classification Report:")
        print(
            classification_report(
                y_category_test, predictions, target_names=category_names
            )
        )

        # Show feature importance
        feature_names = [
            "Word Count",
            "Spam Words",
            "Ham Words",
            "Spam Ratio",
            "Has Urgency",
            "Has Money Words",
            "Has Exclamation",
            "Exclamation Count",
            "Avg Word Length",
            "Uppercase Ratio",
        ]

        importances = spam_classifier.feature_importances_
        print("\nFeature Importance (what the model cares about most):")
        for name, importance in zip(feature_names, importances):
            print(f"   {name}: {importance:.3f}")

        return float(accuracy)

    def step_3_build_anomaly_detector(self) -> Tuple[int, float]:
        """Step 3: Build anomaly detection model.

        Returns:
            Tuple containing:
            - num_anomalies: Count of detected anomalies in test set
            - anomaly_percentage: Percentage of test samples flagged as anomalies
        """
        print("\nSTEP 3: ANOMALY DETECTION MODEL")
        print("=" * 50)

        print("Building Isolation Forest for anomaly detection...")
        print("   Isolation Forest = Finds data points that are 'easy to isolate'")
        print("   Anomalies = Emails with unusual patterns")

        # Create the anomaly detector
        contamination_rate = 0.1  # Expect 10% of data to be anomalous
        num_isolation_trees = 100
        anomaly_detector = IsolationForest(
            contamination=contamination_rate,
            random_state=42,
            n_estimators=num_isolation_trees,
        )
        self.anomaly_detector = anomaly_detector

        print(f"   Number of isolation trees: {num_isolation_trees}")
        print(f"   Expected contamination: {contamination_rate * 100}%")

        # Train on scaled training data
        X_train = self.X_train
        if X_train is None:
            raise ValueError("Training data not available.")

        X_train_scaled = self.feature_scaler.transform(X_train)

        print("Training anomaly detector...")
        anomaly_detector.fit(X_train_scaled)
        print("Anomaly detector training complete!")

        # Test anomaly detection
        X_test = self.X_test
        y_category_test = self.y_category_test
        if X_test is None or y_category_test is None:
            raise ValueError("Test data not available.")

        X_test_scaled = self.feature_scaler.transform(X_test)
        anomaly_predictions = anomaly_detector.predict(X_test_scaled)
        anomaly_scores = anomaly_detector.decision_function(X_test_scaled)

        # Count anomalies
        # -1 = anomaly, 1 = normal
        num_anomalies = np.sum(anomaly_predictions == -1)
        anomaly_percentage = (num_anomalies / len(anomaly_predictions)) * 100

        print("\nAnomaly Detection Results:")
        print(f"   Total test samples: {len(anomaly_predictions)}")
        print(f"   Detected anomalies: {num_anomalies}")
        print(f"   Anomaly rate: {anomaly_percentage:.1f}%")

        # Show some examples
        print("\nExample Anomaly Scores (lower = more anomalous):")
        for i in range(min(5, len(anomaly_scores))):
            status = "ANOMALY" if anomaly_predictions[i] == -1 else "NORMAL"
            category_name = self.data_generator.get_category_name(y_category_test[i])
            print(f"   {category_name}: {anomaly_scores[i]:.3f} ({status})")

        return num_anomalies, anomaly_percentage

    def step_4_build_spam_score_predictor(self) -> Tuple[float, float]:
        """Step 4: Build spam score prediction model.

        Returns:
            Tuple containing:
            - mae: Mean Absolute Error of predictions
            - r2: R-squared score (coefficient of determination)
        """
        print("\nSTEP 4: SPAM SCORE PREDICTION MODEL")
        print("=" * 50)

        print("Building Random Forest Regressor for spam score prediction...")
        print("   Regression = Predicting continuous numbers (0.0 to 1.0 spam score)")
        print("   Classification = Predicting categories (spam or ham)")

        # Create the spam score predictor
        spam_score_predictor = RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10, min_samples_split=5
        )
        self.spam_score_predictor = spam_score_predictor

        print(f"   Forest size: {spam_score_predictor.n_estimators} trees")  # type: ignore

        # Train the model
        X_train = self.X_train
        y_spam_score_train = self.y_spam_score_train
        if X_train is None or y_spam_score_train is None:
            raise ValueError("Training data not available.")

        X_train_scaled = self.feature_scaler.transform(X_train)

        print("Training spam score predictor...")
        spam_score_predictor.fit(X_train_scaled, y_spam_score_train)
        print("Spam score predictor training complete!")

        # Test the model
        X_test = self.X_test
        y_spam_score_test = self.y_spam_score_test
        y_category_test = self.y_category_test

        if X_test is None or y_spam_score_test is None or y_category_test is None:
            raise ValueError("Test data not available.")

        X_test_scaled = self.feature_scaler.transform(X_test)
        spam_score_predictions = spam_score_predictor.predict(X_test_scaled)

        mse = mean_squared_error(y_spam_score_test, spam_score_predictions)
        mae = mean_absolute_error(y_spam_score_test, spam_score_predictions)
        r2 = r2_score(y_spam_score_test, spam_score_predictions)

        print("\nSpam Score Prediction Performance:")
        print(f"   Mean Absolute Error: {mae:.3f}")
        print(f"   Mean Squared Error: {mse:.3f}")
        print(f"   R2 Score: {r2:.3f} (closer to 1.0 = better)")

        # Show some examples
        print("\nExample Spam Score Predictions:")
        for i in range(min(5, len(spam_score_predictions))):
            actual = y_spam_score_test[i]
            predicted = spam_score_predictions[i]
            category_name = self.data_generator.get_category_name(y_category_test[i])
            print(f"   {category_name}: Actual={actual:.3f}, Predicted={predicted:.3f}")

        return mae, r2

    def step_5_test_on_new_data(self) -> None:
        """Step 5: Test all models on completely new data.

        Runs inference on predefined test cases to demonstrate
        how the trained models handle real-world email examples.
        """
        print("\nSTEP 5: TESTING ON NEW DATA")
        print("=" * 50)

        # Create some test cases
        test_cases = [
            {
                "name": "Obvious Spam Email",
                "text": "Congratulations! You have won a free prize! Click now to claim your cash reward! Limited time offer!",
                "expected": "Should be classified as Spam, high spam score",
            },
            {
                "name": "Work Email",
                "text": "Hi team, please review the attached report before tomorrow's meeting. Thanks for your feedback.",
                "expected": "Should be classified as Ham, low spam score",
            },
            {
                "name": "Subtle Spam",
                "text": "Hello, we have an exclusive deal just for you. Limited discount available now.",
                "expected": "Should be Spam with moderate confidence",
            },
            {
                "name": "Unusual Email",
                "text": "xyz123 qwerty asdf random gibberish text with no meaning whatsoever",
                "expected": "Should be anomalous, unpredictable classification",
            },
        ]

        print("Testing models on new, realistic scenarios...")

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i}: {test_case['name']} ---")
            print(f"Email text: {test_case['text'][:60]}...")
            print(f"Expected: {test_case['expected']}")

            # Extract features
            features = self.data_generator.extract_features(test_case["text"])
            features_array = np.array([features])  # Shape: (1, 10)
            features_scaled = self.feature_scaler.transform(features_array)

            # Run all three models
            # 1. Spam Classification
            spam_classifier = self.spam_classifier
            spam_score_predictor = self.spam_score_predictor
            anomaly_detector = self.anomaly_detector

            if (
                spam_classifier is None
                or spam_score_predictor is None
                or anomaly_detector is None
            ):
                raise ValueError("Models not trained. Run steps 2-4 first.")

            category_pred = spam_classifier.predict(features_scaled)[0]
            category_proba = spam_classifier.predict_proba(features_scaled)[0]
            category_confidence = max(category_proba)
            category_name = self.data_generator.get_category_name(category_pred)

            # 2. Spam Score Prediction
            spam_score = spam_score_predictor.predict(features_scaled)[0]

            # 3. Anomaly Detection
            anomaly_score = anomaly_detector.decision_function(features_scaled)[0]
            is_anomaly = anomaly_score < -0.1  # Threshold for anomaly

            print("AI Analysis Results:")
            print(
                f"   Classification: {category_name} (confidence: {category_confidence:.3f})"
            )
            spam_label = (
                "HIGH SPAM"
                if spam_score > 0.7
                else "MODERATE" if spam_score > 0.4 else "LOW SPAM"
            )
            anomaly_label = "ANOMALOUS" if is_anomaly else "NORMAL"
            print(f"   Spam Score: {spam_score:.3f} ({spam_label})")
            print(f"   Anomaly Status: {anomaly_label} (score: {anomaly_score:.3f})")

    def run_complete_tutorial(self) -> None:
        """Run the complete tutorial.

        Executes all five steps of the ML pipeline in sequence:
        data generation, classification, anomaly detection,
        regression, and testing on new data.
        """
        # Step 1: Data
        self.step_1_generate_and_explore_data()

        # Step 2: Classification
        accuracy = self.step_2_build_spam_classifier()

        # Step 3: Anomaly Detection
        _, anomaly_rate = self.step_3_build_anomaly_detector()

        # Step 4: Regression
        _, r2 = self.step_4_build_spam_score_predictor()

        # Step 5: Real-world testing
        self.step_5_test_on_new_data()

        # Summary
        print("\nTUTORIAL COMPLETE!")
        print("=" * 50)
        print("What you've learned:")
        print(f"   Random Forest Classification (accuracy: {accuracy:.3f})")
        print(f"   Isolation Forest Anomaly Detection ({anomaly_rate:.1f}% anomalies)")
        print(f"   Random Forest Regression (R2: {r2:.3f})")
        print("   Feature Engineering and Data Preprocessing")
        print("   Model Training, Testing, and Evaluation")

        print("\nNext steps:")
        print("   Try different algorithms (SVM, Neural Networks, etc.)")
        print("   Experiment with feature engineering")
        print("   Add data visualization with matplotlib")
        print("   Apply to your own datasets!")


# Run the tutorial
if __name__ == "__main__":
    tutorial = EmailSpamMLTutorial()
    tutorial.run_complete_tutorial()
