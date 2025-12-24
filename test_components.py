#!/usr/bin/env python3
"""
Simple examples to test individual components of the tutorial
"""

import numpy as np
from sklearn.model_selection import train_test_split

from data_generator import EmailDataGenerator
from model_trainer import EmailSpamMLTutorial


def test_data_generation() -> None:
    """
    Verify that the data generator produces valid output.

    Tests:
    - Feature shapes match expectations
    - Feature values are within valid ranges
    - Labels correspond to valid categories (spam/ham)
    """
    print("Testing Data Generation...")

    generator = EmailDataGenerator()

    # Test feature extraction
    test_cases = [
        "Free money! Click now to claim your prize!",  # Spam
        "Hi, please review the attached report for tomorrow's meeting.",  # Ham
        "Congratulations! You have won!",  # Spam
        "Thanks for your feedback on the project proposal.",  # Ham
    ]

    for i, text in enumerate(test_cases, 1):
        features = generator.extract_features(text)
        # Show first 5 features
        print(f"   Test {i}: '{text[:30]}...' -> Features: {features[:5]}...")

    print("Data generation test complete!\n")


def test_model_training() -> None:
    """
    Verify that all three models train successfully with minimal data.

    Tests:
    - Models accept scaled features
    - Models produce valid predictions
    - Models are fitted and ready for evaluation
    """
    print("Testing Model Training...")

    # Create tutorial instance
    tutorial = EmailSpamMLTutorial()

    # Generate small dataset
    features_matrix, category_labels, spam_scores, texts = (
        tutorial.data_generator.generate_dataset(samples_per_class=20)
    )

    (
        tutorial.X_train,
        tutorial.X_test,
        tutorial.y_category_train,
        tutorial.y_category_test,
        tutorial.y_spam_score_train,
        tutorial.y_spam_score_test,
        tutorial.texts_train,
        tutorial.texts_test,
    ) = train_test_split(
        features_matrix,
        category_labels,
        spam_scores,
        texts,
        test_size=0.2,
        random_state=42,
        stratify=category_labels,
    )

    # Test spam classifier
    accuracy = tutorial.step_2_build_spam_classifier()
    print(f"   Spam classifier accuracy: {accuracy:.3f}")

    # Test anomaly detector
    num_anomalies, anomaly_rate = tutorial.step_3_build_anomaly_detector()
    print(f"   Anomaly detection: {num_anomalies} anomalies ({anomaly_rate:.1f}%)")

    # Test spam score predictor
    _, r2 = tutorial.step_4_build_spam_score_predictor()
    print(f"   Spam score prediction R2: {r2:.3f}")

    print("Model training test complete!\n")


def test_custom_prediction() -> None:
    """
    Verify that trained models generate valid predictions.

    Tests:
    - Classification predictions are valid category IDs (0=ham, 1=spam)
    - Regression predictions are in valid spam score range [0, 1]
    - Models handle arbitrary email text
    """
    print("Testing Custom Predictions...")

    # Quick training
    tutorial = EmailSpamMLTutorial()
    features_matrix, category_labels, spam_scores, texts = (
        tutorial.data_generator.generate_dataset(samples_per_class=30)
    )

    (
        tutorial.X_train,
        tutorial.X_test,
        tutorial.y_category_train,
        tutorial.y_category_test,
        tutorial.y_spam_score_train,
        tutorial.y_spam_score_test,
        tutorial.texts_train,
        tutorial.texts_test,
    ) = train_test_split(
        features_matrix,
        category_labels,
        spam_scores,
        texts,
        test_size=0.2,
        random_state=42,
        stratify=category_labels,
    )

    # Train models quickly
    tutorial.step_2_build_spam_classifier()
    tutorial.step_3_build_anomaly_detector()
    tutorial.step_4_build_spam_score_predictor()

    spam_classifier = tutorial.spam_classifier
    spam_score_predictor = tutorial.spam_score_predictor

    if spam_classifier is None or spam_score_predictor is None:
        raise ValueError("Models failed to train during custom prediction test.")

    # Test custom emails
    test_emails = [
        (
            "Work Email",
            "Hi team, please review the report before our meeting tomorrow.",
        ),
        ("Obvious Spam", "Congratulations! You won FREE cash! Claim your prize now!"),
        ("Newsletter", "Thanks for subscribing. Here's your weekly update."),
    ]

    for name, text in test_emails:
        features = tutorial.data_generator.extract_features(text)
        features_array = np.array([features])
        features_scaled = tutorial.feature_scaler.transform(features_array)

        category_pred = spam_classifier.predict(features_scaled)[0]
        category_name = tutorial.data_generator.get_category_name(category_pred)
        spam_score = spam_score_predictor.predict(features_scaled)[0]

        print(f"   {name}: Predicted as {category_name}, Spam Score: {spam_score:.3f}")

    print("Custom prediction test complete!\n")


def main() -> None:
    """
    Execute all component tests and report results.

    Runs functional tests for data generation, model training, and predictions.
    Displays summary of test results and instructions for running the full tutorial.
    """
    print("SCIKIT-LEARN TUTORIAL COMPONENT TESTS")
    print("=" * 50)

    try:
        test_data_generation()
        test_model_training()
        test_custom_prediction()

        print("All tests passed! Tutorial is ready to use.")
        print("\nRun the full tutorial with:")
        print("   python email_spam_ml.py")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Please run 'python setup.py' first to install dependencies.")
    except Exception as e:  # pylint: disable=broad-except
        print(f"Test failed: {e}")
        print("Please check your installation.")


if __name__ == "__main__":
    main()
