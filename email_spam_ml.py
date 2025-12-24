#!/usr/bin/env python3
"""
Complete Scikit-Learn Tutorial: Email Spam/Ham Classification
Learn machine learning by building real spam detection models!
"""

import argparse

import numpy as np

from model_evaluator import ModelEvaluator
from model_trainer import EmailSpamMLTutorial


def main() -> None:
    """
    Entry point for the scikit-learn email spam classification tutorial.

    Parses command-line arguments and dispatches to appropriate execution mode
    (basic, advanced, or interactive). Provides flexible ways to explore the
    ML pipeline based on user preference.
    """
    parser = argparse.ArgumentParser(
        description="Scikit-Learn Email Spam Classification Tutorial"
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "advanced", "interactive"],
        default="basic",
        help="Tutorial mode",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Samples per category (default: 100)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualizations (requires matplotlib)",
    )

    args = parser.parse_args()

    print("SCIKIT-LEARN EMAIL SPAM CLASSIFICATION TUTORIAL")
    print("Learn AI by Building Real Spam Detection Models!")
    print("=" * 60)

    if args.mode == "basic":
        # Run basic tutorial
        tutorial = EmailSpamMLTutorial()
        tutorial.run_complete_tutorial()

    elif args.mode == "advanced":
        # Run with advanced analysis
        tutorial = EmailSpamMLTutorial()
        tutorial.run_complete_tutorial()

        if args.visualize:
            evaluator = ModelEvaluator(tutorial)
            evaluator.create_comprehensive_report()

    elif args.mode == "interactive":
        # Interactive mode
        run_interactive_tutorial()


def run_interactive_tutorial() -> None:
    """
    Execute the ML pipeline in interactive step-by-step mode.

    Prompts user for input at each stage of the ML pipeline, allowing
    learners to understand each step individually before proceeding.
    Also enables users to test custom email text against the trained models.
    """
    print("\nINTERACTIVE MODE")
    print("Let's build models step by step!")

    tutorial = EmailSpamMLTutorial()

    input("Press Enter to start with data generation...")
    tutorial.step_1_generate_and_explore_data()

    input("\nPress Enter to build spam classifier...")
    tutorial.step_2_build_spam_classifier()

    input("\nPress Enter to build anomaly detector...")
    tutorial.step_3_build_anomaly_detector()

    input("\nPress Enter to build spam score predictor...")
    tutorial.step_4_build_spam_score_predictor()

    input("\nPress Enter to test on new data...")
    tutorial.step_5_test_on_new_data()

    print("\nInteractive tutorial complete!")

    # Let user test their own data
    while True:
        print("\nTest your own email text!")
        print("Enter email text (or 'quit' to exit):")
        user_input = input("Email: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            break

        test_custom_email(tutorial, user_input)


def test_custom_email(tutorial, email_text: str) -> None:
    """
    Test trained models on custom email text.

    Allows users to input their own email text and see how the trained
    models classify it as spam/ham, predict spam score, and detect anomalies.

    Args:
        tutorial: EmailSpamMLTutorial instance with trained models.
        email_text: The email text to analyze.
    """
    print(f"\nAnalyzing email: {email_text[:50]}...")

    # Extract features
    features = tutorial.data_generator.extract_features(email_text)
    features_array = np.array([features])
    features_scaled = tutorial.feature_scaler.transform(features_array)

    # Run all models
    category_pred = tutorial.spam_classifier.predict(features_scaled)[0]
    category_name = tutorial.data_generator.get_category_name(category_pred)
    category_confidence = max(
        tutorial.spam_classifier.predict_proba(features_scaled)[0]
    )

    spam_score = tutorial.spam_score_predictor.predict(features_scaled)[0]

    anomaly_score = tutorial.anomaly_detector.decision_function(features_scaled)[0]
    is_anomaly = anomaly_score < -0.1

    print("AI Analysis:")
    print(f"   Classification: {category_name} (confidence: {category_confidence:.3f})")
    spam_label = (
        "HIGH SPAM"
        if spam_score > 0.7
        else "MODERATE" if spam_score > 0.4 else "LOW SPAM"
    )
    print(f"   Spam Score: {spam_score:.3f} ({spam_label})")
    print(f"   Anomaly: {'YES' if is_anomaly else 'NO'} (score: {anomaly_score:.3f})")
    print(f"   Features: {features}")


if __name__ == "__main__":
    main()
