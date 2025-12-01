#!/usr/bin/env python3
"""
🎓 Complete Scikit-Learn Tutorial: Network Security AI
Learn machine learning by building real security models!
"""

import argparse

import numpy as np

from model_evaluator import ModelEvaluator
from model_trainer import NetworkSecurityMLTutorial


def main() -> None:
    """
    Entry point for the scikit-learn network security tutorial application.

    Parses command-line arguments and dispatches to appropriate execution mode
    (basic, advanced, or interactive). Provides flexible ways to explore the
    ML pipeline based on user preference.
    """
    parser = argparse.ArgumentParser(
        description="Scikit-Learn Network Security Tutorial"
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
        help="Samples per device type (default: 100)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualizations (requires matplotlib)",
    )

    args = parser.parse_args()

    print("🎓 SCIKIT-LEARN NETWORK SECURITY TUTORIAL")
    print("🤖 Learn AI by Building Real Security Models!")
    print("=" * 60)

    if args.mode == "basic":
        # Run basic tutorial
        tutorial = NetworkSecurityMLTutorial()
        tutorial.run_complete_tutorial()

    elif args.mode == "advanced":
        # Run with advanced analysis
        tutorial = NetworkSecurityMLTutorial()
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
    Also enables users to test custom network port configurations against
    the trained models.
    """
    print("\n🎮 INTERACTIVE MODE")
    print("Let's build models step by step!")

    tutorial = NetworkSecurityMLTutorial()

    input("Press Enter to start with data generation...")
    tutorial.step_1_generate_and_explore_data()

    input("\nPress Enter to build device classifier...")
    tutorial.step_2_build_device_classifier()

    input("\nPress Enter to build anomaly detector...")
    tutorial.step_3_build_anomaly_detector()

    input("\nPress Enter to build risk predictor...")
    tutorial.step_4_build_risk_predictor()

    input("\nPress Enter to test on new data...")
    tutorial.step_5_test_on_new_data()

    print("\n🏆 Interactive tutorial complete!")

    # Let user test their own data
    while True:
        print("\n🧪 Test your own network data!")
        print("Enter port numbers separated by commas (or 'quit' to exit):")
        user_input = input("Ports: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            break

        try:
            ports = [int(p.strip()) for p in user_input.split(",")]
            test_custom_device(tutorial, ports)
        except ValueError:
            print("❌ Invalid input. Please enter numbers separated by commas.")


def test_custom_device(tutorial, ports: list) -> None:
    """
    Test trained models on custom network port configuration.

    Allows users to input their own port lists and see how the trained
    models classify the device, predict risk, and detect anomalies.

    Args:
        tutorial: NetworkSecurityMLTutorial instance with trained models.
        ports: List of port numbers to analyze.
    """
    print(f"\n🔍 Analyzing device with ports: {ports}")

    # Extract features
    features = tutorial.data_generator.extract_features(ports)
    features_array = np.array([features])
    features_scaled = tutorial.feature_scaler.transform(features_array)

    # Run all models
    device_pred = tutorial.device_classifier.predict(features_scaled)[0]
    device_name = tutorial.data_generator.get_device_name(device_pred)
    device_confidence = max(
        tutorial.device_classifier.predict_proba(features_scaled)[0]
    )

    risk_score = tutorial.risk_predictor.predict(features_scaled)[0]

    anomaly_score = tutorial.anomaly_detector.decision_function(features_scaled)[0]
    is_anomaly = anomaly_score < -0.1

    print("🤖 AI Analysis:")
    print(f"   Device Type: {device_name} (confidence: {device_confidence:.3f})")
    risk_label = "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.4 else "LOW"
    print(f"   Risk Level: {risk_score:.3f} ({risk_label})")
    print(f"   Anomaly: {'YES' if is_anomaly else 'NO'} (score: {anomaly_score:.3f})")


if __name__ == "__main__":
    main()
