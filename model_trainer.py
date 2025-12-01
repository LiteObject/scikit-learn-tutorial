"""Machine learning model training and orchestration.

This module manages the complete ML pipeline for network security analysis.
The NetworkSecurityMLTutorial class trains three different scikit-learn models:
- RandomForestClassifier for device type identification
- IsolationForest for anomaly detection
- RandomForestRegressor for risk score prediction

All features are standardized using StandardScaler before training to ensure
consistent model performance.
"""

# pylint: disable=invalid-name
# Pylint warns about variables such as X and y that are standard in ML literature.

from typing import Optional

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

from data_generator import NetworkDataGenerator


class NetworkSecurityMLTutorial:
    """
    Complete ML pipeline for network security classification and risk assessment.

    This class orchestrates the full machine learning workflow:
    1. Generates synthetic network data
    2. Trains three different model types (classification, anomaly detection, regression)
    3. Evaluates performance using standard ML metrics

    Uses StandardScaler for feature normalization across all models.
    """

    def __init__(self) -> None:
        """
        Initialize the ML tutorial with data generator and model instances.

        Sets up:
        - Data generator for synthetic network traffic
        - Feature scaler (StandardScaler) for consistent preprocessing
        - Three model types: RandomForestClassifier, IsolationForest, RandomForestRegressor
        - Train/test split arrays for evaluation
        """
        self.data_generator = NetworkDataGenerator()

        # Initialize models (we'll train these step by step)
        self.device_classifier: Optional[RandomForestClassifier] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.risk_predictor: Optional[RandomForestRegressor] = None
        self.feature_scaler = StandardScaler()

        # Store data for analysis
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_device_train: Optional[np.ndarray] = None
        self.y_device_test: Optional[np.ndarray] = None
        self.y_risk_train: Optional[np.ndarray] = None
        self.y_risk_test: Optional[np.ndarray] = None

    def step_1_generate_and_explore_data(self):
        """Step 1: Generate data and explore it"""
        print("🎯 STEP 1: DATA GENERATION AND EXPLORATION")
        print("=" * 50)

        # Generate dataset
        X, y_device, y_risk = self.data_generator.generate_dataset(
            samples_per_class=100
        )

        print("\n📊 Dataset Overview:")
        print(f"   Total samples: {len(X)}")
        print(f"   Features per sample: {X.shape[1]}")
        print(f"   Device types: {len(np.unique(y_device))}")

        # Show feature names for understanding
        feature_names = [
            "Total Ports",
            "Has SSH",
            "Has HTTP",
            "Has HTTPS",
            "Has Telnet",
            "Has RDP",
            "Has SMB",
            "Has FTP",
            "Port Spread",
            "High Ports",
        ]

        print("\n🔧 Features we're using:")
        for i, name in enumerate(feature_names):
            print(f"   [{i}] {name}")

        # Show some actual data samples
        print("\n📝 Sample Data (first 3 devices):")
        for i in range(3):
            device_name = self.data_generator.get_device_name(y_device[i])
            print(f"   Sample {i+1}: {device_name}")
            print(f"      Features: {X[i]}")
            print(f"      Risk Score: {y_risk[i]:.3f}")

        # Split data for training and testing
        X_train, X_test, y_device_train, y_device_test, y_risk_train, y_risk_test = (
            train_test_split(
                X,
                y_device,
                y_risk,
                test_size=0.2,  # 20% for testing
                random_state=42,  # Reproducible results
                stratify=y_device,  # Ensure balanced split across device types
            )
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_device_train = y_device_train
        self.y_device_test = y_device_test
        self.y_risk_train = y_risk_train
        self.y_risk_test = y_risk_test

        print("\n✂️ Data Split:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")

        return X, y_device, y_risk

    def step_2_build_device_classifier(self):
        """Step 2: Build and train device classification model"""
        print("\n🎯 STEP 2: DEVICE CLASSIFICATION MODEL")
        print("=" * 50)

        print("🌳 Building Random Forest Classifier...")
        print("   Random Forest = Many decision trees voting together")

        # Create the model with specific parameters
        device_classifier = RandomForestClassifier(
            n_estimators=100,  # Number of trees in the forest
            random_state=42,  # For reproducible results
            max_depth=10,  # Maximum depth of each tree
            min_samples_split=5,  # Minimum samples to split a node
            min_samples_leaf=2,  # Minimum samples in a leaf
        )
        self.device_classifier = device_classifier

        print(f"   🌲 Forest size: {device_classifier.n_estimators} trees")  # type: ignore
        print(f"   📏 Max depth: {device_classifier.max_depth}")  # type: ignore

        # Scale features (normalize them)
        print("📊 Scaling features...")
        X_train = self.X_train
        X_test = self.X_test
        y_device_train = self.y_device_train

        if X_train is None or X_test is None or y_device_train is None:
            raise ValueError("Data not generated. Run step 1 first.")

        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)

        # Train the model
        print("🎓 Training the model...")
        device_classifier.fit(X_train_scaled, y_device_train)
        print("✅ Model training complete!")

        # Test the model
        print("\n🧪 Testing the model...")
        y_device_test = self.y_device_test
        if y_device_test is None:
            raise ValueError("Test data not available.")

        predictions = device_classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_device_test, predictions)

        print(f"🎯 Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

        # Show detailed results
        device_names = [self.data_generator.get_device_name(i) for i in range(6)]
        print("\n📋 Detailed Classification Report:")
        print(
            classification_report(y_device_test, predictions, target_names=device_names)
        )

        # Show feature importance
        feature_names = [
            "Total Ports",
            "Has SSH",
            "Has HTTP",
            "Has HTTPS",
            "Has Telnet",
            "Has RDP",
            "Has SMB",
            "Has FTP",
            "Port Spread",
            "High Ports",
        ]

        importances = device_classifier.feature_importances_
        print("\n🔥 Feature Importance (what the model cares about most):")
        for name, importance in zip(feature_names, importances):
            print(f"   {name}: {importance:.3f}")

        return accuracy

    def step_3_build_anomaly_detector(self):
        """Step 3: Build anomaly detection model"""
        print("\n🎯 STEP 3: ANOMALY DETECTION MODEL")
        print("=" * 50)

        print("🕵️ Building Isolation Forest for anomaly detection...")
        print("   Isolation Forest = Finds data points that are 'easy to isolate'")
        print("   Anomalies = Things that don't fit normal patterns")

        # Create the anomaly detector
        contamination_rate = 0.1  # Expect 10% of data to be anomalous
        num_isolation_trees = 100
        anomaly_detector = IsolationForest(
            contamination=contamination_rate,
            random_state=42,
            n_estimators=num_isolation_trees,
        )
        self.anomaly_detector = anomaly_detector

        print(f"   🌲 Number of isolation trees: {num_isolation_trees}")
        print(f"   ⚠️ Expected contamination: {contamination_rate * 100}%")

        # Train on scaled training data
        X_train = self.X_train
        if X_train is None:
            raise ValueError("Training data not available.")

        X_train_scaled = self.feature_scaler.transform(X_train)

        print("🎓 Training anomaly detector...")
        anomaly_detector.fit(X_train_scaled)
        print("✅ Anomaly detector training complete!")

        # Test anomaly detection
        X_test = self.X_test
        y_device_test = self.y_device_test
        if X_test is None or y_device_test is None:
            raise ValueError("Test data not available.")

        X_test_scaled = self.feature_scaler.transform(X_test)
        anomaly_predictions = anomaly_detector.predict(X_test_scaled)
        anomaly_scores = anomaly_detector.decision_function(X_test_scaled)

        # Count anomalies
        # -1 = anomaly, 1 = normal
        num_anomalies = np.sum(anomaly_predictions == -1)
        anomaly_percentage = (num_anomalies / len(anomaly_predictions)) * 100

        print("\n🔍 Anomaly Detection Results:")
        print(f"   Total test samples: {len(anomaly_predictions)}")
        print(f"   Detected anomalies: {num_anomalies}")
        print(f"   Anomaly rate: {anomaly_percentage:.1f}%")

        # Show some examples
        print("\n🔬 Example Anomaly Scores (lower = more anomalous):")
        for i in range(min(5, len(anomaly_scores))):
            status = "ANOMALY" if anomaly_predictions[i] == -1 else "NORMAL"
            device_type = self.data_generator.get_device_name(y_device_test[i])
            print(f"   {device_type}: {anomaly_scores[i]:.3f} ({status})")

        return num_anomalies, anomaly_percentage

    def step_4_build_risk_predictor(self):
        """Step 4: Build risk prediction model"""
        print("\n🎯 STEP 4: RISK PREDICTION MODEL")
        print("=" * 50)

        print("⚡ Building Random Forest Regressor for risk prediction...")
        print("   Regression = Predicting continuous numbers (0.0 to 1.0 risk)")
        print("   Classification = Predicting categories (device types)")

        # Create the risk predictor
        risk_predictor = RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10, min_samples_split=5
        )
        self.risk_predictor = risk_predictor

        print(f"   🌲 Forest size: {risk_predictor.n_estimators} trees")  # type: ignore

        # Train the model
        X_train = self.X_train
        y_risk_train = self.y_risk_train
        if X_train is None or y_risk_train is None:
            raise ValueError("Training data not available.")

        X_train_scaled = self.feature_scaler.transform(X_train)

        print("🎓 Training risk predictor...")
        risk_predictor.fit(X_train_scaled, y_risk_train)
        print("✅ Risk predictor training complete!")

        # Test the model
        X_test = self.X_test
        y_risk_test = self.y_risk_test
        y_device_test = self.y_device_test

        if X_test is None or y_risk_test is None or y_device_test is None:
            raise ValueError("Test data not available.")

        X_test_scaled = self.feature_scaler.transform(X_test)
        risk_predictions = risk_predictor.predict(X_test_scaled)

        mse = mean_squared_error(y_risk_test, risk_predictions)
        mae = mean_absolute_error(y_risk_test, risk_predictions)
        r2 = r2_score(y_risk_test, risk_predictions)

        print("\n📊 Risk Prediction Performance:")
        print(f"   Mean Absolute Error: {mae:.3f}")
        print(f"   Mean Squared Error: {mse:.3f}")
        print(f"   R² Score: {r2:.3f} (closer to 1.0 = better)")

        # Show some examples
        print("\n🔬 Example Risk Predictions:")
        for i in range(min(5, len(risk_predictions))):
            actual = y_risk_test[i]
            predicted = risk_predictions[i]
            device_type = self.data_generator.get_device_name(y_device_test[i])
            print(f"   {device_type}: Actual={actual:.3f}, Predicted={predicted:.3f}")

        return mae, r2

    def step_5_test_on_new_data(self):
        """Step 5: Test all models on completely new data"""
        print("\n🎯 STEP 5: TESTING ON NEW DATA")
        print("=" * 50)

        # Create some test cases
        test_cases = [
            {
                "name": "Typical Web Server",
                "ports": [22, 80, 443],
                "expected": "Should be classified as Linux Server, low-medium risk",
            },
            {
                "name": "Suspicious Device",
                "ports": [23, 21, 3389, 445, 135],
                "expected": "Should be high risk, possibly anomalous",
            },
            {
                "name": "Simple IoT Device",
                "ports": [80],
                "expected": "Should be IoT device, low risk",
            },
            {
                "name": "Unusual Device",
                "ports": [1337, 31337, 8080, 9999],
                "expected": "Should be anomalous, unknown risk",
            },
        ]

        print("🧪 Testing models on new, realistic scenarios...")

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i}: {test_case['name']} ---")
            print(f"Open ports: {test_case['ports']}")
            print(f"Expected: {test_case['expected']}")

            # Extract features
            features = self.data_generator.extract_features(test_case["ports"])
            features_array = np.array([features])  # Shape: (1, 10)
            features_scaled = self.feature_scaler.transform(features_array)

            # Run all three models
            # 1. Device Classification
            device_classifier = self.device_classifier
            risk_predictor = self.risk_predictor
            anomaly_detector = self.anomaly_detector

            if (
                device_classifier is None
                or risk_predictor is None
                or anomaly_detector is None
            ):
                raise ValueError("Models not trained. Run steps 2-4 first.")

            device_pred = device_classifier.predict(features_scaled)[0]
            device_proba = device_classifier.predict_proba(features_scaled)[0]
            device_confidence = max(device_proba)
            device_name = self.data_generator.get_device_name(device_pred)

            # 2. Risk Prediction
            risk_score = risk_predictor.predict(features_scaled)[0]

            # 3. Anomaly Detection
            anomaly_score = anomaly_detector.decision_function(features_scaled)[0]
            is_anomaly = anomaly_score < -0.1  # Threshold for anomaly

            print("🤖 AI Analysis Results:")
            print(
                f"   Device Type: {device_name} (confidence: {device_confidence:.3f})"
            )
            risk_label = (
                "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.4 else "LOW"
            )
            anomaly_label = "ANOMALOUS" if is_anomaly else "NORMAL"
            print(f"   Risk Score: {risk_score:.3f} ({risk_label})")
            print(f"   Anomaly Status: {anomaly_label} (score: {anomaly_score:.3f})")

    def run_complete_tutorial(self):
        """Run the complete tutorial"""
        # Step 1: Data
        self.step_1_generate_and_explore_data()

        # Step 2: Classification
        accuracy = self.step_2_build_device_classifier()

        # Step 3: Anomaly Detection
        _, anomaly_rate = self.step_3_build_anomaly_detector()

        # Step 4: Regression
        _, r2 = self.step_4_build_risk_predictor()

        # Step 5: Real-world testing
        self.step_5_test_on_new_data()

        # Summary
        print("\n🏆 TUTORIAL COMPLETE!")
        print("=" * 50)
        print("🎯 What you've learned:")
        print(f"   ✅ Random Forest Classification (accuracy: {accuracy:.3f})")
        print(
            f"   ✅ Isolation Forest Anomaly Detection ({anomaly_rate:.1f}% anomalies)"
        )
        print(f"   ✅ Random Forest Regression (R²: {r2:.3f})")
        print("   ✅ Feature Engineering and Data Preprocessing")
        print("   ✅ Model Training, Testing, and Evaluation")

        print("\n🚀 Next steps:")
        print("   📚 Try different algorithms (SVM, Neural Networks, etc.)")
        print("   🔧 Experiment with feature engineering")
        print("   📊 Add data visualization with matplotlib")
        print("   🌐 Apply to your own datasets!")


# Run the tutorial
if __name__ == "__main__":
    tutorial = NetworkSecurityMLTutorial()
    tutorial.run_complete_tutorial()
