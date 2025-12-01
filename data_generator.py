"""Synthetic network data generation and feature engineering.

This module handles creating realistic network traffic patterns for training
machine learning models. The NetworkDataGenerator class simulates different
device types (IoT, servers, workstations, etc.) with typical port configurations
and risk profiles. It performs critical feature engineering to convert raw port
data into numerical features suitable for scikit-learn models.
"""

import random
from typing import List, Tuple

import numpy as np


class NetworkDataGenerator:
    """
    Generates synthetic network traffic data for security classification.

    This class simulates realistic network patterns by creating device profiles
    with specific port configurations. It performs feature engineering to convert
    raw port data into a 10-dimensional feature vector suitable for ML models.
    """

    def __init__(self):
        # Define realistic device patterns based on real networks
        self.device_patterns = {
            0: {
                "name": "IoT Device",
                "typical_ports": [80],
                "base_risk": 0.3,
                "description": "Smart home devices, cameras, sensors",
            },
            1: {
                "name": "Linux Server",
                "typical_ports": [22, 80, 443],
                "base_risk": 0.4,
                "description": "Web servers, application servers",
            },
            2: {
                "name": "Windows PC",
                "typical_ports": [135, 3389, 445],
                "base_risk": 0.6,
                "description": "Desktop computers, workstations",
            },
            3: {
                "name": "Printer",
                "typical_ports": [631, 9100],
                "base_risk": 0.2,
                "description": "Network printers, scanners",
            },
            4: {
                "name": "Router/Gateway",
                "typical_ports": [22, 80, 443, 8080],
                "base_risk": 0.5,
                "description": "Network infrastructure devices",
            },
            5: {
                "name": "Vulnerable Device",
                "typical_ports": [23, 21, 3389, 445],
                "base_risk": 0.9,
                "description": "Devices with high-risk services",
            },
        }

    def extract_features(self, ports: List[int]) -> List[float]:
        """
        Extract a 10-dimensional feature vector from a list of open ports.

        This is the core feature engineering step. Each port configuration
        gets transformed into numerical features that represent network
        characteristics. Examples:
        - Presence of well-known ports (SSH, RDP, SMB, FTP)
        - Port range (max - min port number)
        - Number of high ports (>1024) indicating services

        Args:
            ports: List of integer port numbers.

        Returns:
            List of 10 floats representing extracted features:
            [0]: Total number of open ports
            [1]: SSH port (22) presence
            [2]: HTTP port (80) presence
            [3]: HTTPS port (443) presence
            [4]: Telnet port (23) presence
            [5]: RDP port (3389) presence
            [6]: SMB port (445) presence
            [7]: FTP port (21) presence
            [8]: Port range (max - min)
            [9]: High ports count (>1024)

        This is called FEATURE ENGINEERING - the most important part of ML!
        """
        if not ports:
            return [0.0] * 10

        # Feature 0: Total number of open ports
        num_ports = len(ports)

        # Features 1-7: Presence of specific important ports (binary features)
        has_ssh = 1 if 22 in ports else 0  # SSH (secure remote access)
        has_http = 1 if 80 in ports else 0  # HTTP (web server)
        has_https = 1 if 443 in ports else 0  # HTTPS (secure web)
        has_telnet = 1 if 23 in ports else 0  # Telnet (insecure!)
        has_rdp = 1 if 3389 in ports else 0  # Remote Desktop
        has_smb = 1 if 445 in ports else 0  # File sharing
        has_ftp = 1 if 21 in ports else 0  # File transfer

        # Feature 8: Port range spread (max - min)
        port_spread = max(ports) - min(ports) if len(ports) > 1 else 0

        # Feature 9: Number of high ports (> 1024)
        high_ports = len([p for p in ports if p > 1024])

        # Return as list of floats (required by scikit-learn)
        return [
            float(num_ports),  # [0] Total ports
            float(has_ssh),  # [1] Has SSH
            float(has_http),  # [2] Has HTTP
            float(has_https),  # [3] Has HTTPS
            float(has_telnet),  # [4] Has Telnet (risky!)
            float(has_rdp),  # [5] Has RDP (risky!)
            float(has_smb),  # [6] Has SMB (risky!)
            float(has_ftp),  # [7] Has FTP (risky!)
            float(port_spread),  # [8] Port range
            float(high_ports),  # [9] High port count
        ]

    def generate_sample(self, device_type: int) -> Tuple[List[float], int, float]:
        """
        Generate a single realistic training sample for a specific device type.

        Simulates typical port patterns with realistic noise (30% chance of additional
        ports). Risk scores have Gaussian noise added to reflect real-world variation.

        Args:
            device_type: Integer ID of the device type (0-5).

        Returns:
            Tuple containing:
            - features: List of 10 floats representing engineered features
            - device_type: The input device type ID
            - risk_score: Float between 0.0 and 1.0 indicating network risk
        """
        pattern = self.device_patterns[device_type]

        # Start with typical ports for this device type
        ports = pattern["typical_ports"].copy()

        # Add realistic noise (30% chance of additional ports)
        if random.random() > 0.7:
            # Add 1-3 random ports to simulate real-world variation
            extra_ports = random.sample(range(1024, 65535), random.randint(1, 3))
            ports.extend(extra_ports)

        # Extract features
        sample_features = self.extract_features(ports)

        # Generate risk score with some noise
        base_risk = pattern["base_risk"]
        # Add Gaussian noise (normal distribution) to make it realistic
        risk_noise = random.gauss(0, 0.1)  # mean=0, std=0.1
        risk_score = max(0.0, min(1.0, base_risk + risk_noise))

        return sample_features, device_type, risk_score

    def generate_dataset(
        self, samples_per_class: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a complete balanced training dataset across all device types.

        Creates multiple samples for each device type (6 types total) using
        generate_sample(). Returns data in NumPy array format suitable for
        scikit-learn models.

        Args:
            samples_per_class: Number of samples to generate per device type (default 100).

        Returns:
            Tuple containing:
            - X: Feature matrix of shape (600, 10) with scaled and normalized data
            - y_device: Device type labels of shape (600,)
            - y_risk: Risk score labels of shape (600,)
        """

        all_features = []
        all_device_labels = []
        all_risk_scores = []

        print("🏭 Generating training data...")
        print(f"📊 Creating {samples_per_class} samples per device type...")

        # Generate samples for each device type
        for device_type in range(6):  # 0-5 device types
            device_name = self.device_patterns[device_type]["name"]
            print(f"   📱 Generating {device_name} samples...")

            for _ in range(samples_per_class):
                sample_features, label, risk = self.generate_sample(device_type)
                all_features.append(sample_features)
                all_device_labels.append(label)
                all_risk_scores.append(risk)

        # Convert to numpy arrays (required by scikit-learn)
        features_matrix = np.array(all_features)  # Features: shape (600, 10)
        device_labels = np.array(all_device_labels)  # Device labels: shape (600,)
        risk_scores = np.array(all_risk_scores)  # Risk scores: shape (600,)

        print(f"✅ Generated {len(features_matrix)} total samples")
        print(f"📏 Feature matrix shape: {features_matrix.shape}")
        print(f"🏷️ Device labels shape: {device_labels.shape}")
        print(f"⚡ Risk scores shape: {risk_scores.shape}")

        return features_matrix, device_labels, risk_scores

    def get_device_name(self, device_type: int) -> str:
        """
        Retrieve the human-readable name for a device type.

        Args:
            device_type: Integer identifier for the device type (0-5).

        Returns:
            String name of the device type (e.g., 'Linux Server', 'Windows PC').
        """
        return self.device_patterns[device_type]["name"]


# Test the data generator
if __name__ == "__main__":
    generator = NetworkDataGenerator()

    # Test feature extraction
    test_ports = [22, 80, 443, 3389]  # SSH + HTTP + HTTPS + RDP
    features = generator.extract_features(test_ports)
    print(f"Test ports {test_ports} -> Features: {features}")

    # Generate small dataset
    sample_features_matrix, sample_device_labels, sample_risk_scores = (
        generator.generate_dataset(samples_per_class=10)
    )
    print("\nSample features (first 3 rows):")
    print(sample_features_matrix[:3])
    print(f"\nSample device labels (first 10): {sample_device_labels[:10]}")
    print(f"Sample risk scores (first 10): {sample_risk_scores[:10]}")
