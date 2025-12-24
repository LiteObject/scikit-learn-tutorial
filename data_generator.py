"""Synthetic email data generation and feature engineering.

This module handles creating realistic email samples for training
machine learning models. The EmailDataGenerator class simulates different
email types (spam and ham) with typical characteristics and patterns.
It performs critical feature engineering to convert raw email text
into numerical features suitable for scikit-learn models.
"""

import random
from typing import List, Tuple

import numpy as np


class EmailDataGenerator:
    """
    Generates synthetic email data for spam/ham classification.

    This class simulates realistic email patterns by creating message samples
    with specific word patterns and characteristics. It performs feature
    engineering to convert raw text into numerical features suitable for ML models.
    """

    def __init__(self):
        # Define spam and ham vocabulary patterns
        self.spam_words = [
            "free",
            "winner",
            "cash",
            "prize",
            "urgent",
            "click",
            "offer",
            "limited",
            "congratulations",
            "money",
            "credit",
            "discount",
            "deal",
            "buy",
            "cheap",
            "earn",
            "income",
            "million",
            "opportunity",
            "reward",
            "bonus",
            "gift",
            "claim",
            "selected",
            "exclusive",
            "guaranteed",
            "instant",
            "sale",
        ]

        self.ham_words = [
            "meeting",
            "project",
            "report",
            "schedule",
            "team",
            "review",
            "update",
            "tomorrow",
            "thanks",
            "attached",
            "please",
            "regards",
            "discuss",
            "feedback",
            "question",
            "help",
            "information",
            "confirm",
            "available",
            "deadline",
            "proposal",
            "draft",
            "call",
            "agenda",
            "notes",
            "summary",
            "forward",
            "appreciate",
            "colleague",
        ]

        # Common neutral words that appear in both
        self.neutral_words = [
            "the",
            "a",
            "is",
            "are",
            "to",
            "for",
            "you",
            "your",
            "this",
            "that",
            "we",
            "will",
            "be",
            "have",
            "from",
            "with",
            "can",
            "get",
        ]

        # Email templates for realistic generation
        self.spam_templates = [
            "You have been {word1}! {word2} your {word3} now!",
            "{word1}! {word2} offer just for you! {word3} today!",
            "Urgent: {word1} {word2} waiting. {word3} immediately!",
            "{word1} {word2}! Don't miss this {word3}!",
            "Congratulations! You're a {word1}! Claim your {word2} {word3}!",
        ]

        self.ham_templates = [
            "Hi, just wanted to {word1} about the {word2}. {word3}.",
            "Please {word1} the {word2} when you get a chance. {word3}.",
            "Following up on our {word1}. The {word2} looks good. {word3}.",
            "Can you {word1} the {word2}? Let me know. {word3}.",
            "Thanks for the {word1}. I'll {word2} the {word3} soon.",
        ]

        # Category definitions for classification
        self.categories = {
            0: {
                "name": "Ham",
                "description": "Legitimate email (not spam)",
                "base_spam_score": 0.1,
            },
            1: {
                "name": "Spam",
                "description": "Unwanted promotional or scam email",
                "base_spam_score": 0.9,
            },
        }

    def generate_email_text(self, is_spam: bool) -> str:
        """
        Generate realistic email text based on spam/ham patterns.

        Args:
            is_spam: True if generating spam, False for ham.

        Returns:
            Generated email text string.
        """
        if is_spam:
            template = random.choice(self.spam_templates)
            words = random.sample(self.spam_words, 3)
            text = template.format(word1=words[0], word2=words[1], word3=words[2])
            # Randomly add more spam words
            for _ in range(random.randint(2, 5)):
                text += f" {random.choice(self.spam_words)}"
        else:
            template = random.choice(self.ham_templates)
            words = random.sample(self.ham_words, 3)
            text = template.format(word1=words[0], word2=words[1], word3=words[2])
            # Randomly add more ham words
            for _ in range(random.randint(2, 5)):
                text += f" {random.choice(self.ham_words)}"

        # Add some neutral words for realism
        neutral = " ".join(random.sample(self.neutral_words, random.randint(3, 6)))
        text = f"{neutral} {text}"

        return text.lower()

    def extract_features(self, email_text: str) -> List[float]:
        """
        Extract a 10-dimensional feature vector from email text.

        This is the core feature engineering step. Each email
        gets transformed into numerical features that represent
        spam/ham characteristics. Examples:
        - Count of spam-related words
        - Count of ham-related words
        - Presence of urgency indicators
        - Text length characteristics

        Args:
            email_text: The email text content.

        Returns:
            List of 10 floats representing extracted features:
            [0]: Total word count
            [1]: Spam word count
            [2]: Ham word count
            [3]: Spam to total word ratio
            [4]: Has urgency words (urgent, act now, immediately)
            [5]: Has money words (free, cash, money, prize)
            [6]: Has exclamation marks
            [7]: Exclamation mark count
            [8]: Average word length
            [9]: Uppercase word ratio (before lowercasing)

        This is called FEATURE ENGINEERING - the most important part of ML!
        """
        if not email_text:
            return [0.0] * 10

        words = email_text.lower().split()

        # Feature 0: Total word count
        word_count = len(words)

        # Feature 1: Count of spam-related words
        spam_word_count = sum(1 for w in words if w in self.spam_words)

        # Feature 2: Count of ham-related words
        ham_word_count = sum(1 for w in words if w in self.ham_words)

        # Feature 3: Spam to total word ratio
        spam_ratio = spam_word_count / word_count if word_count > 0 else 0

        # Feature 4: Has urgency words
        urgency_words = {"urgent", "act", "now", "immediately", "hurry", "limited"}
        has_urgency = 1 if any(w in urgency_words for w in words) else 0

        # Feature 5: Has money-related words
        money_words = {"free", "cash", "money", "prize", "winner", "million", "dollar"}
        has_money = 1 if any(w in money_words for w in words) else 0

        # Feature 6: Has exclamation marks
        has_exclamation = 1 if "!" in email_text else 0

        # Feature 7: Exclamation mark count
        exclamation_count = email_text.count("!")

        # Feature 8: Average word length
        avg_word_length = (
            sum(len(w) for w in words) / word_count if word_count > 0 else 0
        )

        # Feature 9: Uppercase ratio (calculate before lowercasing in real scenario)
        uppercase_ratio = spam_word_count / word_count if word_count > 0 else 0

        return [
            float(word_count),  # [0] Total words
            float(spam_word_count),  # [1] Spam words
            float(ham_word_count),  # [2] Ham words
            float(spam_ratio),  # [3] Spam ratio
            float(has_urgency),  # [4] Has urgency
            float(has_money),  # [5] Has money words
            float(has_exclamation),  # [6] Has exclamation
            float(exclamation_count),  # [7] Exclamation count
            float(avg_word_length),  # [8] Avg word length
            float(uppercase_ratio),  # [9] Uppercase ratio
        ]

    def generate_sample(self, category: int) -> Tuple[List[float], int, float, str]:
        """
        Generate a single realistic training sample for spam or ham.

        Simulates typical email patterns with realistic noise.
        Spam scores have Gaussian noise added to reflect real-world variation.

        Args:
            category: Integer ID of the category (0=ham, 1=spam).

        Returns:
            Tuple containing:
            - features: List of 10 floats representing engineered features
            - category: The input category ID (0 or 1)
            - spam_score: Float between 0.0 and 1.0 indicating spam likelihood
            - email_text: The generated email text
        """
        is_spam = category == 1
        email_text = self.generate_email_text(is_spam)

        # Extract features
        sample_features = self.extract_features(email_text)

        # Generate spam score with some noise
        base_score = self.categories[category]["base_spam_score"]
        score_noise = random.gauss(0, 0.1)  # mean=0, std=0.1
        spam_score = max(0.0, min(1.0, base_score + score_noise))

        return sample_features, category, spam_score, email_text

    def generate_dataset(
        self, samples_per_class: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Generate a complete balanced training dataset of spam and ham emails.

        Creates multiple samples for each category (2 categories total) using
        generate_sample(). Returns data in NumPy array format suitable for
        scikit-learn models.

        Args:
            samples_per_class: Number of samples to generate per category (default 100).

        Returns:
            Tuple containing:
            - X: Feature matrix of shape (200, 10) with numerical features
            - y_category: Category labels of shape (200,) - 0=ham, 1=spam
            - y_spam_score: Spam score labels of shape (200,)
            - texts: List of email text strings
        """

        all_features = []
        all_category_labels = []
        all_spam_scores = []
        all_texts = []

        print("Generating training data...")
        print(f"Creating {samples_per_class} samples per category...")

        # Generate samples for each category
        for category in range(2):  # 0=ham, 1=spam
            category_name = self.categories[category]["name"]
            print(f"   Generating {category_name} samples...")

            for _ in range(samples_per_class):
                sample_features, label, spam_score, text = self.generate_sample(
                    category
                )
                all_features.append(sample_features)
                all_category_labels.append(label)
                all_spam_scores.append(spam_score)
                all_texts.append(text)

        # Convert to numpy arrays (required by scikit-learn)
        features_matrix = np.array(all_features)
        category_labels = np.array(all_category_labels)
        spam_scores = np.array(all_spam_scores)

        print(f"Generated {len(features_matrix)} total samples")
        print(f"Feature matrix shape: {features_matrix.shape}")
        print(f"Category labels shape: {category_labels.shape}")
        print(f"Spam scores shape: {spam_scores.shape}")

        return features_matrix, category_labels, spam_scores, all_texts

    def get_category_name(self, category: int) -> str:
        """
        Retrieve the human-readable name for a category.

        Args:
            category: Integer identifier for the category (0=ham, 1=spam).

        Returns:
            String name of the category ('Ham' or 'Spam').
        """
        return self.categories[category]["name"]

    # Backwards compatibility alias
    def get_device_name(self, category: int) -> str:
        """Alias for get_category_name for backwards compatibility."""
        return self.get_category_name(category)


# Test the data generator
if __name__ == "__main__":
    generator = EmailDataGenerator()

    # Test email generation
    print("Sample spam email:")
    spam_text = generator.generate_email_text(is_spam=True)
    print(f"  {spam_text}")

    print("\nSample ham email:")
    ham_text = generator.generate_email_text(is_spam=False)
    print(f"  {ham_text}")

    # Test feature extraction
    print("\nFeature extraction example:")
    features = generator.extract_features(spam_text)
    print(f"  Spam text features: {features}")

    features = generator.extract_features(ham_text)
    print(f"  Ham text features: {features}")

    # Generate small dataset
    print("\n" + "=" * 50)
    sample_features_matrix, sample_category_labels, sample_spam_scores, sample_texts = (
        generator.generate_dataset(samples_per_class=10)
    )
    print("\nSample features (first 3 rows):")
    print(sample_features_matrix[:3])
    print(f"\nSample category labels (first 10): {sample_category_labels[:10]}")
    print(f"Sample spam scores (first 5): {sample_spam_scores[:5]}")
    print(f"\nSample email texts:")
    for i in range(3):
        label = "SPAM" if sample_category_labels[i] == 1 else "HAM"
        print(f"  [{label}] {sample_texts[i][:60]}...")
