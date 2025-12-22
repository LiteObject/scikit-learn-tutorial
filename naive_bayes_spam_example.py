"""
Naive Bayes Spam Detection Example

This script demonstrates how to use MultinomialNB for text classification.
It builds a simple spam detector using scikit-learn's pipeline.

Key ML concepts demonstrated:
- Supervised learning: Training a model using labeled data
- Classification: Predicting discrete categories (spam vs ham)
- Feature extraction: Converting raw text into numeric feature vectors
- Training vs inference: Learning from data, then making predictions
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


def main():
    """
    Train a Naive Bayes spam detector and test it on new messages.

    This function demonstrates:
    - Creating a text classification pipeline with CountVectorizer and MultinomialNB
    - Training the model on labeled message data
    - Making predictions on unseen messages
    - Displaying probability scores for each prediction
    """

    # ==========================================================================
    # TRAINING DATA (what the model learns from)
    # ==========================================================================
    # Raw input: Text messages that will be transformed into features.
    # These are NOT features yet - they need to be converted to numeric vectors.
    messages: list[str] = [
        "Buy now! Limited time offer!",
        "Meeting at 3pm tomorrow",
        "You've won a prize! Click here!",
        "Can you review the project proposal?",
        "Free gift card waiting for you",
        "Let's grab coffee after work",
        "Urgent: Update your password now!",
        "Thanks for your help yesterday",
    ]

    # Labels (y): The target variable we want the model to predict.
    # Also called "ground truth" - these were assigned by human annotators.
    # This is supervised learning: we provide the correct
    # answers (label: spam or ham) during training.
    labels: list[int] = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham (not spam)

    # ==========================================================================
    # MODEL PIPELINE (feature extraction + classification)
    # ==========================================================================
    # A pipeline chains multiple steps together:
    # Step 1 - Feature extraction: CountVectorizer converts raw text into
    #          numeric feature vectors using "bag of words" representation.
    #
    #          Example:
    #          Vocabulary:   ["buy", "now", "limited", "time", "offer", "meeting", "3pm", "tomorrow", ...]
    #          "Buy now!" -> [1, 1, 0, 0, ...]
    #
    #          Counts word occurrences in each message. Each position = one word from vocabulary.
    #          If a message said "Buy buy buy now!", the vector would be [3, 1, 0, 0, ...]
    #          This is a simplification - it loses context like word order
    #          and meaning - but it works surprisingly well for spam detection because
    #          certain words ("free", "winner", "click") appear much more often in spam.
    # Step 2 - Classification: MultinomialNB uses these features to learn
    #          patterns that distinguish spam from ham.
    #
    #          Example:
    #          New message: "Free money!" -> [1, 1, 0, 0, ...]
    #          Spam score: P(spam) × P(free|spam) × P(money|spam) = 0.5 × 0.8 × 0.6 = 0.24
    #          Ham score:  P(ham) × P(free|ham) × P(money|ham)   = 0.5 × 0.05 × 0.1 = 0.0025
    #          0.24 > 0.0025 → Predict: SPAM
    #
    #          "Multinomial" means it works with word counts, not just presence/absence.
    spam_detector = Pipeline(
        [("vectorizer", CountVectorizer()), ("classifier", MultinomialNB())]
    )

    # ==========================================================================
    # TRAINING (model learns patterns from labeled data)
    # ==========================================================================
    # fit() is where learning happens. The model analyzes the training data
    # and learns which word patterns are associated with spam vs ham.
    spam_detector.fit(messages, labels)
    print("Model trained on", len(messages), "messages\n")

    # ==========================================================================
    # INFERENCE / PREDICTION (applying the trained model to new data)
    # ==========================================================================
    # Test data: Unseen messages the model has never encountered during training.
    # This tests whether the model can generalize to new examples.
    new_messages: list[str] = [
        "Free money waiting for you!",
        "Are you coming to the meeting?",
        "Click here to claim your reward",
        "Can we reschedule our call to Friday?",
        "You have been selected as a winner!",
    ]

    # predict() applies the trained model to make predictions on new data.
    # The pipeline automatically: (1) extracts features, (2) classifies.
    predictions = spam_detector.predict(new_messages)

    print("Predictions:")
    print("-" * 50)
    for msg, pred in zip(new_messages, predictions):
        status = "SPAM" if pred == 1 else "Not spam"
        print(f"{status:>{len('Not spam')}}: '{msg}'")

    # ==========================================================================
    # PROBABILITY SCORES (model confidence)
    # ==========================================================================
    # predict_proba() returns the model's confidence for each class.
    # Note: Naive Bayes probabilities are often poorly calibrated (overconfident),
    # but the relative ordering is usually reliable for ranking.
    print("\nProbability scores:")
    print("-" * 50)
    probabilities = spam_detector.predict_proba(new_messages)
    for msg, probs in zip(new_messages, probabilities):
        ham_prob, spam_prob = probs
        print(f"Ham: {ham_prob:.2%}, Spam: {spam_prob:.2%} -> '{msg[:40]}...'")


if __name__ == "__main__":
    main()
