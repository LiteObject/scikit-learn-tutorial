"""
CountVectorizer Demo

CountVectorizer is a feature extraction tool from sklearn.

This script demonstrates the basic usage of Scikit-Learn's CountVectorizer
for converting text into numeric feature vectors (Bag of Words).

It highlights:
- How text is tokenized
- How vocabulary is built
- How to inspect the resulting feature matrix
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

message = [
    "I love code.",
    "I love Python code.",
]

# By default, CountVectorizer ignores single characters like "I" or "a".
# It assumes that single letters (like "I", "a") are usually not useful
# for determining the topic of a text.
# Example:
# "I love code" -> 1 "code", 1 "love", 0 "python"
# "I love Python code" -> 1 "code", 1 "love", 1 "python"
#
# Use a custom token_pattern to include single-letter words like "I"
# vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b")

# Initialize the "Bag of Words" model
vec = CountVectorizer()

# 1. Fit and Transform (Returns a Sparse Matrix)
sparse_matrix = vec.fit_transform(message)

print("\nSparse Matrix (Stores only non-zero values):")
print(sparse_matrix)

# 2. Convert to Dense Array (for readability ONLY)
#    In real projects, you rarely do this.
#    We do it here so we can print the matrix and see the 0s.
#    Scikit-learn models accept the sparse_matrix directly.
result = sparse_matrix.toarray()  # type: ignore

print("\nDense Array (Stores every value, including zeros):")
print(result)

# Print the words that correspond to the columns
print("\nVocabulary:")
print(vec.get_feature_names_out())

print("\nReadable DataFrame:")
print(pd.DataFrame(result, columns=vec.get_feature_names_out()))
