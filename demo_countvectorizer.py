"""
CountVectorizer Demo

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
result = vec.fit_transform(message).toarray()
print(result)

# Print the words that correspond to the columns
print("\nVocabulary:")
print(vec.get_feature_names_out())

print("\nReadable DataFrame:")
print(pd.DataFrame(result, columns=vec.get_feature_names_out()))
