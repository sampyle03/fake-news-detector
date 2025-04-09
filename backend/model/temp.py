import pandas as pd
import os
from collections import defaultdict

def find_similar_words(my_dict):
    def get_edits(word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletions = [a + b[1:] for a, b in splits if b]
        insertions = [a + c + b for a, b in splits for c in letters]
        substitutions = [a + c + b[1:] for a, b in splits if b for c in letters if c != b[0]]
        return set(deletions + insertions + substitutions)
    
    # Preprocess the dictionary into individual words with their values
    word_value_pairs = []
    for key, value in my_dict.items():
        phrase = key[0]
        words = phrase.split()
        for word in words:
            word_value_pairs.append((word, value))
    
    # Create a map from normalized (lowercase) words to their original forms and values
    word_map = defaultdict(list)
    for word, value in word_value_pairs:
        normalized = word.lower()
        word_map[normalized].append((word, value))
    
    similar_pairs = []
    
    # Process exact matches (same normalized word, different occurrences)
    for entries in word_map.values():
        # Add all unique pairs within the same normalized word
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                similar_pairs.append((entries[i], entries[j]))
    
    # Process edits with one character difference
    processed_words = set()
    for word in list(word_map.keys()):  # Iterate over a static list of keys
        # Generate all possible single-edit variations
        edits = get_edits(word)
        for edit in edits:
            # Check if the edit exists and avoid reverse duplicates with lex order
            if edit in word_map and edit > word:
                for orig_word, orig_val in word_map[word]:
                    for edit_word, edit_val in word_map[edit]:
                        similar_pairs.append(((orig_word, orig_val), (edit_word, edit_val)))
    
    return similar_pairs

# Example usage with the provided dictionary:
current_dir = os.path.dirname(__file__)
words_in_corpus =  pd.read_pickle(os.path.join(current_dir, "../data/pickle/words_in_corpus.pkl"))
similar_words = find_similar_words(words_in_corpus)
for pair in similar_words:
    print(pair)