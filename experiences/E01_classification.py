"""E01: Classifying messages: is length enough? (chapters 1, 4).

The experiment's code is the `classification` function in `socle_experiences.py`, which
also gathers the functions shared by the other programs (model loading,
manifest writing). This program runs only this experiment.

Usage: python experiences/E01_classification.py
Output: outputs/E01_classification/metadata.json
"""
from socle_experiences import classification

if __name__ == "__main__":
    print("DÉBUT E01", flush=True)
    classification()
