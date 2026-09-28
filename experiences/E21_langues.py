"""E21: Same question, four languages (chapter 11).

The experiment's code is the `languages` function in `socle_experiences.py`, which
also gathers the functions shared by the other programs (model loading,
manifest writing). This program runs only this experiment.

Usage: python experiences/E21_langues.py
Output: outputs/E21_langues/metadata.json
"""
from socle_experiences import languages

if __name__ == "__main__":
    print("DÉBUT E21", flush=True)
    languages()
