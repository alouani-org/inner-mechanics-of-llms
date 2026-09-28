"""E07: Lexical and dense retrieval (chapter 5).

The experiment's code is the `retrieval` function in `socle_experiences.py`, which
also gathers the functions shared by the other programs (model loading,
manifest writing). This program runs only this experiment.

Usage: python experiences/E07_recherche.py
Output: outputs/E07_recherche/metadata.json
"""
from socle_experiences import retrieval

if __name__ == "__main__":
    print("DÉBUT E07", flush=True)
    retrieval()
