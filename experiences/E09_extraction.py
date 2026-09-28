"""E09: Free and constrained extraction (chapter 6).

The experiment's code is the `extraction` function in `socle_experiences.py`, which
also gathers the functions shared by the other programs (model loading,
manifest writing). This program runs only this experiment.

Usage: python experiences/E09_extraction.py
Output: outputs/E09_extraction/metadata.json
"""
from socle_experiences import extraction

if __name__ == "__main__":
    print("DÉBUT E09", flush=True)
    extraction()
