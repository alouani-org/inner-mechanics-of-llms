"""E16: Causal intervention (activation patching) (chapter 9).

The experiment's code is the `causal` function in `socle_experiences.py`, which
also gathers the functions shared by the other programs (model loading,
manifest writing). This program runs only this experiment.

Usage: python experiences/E16_intervention.py
Output: outputs/E16_intervention/metadata.json
"""
from socle_experiences import causal

if __name__ == "__main__":
    print("DÉBUT E16", flush=True)
    causal()
