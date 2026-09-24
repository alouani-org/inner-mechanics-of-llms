"""E16 : Intervention causale (patching d'activation) (chapitre 9).

Le code de l'expérience est la fonction `causal` de `socle_experiences.py`, qui
réunit aussi les fonctions communes aux autres programmes (chargement des modèles,
écriture des manifestes). Ce programme ne lance que cette expérience.

Usage : python experiences/E16_intervention.py
Sortie : outputs/E16_intervention/metadata.json
"""
from socle_experiences import causal

if __name__ == "__main__":
    print("DÉBUT E16", flush=True)
    causal()
