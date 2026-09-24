"""E09 : Extraction libre et contrainte (chapitre 6).

Le code de l'expérience est la fonction `extraction` de `socle_experiences.py`, qui
réunit aussi les fonctions communes aux autres programmes (chargement des modèles,
écriture des manifestes). Ce programme ne lance que cette expérience.

Usage : python experiences/E09_extraction.py
Sortie : outputs/E09_extraction/metadata.json
"""
from socle_experiences import extraction

if __name__ == "__main__":
    print("DÉBUT E09", flush=True)
    extraction()
