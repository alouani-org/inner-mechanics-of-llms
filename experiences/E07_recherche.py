"""E07 : Recherche lexicale et dense (chapitre 5).

Le code de l'expérience est la fonction `retrieval` de `socle_experiences.py`, qui
réunit aussi les fonctions communes aux autres programmes (chargement des modèles,
écriture des manifestes). Ce programme ne lance que cette expérience.

Usage : python experiences/E07_recherche.py
Sortie : outputs/E07_recherche/metadata.json
"""
from socle_experiences import retrieval

if __name__ == "__main__":
    print("DÉBUT E07", flush=True)
    retrieval()
