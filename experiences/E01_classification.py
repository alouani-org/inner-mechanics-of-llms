"""E01 : Classer des messages : la longueur suffit-elle ? (chapitres 1, 4).

Le code de l'expérience est la fonction `classification` de `socle_experiences.py`, qui
réunit aussi les fonctions communes aux autres programmes (chargement des modèles,
écriture des manifestes). Ce programme ne lance que cette expérience.

Usage : python experiences/E01_classification.py
Sortie : outputs/E01_classification/metadata.json
"""
from socle_experiences import classification

if __name__ == "__main__":
    print("DÉBUT E01", flush=True)
    classification()
