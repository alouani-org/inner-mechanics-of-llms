"""E21 : Même question, quatre langues (chapitre 11).

Le code de l'expérience est la fonction `languages` de `socle_experiences.py`, qui
réunit aussi les fonctions communes aux autres programmes (chargement des modèles,
écriture des manifestes). Ce programme ne lance que cette expérience.

Usage : python experiences/E21_langues.py
Sortie : outputs/E21_langues/metadata.json
"""
from socle_experiences import languages

if __name__ == "__main__":
    print("DÉBUT E21", flush=True)
    languages()
