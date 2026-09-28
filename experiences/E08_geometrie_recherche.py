"""E08: reading dense retrieval scores: background level, margins and random control.

Chapter 5. Reuses the base and the questions of E07
(socle_experiences.py, constants DOCS and QUERIES) and the same encoder. Measures:
- question–document cosines, split between relevant and non-relevant pairs;
- the mean cosine between unrelated documents (background level of the space);
- the same computation on random vectors of the same dimension (control);
- for each question, the gap between the best score and that of the reference.

Usage: python experiences/E08_geometrie_recherche.py
Output: outputs/E08_geometrie_recherche/metadata.json
"""
import time

import numpy as np

from socle_experiences import DOCS, QUERIES, encoder, save


def cosinus_hors_diagonale(vecteurs):
    """Mean cosine between distinct vectors (vectors assumed normalized)."""
    similarites = vecteurs @ vecteurs.T
    n = len(vecteurs)
    return float((similarites.sum() - np.trace(similarites)) / (n * (n - 1)))


def main():
    debut = time.perf_counter()
    identifiants = [i for i, _ in DOCS]
    textes = [t for _, t in DOCS]
    questions = [q for _, q in QUERIES]
    vecteurs, revision = encoder(textes + questions)
    documents, requetes = vecteurs[:len(textes)], vecteurs[len(textes):]
    scores = requetes @ documents.T

    pertinents, autres, lignes = [], [], []
    for i, (reference, question) in enumerate(QUERIES):
        j = identifiants.index(reference)
        pertinents.append(float(scores[i, j]))
        autres.extend(float(scores[i, k]) for k in range(len(textes)) if k != j)
        meilleur = int(np.argmax(scores[i]))
        lignes.append({"question": question, "reference": reference,
                       "score_reference": float(scores[i, j]),
                       "premier": identifiants[meilleur], "score_premier": float(scores[i, meilleur]),
                       "ecart_au_premier": float(scores[i, meilleur] - scores[i, j])})

    generateur = np.random.default_rng(42)
    aleatoires = generateur.standard_normal(documents.shape)
    aleatoires /= np.linalg.norm(aleatoires, axis=1, keepdims=True)

    save("E08_geometrie_recherche", {
        "dimension": int(documents.shape[1]),
        "cosinus_pertinents": {"moyenne": float(np.mean(pertinents)), "min": float(np.min(pertinents)),
                               "max": float(np.max(pertinents))},
        "cosinus_non_pertinents": {"moyenne": float(np.mean(autres)), "min": float(np.min(autres)),
                                   "max": float(np.max(autres))},
        "fond_entre_documents": cosinus_hors_diagonale(documents),
        "fond_vecteurs_aleatoires": cosinus_hors_diagonale(aleatoires),
        "questions": lignes,
    }, "Que signifie un cosinus de recherche dense, et à quoi le comparer ?",
        ["Même base, mêmes questions et même encodeur que E2",
         "Témoin : vecteurs aléatoires de même dimension, normalisés"],
        ["Huit documents et huit questions fabriqués", "Un encodeur ; aucune généralisation à d'autres espaces",
         "Le niveau de fond dépend de la collection mesurée"],
        debut, {"MiniLM": revision})


if __name__ == "__main__":
    main()
