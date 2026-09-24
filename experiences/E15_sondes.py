"""E15 : sondes linéaires : que lit-on dans les états internes, et contre quelle explication ?

Chapitre « Inspecter une prédiction ». Deux questions, mêmes représentations
(dernier token de la phrase, treize lectures de GPT-2) :

1. Sonde de sujet : phrases « capitale d'un pays » contre phrases de préférence
   personnelle. Explication rivale : les mots eux-mêmes suffisent. Témoin : un
   classificateur lexical (TF-IDF) sur les mêmes plis.
2. Sonde de connaissance : le pays appartient-il à l'Europe ? Le continent
   n'est écrit nulle part dans la phrase. Les plis sont groupés par pays : un
   pays du test n'a jamais été vu à l'entraînement. Témoins : classificateur
   lexical sur caractères, étiquettes permutées (plis identiques) et « tâche de
   contrôle » attribuant à chaque pays une étiquette arbitraire.

Usage : python experiences/E15_sondes.py
Sortie : outputs/E15_sondes/metadata.json
"""
import time

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from socle_experiences import model, snapshot, save

EUROPE = ["France", "Germany", "Italy", "Spain", "Portugal", "Poland", "Greece", "Sweden",
          "Norway", "Finland", "Denmark", "Austria", "Hungary", "Ireland", "Belgium",
          "Netherlands", "Switzerland", "Romania", "Bulgaria", "Croatia"]
AILLEURS = ["Japan", "China", "India", "Brazil", "Argentina", "Mexico", "Canada", "Egypt",
            "Kenya", "Nigeria", "Peru", "Chile", "Colombia", "Vietnam", "Thailand",
            "Indonesia", "Australia", "Morocco", "Iran", "Pakistan"]
GABARITS_PAYS = ["The capital of {} is", "{} is a country whose capital is"]
PREFERENCES = [f"The favorite {objet} of my {proche} is"
               for objet in ["color", "food", "song", "sport", "book", "movie", "flower", "drink",
                             "game", "animal"]
               for proche in ["sister", "brother", "neighbor", "friend"]]
N_PLIS = 5
N_PERMUTATIONS = 20


def etats(m, tok, phrases):
    """Pour chaque phrase, l'état du dernier token à chacune des treize lectures."""
    lignes = []
    for phrase in phrases:
        with torch.no_grad():
            h = m(**tok(phrase, return_tensors="pt"), output_hidden_states=True).hidden_states
        lignes.append(torch.stack([x[0, -1] for x in h]).numpy())
    return np.stack(lignes, axis=1)            # (lectures, phrases, largeur)


def exactitude_croisee(X, y, plis, fabrique):
    """Exactitude moyenne sur des plis fixés d'avance (mêmes indices pour tous les témoins)."""
    scores = []
    for entrainement, test in plis:
        modele = fabrique().fit(X[entrainement] if not isinstance(X, list) else [X[i] for i in entrainement],
                                y[entrainement])
        Xt = X[test] if not isinstance(X, list) else [X[i] for i in test]
        scores.append(float((modele.predict(Xt) == y[test]).mean()))
    return float(np.mean(scores))


def sonde():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1, random_state=42))


def profil(representations, y, plis, generateur, controle_par_groupe=None):
    """Exactitude réelle, moyenne des permutations et tâche de contrôle, lecture par lecture."""
    lignes = []
    for lecture, X in enumerate(representations):
        reel = exactitude_croisee(X, y, plis, sonde)
        permutations = [exactitude_croisee(X, generateur.permutation(y), plis, sonde)
                        for _ in range(N_PERMUTATIONS)]
        ligne = {"lecture": lecture, "exactitude": reel,
                 "permutation_moyenne": float(np.mean(permutations)),
                 "permutation_max": float(np.max(permutations))}
        if controle_par_groupe is not None:
            ligne["tache_controle"] = exactitude_croisee(X, controle_par_groupe, plis, sonde)
        lignes.append(ligne)
    return lignes


def main():
    debut = time.perf_counter()
    m, tok = model()
    generateur = np.random.default_rng(42)
    pays = EUROPE + AILLEURS

    # 1. Sonde de sujet : 40 phrases de capitale contre 40 phrases de préférence.
    capitales = [GABARITS_PAYS[0].format(p) for p in pays]
    phrases_sujet = capitales + PREFERENCES
    y_sujet = np.array([1] * len(capitales) + [0] * len(PREFERENCES))
    groupes_sujet = np.arange(len(phrases_sujet)) % 40
    plis_sujet = list(GroupKFold(N_PLIS).split(phrases_sujet, y_sujet, groupes_sujet))
    lexical_sujet = exactitude_croisee(phrases_sujet, y_sujet, plis_sujet,
                                       lambda: make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=2000)))
    profil_sujet = profil(etats(m, tok, phrases_sujet), y_sujet, plis_sujet, generateur)

    # 2. Sonde de connaissance : Europe ou non, deux gabarits, plis groupés par pays.
    phrases_pays = [g.format(p) for p in pays for g in GABARITS_PAYS]
    y_pays = np.array([1 if p in EUROPE else 0 for p in pays for _ in GABARITS_PAYS])
    groupes_pays = np.array([i for i in range(len(pays)) for _ in GABARITS_PAYS])
    ordre = generateur.permutation(len(pays))
    groupes_melanges = ordre[groupes_pays]      # répartition des pays entre plis, fixée une fois
    plis_pays = list(GroupKFold(N_PLIS).split(phrases_pays, y_pays, groupes_melanges))
    etiquette_arbitraire = generateur.integers(0, 2, len(pays))
    controle = etiquette_arbitraire[groupes_pays]
    lexical_pays = exactitude_croisee(phrases_pays, y_pays, plis_pays,
                                      lambda: make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)),
                                                            LogisticRegression(max_iter=2000)))
    profil_pays = profil(etats(m, tok, phrases_pays), y_pays, plis_pays, generateur, controle)

    save("E15_sondes", {
        "sujet": {"n": len(phrases_sujet), "lexical_tfidf": lexical_sujet, "profil": profil_sujet},
        "connaissance": {"n": len(phrases_pays), "n_pays": len(pays), "lexical_caracteres": lexical_pays,
                         "profil": profil_pays, "plis_groupes_par_pays": True},
        "plis": N_PLIS, "permutations": N_PERMUTATIONS,
    }, "Une sonde précise lit-elle le sujet écrit, ou une connaissance absente du texte ?",
        ["Plis fixés et identiques pour la sonde et ses témoins",
         "Témoin lexical sur les mêmes plis",
         "Plis groupés par pays pour la sonde de connaissance",
         "Vingt permutations d'étiquettes ; tâche de contrôle par pays"],
        ["Liste de pays et partition Europe/ailleurs choisies par l'auteur de l'expérience",
         "Décodabilité seulement : aucune intervention ne teste l'usage de l'information par le modèle",
         "Deux gabarits anglais ; un modèle"],
        debut, {"gpt2": snapshot("gpt2")[1]})


if __name__ == "__main__":
    main()
