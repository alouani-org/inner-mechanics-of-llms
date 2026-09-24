"""E14 : lire l'attention de GPT-2, puis tester une lecture par une ablation.

Chapitre « Inspecter une prédiction ». Trois mesures :

1. Sur le prompt amorcé, poids d'attention du dernier token vers le sujet
   (« France ») et vers le premier token, couche par couche ; même lecture sur
   une phrase témoin sans relation géographique.
2. Têtes d'induction : sur des séquences de tokens aléatoires répétées, poids que
   chaque tête accorde, depuis la seconde occurrence d'un token, au token qui
   suivait sa première occurrence. Témoin : seconde moitié non répétée.
3. Passage à l'intervention : neutraliser les trois têtes au plus fort score
   (sélectionnées sur des séquences de découverte) et mesurer la perte sur la
   seconde moitié de séquences de validation, comparée à la neutralisation de
   trois têtes tirées au hasard.

Usage : python experiences/E14_attention.py
Sortie : outputs/E14_attention/metadata.json
"""
import time

import numpy as np
import torch

from socle_experiences import model, snapshot, save

PROMPT = "The capital of Germany is Berlin. The capital of France is"
TEMOIN = "The cat slept on the old mat while the dog was"
LONGUEUR = 25          # tokens aléatoires par moitié
N_DECOUVERTE = 8       # séquences servant à choisir les têtes
N_VALIDATION = 8       # séquences servant à mesurer l'effet de l'ablation
N_TIRAGES_TEMOINS = 10 # ensembles de trois têtes tirées au hasard


def attentions(m, ids):
    """Tuple de 12 tenseurs (têtes, positions, positions) pour une séquence d'identifiants."""
    with torch.no_grad():
        sortie = m(torch.tensor([ids]), output_attentions=True)
    return [a[0] for a in sortie.attentions]


def lecture_dernier_token(m, tok, texte, mot=None):
    """Par couche : attention maximale (sur les têtes) vers un mot, et moyenne vers la position 0."""
    ids = tok.encode(texte)
    position_mot = None
    if mot is not None:
        position_mot = ids.index(tok.encode(" " + mot)[0])
    lignes = []
    for couche, a in enumerate(attentions(m, ids)):
        depuis_fin = a[:, -1, :]                      # (têtes, positions)
        ligne = {"couche": couche,
                 "vers_premier_token_moyenne": float(depuis_fin[:, 0].mean())}
        if position_mot is not None:
            ligne["vers_mot_max"] = float(depuis_fin[:, position_mot].max())
            ligne["tete_max"] = int(depuis_fin[:, position_mot].argmax())
        lignes.append(ligne)
    return {"texte": texte, "tokens": [tok.decode([i]) for i in ids],
            "position_mot": position_mot, "couches": lignes}


def sequence_repetee(generateur, tok, repeter=True):
    """Un token de début, L tokens aléatoires, puis les mêmes (ou d'autres) L tokens."""
    premiere = generateur.integers(1000, 20000, LONGUEUR).tolist()
    seconde = premiere if repeter else generateur.integers(1000, 20000, LONGUEUR).tolist()
    return [tok.eos_token_id] + premiere + seconde


def score_induction(a):
    """Poids moyen de la position i vers la position i - L + 1, sur la seconde moitié."""
    positions = range(LONGUEUR + 1, 2 * LONGUEUR + 1)
    return torch.stack([a[:, i, i - LONGUEUR + 1] for i in positions], dim=1).mean(1)


def perte_seconde_moitie(m, ids):
    """Perte moyenne de prédiction du token suivant, sur la seconde moitié."""
    entree = torch.tensor([ids])
    with torch.no_grad():
        logits = m(entree).logits[0]
    logprob = logits[:-1].log_softmax(-1)
    cibles = entree[0, 1:]
    pertes = -logprob[torch.arange(len(cibles)), cibles]
    return float(pertes[LONGUEUR:].mean())


def neutraliser_tetes(m, tetes):
    """Accrocher des hooks qui annulent la sortie de certaines têtes avant la projection c_proj."""
    largeur = m.config.n_embd // m.config.n_head
    crochets = []
    par_couche = {}
    for couche, tete in tetes:
        par_couche.setdefault(couche, []).append(tete)
    for couche, liste in par_couche.items():
        def annuler(module, arguments, liste=liste):
            x = arguments[0].clone()
            for tete in liste:
                x[..., tete * largeur:(tete + 1) * largeur] = 0.0
            return (x,)
        crochets.append(m.transformer.h[couche].attn.c_proj.register_forward_pre_hook(annuler))
    return crochets


def perte_avec_ablation(m, sequences, tetes):
    crochets = neutraliser_tetes(m, tetes)
    try:
        return float(np.mean([perte_seconde_moitie(m, s) for s in sequences]))
    finally:
        for crochet in crochets:
            crochet.remove()


def main():
    debut = time.perf_counter()
    m, tok = model()
    generateur = np.random.default_rng(42)

    lecture = lecture_dernier_token(m, tok, PROMPT, "France")
    temoin = lecture_dernier_token(m, tok, TEMOIN)

    decouverte = [sequence_repetee(generateur, tok) for _ in range(N_DECOUVERTE)]
    validation = [sequence_repetee(generateur, tok) for _ in range(N_VALIDATION)]
    sans_repetition = [sequence_repetee(generateur, tok, repeter=False) for _ in range(N_VALIDATION)]

    scores = torch.stack([torch.stack([score_induction(a) for a in attentions(m, s)]) for s in decouverte]).mean(0)
    scores_temoin = torch.stack([torch.stack([score_induction(a) for a in attentions(m, s)])
                                 for s in sans_repetition]).mean(0)
    classement = sorted(((float(scores[c, t]), c, t) for c in range(scores.shape[0])
                         for t in range(scores.shape[1])), reverse=True)
    meilleures = [(c, t) for _, c, t in classement[:3]]

    perte_intacte = float(np.mean([perte_seconde_moitie(m, s) for s in validation]))
    perte_sans_repetition = float(np.mean([perte_seconde_moitie(m, s) for s in sans_repetition]))
    perte_ablation = perte_avec_ablation(m, validation, meilleures)
    toutes = [(c, t) for c in range(m.config.n_layer) for t in range(m.config.n_head)]
    candidates = [h for h in toutes if h not in meilleures]
    temoins = []
    for _ in range(N_TIRAGES_TEMOINS):
        choix = [candidates[i] for i in generateur.choice(len(candidates), 3, replace=False)]
        temoins.append({"tetes": [{"couche": c, "tete": t} for c, t in choix],
                        "perte": perte_avec_ablation(m, validation, choix)})

    save("E14_attention", {
        "lecture_prompt": lecture, "lecture_temoin": temoin,
        "induction": {
            "longueur_moitie": LONGUEUR,
            "score_par_couche_max": [float(scores[c].max()) for c in range(scores.shape[0])],
            "score_par_couche_moyen": [float(scores[c].mean()) for c in range(scores.shape[0])],
            "temoin_non_repete_max": [float(scores_temoin[c].max()) for c in range(scores.shape[0])],
            "dix_meilleures_tetes": [{"couche": c, "tete": t, "score": s} for s, c, t in classement[:10]],
        },
        "ablation": {
            "tetes_neutralisees": [{"couche": c, "tete": t} for c, t in meilleures],
            "perte_intacte": perte_intacte,
            "perte_sans_repetition": perte_sans_repetition,
            "perte_apres_ablation": perte_ablation,
            "temoins_aleatoires": temoins,
        },
    }, "Que montrent les poids d'attention, et une tête à fort score d'induction compte-t-elle pour la copie ?",
        ["Phrase témoin sans relation géographique", "Séquences non répétées comme témoin du score d'induction",
         "Têtes choisies sur des séquences de découverte, effet mesuré sur d'autres séquences",
         "Ablation de trois têtes aléatoires, dix tirages"],
        ["Un poids d'attention décrit un mélange, pas une cause",
         "Séquences aléatoires artificielles : la copie de texte naturel n'est pas testée",
         "L'ablation par mise à zéro est une intervention parmi d'autres ; elle ne prouve pas un circuit complet",
         "Un modèle et une graine"],
        debut, {"gpt2": snapshot("gpt2")[1]})


if __name__ == "__main__":
    main()
