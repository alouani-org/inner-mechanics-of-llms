"""E14: read GPT-2's attention, then test a reading with an ablation.

Chapter 8. Three measurements:

1. On the primed prompt, attention weights from the last token to the subject
   ("France") and to the first token, layer by layer; same reading on
   a control sentence without a geographic relation.
2. Induction heads: on repeated sequences of random tokens, weight that
   each head gives, from the second occurrence of a token, to the token that
   followed its first occurrence. Control: non-repeated second half.
3. Moving to intervention: knock out the three highest-scoring heads
   (selected on discovery sequences) and measure the loss on the
   second half of validation sequences, compared with knocking out
   three randomly drawn heads.

Usage: python experiences/E14_attention.py
Output: outputs/E14_attention/metadata.json
"""
import time

import numpy as np
import torch

from socle_experiences import model, snapshot, save

PROMPT = "The capital of Germany is Berlin. The capital of France is"
TEMOIN = "The cat slept on the old mat while the dog was"
LONGUEUR = 25          # random tokens per half
N_DECOUVERTE = 8       # sequences used to choose the heads
N_VALIDATION = 8       # sequences used to measure the ablation effect
N_TIRAGES_TEMOINS = 10 # sets of three randomly drawn heads


def attentions(m, ids):
    """Tuple of 12 tensors (heads, positions, positions) for a sequence of ids."""
    with torch.no_grad():
        sortie = m(torch.tensor([ids]), output_attentions=True)
    return [a[0] for a in sortie.attentions]


def lecture_dernier_token(m, tok, texte, mot=None):
    """Per layer: maximum attention (over heads) to a word, and mean attention to position 0."""
    ids = tok.encode(texte)
    position_mot = None
    if mot is not None:
        position_mot = ids.index(tok.encode(" " + mot)[0])
    lignes = []
    for couche, a in enumerate(attentions(m, ids)):
        depuis_fin = a[:, -1, :]                      # (heads, positions)
        ligne = {"couche": couche,
                 "vers_premier_token_moyenne": float(depuis_fin[:, 0].mean())}
        if position_mot is not None:
            ligne["vers_mot_max"] = float(depuis_fin[:, position_mot].max())
            ligne["tete_max"] = int(depuis_fin[:, position_mot].argmax())
        lignes.append(ligne)
    return {"texte": texte, "tokens": [tok.decode([i]) for i in ids],
            "position_mot": position_mot, "couches": lignes}


def sequence_repetee(generateur, tok, repeter=True):
    """A start token, L random tokens, then the same (or other) L tokens."""
    premiere = generateur.integers(1000, 20000, LONGUEUR).tolist()
    seconde = premiere if repeter else generateur.integers(1000, 20000, LONGUEUR).tolist()
    return [tok.eos_token_id] + premiere + seconde


def score_induction(a):
    """Mean weight from position i to position i - L + 1, over the second half."""
    positions = range(LONGUEUR + 1, 2 * LONGUEUR + 1)
    return torch.stack([a[:, i, i - LONGUEUR + 1] for i in positions], dim=1).mean(1)


def perte_seconde_moitie(m, ids):
    """Mean next-token prediction loss over the second half."""
    entree = torch.tensor([ids])
    with torch.no_grad():
        logits = m(entree).logits[0]
    logprob = logits[:-1].log_softmax(-1)
    cibles = entree[0, 1:]
    pertes = -logprob[torch.arange(len(cibles)), cibles]
    return float(pertes[LONGUEUR:].mean())


def neutraliser_tetes(m, tetes):
    """Attach hooks that zero the output of some heads before the c_proj projection."""
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
