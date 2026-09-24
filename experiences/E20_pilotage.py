"""E20 : mettre la direction de pilotage à l'épreuve : témoins aléatoires, sens inverse, effets collatéraux.

Chapitre « Adapter et contrôler ». Le programme `E13_inspection.py` ajoute, à la
sortie du bloc 8 et à la dernière position, la direction normalisée
état(« The capital of France is ») − état(« The capital of Italy is »). La
probabilité de « ·Paris » monte. Trois questions restent ouvertes :
une direction quelconque de même norme ferait-elle autant ? le sens opposé
favorise-t-il « ·Rome » ? que devient le reste du comportement ?

Règle écrite avant l'exécution (23 septembre 2026) :
- texte cible : « France has its capital in » (non saturé dans inspection) ;
- intensités : −40, −20, 0, 20, 40, 80, 160 ;
- témoin : trente directions aléatoires unitaires (graine 0) à l'intensité 40 ;
  la direction est dite plus efficace que le hasard si sa hausse de
  P(·Paris) dépasse le 95e centile des hausses aléatoires ;
- sens inverse : aux intensités négatives, P(·Rome) doit augmenter ;
- effets collatéraux, sur trois textes sans rapport avec la France : pour
  chaque intensité, probabilité de la réponse attendue et divergence de
  Kullback-Leibler entre les distributions du prochain token avant et après ;
- dose tolérable : plus forte intensité positive pour laquelle la réponse
  attendue de chaque texte témoin garde au moins la moitié de sa probabilité
  initiale.

Usage : python experiences/E20_pilotage.py
Sortie : outputs/E20_pilotage/metadata.json
"""
import time

import numpy as np
import torch

from socle_experiences import model, snapshot, save

COUCHE = 8
CIBLE = "France has its capital in"
INTENSITES = [-40.0, -20.0, 0.0, 20.0, 40.0, 80.0, 160.0]
TEMOINS = [("The capital of Japan is", " Tokyo"), ("The capital of Spain is", " Madrid"),
           ("Water is made of hydrogen and", " oxygen")]
N_ALEATOIRES = 30


def etat(m, tok, texte):
    """Sortie du bloc COUCHE à la dernière position."""
    boite = {}
    def capter(module, entrees, sortie):
        boite["v"] = (sortie[0] if isinstance(sortie, tuple) else sortie)[0, -1].detach().clone()
    poignee = m.transformer.h[COUCHE].register_forward_hook(capter)
    try:
        with torch.no_grad():
            m(**tok(texte, return_tensors="pt"))
    finally:
        poignee.remove()
    return boite["v"]


def distribution(m, tok, texte, ajout=None):
    """Probabilités du prochain token, avec un vecteur ajouté à la dernière position si demandé."""
    def ajouter(module, entrees, sortie):
        h = (sortie[0] if isinstance(sortie, tuple) else sortie).clone()
        h[:, -1] += ajout
        return (h,) + tuple(sortie[1:]) if isinstance(sortie, tuple) else h
    poignee = m.transformer.h[COUCHE].register_forward_hook(ajouter) if ajout is not None else None
    try:
        with torch.no_grad():
            logits = m(**tok(texte, return_tensors="pt")).logits[0, -1]
    finally:
        if poignee is not None:
            poignee.remove()
    return logits.softmax(-1)


def kl(p, q):
    """Divergence KL(p ‖ q) en nats."""
    return float((p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum())


def main():
    debut = time.perf_counter()
    m, tok = model()
    paris, rome = tok.encode(" Paris")[0], tok.encode(" Rome")[0]
    direction = etat(m, tok, "The capital of France is") - etat(m, tok, "The capital of Italy is")
    norme_brute = float(direction.norm())
    direction = direction / direction.norm()

    base = distribution(m, tok, CIBLE)
    doses = []
    for alpha in INTENSITES:
        p = distribution(m, tok, CIBLE, alpha * direction)
        ligne = {"intensite": alpha, "p_paris": float(p[paris]), "p_rome": float(p[rome]),
                 "premier": tok.decode(int(p.argmax())), "temoins": []}
        for texte, attendu in TEMOINS:
            i = tok.encode(attendu)[0]
            p0, p1 = distribution(m, tok, texte), distribution(m, tok, texte, alpha * direction)
            ligne["temoins"].append({"texte": texte, "attendu": attendu, "p_avant": float(p0[i]),
                                     "p_apres": float(p1[i]), "kl": kl(p0, p1),
                                     "premier_apres": tok.decode(int(p1.argmax()))})
        doses.append(ligne)

    generateur = np.random.default_rng(0)
    hausses_aleatoires = []
    for _ in range(N_ALEATOIRES):
        u = torch.from_numpy(generateur.standard_normal(direction.shape[0]).astype(np.float32))
        u = u / u.norm()
        hausses_aleatoires.append(float(distribution(m, tok, CIBLE, 40.0 * u)[paris] - base[paris]))
    hausse_direction = next(d["p_paris"] for d in doses if d["intensite"] == 40.0) - float(base[paris])

    tolerables = [d["intensite"] for d in doses if d["intensite"] > 0
                  and all(t["p_apres"] >= 0.5 * t["p_avant"] for t in d["temoins"])]
    save("E20_pilotage", {
        "couche": COUCHE, "texte_cible": CIBLE, "norme_difference_brute": norme_brute,
        "doses": doses, "hausse_direction_40": hausse_direction,
        "hausses_aleatoires_40": hausses_aleatoires,
        "centile_95_aleatoire": float(np.percentile(hausses_aleatoires, 95)),
        "plus_efficace_que_hasard": bool(hausse_direction > np.percentile(hausses_aleatoires, 95)),
        "rome_monte_si_negatif": bool(all(d["p_rome"] > float(base[rome]) for d in doses if d["intensite"] < 0)),
        "dose_tolerable": max(tolerables) if tolerables else None,
    }, "La direction de pilotage fait-elle mieux qu'une direction quelconque, et à quel prix pour le reste ?",
        ["Trente directions aléatoires de même norme", "Intensités négatives", "Trois textes témoins sans rapport"],
        ["Une direction construite sur deux phrases, un site, un modèle",
         "Prochain token seulement ; aucune continuation longue évaluée",
         "Trois textes témoins ne mesurent pas toutes les régressions possibles"],
        debut, {"gpt2": snapshot("gpt2")[1]})


if __name__ == "__main__":
    main()
