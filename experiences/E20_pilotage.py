"""E20: put the steering direction to the test: random controls, reverse direction, side effects.

Chapter 10. The program `E13_inspection.py` adds, at the
output of block 8 and at the last position, the normalized direction
state("The capital of France is") − state("The capital of Italy is"). The
probability of "·Paris" rises. Three questions remain open:
would any direction of the same norm do as much? does the opposite direction
favor "·Rome"? what happens to the rest of the behavior?

Rule written before execution (23 September 2026):
- target text: "France has its capital in" (not saturated in inspection);
- intensities: −40, −20, 0, 20, 40, 80, 160;
- control: thirty random unit directions (seed 0) at intensity 40;
  the direction is called more effective than chance if its increase in
  P(·Paris) exceeds the 95th percentile of the random increases;
- reverse direction: at negative intensities, P(·Rome) must increase;
- side effects, on three texts unrelated to France: for
  each intensity, probability of the expected answer and
  Kullback-Leibler divergence between the next-token distributions before and after;
- tolerable dose: highest positive intensity for which the expected
  answer of each control text keeps at least half of its initial
  probability.

Usage: python experiences/E20_pilotage.py
Output: outputs/E20_pilotage/metadata.json
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
    """Output of block COUCHE at the last position."""
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
    """Next-token probabilities, with a vector added at the last position if requested."""
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
    """KL(p ‖ q) divergence in nats."""
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
