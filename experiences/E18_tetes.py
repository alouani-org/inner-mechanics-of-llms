"""E18: replace the output of a single attention head: selection, transfer and published comparison.

Chapter 9. The task is that of E16: in
"When A and B went to the store, A gave a book to", the expected name is B
(clean condition); in the corrupted condition, the second subject becomes B and
the expected name becomes A. Metric: logit(B) − logit(A) at the last position.

The site is finer than in E16: instead of replacing the full state of a
position, we replace the contribution of a single head (its 64 components,
before the attention output projection), at the last position.

Rule written before execution (23 September 2026):
- discovery on four name pairs, store template: map of the
  144 heads (12 layers × 12 heads), restoration normalized by the gap;
- selection: the three heads with the highest mean restoration;
- validation on eight other pairs, in two templates: the store one
  and a new template ("Then, A and B had a long argument. Afterwards, A
  said to");
- validation measurements: restoration of each selected head, joint restoration
  of the three, joint restoration of thirty triplets drawn at
  random among the other heads (seed 0), and joint reverse
  intervention (corrupted outputs of the three heads in the clean run);
- transfer criterion: the mean joint restoration of the three heads
  exceeds, in each template, the 95th percentile of the random triplets;
- external comparison, with no role in the selection: the same measurements for
  heads 9.6, 9.9 and 10.0, named "Name Mover Heads" by Wang et al.
  (2022) with another method (path patching) and another corruption.

Usage: python experiences/E18_tetes.py
Output: outputs/E18_tetes/metadata.json
"""
import time

import numpy as np
import torch

from socle_experiences import model, snapshot, save

MAGASIN = "When {a} and {b} went to the store, {s} gave a book to"
DISPUTE = "Then, {a} and {b} had a long argument. Afterwards, {s} said to"
DECOUVERTE = [("John", "Mary"), ("Paul", "Alice"), ("James", "Sarah"), ("David", "Anna")]
VALIDATION = [("Peter", "Lucy"), ("Robert", "Jane"), ("Michael", "Emma"), ("Thomas", "Laura"),
              ("Mark", "Kate"), ("Daniel", "Rose"), ("George", "Helen"), ("Henry", "Grace")]
TETES_PUBLIEES = [(9, 6), (9, 9), (10, 0)]
N_TRIPLETS = 30


def un_token(tok, nom):
    ids = tok.encode(" " + nom)
    assert len(ids) == 1, nom
    return ids[0]


def contributions_tetes(m, ids):
    """Input of each layer's output projection: the 12 concatenated heads, last position."""
    boites = {}
    poignees = []
    for couche, bloc in enumerate(m.transformer.h):
        def capter(module, entrees, couche=couche):
            boites[couche] = entrees[0][0, -1].detach().clone()
        poignees.append(bloc.attn.c_proj.register_forward_pre_hook(capter))
    try:
        with torch.no_grad():
            m(torch.tensor([ids]))
    finally:
        for p in poignees:
            p.remove()
    return boites


def ecart_avec_tetes(m, ids, io, s, remplacements=None):
    """logit(io) − logit(s); remplacements = {(layer, head): vector of 64 components}."""
    d = m.config.n_embd // m.config.n_head
    poignees = []
    par_couche = {}
    for (couche, tete), valeur in (remplacements or {}).items():
        par_couche.setdefault(couche, []).append((tete, valeur))
    for couche, liste in par_couche.items():
        def remplacer_tete(module, entrees, liste=liste):
            x = entrees[0].clone()
            for tete, valeur in liste:
                x[0, -1, tete * d:(tete + 1) * d] = valeur
            return (x,) + tuple(entrees[1:])
        poignees.append(m.transformer.h[couche].attn.c_proj.register_forward_pre_hook(remplacer_tete))
    try:
        with torch.no_grad():
            logits = m(torch.tensor([ids])).logits[0, -1]
    finally:
        for p in poignees:
            p.remove()
    return float(logits[io] - logits[s])


def preparer(m, tok, gabarit, a, b):
    """Clean and corrupted texts, name ids, contributions and reference differences."""
    propre = tok.encode(gabarit.format(a=a, b=b, s=a))
    corrompu = tok.encode(gabarit.format(a=a, b=b, s=b))
    assert len(propre) == len(corrompu)
    io, s = un_token(tok, b), un_token(tok, a)
    e_p, e_c = ecart_avec_tetes(m, propre, io, s), ecart_avec_tetes(m, corrompu, io, s)
    return {"paire": f"{a}/{b}", "propre": propre, "corrompu": corrompu, "io": io, "s": s,
            "ecart_propre": e_p, "ecart_corrompu": e_c,
            "c_propre": contributions_tetes(m, propre), "c_corrompu": contributions_tetes(m, corrompu)}


def morceau(contributions, couche, tete, d=64):
    return contributions[couche][tete * d:(tete + 1) * d]


def restauration(m, cas, tetes):
    """Share of the clean − corrupted gap recovered by placing the clean heads in the corrupted run."""
    valeurs = {(c, h): morceau(cas["c_propre"], c, h) for c, h in tetes}
    apres = ecart_avec_tetes(m, cas["corrompu"], cas["io"], cas["s"], valeurs)
    return (apres - cas["ecart_corrompu"]) / (cas["ecart_propre"] - cas["ecart_corrompu"])


def perte_inverse(m, cas, tetes):
    """Share of the clean difference lost by placing the corrupted heads in the clean run."""
    valeurs = {(c, h): morceau(cas["c_corrompu"], c, h) for c, h in tetes}
    apres = ecart_avec_tetes(m, cas["propre"], cas["io"], cas["s"], valeurs)
    return (cas["ecart_propre"] - apres) / (cas["ecart_propre"] - cas["ecart_corrompu"])


def mesurer(m, cas_liste, tetes, triplets):
    """Validation measurements for a set of heads on a list of cases."""
    individuelles = {f"{c}.{h}": [restauration(m, cas, [(c, h)]) for cas in cas_liste] for c, h in tetes}
    conjointe = [restauration(m, cas, tetes) for cas in cas_liste]
    inverse = [perte_inverse(m, cas, tetes) for cas in cas_liste]
    hasard = [float(np.mean([restauration(m, cas, t) for cas in cas_liste])) for t in triplets]
    return {"individuelles": individuelles, "conjointe": conjointe, "conjointe_moyenne": float(np.mean(conjointe)),
            "inverse": inverse, "inverse_moyenne": float(np.mean(inverse)),
            "hasard_moyennes": hasard, "hasard_centile_95": float(np.percentile(hasard, 95)),
            "depasse_centile_95": bool(np.mean(conjointe) > np.percentile(hasard, 95))}


def main():
    debut = time.perf_counter()
    m, tok = model()
    n_couches, n_tetes = m.config.n_layer, m.config.n_head

    decouverte = [preparer(m, tok, MAGASIN, a, b) for a, b in DECOUVERTE]
    carte = np.zeros((n_couches, n_tetes))
    for c in range(n_couches):
        for h in range(n_tetes):
            carte[c, h] = np.mean([restauration(m, cas, [(c, h)]) for cas in decouverte])
    ordre = np.dstack(np.unravel_index(np.argsort(-carte, axis=None), carte.shape))[0]
    retenues = [(int(c), int(h)) for c, h in ordre[:3]]

    generateur = np.random.default_rng(0)
    autres = [(c, h) for c in range(n_couches) for h in range(n_tetes) if (c, h) not in retenues]
    triplets = [[autres[i] for i in generateur.choice(len(autres), 3, replace=False)] for _ in range(N_TRIPLETS)]

    validation = {}
    for nom, gabarit in [("magasin", MAGASIN), ("dispute", DISPUTE)]:
        cas_liste = [preparer(m, tok, gabarit, a, b) for a, b in VALIDATION]
        validation[nom] = {
            "ecarts": [{"paire": c["paire"], "propre": c["ecart_propre"], "corrompu": c["ecart_corrompu"]}
                       for c in cas_liste],
            "retenues": mesurer(m, cas_liste, retenues, triplets),
            "publiees": mesurer(m, cas_liste, TETES_PUBLIEES, triplets),
        }

    save("E18_tetes", {
        "gabarits": {"magasin": MAGASIN, "dispute": DISPUTE},
        "paires_decouverte": [f"{a}/{b}" for a, b in DECOUVERTE],
        "paires_validation": [f"{a}/{b}" for a, b in VALIDATION],
        "ecarts_decouverte": [{"paire": c["paire"], "propre": c["ecart_propre"], "corrompu": c["ecart_corrompu"]}
                              for c in decouverte],
        "carte_decouverte": carte.tolist(),
        "dix_premieres": [{"tete": f"{int(c)}.{int(h)}", "restauration": float(carte[c, h])} for c, h in ordre[:10]],
        "retenues": [f"{c}.{h}" for c, h in retenues],
        "publiees": [f"{c}.{h}" for c, h in TETES_PUBLIEES],
        "triplets_aleatoires": [[f"{c}.{h}" for c, h in t] for t in triplets],
        "validation": validation,
    }, "Quelles têtes, remplacées seules à la dernière position, restaurent la préférence propre, et ce choix se transfère-t-il ?",
        ["Sélection sur quatre paires, validation sur huit autres et dans un second gabarit",
         "Trente triplets aléatoires de têtes, même opération", "Intervention inverse conjointe",
         "Comparaison externe avec trois têtes publiées, hors sélection"],
        ["Deux gabarits anglais fabriqués, douze paires de noms, un modèle",
         "Contribution d'une tête à la dernière position seulement ; effets en aval recalculés",
         "Corruption ABB, différente de celle de Wang et al. (2022) : comparaison, pas réplication"],
        debut, {"gpt2": snapshot("gpt2")[1]})


if __name__ == "__main__":
    main()
