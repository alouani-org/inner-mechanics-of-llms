"""E18 : remplacer la sortie d'une seule tête d'attention : sélection, transfert et comparaison publiée.

Chapitre « Intervenir pour expliquer ». La tâche est celle de E16 : dans
« When A and B went to the store, A gave a book to », le nom attendu est B
(condition propre) ; dans la condition corrompue, le second sujet devient B et
le nom attendu devient A. Métrique : logit(B) − logit(A) à la dernière position.

Le site est plus fin que dans E16 : au lieu de remplacer l'état complet d'une
position, on remplace la contribution d'une seule tête (ses 64 composantes,
avant la projection de sortie de l'attention), à la dernière position.

Règle écrite avant l'exécution (23 septembre 2026) :
- découverte sur quatre paires de noms, gabarit du magasin : carte des
  144 têtes (12 couches × 12 têtes), restauration normalisée par le fossé ;
- sélection : les trois têtes de plus forte restauration moyenne ;
- validation sur huit autres paires, dans deux gabarits : celui du magasin
  et un gabarit nouveau (« Then, A and B had a long argument. Afterwards, A
  said to ») ;
- mesures de validation : restauration de chaque tête retenue, restauration
  conjointe des trois, restauration conjointe de trente triplets tirés au
  hasard parmi les autres têtes (graine 0), et intervention inverse
  conjointe (sorties corrompues des trois têtes dans le calcul propre) ;
- critère de transfert : la restauration conjointe moyenne des trois têtes
  dépasse, dans chaque gabarit, le 95e centile des triplets aléatoires ;
- comparaison externe, sans rôle dans la sélection : les mêmes mesures pour
  les têtes 9.6, 9.9 et 10.0, nommées « Name Mover Heads » par Wang et al.
  (2022) avec une autre méthode (path patching) et une autre corruption.

Usage : python experiences/E18_tetes.py
Sortie : outputs/E18_tetes/metadata.json
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
    """Entrée de la projection de sortie de chaque couche : les 12 têtes concaténées, dernière position."""
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
    """logit(io) − logit(s) ; remplacements = {(couche, tête): vecteur de 64 composantes}."""
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
    """Textes propre et corrompu, identifiants des noms, contributions et écarts de référence."""
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
    """Part du fossé propre − corrompu retrouvée en plaçant les têtes propres dans le calcul corrompu."""
    valeurs = {(c, h): morceau(cas["c_propre"], c, h) for c, h in tetes}
    apres = ecart_avec_tetes(m, cas["corrompu"], cas["io"], cas["s"], valeurs)
    return (apres - cas["ecart_corrompu"]) / (cas["ecart_propre"] - cas["ecart_corrompu"])


def perte_inverse(m, cas, tetes):
    """Part de l'écart propre perdue en plaçant les têtes corrompues dans le calcul propre."""
    valeurs = {(c, h): morceau(cas["c_corrompu"], c, h) for c, h in tetes}
    apres = ecart_avec_tetes(m, cas["propre"], cas["io"], cas["s"], valeurs)
    return (cas["ecart_propre"] - apres) / (cas["ecart_propre"] - cas["ecart_corrompu"])


def mesurer(m, cas_liste, tetes, triplets):
    """Mesures de validation pour un ensemble de têtes sur une liste de cas."""
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
