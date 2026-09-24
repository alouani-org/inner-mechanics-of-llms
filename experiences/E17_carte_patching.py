"""E17 : carte de restauration couche × position, puis contrôles aux sites retenus.

Chapitre « Intervenir pour expliquer ». Paires propre/corrompue qui ne diffèrent
que par le pays interrogé (« The capital of France is » / « … of Italy is »),
précédées du même exemple amorcé. Métrique : différence de logits entre la
capitale du pays propre et celle du pays corrompu, à la dernière position.

Règle écrite avant l'exécution (23 septembre 2026) :
- découverte sur trois paires : carte complète couche × position ;
- site « sujet » = couche où la restauration moyenne à la position du pays est
  la plus forte parmi les couches 1 à 9 (on écarte la couche 0, simple lecture
  de l'entrée, et les dernières couches) ;
- site « lecture » = couche la plus basse où la restauration moyenne à la
  dernière position dépasse 0,5 ;
- validation sur d'autres paires : restauration aux deux sites, auto-patch
  (identité attendue), intervention inverse et vingt perturbations aléatoires
  de même norme que l'écart propre − corrompu, au même site.

Usage : python experiences/E17_carte_patching.py
Sortie : outputs/E17_carte_patching/metadata.json
"""
import time

import numpy as np
import torch

from socle_experiences import model, snapshot, save

AMORCE = "The capital of Germany is Berlin. The capital of {} is"
PAIRES_CANDIDATES = [("France", "Paris", "Italy", "Rome"), ("Spain", "Madrid", "Japan", "Tokyo"),
                     ("Russia", "Moscow", "Egypt", "Cairo"), ("Greece", "Athens", "Poland", "Warsaw"),
                     ("Italy", "Rome", "France", "Paris"), ("Japan", "Tokyo", "Spain", "Madrid"),
                     ("Egypt", "Cairo", "Russia", "Moscow"), ("Poland", "Warsaw", "Greece", "Athens"),
                     ("Norway", "Oslo", "Portugal", "Lisbon"), ("Portugal", "Lisbon", "Norway", "Oslo")]
N_DECOUVERTE = 3
N_PERTURBATIONS = 20


def un_token(tok, mot):
    ids = tok.encode(" " + mot)
    return ids[0] if len(ids) == 1 else None


def paires_valides(tok):
    """Garder les paires dont pays et capitales sont un seul token : positions alignées."""
    valides = []
    for pays, capitale, autre_pays, autre_capitale in PAIRES_CANDIDATES:
        ids = [un_token(tok, x) for x in (pays, capitale, autre_pays, autre_capitale)]
        if None not in ids:
            valides.append({"propre": AMORCE.format(pays), "corrompu": AMORCE.format(autre_pays),
                            "cible": ids[1], "concurrente": ids[3],
                            "nom": f"{pays}/{autre_pays}"})
    return valides


def ecart(m, ids, cible, concurrente, crochet=None, couche=None):
    """Différence de logits cible − concurrente à la dernière position, avec un crochet éventuel."""
    poignee = m.transformer.h[couche].register_forward_hook(crochet) if crochet else None
    try:
        with torch.no_grad():
            logits = m(torch.tensor([ids])).logits[0, -1]
    finally:
        if poignee is not None:
            poignee.remove()
    return float(logits[cible] - logits[concurrente])


def remplacer(position, valeur):
    """Crochet qui remplace, à une position, la sortie d'un bloc par une valeur donnée."""
    def crochet(module, entrees, sortie):
        h = (sortie[0] if isinstance(sortie, tuple) else sortie).clone()
        h[0, position] = valeur
        return (h,) + tuple(sortie[1:]) if isinstance(sortie, tuple) else h
    return crochet


def capturer(m, ids, couche):
    """Sortie brute du bloc demandé (avant la normalisation finale), toutes positions."""
    boite = {}
    def crochet(module, entrees, sortie):
        boite["h"] = (sortie[0] if isinstance(sortie, tuple) else sortie)[0].detach().clone()
    poignee = m.transformer.h[couche].register_forward_hook(crochet)
    try:
        with torch.no_grad():
            m(torch.tensor([ids]))
    finally:
        poignee.remove()
    return boite["h"]


def carte(m, tok, paire):
    """Restauration normalisée pour chaque (couche, position)."""
    propre, corrompu = tok.encode(paire["propre"]), tok.encode(paire["corrompu"])
    assert len(propre) == len(corrompu)
    e_propre = ecart(m, propre, paire["cible"], paire["concurrente"])
    e_corrompu = ecart(m, corrompu, paire["cible"], paire["concurrente"])
    fosse = e_propre - e_corrompu
    lignes = []
    for couche in range(m.config.n_layer):
        h_propre = capturer(m, propre, couche)
        lignes.append([(ecart(m, corrompu, paire["cible"], paire["concurrente"],
                              remplacer(p, h_propre[p]), couche) - e_corrompu) / fosse
                       for p in range(len(propre))])
    position_pays = next(i for i, (a, b) in enumerate(zip(propre, corrompu)) if a != b)
    return {"nom": paire["nom"], "tokens": [tok.decode([i]) for i in propre],
            "position_pays": position_pays, "ecart_propre": e_propre, "ecart_corrompu": e_corrompu,
            "restauration": lignes}


def controles(m, tok, paire, couche, position, generateur):
    """Au site donné : patch propre, auto-patch, inverse et perturbations de même norme."""
    propre, corrompu = tok.encode(paire["propre"]), tok.encode(paire["corrompu"])
    c, k = paire["cible"], paire["concurrente"]
    h_p, h_c = capturer(m, propre, couche), capturer(m, corrompu, couche)
    e_p, e_c = ecart(m, propre, c, k), ecart(m, corrompu, c, k)
    delta = h_p[position] - h_c[position]
    effet = ecart(m, corrompu, c, k, remplacer(position, h_p[position]), couche) - e_c
    identite = ecart(m, corrompu, c, k, remplacer(position, h_c[position]), couche) - e_c
    inverse = ecart(m, propre, c, k, remplacer(position, h_c[position]), couche) - e_p
    perturbations = []
    for _ in range(N_PERTURBATIONS):
        bruit = torch.from_numpy(generateur.standard_normal(delta.shape[0]).astype(np.float32))
        bruit = bruit / bruit.norm() * delta.norm()
        perturbations.append(ecart(m, corrompu, c, k, remplacer(position, h_c[position] + bruit), couche) - e_c)
    return {"paire": paire["nom"], "couche": couche, "position": position, "fosse": e_p - e_c,
            "effet": effet, "identite": identite, "inverse": inverse,
            "perturbations": perturbations}


def main():
    debut = time.perf_counter()
    m, tok = model()
    generateur = np.random.default_rng(42)
    paires = paires_valides(tok)
    decouverte, validation = paires[:N_DECOUVERTE], paires[N_DECOUVERTE:]
    cartes = [carte(m, tok, p) for p in decouverte]

    position_pays = cartes[0]["position_pays"]
    assert all(c["position_pays"] == position_pays for c in cartes)
    derniere = len(cartes[0]["tokens"]) - 1
    moyenne = np.mean([np.array(c["restauration"]) for c in cartes], axis=0)
    profil_pays = moyenne[:, position_pays].tolist()
    profil_lecture = moyenne[:, derniere].tolist()
    couche_sujet = int(1 + np.argmax(moyenne[1:10, position_pays]))
    couche_lecture = int(next(k for k, v in enumerate(profil_lecture) if v > 0.5))

    verifications = []
    for paire in validation:
        verifications.append(controles(m, tok, paire, couche_sujet, position_pays, generateur))
        verifications.append(controles(m, tok, paire, couche_lecture, derniere, generateur))

    save("E17_carte_patching", {
        "paires_decouverte": [p["nom"] for p in decouverte], "paires_validation": [p["nom"] for p in validation],
        "cartes_decouverte": cartes, "profil_pays": profil_pays, "profil_lecture": profil_lecture,
        "couche_sujet": couche_sujet, "couche_lecture": couche_lecture,
        "position_pays": position_pays, "position_lecture": derniere,
        "validation": verifications,
    }, "Où le remplacement d'un état restaure-t-il la préférence propre, et ces sites passent-ils leurs contrôles ?",
        ["Paires alignées token à token ; mêmes positions", "Sites choisis sur la découverte, contrôles sur la validation",
         "Auto-patch, intervention inverse et perturbations de même norme au même site",
         "Cellule tautologique (dernier bloc, dernière position) signalée"],
        ["Six paires de pays, un gabarit amorcé anglais, un modèle",
         "État complet d'une position : pas de tête ni de neurone isolé",
         "Restauration normalisée par le fossé de chaque paire ; effet local, pas circuit complet"],
        debut, {"gpt2": snapshot("gpt2")[1]})


if __name__ == "__main__":
    main()
