"""E04 : intervalles de rééchantillonnage pour trois résultats du livre, sans nouveau calcul de modèle.

Chapitres « Définir ce qu'on mesure », « Classer » et « Trouver les bons
documents ». Ce programme ne fait tourner aucun modèle : il relit des sorties
existantes (outputs/E05_validation_classificateur, outputs/E07_recherche) et la table d'accord fabriquée
de E03_calculs_guides.py, puis demande quelle incertitude accompagne chaque
nombre.

Règle écrite avant l'exécution (23 septembre 2026) :
- rééchantillonnage avec remise des unités observées (paires d'annotations,
  messages, questions), 2 000 tirages, graine 0, intervalle des centiles 2,5 et
  97,5 ;
- pour les comparaisons, rééchantillonnage **apparié** : on tire des messages
  ou des questions, et l'on garde pour chacun les résultats des deux méthodes ;
- une différence est dite établie seulement si son intervalle exclut zéro ;
- pour la validation du classificateur (E05), on donne aussi l'intervalle de Wilson et le test
  exact de McNemar, qui ne regarde que les messages dont le sort a changé.

Usage : python experiences/E04_intervalles.py
Sortie : outputs/E04_intervalles/resultats.json
"""
import json
import math
import random
from pathlib import Path

from E03_calculs_guides import kappa

RACINE = Path(__file__).resolve().parents[1]
TIRAGES = 2000
GRAINE = 0


def intervalle(valeurs):
    """Centiles 2,5 et 97,5 d'une liste de valeurs rééchantillonnées."""
    v = sorted(valeurs)
    return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]


def reechantillonner(unites, statistique, hasard):
    """Statistique recalculée sur TIRAGES échantillons tirés avec remise parmi les unités."""
    n = len(unites)
    return [statistique([unites[hasard.randrange(n)] for _ in range(n)]) for _ in range(TIRAGES)]


def wilson(succes, n, z=1.96):
    """Intervalle de Wilson pour une proportion."""
    p = succes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    demi = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - demi, centre + demi]


def mcnemar_exact(gagnes, perdus):
    """Probabilité bilatérale exacte d'un déséquilibre au moins aussi grand, si chaque changement était à pile ou face."""
    n = gagnes + perdus
    k = min(gagnes, perdus)
    queue = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * queue)


def kappa_de_paires(paires):
    table = [[0, 0], [0, 0]]
    for a, b in paires:
        table[a][b] += 1
    return kappa(table)["kappa"]


def main():
    hasard = random.Random(GRAINE)
    resultats = {"tirages": TIRAGES, "graine": GRAINE}

    # 1. Kappa de la table fabriquée [[40, 10], [5, 45]] : cent paires d'annotations.
    table = [[40, 10], [5, 45]]
    paires = [(i, j) for i in range(2) for j in range(2) for _ in range(table[i][j])]
    resultats["kappa"] = {"table": table, "valeur": kappa(table)["kappa"],
                          "intervalle": intervalle(reechantillonner(paires, kappa_de_paires, hasard))}

    # 2. Validation du classificateur (E05) : 24 messages, avant et après équilibrage.
    v = json.loads((RACINE / "outputs/E05_validation_classificateur/metadata.json").read_text(encoding="utf-8"))["results"]
    avant = [int(l["reference"] == l["prediction"]) for l in v["avant_equilibrage"]["lignes"]]
    apres = [int(l["reference"] == l["prediction"]) for l in v["apres_equilibrage"]["lignes"]]
    messages = list(zip(avant, apres))
    gagnes = sum(1 for a, b in messages if a == 0 and b == 1)
    perdus = sum(1 for a, b in messages if a == 1 and b == 0)
    diff = reechantillonner(messages, lambda e: sum(b - a for a, b in e) / len(e), hasard)
    resultats["E1_validation"] = {
        "n": len(messages), "exactitude_avant": sum(avant) / len(avant), "exactitude_apres": sum(apres) / len(apres),
        "intervalle_apres_reechantillonnage": intervalle(reechantillonner(apres, lambda e: sum(e) / len(e), hasard)),
        "intervalle_apres_wilson": wilson(sum(apres), len(apres)),
        "difference": sum(apres) / len(apres) - sum(avant) / len(avant), "intervalle_difference": intervalle(diff),
        "messages_repares": gagnes, "messages_casses": perdus, "mcnemar_p": mcnemar_exact(gagnes, perdus)}

    # 3. E07 : huit questions, rangs des deux méthodes.
    e2 = json.loads((RACINE / "outputs/E07_recherche/metadata.json").read_text(encoding="utf-8"))["results"]
    questions = list(zip(e2["lexical"]["ranks"], e2["dense"]["ranks"]))
    mrr = lambda e: sum(1 / a - 1 / b for a, b in e) / len(e)
    r1 = lambda e: sum(int(a == 1) - int(b == 1) for a, b in e) / len(e)
    resultats["E2"] = {"n": len(questions), "difference_mrr_lexical_moins_dense": mrr(questions),
                       "intervalle_mrr": intervalle(reechantillonner(questions, mrr, hasard)),
                       "difference_rappel1_lexical_moins_dense": r1(questions),
                       "intervalle_rappel1": intervalle(reechantillonner(questions, r1, hasard))}

    for cle, bloc, champ in [("kappa", resultats["kappa"], "intervalle"),
                             ("E1", resultats["E1_validation"], "intervalle_difference"),
                             ("E2_mrr", resultats["E2"], "intervalle_mrr"), ("E2_r1", resultats["E2"], "intervalle_rappel1")]:
        bas, haut = bloc[champ]
        resultats.setdefault("zero_exclu", {})[cle] = bool(bas > 0 or haut < 0)

    sortie = RACINE / "outputs/E04_intervalles"
    sortie.mkdir(parents=True, exist_ok=True)
    (sortie / "resultats.json").write_text(json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
