"""E04: resampling intervals for three results of the book, with no new model computation.

Chapters 3, 4 and 5 of the book.
This program runs no model: it rereads existing outputs
(outputs/E05_validation_classificateur, outputs/E07_recherche) and the fabricated agreement table
of E03_calculs_guides.py, then asks what uncertainty goes with each
number.

Rule written before execution (23 September 2026):
- resampling with replacement of the observed units (annotation pairs,
  messages, questions), 2,000 draws, seed 0, interval of the 2.5 and
  97.5 percentiles;
- for comparisons, **paired** resampling: we draw messages
  or questions, and keep for each one the results of both methods;
- a difference is called established only if its interval excludes zero;
- for the classifier validation (E05), we also give the Wilson interval and the
  exact McNemar test, which looks only at messages whose outcome changed.

Usage: python experiences/E04_intervalles.py
Output: outputs/E04_intervalles/resultats.json
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
    """2.5 and 97.5 percentiles of a list of resampled values."""
    v = sorted(valeurs)
    return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]


def reechantillonner(unites, statistique, hasard):
    """Statistic recomputed on TIRAGES samples drawn with replacement from the units."""
    n = len(unites)
    return [statistique([unites[hasard.randrange(n)] for _ in range(n)]) for _ in range(TIRAGES)]


def wilson(succes, n, z=1.96):
    """Wilson interval for a proportion."""
    p = succes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    demi = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - demi, centre + demi]


def mcnemar_exact(gagnes, perdus):
    """Exact two-sided probability of an imbalance at least as large, if each change were a coin flip."""
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

    # 1. Kappa of the fabricated table [[40, 10], [5, 45]]: one hundred annotation pairs.
    table = [[40, 10], [5, 45]]
    paires = [(i, j) for i in range(2) for j in range(2) for _ in range(table[i][j])]
    resultats["kappa"] = {"table": table, "valeur": kappa(table)["kappa"],
                          "intervalle": intervalle(reechantillonner(paires, kappa_de_paires, hasard))}

    # 2. Classifier validation (E05): 24 messages, before and after balancing.
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

    # 3. E07: eight questions, ranks of the two methods.
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
