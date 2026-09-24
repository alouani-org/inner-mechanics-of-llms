"""E03 : calculs didactiques sur données fabriquées : aucun résultat de modèle."""
from pathlib import Path
import hashlib
import json
import math
import ast

ROOT = Path(__file__).resolve().parents[1]


def softmax(scores):
    maximum = max(scores)
    poids = [math.exp(score - maximum) for score in scores]
    total = sum(poids)
    return [poids_i / total for poids_i in poids]


def melanger(coefficients, valeurs):
    return [sum(a * v[j] for a, v in zip(coefficients, valeurs))
            for j in range(len(valeurs[0]))]


def metriques(vp, fp, fn, vn):
    precision = vp / (vp + fp)
    rappel = vp / (vp + fn)
    return {"precision": precision, "rappel": rappel,
            "f1": 2 * precision * rappel / (precision + rappel),
            "exactitude": (vp + vn) / (vp + fp + fn + vn)}


def correction_acc(observe, tpr, fpr):
    if abs(tpr - fpr) < 1e-12:
        raise ValueError("Instrument non identifiable : TPR = FPR")
    return (observe - fpr) / (tpr - fpr)


def cosinus(a, b):
    if len(a) != len(b):
        raise ValueError("Dimensions différentes")
    norme_a = math.sqrt(sum(x*x for x in a))
    norme_b = math.sqrt(sum(x*x for x in b))
    if norme_a == 0 or norme_b == 0:
        raise ValueError("Cosinus non défini pour le vecteur nul")
    return sum(x*y for x,y in zip(a,b)) / (norme_a * norme_b)


def attention(requete, cles, valeurs):
    """Une tête d'attention pour une position : scores, poids softmax, mélange des valeurs."""
    d = len(requete)
    scores = [sum(q * k for q, k in zip(requete, cle)) / math.sqrt(d) for cle in cles]
    poids = softmax(scores)
    return scores, poids, melanger(poids, valeurs)


def kappa(table):
    """Kappa de Cohen pour deux annotateurs ; table[i][j] = effectif (annotateur 1 = i, annotateur 2 = j)."""
    total = sum(sum(ligne) for ligne in table)
    observe = sum(table[i][i] for i in range(len(table))) / total
    marges_1 = [sum(ligne) / total for ligne in table]
    marges_2 = [sum(table[i][j] for i in range(len(table))) / total for j in range(len(table))]
    attendu = sum(a * b for a, b in zip(marges_1, marges_2))
    return {"accord_observe": observe, "accord_attendu": attendu,
            "kappa": (observe - attendu) / (1 - attendu)}


def simuler_acc(prevalence, tpr, fpr, n_positifs, n_negatifs, n_corpus, tirages, graine):
    """Variabilité de l'ACC quand TPR et FPR sont estimés sur une petite calibration (simulation)."""
    import random
    hasard = random.Random(graine)
    estimations = []
    for _ in range(tirages):
        tpr_estime = sum(hasard.random() < tpr for _ in range(n_positifs)) / n_positifs
        fpr_estime = sum(hasard.random() < fpr for _ in range(n_negatifs)) / n_negatifs
        signales = sum(hasard.random() < (tpr if hasard.random() < prevalence else fpr)
                       for _ in range(n_corpus))
        if abs(tpr_estime - fpr_estime) < 1e-12:
            continue
        estimations.append(correction_acc(signales / n_corpus, tpr_estime, fpr_estime))
    estimations.sort()
    return {"prevalence": prevalence, "tpr": tpr, "fpr": fpr, "separation": tpr - fpr,
            "n_positifs": n_positifs, "n_negatifs": n_negatifs, "n_corpus": n_corpus,
            "tirages_valides": len(estimations),
            "quantile_2_5": estimations[int(0.025 * len(estimations))],
            "mediane": estimations[len(estimations) // 2],
            "quantile_97_5": estimations[int(0.975 * len(estimations)) - 1]}


def cout_scenario(n_documents, couverture, erreur_acceptes, cout_revue, cout_erreur):
    """Coût d'une politique d'abstention : revue des refus + erreurs acceptées (unités arbitraires)."""
    acceptes = n_documents * couverture
    refuses = n_documents - acceptes
    erreurs = acceptes * erreur_acceptes
    return {"revues": refuses, "erreurs_acceptees": erreurs,
            "cout_total": refuses * cout_revue + erreurs * cout_erreur}


def tfidf(documents):
    """TF-IDF lissé comme scikit-learn : idf = ln((1 + n) / (1 + df)) + 1, puis norme L2 par document."""
    vocabulaire = sorted({mot for doc in documents for mot in doc.split()})
    n = len(documents)
    df = {mot: sum(mot in doc.split() for doc in documents) for mot in vocabulaire}
    idf = {mot: math.log((1 + n) / (1 + df[mot])) + 1 for mot in vocabulaire}
    vecteurs = []
    for doc in documents:
        mots = doc.split()
        brut = [mots.count(mot) * idf[mot] for mot in vocabulaire]
        norme = math.sqrt(sum(x * x for x in brut))
        vecteurs.append([x / norme for x in brut])
    return vocabulaire, idf, vecteurs


def probabilite_logistique(poids, biais, vecteur):
    """Régression logistique : score linéaire, puis sigmoïde."""
    score = biais + sum(w * x for w, x in zip(poids, vecteur))
    return score, 1 / (1 + math.exp(-score))


def main():
    libre = softmax([2, 1, 0])
    contraint = softmax([2, -math.inf, 0])
    # Les propriétés contrôlées sont indépendantes des arrondis imprimés.
    assert abs(sum(libre) - 1) < 1e-12
    assert all(abs(a-b) < 1e-12 for a,b in zip(libre, softmax([12,11,10])))
    assert contraint[1] == 0
    assert abs(libre[0]/libre[2] - contraint[0]/contraint[2]) < 1e-12
    coefficients = [.75, .25]
    valeurs = [[2, 0], [0, 4]]
    melange = melanger(coefficients, valeurs)
    assert melange == [1.5, 1.0]
    m = metriques(6, 2, 4, 88)
    assert abs(m['f1'] - 2/3) < 1e-12
    assert abs(correction_acc(.30, .80, .10) - 2/7) < 1e-12
    try:
        correction_acc(.3, .5, .5)
    except ValueError:
        pass
    else:
        raise AssertionError("Le cas non identifiable doit être refusé")
    resultats = {'statut': 'données fabriquées, calculs didactiques',
                 'logits': [2,1,0], 'softmax':libre, 'masque':contraint,
                 'coefficients':coefficients,'valeurs':valeurs,'melange':melange,
                 'confusion':{'vp':6,'fp':2,'fn':4,'vn':88},'metriques':m,
                 'acc':{'observe':.30,'tpr':.80,'fpr':.10,
                        'corrige':correction_acc(.30,.80,.10)},
                 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    resultats['geometrie'] = {
        'a':[1,1,0], 'b':[2,2,0], 'c':[1,0,0],
        'cosinus_ab':cosinus([1,1,0],[2,2,0]),
        'cosinus_ac':cosinus([1,1,0],[1,0,0]),
        'u':[1,0,1], 'v':[1,0,-1],
        'cosinus_uv':cosinus([1,0,1],[1,0,-1]),
        'projection_u':[1,0], 'projection_v':[1,0]}
    assert abs(resultats['geometrie']['cosinus_ab']-1)<1e-12
    assert resultats['geometrie']['cosinus_uv']==0
    assert resultats['geometrie']['projection_u']==resultats['geometrie']['projection_v']
    # Attention d'une position vers trois sources ; vecteurs fabriqués de dimension 2.
    requete = [1.0, 0.0]
    cles = [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
    valeurs_att = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    scores_att, poids_att, sortie_att = attention(requete, cles, valeurs_att)
    assert abs(sum(poids_att) - 1) < 1e-12
    resultats['attention'] = {'requete': requete, 'cles': cles, 'valeurs': valeurs_att,
                              'scores': scores_att, 'poids': poids_att, 'sortie': sortie_att}
    # Deux annotateurs, deux catégories : 100 textes fabriqués.
    table = [[40, 10], [5, 45]]
    k = kappa(table)
    assert abs(k['accord_observe'] - 0.85) < 1e-12
    resultats['kappa'] = {'table': table, **k}
    # Même proportion réelle, deux instruments de séparation différente.
    resultats['simulation_acc'] = [
        simuler_acc(0.20, 0.85, 0.05, 40, 80, 2000, 2000, 42),
        simuler_acc(0.20, 0.45, 0.10, 40, 80, 2000, 2000, 42)]
    # Deux politiques fabriquées, quatre hypothèses de coût d'une erreur acceptée.
    politiques = {'large': (0.90, 0.10), 'prudente': (0.60, 0.02)}
    resultats['scenario_cout'] = {
        'n_documents': 1000, 'cout_revue': 1, 'politiques': politiques,
        'lignes': [{'cout_erreur': c, **{nom: cout_scenario(1000, cv, er, 1, c)['cout_total']
                                          for nom, (cv, er) in politiques.items()}}
                   for c in [2, 5, 10, 20]]}
    # TF-IDF à la main sur trois messages fabriqués, puis une décision logistique aux poids choisis.
    messages = ['serveur arrêté merci', 'serveur documentation merci', 'documentation tarif']
    vocabulaire, idf, vecteurs = tfidf(messages)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        reference = TfidfVectorizer(token_pattern=r'\S+').fit(messages)
        attendu = reference.transform(messages).toarray()
        ordre = [reference.vocabulary_[m] for m in vocabulaire]
        assert all(abs(a - attendu[i][j]) < 1e-12 for i, v in enumerate(vecteurs) for a, j in zip(v, ordre))
        verifie = True
    except ImportError:
        verifie = False
    poids = {'arrêté': 2.0, 'documentation': -2.0}
    w = [poids.get(m, 0.0) for m in vocabulaire]
    decisions = [probabilite_logistique(w, 0.0, v) for v in vecteurs]
    resultats['tfidf'] = {'messages': messages, 'vocabulaire': vocabulaire, 'idf': idf, 'vecteurs': vecteurs,
                          'verifie_avec_scikit_learn': verifie, 'poids_choisis': poids,
                          'scores': [d[0] for d in decisions], 'probabilites': [d[1] for d in decisions]}
    dest = ROOT/'outputs/E03_calculs_guides'
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'resultats.json').write_text(json.dumps(resultats,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Calculs didactiques et propriétés contrôlés ; résultats enregistrés.')



if __name__ == '__main__':
    main()
