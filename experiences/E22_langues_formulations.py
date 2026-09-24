"""E22 : faire varier la formulation dans chaque langue avant de comparer les langues.

Chapitre « Changer de langue et de terrain ». E21 pose la même question factuelle
une fois par langue. Si l'on change seulement la formulation, sans changer de
langue, le rang de « Paris » varie-t-il plus ou moins qu'entre les langues ?

Règle écrite avant l'exécution (23 septembre 2026) :
- trois formulations par langue : celle de E21, une variante amorcée par un
  exemple (« … l'Allemagne … Berlin. »), une paraphrase (« … s'appelle ») ;
- mesure : rang du token « ·Paris » au prochain token, et sa probabilité ;
- écart d'une série = max − min de log10(rang) ;
- écart entre langues = écart des quatre formulations de E21 ;
- écart interne moyen = moyenne, sur les quatre langues, de l'écart des trois
  formulations de la langue ;
- conclusion prévue : si l'écart interne moyen est au moins égal à l'écart
  entre langues, le tableau de E21 ne peut pas servir à ordonner les langues,
  même pour cette seule question.

Usage : python experiences/E22_langues_formulations.py
Sortie : outputs/E22_langues_formulations/metadata.json
"""
import math
import time

import torch

from socle_experiences import model, snapshot, save

FORMULATIONS = {
    "fr": {"E7": "La capitale de la France est",
           "amorcée": "La capitale de l'Allemagne est Berlin. La capitale de la France est",
           "paraphrase": "La ville capitale de la France s'appelle"},
    "en": {"E7": "The capital of France is",
           "amorcée": "The capital of Germany is Berlin. The capital of France is",
           "paraphrase": "The capital city of France is called"},
    "es": {"E7": "La capital de Francia es",
           "amorcée": "La capital de Alemania es Berlín. La capital de Francia es",
           "paraphrase": "La ciudad capital de Francia se llama"},
    "pt": {"E7": "A capital da França é",
           "amorcée": "A capital da Alemanha é Berlim. A capital da França é",
           "paraphrase": "A cidade capital da França se chama"},
}


def lire_cible(m, tok, texte, cible):
    """Rang (1 = premier) et probabilité du token cible au prochain token."""
    ids = tok(texte, return_tensors="pt")
    with torch.no_grad():
        logits = m(**ids).logits[0, -1]
    rang = int((logits > logits[cible]).sum()) + 1
    return {"texte": texte, "tokens": tok.convert_ids_to_tokens(ids.input_ids[0]),
            "n_tokens": int(ids.input_ids.shape[1]), "rang": rang,
            "probabilite": float(logits.softmax(-1)[cible]), "premier": tok.decode(int(logits.argmax()))}


def ecart(rangs):
    valeurs = [math.log10(r) for r in rangs]
    return max(valeurs) - min(valeurs)


def main():
    debut = time.perf_counter()
    m, tok = model()
    cible = tok.encode(" Paris")
    assert len(cible) == 1
    lignes = {langue: {nom: lire_cible(m, tok, texte, cible[0]) for nom, texte in series.items()}
              for langue, series in FORMULATIONS.items()}
    entre_langues = ecart([lignes[l]["E7"]["rang"] for l in lignes])
    internes = {l: ecart([x["rang"] for x in lignes[l].values()]) for l in lignes}
    interne_moyen = sum(internes.values()) / len(internes)
    save("E22_langues_formulations", {
        "lignes": lignes, "ecart_entre_langues_log10": entre_langues,
        "ecarts_internes_log10": internes, "ecart_interne_moyen_log10": interne_moyen,
        "classement_E7_interpretable": bool(interne_moyen < entre_langues),
    }, "La formulation fait-elle varier le rang de la réponse autant que la langue ?",
        ["Trois formulations par langue, dont celle de E7", "Même modèle, même token cible",
         "Règle de comparaison des écarts écrite avant l'exécution"],
        ["Une seule question factuelle ; trois formulations ne couvrent pas l'espace des formulations",
         "Traductions faites par l'auteur, non validées par des locuteurs",
         "Le token « ·Paris » n'est pas la graphie de toutes les réponses correctes possibles"],
        debut, {"gpt2": snapshot("gpt2")[1]})


if __name__ == "__main__":
    main()
