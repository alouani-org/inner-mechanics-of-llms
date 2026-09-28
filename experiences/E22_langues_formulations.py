"""E22: vary the wording within each language before comparing languages.

Chapter 11. E21 asks the same factual question
once per language. If we change only the wording, without changing
language, does the rank of "Paris" vary more or less than across languages?

Rule written before execution (23 September 2026):
- three wordings per language: that of E21, a variant primed by an
  example ("… l'Allemagne … Berlin."), a paraphrase ("… s'appelle");
- measurement: rank of the token "·Paris" at the next token, and its probability;
- spread of a series = max − min of log10(rank);
- spread across languages = spread of the four E21 wordings;
- mean internal spread = mean, over the four languages, of the spread of the three
  wordings of the language;
- planned conclusion: if the mean internal spread is at least equal to the spread
  across languages, the E21 table cannot be used to rank the languages,
  even for this single question.

Usage: python experiences/E22_langues_formulations.py
Output: outputs/E22_langues_formulations/metadata.json
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
    """Rank (1 = first) and probability of the target token at the next token."""
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
