"""E02 : suivre une phrase dans GPT-2 : tokens, états internes, logits et masque.

Chapitre « Du texte aux scores ». Programme d'observation, sans intervention sur
le calcul : il enregistre ce que produit chaque étape pour que le livre puisse
lire des sorties réelles. CPU imposé, modèle et révision fixés par modeles.json.

Usage : python experiences/E02_calcul_observe.py
Sortie : outputs/E02_calcul_observe/metadata.json
"""
import json
import math
import time

import torch

from socle_experiences import model, snapshot, save

# Le même fait, sans puis avec un exemple qui amorce le patron de phrase.
PROMPT_BRUT = "La capitale de la France est"
PROMPT_AMORCE = "La capitale de l'Allemagne est Berlin. La capitale de la France est"

# Même contenu dans quatre langues : on compte ce que produit le tokenizer.
PHRASES_PARALLELES = {
    "en": "The internal representation depends on context.",
    "es": "La representación interna depende del contexto.",
    "pt": "A representação interna depende do contexto.",
    "fr": "La représentation interne dépend du contexte.",
}

# Prompt et grammaire de l'expérience E09 (socle_experiences.py, fonction extraction),
# recopiés à l'identique pour observer le masque sur un document précis.
PROMPT_E3 = ('Extraire la ville du rendez-vous. Si elle manque, écrire null. Répondre en JSON.\n'
             'Texte: Le rendez-vous est à Paris.\nJSON: {"ville":"Paris"}\n'
             'Texte: Aucun lieu annoncé.\nJSON: {"ville":null}\n')
DOCUMENT_E3 = "Le bureau est à Paris, mais le rendez-vous se tiendra à Lyon."
VALEURS_AUTORISEES = ["Paris", "Lyon", "Rome", None]


def decouper(tok, texte):
    """Identifiants et fragments produits par le tokenizer, dans l'ordre."""
    ids = tok.encode(texte, add_special_tokens=False)
    return [{"id": i, "fragment": tok.decode([i])} for i in ids]


def trajectoire(m, tok, texte):
    """Norme et orientation de l'état de la dernière position, lecture par lecture."""
    entree = tok(texte, return_tensors="pt")
    with torch.no_grad():
        etats = m(**entree, output_hidden_states=True).hidden_states
    lignes = []
    precedent = None
    premier = etats[0][0, -1]
    for numero, etat in enumerate(etats):
        v = etat[0, -1]
        lignes.append({
            "lecture": numero,
            "forme": list(etat.shape),
            "norme": float(v.norm()),
            "cos_lecture_0": float(torch.nn.functional.cosine_similarity(v, premier, dim=0)),
            "cos_precedente": None if precedent is None else
            float(torch.nn.functional.cosine_similarity(v, precedent, dim=0)),
        })
        precedent = v
    return lignes


def meilleurs_candidats(m, tok, texte, k=5, cible=" Paris"):
    """Les k tokens de plus haut score à la dernière position, et le sort de la cible."""
    entree = tok(texte, return_tensors="pt")
    with torch.no_grad():
        logits = m(**entree).logits[0, -1]
    probabilites = logits.softmax(-1)
    valeurs, indices = torch.topk(logits, k)
    id_cible = tok.encode(cible, add_special_tokens=False)
    assert len(id_cible) == 1, "la cible doit être un seul token"
    id_cible = id_cible[0]
    return {
        "prompt": texte,
        "n_tokens": entree.input_ids.shape[1],
        "candidats": [{"rang": r + 1, "token": tok.decode([int(i)]), "logit": float(v),
                       "probabilite": float(probabilites[i])}
                      for r, (v, i) in enumerate(zip(valeurs, indices))],
        "cible": cible,
        "logit_cible": float(logits[id_cible]),
        "probabilite_cible": float(probabilites[id_cible]),
        "rang_cible": int((logits > logits[id_cible]).sum()) + 1,
        "logit_moyen": float(logits.mean()),
    }


def appliquer_masque(logits, autorises):
    """Mettre à -inf les tokens interdits, puis renormaliser par softmax."""
    masque = torch.full_like(logits, -math.inf)
    masque[autorises] = 0.0
    return (logits + masque).softmax(-1)


def observer_masque(m, tok):
    """Suivre, pas à pas, le décodage contraint de E09 sur un document ambigu."""
    options = [json.dumps({"ville": v}, ensure_ascii=False, separators=(",", ":"))
               for v in VALEURS_AUTORISEES]
    sequences = [tok.encode(o, add_special_tokens=False) + [tok.eos_token_id] for o in options]
    prompt = PROMPT_E3 + "Texte: " + DOCUMENT_E3 + "\nJSON:"
    prefixe = tok.encode(prompt, add_special_tokens=False)
    produits = []
    etapes = []
    while True:
        compatibles = [s for s in sequences if s[:len(produits)] == produits and len(s) > len(produits)]
        if not compatibles:
            break
        autorises = sorted({s[len(produits)] for s in compatibles})
        with torch.no_grad():
            logits = m(torch.tensor([prefixe + produits])).logits[0, -1]
        libre = logits.softmax(-1)
        contraint = appliquer_masque(logits, autorises)
        haut = torch.topk(libre, 5)
        choisi = int(contraint.argmax())
        etapes.append({
            "etape": len(produits) + 1,
            "libres_top5": [{"token": tok.decode([int(i)]), "probabilite": float(p)}
                            for p, i in zip(haut.values, haut.indices)],
            "autorises": [{"token": tok.decode([a]), "probabilite_libre": float(libre[a]),
                           "probabilite_contrainte": float(contraint[a])} for a in autorises],
            "masse_autorisee": float(libre[autorises].sum()),
            "choisi": tok.decode([choisi]),
        })
        produits.append(choisi)
        if choisi == tok.eos_token_id:
            break
    # Score de chaque objet complet : somme des log-probabilités de ses tokens.
    scores = {}
    for option, sequence in zip(options, sequences):
        total = 0.0
        for n, token in enumerate(sequence):
            with torch.no_grad():
                logits = m(torch.tensor([prefixe + sequence[:n]])).logits[0, -1]
            total += float(logits.log_softmax(-1)[token])
        scores[option] = total
    return {"document": DOCUMENT_E3, "options": options,
            "decoupage_options": {o: [tok.decode([i]) for i in s[:-1]] for o, s in zip(options, sequences)},
            "etapes": etapes, "sortie": tok.decode(produits, skip_special_tokens=True),
            "log_probabilite_objet_complet": scores}


def main():
    debut = time.perf_counter()
    m, tok = model()
    resultats = {
        "tokens_brut": decouper(tok, PROMPT_BRUT),
        "tokens_amorce": decouper(tok, PROMPT_AMORCE),
        "langues": {langue: {"phrase": phrase, "caracteres": len(phrase),
                             "tokens": len(decouper(tok, phrase)),
                             "fragments": [t["fragment"] for t in decouper(tok, phrase)]}
                    for langue, phrase in PHRASES_PARALLELES.items()},
        "trajectoire_amorce": trajectoire(m, tok, PROMPT_AMORCE),
        "candidats_brut": meilleurs_candidats(m, tok, PROMPT_BRUT),
        "candidats_amorce": meilleurs_candidats(m, tok, PROMPT_AMORCE),
        "dimensions": {"couches": m.config.n_layer, "largeur": m.config.n_embd,
                       "tetes": m.config.n_head, "vocabulaire": m.config.vocab_size},
        "masque_e3": observer_masque(m, tok),
    }
    save("E02_calcul_observe", resultats,
         "Que produit chaque étape du calcul pour une phrase, et que change un masque de décodage ?",
         ["Même modèle et même révision pour toutes les lectures",
          "Mêmes positions lues pour les deux prompts (dernier token)",
          "Prompt E09 recopié à l'identique ; masque appliqué aux mêmes logits"],
         ["Observation d'une phrase et d'un document : illustration, pas statistique",
          "Les nombres de tokens décrivent le tokenizer, pas la compétence du modèle dans une langue",
          "Aucune intervention sur les états : aucune conclusion causale sur le rôle d'une couche"],
         debut, {"gpt2": snapshot("gpt2")[1]})


if __name__ == "__main__":
    main()
