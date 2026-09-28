"""E05: does the repaired classifier hold up on new messages?

Chapter 4. The repair of E01 was
decided after seeing the test: that same test can no longer validate it. This
program applies the two lexical pipelines saved by E01 (before and after
balancing) to 24 messages written for this validation, after the repair and
without any later tuning. Half incidents, half information requests;
half of each group carries the E01 politeness formula. Part of the
vocabulary is absent from training, by construction.

Usage: python experiences/E05_validation_classificateur.py   (after E01)
Output: outputs/E05_validation_classificateur/metadata.json
"""
import time
from pathlib import Path

import joblib

from socle_experiences import OUT, save

SUFFIXE = " Merci de traiter ce message transmis ce matin par notre équipe technique."
INCIDENTS = ["L'imprimante du deuxième étage ne s'allume plus", "Le formulaire renvoie une erreur à l'envoi",
             "Le paiement est refusé depuis ce matin", "Impossible d'ouvrir la session sur le poste",
             "La connexion au serveur échoue", "Le site affiche une page blanche",
             "Le logiciel se ferme sans prévenir", "Les courriels ne partent plus",
             "Le compte client est bloqué", "La commande reste en attente d'erreur",
             "Le réseau Wi-Fi est coupé dans la salle", "L'application ne répond plus"]
INFORMATIONS = ["Pouvez-vous m'envoyer la notice d'utilisation", "Je souhaite connaître les délais de livraison",
                "Quels sont vos horaires d'ouverture", "Je voudrais le catalogue des formations",
                "Où trouver la documentation de l'interface", "Je cherche le tarif de l'abonnement annuel",
                "Existe-t-il une présentation du service", "Je souhaite recevoir le guide de démarrage",
                "Pouvez-vous m'indiquer la procédure d'inscription", "Je demande la liste des options disponibles",
                "Quelle est l'adresse de votre agence", "Je voudrais consulter les conditions générales"]


def construire():
    """Same proportion of messages with and without suffix in each category."""
    textes, references = [], []
    for categorie, liste in [(1, INCIDENTS), (0, INFORMATIONS)]:
        for n, texte in enumerate(liste):
            textes.append(texte + "." + (SUFFIXE if n % 2 == 0 else ""))
            references.append(categorie)
    return textes, references


def evaluer(chemin, textes, references):
    pipeline = joblib.load(chemin)      # artifact produced by E01 in this repository
    predictions = [int(p) for p in pipeline.predict(textes)]
    lignes = [{"texte": t, "reference": r, "prediction": p, "suffixe": SUFFIXE.strip() in t}
              for t, r, p in zip(textes, references, predictions)]
    def exactitude(sous):
        return sum(l["reference"] == l["prediction"] for l in sous) / len(sous)
    return {"exactitude": exactitude(lignes),
            "avec_suffixe": exactitude([l for l in lignes if l["suffixe"]]),
            "sans_suffixe": exactitude([l for l in lignes if not l["suffixe"]]),
            "incidents": exactitude([l for l in lignes if l["reference"] == 1]),
            "informations": exactitude([l for l in lignes if l["reference"] == 0]),
            "lignes": lignes}


def main():
    debut = time.perf_counter()
    textes, references = construire()
    dossier = OUT / "E01_classification"
    resultats = {"n": len(textes),
                 "avant_equilibrage": evaluer(dossier / "classificateur.joblib", textes, references),
                 "apres_equilibrage": evaluer(dossier / "classificateur-equilibre.joblib", textes, references)}
    save("E05_validation_classificateur", resultats,
         "La réparation post hoc de E01 tient-elle sur des messages écrits après elle ?",
         ["Messages écrits après la réparation, sans réglage ultérieur",
          "Suffixe présent dans la moitié de chaque catégorie",
          "Mêmes messages pour les deux pipelines"],
         ["Messages fabriqués par l'auteur de l'expérience, pas un flux réel",
          "24 messages : une estimation grossière",
          "Seuls les pipelines lexicaux sont sauvegardés par E01 ; la sonde n'est pas réévaluée"],
         debut)


if __name__ == "__main__":
    main()
