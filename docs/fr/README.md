# Guide français

[Accueil](../../README.md) · **Français** · [English](../en/README.md) · [Español](../es/README.md) · [Português](../pt/README.md)

Ce dépôt est le laboratoire du livre **La mécanique interne des LLM** (tome 2) de Mustapha Alouani, publié en français, anglais, espagnol et portugais. Le livre explique les questions, les mécanismes et la lecture des résultats ; ce dépôt permet de refaire les 24 expériences, d'examiner leurs sorties et de les prolonger. Tout fonctionne sur un processeur ordinaire (CPU), sans carte graphique ni service payant.

## 1. Préparer l'environnement

Exécuter les commandes depuis la racine du dépôt. L'environnement de référence utilise Python 3.13.3 et les versions consignées dans `requirements-cpu.txt` ; les autres versions ne sont pas annoncées comme testées. Prévoir une connexion pour l'installation et le premier téléchargement des modèles ; les expériences utilisent ensuite les copies locales.

```bash
python -m venv .venv
```

Activer l'environnement : `.venv\Scripts\Activate.ps1` sous PowerShell, ou `source .venv/bin/activate` sous Linux/macOS. Puis :

```bash
python -m pip install -r requirements-cpu.txt
python experiences/00_telecharger_modeles.py
python experiences/00_telecharger_modeles.py --verifier-seulement
```

Les poids sont téléchargés depuis leurs distributions officielles, aux révisions fixées par `modeles.json` ; leurs licences propres restent applicables. Le parcours initial a été produit sur CPU sous Windows, les compléments sur CPU sous Linux, où le programme principal a aussi été réexécuté sans modification (`outputs/verification/reexecution-linux.json`). macOS n'a pas été testé.

## 2. Les 24 expériences, dans l'ordre du livre

Chaque expérience porte un identifiant **E01 à E24**, attribué dans l'ordre où le livre la présente. Le même identifiant se retrouve partout : dans le texte du livre, au début du nom du programme (`experiences/E09_extraction.py`) et au début du nom de son dossier de sortie (`outputs/E09_extraction/`). E06 compte deux programmes : le premier applique le classificateur, le second vérifie cette application.

| Id. | Chap. | Programme (`experiences/`) | Question | Nature |
|---|---|---|---|---|
| **E01** | 1, 4 | `E01_classification.py` | Un classificateur peut-il réussir grâce à la longueur des messages plutôt qu'à la tâche ? | exploratoire |
| **E02** | 2 | `E02_calcul_observe.py` | Que calcule GPT-2 sur une phrase : tokens, états internes, logits, masque ? | observation |
| **E03** | 2, 3, 5, 7, 12 | `E03_calculs_guides.py` | Softmax, masque, métriques, kappa, ACC : les calculs du cours refaits par programme | données fabriquées |
| **E04** | 3, 4, 5 | `E04_intervalles.py` | Quelle incertitude accompagne trois résultats du livre ? (après E05 et E07) | relecture de sorties |
| **E05** | 4 | `E05_validation_classificateur.py` | La réparation de E01 tient-elle sur 24 messages écrits après elle ? (après E01) | validation |
| **E06** | 4 | `E06_prediction.py`, `E06_verifier_prediction.py` | Le pipeline sauvegardé s'applique-t-il à un fichier de messages, et avec quelles erreurs ? (après E01) | contrôle fonctionnel |
| **E07** | 5 | `E07_recherche.py` | Quelle recherche, lexicale ou dense, retrouve le paragraphe pertinent ? | exploratoire |
| **E08** | 5 | `E08_geometrie_recherche.py` | Que valent les scores de la recherche dense face au niveau de fond ? | observation et témoin |
| **E09** | 6 | `E09_extraction.py` | Un format contraint garantit-il le bon champ ? | contraste |
| **E10** | 6, 7 | `E10_extraction_defis.py` | Pourquoi le décodage contraint répond-il toujours null, et quelle règle le corrige ? (environ 5 min) | règle écrite avant exécution |
| **E11** | 7 | `E11_choix_abstention.py` | Que gagne et que manque une règle d'abstention ? (après E09) | exploratoire |
| **E12** | 7 | `E12_latence.py` | Combien coûtent en temps l'extraction libre et l'extraction contrainte, échauffement compris ? | mesures répétées |
| **E13** | 8, 10 | `E13_inspection.py` | Que lit la Logit Lens couche après couche, et que fait un pilotage contrôlé ? | lecture et pilotage |
| **E14** | 8 | `E14_attention.py` | Que montrent les poids d'attention, et une ablation confirme-t-elle la lecture ? | lecture puis ablation |
| **E15** | 8 | `E15_sondes.py` | Une sonde lit-elle une information, ou un indice plus simple ? | témoins |
| **E16** | 9 | `E16_intervention.py` | Remplacer une activation restaure-t-il la réponse sur des noms non vus ? | découverte puis validation |
| **E17** | 9 | `E17_carte_patching.py` | Où, couche par couche et position par position, se joue la réponse ? | règle écrite avant exécution |
| **E18** | 9 | `E18_tetes.py` | Quelles têtes d'attention portent l'effet, et cela tient-il sur un second gabarit ? (environ 4 min) | règle écrite avant exécution |
| **E19** | 10 | `E19_adaptation.py` | Qu'est-ce qui change après LoRA, et retirer l'adaptateur restaure-t-il le modèle ? | exploratoire |
| **E20** | 10 | `E20_pilotage.py` | La direction de pilotage résiste-t-elle aux directions aléatoires et au sens inverse ? | témoins |
| **E21** | 11 | `E21_langues.py` | Que change la langue pour une même question factuelle ? | observation |
| **E22** | 11 | `E22_langues_formulations.py` | La formulation pèse-t-elle autant que la langue ? | règle écrite avant exécution |
| **E23** | 11 | `E23_langues_tache.py` | Une même extraction réussit-elle dans quatre langues ? | textes parallèles fabriqués |
| **E24** | 12 | `E24_corpus_historique.py` | L'arithmétique de la quantification historique est-elle cohérente ? | sorties historiques |

Le fichier `experiences/socle_experiences.py` n'est pas une expérience : il réunit le code commun (chargement des modèles aux révisions fixées, écriture des manifestes) et le code de E01, E07, E09, E16 et E21, que lancent les programmes du même nom.

## 3. Lancer le parcours ou une seule expérience

```bash
python run_cpu.py              # les 24 expériences, dans l'ordre du parcours
python run_cpu.py E09 E11      # seulement celles-ci
python experiences/E09_extraction.py
```

Le lanceur garde un journal par programme dans `outputs/verification/` et s'arrête dès qu'une commande échoue. Quelques expériences relisent les sorties d'une autre : E05 et E06 après E01, E11 après E09, E04 après E05 et E07. Le lanceur respecte cet ordre. Une variante ne doit pas modifier en silence les paramètres du livre : conserver les anciennes sorties et documenter le nouveau protocole.

## 4. Appliquer le classificateur (E06)

```bash
python experiences/E06_prediction.py data/messages-exemple.txt outputs/E06_prediction/predictions-exemple.csv
python experiences/E06_verifier_prediction.py
```

Une ligne UTF-8 représente un message. Le contrôle fonctionnel ne garantit pas des décisions correctes : une erreur sémantique du petit pipeline est volontairement conservée et discutée dans le livre. Ne charger que les fichiers `joblib` produits par votre propre parcours ; ce format peut exécuter du code.

## 5. Comment les expériences sont construites

Toutes suivent la méthode présentée au chapitre 1 du livre : une **question** posée avant de regarder le résultat ; **une seule chose changée**, le reste maintenu fixe ; un **témoin** qui dit ce que l'on obtiendrait sans l'effet supposé ; une **mesure** choisie d'avance ; une **règle de décision** ; et la **portée** de la conclusion, écrite avec ses limites.

Chaque expérience a l'un de ces statuts, indiqué dans la colonne « Nature » :

- **exploratoire** : elle rend un mécanisme visible sur un petit jeu de données ; son résultat suggère, il ne prouve pas ;
- **règle écrite avant exécution** : la règle de décision est écrite dans l'en-tête du programme avant le premier lancement ; elle n'a pas été ajustée ensuite ;
- **données fabriquées**, **sorties historiques**, **relecture de sorties** : calculs sans nouveau modèle, sur des données explicitement construites ou sur des agrégats conservés d'une campagne précédente.

Chaque dossier de sortie contient un **manifeste** `metadata.json` : la question, les contrôles, les limites, l'empreinte du programme, les révisions des modèles, la graine, le processeur, les versions des bibliothèques et la durée. Le lire avant de comparer deux scores. Les durées incluent les chargements et ne constituent pas un banc d'essai.

## 6. Diagnostiquer

- Modèle absent : relancer la préparation en ligne, puis vérifier les révisions ; ne pas remplacer `modeles.json` en silence.
- Sortie d'une autre expérience absente : lancer d'abord celle dont elle dépend (section 3).
- Résultat différent : comparer versions, entrées et métriques avant de changer une tolérance.
- E24 ne valide ni l'annotation ni la représentativité des participants au Grand Débat : les fichiers `historique-*` sont des agrégats conservés, pas un nouvel encodage.

`python tests/check_package.py` vérifie le dépôt : programmes lisibles, un programme et un dossier par identifiant, et, pour chaque manifeste, que le programme présent est bien celui qui l'a produit.

## 7. Ancienne numérotation

Avant le 23 septembre 2026, les expériences s'appelaient E1 à E8, complétées de noms (E3-defis, patching…). Les manifestes et les journaux d'exécution gardent ces anciens noms, qui étaient en vigueur au moment du calcul ; ils n'ont pas été réécrits. `RENUMEROTATION.json` donne la correspondance et, pour chaque programme renommé, les lignes modifiées : en les remettant à leur état d'origine, on retrouve exactement l'empreinte enregistrée dans le manifeste.

| Ancien nom | Identifiant | Dossier de sortie |
|---|---|---|
| E1 | **E01** | `outputs/E01_classification/` |
| calcul-observe | **E02** | `outputs/E02_calcul_observe/` |
| calculs | **E03** | `outputs/E03_calculs_guides/` |
| intervalles | **E04** | `outputs/E04_intervalles/` |
| E1-validation | **E05** | `outputs/E05_validation_classificateur/` |
| prédiction | **E06** | `outputs/E06_prediction/` |
| E2 | **E07** | `outputs/E07_recherche/` |
| E2-geometrie | **E08** | `outputs/E08_geometrie_recherche/` |
| E3 | **E09** | `outputs/E09_extraction/` |
| E3-defis | **E10** | `outputs/E10_extraction_defis/` |
| E4 | **E11** | `outputs/E11_choix_abstention/` |
| latence | **E12** | `outputs/E12_latence/` |
| inspection | **E13** | `outputs/E13_inspection/` |
| attention | **E14** | `outputs/E14_attention/` |
| sondes | **E15** | `outputs/E15_sondes/` |
| E5 | **E16** | `outputs/E16_intervention/` |
| patching | **E17** | `outputs/E17_carte_patching/` |
| tetes | **E18** | `outputs/E18_tetes/` |
| E6 | **E19** | `outputs/E19_adaptation/` |
| pilotage | **E20** | `outputs/E20_pilotage/` |
| E7 | **E21** | `outputs/E21_langues/` |
| E7-formulations | **E22** | `outputs/E22_langues_formulations/` |
| E7-tache | **E23** | `outputs/E23_langues_tache/` |
| E8 | **E24** | `outputs/E24_corpus_historique/` |

<!-- livre:debut -->
### 📕 Le livre

**La mécanique interne des LLM** — Tome 2 — Comprendre les mécanismes, construire des outils, vérifier leurs résultats, par Mustapha Alouani.

- Broché — *à paraître*
- Kindle — *à paraître*
- [Site de l'auteur](https://alouani.org)

Autres éditions : 🇬🇧 The Inner Mechanics of LLMs · 🇪🇸 La mecánica interna de los LLM · 🇧🇷 A mecânica interna dos LLM

Du même auteur : **La Mécanique des LLM** — Théorie, architecture et pratique pour l'ingénieur — [Broché](https://amzn.eu/d/3oREERI) · [Kindle](https://amzn.eu/d/b7sG5iw) · [scripts du livre](https://github.com/alouani-org/mecanics-of-llms)
<!-- livre:fin -->
