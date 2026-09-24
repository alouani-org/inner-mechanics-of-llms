# 🎓 La mécanique interne des LLM — laboratoire compagnon · companion lab

> **🌍 Choisir sa langue · Choose your language · Elegir idioma · Escolher a língua**
>
> [🇫🇷 Français](docs/fr/README.md) · [🇬🇧 English](docs/en/README.md) · [🇪🇸 Español](docs/es/README.md) · [🇧🇷 Português](docs/pt/README.md)

**24 expériences reproductibles sur CPU** pour le livre *La mécanique interne des LLM* (tome 2) de Mustapha Alouani : comprendre ce que calcule un modèle de langage, construire des outils de classement, de recherche et d'extraction, inspecter et modifier un modèle, puis vérifier chaque résultat. Chaque expérience porte le même identifiant dans le livre, dans le nom de son programme et dans celui de son dossier de sortie.

**24 reproducible CPU experiments** for the book *La mécanique interne des LLM* (volume 2, in French) by Mustapha Alouani: understand what a language model computes, build classification, retrieval and extraction tools, inspect and modify a model, then check every result. Each experiment carries the same identifier in the book, in its program name and in its output folder.

<!-- livre:debut -->
### 📕 Le livre · The book · El libro · O livro

**La mécanique interne des LLM** — Tome 2 — Comprendre les mécanismes, construire des outils, vérifier leurs résultats — Mustapha Alouani (édition française · French edition)

- **Broché · Paperback · Tapa blanda · Capa comum** : *à paraître · coming soon · próximamente · em breve*
- **Kindle** : *à paraître · coming soon · próximamente · em breve*
- **Site** : [alouani.org](https://alouani.org)

Du même auteur · By the same author : **La Mécanique des LLMs** — [Broché · Paperback](https://amzn.eu/d/3oREERI) · [Kindle](https://amzn.eu/d/b7sG5iw) · [scripts](https://github.com/alouani-org/mecanics-of-llms)
<!-- livre:fin -->

---

## 🇫🇷 Français

👉 **[Guide complet : installation, les 24 expériences, méthode, diagnostic](docs/fr/README.md)**

## 🇬🇧 English

👉 **[Full guide: setup, the 24 experiments, method, troubleshooting](docs/en/README.md)**

## 🇪🇸 Español

👉 **[Guía completa: instalación, los 24 experimentos, método, diagnóstico](docs/es/README.md)**

## 🇧🇷 Português

👉 **[Guia completo: instalação, as 24 experiências, método, diagnóstico](docs/pt/README.md)**

---

## 🚀 Démarrage rapide · Quick start

```bash
python -m venv .venv
# Windows : .venv\Scripts\Activate.ps1   —   Linux/macOS : source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python experiences/00_telecharger_modeles.py
python run_cpu.py                 # E01 → E24
python tests/check_package.py     # contrôles de provenance · provenance checks
```

Aucun GPU, aucune API payante · No GPU, no paid API.

## 🔬 Les 24 expériences · The 24 experiments

| ID | Chap. | Program (`experiences/`) | Question |
|---|---|---|---|
| **E01** | 1, 4 | `E01_classification.py` | Can a classifier succeed by exploiting message length instead of the task? |
| **E02** | 2 | `E02_calcul_observe.py` | What does GPT-2 compute on a sentence: tokens, hidden states, logits, mask? |
| **E03** | 2, 3, 5, 7, 12 | `E03_calculs_guides.py` | Softmax, mask, metrics, kappa, ACC: the course calculations, redone by a program |
| **E04** | 3, 4, 5 | `E04_intervalles.py` | What uncertainty comes with three results of the book? (after E05 and E07) |
| **E05** | 4 | `E05_validation_classificateur.py` | Does the E01 repair hold on 24 messages written after it? (after E01) |
| **E06** | 4 | `E06_prediction.py`, `E06_verifier_prediction.py` | Does the saved pipeline apply to a file of messages, and with which errors? (after E01) |
| **E07** | 5 | `E07_recherche.py` | Which retrieval, lexical or dense, finds the relevant paragraph? |
| **E08** | 5 | `E08_geometrie_recherche.py` | What are dense-retrieval scores worth against the background level? |
| **E09** | 6 | `E09_extraction.py` | Does a constrained format guarantee the right field? |
| **E10** | 6, 7 | `E10_extraction_defis.py` | Why does constrained decoding always answer null, and which rule fixes it? (about 5 min) |
| **E11** | 7 | `E11_choix_abstention.py` | What does an abstention rule gain and miss? (after E09) |
| **E12** | 7 | `E12_latence.py` | How much time do free and constrained extraction cost, warm-up included? |
| **E13** | 8, 10 | `E13_inspection.py` | What does the Logit Lens read layer by layer, and what does controlled steering do? |
| **E14** | 8 | `E14_attention.py` | What do attention weights show, and does an ablation confirm the reading? |
| **E15** | 8 | `E15_sondes.py` | Does a probe read information, or a simpler cue? |
| **E16** | 9 | `E16_intervention.py` | Does replacing an activation restore the answer on unseen names? |
| **E17** | 9 | `E17_carte_patching.py` | Where, layer by layer and position by position, is the answer decided? |
| **E18** | 9 | `E18_tetes.py` | Which attention heads carry the effect, and does it hold on a second template? (about 4 min) |
| **E19** | 10 | `E19_adaptation.py` | What changes after LoRA, and does removing the adapter restore the model? |
| **E20** | 10 | `E20_pilotage.py` | Does the steering direction survive random directions and the reversed sign? |
| **E21** | 11 | `E21_langues.py` | What does the language change for the same factual question? |
| **E22** | 11 | `E22_langues_formulations.py` | Does phrasing weigh as much as language? |
| **E23** | 11 | `E23_langues_tache.py` | Does the same extraction succeed in four languages? |
| **E24** | 12 | `E24_corpus_historique.py` | Is the arithmetic of the historical quantification consistent? |

Questions in French, Spanish and Portuguese, and the type of each experiment (exploratory, rule written before running, fabricated data…), are in the guides.

## 📂 Structure

```
inner-mechanics-of-llms/
├── experiences/          ← 00_telecharger_modeles.py, E01_… à/to E24_…, socle_experiences.py (code commun · shared code)
├── outputs/              ← un dossier par expérience · one folder per experiment (E01_classification/ … E24_corpus_historique/)
│   ├── historique/       ← agrégats conservés de la campagne précédente · retained historical aggregates
│   └── verification/     ← journaux d'exécution et contrôles · run logs and checks
├── data/                 ← exemples fabriqués pour le livre · fabricated teaching examples
├── docs/fr|en|es|pt/     ← guides
├── tests/check_package.py
├── outils/liens_livre.py ← met à jour les liens du livre · updates the book links (livre.json)
├── run_cpu.py            ← parcours complet · full path
├── modeles.json          ← révisions exactes des modèles · exact model revisions
├── requirements-cpu.txt
├── RENUMEROTATION.json   ← ancienne numérotation (E1–E8) → E01–E24 · former numbering
├── livre.json            ← liens du livre · book links
└── DATA_AND_MODELS.md
```

## 🧭 Méthode · Method

Chaque expérience pose une question avant de regarder le résultat, change une seule chose, garde un témoin, fixe sa mesure et sa règle de décision d'avance, et écrit la portée de sa conclusion. Son manifeste `metadata.json` conserve la question, les contrôles, les limites, l'empreinte du programme et les révisions des modèles. Ce sont de petits exemples contrôlés pour comprendre un mécanisme, pas des bancs d'essai industriels.

Each experiment asks its question before looking at the result, changes one thing, keeps a control, fixes its measure and decision rule in advance, and states the scope of its conclusion. Its `metadata.json` manifest keeps the question, controls, limits, program hash and model revisions. These are small controlled examples for understanding a mechanism, not industrial benchmarks.

---

## 📜 Licence · License

Code et exemples fabriqués : licence MIT (`LICENSE`). Les poids des modèles pré-entraînés ne sont pas redistribués ; leurs licences propres s'appliquent (`DATA_AND_MODELS.md`).

Code and fabricated examples: MIT licence (`LICENSE`). Pretrained model weights are not redistributed; their own licences apply (`DATA_AND_MODELS.md`).

**Bonne expérimentation ! · Happy experimenting! · ¡Buena experimentación! · Boa experimentação!**
