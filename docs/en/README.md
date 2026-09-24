# English guide

[Home](../../README.md) · [Français](../fr/README.md) · **English** · [Español](../es/README.md) · [Português](../pt/README.md)

This repository is the laboratory of the book **The Inner Mechanics of LLMs** (volume 2) by Mustapha Alouani, published in English, French, Spanish and Portuguese. The book explains the questions, the mechanisms and how to read the results; this repository lets you rerun the 24 experiments, inspect their outputs and extend them. Everything runs on an ordinary processor (CPU), with no GPU and no paid service.

## 1. Setup

Run commands from the repository root. The reference environment uses **Python 3.13.3** and the versions fixed in `requirements-cpu.txt`; other versions are not claimed as tested. Internet access is needed for installation and the first model download; experiments then use local copies.

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on Linux/macOS. Then:

```bash
python -m pip install -r requirements-cpu.txt
python experiences/00_telecharger_modeles.py
python experiences/00_telecharger_modeles.py --verifier-seulement
```

Weights are downloaded from their official distributions at the revisions fixed in `modeles.json`; their own licences apply. The initial path was produced on CPU under Windows, the additions on CPU under Linux, where the main program was also rerun unchanged (`outputs/verification/reexecution-linux.json`). macOS has not been tested.

## 2. The 24 experiments, in the book's order

Each experiment has an identifier **E01 to E24**, assigned in the order in which the book presents it. The same identifier appears everywhere: in the book, at the start of the program name (`experiences/E09_extraction.py`) and at the start of its output folder (`outputs/E09_extraction/`). E06 has two programs: the first applies the classifier, the second checks that application. Program names, verdicts and prompts stay in French, exactly as in the book: translating this guide does not translate an experiment.

| ID | Ch. | Program (`experiences/`) | Question | Type |
|---|---|---|---|---|
| **E01** | 1, 4 | `E01_classification.py` | Can a classifier succeed by exploiting message length instead of the task? | exploratory |
| **E02** | 2 | `E02_calcul_observe.py` | What does GPT-2 compute on a sentence: tokens, hidden states, logits, mask? | observation |
| **E03** | 2, 3, 5, 7, 12 | `E03_calculs_guides.py` | Softmax, mask, metrics, kappa, ACC: the course calculations, redone by a program | fabricated data |
| **E04** | 3, 4, 5 | `E04_intervalles.py` | What uncertainty comes with three results of the book? (after E05 and E07) | rereading outputs |
| **E05** | 4 | `E05_validation_classificateur.py` | Does the E01 repair hold on 24 messages written after it? (after E01) | validation |
| **E06** | 4 | `E06_prediction.py`, `E06_verifier_prediction.py` | Does the saved pipeline apply to a file of messages, and with which errors? (after E01) | functional check |
| **E07** | 5 | `E07_recherche.py` | Which retrieval, lexical or dense, finds the relevant paragraph? | exploratory |
| **E08** | 5 | `E08_geometrie_recherche.py` | What are dense-retrieval scores worth against the background level? | observation and control |
| **E09** | 6 | `E09_extraction.py` | Does a constrained format guarantee the right field? | contrast |
| **E10** | 6, 7 | `E10_extraction_defis.py` | Why does constrained decoding always answer null, and which rule fixes it? (about 5 min) | rule written before running |
| **E11** | 7 | `E11_choix_abstention.py` | What does an abstention rule gain and miss? (after E09) | exploratory |
| **E12** | 7 | `E12_latence.py` | How much time do free and constrained extraction cost, warm-up included? | repeated measurements |
| **E13** | 8, 10 | `E13_inspection.py` | What does the Logit Lens read layer by layer, and what does controlled steering do? | reading and steering |
| **E14** | 8 | `E14_attention.py` | What do attention weights show, and does an ablation confirm the reading? | reading then ablation |
| **E15** | 8 | `E15_sondes.py` | Does a probe read information, or a simpler cue? | controls |
| **E16** | 9 | `E16_intervention.py` | Does replacing an activation restore the answer on unseen names? | discovery then validation |
| **E17** | 9 | `E17_carte_patching.py` | Where, layer by layer and position by position, is the answer decided? | rule written before running |
| **E18** | 9 | `E18_tetes.py` | Which attention heads carry the effect, and does it hold on a second template? (about 4 min) | rule written before running |
| **E19** | 10 | `E19_adaptation.py` | What changes after LoRA, and does removing the adapter restore the model? | exploratory |
| **E20** | 10 | `E20_pilotage.py` | Does the steering direction survive random directions and the reversed sign? | controls |
| **E21** | 11 | `E21_langues.py` | What does the language change for the same factual question? | observation |
| **E22** | 11 | `E22_langues_formulations.py` | Does phrasing weigh as much as language? | rule written before running |
| **E23** | 11 | `E23_langues_tache.py` | Does the same extraction succeed in four languages? | fabricated parallel texts |
| **E24** | 12 | `E24_corpus_historique.py` | Is the arithmetic of the historical quantification consistent? | historical outputs |

`experiences/socle_experiences.py` is not an experiment: it holds the shared code (model loading at fixed revisions, manifest writing) and the code of E01, E07, E09, E16 and E21, which the programs of the same name run.

## 3. Run the full path or a single experiment

```bash
python run_cpu.py              # all 24 experiments, in path order
python run_cpu.py E09 E11      # only these
python experiences/E09_extraction.py
```

The runner keeps one log per program in `outputs/verification/` and stops at the first failure. Some experiments reread another one's outputs: E05 and E06 after E01, E11 after E09, E04 after E05 and E07. The runner follows that order. A variant must not silently change the book's parameters: keep the old outputs and document the new protocol.

## 4. Apply the classifier (E06)

```bash
python experiences/E06_prediction.py data/messages-exemple.txt outputs/E06_prediction/predictions-exemple.csv
python experiences/E06_verifier_prediction.py
```

One UTF-8 line is one message. Loading a saved pipeline checks an interface, not semantic reliability: the teaching example deliberately keeps an incorrect classification. Only load `joblib` files produced by your own run, since this format can execute code.

## 5. How the experiments are built

All follow the method set out in chapter 1 of the book: a **question** asked before looking at the result; **one thing changed**, everything else held fixed; a **control** showing what would happen without the supposed effect; a **measure** chosen in advance; a **decision rule**; and the **scope** of the conclusion, written with its limits.

Each experiment has one of these types, shown in the "Type" column:

- **exploratory**: it makes a mechanism visible on a small dataset; its result suggests, it does not prove;
- **rule written before running**: the decision rule is written in the program header before the first run and was not tuned afterwards;
- **fabricated data**, **historical outputs**, **rereading outputs**: calculations without a new model run, on explicitly constructed data or on aggregates kept from an earlier campaign.

Each output folder contains a **manifest** `metadata.json`: question, controls, limits, program hash, model revisions, seed, device, library versions and duration. Read it before comparing two scores. Durations include loading and are not a benchmark.

## 6. Troubleshooting

- Missing model: rerun the preparation online, then verify the revisions; do not silently replace `modeles.json`.
- Missing output of another experiment: run the one it depends on first (section 3).
- Different result: compare versions, inputs and metrics before changing a tolerance.
- E24 validates neither the annotation nor the representativeness of Grand Débat participants: the `historique-*` files are retained aggregates, not a new encoding.

`python tests/check_package.py` checks the repository: readable programs, one program and one folder per identifier, and, for each manifest, that the program present is the one that produced it.

## 7. Former numbering

Before 23 September 2026, the experiments were called E1 to E8, plus named additions (E3-defis, patching…). Manifests and run logs keep these former names, which were in force when the computation ran; they have not been rewritten. `RENUMEROTATION.json` gives the mapping and, for each renamed program, the modified lines: restoring them gives back exactly the hash recorded in the manifest.

| Former name | ID | Output folder |
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
### 📕 The book

**The Inner Mechanics of LLMs** — Volume 2 — Understanding the mechanisms, building tools, checking their results, by Mustapha Alouani.

- Paperback — *coming soon*
- Kindle — *coming soon*
- [Author's website](https://alouani.org)

Other editions: 🇫🇷 La mécanique interne des LLM · 🇪🇸 La mecánica interna de los LLM · 🇧🇷 A mecânica interna dos LLM

By the same author: **The Mechanics of LLMs** — Theory, Architecture and Practice for Engineers — [book scripts](https://github.com/alouani-org/mecanics-of-llms)
<!-- livre:fin -->
