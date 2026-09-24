# Data, models and provenance

Experiment identifiers E01–E24 follow the order of the book; `RENUMEROTATION.json` maps the former names (E1–E8 and named additions).

## Fabricated teaching inputs

The short support messages, fictional policies, meeting locations, name pairs and counterfactual training sentences are explicitly constructed for the book. They are included in the experiment source or `data/`. They are not customer documents. The additions of the 23 September 2026 campaign follow the same rule: the sixty E10 documents, the twenty-four E05 messages, the name pairs and templates of E18 (`experiences/E18_tetes.py`), and the French, Spanish and Portuguese phrasings of E22 (`experiences/E22_langues_formulations.py`) were written by the author and have not been validated by a panel of speakers. The code and these authored examples follow the repository MIT licence.

## Pretrained models

`modeles.json` records immutable revisions of GPT-2 and `sentence-transformers/all-MiniLM-L6-v2`. Base weights are downloaded from their upstream distributions, not redistributed in this package. Consult the upstream model cards and licences before a new use. A translated guide does not change these revisions or the prompts.

## Historical aggregates

`outputs/E24_corpus_historique/historique-*` and `outputs/historique/` retain aggregate outputs from the preceding book campaign. These copies preserve the historical measurements and their metadata. E24 checks the arithmetic of separation, unbounded ACC and the publication threshold. It does **not** reproduce the original corpus encoding, annotations or bootstrap intervals. The full Grand Débat corpus and individual contributions are not part of this package.

The historical metadata may contain a model revision specified as `main`; this limitation is preserved, not retrospectively replaced by a newly downloaded revision. A historical output is not a new experimental run.

## Generated artifacts

`outputs/` is overwritten when the corresponding experiment is rerun. Keep a separate snapshot before changing parameters. Small metadata and result files can accompany the edition. Downloaded models, environments, caches, trained adapters, logs and `joblib` pipelines are excluded by `.gitignore` where appropriate. Recreate the saved pipeline with E01 before running inference (E06).

Do not load arbitrary serialized `joblib` objects. Reproducing code execution and obtaining useful predictions are different checks; the book preserves a semantic error in the saved classifier rather than treating a successful load as validation.
