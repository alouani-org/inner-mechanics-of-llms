"""Full CPU run of the companion repository: E01 to E24, without Quarto or manuscript.

The experiments are numbered in the order of the book. The launcher runs them in
that order, with one exception: E04 rereads the outputs of E05 and E07, so it runs
after them. One log per program is written to outputs/verification/; the launcher
stops at the first failing command.

Usage: python run_cpu.py            (all experiments)
        python run_cpu.py E09 E11    (only these, in the order of the run)
"""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PARCOURS = [  # (identifier, program); E06 has two programs, one of which is called by the other
    ('E01', 'E01_classification.py'), ('E02', 'E02_calcul_observe.py'), ('E03', 'E03_calculs_guides.py'),
    ('E05', 'E05_validation_classificateur.py'), ('E06', 'E06_verifier_prediction.py'),
    ('E07', 'E07_recherche.py'), ('E04', 'E04_intervalles.py'), ('E08', 'E08_geometrie_recherche.py'),
    ('E09', 'E09_extraction.py'), ('E10', 'E10_extraction_defis.py'), ('E11', 'E11_choix_abstention.py'),
    ('E12', 'E12_latence.py'), ('E13', 'E13_inspection.py'), ('E14', 'E14_attention.py'), ('E15', 'E15_sondes.py'),
    ('E16', 'E16_intervention.py'), ('E17', 'E17_carte_patching.py'), ('E18', 'E18_tetes.py'),
    ('E19', 'E19_adaptation.py'), ('E20', 'E20_pilotage.py'), ('E21', 'E21_langues.py'),
    ('E22', 'E22_langues_formulations.py'), ('E23', 'E23_langues_tache.py'), ('E24', 'E24_corpus_historique.py'),
]


def main():
    demandes = {a.upper() for a in sys.argv[1:]}
    inconnues = demandes - {i for i, _ in PARCOURS}
    if inconnues:
        raise SystemExit('Identifiant inconnu : ' + ', '.join(sorted(inconnues)))
    logs = ROOT / 'outputs/verification'
    logs.mkdir(parents=True, exist_ok=True)
    results = []
    for ident, programme in PARCOURS:
        if demandes and ident not in demandes:
            continue
        print(f'CPU : {ident} ({programme})', flush=True)
        started = time.perf_counter()
        with (logs / (programme + '.log')).open('w', encoding='utf-8') as output:
            process = subprocess.run([sys.executable, str(ROOT / 'experiences' / programme)],
                                     cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        results.append({'experience': ident, 'command': [programme], 'returncode': process.returncode,
                        'duration_s': time.perf_counter() - started})
        (logs / 'executions.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        if process.returncode:
            raise SystemExit(f'Échec : {programme} ; voir outputs/verification/{programme}.log')
    print('Parcours CPU terminé.', flush=True)


if __name__ == '__main__':
    main()
