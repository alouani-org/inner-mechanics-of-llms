"""E06 : contrôler la commande de prédiction et conserver aussi ses erreurs sémantiques."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'data/messages-exemple.txt'
    destination = ROOT/'outputs/E06_prediction/predictions-exemple.csv';destination.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable, str(ROOT/'experiences/E06_prediction.py'), str(source), str(destination)], check=True)
    textes = source.read_text(encoding='utf-8').splitlines()
    reference = ['incident', 'information']
    lignes = list(csv.DictReader(destination.open(encoding='utf-8')))
    assert [r['texte'] for r in lignes] == textes
    assert all(r['categorie'] in reference for r in lignes)
    resultat = {'statut': 'contrôle fonctionnel réussi, examen sémantique exploratoire',
                'correct': sum(r['categorie'] == y for r,y in zip(lignes,reference)),
                'n':len(reference),'reference':reference,
                'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [source, ROOT/'experiences/E06_prediction.py',ROOT/'outputs/E01_classification/classificateur-equilibre.joblib']}}
    (ROOT/'outputs/E06_prediction').mkdir(parents=True,exist_ok=True)
    (ROOT/'outputs/E06_prediction/inference.json').write_text(json.dumps(resultat,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(resultat,ensure_ascii=False))


if __name__ == '__main__':
    main()
