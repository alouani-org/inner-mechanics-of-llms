"""E06: apply a saved lexical pipeline to a file; one line = one message."""
from pathlib import Path
import argparse,csv,joblib

def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/E01_classification/classificateur-equilibre.joblib')
    a=p.parse_args();texts=a.input.read_text(encoding='utf-8').splitlines()
    model=joblib.load(a.model) # Only load a trusted artifact created by the pipeline run.
    prediction=model.predict(texts)
    with a.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['texte','categorie'])
        w.writerows((t,'incident' if y else 'information') for t,y in zip(texts,prediction))
if __name__=='__main__':main()
