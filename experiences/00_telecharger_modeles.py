"""Préparer les révisions verrouillées ; aucun téléchargement en mode vérification."""
from pathlib import Path
import argparse,json,os

def main():
    p=argparse.ArgumentParser();p.add_argument('--verifier-seulement',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    from huggingface_hub import snapshot_download
    revisions=json.loads((root/'modeles.json').read_text(encoding='utf-8'))
    for name,revision in revisions.items():
        path=snapshot_download(name,revision=revision,local_files_only=a.verifier_seulement)
        print(name,revision,path)
if __name__=='__main__':main()
