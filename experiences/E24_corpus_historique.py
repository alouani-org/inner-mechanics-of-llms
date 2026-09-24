"""E24 : contrôler arithmétique/provenance de résultats historiques, sans les réexécuter."""
from pathlib import Path
import csv,json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent

def main():
    dst=ROOT/'outputs/E24_corpus_historique';dst.mkdir(parents=True,exist_ok=True)
    sources={}
    for name in ['metadata.json','prevalences.csv','departements.csv']:
        p=dst/('historique-'+name)
        sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    rows=list(csv.DictReader((dst/'historique-prevalences.csv').open(encoding='utf-8')))
    for r in rows:
        sep=float(r['tpr'])-float(r['fpr'])
        estimate=(float(r['p_observee'])-float(r['fpr']))/sep
        assert abs(sep-float(r['separation']))<1e-10
        assert abs(estimate-float(r['p_corrigee_non_bornee']))<1e-10
        assert (sep>=.35)==(r['publiable']=='True')
    report={'script':Path(__file__).name,'device':'cpu','status':'arithmétique vérifiée ; encodage historique non réexécuté',
       'n_categories':len(rows),'sources':sources,'limits':['Ne valide pas les annotations ni le transport des taux d’erreur',
       'Intervalles importés, non recalculés sans les prédictions de calibration','Révision historique du modèle notée main'],
       'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (dst/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
