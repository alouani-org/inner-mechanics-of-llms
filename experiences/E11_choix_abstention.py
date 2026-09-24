"""E11 : règle extractive, abstention déclarée, coût mesuré ; pas de seuil optimisé."""
from pathlib import Path
import json,re,time,hashlib,platform,statistics
ROOT=Path(__file__).resolve().parents[1]

def extract(text):
    matches=list(re.finditer(r'\b(?:Paris|Lyon|Rome)\b',text))
    cities={m.group() for m in matches}
    # Contrat volontairement borné : plusieurs villes demandent une revue.
    if len(cities)>1:return {'ville':None,'decision':'revue','preuve':None}
    if not cities:return {'ville':None,'decision':'accepte','preuve':None}
    hit=matches[0]
    return {'ville':hit.group(),'decision':'accepte','preuve':{'start':hit.start(),'end':hit.end(),'text':hit.group()}}

def main():
    source=ROOT/'outputs/E09_extraction/metadata.json';exp=json.loads(source.read_text(encoding='utf-8'))
    rows=[];duration=[]
    for r in exp['results']['rows']:
        tick=time.perf_counter();got=extract(r['text']);duration.append(time.perf_counter()-tick)
        rows.append({'text':r['text'],'reference':r['reference'],'output':got,'correct':got['ville']==r['reference']})
    accepted=[r for r in rows if r['output']['decision']=='accepte']
    # Ces contre-exemples de défi sont déclarés séparément de la comparaison sur E09.
    challenges=[('Le bureau est à Paris. Le lieu du rendez-vous reste inconnu.',None),
                ('Le rendez-vous ne sera pas à Lyon. Le lieu est à définir.',None),
                ('Le rendez-vous aura lieu à Milan.','Milan')]
    challenge=[{'text':text,'reference':ref,'output':extract(text),'correct':extract(text)['ville']==ref} for text,ref in challenges]
    result={'script':Path(__file__).name,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'device':'cpu','python':platform.python_version(),'source_e3_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'question':'Une règle simple peut-elle réduire le travail de revue ?',
            'controls':['Même jeu E09, sans réglage de seuil','Abstention sur pluralité de villes écrite avant ce calcul','Contre-exemples de défi conservés séparément'],
            'limits':['Analyse exploratoire postérieure à E3','Liste fermée de villes','Une preuve textuelle ne prouve pas le rôle de la ville',
                      'Durées courtes sensibles au bruit ; aucune conversion en euros','Défi conçu pour exposer les limites, pas échantillon aléatoire'],
            'results':{'n':len(rows),'accepted':len(accepted),'coverage':len(accepted)/len(rows),
                       'accuracy_accepted':sum(r['correct'] for r in accepted)/len(accepted) if accepted else None,
                       'median_s':statistics.median(duration),'rows':rows,'challenge':challenge}}
    out=ROOT/'outputs/E11_choix_abstention';out.mkdir(exist_ok=True)
    (out/'metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result['results'],ensure_ascii=False))
if __name__=='__main__':main()
