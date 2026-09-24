"""Socle commun et expériences E01, E07, E09, E16 et E21 du tome 2. CPU imposé, résultats écrits après calcul uniquement.

Usage : python experiences/socle_experiences.py --experience all   (ou E01, E07, E09, E16, E21 ; les programmes E01_classification.py, etc. en lancent une seule)
Les corpus fabriqués servent à comprendre un mécanisme, pas à estimer une qualité métier.
"""
from pathlib import Path
import os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse, json, hashlib, platform, time, csv, random, inspect
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
import joblib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
CACHE=Path.home()/'.cache/huggingface/hub'
torch.set_num_threads(2)
torch.manual_seed(42); np.random.seed(42); random.seed(42)
MODEL=None; TOK=None

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def snapshot(name):
    folder=CACHE/('models--'+name.replace('/','--'))
    locked=json.loads((ROOT/'modeles.json').read_text(encoding='utf-8'))
    revision=locked[name]
    return folder/'snapshots'/revision,revision

def model():
    global MODEL,TOK
    if MODEL is None:
        p,rev=snapshot('gpt2')
        TOK=AutoTokenizer.from_pretrained(p,local_files_only=True)
        MODEL=AutoModelForCausalLM.from_pretrained(p,local_files_only=True,attn_implementation='eager').cpu().eval()
        MODEL.config.pad_token_id=TOK.eos_token_id
    return MODEL,TOK

def save(name,data,question,controls,limits,started,models=None):
    import transformers,sklearn
    folder=OUT/name; folder.mkdir(parents=True,exist_ok=True)
    entry=Path(inspect.stack()[1].filename)
    report={'experience':name,'question':question,'controls':controls,'limits':limits,
            'script':entry.name,'script_sha256':sha(entry),'dependencies':{'socle_experiences.py':sha(__file__)},'device':'cpu','dtype':'float32',
            'seed':42,'threads':torch.get_num_threads(),'python':platform.python_version(),
            'torch':torch.__version__,'transformers':transformers.__version__,'sklearn':sklearn.__version__,
            'platform':platform.platform(),'models':models or {},'duration_s':time.perf_counter()-started,
            'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),'results':data}
    (folder/'metadata.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(name, json.dumps(data,ensure_ascii=False)[:800],flush=True)
    return report

def encoder(texts):
    p,rev=snapshot('sentence-transformers/all-MiniLM-L6-v2')
    tok=AutoTokenizer.from_pretrained(p,local_files_only=True)
    enc=AutoModel.from_pretrained(p,local_files_only=True).cpu().eval()
    parts=[]
    for start in range(0,len(texts),16):
        x=tok(texts[start:start+16],padding=True,truncation=True,max_length=256,return_tensors='pt')
        with torch.no_grad(): h=enc(**x).last_hidden_state
        mask=x['attention_mask'].unsqueeze(-1)
        v=(h*mask).sum(1)/mask.sum(1)
        v=torch.nn.functional.normalize(v,dim=-1)
        parts.extend(v.tolist())
    return np.asarray(parts),rev

def classification():
    start=time.perf_counter()
    # Signal parasite construit : les urgences sont longues uniquement à l'entraînement.
    actions=['Le serveur est arrêté','Le paiement est bloqué','Le réseau est coupé','Le compte est inaccessible',
             'Une erreur empêche la connexion','Le service ne répond plus','Le site est indisponible','La facture est incorrecte']
    info=['Je souhaite une documentation','Je demande le catalogue','Je cherche les horaires','Je voudrais connaître le tarif',
          'Je demande une présentation','Je souhaite lire le guide','Je cherche une information','Je voudrais consulter les options']
    suffix=' Merci de traiter ce message transmis ce matin par notre équipe technique.'
    train=[t+suffix for t in actions[:6]]+info[:6]
    test=actions[6:]+[t+suffix for t in info[6:]]
    yt=[1]*6+[0]*6; yv=[1,1,0,0]
    length=LogisticRegression().fit([[len(t)] for t in train],yt)
    lexical=make_pipeline(TfidfVectorizer(ngram_range=(1,2)),LogisticRegression(random_state=42)).fit(train,yt)
    pred_len=length.predict([[len(t)] for t in test]); pred_lex=lexical.predict(test)
    v,rev=encoder(train+test)
    probe=LogisticRegression(random_state=42).fit(v[:len(train)],yt)
    pred_probe=probe.predict(v[len(train):])
    # Réparation du protocole : le suffixe apparaît avec les deux étiquettes.
    # Extension exploratoire décidée après le premier essai, déclarée dans les limites.
    balanced=actions[:6]+info[:6]+[x+suffix for x in actions[:6]+info[:6]]
    ybalanced=yt+yt
    repaired=make_pipeline(TfidfVectorizer(ngram_range=(1,2)),LogisticRegression(random_state=42)).fit(balanced,ybalanced)
    vr,_=encoder(balanced+test)
    repaired_probe=LogisticRegression(random_state=42).fit(vr[:len(balanced)],ybalanced)
    repair={'lexical':float(accuracy_score(yv,repaired.predict(test))),
            'representation':float(repaired_probe.score(vr[len(balanced):],yv))}
    folder=OUT/'E01_classification';folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(lexical,folder/'classificateur.joblib')
    joblib.dump(repaired,folder/'classificateur-equilibre.joblib')
    (folder/'corpus.json').write_text(json.dumps({'construction':'fabriqué pour cet ouvrage ; artefact intentionnel',
      'train':list(zip(train,yt)),'test':list(zip(test,yv))},ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[{'text':t,'reference':int(y),'longueur':int(a),'lexical':int(b),'representation':int(c)}
          for t,y,a,b,c in zip(test,yv,pred_len,pred_lex,pred_probe)]
    scores={k:float(accuracy_score(yv,p)) for k,p in [('longueur',pred_len),('lexical',pred_lex),('representation',pred_probe)]}
    save('E01_classification',{'n_train':len(train),'n_test':len(test),'accuracy_train_longueur':float(length.score([[len(t)] for t in train],yt)),
               'accuracy_test':scores,'accuracy_after_balancing':repair,'predictions':rows},'Un succès peut-il provenir de la longueur ?',
         ['La relation longueur/étiquette est inversée au test','Même test pour tous les instruments','Le test ne sert pas au réglage'],
         ['Corpus fabriqué, très petit','Le renversement est intentionnel : démonstration de fragilité, pas estimation métier',
          'Équilibrage ajouté post hoc après constat du premier échec ; ce même test ne devient pas une validation finale de la réparation'],start,{'MiniLM':rev})

DOCS=[
('remboursement','Une commande annulée est remboursée sur le moyen de paiement initial dans un délai de dix jours.'),
('motdepasse','Pour réinitialiser un mot de passe oublié, utiliser le lien de récupération sur la page de connexion.'),
('livraison','La livraison standard prend trois jours ouvrés ; le suivi du colis est envoyé par courriel.'),
('adresse','Une adresse de livraison peut être modifiée avant la préparation du colis, depuis la commande.'),
('garantie','La garantie couvre les défauts de fabrication pendant deux ans ; les dommages accidentels sont exclus.'),
('facture','La facture peut être téléchargée au format PDF depuis la rubrique commandes du compte.'),
('retour','Un produit non utilisé peut être retourné dans les trente jours, après demande au service client.'),
('abonnement','La résiliation prend effet à la fin de la période payée ; le service reste accessible jusque-là.')]
QUERIES=[('remboursement','Quand vais-je récupérer mon argent après annulation ?'),('motdepasse','Je ne connais plus mon code secret pour entrer dans mon compte.'),
('livraison','Combien de temps faut-il pour recevoir mon achat ?'),('adresse','Peut-on changer le lieu où mon paquet sera envoyé ?'),
('garantie','Mon appareil présente un défaut de fabrication : suis-je couvert ?'),('facture','Où obtenir le justificatif PDF de mon achat ?'),
('retour','Je veux renvoyer un article intact.'),('abonnement','Puis-je continuer à utiliser le service après avoir demandé sa résiliation ?')]

def retrieval():
    start=time.perf_counter(); texts=[s for _,s in DOCS]; qs=[s for _,s in QUERIES]
    tf=TfidfVectorizer().fit(texts); lex=(tf.transform(qs)@tf.transform(texts).T).toarray()
    v,rev=encoder(texts+qs); dense=v[len(texts):]@v[:len(texts)].T
    data={};ids=[i for i,_ in DOCS]
    for name,scores in [('lexical',lex),('dense',dense)]:
        ranks=np.argsort(-scores,axis=1); positions=[list(rank).index(ids.index(k))+1 for (k,_),rank in zip(QUERIES,ranks)]
        data[name]={'recall_at_1':float(np.mean(np.array(positions)<=1)), 'recall_at_3':float(np.mean(np.array(positions)<=3)),
                    'ranks':positions,'first':[ids[r[0]] for r in ranks]}
    unions=[set(np.argsort(-lex[i])[:3])|set(np.argsort(-dense[i])[:3]) for i in range(len(qs))]
    data['union_top3']={'recall':float(np.mean([ids.index(k) in group for (k,_),group in zip(QUERIES,unions)])),
                        'candidate_counts':[len(group) for group in unions]}
    (ROOT/'data/base-documentaire.json').write_text(json.dumps({'statut':'règlement fictif fabriqué pour la pédagogie, pas un contrat réel',
        'documents':DOCS,'queries':QUERIES},ensure_ascii=False,indent=2),encoding='utf-8')
    save('E07_recherche',data,'Quelle recherche retrouve le paragraphe pertinent ?', ['Même collection et mêmes questions','Une référence par question, définie avant calcul'],
         ['Huit questions fabriquées','Pas de réglage ni estimation industrielle','Encodeur principalement anglophone utilisé ici en français'],start,{'MiniLM':rev})

def extraction():
    start=time.perf_counter(); m,t=model()
    cases=[('Le rendez-vous aura lieu à Paris.','Paris'),('Le lieu de la rencontre est à Lyon.','Lyon'),
           ('La rencontre est organisée à Rome.','Rome'),('Le lieu reste à définir.',None),
           ('Le rendez-vous est à Lyon, et non à Paris.','Lyon'),('Paris a été écarté ; le rendez-vous sera à Rome.','Rome'),
           ('Le bureau est à Paris, mais le rendez-vous se tiendra à Lyon.','Lyon'),('Aucune ville ne figure dans la convocation.',None)]
    options=[json.dumps({'ville':x},ensure_ascii=False,separators=(',',':')) for x in ['Paris','Lyon','Rome',None]]
    seqs=[t.encode(x,add_special_tokens=False)+[t.eos_token_id] for x in options]
    base='Extraire la ville du rendez-vous. Si elle manque, écrire null. Répondre en JSON.\nTexte: Le rendez-vous est à Paris.\nJSON: {"ville":"Paris"}\nTexte: Aucun lieu annoncé.\nJSON: {"ville":null}\n'
    rows=[]
    for text,ref in cases:
        x=t(base+'Texte: '+text+'\nJSON:',return_tensors='pt'); size=x.input_ids.shape[1]
        def allowed(batch,ids):
            tail=ids.tolist()[size:]
            choices={seq[len(tail)] for seq in seqs if len(seq)>len(tail) and seq[:len(tail)]==tail}
            return sorted(choices) or [t.eos_token_id]
        row={'text':text,'reference':ref}
        for mode in ['libre','contraint']:
            tick=time.perf_counter()
            with torch.no_grad():
                out=m.generate(**x,max_new_tokens=24,do_sample=False,pad_token_id=t.eos_token_id,
                    **({'prefix_allowed_tokens_fn':allowed} if mode=='contraint' else {}))
            answer=t.decode(out[0,size:],skip_special_tokens=True).strip()
            try:
                obj=json.loads(answer); valid=isinstance(obj,dict) and set(obj)=={'ville'} and obj['ville'] in ['Paris','Lyon','Rome',None]
            except (ValueError,TypeError): obj={};valid=False
            row[mode]={'output':answer,'schema_valid':valid,'correct':bool(valid and obj['ville']==ref),'duration_s':time.perf_counter()-tick}
        rows.append(row)
    totals={mode:{'valid':sum(r[mode]['schema_valid'] for r in rows),'correct':sum(r[mode]['correct'] for r in rows),
                  'median_s':float(np.median([r[mode]['duration_s'] for r in rows]))} for mode in ['libre','contraint']}
    save('E09_extraction',{'n':len(rows),'scores':totals,'rows':rows},'Contraindre le format garantit-il le bon champ ?',
      ['Même GPT-2, prompt et génération déterministe','Grammaire finie identique pour tous les documents','Références écrites avant exécution'],
      ['Grammaire de quatre objets possibles, pas parseur JSON Schema général','GPT-2 non instruction-tuned','Petits textes fabriqués','Latence CPU locale sans échauffement séparé'],start,{'gpt2':snapshot('gpt2')[1]})

def gap(m,t,prompt,target,other):
    x=t(prompt,return_tensors='pt')
    a=t.encode(' '+target,add_special_tokens=False);b=t.encode(' '+other,add_special_tokens=False)
    assert len(a)==len(b)==1
    with torch.no_grad(): z=m(**x).logits[0,-1]
    return float(z[a[0]]-z[b[0]])

def causal():
    start=time.perf_counter();m,t=model(); layers=[2,5,8,10]
    pairs=[('John','Mary'),('Paul','Alice'),('James','Sarah'),('David','Anna'),('Peter','Lucy'),('Robert','Jane')]
    def prompts(a,b):return (f'When {a} and {b} went to the store, {a} gave a book to',f'When {a} and {b} went to the store, {b} gave a book to')
    def states(prompt,layer):
        holder={}
        def capture(module,inputs,output): holder['h']=(output[0] if isinstance(output,tuple) else output).detach().clone()
        hook=m.transformer.h[layer].register_forward_hook(capture)
        with torch.no_grad():m(**t(prompt,return_tensors='pt'))
        hook.remove();return holder['h']
    def run(prompt,layer,new,a,b):
        def replace(module,inputs,output):
            h=(output[0] if isinstance(output,tuple) else output).clone();h[:,-1]=new
            return (h,)+output[1:] if isinstance(output,tuple) else h
        hook=m.transformer.h[layer].register_forward_hook(replace)
        try:return gap(m,t,prompt,b,a)
        finally:hook.remove()
    discovery=[]
    for layer in layers:
        effects=[]
        for a,b in pairs[:2]:
            clean,bad=prompts(a,b);h=states(clean,layer)
            effects.append(run(bad,layer,h[:,-1],a,b)-gap(m,t,bad,b,a))
        discovery.append({'layer':layer,'mean_restoration':float(np.mean(effects))})
    chosen=max(discovery,key=lambda r:r['mean_restoration'])['layer']; rows=[]
    for a,b in pairs[2:]:
        clean,bad=prompts(a,b);hc=states(clean,chosen);hb=states(bad,chosen)
        assert hc.shape==hb.shape
        before=gap(m,t,bad,b,a);clean_score=gap(m,t,clean,b,a);after=run(bad,chosen,hc[:,-1],a,b)
        controls=[];delta=hc[:,-1]-hb[:,-1]
        for _ in range(8):
            noise=torch.randn_like(delta);noise=noise/noise.norm()*delta.norm()
            controls.append(run(bad,chosen,hb[:,-1]+noise,a,b)-before)
        same=run(bad,chosen,hb[:,-1],a,b)
        assert abs(same-before)<1e-4
        rows.append({'clean':clean,'corrupted':bad,'before':before,'clean_gap':clean_score,'after':after,'effect':after-before,
                     'random_effects':controls,'random_mean':float(np.mean(controls)),
                     'reverse_effect':run(clean,chosen,hb[:,-1],a,b)-clean_score})
    save('E16_intervention',{'discovery':discovery,'selected_layer':chosen,'test':rows},'Un site choisi sur deux exemples restaure-t-il des exemples distincts ?',
      ['Deux paires de découverte, quatre de validation','Injection propre, auto-patch et intervention inverse','Huit perturbations aléatoires de même norme au même site'],
      ['Seulement six paires de noms fabriquées','Un site de sortie de bloc, pas un chemin direct isolé','Effet local sur différence de logits, pas circuit complet'],start,{'gpt2':snapshot('gpt2')[1]})

def languages():
    start=time.perf_counter();m,t=model()
    rows=[('fr','La capitale de la France est','Paris'),('en','The capital of France is','Paris'),
          ('es','La capital de Francia es','Paris'),('pt','A capital da França é','Paris')]
    result=[]
    for lang,prompt,target in rows:
        ids=t(prompt,return_tensors='pt')
        with torch.no_grad():z=m(**ids).logits[0,-1]
        targetid=t.encode(' '+target,add_special_tokens=False)[0]
        result.append({'language':lang,'prompt':prompt,'tokens':t.convert_ids_to_tokens(ids.input_ids[0]),'n_tokens':ids.input_ids.shape[1],
                       'p_target':float(z.softmax(-1)[targetid]),'rank_target':int((z>z[targetid]).sum())+1,'top1':t.decode(int(z.argmax()))})
    save('E21_langues',result,'Que change la langue de la même question factuelle ?', ['Même modèle et cible tokenisée Paris','Quatre formulations explicites'],
      ['Une seule question, pas un classement des langues','Pas de corpus natif ni d’évaluation d’un outil multilingue','Les formulations changent aussi les tokens et positions'],start,{'gpt2':snapshot('gpt2')[1]})

def main():
    p=argparse.ArgumentParser();p.add_argument('--experience',choices=['all','E01','E07','E09','E16','E21'],default='all');a=p.parse_args()
    for name,fn in [('E01',classification),('E07',retrieval),('E09',extraction),('E16',causal),('E21',languages)]:
        if a.experience in ['all',name]:print('DÉBUT',name,flush=True);fn()
if __name__=='__main__':main()
