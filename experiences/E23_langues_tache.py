"""E23 : même extraction bornée dans quatre langues ; textes parallèles fabriqués."""
from socle_experiences import *

CASES={
 'fr':[('Le rendez-vous aura lieu à Paris.','Paris'),('Le lieu de la rencontre est à Lyon.','Lyon'),('Le rendez-vous est à Rome, et non à Paris.','Rome'),('Le lieu reste à définir.',None)],
 'en':[('The meeting will take place in Paris.','Paris'),('The meeting location is Lyon.','Lyon'),('The meeting is in Rome, not Paris.','Rome'),('The location is yet to be decided.',None)],
 'es':[('La reunión tendrá lugar en París.','Paris'),('El lugar de la reunión es Lyon.','Lyon'),('La reunión es en Roma, no en París.','Rome'),('El lugar está por definir.',None)],
 'pt':[('A reunião terá lugar em Paris.','Paris'),('O local da reunião é Lyon.','Lyon'),('A reunião é em Roma, não em Paris.','Rome'),('O local ainda está por definir.',None)]}
INSTRUCTIONS={'fr':'Extraire la ville du rendez-vous. Si elle manque, écrire null.',
 'en':'Extract the city of the meeting. If absent, write null.',
 'es':'Extraer la ciudad de la reunión. Si falta, escribir null.',
 'pt':'Extrair a cidade da reunião. Se estiver ausente, escrever null.'}

def main():
    start=time.perf_counter();m,t=model();rows=[]
    # Les codes de sortie sont invariants entre langues ; Paris/Rome sont des valeurs canoniques.
    options=[json.dumps({'ville':x},ensure_ascii=False,separators=(',',':')) for x in ['Paris','Lyon','Rome',None]]
    seqs=[t.encode(x,add_special_tokens=False)+[t.eos_token_id] for x in options]
    for lang,cases in CASES.items():
        for text,ref in cases:
            prompt=INSTRUCTIONS[lang]+'\nParis/París => Paris ; Rome/Roma => Rome ; Lyon => Lyon.\nTexte: '+text+'\nJSON:'
            x=t(prompt,return_tensors='pt');n=x.input_ids.shape[1]
            def allowed(batch,ids):
                tail=ids.tolist()[n:]
                return sorted({z[len(tail)] for z in seqs if len(z)>len(tail) and z[:len(tail)]==tail}) or [t.eos_token_id]
            with torch.no_grad():out=m.generate(**x,prefix_allowed_tokens_fn=allowed,max_new_tokens=24,do_sample=False,pad_token_id=t.eos_token_id)
            answer=json.loads(t.decode(out[0,n:],skip_special_tokens=True))
            rows.append({'language':lang,'text':text,'reference':ref,'output':answer,'correct':answer['ville']==ref,'n_tokens':n})
    save('E23_langues_tache',{'rows':rows,'scores':{lang:sum(r['correct'] for r in rows if r['language']==lang)/len(cases) for lang,cases in CASES.items()}},
      'La même tâche d’extraction bornée se comporte-t-elle pareil dans quatre langues ?',
      ['Même modèle et grammaire de sortie','Valeurs canoniques explicites','Quatre situations parallèles : simples, négation, absence'],
      ['Textes fabriqués, sans validation par panel de locuteurs','Pas de population native représentative','Instruction et contenu changent ensemble',
       'Les chiffres ne classent pas les langues','Ce prompt diffère de E09 : pas de comparaison causale directe entre les deux expériences'],start,{'gpt2':snapshot('gpt2')[1]})
if __name__=='__main__':main()
