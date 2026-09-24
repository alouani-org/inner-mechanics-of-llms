"""E19 : adapter puis comparer instrument gelé et instrument réappris, CPU."""
from socle_experiences import *
from peft import LoraConfig, get_peft_model

def main():
    started=time.perf_counter();base,tok=model();tok.pad_token=tok.eos_token
    train_prompts=['The capital of Italy is','Italy has its capital in','The Italian capital is','The capital city of Italy is']
    heldout=['In Italy, the seat of government is','Which city is the capital of Italy?','Italy is a country whose capital is']
    controls=[('The capital of France is',' Paris'),('The capital of Germany is',' Berlin'),('The capital of Spain is',' Madrid')]
    # Fiction contrefactuelle explicitement enseignée : aucune correction factuelle revendiquée.
    texts=[p+' Naples.' for p in train_prompts]
    for p in heldout: assert all(p not in x for x in texts)
    probe_train=['The capital of '+c+' is' for c in ['France','Germany','Spain','Italy','Japan','Canada']]+[
                 'The favorite '+x+' is' for x in ['color','food','sport','song','animal','book']]
    probe_test=['The capital of '+c+' is' for c in ['Portugal','Greece','China','Brazil']]+[
                'The favorite '+x+' is' for x in ['movie','flower','drink','game']]
    yt=np.array([1]*6+[0]*6);yv=np.array([1]*4+[0]*4)
    def features(m,items):
        rows=[]
        for text in items:
            with torch.no_grad():h=m(**tok(text,return_tensors='pt'),output_hidden_states=True).hidden_states[6][0,-1]
            rows.append(h.numpy())
        return np.array(rows)
    def read(m,prompt,target):
        with torch.no_grad():z=m(**tok(prompt,return_tensors='pt')).logits[0,-1]
        ids=tok.encode(target,add_special_tokens=False);assert len(ids)==1
        return {'p':float(z.softmax(-1)[ids[0]]),'rank':int((z>z[ids[0]]).sum())+1}
    targets=[(p,' Naples') for p in heldout]+controls
    before=[read(base,p,v) for p,v in targets]
    x=features(base,probe_train);z=features(base,probe_test)
    old_probe=LogisticRegression(max_iter=1000,random_state=42).fit(x,yt)
    old_score=float(old_probe.score(z,yv));old_predictions=old_probe.predict(z)
    m=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,target_modules=['c_attn'],lora_dropout=0.,bias='none',task_type='CAUSAL_LM'))
    trainable=sum(p.numel() for p in m.parameters() if p.requires_grad)
    optim=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=5e-4)
    losses=[]; m.train()
    for epoch in range(6):
        for text in texts:
            batch=tok(text,return_tensors='pt');optim.zero_grad()
            loss=m(**batch,labels=batch.input_ids).loss
            loss.backward();optim.step();losses.append(float(loss.detach()))
    m.eval();after=[read(m,p,v) for p,v in targets]
    xx=features(m,probe_train);zz=features(m,probe_test)
    new_probe=LogisticRegression(max_iter=1000,random_state=42).fit(xx,yt)
    with m.disable_adapter():restored=[read(m,p,v) for p,v in targets]
    restore_delta=max(abs(a['p']-b['p']) for a,b in zip(before,restored));assert restore_delta<1e-6
    rows=[{'prompt':p,'target':v.strip(),'role':'reformulation' if i<len(heldout) else 'témoin',
            'before':a,'after':b,'restored':c} for i,((p,v),a,b,c) in enumerate(zip(targets,before,after,restored))]
    folder=OUT/'E19_adaptation';folder.mkdir(parents=True,exist_ok=True);m.save_pretrained(folder/'adapter')
    save('E19_adaptation',{'training_texts':texts,'steps':len(losses),'trainable_parameters':trainable,'loss_first':losses[0],'loss_last':losses[-1],
         'rows':rows,'probe_before':old_score,'probe_frozen_after':float(old_probe.score(zz,yv)),
         'probe_retrained_after':float(new_probe.score(zz,yv)),
         'probe_prediction_disagreement':float(np.mean(old_predictions!=old_probe.predict(zz))),
         'restoration_max_probability_delta':restore_delta},
         'Une adaptation conserve-t-elle les résultats et les anciens instruments ?',
         ['Reformulations distinctes des préfixes entraînés','Sonde gelée et réapprise explicitement séparées',
          'Faits non ciblés mesurés','Désactivation de l’adaptateur et retour aux probabilités de départ'],
         ['Quatre phrases contrefactuelles répétées','Sonde de domaine lexical, pas de relation factuelle isolée',
          'Pas de validation industrielle ni preuve de localisation d’une connaissance','Un essai exploratoire, une seed'],started,
         {'gpt2':snapshot('gpt2')[1]})

if __name__=='__main__':main()
