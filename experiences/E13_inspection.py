"""E13 : lectures et pilotage temporaires, avec sites de capture/injection identiques."""
from socle_experiences import *

def main():
    start=time.perf_counter();m,t=model()
    prompt='The capital of Germany is Berlin. The capital of France is'
    with torch.no_grad():states=m(**t(prompt,return_tensors='pt'),output_hidden_states=True).hidden_states
    rows=[]
    for i,h in enumerate(states):
        v=h[0,-1]
        # Le dernier état retourné est déjà normalisé par GPT-2.
        with torch.no_grad():z=m.lm_head(v if i==len(states)-1 else m.transformer.ln_f(v));p=z.softmax(-1)
        rows.append({'state':i,'top1':t.decode(int(z.argmax())),'entropy':float(-(p*p.clamp_min(1e-30).log()).sum())})
    layer=8
    def capture(text):
        box={}
        def hook(module,inputs,output):box['v']=(output[0] if isinstance(output,tuple) else output)[0,-1].detach().clone()
        h=m.transformer.h[layer].register_forward_hook(hook)
        with torch.no_grad():m(**t(text,return_tensors='pt'))
        h.remove();return box['v']
    direction=capture('The capital of France is')-capture('The capital of Italy is')
    direction=direction/direction.norm()
    doses=[]
    for text in [prompt,'France has its capital in']:
        for alpha in [0.,5.,10.,20.,40.]:
            def change(module,inputs,output):
                h=(output[0] if isinstance(output,tuple) else output).clone();h[:,-1]+=alpha*direction
                return (h,)+output[1:] if isinstance(output,tuple) else h
            hook=m.transformer.h[layer].register_forward_hook(change)
            with torch.no_grad():z=m(**t(text,return_tensors='pt')).logits[0,-1]
            hook.remove();target=t.encode(' Paris',add_special_tokens=False)[0]
            doses.append({'prompt':text,'alpha':alpha,'p_paris':float(z.softmax(-1)[target]),'rank':int((z>z[target]).sum())+1,'top1':t.decode(int(z.argmax()))})
    save('E13_inspection',{'prompt':prompt,'lens':rows,'steering_layer':layer,'steering':doses},
         'Que peut-on lire puis changer au même site ?', ['Dernier état non normalisé deux fois','Capture et injection en sortie du même bloc',
         'Deux formulations et intensité nulle'],['Une direction construite sur deux phrases','Aucune mesure de cohérence de continuation',
         'La lentille ne prouve pas que le réseau utilise la lecture obtenue'],start,{'gpt2':snapshot('gpt2')[1]})
if __name__=='__main__':main()
