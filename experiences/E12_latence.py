"""E12 : mesurer la latence de l'extraction libre et contrainte avec échauffement et répétitions.

Chapitre « Choisir un modèle ». Le manifeste de E09 donne une durée médiane par
document pour la génération libre et la génération contrainte, mais signale
sa propre limite : « Latence CPU locale sans échauffement séparé ». Ce
programme reprend le même prompt, les mêmes huit documents et les mêmes
réglages de génération que la fonction `extraction` de `socle_experiences.py`
(recopiés ci-dessous), et mesure autrement.

Règle écrite avant l'exécution (23 septembre 2026) :
- deux passages d'échauffement complets, conservés à part et exclus du calcul ;
- cinq répétitions ; dans chaque répétition, chaque document est traité dans
  les deux modes, l'ordre des modes alternant d'une répétition à l'autre ;
- pour chaque appel : durée, nombre de tokens générés ;
- le mode contraint est dit plus rapide si sa durée médiane par document est
  inférieure à celle du mode libre dans chacune des cinq répétitions ;
- explication examinée : la différence vient-elle du nombre de tokens
  générés ou du coût de chaque token ? On compare la durée par token généré.

Usage : python experiences/E12_latence.py
Sortie : outputs/E12_latence/metadata.json
"""
import json
import time

import numpy as np
import torch

from socle_experiences import model, snapshot, save

DOCUMENTS = ['Le rendez-vous aura lieu à Paris.', 'Le lieu de la rencontre est à Lyon.',
             'La rencontre est organisée à Rome.', 'Le lieu reste à définir.',
             'Le rendez-vous est à Lyon, et non à Paris.', 'Paris a été écarté ; le rendez-vous sera à Rome.',
             'Le bureau est à Paris, mais le rendez-vous se tiendra à Lyon.', 'Aucune ville ne figure dans la convocation.']
BASE = ('Extraire la ville du rendez-vous. Si elle manque, écrire null. Répondre en JSON.\n'
        'Texte: Le rendez-vous est à Paris.\nJSON: {"ville":"Paris"}\n'
        'Texte: Aucun lieu annoncé.\nJSON: {"ville":null}\n')
N_ECHAUFFEMENT = 2
N_REPETITIONS = 5


def generer(m, tok, texte, mode, sequences):
    """Une extraction ; renvoie la durée en secondes et le nombre de tokens générés."""
    x = tok(BASE + 'Texte: ' + texte + '\nJSON:', return_tensors='pt')
    taille = x.input_ids.shape[1]
    def autorises(lot, ids):
        suite = ids.tolist()[taille:]
        choix = {s[len(suite)] for s in sequences if len(s) > len(suite) and s[:len(suite)] == suite}
        return sorted(choix) or [tok.eos_token_id]
    debut = time.perf_counter()
    with torch.no_grad():
        sortie = m.generate(**x, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id,
                            **({'prefix_allowed_tokens_fn': autorises} if mode == 'contraint' else {}))
    return time.perf_counter() - debut, int(sortie.shape[1] - taille)


def passage(m, tok, ordre, sequences):
    """Tous les documents, dans les deux modes, selon l'ordre de modes donné."""
    lignes = []
    for i, texte in enumerate(DOCUMENTS):
        for mode in ordre:
            duree, n = generer(m, tok, texte, mode, sequences)
            lignes.append({'document': i, 'mode': mode, 'duree_s': duree, 'tokens': n})
    return lignes


def resume(lignes, mode):
    d = [l['duree_s'] for l in lignes if l['mode'] == mode]
    n = [l['tokens'] for l in lignes if l['mode'] == mode]
    return {'mediane_s': float(np.median(d)), 'p10_s': float(np.percentile(d, 10)),
            'p90_s': float(np.percentile(d, 90)), 'tokens_median': float(np.median(n)),
            's_par_token_median': float(np.median([a / b for a, b in zip(d, n)]))}


def main():
    debut = time.perf_counter()
    m, tok = model()
    options = [json.dumps({'ville': v}, ensure_ascii=False, separators=(',', ':')) for v in ['Paris', 'Lyon', 'Rome', None]]
    sequences = [tok.encode(o, add_special_tokens=False) + [tok.eos_token_id] for o in options]

    echauffement = [passage(m, tok, ['libre', 'contraint'], sequences) for _ in range(N_ECHAUFFEMENT)]
    repetitions = []
    for r in range(N_REPETITIONS):
        ordre = ['libre', 'contraint'] if r % 2 == 0 else ['contraint', 'libre']
        lignes = passage(m, tok, ordre, sequences)
        repetitions.append({'ordre': ordre, 'lignes': lignes,
                            'libre': resume(lignes, 'libre'), 'contraint': resume(lignes, 'contraint')})
    toutes = [l for r in repetitions for l in r['lignes']]
    save('E12_latence', {
        'threads': torch.get_num_threads(),
        'premier_appel_s': echauffement[0][0]['duree_s'],
        'echauffement': [{'libre': resume(p, 'libre'), 'contraint': resume(p, 'contraint')} for p in echauffement],
        'repetitions': repetitions,
        'global': {'libre': resume(toutes, 'libre'), 'contraint': resume(toutes, 'contraint')},
        'contraint_plus_rapide_partout': bool(all(r['contraint']['mediane_s'] < r['libre']['mediane_s'] for r in repetitions)),
    }, "Le mode contraint est-il plus rapide, et pourquoi ?",
        ['Deux passages d’échauffement exclus', 'Cinq répétitions, ordre des modes alterné',
         'Durée rapportée au nombre de tokens générés'],
        ['Une machine, un nombre de fils d’exécution, huit documents courts',
         'Latence mesurée dans le processus Python, sans serveur ni réseau',
         'Autres charges de la machine non contrôlées'],
        debut, {'gpt2': snapshot('gpt2')[1]})


if __name__ == '__main__':
    main()
