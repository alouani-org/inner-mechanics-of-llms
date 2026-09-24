"""Contrôles en lecture seule du dépôt compagnon ; à lancer après python run_cpu.py.

Vérifie notamment que chaque manifeste de outputs/ a bien été produit par le programme
qui porte aujourd'hui son identifiant : si le programme a été renommé (RENUMEROTATION.json),
son contenu d'origine est reconstitué ligne à ligne puis comparé à l'empreinte du manifeste.
"""
from pathlib import Path
import ast
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'outils'))
from liens_livre import blocs_a_jour  # noqa: E402

RENUM = json.loads((ROOT / 'RENUMEROTATION.json').read_text(encoding='utf-8'))
PAR_ANCIEN = {p['ancien']: p for p in RENUM['programmes']}
IDS = [e['id'] for e in RENUM['experiences']]


def empreinte_du_programme(nom):
    """Empreinte du programme `nom` tel qu'il était à l'exécution (nom ancien ou actuel)."""
    actuel = ROOT / 'experiences' / nom
    if actuel.is_file():
        return hashlib.sha256(actuel.read_bytes()).hexdigest()
    p = PAR_ANCIEN.get(nom)
    if not p:
        return None
    lignes = (ROOT / 'experiences' / p['nouveau']).read_bytes().decode('utf-8').split('\n')
    for c in p['lignes_modifiees']:
        if lignes[c['ligne'] - 1] != c['apres']:
            return None
        lignes[c['ligne'] - 1] = c['avant']
    return hashlib.sha256('\n'.join(lignes).encode('utf-8')).hexdigest()


def main():
    errors, checks = [], []

    def check(ok, name):
        checks.append({'check': name, 'ok': bool(ok)})
        if not ok:
            errors.append(name)

    programmes = sorted((ROOT / 'experiences').glob('*.py'))
    for p in programmes:
        texte = p.read_text(encoding='utf-8')
        ast.parse(texte)
        check('llm-fr' not in texte and 'llm2-fr' not in texte, 'Aucun chemin du manuscrit : ' + p.name)
    # Numérotation : un programme et un dossier de sortie par identifiant, dans l'ordre du livre.
    check(IDS == [f'E{i:02d}' for i in range(1, len(IDS) + 1)], 'Identifiants E01 à E%02d consécutifs' % len(IDS))
    for ident in IDS:
        check(any(p.name.startswith(ident + '_') for p in programmes), 'Programme ' + ident)
        check(any(d.name.startswith(ident + '_') for d in (ROOT / 'outputs').iterdir() if d.is_dir()), 'Sortie ' + ident)
    for p in RENUM['programmes']:
        check(empreinte_du_programme(p['ancien']) == p['sha256_ancien'], 'Renumérotation réversible : ' + p['nouveau'])
    # Provenance des manifestes.
    for m_path in sorted((ROOT / 'outputs').glob('E*/metadata.json')):
        exp = m_path.parent.name
        m = json.loads(m_path.read_text(encoding='utf-8'))
        check(m['device'] == 'cpu', 'CPU ' + exp)
        check(empreinte_du_programme(m['script']) == m['script_sha256'], 'Source ' + exp)
        for source, h in m.get('dependencies', {}).items():
            check(empreinte_du_programme(source) == h, 'Dépendance ' + exp + ' ' + source)
    calculs = json.loads((ROOT / 'outputs/E03_calculs_guides/resultats.json').read_text(encoding='utf-8'))
    check(empreinte_du_programme('exemples_calculables.py') == calculs['script_sha256']
          or empreinte_du_programme('E03_calculs_guides.py') == calculs['script_sha256'], 'Source E03')
    check((ROOT / 'outputs/E04_intervalles/resultats.json').is_file(), 'Sortie E04')
    e19 = json.loads((ROOT / 'outputs/E19_adaptation/metadata.json').read_text(encoding='utf-8'))['results']
    check(e19['restoration_max_probability_delta'] < 1e-6, 'E19 : retour arrière LoRA')
    e09 = json.loads((ROOT / 'outputs/E09_extraction/metadata.json').read_text(encoding='utf-8'))['results']
    check(all(r['contraint']['schema_valid'] for r in e09['rows']), 'E09 : schéma respecté en décodage contraint')
    # Documentation : guides présents, liens internes valides, liens du livre à jour.
    guides = [ROOT / 'README.md', *sorted((ROOT / 'docs').rglob('*.md'))]
    for lang in ['fr', 'en', 'es', 'pt']:
        check((ROOT / 'docs' / lang / 'README.md').is_file(), 'Guide ' + lang)
    for g in guides:
        for target in re.findall(r'\]\(([^)\s]+)\)', g.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#') or target.startswith('mailto:'):
                continue
            check((g.parent / target.split('#')[0]).exists(), 'Lien ' + str(g.relative_to(ROOT)) + ' → ' + target)
    for nom, ok in blocs_a_jour(ROOT):
        check(ok, 'Liens du livre à jour : ' + nom)
    result = {'checks': checks, 'errors': errors, 'scope': 'logiciel et provenance, pas validité industrielle'}
    (ROOT / 'outputs/verification/package.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'checks': len(checks), 'errors': errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
