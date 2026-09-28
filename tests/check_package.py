"""Read-only checks of the companion repository; to run after python run_cpu.py.

Checks in particular that each manifest in outputs/ was indeed produced by the program
that now bears its identifier: if the program was renamed (RENUMEROTATION.json),
its original content is rebuilt line by line and then compared with the manifest's hash.
Later changes are recorded step by step, oldest first (ETAPES below): comments translated into
English (TRADUCTION_COMMENTAIRES.json, executed code unchanged, checked), then English labels
(ETIQUETTES_ANGLAISES.json, code changed, affected experiments rerun). Undoing the steps rebuilds
every version a manifest may cite.
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
ETAPES = [json.loads((ROOT / nom).read_text(encoding='utf-8'))
          for nom in ['TRADUCTION_COMMENTAIRES.json', 'ETIQUETTES_ANGLAISES.json']]
PAR_ETAPE = [{p['fichier']: p for p in e['programmes']} for e in ETAPES]


def etats_du_fichier(relatif):
    """Successive states of a file, current first, each as a list of lines (None if a step does not apply)."""
    chemin = ROOT / relatif
    if not chemin.is_file():
        return []
    lignes = chemin.read_bytes().decode('utf-8').split('\n')
    etats = [list(lignes)]
    for par_fichier in reversed(PAR_ETAPE):
        t = par_fichier.get(relatif)
        for c in (t['lignes_modifiees'] if t else []):
            if lignes[c['ligne'] - 1] != c['apres']:
                return etats + [None]
            lignes[c['ligne'] - 1] = c['avant']
        etats.append(list(lignes))
    return etats


def lignes_executees(relatif):
    """Lines of the oldest recorded version (the one in force before the recorded steps)."""
    etats = etats_du_fichier(relatif)
    return etats[-1] if etats else None


def code_execute(texte):
    """Syntax tree without docstrings and comments: equal trees mean the same executed code."""
    arbre = ast.parse(texte)
    for n in ast.walk(arbre):
        if (isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
                and isinstance(n.body[0].value.value, str)):
            n.body[0].value.value = ''
    return ast.dump(arbre)


def empreinte(lignes):
    return None if lignes is None else hashlib.sha256('\n'.join(lignes).encode('utf-8')).hexdigest()


def versions_du_programme(nom):
    """Hashes of every recorded version of program `nom` (current name or name before renumbering)."""
    etats = etats_du_fichier('experiences/' + nom)
    if etats:
        return [empreinte(e) for e in etats if e is not None]
    p = PAR_ANCIEN.get(nom)
    if not p:
        return []
    lignes = lignes_executees('experiences/' + p['nouveau'])
    if lignes is None:
        return []
    lignes = list(lignes)
    for c in p['lignes_modifiees']:
        if lignes[c['ligne'] - 1] != c['apres']:
            return []
        lignes[c['ligne'] - 1] = c['avant']
    return [empreinte(lignes)]


def empreinte_du_programme(nom):
    """Hash of the oldest recorded version of program `nom` (old or current name)."""
    v = versions_du_programme(nom)
    return v[-1] if v else None


def version_connue(nom, h):
    """True if `h` is the hash of a recorded version of program `nom`."""
    return h in versions_du_programme(nom)


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
    # Numbering: one program and one output folder per identifier, in the order of the book.
    check(IDS == [f'E{i:02d}' for i in range(1, len(IDS) + 1)], 'Identifiants E01 à E%02d consécutifs' % len(IDS))
    for ident in IDS:
        check(any(p.name.startswith(ident + '_') for p in programmes), 'Programme ' + ident)
        check(any(d.name.startswith(ident + '_') for d in (ROOT / 'outputs').iterdir() if d.is_dir()), 'Sortie ' + ident)
    for k, etape in enumerate(ETAPES):
        for t in etape['programmes']:
            etats = etats_du_fichier(t['fichier'])
            n = len(ETAPES)
            apres, avant = etats[n - 1 - k] if len(etats) > n - 1 - k else None, etats[n - k] if len(etats) > n - k else None
            check(empreinte(apres) == t['sha256_apres'], 'Étape ' + str(k + 1) + ', version enregistrée : ' + t['fichier'])
            check(empreinte(avant) == t['sha256_avant'], 'Étape ' + str(k + 1) + ', version précédente reconstituée : ' + t['fichier'])
            if etape.get('code_identique'):
                check(apres is not None and avant is not None
                      and code_execute('\n'.join(apres)) == code_execute('\n'.join(avant)),
                      'Commentaires seuls traduits, code exécuté identique : ' + t['fichier'])
        for exp in etape.get('experiences_relancees', []):
            m = json.loads((ROOT / 'outputs' / exp / 'metadata.json').read_text(encoding='utf-8'))
            check(m['script_sha256'] == empreinte(etats_du_fichier('experiences/' + m['script'])[0]),
                  'Relancée avec la version actuelle : ' + exp)
    for p in RENUM['programmes']:
        check(version_connue(p['ancien'], p['sha256_ancien']), 'Renumérotation réversible : ' + p['nouveau'])
    # Manifest provenance.
    for m_path in sorted((ROOT / 'outputs').glob('E*/metadata.json')):
        exp = m_path.parent.name
        m = json.loads(m_path.read_text(encoding='utf-8'))
        check(m['device'] == 'cpu', 'CPU ' + exp)
        check(version_connue(m['script'], m['script_sha256']), 'Source ' + exp)
        for source, h in m.get('dependencies', {}).items():
            check(version_connue(source, h), 'Dépendance ' + exp + ' ' + source)
    calculs = json.loads((ROOT / 'outputs/E03_calculs_guides/resultats.json').read_text(encoding='utf-8'))
    check(version_connue('exemples_calculables.py', calculs['script_sha256'])
          or version_connue('E03_calculs_guides.py', calculs['script_sha256']), 'Source E03')
    check((ROOT / 'outputs/E04_intervalles/resultats.json').is_file(), 'Sortie E04')
    e19 = json.loads((ROOT / 'outputs/E19_adaptation/metadata.json').read_text(encoding='utf-8'))['results']
    check(e19['restoration_max_probability_delta'] < 1e-6, 'E19 : retour arrière LoRA')
    e09 = json.loads((ROOT / 'outputs/E09_extraction/metadata.json').read_text(encoding='utf-8'))['results']
    check(all(r['contraint']['schema_valid'] for r in e09['rows']), 'E09 : schéma respecté en décodage contraint')
    # Documentation: guides present, valid internal links, book links up to date.
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
