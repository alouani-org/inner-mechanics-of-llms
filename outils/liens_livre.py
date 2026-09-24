"""Met à jour les liens vers le livre dans les README à partir de livre.json.

Chaque README contient un bloc délimité par <!-- livre:debut --> et <!-- livre:fin -->.
Ce programme le réécrit dans la langue du fichier (README racine : bloc compact en quatre langues).
Un lien vide dans livre.json est affiché « à paraître » : il suffit de le renseigner
puis de relancer ce programme pour que tous les guides pointent vers la bonne page.

Usage : python outils/liens_livre.py              (réécrit les blocs)
        python outils/liens_livre.py --verifier   (signale les blocs périmés, sans écrire)
"""
from pathlib import Path
import json
import re
import sys

DEBUT, FIN = '<!-- livre:debut -->', '<!-- livre:fin -->'
BLOC = re.compile(re.escape(DEBUT) + r'.*?' + re.escape(FIN), re.S)

TEXTES = {
    'fr': {'titre': '📕 Le livre', 'broche': 'Broché', 'kindle': 'Kindle', 'bientot': 'à paraître',
           'edition': '', 'depot': 'Dépôt compagnon', 'site': "Site de l'auteur", 'fiche': "Page auteur Amazon",
           'meme': 'Du même auteur', 'depot_meme': 'scripts du livre', 'par': 'par'},
    'en': {'titre': '📕 The book', 'broche': 'Paperback', 'kindle': 'Kindle', 'bientot': 'coming soon',
           'edition': ' (French edition)', 'depot': 'Companion repository', 'site': "Author's website",
           'fiche': 'Amazon author page', 'meme': 'By the same author', 'depot_meme': 'book scripts', 'par': 'by'},
    'es': {'titre': '📕 El libro', 'broche': 'Tapa blanda', 'kindle': 'Kindle', 'bientot': 'próximamente',
           'edition': ' (edición francesa)', 'depot': 'Repositorio complementario', 'site': 'Sitio del autor',
           'fiche': 'Página de autor en Amazon', 'meme': 'Del mismo autor', 'depot_meme': 'scripts del libro', 'par': 'de'},
    'pt': {'titre': '📕 O livro', 'broche': 'Capa comum', 'kindle': 'Kindle', 'bientot': 'em breve',
           'edition': ' (edição francesa)', 'depot': 'Repositório complementar', 'site': 'Site do autor',
           'fiche': 'Página do autor na Amazon', 'meme': 'Do mesmo autor', 'depot_meme': 'scripts do livro', 'par': 'de'},
}


def lien(texte, url, t):
    return f'[{texte}]({url})' if url else f'{texte} — *{t["bientot"]}*'


def bloc(livre, langue):
    t = TEXTES[langue]
    a, m = livre['achat'], livre['du_meme_auteur']
    lignes = [f'### {t["titre"]}', '',
              f'**{livre["titre"]}**{t["edition"]} — {livre["sous_titre"]}, {t["par"]} {livre["auteur"]}.', '',
              f'- {lien(t["broche"], a.get("broche"), t)}',
              f'- {lien(t["kindle"], a.get("kindle"), t)}']
    if livre['autres_liens'].get('fiche_auteur_amazon'):
        lignes.append(f'- [{t["fiche"]}]({livre["autres_liens"]["fiche_auteur_amazon"]})')
    if livre.get('site_auteur'):
        lignes.append(f'- [{t["site"]}]({livre["site_auteur"]})')
    lignes += ['', f'{t["meme"]} : **{m["titre"]}** — [{t["broche"]}]({m["broche"]}) · [{t["kindle"]}]({m["kindle"]}) · '
                   f'[{t["depot_meme"]}]({m["depot"]})']
    return '\n'.join(lignes)


def bloc_multilingue(livre):
    """Bloc compact des quatre langues, pour le README racine."""
    a, m = livre['achat'], livre['du_meme_auteur']
    bientot = 'à paraître · coming soon · próximamente · em breve'
    def l(url):
        return f'[Amazon]({url})' if url else f'*{bientot}*'
    lignes = ['### 📕 Le livre · The book · El libro · O livro', '',
              f'**{livre["titre"]}** — {livre["sous_titre"]} — {livre["auteur"]} (édition française · French edition)', '',
              f'- **Broché · Paperback · Tapa blanda · Capa comum** : {l(a.get("broche"))}',
              f'- **Kindle** : {l(a.get("kindle"))}']
    if livre['autres_liens'].get('fiche_auteur_amazon'):
        lignes.append(f'- **Auteur · Author** : [Amazon]({livre["autres_liens"]["fiche_auteur_amazon"]})')
    if livre.get('site_auteur'):
        lignes.append(f'- **Site** : [{livre["site_auteur"].split("//")[-1]}]({livre["site_auteur"]})')
    lignes += ['', f'Du même auteur · By the same author : **{m["titre"]}** — [Broché · Paperback]({m["broche"]}) · '
                   f'[Kindle]({m["kindle"]}) · [scripts]({m["depot"]})']
    return '\n'.join(lignes)


def contenu_attendu(racine, livre, fichier):
    """Texte du fichier une fois son bloc à jour (None s'il n'a pas de bloc)."""
    texte = fichier.read_text(encoding='utf-8')
    if DEBUT not in texte:
        return None
    rel = fichier.relative_to(racine).parts
    if rel[0] == 'docs' and rel[1] in TEXTES:
        corps = bloc(livre, rel[1])
    else:
        corps = bloc_multilingue(livre)
    return BLOC.sub(lambda _: f'{DEBUT}\n{corps}\n{FIN}', texte)


def fichiers(racine):
    return [racine / 'README.md', *sorted((racine / 'docs').glob('*/README.md'))]


def blocs_a_jour(racine):
    livre = json.loads((racine / 'livre.json').read_text(encoding='utf-8'))
    for f in fichiers(racine):
        attendu = contenu_attendu(racine, livre, f)
        yield str(f.relative_to(racine)), attendu is not None and attendu == f.read_text(encoding='utf-8')


def main():
    racine = Path(__file__).resolve().parents[1]
    livre = json.loads((racine / 'livre.json').read_text(encoding='utf-8'))
    verifier = '--verifier' in sys.argv
    perimes = []
    for f in fichiers(racine):
        attendu = contenu_attendu(racine, livre, f)
        if attendu is None:
            perimes.append(f'{f.relative_to(racine)} (bloc absent)')
        elif attendu != f.read_text(encoding='utf-8'):
            perimes.append(str(f.relative_to(racine)))
            if not verifier:
                f.write_text(attendu, encoding='utf-8')
    if verifier:
        print('Blocs à mettre à jour : ' + ', '.join(perimes) if perimes else 'Tous les blocs sont à jour.')
        sys.exit(1 if perimes else 0)
    print(f'{len(perimes)} fichier(s) mis à jour.' if perimes else 'Rien à changer.')


if __name__ == '__main__':
    main()
