# Guia em português

[Início](../../README.md) · [Français](../fr/README.md) · [English](../en/README.md) · [Español](../es/README.md) · **Português**

Este repositório é o laboratório do livro **La mécanique interne des LLM** (volume 2, em francês) de Mustapha Alouani. O livro explica as perguntas, os mecanismos e a leitura dos resultados; este repositório permite refazer as 24 experiências, examinar as suas saídas e prolongá-las. Tudo funciona num processador comum (CPU), sem placa gráfica nem serviço pago.

## 1. Preparar o ambiente

Executar os comandos a partir da raiz do repositório. O ambiente de referência usa **Python 3.13.3** e as versões fixadas em `requirements-cpu.txt`; outras versões não são anunciadas como testadas. É necessária ligação à internet para a instalação e o primeiro descarregamento dos modelos; depois, as experiências usam as cópias locais.

```bash
python -m venv .venv
```

Ativar com `.venv\Scripts\Activate.ps1` no PowerShell, ou `source .venv/bin/activate` em Linux/macOS. Depois:

```bash
python -m pip install -r requirements-cpu.txt
python experiences/00_telecharger_modeles.py
python experiences/00_telecharger_modeles.py --verifier-seulement
```

Os pesos são descarregados das distribuições oficiais, nas revisões fixadas em `modeles.json`; aplicam-se as respetivas licenças. O percurso inicial foi produzido em CPU com Windows, os complementos em CPU com Linux, onde o programa principal também foi reexecutado sem alterações (`outputs/verification/reexecution-linux.json`). O macOS não foi testado.

## 2. As 24 experiências, pela ordem do livro

Cada experiência tem um identificador **E01 a E24**, atribuído pela ordem em que o livro a apresenta. O mesmo identificador aparece em todo o lado: no livro, no início do nome do programa (`experiences/E09_extraction.py`) e no início da pasta de saída (`outputs/E09_extraction/`). E06 tem dois programas: o primeiro aplica o classificador, o segundo verifica essa aplicação. Os nomes dos programas, os veredictos e os prompts ficam em francês, como no livro: traduzir este guia não traduz uma experiência.

| Id. | Cap. | Programa (`experiences/`) | Pergunta | Tipo |
|---|---|---|---|---|
| **E01** | 1, 4 | `E01_classification.py` | Um classificador pode acertar explorando o comprimento das mensagens em vez da tarefa? | exploratório |
| **E02** | 2 | `E02_calcul_observe.py` | O que o GPT-2 calcula numa frase: tokens, estados internos, logits, máscara? | observação |
| **E03** | 2, 3, 5, 7, 12 | `E03_calculs_guides.py` | Softmax, máscara, métricas, kappa, ACC: os cálculos do curso refeitos por programa | dados fabricados |
| **E04** | 3, 4, 5 | `E04_intervalles.py` | Que incerteza acompanha três resultados do livro? (depois de E05 e E07) | releitura de saídas |
| **E05** | 4 | `E05_validation_classificateur.py` | A correção de E01 resiste a 24 mensagens escritas depois dela? (depois de E01) | validação |
| **E06** | 4 | `E06_prediction.py`, `E06_verifier_prediction.py` | O pipeline salvo aplica-se a um ficheiro de mensagens, e com que erros? (depois de E01) | controlo funcional |
| **E07** | 5 | `E07_recherche.py` | Que pesquisa, lexical ou densa, encontra o parágrafo pertinente? | exploratório |
| **E08** | 5 | `E08_geometrie_recherche.py` | Quanto valem as pontuações da pesquisa densa face ao nível de fundo? | observação e testemunha |
| **E09** | 6 | `E09_extraction.py` | Um formato restrito garante o campo certo? | contraste |
| **E10** | 6, 7 | `E10_extraction_defis.py` | Porque é que a descodificação restrita responde sempre null, e que regra o corrige? (cerca de 5 min) | regra escrita antes da execução |
| **E11** | 7 | `E11_choix_abstention.py` | O que ganha e o que perde uma regra de abstenção? (depois de E09) | exploratório |
| **E12** | 7 | `E12_latence.py` | Quanto tempo custam a extração livre e a restrita, aquecimento incluído? | medições repetidas |
| **E13** | 8, 10 | `E13_inspection.py` | O que lê a Logit Lens camada a camada, e o que faz uma pilotagem controlada? | leitura e pilotagem |
| **E14** | 8 | `E14_attention.py` | O que mostram os pesos de atenção, e uma ablação confirma a leitura? | leitura e depois ablação |
| **E15** | 8 | `E15_sondes.py` | Uma sonda lê uma informação, ou um indício mais simples? | testemunhas |
| **E16** | 9 | `E16_intervention.py` | Substituir uma ativação restaura a resposta com nomes não vistos? | descoberta e depois validação |
| **E17** | 9 | `E17_carte_patching.py` | Onde, camada a camada e posição a posição, se decide a resposta? | regra escrita antes da execução |
| **E18** | 9 | `E18_tetes.py` | Que cabeças de atenção transportam o efeito, e isso resiste a um segundo modelo de frase? (cerca de 4 min) | regra escrita antes da execução |
| **E19** | 10 | `E19_adaptation.py` | O que muda depois do LoRA, e retirar o adaptador restaura o modelo? | exploratório |
| **E20** | 10 | `E20_pilotage.py` | A direção de pilotagem resiste a direções aleatórias e ao sentido inverso? | testemunhas |
| **E21** | 11 | `E21_langues.py` | O que muda a língua para uma mesma pergunta factual? | observação |
| **E22** | 11 | `E22_langues_formulations.py` | A formulação pesa tanto como a língua? | regra escrita antes da execução |
| **E23** | 11 | `E23_langues_tache.py` | A mesma extração tem êxito em quatro línguas? | textos paralelos fabricados |
| **E24** | 12 | `E24_corpus_historique.py` | A aritmética da quantificação histórica é coerente? | saídas históricas |

`experiences/socle_experiences.py` não é uma experiência: reúne o código comum (carregamento dos modelos em revisões fixas, escrita dos manifestos) e o código de E01, E07, E09, E16 e E21, que os programas com o mesmo nome executam.

## 3. Executar o percurso ou uma só experiência

```bash
python run_cpu.py              # as 24 experiências, pela ordem do percurso
python run_cpu.py E09 E11      # só estas
python experiences/E09_extraction.py
```

O lançador guarda um registo por programa em `outputs/verification/` e para na primeira falha. Algumas experiências releem as saídas de outra: E05 e E06 depois de E01, E11 depois de E09, E04 depois de E05 e E07. O lançador respeita essa ordem. Uma variante não deve alterar em silêncio os parâmetros do livro: conservar as saídas anteriores e documentar o novo protocolo.

## 4. Aplicar o classificador (E06)

```bash
python experiences/E06_prediction.py data/messages-exemple.txt outputs/E06_prediction/predictions-exemple.csv
python experiences/E06_verifier_prediction.py
```

Uma linha UTF-8 é uma mensagem. Carregar um pipeline salvo verifica uma interface, não a fiabilidade semântica: o exemplo didático conserva de propósito uma classificação incorreta. Carregar apenas ficheiros `joblib` produzidos pela sua própria execução, porque este formato pode executar código.

## 5. Como as experiências são construídas

Todas seguem o método do capítulo 1 do livro: uma **pergunta** feita antes de olhar para o resultado; **uma só coisa alterada**, o resto fixo; uma **testemunha** que mostra o que aconteceria sem o efeito suposto; uma **medida** escolhida de antemão; uma **regra de decisão**; e o **alcance** da conclusão, escrito com os seus limites.

Cada experiência tem um destes tipos, indicado na coluna «Tipo»:

- **exploratório**: torna visível um mecanismo num pequeno conjunto de dados; o resultado sugere, não prova;
- **regra escrita antes da execução**: a regra de decisão está escrita no cabeçalho do programa antes da primeira execução e não foi ajustada depois;
- **dados fabricados**, **saídas históricas**, **releitura de saídas**: cálculos sem nova execução de modelo, sobre dados construídos explicitamente ou sobre agregados conservados de uma campanha anterior.

Cada pasta de saída contém um **manifesto** `metadata.json`: pergunta, controlos, limites, impressão digital do programa, revisões dos modelos, semente, processador, versões das bibliotecas e duração. Lê-lo antes de comparar duas pontuações. As durações incluem os carregamentos e não constituem um banco de ensaio.

## 6. Diagnosticar

- Modelo em falta: relançar a preparação com ligação e verificar as revisões; não substituir `modeles.json` em silêncio.
- Falta a saída de outra experiência: executar primeiro aquela de que depende (secção 3).
- Resultado diferente: comparar versões, entradas e métricas antes de mudar uma tolerância.
- E24 não valida nem a anotação nem a representatividade dos participantes do Grand Débat: os ficheiros `historique-*` são agregados conservados, não uma nova codificação.

`python tests/check_package.py` verifica o repositório: programas legíveis, um programa e uma pasta por identificador e, para cada manifesto, que o programa presente é o que o produziu.

## 7. Numeração anterior

Antes de 23 de setembro de 2026, as experiências chamavam-se E1 a E8, com complementos nomeados (E3-defis, patching…). Os manifestos e os registos de execução mantêm esses nomes, em vigor no momento do cálculo; não foram reescritos. `RENUMEROTATION.json` dá a correspondência e, para cada programa renomeado, as linhas alteradas: repondo-as, obtém-se exatamente a impressão digital registada no manifesto.

| Nome anterior | Id. | Pasta de saída |
|---|---|---|
| E1 | **E01** | `outputs/E01_classification/` |
| calcul-observe | **E02** | `outputs/E02_calcul_observe/` |
| calculs | **E03** | `outputs/E03_calculs_guides/` |
| intervalles | **E04** | `outputs/E04_intervalles/` |
| E1-validation | **E05** | `outputs/E05_validation_classificateur/` |
| prédiction | **E06** | `outputs/E06_prediction/` |
| E2 | **E07** | `outputs/E07_recherche/` |
| E2-geometrie | **E08** | `outputs/E08_geometrie_recherche/` |
| E3 | **E09** | `outputs/E09_extraction/` |
| E3-defis | **E10** | `outputs/E10_extraction_defis/` |
| E4 | **E11** | `outputs/E11_choix_abstention/` |
| latence | **E12** | `outputs/E12_latence/` |
| inspection | **E13** | `outputs/E13_inspection/` |
| attention | **E14** | `outputs/E14_attention/` |
| sondes | **E15** | `outputs/E15_sondes/` |
| E5 | **E16** | `outputs/E16_intervention/` |
| patching | **E17** | `outputs/E17_carte_patching/` |
| tetes | **E18** | `outputs/E18_tetes/` |
| E6 | **E19** | `outputs/E19_adaptation/` |
| pilotage | **E20** | `outputs/E20_pilotage/` |
| E7 | **E21** | `outputs/E21_langues/` |
| E7-formulations | **E22** | `outputs/E22_langues_formulations/` |
| E7-tache | **E23** | `outputs/E23_langues_tache/` |
| E8 | **E24** | `outputs/E24_corpus_historique/` |

<!-- livre:debut -->
### 📕 O livro

**La mécanique interne des LLM** (edição francesa) — Tome 2 — Comprendre les mécanismes, construire des outils, vérifier leurs résultats, de Mustapha Alouani.

- Capa comum — *em breve*
- Kindle — *em breve*
- [Site do autor](https://alouani.org)

Do mesmo autor : **La Mécanique des LLMs** — [Capa comum](https://amzn.eu/d/3oREERI) · [Kindle](https://amzn.eu/d/b7sG5iw) · [scripts do livro](https://github.com/alouani-org/mecanics-of-llms)
<!-- livre:fin -->
