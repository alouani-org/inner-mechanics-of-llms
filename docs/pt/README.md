# Guia em português

[Início](../../README.md) · [Français](../fr/README.md) · [English](../en/README.md) · [Español](../es/README.md) · **Português**

Este repositório é o laboratório do livro **A mecânica interna dos LLM** (volume 2), de Mustapha Alouani, publicado em português, francês, inglês e espanhol. O livro explica as perguntas, os mecanismos e a leitura dos resultados; este repositório permite que você refaça as 24 experiências, examine as saídas delas e vá além. Tudo roda em um processador comum (CPU), sem placa de vídeo nem serviço pago.

## 1. Preparar o ambiente

Execute os comandos a partir da raiz do repositório. O ambiente de referência usa **Python 3.13.3** e as versões fixadas em `requirements-cpu.txt`; outras versões não são declaradas como testadas. Você vai precisar de conexão com a internet para a instalação e para o primeiro download dos modelos; depois disso, as experiências usam as cópias locais.

```bash
python -m venv .venv
```

Ative com `.venv\Scripts\Activate.ps1` no PowerShell, ou `source .venv/bin/activate` no Linux/macOS. Em seguida:

```bash
python -m pip install -r requirements-cpu.txt
python experiences/00_telecharger_modeles.py
python experiences/00_telecharger_modeles.py --verifier-seulement
```

Os pesos são baixados das distribuições oficiais, nas revisões fixadas em `modeles.json`; valem as licenças de cada um. O percurso inicial foi gerado em CPU no Windows, e os complementos em CPU no Linux, onde o programa principal também foi executado de novo sem alterações (`outputs/verification/reexecution-linux.json`). O macOS não foi testado.

## 2. As 24 experiências, na ordem do livro

Cada experiência tem um identificador de **E01 a E24**, atribuído na ordem em que o livro a apresenta. O mesmo identificador aparece em todo lugar: no livro, no início do nome do programa (`experiences/E09_extraction.py`) e no início da pasta de saída (`outputs/E09_extraction/`). A E06 tem dois programas: o primeiro aplica o classificador, o segundo verifica essa aplicação. Os nomes dos programas, os veredictos e os prompts ficam em francês, exatamente como no livro: traduzir este guia não traduz uma experiência.

| Id. | Cap. | Programa (`experiences/`) | Pergunta | Tipo |
|---|---|---|---|---|
| **E01** | 1, 4 | `E01_classification.py` | Um classificador consegue acertar explorando o comprimento das mensagens em vez da tarefa? | exploratório |
| **E02** | 2 | `E02_calcul_observe.py` | O que o GPT-2 calcula em uma frase: tokens, estados internos, logits, máscara? | observação |
| **E03** | 2, 3, 5, 7, 12 | `E03_calculs_guides.py` | Softmax, máscara, métricas, kappa, ACC: os cálculos do curso refeitos por programa | dados fabricados |
| **E04** | 3, 4, 5 | `E04_intervalles.py` | Que incerteza acompanha três resultados do livro? (depois de E05 e E07) | releitura de saídas |
| **E05** | 4 | `E05_validation_classificateur.py` | A correção feita na E01 se sustenta diante de 24 mensagens escritas depois dela? (depois de E01) | validação |
| **E06** | 4 | `E06_prediction.py`, `E06_verifier_prediction.py` | O pipeline salvo se aplica a um arquivo de mensagens, e com quais erros? (depois de E01) | controle funcional |
| **E07** | 5 | `E07_recherche.py` | Qual busca, lexical ou densa, encontra o parágrafo pertinente? | exploratório |
| **E08** | 5 | `E08_geometrie_recherche.py` | Quanto valem as pontuações da busca densa em relação ao nível de fundo? | observação e controle |
| **E09** | 6 | `E09_extraction.py` | Um formato restrito garante o campo certo? | contraste |
| **E10** | 6, 7 | `E10_extraction_defis.py` | Por que a decodificação restrita sempre responde null, e que regra corrige isso? (cerca de 5 min) | regra escrita antes da execução |
| **E11** | 7 | `E11_choix_abstention.py` | O que uma regra de abstenção ganha e o que ela perde? (depois de E09) | exploratório |
| **E12** | 7 | `E12_latence.py` | Quanto tempo custam a extração livre e a restrita, incluindo o aquecimento? | medições repetidas |
| **E13** | 8, 10 | `E13_inspection.py` | O que a Logit Lens lê camada por camada, e o que faz uma pilotagem controlada? | leitura e pilotagem |
| **E14** | 8 | `E14_attention.py` | O que os pesos de atenção mostram, e uma ablação confirma essa leitura? | leitura e depois ablação |
| **E15** | 8 | `E15_sondes.py` | Uma sonda lê uma informação ou um indício mais simples? | controles |
| **E16** | 9 | `E16_intervention.py` | Substituir uma ativação restaura a resposta com nomes não vistos? | descoberta e depois validação |
| **E17** | 9 | `E17_carte_patching.py` | Onde, camada por camada e posição por posição, a resposta é decidida? | regra escrita antes da execução |
| **E18** | 9 | `E18_tetes.py` | Quais cabeças de atenção carregam o efeito, e isso se sustenta com um segundo template de frase? (cerca de 4 min) | regra escrita antes da execução |
| **E19** | 10 | `E19_adaptation.py` | O que muda depois do LoRA, e remover o adaptador restaura o modelo? | exploratório |
| **E20** | 10 | `E20_pilotage.py` | A direção de pilotagem se sustenta diante de direções aleatórias e do sentido inverso? | controles |
| **E21** | 11 | `E21_langues.py` | O que a língua muda para uma mesma pergunta factual? | observação |
| **E22** | 11 | `E22_langues_formulations.py` | A formulação pesa tanto quanto a língua? | regra escrita antes da execução |
| **E23** | 11 | `E23_langues_tache.py` | A mesma extração dá certo em quatro línguas? | textos paralelos fabricados |
| **E24** | 12 | `E24_corpus_historique.py` | A aritmética da quantificação histórica é coerente? | saídas históricas |

`experiences/socle_experiences.py` não é uma experiência: ele reúne o código comum (carregamento dos modelos em revisões fixas, escrita dos manifestos) e o código de E01, E07, E09, E16 e E21, que os programas de mesmo nome executam.

## 3. Executar o percurso ou uma única experiência

```bash
python run_cpu.py              # as 24 experiências, pela ordem do percurso
python run_cpu.py E09 E11      # só estas
python experiences/E09_extraction.py
```

O script de execução salva um registro por programa em `outputs/verification/` e para na primeira falha. Algumas experiências releem as saídas de outra: E05 e E06 depois de E01, E11 depois de E09, E04 depois de E05 e E07. O script respeita essa ordem. Uma variante não deve alterar silenciosamente os parâmetros do livro: preserve as saídas anteriores e documente o novo protocolo.

## 4. Aplicar o classificador (E06)

```bash
python experiences/E06_prediction.py data/messages-exemple.txt outputs/E06_prediction/predictions-exemple.csv
python experiences/E06_verifier_prediction.py
```

Cada linha UTF-8 é uma mensagem. Carregar um pipeline salvo verifica uma interface, não a confiabilidade semântica: o exemplo didático mantém de propósito uma classificação incorreta. Carregue apenas arquivos `joblib` gerados pela sua própria execução, porque esse formato pode executar código.

## 5. Como as experiências são construídas

Todas seguem o método do capítulo 1 do livro: uma **pergunta** feita antes de olhar o resultado; **uma única coisa alterada**, com o resto fixo; um **controle** que mostra o que aconteceria sem o efeito suposto; uma **medida** escolhida de antemão; uma **regra de decisão**; e o **alcance** da conclusão, escrito com seus limites.

Cada experiência tem um destes tipos, indicado na coluna “Tipo”:

- **exploratório**: torna visível um mecanismo em um pequeno conjunto de dados; o resultado sugere, não prova;
- **regra escrita antes da execução**: a regra de decisão está escrita no cabeçalho do programa antes da primeira execução e não foi ajustada depois;
- **dados fabricados**, **saídas históricas**, **releitura de saídas**: cálculos sem nova execução de modelo, sobre dados construídos explicitamente ou sobre agregados preservados de uma campanha anterior.

Cada pasta de saída contém um **manifesto** `metadata.json`: pergunta, controles, limites, hash do programa, revisões dos modelos, semente, processador, versões das bibliotecas e duração. Leia esse manifesto antes de comparar duas pontuações. As durações incluem os carregamentos e não servem como benchmark.

## 6. Diagnosticar problemas

- Modelo ausente: rode a preparação de novo com conexão e verifique as revisões; não substitua `modeles.json` silenciosamente.
- Falta a saída de outra experiência: execute primeiro aquela da qual ela depende (seção 3).
- Resultado diferente: compare versões, entradas e métricas antes de mudar uma tolerância.
- A E24 não valida nem a anotação nem a representatividade dos participantes do Grand Débat: os arquivos `historique-*` são agregados preservados, não uma nova codificação.

`python tests/check_package.py` verifica o repositório: programas legíveis, um programa e uma pasta por identificador e, para cada manifesto, se o programa presente é o que o gerou.

## 7. Numeração anterior

Antes de 23 de setembro de 2026, as experiências se chamavam E1 a E8, com complementos nomeados (E3-defis, patching…). Os manifestos e os registros de execução mantêm esses nomes, vigentes no momento do cálculo; eles não foram reescritos. `RENUMEROTATION.json` traz a correspondência e, para cada programa renomeado, as linhas alteradas: restaurando essas linhas, você obtém exatamente o hash registrado no manifesto.

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

**A mecânica interna dos LLM** — Volume 2 — Compreender os mecanismos, construir ferramentas, verificar seus resultados, de Mustapha Alouani.

- Capa comum — *em breve*
- Kindle — *em breve*
- [Site do autor](https://alouani.org)

Outras edições: 🇫🇷 La mécanique interne des LLM · 🇬🇧 The Inner Mechanics of LLMs · 🇪🇸 La mecánica interna de los LLM

Do mesmo autor: **A Mecânica dos LLM** — Teoria, Arquitetura e Prática para o Engenheiro — [scripts do livro](https://github.com/alouani-org/mecanics-of-llms)
<!-- livre:fin -->
