# Guía en español

[Inicio](../../README.md) · [Français](../fr/README.md) · [English](../en/README.md) · **Español** · [Português](../pt/README.md)

Este repositorio es el laboratorio del libro **La mécanique interne des LLM** (tomo 2, en francés) de Mustapha Alouani. El libro explica las preguntas, los mecanismos y la lectura de los resultados; este repositorio permite rehacer los 24 experimentos, examinar sus salidas y prolongarlos. Todo funciona en un procesador corriente (CPU), sin tarjeta gráfica ni servicio de pago.

## 1. Preparar el entorno

Ejecutar los comandos desde la raíz del repositorio. El entorno de referencia usa **Python 3.13.3** y las versiones fijadas en `requirements-cpu.txt`; otras versiones no se anuncian como probadas. Se necesita conexión para la instalación y la primera descarga de los modelos; después los experimentos usan las copias locales.

```bash
python -m venv .venv
```

Activar con `.venv\Scripts\Activate.ps1` en PowerShell, o `source .venv/bin/activate` en Linux/macOS. Después:

```bash
python -m pip install -r requirements-cpu.txt
python experiences/00_telecharger_modeles.py
python experiences/00_telecharger_modeles.py --verifier-seulement
```

Los pesos se descargan de sus distribuciones oficiales, en las revisiones fijadas por `modeles.json`; se aplican sus propias licencias. El recorrido inicial se produjo en CPU con Windows, los complementos en CPU con Linux, donde el programa principal también se volvió a ejecutar sin cambios (`outputs/verification/reexecution-linux.json`). macOS no se ha probado.

## 2. Los 24 experimentos, en el orden del libro

Cada experimento tiene un identificador **E01 a E24**, asignado en el orden en que el libro lo presenta. El mismo identificador aparece en todas partes: en el libro, al principio del nombre del programa (`experiences/E09_extraction.py`) y al principio de su carpeta de salida (`outputs/E09_extraction/`). E06 tiene dos programas: el primero aplica el clasificador, el segundo verifica esa aplicación. Los nombres de programas, los veredictos y los prompts permanecen en francés, como en el libro: traducir esta guía no traduce un experimento.

| Id. | Cap. | Programa (`experiences/`) | Pregunta | Tipo |
|---|---|---|---|---|
| **E01** | 1, 4 | `E01_classification.py` | ¿Puede un clasificador acertar aprovechando la longitud de los mensajes en lugar de la tarea? | exploratorio |
| **E02** | 2 | `E02_calcul_observe.py` | ¿Qué calcula GPT-2 sobre una frase: tokens, estados internos, logits, máscara? | observación |
| **E03** | 2, 3, 5, 7, 12 | `E03_calculs_guides.py` | Softmax, máscara, métricas, kappa, ACC: los cálculos del curso rehechos por programa | datos fabricados |
| **E04** | 3, 4, 5 | `E04_intervalles.py` | ¿Qué incertidumbre acompaña a tres resultados del libro? (después de E05 y E07) | relectura de salidas |
| **E05** | 4 | `E05_validation_classificateur.py` | ¿Se sostiene la reparación de E01 en 24 mensajes escritos después? (después de E01) | validación |
| **E06** | 4 | `E06_prediction.py`, `E06_verifier_prediction.py` | ¿Se aplica el pipeline guardado a un archivo de mensajes, y con qué errores? (después de E01) | control funcional |
| **E07** | 5 | `E07_recherche.py` | ¿Qué búsqueda, léxica o densa, encuentra el párrafo pertinente? | exploratorio |
| **E08** | 5 | `E08_geometrie_recherche.py` | ¿Qué valen las puntuaciones de la búsqueda densa frente al nivel de fondo? | observación y testigo |
| **E09** | 6 | `E09_extraction.py` | ¿Garantiza un formato restringido el campo correcto? | contraste |
| **E10** | 6, 7 | `E10_extraction_defis.py` | ¿Por qué la decodificación restringida responde siempre null, y qué regla lo corrige? (unos 5 min) | regla escrita antes de ejecutar |
| **E11** | 7 | `E11_choix_abstention.py` | ¿Qué gana y qué pierde una regla de abstención? (después de E09) | exploratorio |
| **E12** | 7 | `E12_latence.py` | ¿Cuánto tiempo cuestan la extracción libre y la restringida, calentamiento incluido? | mediciones repetidas |
| **E13** | 8, 10 | `E13_inspection.py` | ¿Qué lee la Logit Lens capa por capa, y qué hace un pilotaje controlado? | lectura y pilotaje |
| **E14** | 8 | `E14_attention.py` | ¿Qué muestran los pesos de atención, y confirma una ablación la lectura? | lectura y luego ablación |
| **E15** | 8 | `E15_sondes.py` | ¿Lee una sonda una información, o un indicio más simple? | testigos |
| **E16** | 9 | `E16_intervention.py` | ¿Reemplazar una activación restaura la respuesta con nombres no vistos? | descubrimiento y luego validación |
| **E17** | 9 | `E17_carte_patching.py` | ¿Dónde, capa por capa y posición por posición, se decide la respuesta? | regla escrita antes de ejecutar |
| **E18** | 9 | `E18_tetes.py` | ¿Qué cabezas de atención llevan el efecto, y se sostiene con una segunda plantilla? (unos 4 min) | regla escrita antes de ejecutar |
| **E19** | 10 | `E19_adaptation.py` | ¿Qué cambia tras LoRA, y quitar el adaptador restaura el modelo? | exploratorio |
| **E20** | 10 | `E20_pilotage.py` | ¿Resiste la dirección de pilotaje a direcciones aleatorias y al sentido inverso? | testigos |
| **E21** | 11 | `E21_langues.py` | ¿Qué cambia el idioma para una misma pregunta factual? | observación |
| **E22** | 11 | `E22_langues_formulations.py` | ¿Pesa la formulación tanto como el idioma? | regla escrita antes de ejecutar |
| **E23** | 11 | `E23_langues_tache.py` | ¿Tiene éxito la misma extracción en cuatro idiomas? | textos paralelos fabricados |
| **E24** | 12 | `E24_corpus_historique.py` | ¿Es coherente la aritmética de la cuantificación histórica? | salidas históricas |

`experiences/socle_experiences.py` no es un experimento: reúne el código común (carga de los modelos en revisiones fijas, escritura de los manifiestos) y el código de E01, E07, E09, E16 y E21, que ejecutan los programas del mismo nombre.

## 3. Ejecutar el recorrido o un solo experimento

```bash
python run_cpu.py              # los 24 experimentos, en el orden del recorrido
python run_cpu.py E09 E11      # solo estos
python experiences/E09_extraction.py
```

El lanzador guarda un registro por programa en `outputs/verification/` y se detiene en el primer fallo. Algunos experimentos releen las salidas de otro: E05 y E06 después de E01, E11 después de E09, E04 después de E05 y E07. El lanzador respeta ese orden. Una variante no debe cambiar en silencio los parámetros del libro: conservar las salidas anteriores y documentar el nuevo protocolo.

## 4. Aplicar el clasificador (E06)

```bash
python experiences/E06_prediction.py data/messages-exemple.txt outputs/E06_prediction/predictions-exemple.csv
python experiences/E06_verifier_prediction.py
```

Una línea UTF-8 es un mensaje. Cargar un pipeline guardado comprueba una interfaz, no la fiabilidad semántica: el ejemplo didáctico conserva a propósito una clasificación incorrecta. Cargar solo archivos `joblib` producidos por su propia ejecución, ya que este formato puede ejecutar código.

## 5. Cómo están construidos los experimentos

Todos siguen el método del capítulo 1 del libro: una **pregunta** planteada antes de mirar el resultado; **una sola cosa cambiada**, el resto fijo; un **testigo** que muestra lo que ocurriría sin el efecto supuesto; una **medida** elegida de antemano; una **regla de decisión**; y el **alcance** de la conclusión, escrito con sus límites.

Cada experimento tiene uno de estos tipos, indicado en la columna «Tipo»:

- **exploratorio**: hace visible un mecanismo en un pequeño conjunto de datos; su resultado sugiere, no prueba;
- **regla escrita antes de ejecutar**: la regla de decisión está escrita en la cabecera del programa antes de la primera ejecución y no se ajustó después;
- **datos fabricados**, **salidas históricas**, **relectura de salidas**: cálculos sin nueva ejecución de modelo, sobre datos construidos explícitamente o sobre agregados conservados de una campaña anterior.

Cada carpeta de salida contiene un **manifiesto** `metadata.json`: pregunta, controles, límites, huella del programa, revisiones de los modelos, semilla, procesador, versiones de las bibliotecas y duración. Leerlo antes de comparar dos puntuaciones. Las duraciones incluyen las cargas y no constituyen un banco de pruebas.

## 6. Diagnosticar

- Modelo ausente: volver a lanzar la preparación en línea y verificar las revisiones; no sustituir `modeles.json` en silencio.
- Falta la salida de otro experimento: ejecutar primero aquel del que depende (sección 3).
- Resultado diferente: comparar versiones, entradas y métricas antes de cambiar una tolerancia.
- E24 no valida ni la anotación ni la representatividad de los participantes del Grand Débat: los archivos `historique-*` son agregados conservados, no una nueva codificación.

`python tests/check_package.py` verifica el repositorio: programas legibles, un programa y una carpeta por identificador y, para cada manifiesto, que el programa presente es el que lo produjo.

## 7. Numeración anterior

Antes del 23 de septiembre de 2026, los experimentos se llamaban E1 a E8, más complementos con nombre (E3-defis, patching…). Los manifiestos y los registros de ejecución conservan esos nombres, vigentes en el momento del cálculo; no se han reescrito. `RENUMEROTATION.json` da la correspondencia y, para cada programa renombrado, las líneas modificadas: al restaurarlas se obtiene exactamente la huella registrada en el manifiesto.

| Nombre anterior | Id. | Carpeta de salida |
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
### 📕 El libro

**La mécanique interne des LLM** (edición francesa) — Tome 2 — Comprendre les mécanismes, construire des outils, vérifier leurs résultats, de Mustapha Alouani.

- Tapa blanda — *próximamente*
- Kindle — *próximamente*
- [Sitio del autor](https://alouani.org)

Del mismo autor : **La Mécanique des LLMs** — [Tapa blanda](https://amzn.eu/d/3oREERI) · [Kindle](https://amzn.eu/d/b7sG5iw) · [scripts del libro](https://github.com/alouani-org/mecanics-of-llms)
<!-- livre:fin -->
