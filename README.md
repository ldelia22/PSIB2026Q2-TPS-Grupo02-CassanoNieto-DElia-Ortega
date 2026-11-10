# Clasificación automática de estadios de sueño

TP de Procesamiento de Señales e Imágenes Biomédicas (16.63), Grupo 2. Clasifica cada
época de una polisomnografía en W, N1, N2, N3 o REM con decisiones explicables a partir
de las señales. El detalle del método está en el
[anteproyecto](informe/anteproyecto/anteproyecto.pdf) y su
[devolución](informe/anteproyecto/devolucion.txt).

## Estructura del repo

```
codigo/                  Todo lo que se ejecuta (se instala con pip install -e)
  src/estadios/          Paquete: un módulo por etapa del anteproyecto
  notebooks/             Un notebook por frente; generan las figuras del informe
  scripts/               Descarga, cálculo de descriptores, entrenamiento y evaluación
  app/                   Interfaz de escritorio con Tkinter (etapa 11)
  tests/                 Especificación de cada frente
informe/                 Informe final en LaTeX
  main.tex               Une las secciones; compila en Overleaf o con latexmk
  secciones/             Una por apartado; cada frente completa la suya
  figuras/               Las escriben los notebooks (prefijo a_, b_, c_, h_)
  anteproyecto/          Anteproyecto entregado y devolución de la cátedra
```

Los notebooks importan el paquete (no copian código): lo que se implementa en
`src/estadios/` aparece solo en el notebook y en la interfaz. Cada uno guarda sus
figuras en `informe/figuras/` con el prefijo de su frente, así el informe se
actualiza al volver a correrlos.

| Notebook | Frente | Contenido |
|---|---|---|
| `01_frente_A_senales.ipynb` | A | Etapas 1 a 5 |
| `02_frente_B_descriptores.ipynb` | B | Recorrido del Frente B y tabla de descriptores |
| `02a_frente_B_espectro.ipynb` | B | Etapa 6: multitaper, Welch, bandas y pendiente |
| `02b_frente_B_husos.ipynb` | B | Etapa 7: detección de husos |
| `02c_frente_B_ondas_lentas_woody.ipynb` | B | Etapa 7: ondas lentas y promediado de Woody |
| `02d_frente_B_csp.ipynb` | B | Etapa 8: filtrado espacial CSP |
| `02e_frente_B_eog_emg.ipynb` | B | Etapa 9: EOG y EMG |
| `03_frente_C_clasificacion.ipynb` | C | Elección de umbrales, reglas, suavizado, métricas, hipnogramas |
| `04_frente_C_hipotesis.ipynb` | C | H1 a H4 |

## Clasificación: reglas en lugar de LDA + HMM

El anteproyecto proponía LDA + HMM, pero la devolución aclara que la clasificación del
TP cuatrimestral no puede usar técnicas de IA, y ambos lo son (se entrenan con épocas
etiquetadas). Se reemplazaron por:

1. **Reglas AASM** (`clasificacion/reglas.py`), evaluadas en orden: W (EMG alto, y α alta
   o parpadeos), N3 (δ relativa alta y ondas lentas), REM (atonía, EOG o LF/HF altos, sin
   husos), N2 (al menos un huso) y N1 por descarte. Cada descriptor va en sus unidades:
   potencias relativas como fracción, husos y ondas como conteo por época, y EMG, EOG y
   LF/HF en veces la mediana de la noche del paciente. Una primera versión con
   percentiles de la noche dio κ = 0,245 con datos reales y se descartó.
2. **Continuidad de N2** (AASM): una época que cae en N1 por descarte después de una N2
   sigue siendo N2.
3. **Desempate N1/REM con CSP**: implementado, pero apagado por defecto (`"csp"` no está
   en `grupos`), porque con datos reales empeoraba la clasificación. H3 lo evalúa.
4. **Suavizado por reglas** (`clasificacion/suavizado.py`) en lugar del HMM: épocas
   aisladas y REM mínimo, o filtro de moda.

Los umbrales están en `ConfigClasificacion` y se eligen a mano con los histogramas de
`03_frente_C_clasificacion.ipynb` (pacientes de entrenamiento).

Unidades: cada paciente tiene una noche (~8 h), dividida en ~960 épocas de 30 s; se
clasifica cada época. El flujo es:

| Paso | Con qué pacientes | Cuántas veces |
|---|---|---|
| 1. Calcular descriptores | Todos los elegidos | Una vez por paciente (queda en caché) |
| 2. Elegir los umbrales | Entrenamiento, todos juntos en los histogramas | Una sola vez |
| 3. Clasificar | Cada paciente por separado | Una vez por paciente |
| 4. Evaluar (κ global y por paciente) | Prueba | Al final |

EMG, EOG y LF/HF se normalizan por la mediana de la noche de cada paciente; el histograma
junta las épocas de todas las noches de entrenamiento. El detalle está en el notebook 03. LDA y HMM quedan para
el final de la materia. El detalle y la justificación están en la sección de métodos del
informe.

## Cómo está dividido

El trabajo se reparte en tres frentes. Cada uno entrega una estructura de datos al
siguiente, definida en [`comun/contratos.py`](codigo/src/estadios/comun/contratos.py):

```
Registro ──► Frente A ──► RegistroPreprocesado ──► Frente B ──► TablaDescriptores ──► Frente C ──► hipnograma
```

| Frente | Etapas | Carpetas | Responsable |
|---|---|---|---|
| **A. Señales** | 1 a 5: exploratorio, filtrado, artefactos ocular y cardíaco, QRS, segmentación. También la HRV de la etapa 9 | `datos/`, `preprocesamiento/`, `descriptores/hrv.py` | Cassano Nieto |
| **B. Descriptores** | 6 a 9: multitaper, husos, ondas lentas, Woody, CSP, EOG, EMG | `descriptores/` (salvo `hrv.py`) | *a definir* |
| **C. Clasificación y resultados** | 10, 11 y sección 3: reglas AASM, suavizado, interfaz, métricas, H1 a H4 | `clasificacion/`, `evaluacion/`, `app/` | *a definir* |

La HRV es del Frente A porque se calcula con los latidos que detecta `qrs.py`. El
archivo queda en `descriptores/` porque su salida es un descriptor más de la tabla.

Nadie tiene que esperar a otro frente, porque
[`comun/sintetico.py`](codigo/src/estadios/comun/sintetico.py) genera datos falsos con la misma
forma que los reales:

| Frente | Trabaja con |
|---|---|
| A | `registro_sintetico()` o un registro real |
| B | `preprocesado_sintetico()` |
| C | `tabla_sintetica()` |

### Qué hay en cada etapa

| Etapa | Módulo | Hipótesis |
|---|---|---|
| 1. Exploratorio | `preprocesamiento/exploratorio.py` | |
| 2. Filtrado FIR/IIR | `preprocesamiento/filtrado.py` | |
| 3. Artefacto ocular | `preprocesamiento/ocular.py` | H4 |
| 4. Latidos y artefacto cardíaco | `preprocesamiento/qrs.py`, `cardiaco.py` | H4 |
| 5. Segmentación | `preprocesamiento/segmentacion.py` | |
| 6. Multitaper | `descriptores/espectro.py` | H1 |
| 7. Husos y ondas lentas | `descriptores/husos.py`, `ondas_lentas.py`, `woody.py` | H2 |
| 8. CSP | `descriptores/csp.py` | H3 |
| 9. EOG y EMG | `descriptores/perifericos.py` | |
| 9. HRV (Frente A) | `descriptores/hrv.py` | H4 |
| 10. Reglas AASM y suavizado | `clasificacion/reglas.py`, `suavizado.py`, `modelo.py` | H3 |
| 11. Interfaz | `app/interfaz.py` | |
| Evaluación | `evaluacion/metricas.py`, `hipotesis.py` | H1 a H4 |

### Qué ya funciona

Las tres partes están implementadas y pasan sus tests. El Frente A ya corre sobre los
registros reales y sus decisiones están justificadas en
`notebooks/01_frente_A_senales.ipynb`. Falta elegir los umbrales de las reglas con
datos reales y verificar H1 a H4 sobre ellos.

## Puesta en marcha

Requiere Python 3.10 o superior. Todos los comandos se corren desde `codigo/`.

```bash
cd codigo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

`requirements.txt` instala las dependencias, pytest, Jupyter y el paquete del TP. Las
mismas dependencias figuran en `pyproject.toml`, así que `pip install -e ".[dev]"` también
funciona. No se usa ninguna librería de aprendizaje automático: la devolución no lo
permite en el TP cuatrimestral. En Windows, con PowerShell, el entorno se activa con
`.venv\Scripts\Activate.ps1`.

Los tests de lo que falta implementar figuran como `skipped`. Cada función tiene su
test en `codigo/tests/test_frente_{a,b,c}.py`: al implementarla, el test empieza a correr solo
y sirve de especificación. Para ver qué le falta a tu frente:

```bash
pytest tests/test_frente_a.py -rs
```

### Datos

```bash
python scripts/descargar_datos.py          # 70 registros: unos 7 GB
python scripts/descargar_datos.py --n 3    # Alcanza para empezar
```

La semilla es fija, así que los tres bajamos los mismos registros y tenemos la misma
partición. Los datos no se suben al repo.

Si no tenés espacio, o si el repo está en una carpeta sincronizada con iCloud o Drive,
guardá los datos en otro lado (lo mismo vale para `.venv`):

```bash
export ESTADIOS_DATOS="/Volumes/MiDisco/hmc"
```

### Correr todo

```bash
python scripts/calcular_descriptores.py    # Lento, pero queda en caché
python scripts/entrenar_y_evaluar.py
python app/interfaz.py
jupyter lab notebooks/
```

La interfaz abre una ventana de escritorio y arranca aun sin datos, con el registro
sintético. Usa Tkinter, que viene con Python en Windows y macOS. En Linux se instala con
el paquete `python3-tk` del sistema.

## Reglas de trabajo

- **Contratos:** los cambios en `codigo/src/estadios/comun/` se acuerdan entre los tres, porque
  afectan a todos.
- **Git:** una rama por tarea (`frente-a/filtrado`) y pull request contra `main`.
- **Fuga de datos:** lo que se ajusta con datos (CSP) y la elección de los umbrales de
  las reglas se hacen solo con pacientes de entrenamiento. `ModeloEstadios` ya lo hace así.
- **Parámetros:** van en [`comun/config.py`](codigo/src/estadios/comun/config.py), no fijos en
  el código. Las hipótesis se prueban comparando configuraciones, y cada configuración
  tiene su propio caché.
- **Unidades:** señales en µV, estadios como enteros, latidos como índices de muestra.

## Orden sugerido

1. Definir responsables y resolver las decisiones de abajo.
2. Cada frente implementa lo suyo sobre datos sintéticos hasta que pasen sus tests.
3. A y B prueban la unión con unos pocos registros reales.
4. Se calculan los descriptores de los 70 registros, y C entrena y evalúa.
5. Se verifican H1 a H4 y se termina la interfaz.

## Decisiones abiertas

Surgieron al armar el esqueleto y contrastar el anteproyecto con la base.

1. **Pasa-altos y ondas lentas.** Resuelta. El pasa-altos del EEG bajó a 0,2 Hz para no
   recortar la banda en la que se detectan las ondas lentas.
2. **Cantidad de registros.** El anteproyecto dice 148 PSG, pero la v1.1 publica 151 (no
   existen SN014, SN064 ni SN135). No cambia el método, sí el texto del informe.
3. **Épocas fuera del período de sueño.** Resuelta. Las anotaciones incluyen vigilia
   antes de apagar y después de prender las luces, que es fácil de clasificar e inflaría
   κ. La clasificación y la evaluación usan solo las épocas entre luces apagadas y
   encendidas (`evaluacion/periodo.py`): los notebooks 03 y 04 recortan las tablas con
   `SOLO_PERIODO_DE_SUENO = True`, y `scripts/entrenar_y_evaluar.py` lo hace salvo que
   se pase `--toda-la-noche`. El notebook 03 compara los dos criterios. La interfaz
   muestra la grabación completa.
4. **Épocas no estacionarias.** Resuelta. La etapa 5 las marca y quedan en la tabla, pero
   las reglas no las usan y no se excluyen de la evaluación. El notebook 01 muestra que el
   clasificador acierta menos en ellas y que un tercio de las marcadas de W, N1, N2 y REM
   se clasifica como N3, algo a revisar en la regla de N3.
5. **Saturación.** Medida en el notebook 01. Los tramos que superan 500 µV no participan
   del ajuste del filtro ocular ni de la plantilla cardíaca. Las épocas siguen en la tabla.
6. **Parámetros sin definir.** Los de preprocesamiento se eligieron en el notebook 01.
   Sigue pendiente `componentes_csp`.
7. **Segmentación (devolución).** Resuelta. El análisis se hace por ventanas fijas de
   30 s y la marca de no estacionariedad con modelo AR queda solo como información.
8. **Artefacto cardíaco (devolución).** Resuelta. El promedio alrededor de cada latido
   muestra un artefacto relevante, por lo que la remoción queda activa.
9. **Sin inteligencia artificial (devolución).** Resuelta: LDA y HMM se reemplazaron
   por reglas AASM y suavizado por reglas (ver arriba). Esto cambió
   `ConfigClasificacion` en `comun/config.py` (se fue `usar_hmm`; se agregaron los
   umbrales y el tipo de suavizado), que no afecta la huella del caché de descriptores.
10. **Fracción de ondas lentas (Frente B).** La AASM define N3 por ondas lentas en
    ≥ 20 % de la época. Hoy la regla usa `ondas.cantidad`; un descriptor
    `ondas.fraccion` (tiempo con ondas lentas sobre 30 s) la haría más fiel al manual.
