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
  app/                   Interfaz en Streamlit (etapa 11)
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
| `02_frente_B_descriptores.ipynb` | B | Etapas 6 a 9 |
| `03_frente_C_clasificacion.ipynb` | C | Elección de umbrales, reglas, suavizado, métricas, hipnogramas |
| `04_frente_C_hipotesis.ipynb` | C | H1 a H4 |

## Clasificación: reglas en lugar de LDA + HMM

El anteproyecto proponía LDA + HMM, pero la devolución aclara que la clasificación del
TP cuatrimestral no puede usar técnicas de IA, y ambos lo son (se entrenan con épocas
etiquetadas). Se reemplazaron por:

1. **Reglas AASM** (`clasificacion/reglas.py`), evaluadas en orden sobre el percentil de
   cada descriptor dentro de su noche: W (EMG y α altos), N3 (δ y ondas lentas), REM
   (atonía, EOG o LF/HF altos, pocos husos), N2 (husos) y N1 por descarte.
2. **Desempate N1/REM con CSP**: la época cambia de estadio solo si la potencia CSP queda
   claramente del otro lado (H3).
3. **Suavizado por reglas** (`clasificacion/suavizado.py`) en lugar del HMM: épocas
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

El percentil de cada época se calcula dentro de su noche; el histograma junta las épocas
de todas las noches de entrenamiento. El detalle está al principio del notebook 03. LDA y HMM quedan para el
final de la materia. El detalle y la justificación están en la sección de métodos del
informe.

## Cómo está dividido

El trabajo se reparte en tres frentes. Cada uno entrega una estructura de datos al
siguiente, definida en [`comun/contratos.py`](codigo/src/estadios/comun/contratos.py):

```
Registro ──► Frente A ──► RegistroPreprocesado ──► Frente B ──► TablaDescriptores ──► Frente C ──► hipnograma
```

| Frente | Etapas | Carpetas | Responsable |
|---|---|---|---|
| **A. Señales** | 1 a 5: exploratorio, filtrado, artefactos ocular y cardíaco, segmentación. También la HRV de la etapa 9 | `datos/`, `preprocesamiento/`, `descriptores/hrv.py` | *a definir* |
| **B. Descriptores** | 6 a 9: multitaper, husos, ondas lentas, Woody, CSP, EOG, EMG | `descriptores/` (salvo `hrv.py`) | *a definir* |

La HRV es del Frente A porque se calcula con los latidos que detecta `qrs.py`. El
archivo queda en `descriptores/` porque su salida es un descriptor más de la tabla.
| **C. Clasificación y resultados** | 10, 11 y sección 3: reglas AASM, suavizado, interfaz, métricas, H1 a H4 | `clasificacion/`, `evaluacion/`, `app/` | *a definir* |

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
| 11. Interfaz | `app/streamlit_app.py` | |
| Evaluación | `evaluacion/metricas.py`, `hipotesis.py` | H1 a H4 |

### Qué ya funciona

Contratos, configuración, datos sintéticos, carga de la base, descarga, partición por
paciente, caché de descriptores y la conexión entre etapas (`pipeline.py`,
`descriptores/ensamblado.py`). Del Frente C: reglas, suavizado, modelo, métricas, la
prueba de Wilcoxon, H1, H2, H3, la parte de κ de H4 y la interfaz (las pestañas que
dependen del Frente A avisan qué falta). Todo lo demás lanza
`NotImplementedError` y es lo que falta hacer.

## Puesta en marcha

Requiere Python 3.10 o superior. Todos los comandos se corren desde `codigo/`.

```bash
cd codigo
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install jupyter      # Para los notebooks
pytest
```

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
python -m streamlit run app/streamlit_app.py
jupyter lab notebooks/
```

La interfaz arranca aun sin datos, con el registro sintético.

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

1. **Pasa-altos y ondas lentas.** La etapa 2 filtra a 0,5 Hz, pero la etapa 7 detecta
   ondas lentas entre 0,2 y 4 Hz: la parte de 0,2 a 0,5 Hz ya no llega. Se puede bajar el
   pasa-altos o detectar las ondas sobre la señal sin ese filtro.
2. **Cantidad de registros.** El anteproyecto dice 148 PSG, pero la v1.1 publica 151 (no
   existen SN014, SN064 ni SN135). No cambia el método, sí el texto del informe.
3. **Épocas fuera del período de sueño.** Las anotaciones incluyen vigilia antes de
   apagar y después de prender las luces. Esas épocas son fáciles de clasificar y pueden
   inflar κ. Hay que decidir si se evalúa toda la grabación o solo entre
   `luces_apagadas_s` y `luces_encendidas_s`.
4. **Épocas no estacionarias.** La etapa 5 las marca y quedan en la tabla, pero nada las
   usa todavía. Hay que decidir si se excluyen del entrenamiento o solo se informan.
5. **Saturación.** Los EEG y EOG de SN001 llegan a ±800 µV, el límite del rango del
   equipo. El exploratorio tendría que medir cuánto ocurre, porque esos tramos afectan al
   filtro adaptativo y a la detección de ondas lentas.
6. **Parámetros sin definir.** El anteproyecto no fija `nlms_orden`, `nlms_mu`,
   `wiener_orden`, `ventana_plantilla_s` ni `componentes_csp`. Los valores de `config.py`
   son provisorios y hay que ajustarlos.
7. **Segmentación (devolución).** La cátedra sugiere evaluar si conviene omitir la
   segmentación adaptativa y analizar directamente por ventanas fijas de 30 s.
8. **Artefacto cardíaco (devolución).** Puede no ser necesario: el exploratorio
   (promedio del EEG alrededor de cada latido) tiene que mostrar si es relevante antes
   de implementar la remoción.
9. **Sin inteligencia artificial (devolución).** Resuelta: LDA y HMM se reemplazaron
   por reglas AASM y suavizado por reglas (ver arriba). Esto cambió
   `ConfigClasificacion` en `comun/config.py` (se fue `usar_hmm`; se agregaron los
   umbrales y el tipo de suavizado), que no afecta la huella del caché de descriptores.
10. **Fracción de ondas lentas (Frente B).** La AASM define N3 por ondas lentas en
    ≥ 20 % de la época. Hoy la regla usa `ondas.cantidad`; un descriptor
    `ondas.fraccion` (tiempo con ondas lentas sobre 30 s) la haría más fiel al manual.
