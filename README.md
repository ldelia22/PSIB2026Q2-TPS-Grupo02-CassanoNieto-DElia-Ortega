# Clasificación automática de estadios de sueño

TP de Procesamiento de Señales e Imágenes Biomédicas (16.63), Grupo 2. Clasifica cada
época de una polisomnografía en W, N1, N2, N3 o REM con decisiones explicables a partir
de las señales. El detalle del método está en el anteproyecto.

## Cómo está dividido

El trabajo se reparte en tres frentes. Cada uno entrega una estructura de datos al
siguiente, definida en [`comun/contratos.py`](src/estadios/comun/contratos.py):

```
Registro ──► Frente A ──► RegistroPreprocesado ──► Frente B ──► TablaDescriptores ──► Frente C ──► hipnograma
```

| Frente | Etapas | Carpetas | Responsable |
|---|---|---|---|
| **A. Señales** | 1 a 5: exploratorio, filtrado, artefactos ocular y cardíaco, segmentación | `datos/`, `preprocesamiento/` | *a definir* |
| **B. Descriptores** | 6 a 9: multitaper, husos, ondas lentas, Woody, CSP, EOG, EMG, HRV | `descriptores/` | *a definir* |
| **C. Clasificación y resultados** | 10, 11 y sección 3: LDA, HMM, interfaz, métricas, H1 a H4 | `clasificacion/`, `evaluacion/`, `app/` | *a definir* |

Nadie tiene que esperar a otro frente, porque
[`comun/sintetico.py`](src/estadios/comun/sintetico.py) genera datos falsos con la misma
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
| 9. EOG, EMG y HRV | `descriptores/perifericos.py`, `hrv.py` | H4 |
| 10. LDA y HMM | `clasificacion/lda.py`, `hmm.py` | |
| 11. Interfaz | `app/streamlit_app.py` | |
| Evaluación | `evaluacion/metricas.py`, `hipotesis.py` | H1 a H4 |

### Qué ya funciona

Contratos, configuración, datos sintéticos, carga de la base, descarga, partición por
paciente, caché de descriptores y la conexión entre etapas (`pipeline.py`,
`descriptores/ensamblado.py`, `clasificacion/modelo.py`). Todo lo demás lanza
`NotImplementedError` y es lo que falta hacer.

## Puesta en marcha

Requiere Python 3.10 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Los tests de lo que falta implementar figuran como `skipped`. Cada función tiene su
test en `tests/test_frente_{a,b,c}.py`: al implementarla, el test empieza a correr solo
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
streamlit run app/streamlit_app.py
```

La interfaz arranca aun sin datos, con el registro sintético.

## Reglas de trabajo

- **Contratos:** los cambios en `src/estadios/comun/` se acuerdan entre los tres, porque
  afectan a todos.
- **Git:** una rama por tarea (`frente-a/filtrado`) y pull request contra `main`.
- **Fuga de datos:** todo lo que se aprende de los datos (CSP, LDA, transiciones) se
  ajusta solo con pacientes de entrenamiento. `ModeloEstadios` ya lo hace así.
- **Parámetros:** van en [`comun/config.py`](src/estadios/comun/config.py), no fijos en
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
