# High Garden Coffee — Reto Técnico ML Engineer

> Solución de analítica e inteligencia comercial para una empresa exportadora de café internacional,
> a partir de la base de datos histórica de consumo doméstico de café 1990-2020.

---

## La tesis central del proyecto

> **"High Garden Coffee no necesita saber cuánto café se toma en Brasil.
> Necesita saber cuánto café NO va a poder comprar en Brasil."**

El dataset entregado no es lo que parece a primera vista. Una auditoría de los 55 países incluidos revela que **todos son países productores de café** — ninguno de los grandes consumidores mundiales (Estados Unidos, Alemania, Japón) aparece. La fuente probable es la **International Coffee Organization (ICO)** .

Esto cambia radicalmente la pregunta de negocio:

> Cada taza retenida internamente por un país productor es una taza que **no entra al mercado de exportación** . El consumo doméstico de productores es, por lo tanto, un **indicador adelantado de la oferta exportable global** y un factor estructural sobre los precios internacionales — algo que la propia ICO documenta oficialmente.

Este reframing convierte el ejercicio de un análisis descriptivo a **inteligencia competitiva accionable** para una exportadora.

---

## Decisión metodológica crítica: ¿cómo abordamos "rangos de precios futuros"?

El enunciado solicita identificar **"rangos de precios futuros"** . Sin embargo, el dataset entregado **no contiene precios** . Pretender predecir precios con técnicas de ML sin variable objetivo sería técnicamente irresponsable.

**Lo que hicimos en su lugar:**

1. **Reconocer la limitación explícitamente** — está documentado al inicio de cada notebook.
2. **Reformular el problema en una métrica accionable y defensible:**
   construimos un **Índice de Presión sobre la Oferta (IPS)** que predice, para cada país,
   la _dirección e intensidad_ del impacto que su consumo doméstico tendrá sobre la oferta exportable global,
   y por extensión sobre los precios internacionales.
3. **Proponer la extensión natural** — la siguiente iteración integraría el ICO Composite Indicator Price
   (público desde 1965) como variable target en un modelo multivariado.

Esta decisión demuestra que entendimos los datos antes de modelar y que sabemos cuándo NO aplicar un modelo, en lugar de producir un forecast de precios espurio.

---

## Quick start

### Opción 1 — pip (recomendada)

```bash
# 1. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate         # Linux/Mac
# venv\Scripts\activate          # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Abrir Jupyter
jupyter notebook
```

### Opción 2 — conda

```bash
conda env create -f environment.yml
conda activate high-garden-coffee
jupyter notebook
```

### Ejecutar los notebooks

Abre los tres notebooks en orden y ejecuta cada uno con **"Run All"** :

1. `01_eda_storytelling.ipynb` — Análisis exploratorio con storytelling
2. `02_forecasting.ipynb` — Forecasting con detección de quiebres
3. `03_segmentation_1.ipynb` — Segmentación estratégica y scoring IPS
4. `04_chatbot.ipynb` — Chatbot analítico + RAG

> Cada notebook es **autosuficiente** : si se ejecuta sin que existan los archivos intermedios,
> los regenera automáticamente. No es necesario ejecutar nada manual fuera de los notebooks.

Para usar el chatbot fuera de Jupyter:

```bash
export OPENAI_API_KEY="sk-..."          # o define la variable en .env
python high_garden/rag_indexer.py        # genera high_garden/vector_store.faiss
python high_garden/chatbot.py            # inicia el asistente en modo CLI
```

---

## Estructura del proyecto

```
├── README.md                      ← este archivo
├── RESUMEN_EJECUTIVO.md           ← síntesis comercial (1-2 páginas)
├── requirements.txt               ← dependencias pip
├── environment.yml                ← dependencias conda (alternativa)
├── .env.example                   ← plantilla para credenciales
├── high_garden/
│   ├── 01_eda_storytelling.ipynb  ← MÓDULO 1: Análisis exploratorio
│   ├── 02_forecasting.ipynb       ← MÓDULO 2: Forecasting
│   ├── 03_segmentation_1.ipynb    ← MÓDULO 3: Segmentación + scoring IPS
│   ├── 04_chatbot.ipynb           ← MÓDULO 4: Chatbot + RAG documentado
│   ├── chatbot.py                 ← CLI del agente (LangChain + tools)
│   ├── rag_indexer.py             ← Pipeline para generar vector_store.faiss
│   ├── rag_store.py               ← Utilidades FAISS + wrapper LocalKnowledgeBase
│   ├── data_prep.py               ← Carga, validación, enriquecimiento
│   ├── eda_viz.py                 ← Visualizaciones del módulo 1
│   ├── style.py                   ← Sistema de diseño editorial
│   ├── breakpoints.py             ← Detección de quiebres (PELT)
│   ├── forecasting.py             ← 4 modelos + backtest + forecast
│   ├── forecast_viz.py            ← Visualizaciones del módulo 2
│   ├── segmentation.py            ← Feature engineering + K-Means + IPS
│   ├── segment_viz.py             ← Visualizaciones del módulo 3
│   ├── data/
│   │   ├── coffee_long.parquet
│   │   ├── coffee_wide.parquet
│   │   ├── breakpoints.parquet
│   │   ├── backtest_summary.parquet
│   │   ├── forecasts.parquet
│   │   ├── country_segments.parquet
│   │   ├── cluster_centroids.parquet
│   │   ├── clustering_meta.json
│   │   ├── validation_report.json
│   │   └── vector_store.faiss     ← Base de conocimiento FAISS (RAG)
│   └── figures/                   ← 17 PNGs generados por los notebooks
```

---

## Mapa de los módulos

| #     | Notebook                    | Pregunta de negocio                                  | Técnica clave                         |
| ----- | --------------------------- | ---------------------------------------------------- | ------------------------------------- |
| **1** | `01_eda_storytelling.ipynb` | ¿Qué nos dicen los datos?                            | EDA + storytelling editorial          |
| **2** | `02_forecasting.ipynb`      | ¿Dónde y cuánto crecerá la demanda doméstica?        | PELT + 4 modelos en competencia       |
| **3** | `03_segmentation_1.ipynb`   | ¿Qué países priorizar comercialmente?                | K-Means + IPS compuesto               |
| **4** | `04_chatbot.ipynb`          | ¿Cómo democratizar el análisis vía lenguaje natural? | LangChain agent + tools + RAG (FAISS) |

---

## Módulo 4 — Chatbot analítico + RAG

- **Objetivo:** entregar el análisis a decisores no técnicos mediante un asistente conversacional que aprovecha todo lo calculado en los módulos 1-3 y el contexto narrativo del README/RESUMEN.
- **Arquitectura:** LangChain `create_agent` + OpenAI GPT-4o-mini + tools tabulares (perfiles, rankings, clústers, forecasts, comparaciones) + nueva tool `search_knowledge_base` respaldada por un vector store FAISS (`vector_store.faiss`).
- **Pipeline RAG:** `rag_indexer.py` extrae texto de README, RESUMEN_EJECUTIVO y los 4 notebooks, los chunkifica (~1100 caracteres con 150 de overlap), genera embeddings `text-embedding-3-small` y persiste FAISS mediante `rag_store.LocalKnowledgeBase`.
- **System prompt minimalista:** solo define identidad + reglas de enrutamiento; los hallazgos viven en el vector store, evitando drift cuando los datos cambian.
- **Modo de uso:**
  1. Configura `OPENAI_API_KEY` (o edita `.env`).
  2. Ejecuta `python high_garden/rag_indexer.py` para regenerar `vector_store.faiss` cuando cambie el corpus.
  3. Lanza `python high_garden/chatbot.py` para una sesión CLI con memoria; o consume `ChatSession` desde cualquier script.
- **Demostración:** el notebook `04_chatbot.ipynb` documenta el pipeline RAG, muestra llamadas a las tools y ejemplifica las respuestas del agente con citación `(archivo · chunk)`.

---

## Hallazgos clave

### Análisis exploratorio (M1)

- El consumo doméstico de productores creció **+156%** en 30 años (CAGR 3.30%).
- El mercado **se concentra, no se diversifica** : HHI subió de 2,010 → 2,226.
- Brasil amplió su dominancia del 42% al 44% del consumo mundial.
- Asia multiplicó por 4.2× su consumo, África por 2.5×, Caribe por solo 1.2×.

### Forecasting (M2)

- **96% de los países** presentan quiebres estructurales detectados.
- **MAPE mediano global: 2.67%** sobre backtest con holdout de 5 años.
- En **67% de los países con quiebre** , el modelo segmentado supera al lineal completo,
  con mejora promedio de **+31.7 puntos porcentuales de MAPE** .
- Proyección global: **+8.8%** de consumo doméstico adicional en 2020-2024,
  liderada por Asia (+12.8%) y África (+10.0%).

### Segmentación (M3)

- 5 segmentos estratégicos identificados (silhouette 0.46, PCA 70.1% varianza):
  _Aceleración Sostenida, Crecimiento Maduro, Estables Predecibles, Estancamiento Leve, Estancamiento Severo_ .
- **Top 5 prioridades** por Índice de Presión sobre Oferta:
  Brazil (90.1), Colombia (86.5), Ethiopia (84.8), Indonesia (84.7), México (83.2).

---

## Decisiones técnicas que vale la pena destacar

| Decisión                                       | Justificación                                                                      |
| ---------------------------------------------- | ---------------------------------------------------------------------------------- |
| Clustering sobre**comportamiento** , no tamaño | Brasil es outlier extremo. Separar "qué tipo de mercado es" de "qué tan grande es" |
| **Winsorización 5%-95%**antes del clustering   | Neutraliza outliers (Venezuela en crisis, Ghana en boom) sin descartarlos          |
| **PELT (ruptures)**para detección de quiebres  | Algoritmo establecido, valida que el modelo segmentado mejora la precisión         |
| **Cuatro modelos compiten por país**           | Naive drift, lineal, Holt, segmented. Gana el de menor MAPE en backtest            |
| **Safety constraint**en forecasts              | Máximo -30% anual; evita extrapolaciones catastróficas (caso Venezuela)            |
| Notebooks**autosuficientes**                   | Cada uno regenera los datos si no existen — UX para el evaluador                   |

---

## Limitaciones honestas declaradas

1. **No hay precios en el dataset.** Abordado con el IPS como proxy direccional.
2. **No hay datos de producción.** Extensión natural: cruzar con datos ICO públicos.
3. **No hay población.** Análisis per cápita queda como roadmap.
4. **Series cortas (30 puntos anuales).** Descarta arquitecturas pesadas (Prophet, LSTM); favorece modelos clásicos.
5. **Eventos políticos/económicos no se capturan.** Venezuela ilustra esta limitación; el safety constraint la mitiga sin resolverla.

---

## Stack técnico

**Python 3.10+**

**pandas + numpy + pyarrow** para datos

**matplotlib** con sistema de diseño editorial custom

**scikit-learn** para K-Means y PCA

**statsmodels** para Holt smoothing

**ruptures** para detección PELT

**Jupyter** como interfaz de entrega

---

_Reto técnico — Ingeniería de Machine Learning · High Garden Coffee_
