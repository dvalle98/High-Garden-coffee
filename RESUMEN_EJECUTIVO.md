# Resumen Ejecutivo

**High Garden Coffee — Reto Técnico ML Engineer**

---

## La lectura que cambia el problema

El dataset no representa el mercado consumidor global; representa el **consumo doméstico de los 55 países productores**. Validación: ningún país consumidor (EE.UU., Alemania, Japón) aparece. La fuente es la International Coffee Organization (ICO).

**Implicación directa:** cada taza retenida en origen es una taza que no entra al mercado de exportación. El consumo doméstico de productores es un **indicador adelantado de la oferta exportable global**.

> ⚠️ _Sobre los "precios futuros" del enunciado: el dataset no contiene precios. Construimos un **Índice de Presión sobre la Oferta (IPS)** que predice la dirección e intensidad del impacto sobre los precios globales, basado en la relación oficialmente documentada por la ICO entre consumo doméstico de productores y precios internacionales._

---

## Cobertura de los requisitos mínimos

### ✅ Requisito 1 — Análisis de la información

**Entregado en:** `01_eda_storytelling.ipynb` · 7 visualizaciones editoriales

| Hallazgo                                                       | Implicación de negocio                                                   |
| -------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Consumo de productores creció**+156%** en 30 años (CAGR 3.30%) | La oferta exportable se estrecha estructuralmente                        |
| HHI subió de 2,010 → 2,226 (mercado más concentrado)           | El "boom emergente" intensificó el oligopolio en lugar de diversificarlo |
| Brasil pasó del 42% al 44% de share global                     | Cualquier movimiento de Brasil mueve el mercado mundial                  |
| Asia multiplicó por 4.2×, Caribe solo por 1.2×                 | Estrategias regionales diferenciadas son obligatorias                    |
| Primera contracción global en 30 años: 2019/20 (-0.52%)        | Señal de inflexión a vigilar                                             |

### ✅ Requisito 2 — Solución a las problemáticas de negocio

**Entregado en:** `02_forecasting.ipynb` + `03_segmentation.ipynb` · 10 visualizaciones

**Cuatro problemas de negocio identificados y resueltos:**

| Problema                                    | Técnica analítica                                                                 | Output accionable                                     |
| ------------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------- |
| _¿Dónde crecerá la demanda doméstica?_      | Forecasting con detección de quiebres (PELT) + 4 modelos en competencia           | Proyecciones 2020-2024 por país con intervalos al 80% |
| _¿Qué países agrupan estratégicamente?_     | K-Means sobre comportamiento (no tamaño), winsorización 5%-95%                    | 5 segmentos comerciales con playbook por grupo        |
| _¿Qué países priorizar?_                    | Índice IPS = 30%·tamaño + 30%·crecimiento + 20%·aceleración + 20%·predictibilidad | Ranking de los 55 países                              |
| _¿Hacia dónde va la presión sobre precios?_ | Proyección agregada regional del IPS                                              | Proxy direccional sobre presión exportable            |

### ✅ Requisito 3 — Implementación **y evaluación** de la solución

**Implementación:** 9 módulos Python reusables, 3 notebooks autosuficientes (regeneran datos automáticamente), código modular y documentado.

**Evaluación empírica:**

| Métrica                                              | Valor                           | Interpretación                         |
| ---------------------------------------------------- | ------------------------------- | -------------------------------------- |
| MAPE mediano global del forecasting                  | **2.67%**                       | Alta precisión predictiva              |
| Países con quiebre detectado                         | 96% (51/53)                     | Validación de la innovación técnica    |
| Mejora del segmented vs lineal en países con quiebre | **+31.7pp** en 67% de los casos | Decisión técnica con respaldo empírico |
| Silhouette del clustering (K=5)                      | 0.46                            | Segmentación estadísticamente robusta  |
| Países segmentados                                   | 55 de 55                        | Cobertura completa                     |

### ✅ Requisito 4 — Presentación de los resultados

**Tres audiencias, tres formatos:**

| Pieza                            | Audiencia                      | Formato                                              |
| -------------------------------- | ------------------------------ | ---------------------------------------------------- |
| `README.md`                      | Cualquiera (entry point)       | Tabla de contenidos + decisiones clave + quick start |
| `RESUMEN_EJECUTIVO.md`           | Decisor comercial / no técnico | Este documento (1-2 páginas)                         |
| 3 notebooks (`.ipynb`)           | Evaluador técnico              | Narrativa + código + visualizaciones embebidas       |
| 17 visualizaciones editoriales   | Material de presentación       | PNG en `figures/` capaces de dar informacion precisa |
| Datasets procesados (`.parquet`) | Reutilización futura           | Persistidos en `data/`                               |

---

## Top 5 prioridades comerciales (output principal)

| #   | País          | IPS  | Segmento                  | Acción                                         |
| --- | ------------- | ---- | ------------------------- | ---------------------------------------------- |
| 1   | **Brazil**    | 90.1 | Estable predecible        | Relación largo plazo. 44% del mercado mundial  |
| 2   | **Colombia**  | 86.5 | **Aceleración sostenida** | **Prioridad máxima** — régimen cambiando ahora |
| 3   | **Ethiopia**  | 84.8 | Crecimiento maduro        | Monitorear ratio consumo/producción            |
| 4   | **Indonesia** | 84.7 | Crecimiento maduro        | Estrategia regional sudeste asiático           |
| 5   | **Mexico**    | 83.2 | Estable predecible        | Base operativa confiable                       |

---

## Roadmap natural de extensión

1. **Integrar precios reales** (ICO Composite Indicator Price, público desde 1965) → predicción directa de precios
2. **Integrar datos de producción** → ratio consumo/producción explícito por país
3. **Datos per cápita** (Banco Mundial) → identificar mercados premium ocultos
4. **Chatbot conversacional con LLM** (Módulo bonus en desarrollo) → consultas en lenguaje natural sobre los resultados

---

_Detalles técnicos completos en los tres notebooks del proyecto. Solución end-to-end reproducible en 3 comandos (ver `README.md`)._
