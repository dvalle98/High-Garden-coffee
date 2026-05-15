"""
Chatbot analítico para High Garden Coffee.

Arquitectura:
- LLM: OpenAI (configurable vía .env)
- Framework: LangChain v1.x con create_agent (basado en LangGraph)
- Memoria: lista de mensajes mantenida manualmente (la forma estándar en v1.x)
- Tools: funciones Python que consultan los datasets generados por los módulos 1-3
- RAG: vector store FAISS con narrativas de notebooks, README y resumen ejecutivo

El agente responde preguntas de negocio sobre los 55 países productores de café:
- Perfiles individuales (tamaño, crecimiento, segmento, IPS)
- Rankings y comparaciones
- Información de los clústers
- Forecasts a 5 años
- Preguntas metodológicas/narrativas respaldadas por RAG
"""

import os
import json
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Sequence

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler

from rag_store import LocalKnowledgeBase

warnings.filterwarnings("ignore")
load_dotenv()

# ============================================================================
# CARGA DE DATOS — base de conocimiento del agente
# ============================================================================
BASE = Path(__file__).resolve().parent
if (BASE.parent / "data").exists() and BASE.name == "src":
    BASE = BASE.parent
DATA = BASE / "data"

_segments = pd.read_parquet(DATA / "country_segments.parquet")
_centroids = pd.read_parquet(DATA / "cluster_centroids.parquet")
_long = pd.read_parquet(DATA / "coffee_long.parquet")
_forecasts = pd.read_parquet(DATA / "forecasts.parquet")
with open(DATA / "clustering_meta.json") as f:
    _cluster_meta = json.load(f)

VECTOR_STORE_PATH = BASE / "vector_store.faiss"
_EMBEDDINGS_MODEL: Optional[OpenAIEmbeddings] = None
_KNOWLEDGE_BASE: Optional[LocalKnowledgeBase] = None


def _find_country(country_name: str) -> Optional[str]:
    """Busca un país con matching flexible (case-insensitive, partial match)."""
    countries = _segments["Country"].tolist()
    name_lower = country_name.lower().strip()

    for c in countries:
        if c.lower() == name_lower:
            return c
    for c in countries:
        if name_lower in c.lower() or c.lower() in name_lower:
            return c
    aliases = {
        "vietnam": "Viet Nam",
        "ivory coast": "Côte d'Ivoire",
        "drc": "Democratic Republic of Congo",
        "bolivia": "Bolivia (Plurinational State of)",
        "laos": "Lao People's Democratic Republic",
    }
    if name_lower in aliases:
        return aliases[name_lower]
    return None


def _get_embeddings_model() -> OpenAIEmbeddings:
    global _EMBEDDINGS_MODEL
    if _EMBEDDINGS_MODEL is None:
        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        _EMBEDDINGS_MODEL = OpenAIEmbeddings(model=model_name)
    return _EMBEDDINGS_MODEL


def _get_knowledge_base() -> Optional[LocalKnowledgeBase]:
    global _KNOWLEDGE_BASE
    if _KNOWLEDGE_BASE is None:
        if not VECTOR_STORE_PATH.exists():
            return None
        _KNOWLEDGE_BASE = LocalKnowledgeBase.load(VECTOR_STORE_PATH)
    return _KNOWLEDGE_BASE


# ============================================================================
# TOOLS — funciones del agente
# ============================================================================


@tool
def search_knowledge_base(query: str, top_k: int = 4) -> str:
    """
    Busca evidencia narrativa en la base de conocimiento (README, resumen y notebooks).
    Usa esta tool para justificar decisiones metodológicas o recuperar hallazgos
    textuales que no viven en los .parquet estructurados.
    """

    kb = _get_knowledge_base()
    if kb is None:
        return (
            "La base de conocimiento vectorial no está disponible. "
            "Ejecuta `python high_garden/rag_indexer.py` para generarla."
        )

    embedder = _get_embeddings_model()
    query_vector = embedder.embed_query(query)
    hits = kb.search(query_vector, top_k=max(1, min(int(top_k), 8)))

    if not hits:
        return "No encontré coincidencias relevantes para esa búsqueda."

    lines = []
    for idx, hit in enumerate(hits, 1):
        snippet = " ".join(hit.text.split())
        if len(snippet) > 420:
            snippet = snippet[:420].rstrip() + "..."
        source = hit.metadata.get("source", "desconocido")
        chunk_id = hit.metadata.get("chunk_id", "?")
        lines.append(f"{idx}. {snippet}\n   Fuente: {source} · chunk {chunk_id}")

    return "\n\n".join(lines)


@tool
def get_country_profile(country: str) -> str:
    """
    Obtiene el perfil completo de un país: tamaño, crecimiento proyectado,
    segmento estratégico, IPS y estrategia recomendada.
    Usa esta tool cuando el usuario pregunte por un país específico.
    """
    real_name = _find_country(country)
    if real_name is None:
        return f"País '{country}' no encontrado en el dataset (55 países productores de café)."

    row = _segments[_segments["Country"] == real_name].iloc[0]
    cons_2019 = _long[(_long["Country"] == real_name) &
                      (_long["year"] == 2019)]["consumption_cups"].sum()
    rank = int(_segments["ips"].rank(ascending=False)[row.name])

    return (
        f"**{real_name}** (región: {row['region']})\n"
        f"- Consumo 2019: {cons_2019/1e6:.1f}M tazas ({row['share_global_pct']:.2f}% del mundial)\n"
        f"- Segmento: {row['cluster_name']}\n"
        f"- CAGR proyectado 2019-2024: {row['growth_forecast']:+.2f}% anual\n"
        f"- Aceleración reciente: {row['acceleration']:+.2f} pp\n"
        f"- CAGR histórico 1990-2019: {row['growth_historical']:+.2f}%\n"
        f"- Predictibilidad del forecast: {row['predictability']:.0f}/100\n"
        f"- **IPS: {row['ips']:.1f}** (ranking: posición {rank} de {len(_segments)})\n"
        f"- Estrategia recomendada: {row['strategy']}"
    )


@tool
def get_top_n_ips(n: int = 10) -> str:
    """
    Devuelve los top N países por Índice de Presión sobre Oferta (IPS).
    Estos son los países con mayor presión proyectada sobre la oferta exportable.
    Usa esta tool cuando el usuario pregunte por priorización, ranking o cuáles países priorizar.
    """
    n = max(1, min(int(n), 55))
    top = _segments.nlargest(n, "ips")[
        ["Country", "region", "cluster_name", "ips", "share_global_pct", "growth_forecast"]
    ]
    lines = [f"**Top {n} países por IPS (presión sobre oferta exportable):**\n"]
    for i, row in enumerate(top.itertuples(index=False), 1):
        lines.append(
            f"{i}. {row.Country} ({row.region}) — IPS {row.ips:.1f} | "
            f"Segmento: {row.cluster_name} | "
            f"Share: {row.share_global_pct:.1f}% | CAGR: {row.growth_forecast:+.1f}%"
        )
    return "\n".join(lines)


@tool
def get_cluster_info(cluster_name: str) -> str:
    """
    Devuelve información sobre un clúster estratégico: cuántos países lo componen,
    qué características definen al grupo, y la lista de países que lo integran.
    Los 5 segmentos posibles son: 'Aceleración Sostenida', 'Crecimiento Maduro',
    'Estables Predecibles', 'Estancamiento Leve', 'Estancamiento Severo'.
    """
    available = sorted(_segments["cluster_name"].unique())

    cluster_lower = cluster_name.lower().strip()
    matched = None
    for c in available:
        if c.lower() == cluster_lower or cluster_lower in c.lower():
            matched = c
            break

    if matched is None:
        return f"Clúster '{cluster_name}' no encontrado. Clústers disponibles: {', '.join(available)}"

    members = _segments[_segments["cluster_name"] == matched].sort_values("ips", ascending=False)
    centroid_row = _centroids[_centroids["cluster_name"] == matched].iloc[0]

    countries_str = ", ".join(members["Country"].tolist())

    return (
        f"**Clúster: {matched}** ({len(members)} países)\n\n"
        f"Perfil del segmento (centroide):\n"
        f"- Crecimiento proyectado: {centroid_row['growth_forecast']:+.2f}%\n"
        f"- Aceleración promedio: {centroid_row['acceleration']:+.2f} pp\n"
        f"- Volatilidad (CV): {centroid_row['cv']:.2f}\n"
        f"- Predictibilidad: {centroid_row['predictability']:.0f}/100\n\n"
        f"Países que integran este segmento:\n{countries_str}"
    )


@tool
def get_forecast(country: str) -> str:
    """
    Devuelve la proyección de consumo doméstico 2020-2024 para un país,
    con intervalos de confianza al 80% y el modelo usado.
    Usa esta tool cuando el usuario pregunte por proyecciones o forecasts.
    """
    real_name = _find_country(country)
    if real_name is None:
        return f"País '{country}' no encontrado."

    fc = _forecasts[_forecasts["Country"] == real_name].sort_values("year")
    if fc.empty:
        return f"No hay forecast disponible para {real_name}."

    last_actual = _long[(_long["Country"] == real_name) &
                        (_long["year"] == 2019)]["consumption_cups"].sum()
    model = fc["model_used"].iloc[0]
    had_break = fc["had_breakpoint"].iloc[0]

    lines = [f"**Forecast 2020-2024 para {real_name}** (modelo: {model}, quiebre detectado: {'sí' if had_break else 'no'})\n"]
    lines.append(f"Base: 2019 = {last_actual/1e6:.1f}M tazas\n")
    for year_val in sorted(fc["year"].unique()):
        forecast_grp = _forecasts[(_forecasts["Country"] == real_name) &
                                  (_forecasts["year"] == year_val)]
        total_fc = forecast_grp["forecast"].sum() / 1e6
        total_low = forecast_grp["lower_80"].sum() / 1e6
        total_high = forecast_grp["upper_80"].sum() / 1e6
        lines.append(
            f"  {int(year_val)}: {total_fc:.1f}M tazas  (IC 80%: {total_low:.1f}M - {total_high:.1f}M)"
        )
    final_fc = _forecasts[(_forecasts["Country"] == real_name) &
                          (_forecasts["year"] == 2024)]["forecast"].sum()
    total_growth = (final_fc - last_actual) / last_actual * 100
    lines.append(f"\nCrecimiento total proyectado 2019→2024: {total_growth:+.1f}%")
    return "\n".join(lines)


@tool
def compare_countries(countries: str) -> str:
    """
    Compara 2 a 4 países lado a lado sobre sus métricas clave.
    El parámetro 'countries' debe ser una lista separada por comas, ej: 'Brazil,Vietnam,Colombia'.
    Usa esta tool cuando el usuario pida comparar países.
    """
    names = [c.strip() for c in countries.split(",")]
    if len(names) < 2:
        return "Necesito al menos 2 países para comparar."
    if len(names) > 4:
        names = names[:4]

    resolved = []
    for n in names:
        real = _find_country(n)
        if real:
            resolved.append(real)
        else:
            return f"País '{n}' no encontrado."

    rows = _segments[_segments["Country"].isin(resolved)].set_index("Country")
    rows = rows.reindex(resolved)

    lines = ["| Métrica | " + " | ".join(resolved) + " |"]
    lines.append("|---|" + "|".join(["---"] * len(resolved)) + "|")

    metrics = [
        ("Región",              "region",            lambda x: str(x)),
        ("Segmento",            "cluster_name",      lambda x: str(x)),
        ("Share global (%)",    "share_global_pct",  lambda x: f"{x:.2f}%"),
        ("CAGR proyectado",     "growth_forecast",   lambda x: f"{x:+.2f}%"),
        ("Aceleración",         "acceleration",      lambda x: f"{x:+.2f} pp"),
        ("Predictibilidad",     "predictability",    lambda x: f"{x:.0f}/100"),
        ("IPS",                 "ips",               lambda x: f"{x:.1f}"),
    ]
    for label, col, fmt in metrics:
        lines.append("| " + label + " | " + " | ".join(fmt(rows.loc[c, col]) for c in resolved) + " |")
    return "\n".join(lines)


TOOLS = [search_knowledge_base, get_country_profile, get_top_n_ips, get_cluster_info, get_forecast, compare_countries]


# ============================================================================
# SYSTEM PROMPT — contexto rico del proyecto
# ============================================================================

SYSTEM_PROMPT = """Eres el asistente analítico de High Garden Coffee. Respondes en español profesional y ayudas a interpretar el estudio sobre consumo doméstico de café de 55 países productores.

Reglas de operación:
1. Cuando el usuario salude o pida contexto, preséntate proactivamente y explica qué preguntas puedes resolver.
2. Usa `search_knowledge_base` para cualquier pregunta metodológica, narrativa o estratégica. Resume los hallazgos y cita la fuente en formato `(archivo · chunk)`.
3. Usa las tools tabulares (`get_country_profile`, `get_top_n_ips`, `get_cluster_info`, `get_forecast`, `compare_countries`) cuando el usuario requiera cifras concretas.
4. Si el dato no existe en los datasets ni en la base vectorial, sé explícito sobre la limitación y explica qué información haría falta.
5. Mantén precisión numérica (1-2 decimales) y enlaza tus conclusiones con la evidencia recuperada.
"""


# ============================================================================
# CONSTRUCCIÓN DEL AGENTE (LangChain v1.x con create_agent)
# ============================================================================

def build_agent(model: Optional[str] = None, temperature: float = 0.3, streaming: bool = False):
    """
    Construye el agente conversacional con tools.

    LangChain v1.x usa create_agent (basado en LangGraph). La "memoria" se maneja
    pasando el historial de mensajes en cada invocación.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your"):
        raise ValueError(
            "OPENAI_API_KEY no configurada. Copia .env.example a .env y agrega tu key."
        )

    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key, streaming=streaming)

    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


class StreamingStdOutCallbackHandler(BaseCallbackHandler):
    """Envía los tokens del LLM al stdout en tiempo real."""

    def __init__(self, flush: bool = True):
        self.flush = flush

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=self.flush)


class ChatSession:
    """
    Sesión conversacional con memoria explícita.
    Mantiene la lista de mensajes y la pasa al agente en cada turno.
    """

    def __init__(self, agent=None, streaming: bool = False, **agent_kwargs):
        if agent is None:
            agent = build_agent(streaming=streaming, **agent_kwargs)
        self.agent = agent
        self.messages: List[Dict] = []
        self.streaming = streaming

    def chat(self, user_message: str, callbacks: Optional[Sequence[BaseCallbackHandler]] = None) -> str:
        """Envía un mensaje y devuelve la respuesta, manteniendo contexto."""
        self.messages.append({"role": "user", "content": user_message})
        if callbacks:
            result = self.agent.invoke({"messages": self.messages}, config={"callbacks": callbacks})
        else:
            result = self.agent.invoke({"messages": self.messages})
        all_messages = result["messages"]

        # Reconstruir historial: incluir user + assistant text, ignorar tool calls
        new_history = []
        for m in all_messages:
            content = getattr(m, "content", "")
            mtype = getattr(m, "type", "")
            if mtype == "human" and content:
                new_history.append({"role": "user", "content": content})
            elif mtype == "ai" and content:
                new_history.append({"role": "assistant", "content": content})
        self.messages = new_history

        # La última respuesta del asistente con contenido
        for m in reversed(all_messages):
            if getattr(m, "type", "") == "ai" and getattr(m, "content", ""):
                return m.content
        return "(sin respuesta)"

    def reset(self):
        """Limpia la memoria de la conversación."""
        self.messages = []


# ============================================================================
# MAIN — demo CLI
# ============================================================================

if __name__ == "__main__":
    streaming_env = os.getenv("CHATBOT_STREAMING", "1").lower()
    streaming_enabled = streaming_env not in {"0", "false", "no"}

    print("=" * 70)
    print("HIGH GARDEN COFFEE — Chatbot Analítico")
    print("=" * 70)
    print("\nEscribe tus preguntas. Escribe 'salir' para terminar.")
    print(f"Streaming {'activado' if streaming_enabled else 'desactivado'} (CHATBOT_STREAMING env).\n")

    session = ChatSession(streaming=streaming_enabled)

    while True:
        try:
            user = input("\n👤 Tú: ").strip()
            if not user:
                continue
            if user.lower() in {"salir", "exit", "quit", "q"}:
                print("Hasta luego.")
                break
            if streaming_enabled:
                print("\n🤖 Asistente: ", end="", flush=True)
                handler = StreamingStdOutCallbackHandler()
                response = session.chat(user, callbacks=[handler])
                print()
            else:
                response = session.chat(user)
                print(f"\n🤖 Asistente: {response}")
        except KeyboardInterrupt:
            print("\nInterrumpido.")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
