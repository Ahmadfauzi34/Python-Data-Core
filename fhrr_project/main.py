"""
FHRR Chat UI — Streamlit interface untuk FHRRQueryInterface.
Run: streamlit run main.py
"""
import os
import sys
import time
import streamlit as st

# Pastikan import path = folder ini (agar `from core.x import ...` jalan)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from core.runner import FHRRResearchRunner
from core.topology import FHRRTopologicalLayer
from data.dataset import fhrr_research_dataset
from interface.query_api import FHRRQueryInterface
from memory.knowledge_graph import KnowledgeGraphIngestor
from memory.open_vocab import extend_engine_open_vocab
from agents.discoverer import SelfSupervisedDiscovery
from agents.improver import SelfImprovementEngine


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FHRR Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# System bootstrap (cached antar rerun Streamlit)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Inisialisasi FHRR engine...")
def boot_system(dim: int = 4096, run_training: bool = True):
    runner = FHRRResearchRunner(dim=dim)
    runner.load_dataset(fhrr_research_dataset)

    topo = FHRRTopologicalLayer(runner.engine)
    runner.attach_topology(topo)

    open_vocab = extend_engine_open_vocab(runner.engine)
    kg = KnowledgeGraphIngestor(runner.engine, open_vocab)
    runner.attach_kg(kg)

    discoverer = SelfSupervisedDiscovery(runner.engine)
    improver = SelfImprovementEngine(runner.engine, kg)

    if run_training:
        try:
            runner.run_training()
        except Exception as e:
            st.warning(f"Training dilewati: {e}")

    qi = FHRRQueryInterface(runner).attach_kg(kg).attach_discoverer(discoverer).attach_improver(improver)
    return runner, qi


# -----------------------------------------------------------------------------
# Sidebar — kontrol sistem
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ FHRR Control")

    dim = st.select_slider("Dimensi vektor", options=[512, 1024, 2048, 4096, 8192], value=4096)
    do_train = st.checkbox("Jalankan training saat boot", value=True)

    if st.button("🔄 Re-init system", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.pop("messages", None)
        st.rerun()

    st.divider()
    explain_mode = st.toggle("Mode penjelasan (verbose)", value=False)
    show_debug = st.toggle("Tampilkan info debug", value=False)

    st.divider()
    if st.button("🗑️ Bersihkan chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# -----------------------------------------------------------------------------
# Boot
# -----------------------------------------------------------------------------
runner, qi = boot_system(dim=dim, run_training=do_train)


# -----------------------------------------------------------------------------
# Sidebar — stats (setelah boot)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.subheader("📊 Statistik")
    stats = qi.get_stats()
    st.metric("Total query", stats["total_queries"])
    st.metric("Avg confidence", f"{stats['avg_confidence']:.1%}")
    st.metric("Feedback diterima", stats["feedback_count"])
    if stats["qtype_distribution"]:
        st.caption("Distribusi tipe pertanyaan:")
        st.json(dict(stats["qtype_distribution"]), expanded=False)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("🧠 FHRR Chat Interface")
st.caption("Tanya apa saja tentang dataset — sistem akan memakai HRR + KG + topology untuk menjawab.")


# -----------------------------------------------------------------------------
# Chat state
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Halo! Saya FHRR Assistant. Silakan tanya — contoh: *Siapa yang membangun benteng?* atau *Apa yang dilakukan singa?*",
        }
    ]


# -----------------------------------------------------------------------------
# Quick examples
# -----------------------------------------------------------------------------
EXAMPLES = [
    "Siapa yang memotong kayu?",
    "Apa yang dimakan singa?",
    "Di mana petani bekerja?",
    "Mengapa kayu terbakar?",
    "Apakah air membasahi kain?",
]
with st.expander("💡 Contoh pertanyaan", expanded=False):
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["pending_query"] = ex
            st.rerun()


# -----------------------------------------------------------------------------
# Render history
# -----------------------------------------------------------------------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            with st.expander("🔍 Detail jawaban", expanded=False):
                st.json(msg["meta"])


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
prompt = st.chat_input("Ketik pertanyaan...")
if "pending_query" in st.session_state and not prompt:
    prompt = st.session_state.pop("pending_query")


def answer_with_feedback(user_query: str):
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Berpikir..."):
            t0 = time.time()
            try:
                result = qi.ask(user_query, explain=explain_mode)
            except Exception as e:
                err = f"⚠️ Error saat memproses: `{e}`"
                st.error(err)
                st.session_state["messages"].append({"role": "assistant", "content": err})
                return
            elapsed = time.time() - t0

        body = f"**{result.answer}**\n\n"
        body += f"_Keyakinan: {result.confidence:.1%} · Mekanisme: `{result.mechanism}` · {elapsed*1000:.0f} ms_"
        if explain_mode and result.explanation:
            body += f"\n\n```\n{result.explanation}\n```"
        if result.suggested_followup:
            body += "\n\n**Lanjutan yang disarankan:**\n" + "\n".join(f"- {s}" for s in result.suggested_followup)
        st.markdown(body)

        meta = None
        if show_debug:
            meta = {
                "query": result.query,
                "answer": result.answer,
                "confidence": result.confidence,
                "mechanism": result.mechanism,
                "reasoning": result.reasoning,
                "related_entities": result.related_entities,
            }
            with st.expander("🔍 Detail jawaban", expanded=False):
                st.json(meta)

        st.session_state["messages"].append({"role": "assistant", "content": body, "meta": meta})


if prompt:
    answer_with_feedback(prompt)
