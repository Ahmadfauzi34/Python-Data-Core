import streamlit as st
import numpy as np

# Import semua sistem dari file core Anda
# Pastikan nama-nama class ini sesuai dengan yang ada di fhrr_core.py
from fhrr_core import (
    FHRRResearchRunner, fhrr_research_dataset, extend_engine_open_vocab,
    KnowledgeGraphIngestor, SelfSupervisedDiscovery, FHRRTopologicalLayer,
    SelfImprovementEngine, FHRRQueryInterface, ingest_dataset_to_kg,
)

# Gunakan cache agar AI tidak di-rebuild dari nol setiap kali user mengetik
@st.cache_resource
def init_fhrr_system():
    # Setup sistem persis seperti di Cell 18
    runner = FHRRResearchRunner(dim=4096)
    runner.load_dataset(fhrr_research_dataset)
    
    open_vocab = extend_engine_open_vocab(runner.engine)
    kg = KnowledgeGraphIngestor(runner.engine, open_vocab)

    # Auto-ingest semua observation dataset jadi triple KG
    ingest_dataset_to_kg(kg, fhrr_research_dataset)
    
    discoverer = SelfSupervisedDiscovery(runner.engine, window_size=3)
    topo = FHRRTopologicalLayer(runner.engine)
    runner.attach_topology(topo)
    improver = SelfImprovementEngine(runner.engine, topo, discoverer, kg)
    
    api = FHRRQueryInterface(runner)
    api.attach_kg(kg)
    api.attach_discoverer(discoverer)
    api.attach_improver(improver)
    
    return api

# Muat sistem AI
api = init_fhrr_system()

# Tampilan UI
st.title("🧠 Cognitive AI - FHRR Chat")
st.caption("Berbasis Vector Symbolic Architecture (VSA) & Topological Reasoning")

# Setup memori riwayat chat agar tidak hilang
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Pesan sambutan
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Sistem siap. Tanyakan sesuatu berdasarkan dataset saya (contoh: 'siapa yang makan mangga?')"
    })

# Tampilkan riwayat chat di layar
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Kolom input text di bawah layar
if prompt := st.chat_input("Ketik pertanyaan Anda di sini..."):
    
    # 1. Tampilkan pertanyaan user
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Proses pertanyaan ke dalam FHRR Query Interface (Cell 17)
    with st.spinner("Memproses vektor fasa..."):
        result = api.ask(prompt, explain=False)
    
    # 3. Format balasan dari sistem
    response_text = f"**Jawaban:** {result.answer}\n\n"
    response_text += f"*(Keyakinan: {result.confidence:.1%} | Mekanisme: {result.mechanism})*"
    
    if result.reasoning:
        response_text += f"\n\n**Alasan:** {result.reasoning}"
        
    if result.suggested_followup:
        response_text += "\n\n**Coba tanyakan juga:**\n"
        for fw in result.suggested_followup:
            response_text += f"- {fw}\n"

    # 4. Tampilkan balasan AI
    with st.chat_message("assistant"):
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
