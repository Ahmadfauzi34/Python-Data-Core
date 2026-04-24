import streamlit as st

from fhrr_core import (
    FHRRResearchRunner, extend_engine_open_vocab,
    KnowledgeGraphIngestor, SelfSupervisedDiscovery, FHRRTopologicalLayer,
    SelfImprovementEngine, FHRRQueryInterface,
    load_dataset, list_datasets, ingest_dataset_to_kg, validate_dataset,
)

st.set_page_config(page_title="FHRR Chat", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner="Inisialisasi sistem FHRR...")
def init_fhrr_system(dataset_name: str, dim: int = 4096):
    dataset = load_dataset(dataset_name, strict=False)

    runner = FHRRResearchRunner(dim=dim)
    runner.load_dataset(dataset)

    open_vocab = extend_engine_open_vocab(runner.engine)
    kg = KnowledgeGraphIngestor(runner.engine, open_vocab)
    n_triples = ingest_dataset_to_kg(kg, dataset)

    discoverer = SelfSupervisedDiscovery(runner.engine, window_size=3)
    topo = FHRRTopologicalLayer(runner.engine)
    runner.attach_topology(topo)
    improver = SelfImprovementEngine(runner.engine, topo, discoverer, kg)

    api = FHRRQueryInterface(runner)
    api.attach_kg(kg).attach_discoverer(discoverer).attach_improver(improver)
    return api, dataset, n_triples


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ FHRR")

    available = list_datasets()
    if not available:
        st.error("Tidak ada dataset di `fhrr_project/data/datasets/`.")
        st.stop()
    dataset_name = st.selectbox("Dataset", available, index=0)

    if st.button("🔄 Re-init", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.pop("messages", None)
        st.session_state.pop("pending_rules", None)
        st.rerun()

    if st.button("🗑️ Bersihkan chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    st.divider()
    if st.toggle("Validasi dataset (debug)", value=False):
        ds_preview = load_dataset(dataset_name, strict=False)
        issues = validate_dataset(ds_preview)
        st.caption(f"{sum(1 for i in issues if i.severity == 'error')} error · {sum(1 for i in issues if i.severity == 'warning')} warning")
        with st.expander("Detail", expanded=False):
            for i in issues[:50]:
                st.write(str(i))


# -----------------------------------------------------------------------------
# Boot
# -----------------------------------------------------------------------------
api, dataset, n_triples = init_fhrr_system(dataset_name)


with st.sidebar:
    st.divider()
    st.subheader("📊 Stats")
    st.metric("Observation", len(dataset.get("observations", [])))
    st.metric("QA pairs", len(dataset.get("qa_pairs", [])))
    st.metric("KG triples", n_triples)

    st.markdown("---")
    st.subheader("Otonomi Kognitif")

    with st.expander("💤 Fase Tidur (Konsolidasi)", expanded=False):
        st.write("Menganalisis memori untuk menemukan pola baru.")
        if st.button("Jalankan Analisis", use_container_width=True):
            with st.spinner("Mengkonsolidasi..."):
                new_rules = api.runner.sleep_and_consolidate(dry_run=True)
                st.session_state['pending_rules'] = new_rules
                if not new_rules:
                    st.info("Tidak ada aturan baru yang ditemukan saat ini.")

        if 'pending_rules' in st.session_state and st.session_state['pending_rules']:
            st.write("Aturan Ditemukan:")
            for r in st.session_state['pending_rules']:
                st.code(f"Jika {r['premise']} -> {r['conclusion']} (Conf: {r['confidence']})")
            if st.button("Simpan Permanen ke .auto.yaml", type="primary", use_container_width=True):
                api.runner.consolidator.persist_rules_to_dataset(st.session_state['pending_rules'])
                st.success("Tersimpan!")
                st.session_state['pending_rules'] = []

    with st.expander("👁️ Simulasi Sandbox", expanded=False):
        st.write("Proyeksi tindakan sebelum bertindak.")
        sim_goal = st.text_input("Goal (Aksi yang diinginkan)", value="sapa")
        col1, col2 = st.columns(2)
        with col1:
            act_1 = st.text_input("Opsi Aksi 1", value="sapa")
            act_1_tgt = st.text_input("Target Aksi 1 (Opsional)", value="budi")
        with col2:
            act_2 = st.text_input("Opsi Aksi 2", value="marah")
            act_2_tgt = st.text_input("Target Aksi 2 (Opsional)", value="budi")

        if st.button("Jalankan Simulasi", use_container_width=True):
            with st.spinner("Menjalankan proyeksi..."):
                scen_1 = {"aksi": act_1}
                if act_1_tgt: scen_1["target"] = act_1_tgt

                scen_2 = {"aksi": act_2}
                if act_2_tgt: scen_2["target"] = act_2_tgt

                best_id, best_bindings = api.runner.simulate_and_commit(
                    action_scenarios=[scen_1, scen_2],
                    goal={"aksi": sim_goal},
                    current_state={"agen": "user", "aksi": "tunggu"}
                )
                st.success(f"Skenario terpilih: {best_id} -> {best_bindings}")

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("🧠 Cognitive AI - FHRR Chat")
st.caption(f"Dataset aktif: **{dataset_name}** · Berbasis Vector Symbolic Architecture (VSA) & Topological Reasoning")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Sistem siap. Tanyakan sesuatu berdasarkan dataset (contoh: *siapa yang makan mangga?*)",
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ketik pertanyaan Anda..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Memproses vektor fasa..."):
        result = api.ask(prompt, explain=False)

    body = f"**Jawaban:** {result.answer}\n\n"
    body += f"*(Keyakinan: {result.confidence:.1%} | Mekanisme: {result.mechanism})*"
    if result.reasoning:
        body += f"\n\n**Alasan:** {result.reasoning}"
    if result.suggested_followup:
        body += "\n\n**Coba tanyakan juga:**\n" + "\n".join(f"- {s}" for s in result.suggested_followup)

    with st.chat_message("assistant"):
        st.markdown(body)
    st.session_state.messages.append({"role": "assistant", "content": body})
