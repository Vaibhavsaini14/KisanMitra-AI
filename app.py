"""
KisanMitra AI - Multilingual Crop Advisory Chatbot for Rajasthan Farmers
Built for: 1M1B AI for Sustainability Virtual Internship (AICTE)
SDGs: SDG 2 (Zero Hunger) · SDG 13 (Climate Action) · SDG 15 (Life on Land)
Author: Vaibhav Saini | MBM University, Jodhpur
"""

import streamlit as st
from rag_engine import CropRAG
from advisor import CropAdvisor

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KisanMitra AI — Crop Advisory",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1B6B2E, #145222);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        color: white;
    }
    .advisory-card {
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid;
    }
    .card-diagnosis  { background: #fff7ed; border-color: #ea580c; }
    .card-action     { background: #f0fdf4; border-color: #166534; }
    .card-water      { background: #eff6ff; border-color: #1d4ed8; }
    .card-prevention { background: #faf5ff; border-color: #7c3aed; }
    .card-organic    { background: #f0fdf4; border-color: #16a34a; }
    .farmer-msg {
        background: #dcfce7;
        border-radius: 14px 14px 4px 14px;
        padding: 10px 14px;
        color: #14532d;
        font-size: 14px;
        margin-bottom: 8px;
        display: inline-block;
        max-width: 90%;
        float: right;
        clear: both;
    }
    .sdg-badge {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 20px;
        font-size: 11px;
        background: rgba(255,255,255,0.2);
        color: white;
        margin-right: 4px;
    }
    .source-tag {
        font-size: 12px;
        color: #6b7280;
        border-top: 1px solid #e5e7eb;
        padding-top: 6px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag" not in st.session_state:
    st.session_state.rag = CropRAG()
    st.session_state.rag.build_index()
if "advisor" not in st.session_state:
    st.session_state.advisor = CropAdvisor(st.session_state.rag)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:1.7rem;font-weight:600">🌾 KisanMitra AI</h1>
    <p style="margin:3px 0 8px;opacity:0.7;font-size:0.85rem">
        Multilingual Crop Advisory Chatbot · Rajasthan Farmers
    </p>
    <span class="sdg-badge">SDG 2: Zero Hunger</span>
    <span class="sdg-badge">SDG 13: Climate Action</span>
    <span class="sdg-badge">SDG 15: Life on Land</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌱 Crop Details / फसल विवरण")

    language = st.radio(
        "Language / भाषा",
        ["English", "हिंदी"],
        horizontal=True
    )

    crop = st.selectbox("Crop / फसल", [
        "",
        "Bajra (Pearl Millet) / बाजरा",
        "Jowar (Sorghum) / ज्वार",
        "Wheat / गेहूं",
        "Mustard (Sarson) / सरसों",
        "Moong (Green Gram) / मूंग",
        "Moth Bean / मोठ",
        "Groundnut / मूंगफली",
        "Cumin (Jeera) / जीरा",
        "Coriander (Dhaniya) / धनिया",
        "Cotton / कपास",
        "Chickpea (Chana) / चना",
        "Sesame (Til) / तिल",
        "Maize / मक्का",
        "Fennel (Saunf) / सौंफ",
        "Fenugreek (Methi) / मेथी",
    ])

    district = st.selectbox("District / जिला", [
        "",
        "Jodhpur", "Barmer", "Jaisalmer", "Bikaner", "Nagaur", "Pali",
        "Ajmer", "Bhilwara", "Chittorgarh", "Tonk", "Bundi",
        "Sikar", "Churu", "Jhunjhunu", "Hanumangarh", "Sri Ganganagar",
        "Jaipur", "Alwar", "Kota", "Udaipur", "Dungarpur", "Rajsamand"
    ])

    season = st.selectbox("Season / मौसम", [
        "",
        "Kharif (June–October) / खरीफ",
        "Rabi (November–March) / रबी",
        "Zaid (April–June) / जायद"
    ])

    st.divider()
    st.markdown("**📚 Knowledge Base**")
    st.caption("ICAR Bulletins · Rajasthan Agriculture Dept. · CAZRI Jodhpur · IMD Rainfall Data")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("""
    **Quick Examples:**
    - *Bajra leaves turning yellow in Jodhpur*
    - *बाजरा में पत्तियों पर धब्बे आ रहे हैं*
    - *Mustard aphids — organic solution needed*
    - *जीरे की फसल सूख रही है*
    """)

# ── Main Chat Area ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    # Display chat history
    chat_container = st.container()

    with chat_container:
        if not st.session_state.messages:
            st.info(
                "🌱 **Namaste, Kisan!** Select your crop, district, and season from the sidebar, "
                "then describe your problem below in Hindi or English."
                if language == "English" else
                "🌱 **नमस्ते, किसान!** बाईं ओर से फसल, जिला और मौसम चुनें, फिर नीचे अपनी समस्या लिखें।"
            )

        for msg in st.session_state.messages:
            if msg["role"] == "farmer":
                st.markdown(
                    f'<div class="farmer-msg">{msg["content"]}</div><div style="clear:both"></div>',
                    unsafe_allow_html=True
                )
            else:
                response = msg["content"]
                with st.container():
                    col_a, col_b = st.columns([0.05, 0.95])
                    with col_a:
                        st.markdown("🌾")
                    with col_b:
                        st.markdown(f"**{response.get('summary', 'KisanMitra Advisory')}**")

                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(
                                f'<div class="advisory-card card-diagnosis">'
                                f'<small><b>🔍 DIAGNOSIS</b></small><br>{response.get("diagnosis", "—")}'
                                f'</div>', unsafe_allow_html=True
                            )
                            st.markdown(
                                f'<div class="advisory-card card-water">'
                                f'<small><b>💧 WATER ADVISORY</b></small><br>{response.get("water", "—")}'
                                f'</div>', unsafe_allow_html=True
                            )
                        with c2:
                            st.markdown(
                                f'<div class="advisory-card card-action">'
                                f'<small><b>✅ IMMEDIATE ACTION</b></small><br>{response.get("action", "—")}'
                                f'</div>', unsafe_allow_html=True
                            )
                            st.markdown(
                                f'<div class="advisory-card card-prevention">'
                                f'<small><b>🛡 PREVENTION</b></small><br>{response.get("prevention", "—")}'
                                f'</div>', unsafe_allow_html=True
                            )

                        if response.get("organic"):
                            st.markdown(
                                f'<div class="advisory-card card-organic">'
                                f'<small><b>🌿 ORGANIC ALTERNATIVE</b></small><br>{response.get("organic")}'
                                f'</div>', unsafe_allow_html=True
                            )

                        if response.get("variety"):
                            st.info(f"🌱 **Recommended Variety:** {response['variety']}")

                        st.markdown(
                            f'<div class="source-tag">📚 Sources: {response.get("sources", "ICAR · Rajasthan Agriculture Dept.")}</div>',
                            unsafe_allow_html=True
                        )
                st.divider()

    # ── Input Area ─────────────────────────────────────────────────────────────
    problem = st.text_area(
        "Describe your problem / अपनी समस्या लिखें",
        placeholder=(
            "e.g. Yellow spots on bajra leaves, plants wilting near roots...\n"
            "जैसे: बाजरे की पत्तियों पर पीले धब्बे, जड़ों के पास पौधे सूख रहे हैं..."
        ),
        height=100
    )

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        ask_btn = st.button(
            "🔍 Get Advisory / सलाह लें",
            use_container_width=True,
            type="primary"
        )
    with col_btn2:
        st.caption("Powered by IBM Granite + RAG")

    if ask_btn:
        if not crop:
            st.error("Please select a crop from the sidebar." if language == "English" else "कृपया फसल चुनें।")
        elif not district:
            st.error("Please select a district." if language == "English" else "कृपया जिला चुनें।")
        elif not problem.strip():
            st.error("Please describe your problem." if language == "English" else "कृपया समस्या लिखें।")
        else:
            query = f"Crop: {crop} | District: {district} | Season: {season or 'Not specified'} | Problem: {problem}"

            st.session_state.messages.append({
                "role": "farmer",
                "content": query
            })

            with st.spinner("🤖 KisanMitra AI is analyzing your crop problem..."):
                # RAG retrieval
                context = st.session_state.rag.retrieve(crop, district, problem)
                # Generate advisory
                response = st.session_state.advisor.get_advisory(
                    crop=crop,
                    district=district,
                    season=season,
                    problem=problem,
                    context=context,
                    language=language
                )

            st.session_state.messages.append({
                "role": "ai",
                "content": response
            })
            st.rerun()

# ── Right Panel — Info ─────────────────────────────────────────────────────────
with col2:
    st.markdown("### 📊 Water Zones")
    st.markdown("""
    🔴 **Critical Arid**
    Jodhpur · Barmer · Jaisalmer
    Bikaner · Nagaur · Churu

    🟡 **Moderate Stress**
    Ajmer · Pali · Bhilwara
    Tonk · Sikar · Alwar

    🟢 **Canal Irrigated**
    Hanumangarh · Sri Ganganagar

    🔵 **Good Rainfall**
    Kota · Udaipur · Dungarpur
    """)

    st.divider()
    st.markdown("### 🌾 Crop Seasons")
    st.markdown("""
    **खरीफ (Kharif)**
    Bajra · Jowar · Moong
    Groundnut · Cotton · Til

    **रबी (Rabi)**
    Wheat · Mustard · Jeera
    Chana · Dhaniya · Saunf

    **जायद (Zaid)**
    Moong · Watermelon
    Vegetables
    """)

    st.divider()
    st.markdown("### 📞 Helplines")
    st.markdown("""
    🌾 Kisan Call Centre: **1800-180-1551**
    🌧 IMD Weather: **1800-180-1717**
    💧 Water Dept: **1800-180-6234**
    """)
