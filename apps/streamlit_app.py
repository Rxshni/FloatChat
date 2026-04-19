import streamlit as st
import time
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.engine.ocean_engine import execute_hybrid_workflow, prediction_models, valid_regions, valid_depths
from src.engine.ocean_map import build_ocean_map
from src.engine.query_cache import load_cache

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloatChat",
    page_icon="🌊",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1E3A8A;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #64748B;
        font-style: italic;
        margin-bottom: 20px;
        font-size: 0.95rem;
    }
    .route-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .badge-sql      { background: #DBEAFE; color: #1E40AF; }
    .badge-vector   { background: #D1FAE5; color: #065F46; }
    .badge-hybrid   { background: #FEF3C7; color: #92400E; }
    .badge-predict  { background: #EDE9FE; color: #5B21B6; }
    .stat-card {
        background: #1E293B;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        color: #F1F5F9;
    }
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm **FloatChat** 🌊\n\n"
                "I can help you with:\n"
                "- Historical ocean data (temperature, salinity, oxygen, chlorophyll)\n"
                "- Marine biology and fishing zone recommendations\n"
                "- Future condition forecasts (temp & salinity via ML, O₂ & Chla via climatology)\n\n"
                "Try asking: *\"Where should I fish for tuna in May?\"* or "
                "*\"What will the Arabian Sea surface temperature be in August 2026?\"*"
            )
        }
    ]
if "routes"            not in st.session_state: st.session_state.routes            = []
if "charts"            not in st.session_state: st.session_state.charts            = {}
if "chart_store"       not in st.session_state: st.session_state.chart_store       = {}
if "selected_variable" not in st.session_state: st.session_state.selected_variable = "Temperature"

# ── ROUTE BADGE HELPER ────────────────────────────────────────────────────────
ROUTE_BADGE = {
    "SQL_ONLY":    ('<span class="route-badge badge-sql">🔵 SQL Query</span>',         "Querying historical database..."),
    "VECTOR_ONLY": ('<span class="route-badge badge-vector">🟢 Biology Search</span>', "Searching marine biology knowledge..."),
    "HYBRID":      ('<span class="route-badge badge-hybrid">🟡 Hybrid Analysis</span>',"Combining data + biology knowledge..."),
    "PREDICTION":  ('<span class="route-badge badge-predict">🟣 ML Forecast</span>',   "Running prediction models..."),
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌊 FloatChat")
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate",
        options=["💬 Chat", "🗺️ Ocean Map"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # System status
    st.markdown("**System Status**")
    st.markdown(
        f'<div class="stat-card">🤖 LLM: <b>Groq</b> — Llama 3.3 70B</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="stat-card">🗄️ SQL DB: <b>Neon PostgreSQL</b><br>'
        f'Regions: {", ".join(valid_regions)}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="stat-card">🧠 Vector DB: <b>Pinecone</b></div>',
        unsafe_allow_html=True
    )
    

    # Cache stats
    try:
        cache = load_cache()
        total_hits = sum(e.get("hits", 0) for e in cache.values())
        st.markdown(
            f'<div class="stat-card">⚡ Cache: <b>{len(cache)} entries</b><br>'
            f'Queries saved: <b>{total_hits}</b></div>',
            unsafe_allow_html=True
        )
    except Exception:
        pass

    st.markdown("---")

    # Capabilities
    st.markdown("**What I can do:**")
    st.markdown("""
- 🔵 **SQL** — Historical Argo data queries
- 🟢 **Biology** — Marine species & ecology
- 🟡 **Hybrid** — Fishing zone recommendations
- 🟣 **Predict** — Future temp & salinity forecasts
- 📊 **O₂ & Chla** — Seasonal climatology
    """)

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared. How can I help you?"}
        ]
        st.session_state.routes    = []
        st.session_state.charts    = {}
        st.session_state.chart_store = {}
        st.rerun()

    if st.button("🗑️ Clear Cache", use_container_width=True):
        import os
        cache_file = Path(__file__).resolve().parents[1] / "data" / "query_cache.json"
        if os.path.exists(cache_file):
            os.remove(cache_file)
        st.success("Cache cleared!")
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#94A3B8;'>Data: Argo Float Program<br>"
        "Coverage: Arabian Sea & Bay of Bengal<br>2020 – present</div>",
        unsafe_allow_html=True
    )

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("<h1 class='main-title'>🌊 FloatChat</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-title'>Argo Ocean Intelligence — SQL · Biology · Prediction</p>",
    unsafe_allow_html=True
)

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — CHAT
# ═════════════════════════════════════════════════════════════════════════════
if page == "💬 Chat":

    # Chat input at top level — Streamlit correctly pins it to bottom
    prompt = st.chat_input("Ask about ocean data, fishing zones, or future predictions...")

    # ── History display ───────────────────────────────────────────────────────
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and i > 0:
                route_idx = (i - 1) // 2
                if route_idx < len(st.session_state.routes):
                    badge_html, _ = ROUTE_BADGE.get(
                        st.session_state.routes[route_idx], ("", "")
                    )
                    if badge_html:
                        st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown(message["content"])
            if i in st.session_state.charts:
                st.plotly_chart(
                    st.session_state.charts[i],
                    width='stretch',
                    key=f"chart_history_{i}",
                    config={"displayModeBar": True}
                )

    # ── SUGGESTED QUESTIONS — shown only when conversation is fresh ───────────
# REPLACE WITH THIS:

    # ── SUGGESTED QUESTIONS ───────────────────────────────────────────────────
    if len(st.session_state.messages) == 1:
        st.markdown(
            "<p style='color:#64748B; font-size:0.85rem; margin-bottom:6px;'>"
            "💡 Try one of these:</p>",
            unsafe_allow_html=True
        )
        suggestions = [
            "Plot the temperature in each depth zone of Arabian Sea in June 2023",
            "What are the ideal conditions for fishing tuna in the Bay of Bengal?",
            "Show the surface temperature trend of Arabian Sea across all months of 2023",
            "What will the Arabian Sea surface temperature be in August 2026?",
            "Plot the salinity across all depth zones of Bay of Bengal in March 2024",
            "What causes algal blooms in the Arabian Sea?",
        ]
        cols = st.columns(2)
        for idx, suggestion in enumerate(suggestions):
            with cols[idx % 2]:
                if st.button(suggestion, key=f"suggestion_{idx}", use_container_width=True):
                    st.session_state["pending_prompt"] = suggestion  # only set, don't append
                    st.rerun()

    # ── MERGE typed prompt and button click into one variable ─────────────────
    final_prompt = prompt or st.session_state.pop("pending_prompt", None)

    # ── HANDLE INPUT ──────────────────────────────────────────────────────────
    if final_prompt:

        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with st.chat_message("user"):
            st.markdown(final_prompt)

        q_lower = final_prompt.lower()
        if any(w in q_lower for w in ["will", "predict", "forecast", "2026", "2027", "next month", "future"]):
            likely_route = "PREDICTION"
        elif any(w in q_lower for w in ["where", "should i fish", "good for", "suitable", "recommend"]):
            likely_route = "HYBRID"
        elif any(w in q_lower for w in ["what is", "temperature", "salinity", "oxygen",
                                         "which", "when", "compare", "show", "plot", "chart"]):
            likely_route = "SQL_ONLY"
        else:
            likely_route = "VECTOR_ONLY"

        badge_html, spinner_text = ROUTE_BADGE.get(likely_route, ("", "Processing..."))

        with st.chat_message("assistant"):
            if badge_html:
                st.markdown(badge_html, unsafe_allow_html=True)

            message_placeholder = st.empty()
            full_response       = ""

            try:
                with st.spinner(spinner_text):
                    response_text, chart_figure = execute_hybrid_workflow(
                        final_prompt,                    # ← final_prompt not prompt
                        chat_history=st.session_state.messages
                    )

                full_response = str(response_text)
                sentences     = full_response.replace("\n", " \n ").split(". ")
                displayed     = ""
                for idx, sentence in enumerate(sentences):
                    displayed += sentence
                    if idx < len(sentences) - 1:
                        displayed += ". "
                    message_placeholder.markdown(displayed + "▌")
                    time.sleep(0.06)
                message_placeholder.markdown(full_response)

                chart_to_show = None
                question_key  = final_prompt.strip().lower()  # ← final_prompt not prompt
                if chart_figure is None:
                    chart_to_show = st.session_state.chart_store.get(question_key, None)
                else:
                    st.session_state.chart_store[question_key] = chart_figure
                    chart_to_show = chart_figure

                if chart_to_show is not None:
                    with st.spinner("📊 Generating plot..."):
                        time.sleep(0.3)
                    st.caption("📊 Interactive chart — hover to explore, scroll to zoom")
                    st.plotly_chart(
                        chart_to_show,
                        width='stretch',
                        key=f"chart_{int(time.time() * 1000)}",
                        config={
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "floatchat_chart",
                                "scale": 2
                            }
                        }
                    )

                msg_index = len(st.session_state.messages)
                if chart_to_show is not None:
                    st.session_state.charts[msg_index] = chart_to_show

            except Exception as e:
                full_response = (
                    f"⚠️ Something went wrong while processing your request.\n\n"
                    f"**Details:** `{str(e)}`\n\n"
                    f"Try rephrasing your question or check the terminal for more details."
                )
                message_placeholder.error(full_response)
                likely_route = "SQL_ONLY"

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.routes.append(likely_route)

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — OCEAN MAP
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Ocean Map":

    st.markdown("### 🌊 Live Ocean Conditions")
    st.markdown(
        "<p style='color:#64748B; font-size:0.9rem; margin-top:-10px;'>"
        "Latest Argo float data · Hover over highlighted regions for depth profile details · ",
        unsafe_allow_html=True
    )

    # Variable selector
    col_sel, col_spacer = st.columns([1, 4])
    with col_sel:
        selected_var = st.selectbox(
            "Select variable",
            options=["Temperature", "Salinity", "Oxygen", "Chlorophyll"],
            index=["Temperature", "Salinity", "Oxygen", "Chlorophyll"].index(
                st.session_state.selected_variable
            ),
            key="map_variable_selector",
        )
        st.session_state.selected_variable = selected_var

    # Map
    with st.spinner("🗺️ Loading ocean map..."):
        fig = build_ocean_map(selected_var)

    st.plotly_chart(
        fig,
        width='stretch',
        key=f"ocean_map_{selected_var}",
        config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "floatchat_ocean_map",
                "scale": 2
            }
        }
    )

    st.markdown(
        "<p style='color:#475569; font-size:0.75rem; text-align:center; margin-top:-10px;'>"
        "Source: Argo Float Program (ERDDAP) · "
        "Aggregated monthly averages · "
        "Depth zones: Surface / Epipelagic / Mesopelagic / Bathypelagic</p>",
        unsafe_allow_html=True
    )
