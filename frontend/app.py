import streamlit as st
import requests

st.set_page_config(page_title="نظام أخبار عربي ذكي", layout="wide", page_icon="🗞️")

st.markdown("""
<style>
    .response-box { background:#1e1e2e; padding:20px; border-radius:8px; border-right:4px solid #c8a96e; direction:rtl; font-family:'Arial'; font-size:16px; line-height:1.8; color:#e8e8f0; }
    .tool-badge { background:#2a2a3e; padding:6px 14px; border-radius:20px; font-size:12px; color:#6ec8a0; border:1px solid #6ec8a0; display:inline-block; margin:4px; }
    .meta-box { background:#111118; padding:12px 16px; border-radius:6px; border:1px solid #1e1e2e; font-size:12px; color:#6b6b80; }
</style>
""", unsafe_allow_html=True)

st.title("🗞️ نظام أخبار عربي ذكي")
st.caption("Agentic RAG · Hybrid Search · LangGraph")

with st.form("query_form"):
    query = st.text_input("اكتب سؤالك بالعربية", placeholder="ما هي آخر الأخبار السياسية؟", label_visibility="collapsed")
    submitted = st.form_submit_button("بحث", use_container_width=True)

if submitted and query:
    with st.spinner("جاري البحث..."):
        try:
            resp = requests.post("http://localhost:8000/query", json={"query": query}, timeout=60)
            data = resp.json()

            st.markdown(f'<div class="response-box">{data["response"]}</div>', unsafe_allow_html=True)

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Tool selected**")
                st.markdown(f'<span class="tool-badge">{data["tool_choice"]}</span>', unsafe_allow_html=True)
            with col2:
                st.markdown("**Retrieval loops**")
                st.markdown(f'<div class="meta-box">{data["loop_count"]} loop(s)</div>', unsafe_allow_html=True)
            with col3:
                st.markdown("**Status**")
                st.markdown('<div class="meta-box">✓ Complete</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")