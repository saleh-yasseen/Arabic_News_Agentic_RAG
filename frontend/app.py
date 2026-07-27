import streamlit as st
import requests
import json

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

with st.expander("⚙️ إعدادات متقدمة"):
    adv_col1, adv_col2 = st.columns(2)

    with adv_col1:
        selected_model = st.selectbox(
            "النموذج",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            help="قارن الجودة مقابل السرعة بين النموذجين"
        )
        tool_override = st.selectbox(
            "فرض أداة معينة",
            ["تلقائي (يقرر الوكيل)", "search_news", "summarize_topic", "compare_timeline", "answer_direct"]
        )
        top_k = st.slider("عدد المصادر المسترجعة (top_k)", 1, 15, 5, help="عدد أكبر = سياق أوسع لكن أقل تركيزاً")
        category_filter_ui = st.selectbox("تصفية حسب الفئة", ["الكل", "Politics", "Sports", "Economy", "Technology", "Health", "Culture"])


    with adv_col2:
        retrieval_mode = st.radio(
            "وضع الاسترجاع",
            ["hybrid", "dense", "sparse"],
            horizontal=True,
            help="hybrid = دلالي + حرفي مدمج، dense = دلالي فقط، sparse = حرفي فقط"
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        use_reranker = st.toggle("تفعيل إعادة الترتيب (Reranker)", value=True)
        min_score = st.slider(
            "الحد الأدنى للصلة", 0.0, 1.0, 0.0, 0.05,
            help="يعمل بشكل أدق عند تفعيل Reranker (نطاق 0-1 واضح). بدونه يُطبَّق على درجات RRF الخام، وهي ليست بنفس المقياس البديهي."
        )

with st.form("query_form"):
    query = st.text_input("اكتب سؤالك بالعربية", placeholder="ما هي آخر الأخبار السياسية؟", label_visibility="collapsed")
    submitted = st.form_submit_button("بحث", use_container_width=True)

if submitted and query:
    status_placeholder = st.empty()
    result_data = None
    data = {
        "response": "",
        "tool_choice": "—",
        "loop_count": 0,
        "sources": [],
        "comparison": None,
    }
    try:
        request_payload = {
            "query": query,
            "model": selected_model,
            "tool_override": None if tool_override == "تلقائي (يقرر الوكيل)" else tool_override,
            "retrieval_mode": retrieval_mode,
            "temperature": temperature
        }
        resp = requests.post("http://localhost:8000/query/stream", json=request_payload, stream=True, timeout=60)
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = json.loads(line[len("data: "):])

            if payload["type"] == "status":
                status_placeholder.markdown(
                    f'<div class="meta-box">⏳ {payload["label"]}</div>',
                    unsafe_allow_html=True
                )
            elif payload["type"] == "result":
                result_data = payload
            elif payload["type"] == "error":
                st.error(f"Error: {payload['message']}")

        status_placeholder.empty()

        if result_data:
            data = result_data

        st.markdown(f'<div class="response-box">{data["response"]}</div>', unsafe_allow_html=True)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Tool selected**")
            st.markdown(f'<span class="tool-badge">{data.get("tool_choice", data.get("tool_used", "—"))}</span>', unsafe_allow_html=True)

        with col2:
            st.markdown("**Retrieval loops**")
            st.markdown(f'<div class="meta-box">{data.get("loop_count", 0)} loop(s)</div>', unsafe_allow_html=True)

        with col3:
            st.markdown("**Status**")
            st.markdown('<div class="meta-box">✓ Complete</div>', unsafe_allow_html=True)

        sources = data.get("sources", [])
        if sources:
            st.markdown("---")
            categories_used = sorted(set(s.get("category", "—") for s in sources))
            st.markdown(f"**التحليل** — {len(sources)} مصدر من الفئات: {', '.join(categories_used)}")

            with st.expander(f"عرض المصادر ({len(sources)})"):
                for i, src in enumerate(sources, 1):
                    score = src.get("score")
                    score_text = f" · score: {score:.3f}" if score is not None else ""
                    st.markdown(
                        f'<div class="meta-box" style="margin-bottom:8px;">'
                        f'<strong>{i}. [{src.get("category", "—")}]</strong>{score_text}<br>'
                        f'{src.get("text", "")[:200]}...'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        comparison = data.get("comparison")
        if comparison:
            st.markdown("---")
            st.markdown("**مقارنة الاسترجاع الهجين** — الفرق بين البحث الدلالي والحرفي قبل الدمج")

            comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)

            with comp_col1:
                st.markdown("**🧠 دلالي فقط (Dense)**")
                for i, src in enumerate(comparison.get("dense", [])[:5], 1):
                    st.markdown(
                        f'<div class="meta-box" style="margin-bottom:6px; font-size:11px;">'
                        f'{i}. {src.get("text", "")[:80]}...</div>',
                        unsafe_allow_html=True
                    )

            with comp_col2:
                st.markdown("**🔤 حرفي فقط (BM25)**")
                for i, src in enumerate(comparison.get("sparse", [])[:5], 1):
                    st.markdown(
                        f'<div class="meta-box" style="margin-bottom:6px; font-size:11px;">'
                        f'{i}. {src.get("text", "")[:80]}...</div>',
                        unsafe_allow_html=True
                    )

            with comp_col3:
                st.markdown("**⚡ مدمج (RRF Fusion)**")
                for i, src in enumerate(comparison.get("fused", [])[:5], 1):
                    st.markdown(
                        f'<div class="meta-box" style="margin-bottom:6px; font-size:11px;">'
                        f'{i}. {src.get("text", "")[:80]}...</div>',
                        unsafe_allow_html=True
                    )
            with comp_col4:
                st.markdown("**🏆 بعد إعادة الترتيب (Reranked)**")
                for i, src in enumerate(comparison.get("reranked", [])[:5], 1):
                    st.markdown(
                        f'<div class="meta-box" style="margin-bottom:6px; font-size:11px;">'
                        f'{i}. {src.get("text", "")[:80]}...</div>',
                        unsafe_allow_html=True
                    )
    except Exception as e:
        st.error(f"Error: {e}")
