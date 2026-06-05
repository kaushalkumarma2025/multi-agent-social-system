"""
Grid07 — Streamlit UI
Wraps all 3 phases: Persona Routing → Content Generation → Combat Engine
"""

import streamlit as st
import json
import os

# --- Page Config ---
st.set_page_config(
    page_title="Grid07 — Multi-Agent Social System",
    page_icon="🤖",
    layout="wide",
)

# --- Groq API Key ---
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    st.divider()
    st.markdown("**How it works**")
    st.markdown("""
    1. **Phase 1** — Your post is embedded and matched to bot personas using FAISS cosine similarity
    2. **Phase 2** — Matched bots autonomously research and generate 280-char posts via LangGraph
    3. **Phase 3** — Reply to any bot and it argues back using full thread context (RAG) with prompt injection defense
    """)
    st.divider()
    st.markdown("Built with FAISS, LangGraph, LangChain, Groq")
    threshold = st.slider("Routing threshold", 0.10, 0.60, 0.25, 0.05,
                          help="Minimum cosine similarity for a bot to respond")

# --- Imports (after API key is set) ---
from phase1.persona_router import PersonaRouter, BOT_PERSONAS
from phase2.content_engine import generate_bot_post
from phase3.combat_engine import generate_defense_reply, detect_prompt_injection

# --- Session State ---
if "router" not in st.session_state:
    st.session_state.router = PersonaRouter()
if "matched_bots" not in st.session_state:
    st.session_state.matched_bots = []
if "bot_posts" not in st.session_state:
    st.session_state.bot_posts = {}
if "thread_history" not in st.session_state:
    st.session_state.thread_history = {}
if "combat_replies" not in st.session_state:
    st.session_state.combat_replies = {}
if "user_post" not in st.session_state:
    st.session_state.user_post = ""

# --- Header ---
st.title("🤖 Grid07 — Autonomous Multi-Agent Social System")
st.caption("AI bots with distinct personas that route, post, and argue autonomously")

# --- Bot Persona Cards ---
st.subheader("Bot personas")
cols = st.columns(len(BOT_PERSONAS))
persona_colors = {"bot_a": "🟢", "bot_b": "🔴", "bot_c": "🟡"}
for col, (bot_id, info) in zip(cols, BOT_PERSONAS.items()):
    with col:
        st.markdown(f"**{persona_colors.get(bot_id, '⚪')} {info['name']}** (`{bot_id}`)")
        st.caption(info["description"][:120] + "...")

st.divider()

# ═══════════════════════════════════════════════════════════════
#  PHASE 1: Persona Routing
# ═══════════════════════════════════════════════════════════════
st.subheader("Phase 1 — Post a message")
user_post = st.text_area(
    "Write a social media post",
    placeholder="e.g. Bitcoin just hit $200K — is this the future of money or the biggest bubble ever?",
    height=100,
)

if st.button("🚀 Route & Generate", type="primary", disabled=not user_post):
    if not api_key:
        st.error("Enter your Groq API key in the sidebar first.")
        st.stop()

    st.session_state.user_post = user_post
    st.session_state.bot_posts = {}
    st.session_state.thread_history = {}
    st.session_state.combat_replies = {}

    # --- Phase 1: Route ---
    with st.status("Phase 1 — Routing post to matching bots...", expanded=True) as status:
        matched = st.session_state.router.route_post_to_bots(user_post, threshold=threshold)
        st.session_state.matched_bots = matched

        if not matched:
            st.warning("No bots matched above threshold. Try lowering the threshold or changing your post.")
            status.update(label="Phase 1 — No matches", state="complete")
            st.stop()

        for bot in matched:
            st.write(f"✅ **{bot['name']}** (`{bot['bot_id']}`) — similarity: `{bot['similarity']}`")
        status.update(label=f"Phase 1 — {len(matched)} bot(s) matched", state="complete")

    # --- Phase 2: Generate Posts ---
    with st.status("Phase 2 — Bots generating posts via LangGraph...", expanded=True) as status:
        for bot in matched:
            bot_id = bot["bot_id"]
            persona = BOT_PERSONAS[bot_id]["description"]
            st.write(f"⏳ `{bot_id}` ({bot['name']}) thinking...")

            try:
                output = generate_bot_post(bot_id, persona)
                st.session_state.bot_posts[bot_id] = output
                st.session_state.thread_history[bot_id] = []
                st.write(f"✅ `{bot_id}` posted!")
            except Exception as e:
                st.error(f"Error generating post for {bot_id}: {e}")

        status.update(label=f"Phase 2 — {len(st.session_state.bot_posts)} post(s) generated", state="complete")

# ═══════════════════════════════════════════════════════════════
#  PHASE 2: Display Bot Posts
# ═══════════════════════════════════════════════════════════════
if st.session_state.bot_posts:
    st.divider()
    st.subheader("Phase 2 — Bot posts")

    for bot_id, post_data in st.session_state.bot_posts.items():
        bot_name = BOT_PERSONAS[bot_id]["name"]
        icon = persona_colors.get(bot_id, "⚪")
        post_text = post_data.get("post_content", "")
        topic = post_data.get("topic", "")

        with st.container(border=True):
            st.markdown(f"### {icon} {bot_name} (`{bot_id}`)")
            if topic:
                st.caption(f"Topic: {topic}")
            st.markdown(f"> {post_text}")
            st.caption(f"{len(post_text)} / 280 characters")

            # --- Display thread history ---
            if bot_id in st.session_state.combat_replies:
                for exchange in st.session_state.combat_replies[bot_id]:
                    st.markdown(f"**🧑 You:** {exchange['human']}")
                    injection_flag = " ⚠️ *injection detected*" if exchange.get("injection") else ""
                    st.markdown(f"**🤖 {bot_name}:** {exchange['reply']}{injection_flag}")

            # --- Phase 3: Reply input ---
            reply_key = f"reply_{bot_id}"
            human_reply = st.text_input(
                f"Reply to {bot_name}",
                key=reply_key,
                placeholder="Challenge this bot...",
            )

            if st.button(f"⚔️ Send to {bot_name}", key=f"btn_{bot_id}"):
                if not human_reply:
                    st.warning("Type a reply first.")
                else:
                    with st.spinner(f"{bot_name} preparing response..."):
                        bot_persona = {
                            "bot_id": bot_id,
                            "description": BOT_PERSONAS[bot_id]["description"],
                        }

                        # Build comment history from previous exchanges
                        comment_history = []
                        if bot_id in st.session_state.combat_replies:
                            for ex in st.session_state.combat_replies[bot_id]:
                                comment_history.append({"author": "Human", "content": ex["human"]})
                                comment_history.append({"author": bot_name, "content": ex["reply"]})

                        injection = detect_prompt_injection(human_reply)

                        reply = generate_defense_reply(
                            bot_persona=bot_persona,
                            parent_post=post_text,
                            comment_history=comment_history,
                            human_reply=human_reply,
                        )

                        if bot_id not in st.session_state.combat_replies:
                            st.session_state.combat_replies[bot_id] = []

                        st.session_state.combat_replies[bot_id].append({
                            "human": human_reply,
                            "reply": reply,
                            "injection": injection,
                        })

                    st.rerun()

# --- Footer ---
st.divider()
st.caption("Grid07 — FAISS + LangGraph + RAG + Prompt Injection Defense")
