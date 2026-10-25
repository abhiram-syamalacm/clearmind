"""Shared ClearMind branding for every page. Colors match the logo."""
import base64
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"


def setup_page(title):
    st.set_page_config(page_title=title, page_icon=str(ASSETS / "mark.png"), layout="centered")
    st.logo(str(ASSETS / "wordmark.png"), size="large", icon_image=str(ASSETS / "mark.png"))
    # Brand styling: wordmark header, band pills, risk card.
    st.markdown("""
<style>
    .cm-brand { display:flex; align-items:center; gap:.9rem; margin:.25rem 0 1.25rem; }
    .cm-brand img { width:64px; height:64px; }
    .cm-word { font-family:'Quicksand',sans-serif; font-weight:600; font-size:2.6rem;
               line-height:1; letter-spacing:-.01em; }
    .cm-word .c { color:#1557F4; } .cm-word .m { color:#FFC718; }
    .cm-sub { font-family:'Quicksand',sans-serif; font-size:.8rem; font-weight:600;
              letter-spacing:.45em; text-transform:uppercase; color:#1557F4; margin-top:.35rem; }
    .cm-card { border:1px solid rgba(21,87,244,.18); border-radius:1rem; padding:1.1rem 1.3rem;
               background:linear-gradient(135deg, rgba(21,87,244,.07), rgba(255,199,24,.07)); }
    .cm-risk { font-family:'Quicksand',sans-serif; font-weight:700; font-size:3rem; line-height:1; }
    .cm-pill { display:inline-block; padding:.3rem .9rem; border-radius:999px; font-weight:700;
               font-size:.85rem; letter-spacing:.08em; margin-left:.75rem; vertical-align:middle; }
    .cm-pill.low { background:#1557F4; color:#fff; }
    .cm-pill.medium { background:#FFC718; color:#14213D; }
    .cm-pill.high { background:#E5484D; color:#fff; }
    @media (prefers-color-scheme: dark) { .cm-word .c, .cm-sub { color:#4F80FF; } }
    </style>
    """, unsafe_allow_html=True)


def brand_header(subtitle="for students"):
    mark = base64.b64encode((ASSETS / "mark.png").read_bytes()).decode()
    st.markdown(f"""
<div class="cm-brand">
  <img src="data:image/png;base64,{mark}" alt="">
  <div><div class="cm-word"><span class="c">clear</span><span class="m">mind</span></div>
  <div class="cm-sub">{subtitle}</div></div>
</div>""", unsafe_allow_html=True)
