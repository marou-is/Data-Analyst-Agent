"""Small Streamlit building blocks for the Northstar look."""

from __future__ import annotations

from html import escape

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from analyst.contracts import Severity
from analyst.dashboard.theme import AMBER, BLUE, RED, TEAL

SEVERITY_COLOR = {Severity.ERROR: RED, Severity.WARNING: AMBER, Severity.INFO: TEAL}


def md(text: str) -> str:
    """Markdown-safe text: Streamlit renders ``$...$`` as LaTeX."""
    return text.replace("$", r"\$")


def html_text(text: str) -> str:
    return escape(text).replace("$", "&#36;")


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="eyebrow">{html_text(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="page-subtitle">{html_text(subtitle)}</div>', unsafe_allow_html=True)


def panel(key: str) -> DeltaGenerator:
    """White card container (styled through its ``st-key-panel-*`` class)."""
    return st.container(key=f"panel-{key}")


def prompt_card(key: str) -> DeltaGenerator:
    return st.container(key=f"prompt-{key}")


def panel_heading(title: str, note: str | None = None) -> None:
    html = f'<div class="panel-title">{html_text(title)}</div>'
    if note:
        html += f'<div class="panel-note">{html_text(note)}</div>'
    st.markdown(html, unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{html_text(text)}</div>', unsafe_allow_html=True)


def finding(title: str, body: str, color: str = BLUE) -> None:
    st.markdown(
        f'<div class="finding"><span class="dot" style="background:{color}"></span><b>{html_text(title)}</b>'
        f'<div class="finding-body">{html_text(body)}</div></div>',
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        '<div class="footer"><span>NORTHSTAR ANALYTICS · AUTONOMOUS DATA ANALYST</span>'
        "<span>Designed for evidence-led decisions</span></div>",
        unsafe_allow_html=True,
    )
