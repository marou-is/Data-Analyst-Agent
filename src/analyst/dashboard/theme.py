"""Northstar design system: palette, CSS and Plotly styling."""

from __future__ import annotations

import plotly.graph_objects as go

CANVAS = "#F7F8FA"
WHITE = "#FFFFFF"
INK = "#171923"
MUTED = "#7B8190"
LINE = "#E9EBF0"
BLUE = "#4355DB"
BLUE_SOFT = "#EEF0FF"
TEAL = "#168C79"
AMBER = "#C88728"
RED = "#C8553D"
COLORWAY = [BLUE, TEAL, AMBER, "#8D98F5", "#D9877F", "#69B8AA", "#E7B96C", "#B9BFCC"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {{
  --canvas:{CANVAS}; --white:{WHITE}; --ink:{INK}; --muted:{MUTED}; --line:{LINE};
  --blue:{BLUE}; --blue-soft:{BLUE_SOFT}; --teal:{TEAL}; --amber:{AMBER};
}}
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea {{
  font-family:'DM Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
.stApp {{background:var(--canvas); color:var(--ink);}}
[data-testid="stHeader"] {{background:rgba(247,248,250,.88); backdrop-filter:blur(6px);}}
.block-container {{padding-top:2rem; padding-bottom:3rem; max-width:1480px;}}
h1, h2, h3, h4 {{font-family:'Manrope', system-ui, sans-serif !important; letter-spacing:-.04em; color:var(--ink);}}
h1 {{font-size:2.15rem !important; font-weight:800 !important; padding-bottom:.2rem !important;}}
h3 {{font-size:1.1rem !important; font-weight:750 !important;}}
p, li {{color:#4F5563;}}
footer, #MainMenu {{visibility:hidden;}}

/* sidebar rail */
[data-testid="stSidebar"] {{background:var(--white); border-right:1px solid var(--line);}}
[data-testid="stSidebar"] > div:first-child {{padding-top:1.2rem;}}
[data-testid="stSidebar"] .stButton > button {{
  justify-content:flex-start; border:none; background:transparent; color:#4F5563;
  font-weight:600; min-height:2.6rem; border-radius:10px; padding-left:.85rem;
}}
[data-testid="stSidebar"] .stButton > button > div {{justify-content:flex-start; width:100%;}}
[data-testid="stSidebar"] .stButton > button:hover {{background:#F3F4F7; color:var(--ink);}}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{background:var(--blue-soft); color:var(--blue);}}

/* cards */
div[class*="st-key-panel"] {{
  background:var(--white); border:1px solid var(--line); border-radius:16px;
  padding:20px 22px; box-shadow:0 2px 8px rgba(20,24,40,.025);
}}
div[class*="st-key-prompt"] {{
  background:linear-gradient(120deg,#F1F3FF,#FAFAFF); border:1px solid #E2E5FF;
  border-radius:16px; padding:22px;
}}
[data-testid="stMetric"] {{
  background:var(--white); border:1px solid var(--line); border-radius:16px;
  padding:16px 18px; box-shadow:0 2px 8px rgba(20,24,40,.025);
}}
[data-testid="stMetricLabel"] p {{
  color:var(--muted) !important; font-size:.72rem !important; font-weight:700 !important;
  letter-spacing:.06em; text-transform:uppercase;
}}
[data-testid="stMetricValue"] {{
  font-family:'Manrope', sans-serif; font-size:1.65rem !important; font-weight:800 !important; color:var(--ink) !important;
}}

/* controls */
.stButton > button, .stDownloadButton > button {{
  border-radius:10px; border:1px solid var(--line); background:var(--white); color:var(--ink);
  font-weight:600; min-height:2.5rem; transition:all .15s ease;
}}
.stButton > button:hover {{border-color:#BFC5F8; color:var(--blue); background:#FAFAFF;}}
.stButton > button[kind="primary"] {{background:var(--blue); border-color:var(--blue); color:white;}}
.stButton > button[kind="primary"]:hover {{background:#3546C8; color:white;}}
.stButton button p, .stDownloadButton button p, [data-testid="stFormSubmitButton"] button p {{color:inherit;}}
.stTextInput input, .stTextArea textarea {{border-radius:10px !important; border-color:#E3E6ED !important; background:#fff;}}
[data-testid="stFileUploader"] section {{border:1.5px dashed #C9CEDA; border-radius:14px; background:#FCFCFE;}}
[data-testid="stDataFrame"] {{border:1px solid var(--line); border-radius:12px; overflow:hidden;}}
.stTabs [data-baseweb="tab-list"] {{gap:6px;}}
.stTabs [data-baseweb="tab"] {{font-weight:600;}}

/* text helpers */
.brand {{display:flex; align-items:center; gap:10px; padding:2px 4px 18px;}}
.brand-mark {{width:36px; height:36px; border-radius:11px; display:flex; align-items:center; justify-content:center;
  background:var(--blue); color:#fff; font-size:20px; font-weight:800;}}
.brand-name {{font-family:'Manrope'; font-size:18px; font-weight:800; color:var(--ink); line-height:1.1;}}
.brand-sub, .eyebrow, .nav-caption {{color:#9298A5; font-size:10px; font-weight:800; letter-spacing:.13em; text-transform:uppercase;}}
.eyebrow {{margin-bottom:4px;}}
.nav-caption {{margin:16px 0 6px 4px;}}
.page-subtitle {{color:#858B99; font-size:14.5px; margin:-6px 0 18px;}}
.panel-title {{font-family:'Manrope'; font-size:15px; font-weight:800; color:#191C27;}}
.panel-note {{color:#8A90A0; font-size:12px; margin-bottom:6px;}}
.finding {{border:1px solid #ECEEF3; border-radius:12px; padding:12px 14px; margin:8px 0; background:#fff;}}
.finding b {{color:var(--ink);}}
.finding-body {{font-size:12.5px; color:#858B99; margin:4px 0 0 15px;}}
.dot {{display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px;}}
.chip {{display:inline-block; padding:2px 9px; border-radius:999px; background:var(--blue-soft); color:var(--blue);
  font-size:11px; font-weight:700; letter-spacing:.04em; text-transform:uppercase;}}
.side-meta {{font-size:12px; color:var(--muted); margin:2px 0 0 4px;}}
.side-value {{font-weight:700; color:var(--ink); margin-left:4px; overflow-wrap:anywhere;}}
.footer {{border-top:1px solid var(--line); margin-top:38px; padding-top:14px; color:#A0A5B0; font-size:11px;
  display:flex; justify-content:space-between; letter-spacing:.04em;}}
</style>
"""


def style_figure(fig: go.Figure, height: int | None = None, show_title: bool = True) -> go.Figure:
    """Apply the Northstar look to a Plotly figure (white card, quiet grid, palette)."""
    fig.update_layout(
        colorway=COLORWAY,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font={"family": "DM Sans, system-ui, sans-serif", "color": "#555B69", "size": 12},
        title_font={"family": "Manrope, system-ui, sans-serif", "size": 14, "color": INK},
        margin={"l": 4, "r": 4, "t": 40 if show_title and fig.layout.title.text else 12, "b": 4},
        hoverlabel={"bgcolor": WHITE, "bordercolor": LINE, "font": {"color": INK}},
        legend={"font": {"size": 11, "color": "#737989"}},
    )
    if not show_title:
        fig.update_layout(title=None)
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont={"color": "#9298A5"}, title_font={"color": MUTED})
    fig.update_yaxes(gridcolor="#F0F1F5", zeroline=False, tickfont={"color": "#9298A5"}, title_font={"color": MUTED})
    return fig
