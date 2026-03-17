#!/usr/bin/env python3
import io
import os
import json
import base64
import struct
import glob
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go

# ── binary format constants ──────────────────────────────────────────────────
MAGIC = 0x50474C47
HEADER_FMT = "<IHHHHIIIq"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
GH_DARK = "#0d1117"
GH_GREEN = [
    [0.00, "#161b22"],
    [0.01, "#0e4429"],
    [0.25, "#006d32"],
    [0.60, "#26a641"],
    [1.00, "#39d353"],
]

def get_local_logs():
    """Scans directory for .pagelog files and matches them to source code."""
    logs = []
    for f in glob.glob("*.pagelog"):
        base = os.path.splitext(f)[0]
        source_files = glob.glob(f"{base}.cu")
        if not source_files:
            all_matches = glob.glob(f"{base}.*")
            source_files = [m for m in all_matches if not m.endswith(".pagelog")]
        
        src = source_files[0] if source_files else None
        # Store both paths as a JSON string so the callback can read the source file
        val = json.dumps({"log": f, "src": src})
        label = f"{f} (Source: {src if src else 'None'})"
        logs.append({'label': label, 'value': val})
    return logs

def parse_binary(data_bytes):
    f = io.BytesIO(data_bytes)
    raw = f.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE: return None, None
    magic, _, l1_entries, l2_entries, l3_bytes, l1_s, l2_s, l3_s, num_leaves = struct.unpack(HEADER_FMT, raw)
    if magic != MAGIC: return None, None
    
    hdr = dict(l1_entries=l1_entries, l2_entries=l2_entries, l3_bytes=l3_bytes, 
               l1_shift=l1_s, l2_shift=l2_s, l3_shift=l3_s)
    
    leaves = {}
    for _ in range(num_leaves):
        idx_data = f.read(12)
        if len(idx_data) < 12: break
        l1, l2, offset = struct.unpack("<HHQ", idx_data)
        
        curr = f.tell()
        f.seek(offset)
        leaves[f"{l1},{l2}"] = list(f.read(l3_bytes))
        f.seek(curr)
    return hdr, leaves

# ── VA helpers (Restored to original accuracy) ───────────────────────────────
def get_l1_va(hdr, l1): return l1 << hdr["l1_shift"]
def get_l2_va(hdr, l1, l2): return (l1 << hdr["l1_shift"]) | (l2 << hdr["l2_shift"])
def get_l3_va(hdr, l1, l2, bit): return (l1 << hdr["l1_shift"]) | (l2 << hdr["l2_shift"]) | (bit << hdr["l3_shift"])

# ── Dash UI ──────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)

app.layout = html.Div(style={"backgroundColor": GH_DARK, "minHeight": "100vh", "color": "#e6edf3", "padding": "20px", "fontFamily": "monospace"}, children=[
    html.H2("Shadow Page Table Explorer"),
    
    html.Div(style={"display": "flex", "gap": "20px", "marginBottom": "20px"}, children=[
        html.Div(style={"flex": 1}, children=[
            html.Label("Load Local Log:"),
            dcc.Dropdown(id='local-file-dropdown', options=get_local_logs(), 
                         style={'color': '#000'}, placeholder="Select a file in folder...")
        ]),
        html.Div(style={"flex": 1}, children=[
            html.Label("Or Upload New:"),
            dcc.Upload(id='upload-data', children=html.Div(['Drag/Drop Log']),
                       style={'border': '1px dashed #30363d', 'borderRadius': '5px', 'textAlign': 'center', 'lineHeight': '35px'})
        ])
    ]),

    html.Div(id="breadcrumb", style={"color": "#8b949e", "marginBottom": "10px"}),
    html.Button("← Back", id="back-btn", n_clicks=0, 
                style={"backgroundColor": "#21262d", "color": "#e6edf3", "border": "1px solid #30363d", "padding": "6px 16px", "borderRadius": "6px", "cursor": "pointer", "marginBottom": "12px", "display": "none"}),
    
    dcc.Graph(id="heatmap", config={"scrollZoom": True}),
    
    # Code Preview Section
    html.Div(id="code-container", style={"marginTop": "20px", "display": "none", "border": "1px solid #30363d", "borderRadius": "6px", "padding": "15px", "backgroundColor": "#161b22"}, children=[
        html.H4("Source Code Preview", style={"margin": "0 0 10px 0", "color": "#8b949e"}),
        dcc.Markdown(
            id="source-code-preview", 
            style={"maxHeight": "400px", "overflowY": "auto", "fontSize": "12px"},
            highlight_config={"theme": "dark"} # <-- This forces the dark syntax highlighting theme
        )
    ]),
    
    dcc.Store(id="log-hdr"), 
    dcc.Store(id="log-leaves"),
    dcc.Store(id="nav-state", data={"level": 0, "l1": None, "l2": None}),
])

@app.callback(
    Output("log-hdr", "data"), Output("log-leaves", "data"), 
    Output("source-code-preview", "children"), Output("code-container", "style"),
    Input("upload-data", "contents"), Input("local-file-dropdown", "value")
)
def load_data(upload_contents, dropdown_val):
    ctx = callback_context
    if not ctx.triggered: return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    trigger = ctx.triggered[0]["prop_id"]
    
    code_style_hidden = {"display": "none"}
    code_style_visible = {"marginTop": "20px", "display": "block", "border": "1px solid #30363d", "borderRadius": "6px", "padding": "15px", "backgroundColor": "#161b22"}

    if "upload-data" in trigger and upload_contents:
        _, content_string = upload_contents.split(',')
        hdr, leaves = parse_binary(base64.b64decode(content_string))
        return hdr, leaves, "*Code preview not available for uploaded files.*", code_style_visible
    
    elif "local-file-dropdown" in trigger and dropdown_val:
        paths = json.loads(dropdown_val)
        log_path, src_path = paths.get("log"), paths.get("src")
        
        if os.path.exists(log_path):
            with open(log_path, "rb") as f:
                hdr, leaves = parse_binary(f.read())
            
            code_content = "*Source file not found.*"
            if src_path and os.path.exists(src_path):
                with open(src_path, "r") as f:
                    # Wrap in Markdown code block for syntax highlighting
                    code_content = f"```cpp\n{f.read()}\n```"
            
            return hdr, leaves, code_content, code_style_visible

    return None, None, "", code_style_hidden

@app.callback(
    Output("heatmap", "figure"), Output("nav-state", "data"),
    Output("breadcrumb", "children"), Output("back-btn", "style"),
    Input("heatmap", "clickData"), Input("back-btn", "n_clicks"),
    Input("log-hdr", "data"), Input("log-leaves", "data"),
    State("nav-state", "data")
)
def navigate(click, b_n, hdr, leaves, state):
    if not hdr or not leaves: 
        return go.Figure().update_layout(plot_bgcolor=GH_DARK, paper_bgcolor=GH_DARK), state, "Select a log file to begin", {"display": "none"}
    
    ctx = callback_context
    trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    lvl, l1, l2 = state["level"], state["l1"], state["l2"]
    
    if "back-btn" in trig:
        if lvl == 2: lvl, l2 = 1, None
        elif lvl == 1: lvl, l1 = 0, None
    elif "heatmap" in trig and click:
        pt = click["points"][0]
        if lvl == 0: l1, lvl = int(pt["y"]) * 32 + int(pt["x"]), 1
        elif lvl == 1: l2, lvl = int(pt["y"]) * 64 + int(pt["x"]), 2

    # ── RESTORED: Precise Hover Text Generation ──
    hover_text = []
    
    if lvl == 0:
        rows, cols = 16, 32
        counts = np.zeros(hdr["l1_entries"], dtype=np.int64)
        for k, v in leaves.items():
            counts[int(k.split(',')[0])] += np.unpackbits(np.array(v, dtype=np.uint8)).sum()
        grid = counts.reshape(rows, cols)
        zmax = max(1, grid.max())
        
        for r in range(rows):
            row_h = []
            for c in range(cols):
                idx = r * cols + c
                va = get_l1_va(hdr, idx)
                va_end = get_l1_va(hdr, idx + 1) - 1
                row_h.append(f"L1[{idx}]<br>{va:#018x} – {va_end:#018x}<br>{int(grid[r,c]):,} pages touched")
            hover_text.append(row_h)
            
        title, sub, crumb = "L1 — full 48-bit address space", f"{int(counts.sum()):,} total pages touched", "Root"

    elif lvl == 1:
        rows, cols = 64, 64
        counts = np.zeros(hdr["l2_entries"], dtype=np.int64)
        for k, v in leaves.items():
            k1, k2 = map(int, k.split(','))
            if k1 == l1: counts[k2] += np.unpackbits(np.array(v, dtype=np.uint8)).sum()
        grid = counts.reshape(rows, cols)
        zmax = max(1, grid.max())
        
        for r in range(rows):
            row_h = []
            for c in range(cols):
                idx = r * cols + c
                va = get_l2_va(hdr, l1, idx)
                va_end = get_l2_va(hdr, l1, idx + 1) - 1
                row_h.append(f"L1[{l1}] → L2[{idx}]<br>{va:#018x} – {va_end:#018x}<br>{int(grid[r,c]):,} pages touched")
            hover_text.append(row_h)
            
        va_base = get_l1_va(hdr, l1)
        title, sub, crumb = f"L2 — L1[{l1}]", f"base {va_base:#018x}", f"Root › L1[{l1}]"

    else:
        rows, cols = 128, 256
        bits = np.unpackbits(np.array(leaves.get(f"{l1},{l2}", [0]*hdr["l3_bytes"]), dtype=np.uint8))
        grid = bits[:rows*cols].reshape(rows, cols)
        zmax = 1
        
        for r in range(rows):
            row_h = []
            for c in range(cols):
                bit = r * cols + c
                va = get_l3_va(hdr, l1, l2, bit)
                state_str = "touched" if grid[r,c] else "not touched"
                row_h.append(f"L1[{l1}] L2[{l2}] page[{bit}]<br>{va:#018x}<br>{state_str}")
            hover_text.append(row_h)
            
        va_base = get_l2_va(hdr, l1, l2)
        touched = int(bits.sum())
        title, sub, crumb = f"L3 — L1[{l1}] L2[{l2}]", f"base {va_base:#018x} ({touched:,} pages touched)", f"Root › L1[{l1}] › L2[{l2}]"

    fig = go.Figure(go.Heatmap(
        z=grid, text=hover_text, hovertemplate="%{text}<extra></extra>",
        colorscale=GH_GREEN, showscale=False, zmin=0, zmax=zmax, xgap=1, ygap=1
    ))
    
    fig.update_layout(
        paper_bgcolor=GH_DARK, plot_bgcolor=GH_DARK, font=dict(color="#e6edf3", family="monospace"),
        height=650, uirevision=str(l1) if l1 is not None else "root",
        yaxis=dict(autorange="reversed", gridcolor="#21262d", title="Row"),
        xaxis=dict(gridcolor="#21262d", title="Column"),
        title=dict(text=f"<b>{title}</b><br><span style='font-size:12px;color:#8b949e'>{sub}</span>"),
        margin=dict(t=80, b=50, l=50, r=50)
    )
    
    btn_s = {"display": "block", "backgroundColor": "#21262d", "color": "#e6edf3", "border": "1px solid #30363d", 
             "padding": "6px 16px", "borderRadius": "6px", "cursor": "pointer", "marginBottom": "12px"} if lvl > 0 else {"display": "none"}
    
    return fig, {"level": lvl, "l1": l1, "l2": l2}, crumb, btn_s

server = app.server
if __name__ == "__main__":
    app.run(debug=True)