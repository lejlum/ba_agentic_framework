#!/usr/bin/env python3
"""
Swiss Recycling Assistant - Dashboard (Bachelor Thesis)
=======================================================

"""

import base64
import logging
import uuid
import webbrowser
import threading
import tempfile
import os
import requests
import sys
from pathlib import Path
from typing import Dict, List

import dash
from dash import Dash, html, dcc, Input, Output, State, ctx, ALL
import dash_bootstrap_components as dbc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from chatbot.swiss_waste_agent import build_agent, AgentState

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

agent = build_agent()


# ---------------------------------------------------------------------------
# UI TEXT
# ---------------------------------------------------------------------------
def get_texts(language: str) -> Dict[str, str]:
    if language == "de":
        return {
            "title": "Swiss Recycling Assistant",
            "chat_input": "Stellen Sie eine Frage...",
            "new_chat": "Neuer Chat",
            "language_label": "Sprache",
            "zip_label": "PLZ",
            "confidence": "Konfidenz",
            "welcome_title": "Hi, ich bin dein Swiss Recycling Assistant",
            "welcome_text": "Du bist unsicher, wie du etwas in der Schweiz recyceln sollst? Lade ein Foto deines Abfalls hoch oder stelle mir einfach direkt deine Frage.",
            "no_chats": "Keine Chatverläufe",
            "upload_button": "Bild",
            "image_analyzed": "Bild analysiert",
            "chat_history_label": "Verlauf",
            "detected": "Erkannt",
            "map_button": "Alle Sammelstellen anzeigen",
            "map_title": "Sammelstellen in der Nähe",
        }
    return {
        "title": "Swiss Recycling Assistant",
        "chat_input": "Ask a question...",
        "new_chat": "New Chat",
        "language_label": "Language",
        "zip_label": "ZIP code",
        "confidence": "Confidence",
        "welcome_title": "Hi, I'm your Swiss Recycling Assistant",
        "welcome_text": "Not sure how to recycle something in Switzerland? Upload a picture of your waste item or just ask me directly.",
        "no_chats": "No chat history",
        "upload_button": "Image",
        "image_analyzed": "Image analyzed",
        "chat_history_label": "History",
        "detected": "Detected",
        "map_button": "View all collection points",
        "map_title": "Nearby collection points",
    }


# ---------------------------------------------------------------------------
# DASH APP
# ---------------------------------------------------------------------------
app: Dash = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Swiss Recycling Assistant",
    suppress_callback_exceptions=True,
    assets_folder="assets",
)

server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
        <style>
            * { box-sizing: border-box; }
            body {
                margin: 0; padding: 0; min-height: 100vh; display: flex; flex-direction: column;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                color: #2c3e50;
            }
            #react-entry-point { flex: 1; display: flex; flex-direction: column; }
            #sidebar {
                position: fixed; top: 0; left: 0; bottom: 0; width: 280px;
                background-color: #f8f9fa; border-right: 1px solid #e5e7eb;
                display: flex; flex-direction: column; z-index: 1000;
                transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                overflow: hidden;
            }
            #sidebar.collapsed { width: 72px; }
            #sidebar-header { padding: 20px 16px 16px 16px; flex-shrink: 0; }
            #sidebar-sessions { flex: 1; overflow-y: auto; padding: 12px 16px; }
            #sidebar.collapsed #sidebar-header .sidebar-content { display: none; }
            #sidebar.collapsed #sidebar-sessions { display: none; }
            #main-content {
                margin-left: 280px; padding: 32px 48px 100px 48px;
                min-height: calc(100vh - 100px); background-color: #ffffff;
                transition: margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            #main-content.collapsed { margin-left: 72px; }
            .sidebar-section-label {
                font-size: 11px; font-weight: 600; color: #6b7280; text-transform: uppercase;
                letter-spacing: 0.05em; margin-bottom: 8px; padding-left: 2px;
            }
            .language-dropdown > div > div {
                background-color: #ffffff !important; border: 1px solid #d1d5db !important;
                border-radius: 8px !important; color: #2c3e50 !important; font-size: 14px !important;
            }
            .language-dropdown [class*="-option"] {
                background-color: #ffffff !important; color: #2c3e50 !important;
                padding: 10px 12px !important; cursor: pointer !important; font-size: 14px !important;
            }
            .language-dropdown [class*="-option--is-selected"] { background-color: #4a7ba7 !important; color: #ffffff !important; }
            .language-dropdown [class*="-menu"] {
                background-color: #ffffff !important; border: 1px solid #d1d5db !important;
                border-radius: 8px !important; box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
            }
            .language-dropdown [class*="-indicatorSeparator"],
            .language-dropdown [class*="-IndicatorSeparator"] { display: none !important; }
            .language-dropdown [class*="-dropdownIndicator"],
            .language-dropdown [class*="-DropdownIndicator"] { display: none !important; }
            .language-dropdown svg { display: none !important; }
            .chat-history-item {
                display: flex; align-items: center; padding: 10px 12px; border-radius: 8px;
                margin-bottom: 4px; font-size: 14px; cursor: pointer; border: 1px solid transparent;
                transition: all 0.15s ease;
            }
            .chat-history-item:hover { background-color: #f3f4f6; border-color: #e5e7eb; }
            .chat-history-item.active { background-color: #eff6ff; border-color: #bfdbfe; }
            .chat-history-item .chat-title { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #374151; }
            .chat-history-item.active .chat-title { color: #1e40af; font-weight: 500; }
            .chat-history-item .delete-btn { opacity: 0; transition: opacity 0.15s ease; }
            .chat-history-item:hover .delete-btn { opacity: 1; }
            .chat-input-group {
                display: flex; align-items: stretch; border-radius: 28px;
                background-color: #ffffff; border: 2px solid #e5e7eb;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06); overflow: hidden; transition: all 0.2s ease;
            }
            .chat-input-group:hover, .chat-input-group:focus-within {
                border-color: #4a7ba7; box-shadow: 0 4px 16px rgba(74,123,167,0.2); background-color: #f8fbff;
            }
            .chat-input-group .btn-upload {
                display: inline-flex; align-items: center; justify-content: center;
                padding: 14px 20px; border: none; background-color: transparent;
                color: #4a7ba7; font-size: 14px; cursor: pointer; gap: 8px; white-space: nowrap;
            }
            .btn-upload i { font-size: 16px; }
            .chat-input-group input {
                flex: 1; border: none !important; background-color: transparent !important;
                padding: 14px 24px; font-size: 15px; outline: none !important; box-shadow: none !important;
            }
            .chat-input-group .btn-send {
                border: none !important; background-color: #4a7ba7; color: white;
                padding: 14px 24px; min-width: 60px; transition: background-color 0.2s ease;
            }
            .chat-input-group:hover .btn-send,
            .chat-input-group:focus-within .btn-send { background-color: #3d6687; }
            .app-footer {
                padding: 24px 32px; text-align: center; background-color: #ffffff;
                border-top: 1px solid #e5e7eb; font-size: 13px; color: #6b7280;
            }
            .app-footer a { color: #6b7280; text-decoration: none; font-weight: 500; }
            .app-footer a:hover { color: #4a7ba7; }
            .markdown-content { color: #2c3e50; line-height: 1.6; }
            .markdown-content p { margin-bottom: 1em; }
            .markdown-content p:last-child { margin-bottom: 0; }
            .markdown-content ul, .markdown-content ol { margin-bottom: 1em; padding-left: 1.75em; }
            .markdown-content li { margin-bottom: 0.4em; }
            .markdown-content a { color: #4a7ba7; text-decoration: none; }
            .markdown-content a:hover { text-decoration: underline; }
            .markdown-content strong { font-weight: 600; color: #1a202c; }
            .image-result-card { background-color: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
            /* map card */
            .map-card {
                background-color: #f0f7ff; border: 1px solid #bfdbfe;
                border-radius: 12px; padding: 12px; margin-top: 8px;
                max-width: 65%;
            }
            .map-card iframe {
                width: 100%; height: 280px; border: none;
                border-radius: 8px; margin-top: 8px;
            }
            .map-btn {
                display: inline-block; background-color: #4a7ba7; color: white !important;
                padding: 8px 16px; border-radius: 8px; text-decoration: none !important;
                font-size: 14px; font-weight: 500; margin-top: 8px;
            }
            .map-btn:hover { background-color: #3d6687; }
            ::-webkit-scrollbar { width: 8px; }
            ::-webkit-scrollbar-track { background: #f3f4f6; }
            ::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }
            @media (max-width: 1024px) {
                #sidebar { width: 220px; }
                #main-content { margin-left: 220px; padding: 24px 24px 100px 24px; }
                #main-content.collapsed { margin-left: 48px; }
            }
            @media (max-width: 768px) {
                #sidebar { width: 0px; overflow: hidden; border: none; }
                #sidebar.open { width: 260px; overflow: visible; border-right: 1px solid #e5e7eb; box-shadow: 4px 0 20px rgba(0,0,0,0.15); z-index: 1100; }
                #main-content { margin-left: 0px !important; padding: 16px 16px 100px 16px; }
                #main-content.collapsed { margin-left: 0px; }
                .chat-input-group input { font-size: 16px; }
                .chat-input-group .btn-send { padding: 14px 18px; }
                .map-card { max-width: 100%; }
            }
            @media (max-width: 480px) {
                #main-content { padding: 12px 12px 120px 12px; }
                .app-footer { padding: 16px; font-size: 12px; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

CHAT_BUBBLE_USER = {
    "backgroundColor": "#4a7ba7", "color": "#ffffff", "borderRadius": "16px 16px 4px 16px",
    "padding": "14px 18px", "maxWidth": "65%", "marginLeft": "auto", "fontSize": "15px", "lineHeight": "1.6",
}
CHAT_BUBBLE_BOT = {
    "backgroundColor": "#f3f4f6", "color": "#2c3e50", "borderRadius": "16px 16px 16px 4px",
    "padding": "14px 18px", "maxWidth": "65%", "fontSize": "15px", "lineHeight": "1.6", "border": "1px solid #e5e7eb",
}
AVATAR = {
    "width": "40px", "height": "40px", "borderRadius": "50%", "objectFit": "cover",
    "flexShrink": 0, "border": "2px solid #ffffff", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
}
CHAT_ROW = {"display": "flex", "alignItems": "flex-end", "gap": "12px", "marginBottom": "20px"}


def make_map_card(zip_code: str, language: str):
    """Creates a map card with embedded OSM map centered on the ZIP code."""
    texts = get_texts(language)
    lang_path = "de" if language == "de" else "en"
    map_url = f"https://recycling-map.ch/{lang_path}/karte?zip={zip_code}"

    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "postalcode": zip_code,
                "country": "CH",
                "format": "json",
                "limit": 1
            },
            headers={"User-Agent": "SwissRecyclingAssistant/1.0"},
            timeout=5
        )
        results = resp.json()
        if results:
            lat = float(results[0].get("lat", 47.38))
            lon = float(results[0].get("lon", 8.54))
        else:
            lat, lon = 47.38, 8.54
    except:
        lat, lon = 47.38, 8.54

    # calculate bbox around the coordinates (approx 10km radius)
    delta = 0.04
    bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"
    osm_embed = f"https://www.openstreetmap.org/export/embed.html?bbox={bbox}&layer=mapnik&marker={lat},{lon}"

    return html.Div([
        html.Div(texts["map_title"], style={"fontWeight": "600", "fontSize": "14px", "color": "#1e40af", "marginBottom": "8px"}),
        html.Iframe(src=osm_embed, style={"width": "100%", "height": "250px", "border": "none", "borderRadius": "8px", "marginBottom": "8px"}),
        html.A(texts["map_button"], href=map_url, target="_blank", className="map-btn"),
    ], className="map-card")


app.layout = html.Div([
    dcc.Store(id="language-store", data="en"),
    dcc.Store(id="sessions-store", data={}),
    dcc.Store(id="active-session-store", data=""),
    dcc.Store(id="sidebar-collapsed-store", data=False),
    dcc.Store(id="zip-store", data=""),
    dcc.Store(id="agent-history-store", data={}),

    html.Div([
        html.Div([
            html.Div(
                html.Span("Swiss Recycling Assistant", style={"fontSize": "15px", "fontWeight": "600", "color": "#1a202c"}),
                className="sidebar-content",
                style={"marginTop": "48px", "marginBottom": "20px", "paddingLeft": "2px"}
            ),
            html.Div([
                html.Div([
                    html.Label("Language", id="language-label", className="sidebar-section-label"),
                    dcc.Dropdown(
                        id="language-dropdown",
                        options=[{"label": "English", "value": "en"}, {"label": "Deutsch", "value": "de"}],
                        value="en", clearable=False, className="language-dropdown",
                    ),
                ], style={"marginBottom": "16px"}),
                html.Div([
                    html.Label("ZIP code", id="zip-label", className="sidebar-section-label"),
                    dbc.Input(
                        id="zip-input", placeholder="e.g. 9403", type="text", maxLength=4,
                        style={"borderRadius": "8px", "fontSize": "14px", "border": "1px solid #d1d5db"},
                    ),
                ], style={"marginBottom": "20px"}),
                dbc.Button(
                    [html.I(className="bi bi-plus-lg", style={"marginRight": "8px"}), html.Span("New Chat", id="new-chat-label")],
                    id="new-chat-btn",
                    style={"width": "100%", "marginBottom": "8px", "borderRadius": "10px", "padding": "12px 16px",
                           "fontSize": "14px", "fontWeight": "500", "backgroundColor": "#4a7ba7", "border": "none", "color": "#ffffff"},
                ),
                html.Div("History", id="history-label", className="sidebar-section-label", style={"marginBottom": "8px", "marginTop": "4px"}),
            ], className="sidebar-content"),
        ], id="sidebar-header"),
        html.Div(id="sidebar-sessions"),
    ], id="sidebar"),

    html.Div(
        dbc.Button(
            html.I(className="bi bi-list", style={"fontSize": "20px"}),
            id="sidebar-toggle", color="link",
            style={"color": "#4b5563", "padding": "8px"},
            n_clicks=0,
        ),
        style={"position": "fixed", "top": "10px", "left": "10px", "zIndex": 1002, "backgroundColor": "#f8f9fa", "borderRadius": "8px"}
    ),

    html.Div([
        html.Div(id="chat-content"),
        html.Div(
            html.Div([
                dcc.Upload(
                    id="image-upload",
                    children=html.Div([html.I(className="bi bi-image"), html.Span(id="upload-button-text", children=" Image")], className="btn-upload"),
                    multiple=False, style={"border": "none", "display": "inline-block"},
                ),
                dbc.Input(id="chat-input", placeholder="Ask a question...", type="text", n_submit=0),
                dbc.Button(html.I(className="bi bi-send-fill", style={"fontSize": "16px"}), id="send-button", className="btn-send"),
            ], className="chat-input-group", style={"maxWidth": "880px", "margin": "0 auto", "width": "100%"}),
            style={"marginTop": "40px", "display": "flex", "justifyContent": "center"},
        ),
    ], id="main-content"),

    html.Div([
        html.Div([
            html.A("BAFU", href="https://www.bafu.admin.ch", target="_blank"),
            html.Span(" · ", style={"color": "#d1d5db", "margin": "0 8px"}),
            html.A("IGORA", href="https://www.igora.ch/de/sammelstellen", target="_blank"),
            html.Span(" · ", style={"color": "#d1d5db", "margin": "0 8px"}),
            html.A("PET Recycling", href="https://www.petrecycling.ch/de/sammelstellen", target="_blank"),
            html.Span(" · ", style={"color": "#d1d5db", "margin": "0 8px"}),
            html.A("Swiss Recycle", href="https://www.swissrecycle.ch", target="_blank"),
            html.Span(" · ", style={"color": "#d1d5db", "margin": "0 8px"}),
            html.A("VetroSwiss", href="https://www.vetroswiss.ch/de/sammelstellen", target="_blank"),
        ])
    ], className="app-footer"),
], style={"display": "flex", "flexDirection": "column", "minHeight": "100vh"})


# ---------------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------------

@app.callback(Output("zip-store", "data"), Input("zip-input", "value"))
def save_zip(value):
    return value or ""

@app.callback(Output("language-store", "data"), Input("language-dropdown", "value"))
def update_language(lang):
    return lang or "en"

@app.callback(
    [Output("language-label", "children"), Output("zip-label", "children"),
     Output("new-chat-label", "children"), Output("history-label", "children"),
     Output("chat-input", "placeholder"), Output("upload-button-text", "children")],
    Input("language-store", "data"),
)
def update_labels(language):
    texts = get_texts(language)
    return (texts["language_label"], texts["zip_label"], texts["new_chat"],
            texts["chat_history_label"], texts["chat_input"], f" {texts['upload_button']}")

@app.callback(
    [Output("sidebar", "className"), Output("main-content", "className")],
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar-collapsed-store", "data"),
    prevent_initial_call=True,
)
def toggle_sidebar(n_clicks, collapsed):
    new_collapsed = not collapsed
    sidebar_class = "collapsed" if new_collapsed else "open"
    main_class = "collapsed" if new_collapsed else ""
    return sidebar_class, main_class

@app.callback(Output("sidebar-collapsed-store", "data"), Input("sidebar-toggle", "n_clicks"),
              State("sidebar-collapsed-store", "data"), prevent_initial_call=True)
def save_collapsed(n_clicks, collapsed):
    return not collapsed

@app.callback(
    Output("sidebar-sessions", "children"),
    [Input("sessions-store", "data"), Input("active-session-store", "data"), Input("language-store", "data")],
)
def update_session_list(sessions, active_session, language):
    texts = get_texts(language)
    sessions = sessions or {}
    if not sessions:
        return html.Div(texts["no_chats"], style={
            "color": "#9ca3af", "fontSize": "13px", "padding": "16px 12px", "textAlign": "center", "fontStyle": "italic"
        })
    items = []
    for sid, messages in reversed(list(sessions.items())):
        title = texts["new_chat"]
        for msg in messages:
            if msg.get("role") == "user":
                title = msg.get("content", "")[:40] + ("..." if len(msg.get("content", "")) > 40 else "")
                break
            elif msg.get("role") == "image_result":
                title = texts["image_analyzed"]
                break
        items.append(html.Div([
            html.Div(title, className="chat-title", id={"type": "session-select", "index": sid}),
            dbc.Button(
                html.I(className="bi bi-trash", style={"fontSize": "12px"}),
                id={"type": "session-delete", "index": sid},
                color="link", size="sm", className="delete-btn",
                style={"color": "#9ca3af", "padding": "4px 6px"},
            ),
        ], className=f"chat-history-item {'active' if sid == active_session else ''}"))
    return items

@app.callback(
    Output("chat-content", "children"),
    [Input("language-store", "data"), Input("sessions-store", "data"), Input("active-session-store", "data"),
     Input("zip-store", "data")],
)
def render_chat(language, sessions, active_session, zip_code):
    texts = get_texts(language)
    messages = (sessions or {}).get(active_session, []) if active_session else []
    if not messages:
        return html.Div([
            html.Div([
                html.Div(html.Img(src="/assets/robo_head.png", style={"width": "80px", "height": "80px", "marginBottom": "28px", "opacity": "0.9"}), style={"textAlign": "center"}),
                html.H2(texts["welcome_title"], style={"fontWeight": "600", "marginBottom": "16px", "textAlign": "center", "fontSize": "28px", "color": "#1a202c"}),
                html.P(texts["welcome_text"], style={"color": "#6b7280", "fontSize": "16px", "textAlign": "center", "maxWidth": "560px", "margin": "0 auto"}),
            ], style={"backgroundColor": "#f9fafb", "borderRadius": "20px", "border": "1px solid #e5e7eb", "padding": "48px 32px", "maxWidth": "680px", "margin": "0 auto"}),
        ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "padding": "60px 24px", "minHeight": "65vh"})

    def make_msg(msg):
        if msg.get("role") == "user":
            return html.Div([
                html.Div(msg.get("content", ""), style=CHAT_BUBBLE_USER),
                html.Div(html.I(className="bi bi-person-circle", style={"fontSize": "20px", "color": "#4a7ba7"}),
                         style={**AVATAR, "backgroundColor": "#e8f0f7", "display": "flex", "alignItems": "center", "justifyContent": "center"}),
            ], style={**CHAT_ROW, "justifyContent": "flex-end"})
        elif msg.get("role") == "image_result":
            image_src = msg.get("image_src", "")
            return html.Div([
                html.Img(src="/assets/robo_head.png", style=AVATAR),
                html.Div([html.Div([
                    html.Img(src=image_src, style={"maxWidth": "100%", "maxHeight": "180px", "borderRadius": "8px", "marginBottom": "12px"}) if image_src else None,
                    html.Div(msg.get("result_data", {}).get("content", [])),
                ], className="image-result-card")], style={**CHAT_BUBBLE_BOT, "backgroundColor": "transparent", "border": "none", "padding": "0"}),
            ], style=CHAT_ROW)
        elif msg.get("role") == "location_result":
            # special location message with map card
            zip_used = msg.get("zip_code", zip_code)
            lang_used = msg.get("language", language)
            return html.Div([
                html.Img(src="/assets/robo_head.png", style=AVATAR),
                html.Div([
                    html.Div(dcc.Markdown(msg.get("content", ""), className="markdown-content", style={"margin": 0}), style=CHAT_BUBBLE_BOT),
                    make_map_card(zip_used, lang_used) if zip_used else None,
                ]),
            ], style=CHAT_ROW)
        else:
            return html.Div([
                html.Img(src="/assets/robo_head.png", style=AVATAR),
                html.Div(dcc.Markdown(msg.get("content", ""), className="markdown-content", style={"margin": 0}), style=CHAT_BUBBLE_BOT),
            ], style=CHAT_ROW)

    return html.Div([make_msg(m) for m in messages], style={"padding": "28px 0", "maxWidth": "920px", "margin": "0 auto"})

@app.callback(
    [Output("sessions-store", "data", allow_duplicate=True), Output("active-session-store", "data", allow_duplicate=True)],
    Input("new-chat-btn", "n_clicks"), State("sessions-store", "data"),
    prevent_initial_call=True,
)
def new_chat(n_clicks, sessions):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    sessions = sessions or {}
    new_id = str(uuid.uuid4())[:8]
    sessions[new_id] = []
    return sessions, new_id

@app.callback(
    Output("active-session-store", "data", allow_duplicate=True),
    Input({"type": "session-select", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_session(n_clicks):
    if not ctx.triggered_id or not any(n_clicks):
        raise dash.exceptions.PreventUpdate
    return ctx.triggered_id["index"]

@app.callback(
    [Output("sessions-store", "data", allow_duplicate=True), Output("active-session-store", "data", allow_duplicate=True)],
    Input({"type": "session-delete", "index": ALL}, "n_clicks"),
    [State("sessions-store", "data"), State("active-session-store", "data")],
    prevent_initial_call=True,
)
def delete_session(n_clicks, sessions, active_session):
    if not ctx.triggered_id or not any(n for n in n_clicks if n):
        raise dash.exceptions.PreventUpdate
    sid = ctx.triggered_id["index"]
    sessions = sessions or {}
    if sid in sessions:
        del sessions[sid]
    new_active = active_session if active_session != sid else (list(sessions.keys())[0] if sessions else "")
    return sessions, new_active

@app.callback(
    [Output("sessions-store", "data", allow_duplicate=True),
     Output("active-session-store", "data", allow_duplicate=True),
     Output("agent-history-store", "data", allow_duplicate=True)],
    [Input("send-button", "n_clicks"), Input("chat-input", "n_submit")],
    [State("chat-input", "value"), State("sessions-store", "data"),
     State("active-session-store", "data"), State("language-store", "data"),
     State("zip-store", "data"), State("agent-history-store", "data")],
    prevent_initial_call=True,
)
def send_message(n_clicks, n_submit, user_text, sessions, active_session, language, zip_code, agent_history):
    if (not n_clicks and not n_submit) or not user_text or not user_text.strip():
        raise dash.exceptions.PreventUpdate

    sessions = sessions or {}
    agent_history = agent_history or {}

    if not active_session or active_session not in sessions:
        active_session = str(uuid.uuid4())[:8]
        sessions[active_session] = []

    user_text = user_text.strip()
    sessions[active_session].append({"role": "user", "content": user_text})
    session_state = agent_history.get(active_session, {"scan_history": [], "conv_history": []})

    try:
        state = AgentState(
            user_message=user_text, image_path=None, zip_code=zip_code or None,
            language=language or "en", classification=None, guidelines=None, collection_points=None,
            input_type=None, needs_clarification=False, final_response=None,
            scan_history=session_state["scan_history"], conversation_history=session_state["conv_history"],
        )
        result = agent.invoke(state, config={"configurable": {"thread_id": active_session}})
        response = result["final_response"]
        input_type = result.get("input_type", "text")

        agent_history[active_session] = {
            "scan_history": result.get("scan_history", []),
            "conv_history": result.get("conversation_history", []),
        }
    except Exception as e:
        logger.error(f"Agent error: {e}")
        response = f"Error: {e}"
        input_type = "text"

    # use location_result role when it was a location query with a ZIP code
    # this triggers the map card in the chat
    if input_type == "location" and zip_code:
        sessions[active_session].append({
            "role": "location_result",
            "content": response,
            "zip_code": zip_code,
            "language": language or "en"
        })
    else:
        sessions[active_session].append({"role": "assistant", "content": response})

    return sessions, active_session, agent_history

@app.callback(Output("chat-input", "value"), [Input("send-button", "n_clicks"), Input("chat-input", "n_submit")], prevent_initial_call=True)
def clear_input(n_clicks, n_submit):
    return ""

@app.callback(
    [Output("sessions-store", "data", allow_duplicate=True),
     Output("active-session-store", "data", allow_duplicate=True),
     Output("agent-history-store", "data", allow_duplicate=True)],
    [Input("image-upload", "contents"), State("image-upload", "filename"),
     State("language-store", "data"), State("sessions-store", "data"),
     State("active-session-store", "data"), State("zip-store", "data"), State("agent-history-store", "data")],
    prevent_initial_call=True,
)
def handle_image(contents, filename, language, sessions, active_session, zip_code, agent_history):
    if not contents:
        raise dash.exceptions.PreventUpdate

    sessions = sessions or {}
    agent_history = agent_history or {}
    texts = get_texts(language)

    if not active_session or active_session not in sessions:
        active_session = str(uuid.uuid4())[:8]
        sessions[active_session] = []

    try:
        header, encoded = contents.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        suffix = Path(filename).suffix or ".png"
        tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}{suffix}")
        with open(tmp_path, "wb") as f:
            f.write(img_bytes)

        session_state = agent_history.get(active_session, {"scan_history": [], "conv_history": []})
        state = AgentState(
            user_message="How do I dispose of this?" if language == "en" else "Wie entsorge ich das?",
            image_path=tmp_path, zip_code=zip_code or None, language=language or "en",
            classification=None, guidelines=None, collection_points=None,
            input_type=None, needs_clarification=False, final_response=None,
            scan_history=session_state["scan_history"], conversation_history=session_state["conv_history"],
        )
        result = agent.invoke(state, config={"configurable": {"thread_id": active_session}})
        classification = result.get("classification") or {}
        category = classification.get("category", "unknown").replace("_", " ").title()
        confidence = classification.get("confidence", 0)

        result_content = [
            html.H5(f"{texts['detected']}: {category}", style={"fontWeight": "600", "marginBottom": "8px", "fontSize": "17px", "color": "#1a202c"}),
            html.P(f"{texts['confidence']}: {confidence:.0%}", style={"color": "#10b981", "fontSize": "14px", "marginBottom": "14px", "fontWeight": "500"}),
            dcc.Markdown(result["final_response"], className="markdown-content", style={"marginBottom": 0}),
        ]
        agent_history[active_session] = {
            "scan_history": result.get("scan_history", []),
            "conv_history": result.get("conversation_history", []),
        }
        try:
            os.remove(tmp_path)
        except:
            pass

    except Exception as e:
        logger.error(f"Image error: {e}")
        result_content = [html.Div(f"Error: {e}", style={"color": "#dc2626", "fontWeight": "500"})]

    sessions[active_session].append({"role": "image_result", "image_src": contents, "result_data": {"content": result_content}})
    return sessions, active_session, agent_history


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def open_browser():
    webbrowser.open("http://127.0.0.1:8050/")

if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.5, open_browser).start()
    app.run(debug=True, host="0.0.0.0", port=8050)
