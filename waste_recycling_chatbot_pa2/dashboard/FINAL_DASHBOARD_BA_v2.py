#!/usr/bin/env python3
"""
Swiss Recycling Assistant - Dashboard (Bachelor Thesis)
=======================================================

Changes applied:
  1. Google Maps link directly in Leaflet popup (removed map-click callbacks)
  2. User message shown immediately with animated thinking bubble
  3. City/Ort name input in addition to ZIP
  4. location_result: only map card shown — no duplicate text bubble.
     The Google Maps link (+ address + opening hours) lives in each popup.
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

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # HEIC support unavailable; HEIC uploads will show a format error

from PIL import Image, ImageOps

import dash
from dash import Dash, html, dcc, Input, Output, State, ctx, ALL
from flask import request as flask_request, Response
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
            "city_label": "Ort",
            "zip_placeholder": "z.B. 8820",
            "city_placeholder": "z.B. Wädenswil",
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
        "city_label": "City",
        "zip_placeholder": "e.g. 8820",
        "city_placeholder": "e.g. Wädenswil",
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


def build_map_html(zip_code, lat, lon, language):
    """Build standalone map HTML served via Flask route."""
    home_label   = "Ihr Standort"    if language == "de" else "Your location"
    lbl_address  = "Adresse"         if language == "de" else "Address"
    lbl_hours    = "Oeffnungszeiten" if language == "de" else "Opening hours"
    lbl_no_hours = "Nicht angegeben" if language == "de" else "Not specified"
    cat_glass    = "Glascontainer"   if language == "de" else "Glass"
    cat_pet      = "PET / Plastik"   if language == "de" else "PET / Plastic"
    cat_metal    = "Metall / Alu"    if language == "de" else "Metal"
    cat_centre   = "Entsorgungshof"  if language == "de" else "Recycling Centre"
    btn_all      = "Alle"            if language == "de" else "All"
    btn_glass    = "Glas"            if language == "de" else "Glass"
    btn_metal    = "Metall"          if language == "de" else "Metal"
    lbl_gmaps    = "In Google Maps öffnen" if language == "de" else "Open in Google Maps"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body {{ margin:0; padding:0; font-family: -apple-system, sans-serif; }}
  #map {{ width:100%; height:280px; }}
  #filters {{ padding:6px 8px; background:#f8fafc; border-bottom:1px solid #e2e8f0; display:flex; gap:6px; flex-wrap:wrap; }}
  .filter-btn {{ padding:4px 12px; border-radius:20px; border:1.5px solid #cbd5e1; background:white; font-size:12px; cursor:pointer; transition:all 0.15s; font-weight:500; color:#475569; }}
  .filter-btn.active {{ border-color:#3b82f6; background:#eff6ff; color:#1d4ed8; }}
</style>
</head>
<body>
<div id="filters">
  <button class="filter-btn active" onclick="filterMarkers('all',this)">{btn_all}</button>
  <button class="filter-btn" onclick="filterMarkers('glass',this)" style="border-color:#31a354;color:#166534;">{btn_glass} &#x1F7E2;</button>
  <button class="filter-btn" onclick="filterMarkers('pet',this)" style="border-color:#fd8d3c;color:#9a3412;">PET &#x1F7E0;</button>
  <button class="filter-btn" onclick="filterMarkers('metal',this)" style="border-color:#636363;color:#374151;">{btn_metal} &#x26AB;</button>
  <button class="filter-btn" onclick="filterMarkers('centre',this)" style="border-color:#de2d26;color:#991b1b;">{cat_centre} &#x1F534;</button>
</div>
<div id="map"></div>
<script>
var map = L.map('map').setView([{lat}, {lon}], 15);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);

var homeIcon = L.divIcon({{
    html: '<div style="background:#2563eb;width:16px;height:16px;border-radius:50%;border:3px solid white;box-shadow:0 2px 4px rgba(0,0,0,0.5)"></div>',
    iconSize:[16,16], iconAnchor:[8,8], className:''
}});
L.marker([{lat},{lon}], {{icon:homeIcon}})
 .bindPopup('<b>{home_label}</b><br>ZIP {zip_code}').addTo(map);

var allMarkers = [];

function filterMarkers(type, btn) {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    allMarkers.forEach(function(m) {{
        if (type === 'all' || m.catKey === type) map.addLayer(m.marker);
        else map.removeLayer(m.marker);
    }});
}}

function makeIcon(color) {{
    return L.divIcon({{
        html: '<div style="background:'+color+';width:14px;height:14px;border-radius:50%;border:2px solid white;box-shadow:0 1px 3px rgba(0,0,0,0.4)"></div>',
        iconSize:[14,14], iconAnchor:[7,7], className:''
    }});
}}

function procBM(el) {{
    if(!el.lat||!el.lon) return;
    var tags=el.tags||{{}};
    var name=tags.name||'Collection point';
    var street=tags['addr:street']||''; var number=tags['addr:housenumber']||'';
    var address=street?(street+' '+number).trim():'';
    var hours=tags['opening_hours']||'{lbl_no_hours}';
    var color='#6baed6'; var cat='{cat_glass}'; var catKey='other';
    if(tags['recycling:glass_bottles']==='yes'||tags['recycling:glass']==='yes')
        {{color='#31a354';cat='{cat_glass}';catKey='glass';}}
    else if(tags['recycling:plastic_bottles']==='yes'||tags['recycling:PET']==='yes')
        {{color='#fd8d3c';cat='{cat_pet}';catKey='pet';}}
    else if(tags['recycling:scrap_metal']==='yes'||tags['recycling:metal']==='yes')
        {{color='#636363';cat='{cat_metal}';catKey='metal';}}
    else if(tags.amenity==='waste_disposal'||tags.recycling_type==='centre')
        {{color='#de2d26';cat='{cat_centre}';catKey='centre';}}
    else if(tags.shop==='supermarket')
        {{color='#7c3aed';cat='{cat_pet}';catKey='pet';}}

    var gmaps = 'https://www.google.com/maps?q=' + el.lat + ',' + el.lon;

    var popup='<b>'+name+'</b><br><span style="color:#64748b;font-size:12px">'+cat+'</span>';
    if(address) popup+='<br><small>&#128205; {lbl_address}: '+address+'</small>';
    popup+='<br><small>&#128336; {lbl_hours}: '+hours+'</small>';
    popup+='<br><small><a href="'+gmaps+'" target="_blank" style="color:#2563eb;">{lbl_gmaps}</a></small>';

    var marker=L.marker([el.lat,el.lon],{{icon:makeIcon(color)}})
        .bindPopup(popup).bindTooltip(cat+': '+name)
        .addTo(map);
    allMarkers.push({{marker:marker, catKey:catKey}});
}}
fetch('https://overpass-api.de/api/interpreter', {{
    method:'POST',
    body:'data='+encodeURIComponent('[out:json][timeout:15];(node["amenity"="recycling"](around:3000,{lat},{lon});node["amenity"="waste_disposal"](around:3000,{lat},{lon}););out body;'),
    headers:{{'Content-Type':'application/x-www-form-urlencoded'}}
}}).then(r=>r.json()).then(d=>(d.elements||[]).forEach(procBM)).catch(e=>console.log('q1:',e));
fetch('https://overpass-api.de/api/interpreter', {{
    method:'POST',
    body:'data='+encodeURIComponent('[out:json][timeout:10];node["shop"="supermarket"](around:1500,{lat},{lon});out body;'),
    headers:{{'Content-Type':'application/x-www-form-urlencoded'}}
}}).then(r=>r.json()).then(d=>(d.elements||[]).filter(el=>/coop|migros|denner|lidl|aldi/i.test((el.tags&&el.tags.name)||'')).forEach(procBM)).catch(e=>console.log('q2:',e));
</script>
</body>
</html>"""


@server.route('/map')
def serve_map():
    zip_code = flask_request.args.get('zip', '')
    lat = float(flask_request.args.get('lat', 47.38))
    lon = float(flask_request.args.get('lon', 8.54))
    language = flask_request.args.get('lang', 'en')
    html_content = build_map_html(zip_code, lat, lon, language)
    return Response(html_content, mimetype='text/html')


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
            .map-card {
                background-color: #f0f7ff; border: 1px solid #bfdbfe;
                border-radius: 12px; padding: 12px; margin-top: 8px;
                max-width: 100%;
            }
            .map-card iframe {
                width: 100%; height: 380px; border: none;
                border-radius: 8px; margin-top: 8px;
            }
            .map-btn {
                display: inline-block; background-color: #4a7ba7; color: white !important;
                padding: 8px 16px; border-radius: 8px; text-decoration: none !important;
                font-size: 14px; font-weight: 500; margin-top: 8px;
            }
            .map-btn:hover { background-color: #3d6687; }
            @keyframes pulse { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }
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
                #sidebar.collapsed { width: 0px; overflow: hidden; border: none; }
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


def make_map_card(zip_code: str, language: str, city: str = "", elements: list = None, lat: float = None, lon: float = None):
    """Interactive Leaflet map.
    Geocoding is done entirely client-side (JS → Nominatim) so it is not
    blocked by Python network issues or Dash callback timeouts.
    """
    texts = get_texts(language)
    location_label = zip_code or city or "CH"
    # Safe JS string: escape single quotes
    query_str = (zip_code or city or "Schweiz").replace("'", "\\'")

    home_label    = "Ihr Standort"    if language == "de" else "Your location"
    lbl_address   = "Adresse"         if language == "de" else "Address"
    lbl_hours     = "Oeffnungszeiten" if language == "de" else "Opening hours"
    lbl_no_hours  = "Nicht angegeben" if language == "de" else "Not specified"
    cat_glass     = "Glascontainer"   if language == "de" else "Glass"
    cat_pet       = "PET / Plastik"   if language == "de" else "PET / Plastic"
    cat_metal     = "Metall / Alu"    if language == "de" else "Metal"
    cat_centre    = "Entsorgungshof"  if language == "de" else "Recycling Centre"
    btn_all       = "Alle"            if language == "de" else "All"
    btn_glass     = "Glas"            if language == "de" else "Glass"
    btn_metal     = "Metall"          if language == "de" else "Metal"
    btn_supermarkt = "Laden"          if language == "de" else "Store"
    lbl_gmaps     = "In Google Maps \u00f6ffnen" if language == "de" else "Open in Google Maps"

    map_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body {{ margin:0; padding:0; font-family:-apple-system,sans-serif; }}
  #map {{ width:100%; height:300px; }}
  #filters {{ padding:6px 8px; background:#f8fafc; border-bottom:1px solid #e2e8f0; display:flex; gap:6px; flex-wrap:wrap; }}
  .fb {{ padding:4px 12px; border-radius:20px; border:1.5px solid #cbd5e1; background:white; font-size:12px; cursor:pointer; font-weight:500; color:#475569; }}
  .fb.active {{ border-color:#3b82f6; background:#eff6ff; color:#1d4ed8; }}
</style>
</head>
<body>
<div id="filters">
  <button class="fb active" onclick="fm('all',this)">{btn_all}</button>
  <button class="fb" onclick="fm('glass',this)" style="border-color:#31a354;color:#166534;">{btn_glass}</button>
  <button class="fb" onclick="fm('pet',this)" style="border-color:#fd8d3c;color:#9a3412;">PET</button>
  <button class="fb" onclick="fm('metal',this)" style="border-color:#636363;color:#374151;">{btn_metal}</button>
  <button class="fb" onclick="fm('centre',this)" style="border-color:#de2d26;color:#991b1b;">{cat_centre}</button>
  <button class="fb" onclick="fm('store',this)" style="border-color:#7c3aed;color:#5b21b6;">{btn_supermarkt}</button>
</div>
<div id="map"></div>
<script>
// Map starts zoomed out on Switzerland; client-side geocoding centres it correctly
var map = L.map('map').setView([46.9481, 7.4474], 8);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{attribution:'&copy; OpenStreetMap'}}).addTo(map);

var hi = L.divIcon({{html:'<div style="background:#2563eb;width:16px;height:16px;border-radius:50%;border:3px solid white;box-shadow:0 2px 4px rgba(0,0,0,0.5)"></div>',iconSize:[16,16],iconAnchor:[8,8],className:''}});
var AM = [];

function fm(t,b){{document.querySelectorAll('.fb').forEach(x=>x.classList.remove('active'));b.classList.add('active');AM.forEach(function(m){{if(t==='all'||m.k===t)map.addLayer(m.m);else map.removeLayer(m.m);}});}}
function mi(c){{return L.divIcon({{html:'<div style="background:'+c+';width:14px;height:14px;border-radius:50%;border:2px solid white;box-shadow:0 1px 3px rgba(0,0,0,0.4)"></div>',iconSize:[14,14],iconAnchor:[7,7],className:''}});}}

function addEl(el){{
  if(!el.lat||!el.lon) return;
  var t=el.tags||{{}},n=t.name||'Collection point';
  var s=t['addr:street']||'',nr=t['addr:housenumber']||'';
  var oa=s?(s+' '+nr).trim():'';
  var h=t['opening_hours']||'{lbl_no_hours}';
  var c='#6baed6',cat='Recycling',k='other';
  if(t['recycling:glass_bottles']==='yes'||t['recycling:glass']==='yes'){{c='#31a354';cat='{cat_glass}';k='glass';}}
  else if(t['recycling:plastic_bottles']==='yes'||t['recycling:PET']==='yes'){{c='#fd8d3c';cat='{cat_pet}';k='pet';}}
  else if(t['recycling:scrap_metal']==='yes'||t['recycling:metal']==='yes'){{c='#636363';cat='{cat_metal}';k='metal';}}
  else if(t.amenity==='waste_disposal'||t.recycling_type==='centre'){{c='#de2d26';cat='{cat_centre}';k='centre';}}
  else if(t.shop==='supermarket'){{c='#7c3aed';cat='{btn_supermarkt}';k='store';}}
  function addM(a){{
    var gmaps='https://www.google.com/maps?q='+el.lat+','+el.lon;
    var p='<b>'+n+'</b><br><span style="color:#64748b;font-size:12px">'+cat+'</span>';
    if(a)p+='<br><small>&#128205; {lbl_address}: '+a+'</small>';
    p+='<br><small>&#128336; {lbl_hours}: '+h+'</small>';
    p+='<br><small><a href="'+gmaps+'" target="_blank" style="color:#2563eb;">{lbl_gmaps}</a></small>';
    var mk=L.marker([el.lat,el.lon],{{icon:mi(c)}}).bindPopup(p).bindTooltip(cat+': '+n).addTo(map);
    AM.push({{m:mk,k:k}});
  }}
  if(oa){{addM(oa);}}
  else{{
    fetch('https://nominatim.openstreetmap.org/reverse?lat='+el.lat+'&lon='+el.lon+'&format=json&zoom=18&addressdetails=1')
      .then(function(r){{return r.json();}})
      .then(function(g){{
        var ad=g.address||{{}};
        var road=ad.road||ad.pedestrian||ad.path||'';
        var nr2=ad.house_number||'';
        var v=ad.village||ad.town||ad.city||'';
        addM(road?(road+(nr2?' '+nr2:''))+(v?', '+v:''):g.display_name.split(',').slice(0,2).join(',')||'');
      }}).catch(function(){{addM('');}});
  }}
}}

// fetchOSM: takes JS lat/lon variables so no Python f-string baking needed
function fetchOSM(clat, clon){{
  fetch('https://overpass-api.de/api/interpreter',{{method:'POST',
    body:'data='+encodeURIComponent('[out:json][timeout:15];(node["amenity"="recycling"](around:3000,'+clat+','+clon+');node["amenity"="waste_disposal"](around:3000,'+clat+','+clon+'););out body;'),
    headers:{{'Content-Type':'application/x-www-form-urlencoded'}}}})
  .then(function(r){{return r.json();}})
  .then(function(d){{(d.elements||[]).forEach(addEl);}})
  .catch(function(e){{console.log('q1:',e);}});
  fetch('https://overpass-api.de/api/interpreter',{{method:'POST',
    body:'data='+encodeURIComponent('[out:json][timeout:10];node["shop"="supermarket"](around:1500,'+clat+','+clon+');out body;'),
    headers:{{'Content-Type':'application/x-www-form-urlencoded'}}}})
  .then(function(r){{return r.json();}})
  .then(function(d){{
    (d.elements||[]).filter(function(el){{var n2=(el.tags&&el.tags.name)||'';return /coop|migros|denner|lidl|aldi/i.test(n2);}}).forEach(addEl);
  }}).catch(function(e){{console.log('q2:',e);}});
}}

// Client-side geocoding: browser requests Nominatim directly (reliable, no server dependency)
fetch('https://nominatim.openstreetmap.org/search?q='+encodeURIComponent('{query_str} Schweiz')+'&format=json&limit=1&countrycodes=ch')
  .then(function(r){{return r.json();}})
  .then(function(data){{
    var clat = (data && data[0]) ? parseFloat(data[0].lat) : 46.9481;
    var clon = (data && data[0]) ? parseFloat(data[0].lon) : 7.4474;
    map.setView([clat, clon], 15);
    L.marker([clat, clon],{{icon:hi}}).bindPopup('<b>{home_label}</b><br>{location_label}').addTo(map);
    fetchOSM(clat, clon);
  }})
  .catch(function(){{
    map.setView([46.9481, 7.4474], 15);
    fetchOSM(46.9481, 7.4474);
  }});
</script>
</body>
</html>"""

    return html.Div([
        html.Div(texts["map_title"], style={"fontWeight": "600", "fontSize": "14px", "color": "#1e40af", "marginBottom": "8px"}),
        html.Iframe(srcDoc=map_html, style={"width": "100%", "height": "340px", "border": "none", "borderRadius": "8px"}),
    ], className="map-card", style={"width": "100%"})




app.layout = html.Div([
    dcc.Store(id="language-store", data="en"),
    dcc.Store(id="sessions-store", data={}),
    dcc.Store(id="active-session-store", data=""),
    dcc.Store(id="sidebar-collapsed-store", data=False),
    dcc.Store(id="zip-store", data=""),
    dcc.Store(id="city-store", data=""),
    dcc.Store(id="agent-history-store", data={}),
    dcc.Store(id="pending-message-store", data=None),
    dcc.Store(id="pending-image-store", data=None),
    dcc.Store(id="new-chat-session-id", data=None),

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
                        id="zip-input", placeholder="e.g. 8820", type="text", maxLength=4,
                        style={"borderRadius": "8px", "fontSize": "14px", "border": "1px solid #d1d5db"},
                    ),
                    html.Label("City", id="city-label", className="sidebar-section-label", style={"marginTop": "10px"}),
                    dbc.Input(
                        id="city-input", placeholder="e.g. Wädenswil", type="text",
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
        style={"position": "fixed", "top": "10px", "left": "10px", "zIndex": 1200, "backgroundColor": "#f8f9fa", "borderRadius": "8px"}
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

@app.callback(Output("city-store", "data"), Input("city-input", "value"))
def save_city(value):
    return value or ""

@app.callback(Output("language-store", "data"), Input("language-dropdown", "value"))
def update_language(lang):
    return lang or "en"

@app.callback(
    [Output("language-label", "children"), Output("zip-label", "children"),
     Output("city-label", "children"),
     Output("new-chat-label", "children"), Output("history-label", "children"),
     Output("chat-input", "placeholder"), Output("upload-button-text", "children"),
     Output("zip-input", "placeholder"), Output("city-input", "placeholder")],
    Input("language-store", "data"),
)
def update_labels(language):
    texts = get_texts(language)
    return (texts["language_label"], texts["zip_label"], texts["city_label"],
            texts["new_chat"], texts["chat_history_label"], texts["chat_input"],
            f" {texts['upload_button']}", texts["zip_placeholder"], texts["city_placeholder"])

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
     Input("zip-store", "data"), Input("city-store", "data")],
)
def render_chat(language, sessions, active_session, zip_code, city):
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

        elif msg.get("role") == "thinking":
            return html.Div([
                html.Img(src="/assets/robo_head.png", style=AVATAR),
                html.Div([
                    html.Span("●", style={"animation": "pulse 1s infinite", "marginRight": "4px"}),
                    html.Span("●", style={"animation": "pulse 1s infinite 0.2s", "marginRight": "4px"}),
                    html.Span("●", style={"animation": "pulse 1s infinite 0.4s"}),
                # borderRadius: TL TR BR BL → 4px unten-links = Spitze zeigt zum Avatar (links)
                ], style={**CHAT_BUBBLE_BOT, "fontSize": "13px", "letterSpacing": "3px", "padding": "10px 16px", "borderRadius": "16px 16px 16px 4px"}),
            ], style=CHAT_ROW)

        elif msg.get("role") == "image_result":
            image_src = msg.get("image_src", "")
            return html.Div([
                html.Img(src="/assets/robo_head.png", style=AVATAR),
                html.Div([html.Div([
                    html.Img(src=image_src, style={"maxWidth": "100%", "maxHeight": "180px", "borderRadius": "8px", "marginBottom": "12px"}) if image_src else None,
                    html.Div(msg.get("result_data", {}).get("content", [])),
                ], className="image-result-card")], style={**CHAT_BUBBLE_BOT, "backgroundColor": "transparent", "border": "none", "padding": "0"}),
            ], style=CHAT_ROW)

        # ---------------------------------------------------------------
        # Change 4: location_result — show ONLY the map card.
        #
        # Previously: text bubble (with Google Maps links) + map card
        #             → same links appeared twice (duplicate).
        #
        # Now: just the map card whose popups already contain
        #      address, opening hours AND the Google Maps link.
        #      The agent text is kept in the store but not rendered.
        #
        # Fallback: if no zip/city was stored, render the text response
        #           as a plain assistant bubble so nothing is lost.
        # ---------------------------------------------------------------
        elif msg.get("role") == "location_result":
            zip_used  = msg.get("zip_code", zip_code)
            city_used = msg.get("city", city)
            lang_used = msg.get("language", language)
            if zip_used or city_used:
                loc = city_used or zip_used
                if lang_used == "de":
                    intro = (
                        f"Hier sind Recycling-Standorte in der Nähe von **{loc}**. "
                        f"Nutzen Sie die Filter auf der Karte, um Sammelstellen für "
                        f"Aluminium, Glas, Karton, PET, Sonderabfall und mehr zu finden."
                    )
                else:
                    intro = (
                        f"Here are recycling locations near **{loc}**. "
                        f"Use the filters on the map to find specific collection points "
                        f"for aluminium, glass, cardboard, PET, hazardous waste, and more."
                    )
                return html.Div([
                    html.Img(src="/assets/robo_head.png", style=AVATAR),
                    html.Div([
                        dcc.Markdown(intro, className="markdown-content", style={"marginBottom": "12px"}),
                        make_map_card(zip_used, lang_used, city=city_used),
                    ], style={"flex": "1", "minWidth": "0"}),
                ], style=CHAT_ROW)
            # No location available → plain text fallback
            return html.Div([
                html.Img(src="/assets/robo_head.png", style=AVATAR),
                html.Div(
                    dcc.Markdown(msg.get("content", ""), className="markdown-content", style={"margin": 0}),
                    style={**CHAT_BUBBLE_BOT, "maxWidth": "fit-content"},
                ),
            ], style=CHAT_ROW)

        else:
            return html.Div([
                html.Img(src="/assets/robo_head.png", style=AVATAR),
                html.Div(dcc.Markdown(msg.get("content", ""), className="markdown-content", style={"margin": 0}), style={**CHAT_BUBBLE_BOT, "maxWidth": "fit-content"}),
            ], style=CHAT_ROW)

    return html.Div([make_msg(m) for m in messages], style={"padding": "28px 0", "maxWidth": "920px", "margin": "0 auto"})

@app.callback(
    [Output("sessions-store", "data", allow_duplicate=True), Output("active-session-store", "data", allow_duplicate=True),
     Output("new-chat-session-id", "data")],
    Input("new-chat-btn", "n_clicks"), State("sessions-store", "data"),
    prevent_initial_call=True,
)
def new_chat(n_clicks, sessions):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    sessions = sessions or {}
    new_id = str(uuid.uuid4())[:8]
    sessions[new_id] = []
    return sessions, new_id, new_id

@app.callback(
    [Output("zip-input", "value"), Output("city-input", "value")],
    Input("new-chat-session-id", "data"),
    prevent_initial_call=True,
)
def clear_inputs_on_new_chat(new_session_id):
    if not new_session_id:
        raise dash.exceptions.PreventUpdate
    return "", ""

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


# ---------------------------------------------------------------------------
# Two-step send — Step 1: show user message + thinking bubble
# ---------------------------------------------------------------------------
@app.callback(
    [Output("sessions-store", "data", allow_duplicate=True),
     Output("active-session-store", "data", allow_duplicate=True),
     Output("pending-message-store", "data"),
     Output("chat-input", "value")],
    [Input("send-button", "n_clicks"), Input("chat-input", "n_submit")],
    [State("chat-input", "value"), State("sessions-store", "data"),
     State("active-session-store", "data"), State("language-store", "data"),
     State("zip-store", "data"), State("city-store", "data"),
     State("agent-history-store", "data")],
    prevent_initial_call=True,
)
def send_message_step1(n_clicks, n_submit, user_text, sessions, active_session, language, zip_code, city, agent_history):
    """Step 1: Immediately display user message + thinking bubble, store pending state."""
    if (not n_clicks and not n_submit) or not user_text or not user_text.strip():
        raise dash.exceptions.PreventUpdate

    sessions = sessions or {}
    agent_history = agent_history or {}

    if not active_session or active_session not in sessions:
        active_session = str(uuid.uuid4())[:8]
        sessions[active_session] = []

    user_text = user_text.strip()

    sessions[active_session].append({"role": "user", "content": user_text})
    sessions[active_session].append({"role": "thinking"})

    pending = {
        "user_text": user_text,
        "session_id": active_session,
        "language": language,
        "zip_code": zip_code,
        "city": city,
        "agent_history": agent_history,
    }

    return sessions, active_session, pending, ""


# Two-step send — Step 2: call agent, replace thinking bubble
@app.callback(
    [Output("sessions-store", "data", allow_duplicate=True),
     Output("agent-history-store", "data", allow_duplicate=True)],
    Input("pending-message-store", "data"),
    State("sessions-store", "data"),
    prevent_initial_call=True,
)
def send_message_step2(pending, sessions):
    """Step 2: Call agent and replace thinking bubble with real response."""
    if not pending or not isinstance(pending, dict):
        raise dash.exceptions.PreventUpdate

    sessions = sessions or {}
    user_text = pending.get("user_text", "")
    active_session = pending.get("session_id", "")
    language = pending.get("language", "en")
    zip_code = pending.get("zip_code", "")
    city = pending.get("city", "")
    agent_history = pending.get("agent_history", {})

    if not active_session or active_session not in sessions:
        raise dash.exceptions.PreventUpdate

    session_state = agent_history.get(active_session, {"scan_history": [], "conv_history": []})

    try:
        state = AgentState(
            user_message=user_text, image_path=None, zip_code=zip_code or None,
            language=language or "en", classification=None, guidelines=None, collection_points=None,
            input_type=None, needs_clarification=False, final_response=None,
            osm_elements=None, map_lat=None, map_lon=None,
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

    msgs = sessions[active_session]
    if msgs and msgs[-1].get("role") == "thinking":
        msgs.pop()

    if input_type == "location" and (zip_code or city):
        # Store content for potential fallback, but the bubble won't show it
        # (the map popup already contains address + hours + Google Maps link).
        msgs.append({
            "role": "location_result",
            "content": response,
            "zip_code": zip_code,
            "city": city,
            "language": language or "en",
        })
    else:
        msgs.append({"role": "assistant", "content": response})

    sessions[active_session] = msgs
    return sessions, agent_history


@app.callback(
    [Output("sessions-store", "data", allow_duplicate=True),
     Output("active-session-store", "data", allow_duplicate=True),
     Output("pending-image-store", "data")],
    [Input("image-upload", "contents"), State("image-upload", "filename"),
     State("sessions-store", "data"), State("active-session-store", "data")],
    prevent_initial_call=True,
)
def handle_image_step1(contents, filename, sessions, active_session):
    """Step 1: Show thinking bubble immediately when image is uploaded."""
    if not contents:
        raise dash.exceptions.PreventUpdate
    sessions = sessions or {}
    if not active_session or active_session not in sessions:
        active_session = str(uuid.uuid4())[:8]
        sessions[active_session] = []
    sessions[active_session].append({"role": "thinking"})
    return sessions, active_session, {"contents": contents, "filename": filename}


@app.callback(
    [Output("sessions-store", "data", allow_duplicate=True),
     Output("active-session-store", "data", allow_duplicate=True),
     Output("agent-history-store", "data", allow_duplicate=True)],
    [Input("pending-image-store", "data"),
     State("language-store", "data"), State("sessions-store", "data"),
     State("active-session-store", "data"), State("zip-store", "data"),
     State("city-store", "data"), State("agent-history-store", "data")],
    prevent_initial_call=True,
)
def handle_image_step2(pending, language, sessions, active_session, zip_code, city, agent_history):
    """Step 2: Classify image, replace thinking bubble with result."""
    if not pending or not isinstance(pending, dict):
        raise dash.exceptions.PreventUpdate

    contents = pending["contents"]
    filename = pending["filename"]
    sessions = sessions or {}
    agent_history = agent_history or {}
    texts = get_texts(language)

    if not active_session or active_session not in sessions:
        active_session = str(uuid.uuid4())[:8]
        sessions[active_session] = []

    result_content = None
    tmp_path = None
    normalized_path = None
    try:
        header, encoded = contents.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        suffix = Path(filename).suffix or ".jpg"
        tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}{suffix}")
        with open(tmp_path, "wb") as f:
            f.write(img_bytes)

        # Normalize: handle HEIC/WEBP/PNG/EXIF rotation → plain RGB JPEG
        # pillow-heif (registered at import) lets PIL open HEIC transparently.
        normalized_path = os.path.join(tempfile.gettempdir(), f"norm_{uuid.uuid4().hex}.jpg")
        with Image.open(tmp_path) as img:
            img = ImageOps.exif_transpose(img)  # fix phone rotation metadata
            img = img.convert("RGB")             # HEIC/WEBP/PNG → JPEG-safe
            img.save(normalized_path, "JPEG", quality=90)

        session_state = agent_history.get(active_session, {"scan_history": [], "conv_history": []})
        state = AgentState(
            user_message="How do I dispose of this?" if language == "en" else "Wie entsorge ich das?",
            image_path=normalized_path, zip_code=zip_code or None, language=language or "en",
            classification=None, guidelines=None, collection_points=None,
            input_type=None, needs_clarification=False, final_response=None,
            osm_elements=None, map_lat=None, map_lon=None,
            scan_history=session_state["scan_history"], conversation_history=session_state["conv_history"],
        )
        result = agent.invoke(state, config={"configurable": {"thread_id": active_session}})
        classification = result.get("classification") or {}
        category = classification.get("category", "unknown").replace("_", " ").title()

        result_content = [
            html.H5(f"{texts['detected']}: {category}", style={"fontWeight": "600", "marginBottom": "8px", "fontSize": "17px", "color": "#1a202c"}),
            dcc.Markdown(result["final_response"], className="markdown-content", style={"marginBottom": 0}),
        ]
        agent_history[active_session] = {
            "scan_history": result.get("scan_history", []),
            "conv_history": result.get("conversation_history", []),
        }

    except Exception as e:
        logger.error(f"Image error: {e}")
        result_content = [html.Div(f"Error processing image: {e}", style={"color": "#dc2626", "fontWeight": "500"})]

    finally:
        for p in (tmp_path, normalized_path):
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass

    # Always remove the thinking bubble and append result, even after an error.
    msgs = sessions.get(active_session, [])
    if msgs and msgs[-1].get("role") == "thinking":
        msgs.pop()
    msgs.append({"role": "image_result", "image_src": contents, "result_data": {"content": result_content}})
    sessions[active_session] = msgs
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