#!/usr/bin/env python3
"""Dash dashboard and Flask server for the Swiss waste recycling chatbot.

Implements the chat UI, image upload and classification flow, Leaflet map
integration via an embedded iframe, and all Dash callbacks. Served as a
Gunicorn app via app.py.
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
from urllib.parse import quote as url_quote

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
from chatbot.knowledge_base import Config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

agent = build_agent()


# ---------------------------------------------------------------------------
# CATEGORY LABELS: human-readable names for RECYCLING_GUIDE keys (de + en).
# Fallback for unknown keys: key.replace('_', ' ').title()
# ---------------------------------------------------------------------------
CATEGORY_LABELS: dict = {
    "white_glass":               {"de": "Weissglas",                          "en": "White Glass"},
    "brown_glass":               {"de": "Braunglas",                          "en": "Brown Glass"},
    "green_glass":               {"de": "Grünglas",                           "en": "Green Glass"},
    "metal":                     {"de": "Metall",                             "en": "Metal"},
    "aluminium":                 {"de": "Aluminium / Getränkedose",           "en": "Aluminium / Beverage Can"},
    "paper":                     {"de": "Altpapier",                          "en": "Paper"},
    "cardboard":                 {"de": "Karton / Papiertragtasche",          "en": "Cardboard / Paper Bag"},
    "composite_carton":          {"de": "Getränkekarton (Tetrapack)",         "en": "Composite Carton (Tetrapack)"},
    "pet":                       {"de": "PET-Flasche",                        "en": "PET Bottle"},
    "plastic":                   {"de": "Kunststoff / Folie",                 "en": "Plastic / Film"},
    "rigid_plastic_container":   {"de": "Hartplastikbehälter",                "en": "Rigid Plastic Container"},
    "organic_waste":             {"de": "Bio- / Gartenabfall",                "en": "Organic / Garden Waste"},
    "residual_waste":            {"de": "Kehricht / Restmüll",                "en": "Residual Waste"},
    "non_waste":                 {"de": "Elektronik / Textilien (Sonderwege)","en": "Electronics / Clothing (Special Routes)"},
    "hazardous_waste_(battery)": {"de": "Batterie / Akku",                    "en": "Battery / Accumulator"},
    "damaged_battery":           {"de": "Beschädigter / aufgeblähter Akku",   "en": "Damaged / Swollen Battery"},
    "aerosol_can":               {"de": "Spraydose (Sonderabfall)",           "en": "Aerosol Can (Hazardous Waste)"},
    "incandescent_lamp":         {"de": "Glüh- / Halogenlampe",              "en": "Incandescent / Halogen Bulb"},
    "lamp_special_disposal":     {"de": "LED / Energiespar- / Leuchtstofflampe", "en": "LED / CFL / Fluorescent Tube"},
    "waste_oil":                 {"de": "Altöl / Speiseöl",                  "en": "Waste Oil / Cooking Oil"},
    "thermal_receipt":           {"de": "Kassenzettel (Thermopapier)",        "en": "Receipt (Thermal Paper)"},
    "eco_receipt":               {"de": "Öko-Bon (blauer Kassenzettel)",      "en": "Eco Receipt (Blue Receipt)"},
    # Classifier-only categories (not in RECYCLING_GUIDE but in Config.WASTE_CATEGORIES)
    "plastic_aluminium":         {"de": "Plastik-Aluminium-Verbund",          "en": "Plastic-Aluminium Composite"},
    "white_glass_metal":         {"de": "Weissglas mit Metallverschluss",     "en": "White Glass with Metal Lid"},
}

# ---------------------------------------------------------------------------
# MAP COMPONENT CACHE
# ---------------------------------------------------------------------------
# Caches the map_widget component keyed by message id (mid, a uuid4 hex).
# Goal: return the same Python object on re-renders so Dash serialises
# byte-for-byte identical JSON. React reconciliation then has no reason to
# unmount and remount the iframe. The key= props on the widget div and the
# iframe are the actual mechanism that steers React away from re-mounting.
# Safety: mids are uuid4 hex strings generated once at message creation;
# new-chat sessions get new mids, so no stale or wrong-location data ever
# surfaces from the cache. Capped at _MAP_CACHE_MAX entries to bound RAM.
_MAP_COMPONENT_CACHE: dict = {}
_MAP_CACHE_MAX = 100

# ---------------------------------------------------------------------------
# UI TEXT
# ---------------------------------------------------------------------------
def get_texts(language: str) -> Dict[str, str]:
    """Return UI text strings for the given language ('en' or 'de')."""
    if language == "de":
        return {
            "title": "Swiss Recycling Assistant",
            "chat_input": "Stellen Sie eine Frage...",
            "new_chat": "Neuer Chat",
            "language_label": "Sprache",
            "city_label": "Ort / Gemeinde",
            "city_placeholder": "z.B. Goldach",
            "welcome_title": "Hi, ich bin dein Swiss Recycling Assistant",
            "welcome_text": "Du bist unsicher, wie du etwas in der Schweiz recyceln sollst? Lade ein Foto deines Abfalls hoch oder stelle mir einfach direkt deine Frage.",
            "no_chats": "Keine Chatverläufe",
            "upload_button": "Bild",
            "image_analyzed": "Bild analysiert",
            "chat_history_label": "Verlauf",
            "detected": "Erkannt",
            "map_button": "Alle Sammelstellen anzeigen",
            "map_title": "Sammelstellen in der Nähe",
            "info_button": "Info",
            "info_modal_title": "Über diesen Assistenten",
            "info_block1_title": "Was kann der Assistent?",
            "info_block1_items": [
                "Bild eines Abfall-Gegenstands hochladen → automatische Klassifizierung",
                "Per Text fragen, wie etwas entsorgt wird",
                "Ort/Gemeinde eingeben → Karte mit Sammelstellen in der Nähe",
                "Sprache wählen (Deutsch / English)",
            ],
            "info_block2_img_title": "Bildklassifizierung",
            "info_block2_text_title": "Textfragen",
            "info_block2_text_body": "Per Text kannst du zu allen möglichen Recycling-Themen fragen – nicht nur zu den oben genannten Kategorien. Der Assistent kennt darüber hinaus viele weitere Fälle (z.B. Leuchtmittel, Spraydosen, Altöl, Sonderabfall, beschädigte Akkus u.v.m.).",
            "info_block3_title": "Hinweise & Grenzen",
            "info_block3_items": [
                "Der Assistent bildet Schweizer Richtlinien (insb. Swiss Recycle) ab.",
                "Regionale und kommunale Unterschiede sind möglich – im Zweifel Gemeinde oder lokale Sammelstelle fragen.",
                "Der Bot ersetzt keine offizielle Auskunft; Angaben ohne Gewähr.",
                "Die Bildklassifizierung kann Fehler machen – bei Unsicherheit einfach nachfragen.",
            ],
            "info_close": "Schliessen",
            "info_ba_note": "Dieser Prototyp wurde im Rahmen einer Bachelorarbeit an der ZHAW Wädenswil entwickelt.",
            "footer_ba_note": "Bachelorarbeit · ZHAW Wädenswil",
        }
    return {
        "title": "Swiss Recycling Assistant",
        "chat_input": "Ask a question...",
        "new_chat": "New Chat",
        "language_label": "Language",
        "city_label": "City / Town",
        "city_placeholder": "e.g. Goldach",
        "welcome_title": "Hi, I'm your Swiss Recycling Assistant",
        "welcome_text": "Not sure how to recycle something in Switzerland? Upload a picture of your waste item or just ask me directly.",
        "no_chats": "No chat history",
        "upload_button": "Image",
        "image_analyzed": "Image analyzed",
        "chat_history_label": "History",
        "detected": "Detected",
        "map_button": "View all collection points",
        "map_title": "Nearby collection points",
        "info_button": "Info",
        "info_modal_title": "About this Assistant",
        "info_block1_title": "What can the assistant do?",
        "info_block1_items": [
            "Upload a photo of a waste item → automatic classification",
            "Ask in text how to dispose of something",
            "Enter your city/town → map with nearby collection points",
            "Choose language (Deutsch / English)",
        ],
        "info_block2_img_title": "Image Classification",
        "info_block2_text_title": "Text questions",
        "info_block2_text_body": "Via text you can ask about any recycling topic — not just the categories listed above. The assistant also handles many additional cases (e.g. light bulbs, aerosol cans, waste oil, hazardous waste, damaged batteries, and more).",
        "info_block3_title": "Limits & Notes",
        "info_block3_items": [
            "The assistant reflects Swiss guidelines (mainly Swiss Recycle).",
            "Regional and municipal differences are possible — when in doubt, contact your municipality or local collection point.",
            "This bot does not replace official advice; no guarantee of accuracy.",
            "Image classification can make mistakes — feel free to ask for clarification.",
        ],
        "info_close": "Close",
        "info_ba_note": "This prototype was developed as part of a Bachelor's thesis at ZHAW Wädenswil.",
        "footer_ba_note": "Bachelor's thesis · ZHAW Wädenswil",
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


def build_map_html(city, lat, lon, language):
    """Build standalone Leaflet map HTML for the given city and coordinates."""
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
 .bindPopup('<b>{home_label}</b><br>{city}').addTo(map);

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
    """Flask route that serves the Leaflet map HTML for a given city and coordinates."""
    city     = flask_request.args.get('city', '')
    lat_str  = flask_request.args.get('lat', '')
    lon_str  = flask_request.args.get('lon', '')
    language = flask_request.args.get('lang', 'en')
    lat = float(lat_str) if lat_str else None
    lon = float(lon_str) if lon_str else None
    return Response(build_map_html_str(city, language, lat, lon), mimetype='text/html')


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


def build_map_html_str(city: str, language: str, lat: float = None, lon: float = None) -> str:
    """Build the Leaflet map HTML string for the given city and coordinates."""
    location_label = city or "CH"
    query_str = (city or "Schweiz").replace("'", "\\'")

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

    # When the agent already resolved coordinates, bake them into the JS init block.
    # This skips the client-side Nominatim geocoding call so the map centers instantly.
    init_lat  = lat if lat is not None else 46.9481
    init_lon  = lon if lon is not None else 7.4474
    init_zoom = 15  if lat is not None else 8

    if lat is not None and lon is not None:
        init_js = (
            f"L.marker([{lat}, {lon}],{{icon:hi}})"
            f".bindPopup('<b>{home_label}</b><br>{location_label}').addTo(map);\n"
            f"fetchOSM({lat}, {lon});"
        )
    else:
        init_js = (
            f"fetch('https://nominatim.openstreetmap.org/search?q='+encodeURIComponent('{query_str} Schweiz')+'&format=json&limit=1&countrycodes=ch')\n"
            f"  .then(function(r){{return r.json();}})\n"
            f"  .then(function(data){{\n"
            f"    var clat = (data && data[0]) ? parseFloat(data[0].lat) : 46.9481;\n"
            f"    var clon = (data && data[0]) ? parseFloat(data[0].lon) : 7.4474;\n"
            f"    map.setView([clat, clon], 15);\n"
            f"    L.marker([clat, clon],{{icon:hi}}).bindPopup('<b>{home_label}</b><br>{location_label}').addTo(map);\n"
            f"    fetchOSM(clat, clon);\n"
            f"  }})\n"
            f"  .catch(function(){{\n"
            f"    map.setView([46.9481, 7.4474], 15);\n"
            f"    fetchOSM(46.9481, 7.4474);\n"
            f"  }});"
        )

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
var map = L.map('map').setView([{init_lat}, {init_lon}], {init_zoom});
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

{init_js}
</script>
</body>
</html>"""

    return map_html


def make_map_card(city: str, language: str, lat: float = None, lon: float = None):
    """Wrap build_map_html_str in a Dash card component."""
    print(f"[MAP] make_map_card: lat={lat}, lon={lon}, city={city!r}")
    texts = get_texts(language)
    map_html = build_map_html_str(city, language, lat, lon)
    return html.Div([
        html.Div(texts["map_title"], style={"fontWeight": "600", "fontSize": "14px", "color": "#1e40af", "marginBottom": "8px"}),
        html.Iframe(srcDoc=map_html, style={"width": "100%", "height": "340px", "border": "none", "borderRadius": "8px"}),
    ], className="map-card", style={"width": "100%"})




app.layout = html.Div([
    dcc.Store(id="language-store", data="en"),
    dcc.Store(id="sessions-store", data={}),
    dcc.Store(id="active-session-store", data=""),
    dcc.Store(id="sidebar-collapsed-store", data=False),
    dcc.Store(id="city-store", data=""),
    dcc.Store(id="agent-history-store", data={}),
    dcc.Store(id="pending-message-store", data=None),
    dcc.Store(id="pending-image-store", data=None),
    dcc.Store(id="new-chat-session-id", data=None),

    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id="info-modal-title"), close_button=False),
        dbc.ModalBody(id="info-modal-body"),
        dbc.ModalFooter(
            dbc.Button(id="info-modal-close-btn", n_clicks=0, color="secondary", outline=True, size="sm"),
        ),
    ], id="info-modal", is_open=False, scrollable=True, size="lg"),

    html.Div([
        html.Div([
            html.Div(
                html.Span("Swiss Recycling Assistant", style={"fontSize": "15px", "fontWeight": "600", "color": "#1a202c"}),
                className="sidebar-content",
                style={"marginTop": "48px", "marginBottom": "20px", "paddingLeft": "2px"}
            ),
            html.Div([
                dbc.Button(
                    [html.I(className="bi bi-info-circle", style={"marginRight": "8px"}), html.Span("Info", id="info-btn-label")],
                    id="info-btn",
                    n_clicks=0,
                    style={"width": "100%", "marginBottom": "16px", "borderRadius": "10px", "padding": "10px 16px",
                           "fontSize": "13px", "fontWeight": "500", "backgroundColor": "#f1f5f9",
                           "border": "1px solid #e2e8f0", "color": "#4b5563"},
                ),
                html.Div([
                    html.Label("Language", id="language-label", className="sidebar-section-label"),
                    dcc.Dropdown(
                        id="language-dropdown",
                        options=[{"label": "English", "value": "en"}, {"label": "Deutsch", "value": "de"}],
                        value="en", clearable=False, className="language-dropdown",
                    ),
                ], style={"marginBottom": "16px"}),
                html.Div([
                    html.Label("City / Town", id="city-label", className="sidebar-section-label"),
                    dbc.Input(
                        id="city-input", placeholder="e.g. Goldach", type="text",
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
        ]),
        html.Div(id="footer-ba-note",
                 style={"marginTop": "8px", "fontSize": "11px", "color": "#9ca3af", "fontStyle": "italic"}),
    ], className="app-footer"),
], style={"display": "flex", "flexDirection": "column", "minHeight": "100vh"})


# ---------------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------------

@app.callback(Output("city-store", "data"), Input("city-input", "value"))
def save_city(value):
    """Persist city input to a store so other callbacks can read it without coupling to the input component."""
    return value or ""

@app.callback(Output("language-store", "data"), Input("language-dropdown", "value"))
def update_language(lang):
    """Persist language selection to a store."""
    return lang or "en"

@app.callback(
    [Output("language-label", "children"), Output("city-label", "children"),
     Output("new-chat-label", "children"), Output("history-label", "children"),
     Output("chat-input", "placeholder"), Output("upload-button-text", "children"),
     Output("city-input", "placeholder"), Output("info-btn-label", "children"),
     Output("footer-ba-note", "children")],
    Input("language-store", "data"),
)
def update_labels(language):
    """Update all language-dependent sidebar labels and placeholders when the language changes."""
    texts = get_texts(language)
    return (texts["language_label"], texts["city_label"],
            texts["new_chat"], texts["chat_history_label"], texts["chat_input"],
            f" {texts['upload_button']}", texts["city_placeholder"], texts["info_button"],
            texts["footer_ba_note"])

@app.callback(
    [Output("sidebar", "className"), Output("main-content", "className")],
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar-collapsed-store", "data"),
    prevent_initial_call=True,
)
def toggle_sidebar(n_clicks, collapsed):
    """Toggle sidebar CSS class between collapsed and expanded."""
    new_collapsed = not collapsed
    sidebar_class = "collapsed" if new_collapsed else "open"
    main_class = "collapsed" if new_collapsed else ""
    return sidebar_class, main_class

@app.callback(Output("sidebar-collapsed-store", "data"), Input("sidebar-toggle", "n_clicks"),
              State("sidebar-collapsed-store", "data"), prevent_initial_call=True)
def save_collapsed(n_clicks, collapsed):
    """Persist sidebar collapsed state to a store."""
    return not collapsed

@app.callback(
    Output("sidebar-sessions", "children"),
    [Input("sessions-store", "data"), Input("active-session-store", "data"), Input("language-store", "data")],
)
def update_session_list(sessions, active_session, language):
    """Render the sidebar session list, highlighting the active session."""
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
    [Input("language-store", "data"), Input("sessions-store", "data"), Input("active-session-store", "data")],
    [State("city-store", "data")],
)
def render_chat(language, sessions, active_session, city):
    """Re-render the full chat content whenever sessions or the active session changes."""
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

    # idx gives React a stable key per message so it reuses DOM nodes instead of
    # recreating them, which is critical for keeping the map iframe from reloading.
    def make_msg(msg, idx):
        k = msg.get("mid") or f"msg-{idx}"
        if msg.get("role") == "user":
            return html.Div([
                html.Div(msg.get("content", ""), style=CHAT_BUBBLE_USER),
                html.Div(html.I(className="bi bi-person-circle", style={"fontSize": "20px", "color": "#4a7ba7"}),
                         style={**AVATAR, "backgroundColor": "#e8f0f7", "display": "flex", "alignItems": "center", "justifyContent": "center"}),
            ], key=k, style={**CHAT_ROW, "justifyContent": "flex-end"})

        elif msg.get("role") == "thinking":
            return html.Div([
                html.Img(src="/assets/robo_head.png", style=AVATAR),
                html.Div([
                    html.Span("●", style={"animation": "pulse 1s infinite", "marginRight": "4px"}),
                    html.Span("●", style={"animation": "pulse 1s infinite 0.2s", "marginRight": "4px"}),
                    html.Span("●", style={"animation": "pulse 1s infinite 0.4s"}),
                ], style={**CHAT_BUBBLE_BOT, "fontSize": "13px", "letterSpacing": "3px", "padding": "10px 16px", "borderRadius": "16px 16px 16px 4px"}),
            ], key=k, style=CHAT_ROW)

        elif msg.get("role") == "image_result":
            image_src = msg.get("image_src", "")
            return html.Div([
                html.Img(src="/assets/robo_head.png", style=AVATAR),
                html.Div([html.Div([
                    html.Img(src=image_src, style={"maxWidth": "100%", "maxHeight": "180px", "borderRadius": "8px", "marginBottom": "12px"}) if image_src else None,
                    html.Div(msg.get("result_data", {}).get("content", [])),
                ], className="image-result-card")], style={**CHAT_BUBBLE_BOT, "backgroundColor": "transparent", "border": "none", "padding": "0"}),
            ], key=k, style=CHAT_ROW)

        elif msg.get("role") == "location_result":
            city_used = msg.get("city", city)
            lang_used = msg.get("language", language)
            lat_used  = msg.get("map_lat")
            lon_used  = msg.get("map_lon")
            mid       = msg.get("mid") or f"loc-{idx}"
            if city_used:
                if lang_used == "de":
                    intro = (
                        f"Hier sind Recycling-Standorte in der Nähe von **{city_used}**. "
                        f"Nutzen Sie die Filter auf der Karte, um Sammelstellen für "
                        f"Aluminium, Glas, Karton, PET, Sonderabfall und mehr zu finden."
                    )
                else:
                    intro = (
                        f"Here are recycling locations near **{city_used}**. "
                        f"Use the filters on the map to find specific collection points "
                        f"for aluminium, glass, cardboard, PET, hazardous waste, and more."
                    )
                if mid not in _MAP_COMPONENT_CACHE:
                    # Use a URL-based src so the browser never reloads the iframe
                    # when its src stays identical. srcDoc re-evaluates aggressively.
                    lat_part = f"&lat={lat_used}" if lat_used is not None else ""
                    lon_part = f"&lon={lon_used}" if lon_used is not None else ""
                    map_src = f"/map?city={url_quote(city_used)}&lang={lang_used}{lat_part}{lon_part}"
                    _widget = html.Div([
                        html.Div(get_texts(lang_used)["map_title"], style={"fontWeight": "600", "fontSize": "14px", "color": "#1e40af", "marginBottom": "8px"}),
                        html.Iframe(
                            key=f"iframe-{mid}",
                            id=f"map-iframe-{mid}",
                            src=map_src,
                            style={"width": "100%", "height": "340px", "border": "none", "borderRadius": "8px"},
                        ),
                    ], key=f"mapwidget-{mid}", className="map-card", style={"width": "100%"})
                    if len(_MAP_COMPONENT_CACHE) >= _MAP_CACHE_MAX:
                        _MAP_COMPONENT_CACHE.pop(next(iter(_MAP_COMPONENT_CACHE)))
                    _MAP_COMPONENT_CACHE[mid] = _widget
                map_widget = _MAP_COMPONENT_CACHE[mid]
                return html.Div([
                    html.Img(src="/assets/robo_head.png", style=AVATAR),
                    html.Div([
                        dcc.Markdown(intro, className="markdown-content", style={"marginBottom": "12px"}),
                        map_widget,
                    ], style={"flex": "1", "minWidth": "0"}),
                ], key=k, style=CHAT_ROW)
            return html.Div([
                html.Img(src="/assets/robo_head.png", style=AVATAR),
                html.Div(
                    dcc.Markdown(msg.get("content", ""), className="markdown-content", style={"margin": 0}),
                    style={**CHAT_BUBBLE_BOT, "maxWidth": "fit-content"},
                ),
            ], key=k, style=CHAT_ROW)

        else:
            return html.Div([
                html.Img(src="/assets/robo_head.png", style=AVATAR),
                html.Div(dcc.Markdown(msg.get("content", ""), className="markdown-content", style={"margin": 0}), style={**CHAT_BUBBLE_BOT, "maxWidth": "fit-content"}),
            ], key=k, style=CHAT_ROW)

    return html.Div([make_msg(m, i) for i, m in enumerate(messages)], style={"padding": "28px 0", "maxWidth": "920px", "margin": "0 auto"})


@app.callback(
    [Output("sessions-store", "data", allow_duplicate=True), Output("active-session-store", "data", allow_duplicate=True),
     Output("new-chat-session-id", "data")],
    Input("new-chat-btn", "n_clicks"), State("sessions-store", "data"),
    prevent_initial_call=True,
)
def new_chat(n_clicks, sessions):
    """Create a new empty chat session and make it active."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    sessions = sessions or {}
    new_id = str(uuid.uuid4())[:8]
    sessions[new_id] = []
    return sessions, new_id, new_id

@app.callback(
    Output("city-input", "value"),
    Input("new-chat-session-id", "data"),
    prevent_initial_call=True,
)
def clear_inputs_on_new_chat(new_session_id):
    """Clear the city input when a new chat starts."""
    if not new_session_id:
        raise dash.exceptions.PreventUpdate
    return ""

@app.callback(
    Output("active-session-store", "data", allow_duplicate=True),
    Input({"type": "session-select", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_session(n_clicks):
    """Switch the active session when a history item is clicked."""
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
    """Delete a session and fall back to the first remaining session."""
    if not ctx.triggered_id or not any(n for n in n_clicks if n):
        raise dash.exceptions.PreventUpdate
    sid = ctx.triggered_id["index"]
    sessions = sessions or {}
    if sid in sessions:
        del sessions[sid]
    new_active = active_session if active_session != sid else (list(sessions.keys())[0] if sessions else "")
    return sessions, new_active


# ---------------------------------------------------------------------------
# INFO MODAL: open/close and language-aware body rendering
# ---------------------------------------------------------------------------
@app.callback(
    Output("info-modal", "is_open"),
    [Input("info-btn", "n_clicks"), Input("info-modal-close-btn", "n_clicks")],
    State("info-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_info_modal(_n_open, _n_close, is_open):
    """Open or close the info modal on button clicks."""
    return not is_open


@app.callback(
    [Output("info-modal-title", "children"),
     Output("info-modal-body", "children"),
     Output("info-modal-close-btn", "children")],
    [Input("language-store", "data"), Input("info-modal", "is_open")],
    prevent_initial_call=True,
)
def render_info_modal(language, is_open):
    """Populate info modal content in the current language when the modal opens."""
    if not is_open:
        raise dash.exceptions.PreventUpdate
    texts = get_texts(language)

    # Block 1: what the assistant can do
    block1 = html.Div([
        html.H6(texts["info_block1_title"], style={"fontWeight": "600", "marginBottom": "8px"}),
        html.Ul(
            [html.Li(item, style={"marginBottom": "4px"}) for item in texts["info_block1_items"]],
            style={"paddingLeft": "20px"},
        ),
    ], style={"marginBottom": "20px"})

    # Block 2: classifier categories from Config.WASTE_CATEGORIES (the model's training labels)
    img_classes = Config.WASTE_CATEGORIES
    cat_labels = []
    for key in img_classes:
        label_dict = CATEGORY_LABELS.get(key, {})
        label = label_dict.get(language) or label_dict.get("en") or key.replace("_", " ").title()
        cat_labels.append(label)
    cat_labels.sort()
    mid = (len(cat_labels) + 1) // 2
    col1, col2 = cat_labels[:mid], cat_labels[mid:]

    n = len(img_classes)
    img_heading = (
        f"{texts['info_block2_img_title']} ({n} Kategorien)"
        if language == "de"
        else f"{texts['info_block2_img_title']} ({n} categories)"
    )

    block2 = html.Div([
        html.H6(img_heading, style={"fontWeight": "600", "marginBottom": "8px"}),
        dbc.Row([
            dbc.Col(
                html.Ul([html.Li(l, style={"marginBottom": "2px"}) for l in col1],
                        style={"paddingLeft": "20px", "marginBottom": "0"}),
                md=6,
            ),
            dbc.Col(
                html.Ul([html.Li(l, style={"marginBottom": "2px"}) for l in col2],
                        style={"paddingLeft": "20px", "marginBottom": "0"}),
                md=6,
            ),
        ]),
        html.Div([
            html.Strong(texts["info_block2_text_title"] + ": ", style={"fontSize": "13px"}),
            html.Span(texts["info_block2_text_body"], style={"fontSize": "13px", "color": "#4b5563"}),
        ], style={"marginTop": "14px", "padding": "10px 12px", "backgroundColor": "#f8fafc",
                  "borderRadius": "6px", "border": "1px solid #e2e8f0"}),
    ], style={"marginBottom": "20px"})

    # Block 3: limitations and notes
    block3 = html.Div([
        html.H6(texts["info_block3_title"], style={"fontWeight": "600", "marginBottom": "8px"}),
        html.Ul(
            [html.Li(item, style={"marginBottom": "4px"}) for item in texts["info_block3_items"]],
            style={"paddingLeft": "20px"},
        ),
    ])

    ba_note = html.Div(
        texts["info_ba_note"],
        style={"marginTop": "20px", "fontSize": "11px", "color": "#9ca3af",
               "textAlign": "center", "fontStyle": "italic"},
    )

    body = html.Div([block1, html.Hr(), block2, html.Hr(), block3, ba_note])
    return texts["info_modal_title"], body, texts["info_close"]


# ---------------------------------------------------------------------------
# Two-step send: Step 1 shows the user message and thinking bubble immediately
# ---------------------------------------------------------------------------
@app.callback(
    [Output("sessions-store", "data", allow_duplicate=True),
     Output("active-session-store", "data", allow_duplicate=True),
     Output("pending-message-store", "data"),
     Output("chat-input", "value")],
    [Input("send-button", "n_clicks"), Input("chat-input", "n_submit")],
    [State("chat-input", "value"), State("sessions-store", "data"),
     State("active-session-store", "data"), State("language-store", "data"),
     State("city-store", "data"), State("agent-history-store", "data")],
    prevent_initial_call=True,
)
def send_message_step1(n_clicks, n_submit, user_text, sessions, active_session, language, city, agent_history):
    """Step 1: Immediately display user message + thinking bubble, store pending state."""
    if (not n_clicks and not n_submit) or not user_text or not user_text.strip():
        raise dash.exceptions.PreventUpdate

    sessions = sessions or {}
    agent_history = agent_history or {}

    if not active_session or active_session not in sessions:
        active_session = str(uuid.uuid4())[:8]
        sessions[active_session] = []

    user_text = user_text.strip()

    sessions[active_session].append({"role": "user", "content": user_text, "mid": uuid.uuid4().hex})
    sessions[active_session].append({"role": "thinking", "mid": uuid.uuid4().hex})

    pending = {
        "user_text": user_text,
        "session_id": active_session,
        "language": language,
        "city": city,
        "agent_history": agent_history,
    }

    return sessions, active_session, pending, ""


# Two-step send: Step 2 calls the agent and replaces the thinking bubble
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
    city = pending.get("city", "")
    agent_history = pending.get("agent_history", {})

    if not active_session or active_session not in sessions:
        raise dash.exceptions.PreventUpdate

    session_state = agent_history.get(active_session, {"scan_history": [], "conv_history": []})

    try:
        state = AgentState(
            user_message=user_text, image_path=None, city=city or None,
            language=language or "en", classification=None, guidelines=None, collection_points=None,
            input_type=None, needs_clarification=False, final_response=None,
            osm_elements=None, map_lat=None, map_lon=None,
            scan_history=session_state["scan_history"], conversation_history=session_state["conv_history"],
        )
        result = agent.invoke(state, config={"configurable": {"thread_id": active_session}})
        response = result["final_response"]
        input_type = result.get("input_type", "text")
        print(f"[MAP] step2: input_type={input_type}, map_lat={result.get('map_lat')}, map_lon={result.get('map_lon')}")

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

    if input_type == "location" and city:
        msgs.append({
            "role": "location_result",
            "content": response,
            "city": city,
            "language": language or "en",
            "map_lat": result.get("map_lat"),
            "map_lon": result.get("map_lon"),
            "mid": uuid.uuid4().hex,
        })
    else:
        msgs.append({"role": "assistant", "content": response, "mid": uuid.uuid4().hex})

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
    sessions[active_session].append({"role": "thinking", "mid": uuid.uuid4().hex})
    return sessions, active_session, {"contents": contents, "filename": filename}


@app.callback(
    [Output("sessions-store", "data", allow_duplicate=True),
     Output("active-session-store", "data", allow_duplicate=True),
     Output("agent-history-store", "data", allow_duplicate=True)],
    [Input("pending-image-store", "data"),
     State("language-store", "data"), State("sessions-store", "data"),
     State("active-session-store", "data"),
     State("city-store", "data"), State("agent-history-store", "data")],
    prevent_initial_call=True,
)
def handle_image_step2(pending, language, sessions, active_session, city, agent_history):
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

        # Normalize to JPEG: handle HEIC/WEBP/PNG and fix EXIF rotation before classifying.
        # pillow-heif (registered at import) lets PIL open HEIC transparently.
        normalized_path = os.path.join(tempfile.gettempdir(), f"norm_{uuid.uuid4().hex}.jpg")
        with Image.open(tmp_path) as img:
            img = ImageOps.exif_transpose(img)  # fix phone rotation metadata
            img = img.convert("RGB")             # strip alpha channel and HEIC/WEBP encoding
            img.save(normalized_path, "JPEG", quality=90)

        session_state = agent_history.get(active_session, {"scan_history": [], "conv_history": []})
        state = AgentState(
            user_message="How do I dispose of this?" if language == "en" else "Wie entsorge ich das?",
            image_path=normalized_path, city=city or None, language=language or "en",
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
    msgs.append({"role": "image_result", "image_src": contents, "result_data": {"content": result_content}, "mid": uuid.uuid4().hex})
    sessions[active_session] = msgs
    return sessions, active_session, agent_history


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def open_browser():
    """Open the app in the default browser."""
    webbrowser.open("http://127.0.0.1:8050/")

if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.5, open_browser).start()
    app.run(debug=True, host="0.0.0.0", port=8050)