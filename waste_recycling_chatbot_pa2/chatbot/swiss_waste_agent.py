#!/usr/bin/env python3
"""LangGraph agent for the Swiss waste recycling chatbot.

Handles image classification, text-to-category lookup, geolocation, and
GPT-4o response generation. Coordinates are resolved once in geolocation_node
and passed through AgentState so the dashboard map never geocodes twice.
"""

from pathlib import Path
from typing import TypedDict, Optional, List
from dotenv import load_dotenv
import re
import requests
from huggingface_hub import hf_hub_download

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from chatbot.knowledge_base import WasteClassifier, RECYCLING_GUIDE

load_dotenv()

# ===========================================================================
# SETUP
# ===========================================================================

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

try:
    MODEL_PATH = hf_hub_download(
        repo_id="le7lum/swiss-waste-classifier",
        filename="finetuned_model.pth"
    )
    print(f"[INFO] Model loaded from Hugging Face Hub")
except Exception as e:
    print(f"[DEBUG] HF download failed: {e} - trying local path")
    MODEL_PATH = "./models/baseline/finetuned_model.pth"
    if not Path(MODEL_PATH).exists():
        MODEL_PATH = "../models/baseline/finetuned_model.pth"

classifier = WasteClassifier(model_path=MODEL_PATH)


# ===========================================================================
# TEXT-TO-CATEGORY MAPPING
# ===========================================================================

TEXT_CATEGORY_MAP = {
    # English
    "tetrapack": "composite_carton", "tetra pak": "composite_carton",
    "juice box": "composite_carton", "milk carton": "composite_carton",
    "juice carton": "composite_carton", "milk box": "composite_carton",
    "tetra brik": "composite_carton",

    # Papiertragtaschen (paper carrier bags): Swiss Recycle classifies these as cardboard, not paper.
    # Compound keys are listed before "paper bag" so span-overlap detection blocks the generic "paper" key.
    "paper carrier bag": "cardboard",   "paper bag": "cardboard",
    "papiertragtasche": "cardboard",    "papiertragetasche": "cardboard",
    "papiertüte": "cardboard",          "papiersack": "cardboard",
    "cardboard": "cardboard", "carton": "cardboard",
    "paper": "paper", "newspaper": "paper", "magazine": "paper",

    "plastic bottle": "pet", "pet bottle": "pet", "pet": "pet",

    "glass bottle": "white_glass", "wine bottle": "white_glass",
    "clear glass": "white_glass", "white glass": "white_glass",
    "beer bottle": "brown_glass",
    "brown glass": "brown_glass",
    "green bottle": "green_glass", "green glass": "green_glass",

    # Aerosol cans (hairspray, deodorant, whipped cream): Swiss Recycle classifies these as hazardous
    # waste (Sonderabfall), not aluminium/metal, not residual waste. Reason: pressurised containers
    # with propellant residue are an explosion hazard when compacted. Cantonal rules vary slightly.
    # Compound keys are listed before generic ones: "schlagrahm spraydose" before "spraydose".
    "schlagrahm spraydose": "aerosol_can",   "schlagrahm sprühdose": "aerosol_can",
    "rahm-spraydose": "aerosol_can",         "deodorant spray": "aerosol_can",
    "hair spray": "aerosol_can",             "aerosol can": "aerosol_can",
    "spray can": "aerosol_can",              "hairspray": "aerosol_can",
    "aerosol": "aerosol_can",                "haarspray": "aerosol_can",
    "haarlack": "aerosol_can",               "deospray": "aerosol_can",
    "rasierschaum": "aerosol_can",           "schaumfestiger": "aerosol_can",
    "sprührahm": "aerosol_can",              "spraydose": "aerosol_can",
    "sprühdose": "aerosol_can",

    "aluminium": "aluminium", "aluminum": "aluminium",
    "tin can": "aluminium", "can": "aluminium",
    "nespresso": "aluminium", "coffee capsule": "aluminium", "capsule": "aluminium",
    "foil": "aluminium", "yogurt lid": "aluminium", "tin foil": "aluminium",
    "beverage can": "aluminium",

    # Damaged/swollen batteries: require hazardous disposal with a safety warning.
    # Source: INOBAT (inobat.ch), BAFU guidelines for damaged/swollen Li-ion batteries.
    # NOTE: the image classifier maps all batteries to hazardous_waste_(battery); the text path
    # handles the swollen/damaged subtype, since visual distinction is not possible via the classifier.
    # These compound keys must appear before "battery"/"batterie" in dict order so the
    # knowledge_base_node loop matches the more specific category first.
    "swollen battery": "damaged_battery",       "swollen lithium": "damaged_battery",
    "damaged battery": "damaged_battery",       "leaking battery": "damaged_battery",
    "puffed battery": "damaged_battery",        "bloated battery": "damaged_battery",
    "defekte batterie": "damaged_battery",      "aufgeblähte batterie": "damaged_battery",
    "aufgeblähter akku": "damaged_battery",     "beschädigter akku": "damaged_battery",
    "geschwollene batterie": "damaged_battery", "auslaufende batterie": "damaged_battery",
    "defekter akku": "damaged_battery",         "swollen akku": "damaged_battery",

    "battery": "hazardous_waste_(battery)",
    "batteries": "hazardous_waste_(battery)",

    # Items that are outside the trained recycling categories
    # are mapped to non_waste so the assistant does not invent disposal rules.
    "phone": "non_waste", "mobile": "non_waste",
    "laptop": "non_waste", "electronics": "non_waste",
    "clothes": "non_waste", "clothing": "non_waste",
    "fabric": "non_waste", "shoes": "non_waste",

    "food waste": "organic_waste", "compost": "organic_waste",
    "food scraps": "organic_waste", "vegetable scraps": "organic_waste",
    "fruit scraps": "organic_waste", "garden waste": "organic_waste",

    "shampoo bottle": "rigid_plastic_container",
    "detergent bottle": "rigid_plastic_container",
    "cleaning bottle": "rigid_plastic_container",

    "plastic": "plastic",
    "plastic bag": "plastic",
    "wrapper": "plastic",

    "residual waste": "residual_waste",
    "diaper": "residual_waste", "nappy": "residual_waste",

    # Receipts: differentiated by type (thermal/white vs. eco/blue).
    # Sources: Migros / 20min (blue eco-receipts recyclable, physical print process),
    # INGEDE / VKU (caution: pigments affect paper recycling quality; small amounts are fine).
    # White/classic receipts: residual waste. Blue/eco receipts: paper recycling possible, residual waste if in doubt.
    # Compound eco-receipt keys must appear before generic keys so the loop matches
    # the more specific category first (e.g. "blauen kassenzettel" before "kassenzettel").
    "blue receipt": "eco_receipt",         "blauer kassenzettel": "eco_receipt",
    "blauen kassenzettel": "eco_receipt",  "blauer kassenbon": "eco_receipt",
    "blauen kassenbon": "eco_receipt",     "blauer bon": "eco_receipt",
    "blauen bon": "eco_receipt",           "öko-bon": "eco_receipt",
    "ökobon": "eco_receipt",               "migros kassenzettel": "eco_receipt",
    "migros kassenbon": "eco_receipt",     "migros bon": "eco_receipt",
    "receipt": "thermal_receipt",          "thermal paper": "thermal_receipt",
    "thermal receipt": "thermal_receipt",  "white receipt": "thermal_receipt",
    "kassenzettel": "thermal_receipt",     "kassenbon": "thermal_receipt",
    "kassenbeleg": "thermal_receipt",      "quittung": "thermal_receipt",
    "thermopapier": "thermal_receipt",     "weisser kassenzettel": "thermal_receipt",
    "weissen kassenzettel": "thermal_receipt",

    # German
    "altpapier": "paper", "zeitung": "paper", "zeitschrift": "paper",
    "karton": "cardboard", "pappe": "cardboard",

    "plastikflasche": "pet",
    "pet flasche": "pet",
    "pet-flasche": "pet",

    "glasflasche": "white_glass",
    "weissglas": "white_glass",
    "weißglas": "white_glass",
    "braunglas": "brown_glass",
    "grünglas": "green_glass",

    "alu": "aluminium", "aludose": "aluminium", "dose": "aluminium",
    "nespressokapsel": "aluminium", "kaffekapsel": "aluminium",
    "alufolie": "aluminium", "joghurtdeckel": "aluminium",
    "getränkedose": "aluminium",

    "batterie": "hazardous_waste_(battery)",
    "batterien": "hazardous_waste_(battery)",

    # Not part of the trained waste classes
    "handy": "non_waste",
    "smartphone": "non_waste",
    "elektroschrott": "non_waste",
    "kleider": "non_waste",
    "textilien": "non_waste",
    "schuhe": "non_waste",

    "kompost": "organic_waste", "speisereste": "organic_waste",
    "lebensmittelreste": "organic_waste", "gartenabfall": "organic_waste",

    "milchkarton": "composite_carton",
    "getränkekarton": "composite_carton",
    "tetrapack": "composite_carton",

    "shampooflasche": "rigid_plastic_container",
    "shampooflaschen": "rigid_plastic_container",
    "reinigungsflasche": "rigid_plastic_container",

    "plastik": "plastic",
    "plastiksack": "plastic",
    "plastiktüte": "plastic",
    "verpackungsfolie": "plastic",

    "kehricht": "residual_waste",
    "restmüll": "residual_waste",
    "windel": "residual_waste",

    # Incandescent and halogen lamps: disposed of as residual waste (Kehricht).
    # NOTE: the image classifier has no lamp classes; these categories are text-path only.
    # Keys are long and compound, so there is no false-positive substring risk (e.g. "glühbirne" != "Birne"/fruit).
    "glühlampe": "incandescent_lamp",    "glühlampen": "incandescent_lamp",
    "glühbirne": "incandescent_lamp",    "glühbirnen": "incandescent_lamp",
    "glühbrine": "incandescent_lamp",    "glüehbire": "incandescent_lamp",
    "halogenlampe": "incandescent_lamp", "halogenlampen": "incandescent_lamp",
    "halogenbirne": "incandescent_lamp", "halogen": "incandescent_lamp",
    "light bulb": "incandescent_lamp",   "incandescent": "incandescent_lamp",
    "halogen lamp": "incandescent_lamp", "halogen bulb": "incandescent_lamp",

    # LED, CFL, and fluorescent lamps: returned to SENS collection points under VREG.
    # Longer/compound keys are listed before shorter substrings so the first-match loop hits
    # the more specific term first (e.g. "energiesparlampe" before "sparlampe").
    "led lampe": "lamp_special_disposal",              "led-lampe": "lamp_special_disposal",
    "ledlampe": "lamp_special_disposal",               "led birne": "lamp_special_disposal",
    "energiesparlampe": "lamp_special_disposal",       "sparlampe": "lamp_special_disposal",
    "kompaktleuchtstofflampe": "lamp_special_disposal",
    "leuchtstoffröhre": "lamp_special_disposal",       "leuchtstofflampe": "lamp_special_disposal",
    "neonröhre": "lamp_special_disposal",              "neonlampe": "lamp_special_disposal",
    "led lamp": "lamp_special_disposal",               "energy saving lamp": "lamp_special_disposal",
    "fluorescent tube": "lamp_special_disposal",       "fluorescent": "lamp_special_disposal",

    # Waste oil and cooking oil: separate waste-oil collection, not classified as hazardous waste.
    # Source: Swiss Recycle (swissrecycle.ch/de/wertstoffe-wissen/wertstoffe/oel), supplemented by BAFU.
    # Hazardous waste covers fuels, paints, solvents. Motor oil: collection points and garages. Never drain or residual waste.
    "motorenöl": "waste_oil",   "motoröl": "waste_oil",   "altöl": "waste_oil",
    "getriebeöl": "waste_oil",  "schmieröl": "waste_oil",
    "frittieröl": "waste_oil",  "speiseöl": "waste_oil",  "bratöl": "waste_oil",
    "öl": "waste_oil",          # short key: safe via word-boundary regex in knowledge_base_node
    "motor oil": "waste_oil",   "engine oil": "waste_oil",  "used oil": "waste_oil",
    "cooking oil": "waste_oil", "frying oil": "waste_oil",  "waste oil": "waste_oil",
    "oil": "waste_oil",         # short key: safe via word-boundary regex (won't match "foil", "coil")
}

LOCATION_KEYWORDS = [
    # English
    "where", "location", "collect", "collection point", "recycling centre",
    "recycling center", "near me", "nearby", "drop off", "drop-off",
    "bring", "take", "dispose near", "find", "closest",
    # German
    "wo", "sammelstelle", "entsorgung", "standort", "abgeben",
    "bringen", "nächste", "in der nähe", "sammelstellen", "entsorgungshof"
]


# ===========================================================================
# AGENT STATE
# ===========================================================================

class AgentState(TypedDict):
    """Shared state passed between all LangGraph nodes in the agent graph."""
    user_message: str
    image_path: Optional[str]
    city: Optional[str]
    language: str
    classification: Optional[dict]
    guidelines: Optional[str]
    guidelines_list: Optional[List]   # multi-item: [{"category": str, "guideline": str}, ...]
    collection_points: Optional[str]
    input_type: Optional[str]
    needs_clarification: bool
    final_response: Optional[str]
    osm_elements: Optional[List]
    map_lat: Optional[float]       # set by geolocation_node, used by dashboard
    map_lon: Optional[float]       # set by geolocation_node, used by dashboard
    scan_history: List[dict]
    conversation_history: List[dict]


# ===========================================================================
# GEOLOCATION
# Resolves Swiss city names and ZIP codes to coordinates via geo.admin.ch.
# Coordinates are stored in AgentState so the dashboard map never geocodes twice.
# ===========================================================================

def get_coordinates(city: str):
    """Resolve a Swiss city/town name or ZIP code to (lat, lon, municipality) via geo.admin.ch."""
    city = (city or "").strip()
    if not city:
        return None

    search_text  = city
    search_label = f"city='{city}'"
    origins      = "gg25,zipcode"

    # Switzerland bounding box (WGS84)
    CH_LAT_MIN, CH_LAT_MAX = 45.8, 47.8
    CH_LON_MIN, CH_LON_MAX = 5.9, 10.5

    try:
        response = requests.get(
            "https://api3.geo.admin.ch/rest/services/ech/SearchServer",
            params={
                "searchText": search_text,
                "type": "locations",
                "origins": origins,
                "sr": "4326",
                "lang": "de",
                "limit": "1",
            },
            headers={"User-Agent": "SwissRecyclingAssistant/1.0 (bachelor-thesis-demo)"},
            timeout=8,
        )
        data = response.json()
        results = data.get("results", [])
        if results:
            attrs = results[0]["attrs"]
            lat = float(attrs["lat"])
            lon = float(attrs["lon"])

            # Reject coordinates that fall outside Switzerland
            if not (CH_LAT_MIN <= lat <= CH_LAT_MAX and CH_LON_MIN <= lon <= CH_LON_MAX):
                print(f"[DEBUG] geo.admin.ch: {search_label} → coords ({lat:.4f}, {lon:.4f}) outside CH, rejecting")
                return None

            # label is like "<b>9000 St. Gallen</b>": strip HTML then leading ZIP
            raw_label = attrs.get("label", search_text)
            clean = re.sub(r"<[^>]+>", "", raw_label).strip()
            municipality = re.sub(r"^\d{4}\s+", "", clean).strip() or city
            print(f"[DEBUG] geo.admin.ch ({search_label}): → {municipality} ({lat:.4f}, {lon:.4f})")
            return lat, lon, municipality
    except Exception as e:
        print(f"[DEBUG] geo.admin.ch error: {e}")

    print(f"[DEBUG] Could not resolve location: {search_label}")
    return None


def get_osm_collection_points(lat: float, lon: float, radius: int = 2000):
    """Query the Overpass API for recycling and waste disposal nodes within the given radius (metres)."""
    query = f"""
    [out:json][timeout:15];
    (
      node["amenity"="recycling"](around:{radius},{lat},{lon});
      node["amenity"="waste_disposal"](around:{radius},{lat},{lon});
    );
    out body;
    """
    mirrors = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    for mirror in mirrors:
        try:
            response = requests.post(mirror, data={"data": query}, timeout=20)
            elements = response.json().get("elements", [])
            print(f"[DEBUG] Overpass ({mirror}): {len(elements)} results")
            return elements
        except Exception as e:
            print(f"[DEBUG] Overpass mirror {mirror} failed: {e}")
            continue
    return []


def format_collection_points(elements: list, municipality: str, lang: str) -> str:
    """Format raw OSM collection point elements into a readable text summary in the given language."""
    glass, pet, metal, centres = [], [], [], []

    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name", "")
        street = tags.get("addr:street", "")
        number = tags.get("addr:housenumber", "")
        address = f"{street} {number}".strip() if street else ""
        label = (name if name else "Collection point") + (f", {address}" if address else "")

        if tags.get("amenity") == "waste_disposal" or tags.get("recycling_type") == "centre":
            centres.append(label)
        elif tags.get("recycling:glass_bottles") == "yes" or tags.get("recycling:glass") == "yes":
            glass.append(label)
        elif tags.get("recycling:plastic_bottles") == "yes" or tags.get("recycling:PET") == "yes":
            pet.append(label)
        elif tags.get("recycling:scrap_metal") == "yes" or tags.get("recycling:metal") == "yes":
            metal.append(label)

    location_label = municipality

    if lang == "de":
        out = f"Sammelstellen in {location_label}, 2km Umkreis:\n"
        if centres: out += "\n🏭 Entsorgungshof:\n" + "".join(f"  • {s}\n" for s in centres[:2])
        if glass:   out += "\n🍾 Glascontainer:\n"  + "".join(f"  • {s}\n" for s in glass[:3])
        if pet:     out += "\n♻️ PET / Plastik:\n"  + "".join(f"  • {s}\n" for s in pet[:2])
        if metal:   out += "\n🔩 Metall / Alu:\n"   + "".join(f"  • {s}\n" for s in metal[:2])
        if not (centres or glass or pet or metal):
            out += "Keine OpenStreetMap-Einträge gefunden.\n"
        out += "\n• Batterien: Gratis Rückgabe bei jedem Händler der Batterien verkauft"
    else:
        out = f"Collection points in {location_label}, 2km radius:\n"
        if centres: out += "\n🏭 Recycling centre:\n" + "".join(f"  • {s}\n" for s in centres[:2])
        if glass:   out += "\n🍾 Glass containers:\n"  + "".join(f"  • {s}\n" for s in glass[:3])
        if pet:     out += "\n♻️ PET / Plastic:\n"     + "".join(f"  • {s}\n" for s in pet[:2])
        if metal:   out += "\n🔩 Metal / Aluminium:\n" + "".join(f"  • {s}\n" for s in metal[:2])
        if not (centres or glass or pet or metal):
            out += "No OpenStreetMap entries found.\n"
        out += "\n• Batteries: Free return at any retailer selling batteries"

    return out.strip()


# ===========================================================================
# NODES
# ===========================================================================

def perception_node(state: AgentState) -> AgentState:
    """Classify input as 'image', 'location', or 'text' based on the uploaded file and message content."""
    msg = state["user_message"].lower()
    image_path = state.get("image_path")

    if image_path and Path(image_path).exists():
        input_type = "image"
    elif any(re.search(r'\b' + re.escape(kw) + r'\b', msg, re.UNICODE) for kw in LOCATION_KEYWORDS):
        input_type = "location"
    else:
        input_type = "text"

    print(f"[DEBUG] input type: {input_type}")
    return {**state, "input_type": input_type}


def classifier_node(state: AgentState) -> AgentState:
    """Run the image classifier and update state with the result and scan history."""
    print(f"[DEBUG] classifying: {state['image_path']}")
    result = classifier.classify(state["image_path"])

    history = state.get("scan_history", []) + [{
        "category": result["category"],
        "confidence": round(result["confidence"], 3),
        "image": state["image_path"],
    }]
    return {**state, "classification": result,
            "needs_clarification": result["needs_clarification"],
            "scan_history": history}


_MAX_CATEGORIES = 4  # cap multi-item context to avoid overwhelming the LLM

def knowledge_base_node(state: AgentState) -> AgentState:
    """Look up disposal guidelines for the classified category or matched text terms."""
    lang = state.get("language", "en")
    classification = state.get("classification") or {}
    category_from_classifier = classification.get("category", "")

    if category_from_classifier:
        # Image path: single category from classifier, no multi-match needed.
        guide = RECYCLING_GUIDE.get(category_from_classifier, {})
        gl = guide.get(lang, guide.get("en", "No guidelines found."))
        guidelines_list = [{"category": category_from_classifier, "guideline": gl}]
    else:
        # Text path: collect ALL matching categories with position-based overlap detection.
        # More-specific/compound keys appear earlier in TEXT_CATEGORY_MAP and are matched
        # first; their character spans are recorded so sub-phrases (e.g. "kassenzettel"
        # inside "blauen kassenzettel") don't register as a separate category.
        msg = state["user_message"].lower()
        covered_spans: list = []  # (start, end) of already-claimed char ranges
        guidelines_list = []

        for term, mapped_category in TEXT_CATEGORY_MAP.items():
            if len(guidelines_list) >= _MAX_CATEGORIES:
                break
            if any(item["category"] == mapped_category for item in guidelines_list):
                continue  # category already collected

            pattern = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(pattern, msg, re.UNICODE):
                s, e = match.span()
                overlaps = any(
                    cs <= s < ce or cs < e <= ce or (s <= cs and e >= ce)
                    for cs, ce in covered_spans
                )
                if not overlaps:
                    guide = RECYCLING_GUIDE.get(mapped_category, {})
                    gl = guide.get(lang, guide.get("en", "No guidelines found."))
                    guidelines_list.append({"category": mapped_category, "guideline": gl})
                    covered_spans.append((s, e))
                    print(f"[DEBUG] text mapped '{term}' -> '{mapped_category}' (pos {s}-{e})")
                    break  # one valid occurrence of this term is enough

    single_guideline = guidelines_list[0]["guideline"] if guidelines_list else "No guidelines found."
    return {**state, "guidelines": single_guideline, "guidelines_list": guidelines_list}


def geolocation_node(state: AgentState) -> AgentState:
    """Geocode the city from state, fetch nearby OSM collection points, and store results and coordinates in state."""
    city = (state.get("city") or "").strip()
    lang = state.get("language", "en")

    if not city:
        if lang == "de":
            return {**state, "collection_points": None, "map_lat": None, "map_lon": None,
                    "final_response": "Bitte gib deinen **Ort** (z.B. Goldach) in der Sidebar ein – dann zeige ich dir die nächsten Sammelstellen!"}
        else:
            return {**state, "collection_points": None, "map_lat": None, "map_lon": None,
                    "final_response": "Please enter your **city/town** (e.g. Goldach) in the sidebar – then I'll show you the nearest collection points!"}

    coords = get_coordinates(city)
    if not coords:
        if lang == "de":
            msg = f"Ort '{city}' nicht gefunden. Bitte prüfe die Eingabe."
        else:
            msg = f"Location '{city}' not found. Please check your input."
        return {**state, "collection_points": msg, "map_lat": None, "map_lon": None}

    lat, lon, municipality = coords
    elements = get_osm_collection_points(lat, lon)

    if not elements:
        print(f"[DEBUG] No Overpass results - using fallback text")
        if lang == "de":
            result = (
                f"Sammelstellen in {municipality}:\n\n"
                f"• Batterien: Gratis Rückgabe bei jedem Händler der Batterien verkauft\n"
                f"• PET / Alu: COOP, Migros, Denner Filialen in Ihrer Gemeinde"
            )
        else:
            result = (
                f"Collection points in {municipality}:\n\n"
                f"• Batteries: Free return at any retailer selling batteries\n"
                f"• PET / Alu: COOP, Migros, Denner branches in your municipality"
            )
        return {**state, "collection_points": result,
                "map_lat": lat, "map_lon": lon}

    result = format_collection_points(elements, municipality, lang)
    return {**state, "collection_points": result, "map_lat": lat, "map_lon": lon}


def _strip_map_references(text: str) -> str:
    """Remove sentences containing map-reference phrases when no map was generated.
    Safety net: only called when collection_points is empty."""
    map_phrases = [
        r'schau\s+auf\s+die\s+karte',
        r'auf\s+der\s+karte\s+(?:oben|unten)',
        r'sieh(?:e)?\s+(?:dir\s+)?die\s+karte',
        r'check\s+the\s+map',
        r'on\s+the\s+map\s+(?:above|below)',
        r'see\s+the\s+map',
        r'refer\s+to\s+the\s+map',
        r'view\s+the\s+map',
    ]
    combined = '|'.join(f'(?:{p})' for p in map_phrases)
    # Remove the full sentence (up to . ! ? or line break) containing a map phrase
    sentence_re = re.compile(
        r'[^.!?\n]*(?:' + combined + r')[^.!?\n]*[.!?]?\s*',
        re.IGNORECASE,
    )
    result = sentence_re.sub('', text)
    result = re.sub(r'[ \t]+\n', '\n', result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def response_node(state: AgentState) -> AgentState:
    """Build the LLM prompt from context and guidelines, call GPT-4o, and store the response in state."""
    lang = state.get("language", "en")
    classification = state.get("classification")
    guidelines = state.get("guidelines", "")
    collection_points = state.get("collection_points", "")
    conv_history = state.get("conversation_history", [])
    has_map = bool(collection_points)

    # For a new location query, the map appears below the response in the chat.
    # For a follow-up text query, any map was already shown above in a previous message.
    is_new_location_query = state.get("input_type") == "location"

    if lang == "de":
        map_direction = "unten" if is_new_location_query else "oben"
        system = f"""Du bist ein Experte für Schweizer Abfallwirtschaft nach Swiss Recycle Richtlinien.

Regeln:
- SPRACHE (strikte Anforderung, keine Ausnahme): Antworte AUSSCHLIESSLICH auf Deutsch, unabhängig davon, in welcher Sprache der Nutzer schreibt. Selbst wenn die Eingabe englisch, französisch oder in einer anderen Sprache ist – die gesamte Antwort muss auf Deutsch sein. Das ist eine harte Vorgabe der Benutzeroberfläche.
- Beantworte nur Fragen zur Abfallentsorgung in der Schweiz
- Basiere Antworten NUR auf den bereitgestellten Richtlinien
- Halte Antworten kurz: maximal 2-3 Sätze
- KRITISCH: Wenn der Nutzer nach einer SPEZIFISCHEN Kategorie fragt (Glas, PET, Metall, Papier etc.), antworte NUR zu dieser Kategorie
- Schlage immer eine konkrete Aktion vor
- Verwende: Kehricht, Gemeinde, Entsorgungshof
- Füge KEINE Links hinzu
- Entferne ALLE recycling-map.ch Links aus deiner Antwort
- KARTE (verbindlich – prüfe MAP_AVAILABLE im Kontext): Verweise NUR dann auf die Karte ('Schau auf die Karte {map_direction}'), wenn im Kontext 'MAP_AVAILABLE: yes' steht. Wenn 'MAP_AVAILABLE: no' steht, erwähne KEINE Karte und sage NICHT 'Schau auf die Karte' – nenne stattdessen die Entsorgungsart allgemein (z.B. Rückgabe im Verkaufsgeschäft oder bei einer SENS-Sammelstelle)
- Wenn Sammelstellen und MAP_AVAILABLE: yes im Kontext stehen, bestätige den Ort und verweise auf die Karte – frage NIEMALS erneut nach dem Ort
- Sage NIEMALS dass es ein Problem mit der PLZ oder dem Ort gibt
- Wiederhole keine Informationen die bereits im Gespräch genannt wurden
- Wenn im Kontext MEHRERE Items mit eigenen Entsorgungsregeln aufgeführt sind: Beantworte JEDES Item separat mit seiner eigenen Regel. Übertrage die Regel eines Items NIEMALS auf ein anderes."""
    else:
        map_direction = "below" if is_new_location_query else "above"
        system = f"""You are a Swiss waste management expert following Swiss Recycle guidelines.

Rules:
- LANGUAGE (strict requirement, no exceptions): Answer EXCLUSIVELY in English, regardless of the language the user writes in. Even if the user's message or the waste item name is in German, French, or any other language, your entire response must be in English. This is a hard constraint set by the user interface.
- Only answer questions about waste disposal in Switzerland
- Base answers ONLY on the provided guidelines
- Keep answers concise: maximum 2-3 sentences
- CRITICAL: If the user asks about a SPECIFIC category (glass, PET, metal, paper etc.), ONLY answer about that category
- Always suggest one concrete action
- Use Swiss terms: residual waste, municipality, recycling centre
- Do NOT include any links
- Remove ALL recycling-map.ch links from your response
- MAP RULE (mandatory – check MAP_AVAILABLE in context): Refer to the map ('Check the map {map_direction}') ONLY if the context contains 'MAP_AVAILABLE: yes'. If 'MAP_AVAILABLE: no', do NOT mention any map and do NOT say 'Check the map' — describe the disposal method in general terms instead (e.g. return to retailer or SENS collection point)
- If collection points and MAP_AVAILABLE: yes are in the context, confirm the location and refer to the map – NEVER ask for location again
- NEVER say there is an issue with the ZIP code or location
- Do not repeat information already given in the conversation
- If the context lists MULTIPLE items with separate disposal rules: answer EACH item individually using only its own rule. Never apply one item's disposal rule to another item."""

    context = ""
    guidelines_list = state.get("guidelines_list") or []
    if classification:
        context += f"Detected category: {classification['category'].replace('_', ' ').title()}\n\n"
    if len(guidelines_list) > 1:
        if lang == "de":
            context += "Erkannte Items und Entsorgungsregeln (jedes Item separat beantworten):\n"
        else:
            context += "Detected items and disposal rules (answer each item separately):\n"
        for item in guidelines_list:
            cat_label = item["category"].replace("_", " ").title()
            context += f"- {cat_label}: {item['guideline']}\n\n"
    elif guidelines_list:
        context += f"Swiss Recycle guideline:\n{guidelines_list[0]['guideline']}\n\n"
    elif guidelines:
        context += f"Swiss Recycle guideline:\n{guidelines}\n\n"
    if collection_points:
        context += f"Collection points:\n{collection_points}\n\n"
    # Explicit signal so the LLM never needs to infer map availability from the topic
    if has_map:
        context += "MAP_AVAILABLE: yes – you may refer to the map.\n\n"
    else:
        context += "MAP_AVAILABLE: no – do NOT mention any map or say 'check the map'.\n\n"
    if state.get("needs_clarification") and classification:
        top3 = classification.get("top3_predictions", [])
        if len(top3) >= 2:
            context += f"Uncertain - alternative: {top3[1]['category'].replace('_', ' ')}. Ask a clarification question.\n"

    user_content = f"{context}User: {state['user_message']}" if context else state["user_message"]

    messages = [SystemMessage(content=system)]
    for turn in conv_history[-3:]:
        messages.append(HumanMessage(content=turn["user"]))
        messages.append(SystemMessage(content=turn["assistant"]))
    messages.append(HumanMessage(content=user_content))

    try:
        answer = llm.invoke(messages).content
    except Exception as e:
        if lang == "de":
            answer = f"Es ist ein Fehler aufgetreten. Bitte versuche es erneut. ({e})"
        else:
            answer = f"An error occurred. Please try again. ({e})"

    if not has_map:
        answer = _strip_map_references(answer)

    return {
        **state,
        "final_response": answer,
        "conversation_history": conv_history + [{"user": state["user_message"], "assistant": answer}],
        # Explicitly carry map coords forward so LangGraph doesn't drop them
        "map_lat": state.get("map_lat"),
        "map_lon": state.get("map_lon"),
    }


def clarification_node(state: AgentState) -> AgentState:
    """Return a clarifying question when the image classification confidence is too low."""
    lang = state.get("language", "en")
    classification = state.get("classification", {})
    category = classification.get("category", "")
    top3 = classification.get("top3_predictions", [])
    alt = top3[1]["category"] if len(top3) >= 2 else ""

    if lang == "de":
        if "glass" in category and "glass" in alt:           q = "Welche Farbe hat das Glas? (weiss, braun, grün)"
        elif "plastic" in category or "pet" in category:     q = "PET-Symbol vorhanden? Starr oder flexibel?"
        elif "paper" in category or "cardboard" in category: q = "Dünn (Papier) oder dick und steif (Karton)?"
        elif "aluminium" in category or "metal" in category: q = "Magnetisch? (magnetisch = Stahl, nicht = Aluminium)"
        else: q = "Können Sie Material oder Form genauer beschreiben?"
        answer = f"Das Bild ist nicht eindeutig.\n\nZur Klärung: {q}"
    else:
        if "glass" in category and "glass" in alt:           q = "What colour is the glass? (white, brown, green)"
        elif "plastic" in category or "pet" in category:     q = "PET symbol present? Is it rigid or flexible?"
        elif "paper" in category or "cardboard" in category: q = "Thin (paper) or thick and rigid (cardboard)?"
        elif "aluminium" in category or "metal" in category: q = "Is it magnetic? (magnetic = steel, non-magnetic = aluminium)"
        else: q = "Can you describe the material or shape in more detail?"
        answer = f"The image is not clear enough.\n\nTo clarify: {q}"

    conv_history = state.get("conversation_history", []) + [{"user": state["user_message"], "assistant": answer}]
    return {
        **state,
        "final_response": answer,
        "conversation_history": conv_history,
        "map_lat": state.get("map_lat"),
        "map_lon": state.get("map_lon"),
    }


# ===========================================================================
# ROUTING
# ===========================================================================

def route_after_perception(state: AgentState):
    """Route to classifier, geolocation, or knowledge_base based on the detected input type."""
    if state["input_type"] == "image":    return "classifier"
    if state["input_type"] == "location": return "geolocation"
    return "knowledge_base"

def route_after_classifier(state: AgentState):
    """Route to clarification if confidence is low, otherwise to knowledge_base."""
    return "clarification" if state.get("needs_clarification") else "knowledge_base"

def route_after_knowledge_base(state: AgentState):
    """Always continue to the response node."""
    return "response"

def route_after_geolocation(state: AgentState):
    """End early if geolocation failed with a pre-set response, otherwise continue to response node."""
    if state.get("final_response"):
        return END
    return "response"


# ===========================================================================
# BUILD AGENT
# ===========================================================================

def build_agent():
    """Build and compile the LangGraph agent with all nodes and conditional routing."""
    memory = MemorySaver()
    graph = StateGraph(AgentState)

    graph.add_node("perception",     perception_node)
    graph.add_node("classifier",     classifier_node)
    graph.add_node("knowledge_base", knowledge_base_node)
    graph.add_node("geolocation",    geolocation_node)
    graph.add_node("response",       response_node)
    graph.add_node("clarification",  clarification_node)

    graph.set_entry_point("perception")

    graph.add_conditional_edges("perception",     route_after_perception)
    graph.add_conditional_edges("classifier",     route_after_classifier)
    graph.add_conditional_edges("knowledge_base", route_after_knowledge_base)
    graph.add_conditional_edges("geolocation",    route_after_geolocation)
    graph.add_edge("response",      END)
    graph.add_edge("clarification", END)

    return graph.compile(checkpointer=memory)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    """Run the interactive command-line interface for testing the agent locally."""
    print("=" * 55)
    print("Swiss Waste Recycling Agent - Bachelor Thesis")
    print("=" * 55)

    agent = build_agent()

    language   = input("Language (en/de): ").strip().lower() or "en"
    city       = input("City/town (optional, Enter to skip): ").strip() or None
    session_id = input("Session ID (Enter for 'default'): ").strip() or "default"
    config = {"configurable": {"thread_id": session_id}}

    print("\nAgent ready. Type 'quit' to exit.")
    print("-" * 55)

    scan_history = []
    conv_history = []

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye! 🌱")
            break
        if not user_input:
            continue

        image_path = None
        cleaned = user_input.strip("\"'")
        if any(cleaned.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]):
            if Path(cleaned).exists():
                image_path = cleaned
                user_input = "How do I dispose of this?" if language == "en" else "Wie entsorge ich das?"
            else:
                print(f"File not found: {cleaned}")
                continue

        state = AgentState(
            user_message=user_input, image_path=image_path,
            city=city, language=language,
            classification=None, guidelines=None, collection_points=None,
            input_type=None, needs_clarification=False, final_response=None,
            osm_elements=None, map_lat=None, map_lon=None,
            scan_history=scan_history, conversation_history=conv_history,
        )

        result = agent.invoke(state, config=config)
        print(f"\nAgent: {result['final_response']}")
        scan_history = result.get("scan_history", scan_history)
        conv_history = result.get("conversation_history", conv_history)

        if scan_history:
            print(f"\n── Scan history ({len(scan_history)} item(s)) ──")
            for i, s in enumerate(scan_history, 1):
                print(f"  {i}. {s['category'].replace('_', ' ').title()}")


if __name__ == "__main__":
    main()
