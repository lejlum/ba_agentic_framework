#!/usr/bin/env python3
"""
Swiss Waste Recycling Agent - Bachelor Thesis
FIX: get_coordinates now returns coords for both ZIP codes AND city names.
     map_lat/map_lon are passed through AgentState so the dashboard
     never has to geocode a second time (which caused the wrong city on map).
"""

from pathlib import Path
from typing import TypedDict, Optional, List
from dotenv import load_dotenv
import requests
from huggingface_hub import hf_hub_download

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from chatbot.improved_swiss_waste_chatbot_opensource import WasteClassifier, RECYCLING_GUIDE

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

    "cardboard": "cardboard", "carton": "cardboard",
    "paper": "paper", "newspaper": "paper", "magazine": "paper",

    "plastic bottle": "pet", "pet bottle": "pet", "pet": "pet",

    "glass bottle": "white_glass", "wine bottle": "white_glass",
    "clear glass": "white_glass", "white glass": "white_glass",
    "beer bottle": "brown_glass",
    "brown glass": "brown_glass",
    "green bottle": "green_glass", "green glass": "green_glass",

    "aluminium": "aluminium", "aluminum": "aluminium",
    "tin can": "aluminium", "can": "aluminium",
    "nespresso": "aluminium", "coffee capsule": "aluminium", "capsule": "aluminium",
    "foil": "aluminium", "yogurt lid": "aluminium", "tin foil": "aluminium",
    "spray can": "aluminium", "beverage can": "aluminium",

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
    "receipt": "residual_waste", "thermal paper": "residual_waste",
    "diaper": "residual_waste", "nappy": "residual_waste",

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
    "sprühdose": "aluminium", "getränkedose": "aluminium",

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
    "kassenzettel": "residual_waste",
    "windel": "residual_waste",
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
    user_message: str
    image_path: Optional[str]
    zip_code: Optional[str]        # now accepts ZIP or city name
    language: str
    classification: Optional[dict]
    guidelines: Optional[str]
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
# FIX: accepts both ZIP codes and city/municipality names.
#      Returns (lat, lon, municipality) so the dashboard can use these
#      coordinates directly – no second geocoding in make_map_card.
# ===========================================================================

def get_coordinates(location_input: str):
    """
    Resolves a Swiss ZIP code OR city name to (lat, lon, municipality).
    Uses Nominatim with q="<input>, Switzerland" — this is the most reliable
    approach for both ZIP codes and city names and always returns WGS84.
    geo.admin.ch is skipped because its sr=4326 response omits lat/lon fields
    in practice, causing silent fallbacks to wrong coordinates.
    """
    location_input = location_input.strip()

    # Single strategy: Nominatim free-text with Switzerland constraint.
    # Using q= instead of postalcode= or city= avoids mismatches.
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": f"{location_input}, Switzerland",
                "format": "json",
                "limit": 1,
                "addressdetails": 1,
                "countrycodes": "ch",
            },
            headers={"User-Agent": "SwissRecyclingAssistant/1.0"},
            timeout=8,
        )
        results = response.json()
        if results:
            r = results[0]
            addr = r.get("address", {})
            municipality = (
                addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("municipality")
                or addr.get("county")
                or location_input
            )
            lat = float(r["lat"])
            lon = float(r["lon"])
            print(f"[DEBUG] Nominatim: '{location_input}' → {municipality} ({lat:.4f}, {lon:.4f})")
            return lat, lon, municipality
    except Exception as e:
        print(f"[DEBUG] Nominatim error: {e}")

    print(f"[DEBUG] Could not resolve location: {location_input}")
    return None


def get_osm_collection_points(lat: float, lon: float, radius: int = 2000):
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


def format_collection_points(elements: list, municipality: str, zip_code: str, lang: str) -> str:
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

    location_label = f"{municipality} ({zip_code})" if zip_code != municipality else municipality

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
    msg = state["user_message"].lower()
    image_path = state.get("image_path")

    if image_path and Path(image_path).exists():
        input_type = "image"
    elif any(w in msg for w in LOCATION_KEYWORDS):
        input_type = "location"
    else:
        input_type = "text"

    print(f"[DEBUG] input type: {input_type}")
    return {**state, "input_type": input_type}


def classifier_node(state: AgentState) -> AgentState:
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


def knowledge_base_node(state: AgentState) -> AgentState:
    lang = state.get("language", "en")
    classification = state.get("classification") or {}
    category = classification.get("category", "")

    if not category:
        msg = state["user_message"].lower()
        for term, mapped_category in TEXT_CATEGORY_MAP.items():
            if term in msg:
                category = mapped_category
                print(f"[DEBUG] text mapped '{term}' -> '{category}'")
                break

    guide = RECYCLING_GUIDE.get(category, {})
    guidelines = guide.get(lang, guide.get("en", "No guidelines found."))
    return {**state, "guidelines": guidelines}


def geolocation_node(state: AgentState) -> AgentState:
    """
    FIX: stores resolved lat/lon into state so the dashboard map_card
    can use them directly without geocoding again.
    """
    location_input = (state.get("zip_code") or "").strip()
    lang = state.get("language", "en")

    if not location_input:
        if lang == "de":
            return {**state, "collection_points": None, "map_lat": None, "map_lon": None,
                    "final_response": "Bitte gib deine **PLZ oder deinen Ort** in der Sidebar ein – dann zeige ich dir die nächsten Sammelstellen!"}
        else:
            return {**state, "collection_points": None, "map_lat": None, "map_lon": None,
                    "final_response": "Please enter your **ZIP code or city name** in the sidebar – then I'll show you the nearest collection points!"}

    coords = get_coordinates(location_input)
    if not coords:
        if lang == "de":
            msg = f"Ort '{location_input}' nicht gefunden. Bitte prüfe die Eingabe."
        else:
            msg = f"Location '{location_input}' not found. Please check your input."
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
        # still pass coords so the map renders at the right location
        return {**state, "collection_points": result,
                "map_lat": lat, "map_lon": lon}

    result = format_collection_points(elements, municipality, location_input, lang)
    return {**state, "collection_points": result, "map_lat": lat, "map_lon": lon}


def response_node(state: AgentState) -> AgentState:
    lang = state.get("language", "en")
    classification = state.get("classification")
    guidelines = state.get("guidelines", "")
    collection_points = state.get("collection_points", "")
    conv_history = state.get("conversation_history", [])

    # input_type == "location" → new map generated NOW → appears BELOW
    # input_type != "location" → follow-up → map was in previous message → ABOVE
    is_new_location_query = state.get("input_type") == "location"

    if lang == "de":
        map_direction = "unten" if is_new_location_query else "oben"
        system = f"""Du bist ein Experte für Schweizer Abfallwirtschaft nach Swiss Recycle Richtlinien.

Regeln:
- Beantworte nur Fragen zur Abfallentsorgung in der Schweiz
- Basiere Antworten NUR auf den bereitgestellten Richtlinien
- Halte Antworten kurz: maximal 2-3 Sätze
- KRITISCH: Wenn der Nutzer nach einer SPEZIFISCHEN Kategorie fragt (Glas, PET, Metall, Papier etc.), antworte NUR zu dieser Kategorie
- Schlage immer eine konkrete Aktion vor
- Verwende: Kehricht, Gemeinde, Entsorgungshof
- Füge KEINE Links hinzu – die Karte wird direkt im Chat angezeigt
- Entferne ALLE recycling-map.ch Links aus deiner Antwort
- Wenn Sammelstellen im Kontext vorhanden sind, verweise auf die Karte mit 'Schau auf die Karte {map_direction}' – beschreibe Standorte NIE im Text
- Wenn ein Ort und Sammelstellen im Kontext vorhanden sind, bestätige den Ort und verweise auf die Karte – frage NIEMALS erneut nach dem Ort
- Sage NIEMALS dass es ein Problem mit der PLZ oder dem Ort gibt
- Wiederhole keine Informationen die bereits im Gespräch genannt wurden
- Antworte in der Sprache in der der Nutzer schreibt"""
    else:
        map_direction = "below" if is_new_location_query else "above"
        system = f"""You are a Swiss waste management expert following Swiss Recycle guidelines.

Rules:
- Only answer questions about waste disposal in Switzerland
- Base answers ONLY on the provided guidelines
- Keep answers concise: maximum 2-3 sentences
- CRITICAL: If the user asks about a SPECIFIC category (glass, PET, metal, paper etc.), ONLY answer about that category
- Always suggest one concrete action
- Use Swiss terms: residual waste, municipality, recycling centre
- Do NOT include any links – the map is shown directly in the chat
- Remove ALL recycling-map.ch links from your response
- If collection points are shown in the context, refer to the map with 'Check the map {map_direction}' – never describe locations in text
- If a location and collection points are provided, confirm the location and refer to the map – NEVER ask for location again
- NEVER say there is an issue with the ZIP code or location
- Do not repeat information already given in the conversation
- Answer in the same language the user writes in"""

    context = ""
    if classification:
        context += f"Detected category: {classification['category'].replace('_', ' ').title()} ({classification['confidence']:.1%})\n\n"
    if guidelines:
        context += f"Swiss Recycle guideline:\n{guidelines}\n\n"
    if collection_points:
        context += f"Collection points:\n{collection_points}\n\n"
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

    return {
        **state,
        "final_response": answer,
        "conversation_history": conv_history + [{"user": state["user_message"], "assistant": answer}],
        # Explicitly carry map coords forward so LangGraph doesn't drop them
        "map_lat": state.get("map_lat"),
        "map_lon": state.get("map_lon"),
    }


def clarification_node(state: AgentState) -> AgentState:
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
    if state["input_type"] == "image":    return "classifier"
    if state["input_type"] == "location": return "geolocation"
    return "knowledge_base"

def route_after_classifier(state: AgentState):
    return "clarification" if state.get("needs_clarification") else "knowledge_base"

def route_after_knowledge_base(state: AgentState):
    return "geolocation" if state.get("zip_code") else "response"

def route_after_geolocation(state: AgentState):
    if state.get("final_response"):
        return END
    return "response"


# ===========================================================================
# BUILD AGENT
# ===========================================================================

def build_agent():
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
    print("=" * 55)
    print("Swiss Waste Recycling Agent - Bachelor Thesis")
    print("=" * 55)

    agent = build_agent()

    language   = input("Language (en/de): ").strip().lower() or "en"
    zip_code   = input("ZIP code or city (optional, Enter to skip): ").strip() or None
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
            zip_code=zip_code, language=language,
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
                print(f"  {i}. {s['category'].replace('_', ' ').title()} – {s['confidence']:.0%} confidence")


if __name__ == "__main__":
    main()
