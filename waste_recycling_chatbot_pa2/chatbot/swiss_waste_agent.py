#!/usr/bin/env python3
"""
Swiss Waste Recycling Agent - Bachelor Thesis

Extends the PA2 prototype with a LangGraph agentic loop, GPT-4o,
session memory, and location-aware collection point lookup.

Requirements: pip install langgraph langchain-openai python-dotenv
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

# WasteClassifier and Swiss Recycle guidelines reused directly from PA2
from chatbot.improved_swiss_waste_chatbot_opensource import WasteClassifier, RECYCLING_GUIDE

load_dotenv()

# ===========================================================================
# SETUP
# ===========================================================================

# temperature 0.3 kept the same as PA2 to reduce hallucination risk
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# load model from Hugging Face Hub - works both locally and on deployment
# falls back to local path if HF download fails
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
# Maps common user terms to RECYCLING_GUIDE category keys.
# This improves text-based queries where no image is uploaded.
# ===========================================================================

TEXT_CATEGORY_MAP = {
    # English
    "tetrapack": "composite_carton", "tetra pak": "composite_carton",
    "juice box": "composite_carton", "milk carton": "composite_carton",
    "cardboard": "cardboard", "carton": "cardboard",
    "paper": "paper", "newspaper": "paper", "magazine": "paper",
    "plastic bottle": "pet_bottles", "pet bottle": "pet_bottles", "pet": "pet_bottles",
    "glass bottle": "white_glass", "wine bottle": "white_glass",
    "beer bottle": "brown_glass",
    "aluminium": "aluminium", "aluminum": "aluminium",
    "tin can": "aluminium", "can": "aluminium",
    "battery": "batteries", "batteries": "batteries",
    "phone": "electronic_waste", "mobile": "electronic_waste",
    "laptop": "electronic_waste", "electronics": "electronic_waste",
    "clothes": "textiles", "clothing": "textiles",
    "fabric": "textiles", "shoes": "textiles",
    "food waste": "organic_waste", "compost": "organic_waste",
    "nespresso": "aluminium",
    "coffee capsule": "aluminium",
    "capsule": "aluminium",
    "foil": "aluminium",
    "yogurt lid": "aluminium",
    "tin foil": "aluminium",
    "spray can": "aluminium",
    "beverage can": "aluminium",
    "juice carton": "composite_carton",
    "milk box": "composite_carton",
    "tetra brik": "composite_carton",
    "shampoo bottle": "rigid_plastic_container",
    "detergent bottle": "rigid_plastic_container",
    "cleaning bottle": "rigid_plastic_container",
    "food scraps": "organic_waste",
    "vegetable scraps": "organic_waste",
    "fruit scraps": "organic_waste",
    "garden waste": "organic_waste",
    "receipt": "residual_waste",
    "thermal paper": "residual_waste",
    "diaper": "residual_waste",
    "nappy": "residual_waste",
    # German
    "altpapier": "paper", "zeitung": "paper", "zeitschrift": "paper",
    "karton": "cardboard", "pappe": "cardboard",
    "plastikflasche": "pet_bottles",
    "glasflasche": "white_glass", "weissglas": "white_glass",
    "braunglas": "brown_glass", "grünglas": "green_glass",
    "alu": "aluminium", "aludose": "aluminium", "dose": "aluminium",
    "batterie": "batteries",
    "handy": "electronic_waste", "elektroschrott": "electronic_waste",
    "kleider": "textiles", "textilien": "textiles",
    "kompost": "organic_waste", "speisereste": "organic_waste",
    "nespressokapsel": "aluminium",
    "kaffekapsel": "aluminium",
    "alufolie": "aluminium",
    "joghurtdeckel": "aluminium",
    "sprühdose": "aluminium",
    "getränkedose": "aluminium",
    "milchkarton": "composite_carton",
    "tetrapack": "composite_carton",
    "shampooflaschen": "rigid_plastic_container",
    "lebensmittelreste": "organic_waste",
    "gartenabfall": "organic_waste",
    "kassenzettel": "residual_waste",
    "windel": "residual_waste",
}

# Keywords that trigger the geolocation node
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
    # user input
    user_message: str
    image_path: Optional[str]
    zip_code: Optional[str]
    language: str

    # filled by nodes during the reasoning loop
    classification: Optional[dict]
    guidelines: Optional[str]
    collection_points: Optional[str]
    input_type: Optional[str]
    needs_clarification: bool
    final_response: Optional[str]

    # persisted across conversation turns
    scan_history: List[dict]
    conversation_history: List[dict]


# ===========================================================================
# GEOLOCATION
# Uses geo.admin.ch (ZIP -> coordinates) + Overpass API (coordinates -> OSM
# collection points). Works for all Swiss ZIP codes without a static database.
# ===========================================================================

def get_coordinates(zip_code: str):
    """Returns (lat, lon, municipality) for a Swiss ZIP code, or None."""
    try:
        response = requests.get(
            "https://api3.geo.admin.ch/rest/services/api/SearchServer",
            params={"searchText": zip_code, "type": "locations", "origins": "zipcode"},
            timeout=5
        )
        results = response.json().get("results", [])
        if not results:
            print(f"[DEBUG] ZIP {zip_code} not found")
            return None
        attrs = results[0].get("attrs", {})
        municipality = attrs.get("label", zip_code).replace("<b>", "").replace("</b>", "")
        print(f"[DEBUG] ZIP {zip_code} -> {municipality}")
        return attrs.get("lat"), attrs.get("lon"), municipality
    except Exception as e:
        print(f"[DEBUG] geo.admin.ch error: {e}")
        return None


def get_osm_collection_points(lat: float, lon: float, radius: int = 2000):
    """
    Queries OpenStreetMap for recycling facilities within the given radius.
    Tries two Overpass API mirrors with retry logic.
    Returns empty list if both fail (caller handles fallback).
    """
    query = f"""
    [out:json][timeout:15];
    (
      node["amenity"="recycling"](around:{radius},{lat},{lon});
      node["amenity"="waste_disposal"](around:{radius},{lat},{lon});
    );
    out body;
    """
    # try two Overpass mirrors in case one is down
    mirrors = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter"
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
    """Groups OSM elements by waste type and returns a formatted string."""
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

    if lang == "de":
        out = f"Sammelstellen in {municipality} (PLZ {zip_code}, 2km Umkreis):\n"
        if centres: out += "\n🏭 Entsorgungshof:\n" + "".join(f"  • {s}\n" for s in centres[:2])
        if glass:   out += "\n🍾 Glascontainer:\n"  + "".join(f"  • {s}\n" for s in glass[:3])
        if pet:     out += "\n♻️ PET / Plastik:\n"  + "".join(f"  • {s}\n" for s in pet[:2])
        if metal:   out += "\n🔩 Metall / Alu:\n"   + "".join(f"  • {s}\n" for s in metal[:2])
        if not (centres or glass or pet or metal):
            out += f"Keine OpenStreetMap-Einträge gefunden.\n"
            out += f"👉 Alle Sammelstellen in {municipality}: https://recycling-map.ch/de/karte?zip={zip_code}\n"
        else:
            out += f"\n👉 Alle Sammelstellen anzeigen: https://recycling-map.ch/de/karte?zip={zip_code}"
        out += "\n• Batterien: Gratis Rückgabe bei jedem Händler der Batterien verkauft"
    else:
        out = f"Collection points in {municipality} (ZIP {zip_code}, 2km radius):\n"
        if centres: out += "\n🏭 Recycling centre:\n" + "".join(f"  • {s}\n" for s in centres[:2])
        if glass:   out += "\n🍾 Glass containers:\n"  + "".join(f"  • {s}\n" for s in glass[:3])
        if pet:     out += "\n♻️ PET / Plastic:\n"     + "".join(f"  • {s}\n" for s in pet[:2])
        if metal:   out += "\n🔩 Metal / Aluminium:\n" + "".join(f"  • {s}\n" for s in metal[:2])
        if not (centres or glass or pet or metal):
            out += f"No OpenStreetMap entries found.\n"
            out += f"👉 All collection points in {municipality}: https://recycling-map.ch/en/map?zip={zip_code}\n"
        else:
            out += f"\n👉 View all collection points: https://recycling-map.ch/en/map?zip={zip_code}"
        out += "\n• Batteries: Free return at any retailer selling batteries"

    return out.strip()


# ===========================================================================
# NODES
# ===========================================================================

def perception_node(state: AgentState) -> AgentState:
    # detect input type using expanded keyword list
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
    # run MobileNetV3 classifier from PA2 and append result to scan history
    print(f"[DEBUG] classifying: {state['image_path']}")
    result = classifier.classify(state["image_path"])

    history = state.get("scan_history", []) + [{
        "category": result["category"],
        "confidence": round(result["confidence"], 3),
        "image": state["image_path"]
    }]

    return {**state, "classification": result, "needs_clarification": result["needs_clarification"], "scan_history": history}


def knowledge_base_node(state: AgentState) -> AgentState:
    # fetch Swiss Recycle guidelines for the detected category
    # for text queries: try to map user terms to a known category first
    lang = state.get("language", "en")
    classification = state.get("classification") or {}
    category = classification.get("category", "")

    # if no image was classified, try text-based category mapping
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
    # resolve ZIP code to coordinates, then query OpenStreetMap
    # if Overpass fails completely, fall back to recycling-map.ch link
    zip_code = state.get("zip_code", "")
    lang = state.get("language", "en")

    if not zip_code:
        if lang == "de":
            return {**state, "collection_points": None, "final_response": 
                "Bitte gib deine **PLZ** in der Sidebar ein (Feld 'PLZ') – dann zeige ich dir die nächsten Sammelstellen in deiner Nähe!"}
        else:
            return {**state, "collection_points": None, "final_response": 
                "Please enter your **ZIP code** in the sidebar (field 'ZIP code') – then I'll show you the nearest collection points!"}

    coords = get_coordinates(zip_code)
    if not coords:
        if lang == "de":
            msg = f"PLZ {zip_code} nicht gefunden. Sammelstellen suchen: https://recycling-map.ch/de/karte?zip={zip_code}"
        else:
            msg = f"ZIP {zip_code} not found. Find collection points: https://recycling-map.ch/en/map?zip={zip_code}"
        return {**state, "collection_points": msg}

    lat, lon, municipality = coords
    elements = get_osm_collection_points(lat, lon)

    # if Overpass returned nothing at all (API down), use recycling-map.ch directly
    if elements is None or len(elements) == 0:
        print(f"[DEBUG] No Overpass results - using recycling-map.ch fallback")
        if lang == "de":
            result = (
                f"Sammelstellen in {municipality} (PLZ {zip_code}):\n\n"
                f"👉 Alle Sammelstellen anzeigen: https://recycling-map.ch/de/karte?zip={zip_code}\n\n"
                f"• Batterien: Gratis Rückgabe bei jedem Händler der Batterien verkauft\n"
                f"• PET / Alu: COOP, Migros, Denner Filialen in Ihrer Gemeinde"
            )
        else:
            result = (
                f"Collection points in {municipality} (ZIP {zip_code}):\n\n"
                f"👉 View all collection points: https://recycling-map.ch/en/map?zip={zip_code}\n\n"
                f"• Batteries: Free return at any retailer selling batteries\n"
                f"• PET / Alu: COOP, Migros, Denner branches in your municipality"
            )
        return {**state, "collection_points": result}

    result = format_collection_points(elements, municipality, zip_code, lang)
    return {**state, "collection_points": result}


def response_node(state: AgentState) -> AgentState:
    # build context from all tool nodes and generate a GPT-4o response
    lang = state.get("language", "en")
    classification = state.get("classification")
    guidelines = state.get("guidelines", "")
    collection_points = state.get("collection_points", "")
    conv_history = state.get("conversation_history", [])

    # improved system prompts with concise response instructions
    if lang == "de":
        system = """Du bist ein Experte für Schweizer Abfallwirtschaft nach Swiss Recycle Richtlinien.

Regeln:
- Beantworte nur Fragen zur Abfallentsorgung in der Schweiz
- Basiere Antworten NUR auf den bereitgestellten Richtlinien
- Halte Antworten kurz: maximal 4-5 Sätze
- Nenne immer die spezifische Abfallkategorie beim Namen
- Schlage immer eine konkrete Aktion vor
- Verwende: Kehricht, Gemeinde, Entsorgungshof
- Wenn keine Richtlinie vorhanden ist, empfehle die Gemeinde zu kontaktieren
- Keine zusätzlichen Links, aber recycling-map.ch Links aus dem Kontext beibehalten
- Wiederhole keine Informationen die bereits im Gespräch genannt wurden"""
    else:
        system = """You are a Swiss waste management expert following Swiss Recycle guidelines.

Rules:
- Only answer questions about waste disposal in Switzerland
- Base answers ONLY on the provided guidelines
- Keep answers concise: maximum 4-5 sentences
- Always name the specific waste category
- Always suggest one concrete action
- Use Swiss terms: residual waste, municipality, recycling centre
- If no guideline is available, recommend contacting the municipality
- Do not add extra links, but keep any recycling-map.ch links from the context
- Do not repeat information already given in the conversation"""

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

    # include last 3 turns for better multi-turn context (increased from 2)
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

    return {**state, "final_response": answer, "conversation_history": conv_history + [{"user": state["user_message"], "assistant": answer}]}


def clarification_node(state: AgentState) -> AgentState:
    # ask a targeted follow-up question when confidence < 0.6 (same logic as PA2)
    lang = state.get("language", "en")
    classification = state.get("classification", {})
    category = classification.get("category", "")
    top3 = classification.get("top3_predictions", [])
    alt = top3[1]["category"] if len(top3) >= 2 else ""

    if lang == "de":
        if "glass" in category and "glass" in alt:   q = "Welche Farbe hat das Glas? (weiss, braun, grün)"
        elif "plastic" in category or "pet" in category: q = "PET-Symbol vorhanden? Starr oder flexibel?"
        elif "paper" in category or "cardboard" in category: q = "Dünn (Papier) oder dick und steif (Karton)?"
        elif "aluminium" in category or "metal" in category: q = "Magnetisch? (magnetisch = Stahl, nicht = Aluminium)"
        else: q = "Können Sie Material oder Form genauer beschreiben?"
        answer = f"Das Bild ist nicht eindeutig.\n\nZur Klärung: {q}"
    else:
        if "glass" in category and "glass" in alt:   q = "What colour is the glass? (white, brown, green)"
        elif "plastic" in category or "pet" in category: q = "PET symbol present? Is it rigid or flexible?"
        elif "paper" in category or "cardboard" in category: q = "Thin (paper) or thick and rigid (cardboard)?"
        elif "aluminium" in category or "metal" in category: q = "Is it magnetic? (magnetic = steel, non-magnetic = aluminium)"
        else: q = "Can you describe the material or shape in more detail?"
        answer = f"The image is not clear enough.\n\nTo clarify: {q}"

    conv_history = state.get("conversation_history", []) + [{"user": state["user_message"], "assistant": answer}]
    return {**state, "final_response": answer, "conversation_history": conv_history}


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


# ===========================================================================
# BUILD AGENT
# ===========================================================================

def route_after_geolocation(state: AgentState):
    # if final_response already set (e.g. no ZIP code), skip GPT-4o
    if state.get("final_response"):
        return END
    return "response"

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

    language  = input("Language (en/de): ").strip().lower() or "en"
    zip_code  = input("ZIP code (optional, Enter to skip): ").strip() or None
    session_id = input("Session ID (Enter for 'default'): ").strip() or "default"
    config = {"configurable": {"thread_id": session_id}}

    print("\nAgent ready. Type 'quit' to exit.")
    print("For image analysis: enter the file path.")
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
            user_message=user_input,
            image_path=image_path,
            zip_code=zip_code,
            language=language,
            classification=None,
            guidelines=None,
            collection_points=None,
            input_type=None,
            needs_clarification=False,
            final_response=None,
            scan_history=scan_history,
            conversation_history=conv_history
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
