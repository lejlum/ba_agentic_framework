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

MODEL_PATH = "./models/baseline/finetuned_model.pth"
if not Path(MODEL_PATH).exists():
    MODEL_PATH = "../models/baseline/finetuned_model.pth"

classifier = WasteClassifier(model_path=MODEL_PATH)


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
    """Queries OpenStreetMap for recycling facilities within the given radius."""
    query = f"""
    [out:json][timeout:10];
    (
      node["amenity"="recycling"](around:{radius},{lat},{lon});
      node["amenity"="waste_disposal"](around:{radius},{lat},{lon});
    );
    out body;
    """
    try:
        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=15
        )
        elements = response.json().get("elements", [])
        print(f"[DEBUG] Overpass: {len(elements)} results")
        return elements
    except Exception as e:
        print(f"[DEBUG] Overpass error: {e}")
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
            out += "Keine Einträge in OpenStreetMap für diesen Bereich.\n"
        out += "\n• Batterien: Gratis Rückgabe bei jedem Händler der Batterien verkauft"
    else:
        out = f"Collection points in {municipality} (ZIP {zip_code}, 2km radius):\n"
        if centres: out += "\n🏭 Recycling centre:\n" + "".join(f"  • {s}\n" for s in centres[:2])
        if glass:   out += "\n🍾 Glass containers:\n"  + "".join(f"  • {s}\n" for s in glass[:3])
        if pet:     out += "\n♻️ PET / Plastic:\n"     + "".join(f"  • {s}\n" for s in pet[:2])
        if metal:   out += "\n🔩 Metal / Aluminium:\n" + "".join(f"  • {s}\n" for s in metal[:2])
        if not (centres or glass or pet or metal):
            out += "No entries found in OpenStreetMap for this area.\n"
        out += "\n• Batteries: Free return at any retailer selling batteries"

    return out.strip()


# ===========================================================================
# NODES
# ===========================================================================

def perception_node(state: AgentState) -> AgentState:
    # detect input type: image path, location question, or plain text
    msg = state["user_message"].lower()
    image_path = state.get("image_path")

    if image_path and Path(image_path).exists():
        input_type = "image"
    elif any(w in msg for w in ["where", "wo", "sammelstelle", "collect", "entsorgung", "standort", "location"]):
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
    # fetch Swiss Recycle guidelines for the detected category (from PA2)
    lang = state.get("language", "en")
    classification = state.get("classification") or {}
    category = classification.get("category", "")
    guide = RECYCLING_GUIDE.get(category, {})
    guidelines = guide.get(lang, guide.get("en", "No guidelines found."))
    return {**state, "guidelines": guidelines}


def geolocation_node(state: AgentState) -> AgentState:
    # resolve ZIP code to coordinates, then query OpenStreetMap
    zip_code = state.get("zip_code", "")
    lang = state.get("language", "en")

    if not zip_code:
        return {**state, "collection_points": None}

    coords = get_coordinates(zip_code)
    if not coords:
        msg = f"ZIP {zip_code} not found. See https://www.swissrecycle.ch"
        return {**state, "collection_points": msg}

    lat, lon, municipality = coords
    elements = get_osm_collection_points(lat, lon)
    result = format_collection_points(elements, municipality, zip_code, lang)
    return {**state, "collection_points": result}


def response_node(state: AgentState) -> AgentState:
    # build context from all tool nodes and generate a GPT-4o response
    lang = state.get("language", "en")
    classification = state.get("classification")
    guidelines = state.get("guidelines", "")
    collection_points = state.get("collection_points", "")
    conv_history = state.get("conversation_history", [])

    if lang == "de":
        system = """Du bist ein Experte für Schweizer Abfallwirtschaft nach Swiss Recycle Richtlinien.
- Beantworte nur Fragen zur Abfallentsorgung in der Schweiz
- Basiere Antworten NUR auf den bereitgestellten Richtlinien
- Verwende: Kehricht, Gemeinde, Entsorgungshof
- Keine Links in der Antwort (wird automatisch hinzugefügt)"""
    else:
        system = """You are a Swiss waste management expert following Swiss Recycle guidelines.
- Only answer questions about waste disposal in Switzerland
- Base answers ONLY on the provided guidelines
- Use: residual waste, municipality, recycling centre
- Do not add links (appended automatically)"""

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
    for turn in conv_history[-2:]:
        messages.append(HumanMessage(content=turn["user"]))
        messages.append(SystemMessage(content=turn["assistant"]))
    messages.append(HumanMessage(content=user_content))

    answer = llm.invoke(messages).content

    if "swissrecycle.ch" not in answer.lower():
        answer += "\n\nMore information: https://www.swissrecycle.ch" if lang == "en" \
            else "\n\nWeitere Infos: https://www.swissrecycle.ch"

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
    graph.add_edge("geolocation",   "response")
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

        # detect image paths (strip quotes Windows sometimes adds)
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
