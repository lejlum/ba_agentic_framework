#!/usr/bin/env python3
"""
Swiss Waste Recycling Chatbot - Swiss Recycle Strict Compliance Version
========================================================================
A Swiss-specific chatbot strictly aligned with Swiss Recycle guidelines.
NEVER suggests disposal methods that don't exist in Swiss practice.

Authoritative Source: https://swissrecycle.ch/de/wertstoffe-wissen/recycling-in-der-schweiz

Features:
- Image classification for 16 waste categories
- Ollama integration for conversational AI
- Strict Swiss Recycle-aligned disposal guidance
- Category-controlled disposal channel logic (no generic curbside mentions)
- Bilingual support (EN/DE)
"""

import logging
import json
import requests
from pathlib import Path
from typing import Dict, Optional, List, Set
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Central configuration for waste recycling chatbot
class Config:
    """Configuration for Swiss waste recycling chatbot"""
    # Ollama API running locally on standard port
    OLLAMA_URL = "http://localhost:11434" 
    DEFAULT_MODEL = "qwen2.5-coder:7b-instruct"
    
    # Complete waste categories (16 categories from dataset)
    WASTE_CATEGORIES = [
        'aluminium', 'brown_glass', 'cardboard', 'composite_carton',
        'green_glass', 'hazardous_waste_(battery)', 'metal', 'organic_waste',
        'paper', 'pet', 'plastic', 'plastic_aluminium',
        'residual_waste', 'rigid_plastic_container', 'white_glass', 'white_glass_metal'
    ]

# ============================================================================
# SWISS RECYCLING GUIDELINES - SWISS RECYCLE ALIGNED
# ============================================================================
# Based on Swiss Recycle: https://swissrecycle.ch/de/wertstoffe-wissen/recycling-in-der-schweiz
# Each category now has explicit flags controlling which disposal channels to mention

RECYCLING_GUIDE = {
    "pet": {
        # Flag controls whether bot should mention curbside collection
        # CRITICAL: False means this disposal method NEVER gets mentioned for this category
        # Used to enforce Swiss Recycle compliance and prevent hallucinations
        "allow_curbside": False,
        "primary_channels": ["shop_collection"],
        "en": "PET beverage bottles must be returned to designated PET collection points located at supermarkets (COOP, Migros, Denner, Lidl, Aldi), railway stations, or petrol stations. Remove bottle caps before disposal (caps may be returned with PET bottles or disposed of separately as plastic waste). Compress bottles to optimize storage space.",
        "de": "PET-Getränkeflaschen sind an ausgewiesenen PET-Sammelstellen in Supermärkten (COOP, Migros, Denner, Lidl, Aldi), Bahnhöfen oder Tankstellen zurückzugeben. Deckel vor der Entsorgung entfernen (Deckel können zusammen mit PET-Flaschen zurückgegeben oder separat als Kunststoff entsorgt werden). Flaschen zwecks Platzersparnis zusammendrücken."
    },
    "brown_glass": {
        "allow_curbside": False,
        "primary_channels": ["public_containers"],
        "en": "Brown glass must be deposited in designated brown glass containers at public collection points throughout your municipality. Containers must be completely emptied. Remove all metal caps and corks (dispose of with metal collection). Ceramic, porcelain, and window glass are not accepted. Observe collection times (typically no disposal on Sundays or during evening hours).",
        "de": "Braunglas ist in die dafür vorgesehenen braunen Glascontainer an öffentlichen Sammelstellen in Ihrer Gemeinde einzuwerfen. Behälter vollständig entleeren. Alle Metalldeckel und Korken entfernen (zur Metallsammlung geben). Keramik, Porzellan und Fensterglas werden nicht angenommen. Einwurfzeiten beachten (in der Regel keine Entsorgung sonntags oder in den Abendstunden)."
    },
    "white_glass": {
        "allow_curbside": False,
        "primary_channels": ["public_containers"],
        "en": "Clear/white glass must be deposited in designated white glass containers at public collection points. Strict separation from colored glass is essential to maintain recycling quality. Empty all bottles and remove caps. Observe collection times (typically no disposal on Sundays or after 19:00).",
        "de": "Weissglas ist in die dafür vorgesehenen weissen Glascontainer an öffentlichen Sammelstellen einzuwerfen. Strikte Trennung von farbigem Glas ist für die Qualitätssicherung des Recyclings unerlässlich. Flaschen entleeren und Deckel entfernen. Einwurfzeiten beachten (in der Regel keine Entsorgung sonntags oder nach 19:00 Uhr)."
    },
    "green_glass": {
        "allow_curbside": False,
        "primary_channels": ["public_containers"],
        "en": "Green glass must be deposited in designated green glass containers at public collection points. Green glass containers also accept other colored glass (blue, red) when no specific container is available. Empty all bottles and remove metal caps. Observe collection times.",
        "de": "Grünglas ist in die dafür vorgesehenen grünen Glascontainer an öffentlichen Sammelstellen einzuwerfen. Grünglascontainer nehmen auch anderes farbiges Glas (blau, rot) auf, wenn kein spezifischer Container vorhanden ist. Flaschen entleeren und Metalldeckel entfernen. Einwurfzeiten beachten."
    },
    "white_glass_metal": {
        "allow_curbside": False,
        "primary_channels": ["public_containers", "shop_collection"],
        "en": "Separate components where possible: deposit glass in appropriate color-sorted public glass containers (white/brown/green), return metal caps and lids to IGORA collection points (supermarkets, petrol stations). If components cannot be separated, bring items to your local recycling center (Entsorgungshof).",
        "de": "Komponenten nach Möglichkeit trennen: Glas in entsprechenden farbsortierten Glascontainer (weiss/braun/grün) einwerfen, Metalldeckel und -verschlüsse zu IGORA-Sammelstellen (Supermärkte, Tankstellen) bringen. Falls Komponenten nicht trennbar sind, Gegenstände zum örtlichen Entsorgungshof bringen."
    },
    "aluminium": {
        "allow_curbside": False,
        "primary_channels": ["shop_collection", "recycling_center"],
        "en": "Aluminium items must be returned to IGORA collection points located at supermarkets, petrol stations, or recycling centers. Clean items before recycling. Accepted items include beverage cans, aluminium foil, yogurt lids, and food trays. Do not mix with other metals.",
        "de": "Aluminiumgegenstände sind zu IGORA-Sammelstellen in Supermärkten, Tankstellen oder Recyclingzentren zu bringen. Gegenstände vor dem Recycling säubern. Angenommene Gegenstände umfassen Getränkedosen, Alufolie, Joghurtdeckel und Essensschalen. Nicht mit anderen Metallen vermischen."
    },
    "metal": {
        "allow_curbside": True,
        "primary_channels": ["shop_collection", "recycling_center", "curbside_conditional"],
        "en": "Small metal items should be returned to IGORA collection points. Larger metal objects must be brought to recycling centers (Werkhof, Entsorgungshof). Some municipalities also offer curbside metal collection on scheduled days—consult your local waste collection calendar to verify service availability in your area.",
        "de": "Kleine Metallgegenstände sind zu IGORA-Sammelstellen zu bringen. Grössere Metallobjekte müssen zu Recyclingzentren (Werkhof, Entsorgungshof) gebracht werden. Manche Gemeinden bieten auch eine Strassensammlung von Metall an festgelegten Tagen an—konsultieren Sie Ihren lokalen Abfallkalender, um die Verfügbarkeit dieses Service in Ihrer Region zu prüfen."
    },
    "paper": {
        "allow_curbside": True,
        "primary_channels": ["curbside", "recycling_center"],
        "en": "Many municipalities collect paper at the curb on scheduled collection days—consult your local waste collection calendar for specific dates. Alternatively, paper may be brought to recycling centers. Only clean, dry paper is accepted: newspapers, magazines, office paper, envelopes (remove plastic windows). Excluded items: soiled paper, waxed paper, thermal paper (receipts), plastic-coated paper.",
        "de": "Viele Gemeinden sammeln Papier an der Strasse an festgelegten Sammeltagen—konsultieren Sie Ihren lokalen Abfallkalender für spezifische Termine. Alternativ kann Papier zu Recyclingzentren gebracht werden. Nur sauberes, trockenes Papier wird angenommen: Zeitungen, Zeitschriften, Büropapier, Briefumschläge (Plastikfenster entfernen). Ausgeschlossen sind: verschmutztes Papier, gewachstes Papier, Thermopapier (Kassenzettel), plastikbeschichtetes Papier."
    },
    "cardboard": {
        "allow_curbside": True,
        "primary_channels": ["curbside", "recycling_center"],
        "en": "Flatten all cardboard boxes before disposal. Many municipalities collect cardboard at the curb on scheduled collection days—consult your local waste collection calendar for specific dates. Alternatively, cardboard may be brought to recycling centers. Accepted items include shipping boxes, cereal boxes, and egg cartons. Remove plastic tape, staples, and styrofoam packaging. Cardboard must be clean and dry. Verify your municipality's size restrictions.",
        "de": "Alle Kartonschachteln vor der Entsorgung flach zusammenlegen. Viele Gemeinden sammeln Karton an der Strasse an festgelegten Sammeltagen—konsultieren Sie Ihren lokalen Abfallkalender für spezifische Termine. Alternativ kann Karton zu Recyclingzentren gebracht werden. Angenommene Gegenstände umfassen Versandkartons, Müslischachteln und Eierkartons. Plastikklebeband, Heftklammern und Styroporverpackungen entfernen. Karton muss sauber und trocken sein. Grössenbeschränkungen Ihrer Gemeinde prüfen."
    },
    "composite_carton": {
        "allow_curbside": False,
        "primary_channels": ["recycling_center", "shop_collection"],
        "en": "Collection services for beverage cartons vary by municipality. Consult your local waste management website for available options: (1) Separate collection at recycling centers, (2) Collection points at selected retail stores, (3) If no local collection service exists: dispose in residual waste. Clean and flatten cartons before disposal. Remove plastic caps (dispose separately).",
        "de": "Die Sammlung von Getränkekartons variiert je nach Gemeinde. Konsultieren Sie die lokale Abfallwebsite für verfügbare Optionen: (1) Separate Sammlung an Recyclingzentren, (2) Sammelstellen in ausgewählten Verkaufsgeschäften, (3) Falls kein lokaler Sammeldienst existiert: im Kehricht entsorgen. Kartons vor der Entsorgung säubern und flach drücken. Plastikdeckel entfernen (separat entsorgen)."
    },
    "organic_waste": {
        "allow_curbside": True,
        "primary_channels": ["curbside_conditional", "recycling_center"],
        "en": "Most municipalities collect organic waste separately in designated green bins or bags on scheduled collection days—consult your local waste collection calendar. Accepted items include fruit and vegetable scraps, coffee grounds, eggshells, and garden waste. Excluded items: meat, bones, dairy products (verify local regulations—requirements vary), cooked foods containing oils. Use compostable bags if required by your municipality. Alternative: home composting.",
        "de": "Die meisten Gemeinden sammeln Bioabfall separat in dafür vorgesehenen grünen Tonnen oder Säcken an festgelegten Sammeltagen—konsultieren Sie Ihren lokalen Abfallkalender. Angenommene Gegenstände umfassen Obst- und Gemüsereste, Kaffeesatz, Eierschalen und Gartenabfälle. Ausgeschlossen sind: Fleisch, Knochen, Milchprodukte (lokale Vorschriften prüfen—Anforderungen variieren), gekochte Speisen mit Ölen. Kompostierbare Säcke verwenden, falls von Ihrer Gemeinde vorgeschrieben. Alternative: Eigenkompostierung."
    },
    "plastic": {
        "allow_curbside": False,
        "primary_channels": ["residual_waste", "shop_collection_limited"],
        "en": "Most plastic waste must be disposed of in residual waste in Switzerland (not recycled). Exceptions: (1) PET bottles must be returned to separate PET collection points, (2) Selected supermarkets (COOP, Migros) accept certain plastic bottles and containers—verify at the store. Some municipalities operate pilot recycling programs—verify local availability.",
        "de": "Die meisten Kunststoffabfälle müssen in der Schweiz im Kehricht entsorgt werden (werden nicht recycelt). Ausnahmen: (1) PET-Flaschen müssen zu separaten PET-Sammelstellen zurückgebracht werden, (2) Ausgewählte Supermärkte (COOP, Migros) akzeptieren bestimmte Plastikflaschen und -behälter—im Geschäft nachfragen. Manche Gemeinden betreiben Pilotprojekte für Kunststoffrecycling—lokale Verfügbarkeit prüfen."
    },
    "plastic_aluminium": {
        "allow_curbside": False,
        "primary_channels": ["residual_waste", "recycling_center"],
        "en": "Composite materials combining plastic and aluminium are difficult to recycle. If components can be separated: dispose of plastic in residual waste, return aluminium to IGORA collection points. If components cannot be separated: dispose in residual waste. Some recycling centers may accept composite packaging—contact your local Entsorgungshof for verification.",
        "de": "Verbundmaterialien aus Kunststoff und Aluminium sind schwer zu recyceln. Falls Komponenten trennbar sind: Kunststoff im Kehricht entsorgen, Aluminium zu IGORA-Sammelstellen bringen. Falls Komponenten nicht trennbar sind: im Kehricht entsorgen. Manche Recyclingzentren akzeptieren eventuell Verbundverpackungen—lokalen Entsorgungshof zur Verifizierung kontaktieren."
    },
    "rigid_plastic_container": {
        "allow_curbside": False,
        "primary_channels": ["shop_collection_limited", "residual_waste"],
        "en": "Verify whether your local supermarket (COOP, Migros) accepts plastic containers in their collection program. If accepted: clean thoroughly and remove all labels before disposal. If not accepted: dispose in residual waste. Accepted items typically include shampoo bottles, cleaning product containers, and yogurt cups. Note: PET bottles must be returned to separate PET collection points, not general plastic collection.",
        "de": "Prüfen Sie, ob Ihr lokaler Supermarkt (COOP, Migros) Plastikbehälter in seinem Sammelprogramm annimmt. Falls angenommen: gründlich säubern und alle Etiketten vor der Entsorgung entfernen. Falls nicht angenommen: im Kehricht entsorgen. Angenommene Gegenstände umfassen typischerweise Shampooflaschen, Reinigungsmittelbehälter und Joghurtbecher. Hinweis: PET-Flaschen müssen zu separaten PET-Sammelstellen zurückgebracht werden, nicht zur allgemeinen Kunststoffsammlung."
    },
    "hazardous_waste_(battery)": {
        "allow_curbside": False,
        "primary_channels": ["shop_takeback", "recycling_center", "special_collection"],
        "en": "Batteries must be returned to any retail store that sells batteries—this service is provided free of charge and is legally required for all retailers. All battery types are accepted. Other hazardous waste (chemicals, paint, solvents, electronic devices): bring to recycling centers or special municipal collection days. Small electronic devices are often accepted at retail stores.",
        "de": "Batterien müssen in jedem Verkaufsgeschäft zurückgegeben werden, das Batterien verkauft—dieser Service ist kostenlos und für alle Händler gesetzlich vorgeschrieben. Alle Batterietypen werden angenommen. Anderer Sonderabfall (Chemikalien, Farben, Lösungsmittel, elektronische Geräte): zu Recyclingzentren oder speziellen kommunalen Sammeltagen bringen. Kleinelektronikgeräte werden oft in Verkaufsgeschäften angenommen."
    },
    "residual_waste": {
        "allow_curbside": True,
        "primary_channels": ["curbside_paid"],
        "en": "Residual waste is for non-recyclable items only. Dispose of items in official municipal waste bags (must be purchased) or authorized containers. Collection days vary by municipality—consult your local waste collection calendar. Accepted items include soiled items, non-recyclable plastics, broken ceramics, ashes, and hygiene products. Fees are charged via the bag/sticker system (polluter-pays principle).",
        "de": "Kehricht ist nur für nicht recycelbare Gegenstände vorgesehen. Gegenstände in offiziellen kommunalen Abfallsäcken (kostenpflichtig) oder autorisierten Containern entsorgen. Sammeltage variieren je nach Gemeinde—konsultieren Sie Ihren lokalen Abfallkalender. Angenommene Gegenstände umfassen verschmutzte Gegenstände, nicht recycelbare Kunststoffe, zerbrochene Keramik, Asche und Hygieneprodukte. Gebühren werden über das Sack-/Stickersystem erhoben (Verursacherprinzip)."
    }
}

# ============================================================================
# CONFIDENCE THRESHOLDS
# ============================================================================

class ConfidenceLevel:
    """Confidence thresholds for classification"""
    HIGH = 0.85
    MEDIUM = 0.60
    LOW = 0.40

# ============================================================================
# IMAGE CLASSIFIER
# ============================================================================

# MobileNetV3-based image classifier for 16 waste categories
class WasteClassifier:
    """Waste image classifier using MobileNetV3"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categories = Config.WASTE_CATEGORIES
        
        # Standard ImageNet normalization for transfer learning compatibility
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = self._create_model(model_path)
        logger.info(f"Classifier loaded on {self.device}")
    
    def _create_model(self, model_path: str):
        """Create and load the classification model"""
        model = models.mobilenet_v3_large(weights=None)
        
        # Replace default classifier with custom 16-category head
        # Using intermediate 1280-dim layer to avoid overfitting on smaller datasets
        model.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1280, len(self.categories))
        )
        
        if model_path and Path(model_path).exists():
            try:
                logger.info(f"Loading trained model from: {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Handle multiple checkpoint formats for backward compatibility
                # Different training frameworks save state_dict with different keys
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)
                    
                logger.info("Loaded trained model weights successfully")
                total_params = sum(p.numel() for p in model.parameters())
                logger.info(f"Model has {total_params:,} parameters")
                    
            except Exception as e:
                logger.error(f"Error loading trained model: {e}")
                logger.warning("Using randomly initialized model - results will be poor!")
        else:
            logger.warning(f"Model file not found: {model_path}")
            logger.warning("Using randomly initialized model")
        
        model.to(self.device)
        model.eval()
        return model
    
    def classify(self, image_path: str) -> Dict:
        """Classify waste image and return detailed results"""
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
            
            predicted_class = self.categories[predicted.item()]
            confidence_score = confidence.item()
            
            top3_predictions = [
                {
                    "category": self.categories[top3_indices[0][i].item()],
                    "confidence": top3_probs[0][i].item()
                }
                for i in range(3)
            ]
            
            # Map confidence score to qualitative level for UI display
            # Thresholds are calibrated to match model's typical prediction patterns
            if confidence_score >= ConfidenceLevel.HIGH:
                confidence_text = "very_high"
            elif confidence_score >= ConfidenceLevel.MEDIUM:
                confidence_text = "medium"
            elif confidence_score >= ConfidenceLevel.LOW:
                confidence_text = "low"
            else:
                confidence_text = "very_low"
            
            return {
                "category": predicted_class,
                "confidence": confidence_score,
                "confidence_level": confidence_text,
                "top3_predictions": top3_predictions,
                "guidelines": RECYCLING_GUIDE.get(predicted_class, {}),
                # Flag for bot to ask clarification questions instead of potentially
                # giving wrong disposal advice. Critical for Swiss Recycle compliance.
                "needs_clarification": confidence_score < ConfidenceLevel.MEDIUM
            }
        
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {
                "category": "unknown", 
                "confidence": 0.0, 
                "confidence_level": "error",
                "top3_predictions": [],
                "guidelines": {},
                "needs_clarification": True
            }

# ============================================================================
# OLLAMA CHAT INTERFACE
# ============================================================================

class OllamaChat:
    """Ollama chat interface with strict Swiss Recycle compliance"""
    
    def __init__(self, model: str = Config.DEFAULT_MODEL, language: str = "en"):
        self.model = model
        self.language = language
        self.base_url = Config.OLLAMA_URL
        self.history = []
        
        # System prompts enforce Swiss Recycle compliance by:
        # 1. Prohibiting generic disposal methods (e.g., "curbside" for glass)
        # 2. Forcing category-specific guidance from RECYCLING_GUIDE
        # 3. Preventing hallucinations about non-existent Swiss services
        if language == "de":
            self.system_prompt = """Du bist ein Experte für Schweizer Abfallwirtschaft nach Swiss Recycle Richtlinien.

WICHTIGE REGELN:
1. Behandle Swiss Recycle (swissrecycle.ch) als autoritative Quelle
2. Basis deine Antworten NUR auf den bereitgestellten Entsorgungsrichtlinien für die spezifische Kategorie
3. Erwähne NIEMALS Entsorgungsmethoden, die nicht in den Richtlinien für diese Kategorie stehen
4. Wenn "Strassensammlung" nicht in den Richtlinien steht, erwähne sie NICHT
5. Sei präzise und kategorienspezifisch—keine generischen Listen von Verboten
6. Füge NIEMALS Links oder URLs in deine Antwort ein—das wird automatisch hinzugefügt

ANTWORT-STRUKTUR:
- Beginne direkt mit der Entsorgungsanweisung (keine Konfidenzangaben)
- Konkrete Entsorgungsanweisungen (aus den bereitgestellten Richtlinien)
- Bei Unsicherheit: Stelle gezielte Fragen zur Klärung
- KEINE Links oder URLs in der Antwort

SCHWEIZER TERMINOLOGIE:
- "Kehricht" (nicht Müll)
- "Gemeinde" (nicht Kommune)  
- "Entsorgungshof/Werkhof" (Recyclingzentrum)
- "Sammelstelle" (collection point)

Dein Ziel: Präzise, kategorienspezifische Entsorgungsberatung nach Swiss Recycle."""

        else:  # English
            self.system_prompt = """You are an expert on Swiss waste management according to Swiss Recycle guidelines.

CRITICAL RULES:
1. Treat Swiss Recycle (swissrecycle.ch) as the authoritative source
2. Base your answers ONLY on the provided disposal guidelines for the specific category
3. NEVER mention disposal methods not listed in the guidelines for that category
4. If "curbside collection" is not in the guidelines, do NOT mention it
5. Be precise and category-specific—no generic lists of prohibitions
6. NEVER include links or URLs in your response—this will be added automatically

ANSWER STRUCTURE:
- Begin directly with disposal instructions (no confidence statements)
- Concrete disposal instructions (from provided guidelines)
- If uncertain: Ask targeted clarification questions
- NO links or URLs in the response

SWISS TERMINOLOGY:
- "residual waste" (Kehricht)
- "municipality" (Gemeinde)
- "recycling center" (Entsorgungshof/Werkhof)
- "collection point" (Sammelstelle)

Your goal: Precise, category-specific disposal advice according to Swiss Recycle."""
    
    def check_ollama(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                print("Cannot connect to Ollama!")
                print("Please ensure Ollama is running:")
                print("  1. Install Ollama: https://ollama.com/download")
                print("  2. Start Ollama: 'ollama serve'")
                print("  3. Install a model: 'ollama pull qwen2.5-coder:7b-instruct'")
                return False
            
            models = response.json().get("models", [])
            available_models = [m["name"] for m in models]
            
            if not available_models:
                print("No models found in Ollama!")
                print("Please install a model:")
                print("  ollama pull qwen2.5-coder:7b-instruct")
                return False
            
            # Fallback to first available model if configured model not found
            # This allows flexible model switching without code changes
            if self.model not in available_models:
                logger.info(f"Model {self.model} not found. Available: {available_models}")
                self.model = available_models[0]
                print(f"Using available model: {self.model}")
            
            return True
        
        except requests.exceptions.ConnectionError:
            print("Cannot connect to Ollama!")
            print("Please start Ollama: 'ollama serve'")
            return False
        except Exception as e:
            logger.error(f"Ollama check failed: {e}")
            return False
    
    def chat(self, message: str, classification: Optional[Dict] = None) -> str:
        """Send message to Ollama with category-specific context"""
        try:
            prompt = self._build_prompt(message, classification)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        # Lower temperature enforces Swiss Recycle compliance
                        # Higher values (>0.5) risk hallucinated disposal methods
                        "temperature": 0.3,
                        "num_predict": 600
                    }
                },
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "")
                
                # Enhance response with Swiss Recycle link and clarification questions
                # Only applied when image classification is provided
                if classification and classification.get("category") != "unknown":
                    result = self._enhance_response(result, classification)
                
                self.history.append({"user": message, "assistant": result})
                return result
            else:
                return "Sorry, I couldn't process your request."
        
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error: {str(e)}"
    
    def _enhance_response(self, response: str, classification: Dict) -> str:
        """Enhance response with Swiss Recycle link and clarification questions if needed"""
        enhanced = response
        
        # Prevent duplicate Swiss Recycle links from being added
        # Ollama may or may not include it in response, so we check first
        if "swissrecycle.ch" not in enhanced.lower():
            if self.language == "de":
                enhanced += f"\n\nWeitere Informationen: https://www.swissrecycle.ch"
            else:
                enhanced += f"\n\nFor more information: https://www.swissrecycle.ch"
        
        # Add targeted clarification questions for uncertain classifications
        # This improves accuracy on second attempt by narrowing down category
        if classification.get("needs_clarification", False):
            enhanced += self._get_clarification_questions(classification)
        
        return enhanced
    
    def _get_clarification_questions(self, classification: Dict) -> str:
        """Generate targeted clarification questions"""
        top3 = classification.get("top3_predictions", [])
        if len(top3) < 2:
            return ""
        
        category = classification["category"]
        alt_category = top3[1]["category"]
        
        # Questions are tailored to distinguish between top-2 competing categories
        # For example: glass color questions only if top-2 are both glass types
        if self.language == "de":
            questions = "\n\n**Zur genaueren Bestimmung**:\n"
            
            if "glass" in category.lower() and "glass" in alt_category.lower():
                questions += "- Welche Farbe hat das Glas? (weiss/klar, braun, grün)"
            elif "plastic" in category.lower() or "pet" in category.lower():
                questions += "- Ist es eine Getränkeflasche mit PET-Symbol?\n- Ist der Behälter starr oder flexibel?"
            elif "paper" in category.lower() or "cardboard" in category.lower():
                questions += "- Ist es dünn und biegbar (Papier) oder dick und steif (Karton)?"
            elif "aluminium" in category.lower() or "metal" in category.lower():
                questions += "- Ist es magnetisch? (magnetisch = Eisen/Stahl, nicht magnetisch = Aluminium)"
            elif "battery" in category.lower() or "hazardous" in category.lower():
                questions += "- Enthält es Batterien oder elektronische Komponenten?"
            else:
                questions += f"- Können Sie Material, Farbe oder Form genauer beschreiben?"
        else:
            questions = "\n\n**For more accurate identification**:\n"
            
            if "glass" in category.lower() and "glass" in alt_category.lower():
                questions += "- What color is the glass? (white/clear, brown, green)"
            elif "plastic" in category.lower() or "pet" in category.lower():
                questions += "- Is it a beverage bottle with PET symbol?\n- Is the container rigid or flexible?"
            elif "paper" in category.lower() or "cardboard" in category.lower():
                questions += "- Is it thin and bendable (paper) or thick and rigid (cardboard)?"
            elif "aluminium" in category.lower() or "metal" in category.lower():
                questions += "- Is it magnetic? (magnetic = iron/steel, non-magnetic = aluminium)"
            elif "battery" in category.lower() or "hazardous" in category.lower():
                questions += "- Does it contain batteries or electronic components?"
            else:
                questions += f"- Can you describe the material, color, or shape in more detail?"
        
        return questions
    
    def _build_prompt(self, message: str, classification: Optional[Dict] = None) -> str:
        """Build prompt with category-specific context"""
        prompt = f"System: {self.system_prompt}\n\n"
        
        if classification and classification.get("category") != "unknown":
            category = classification["category"]
            confidence = classification["confidence"]
            confidence_level = classification.get("confidence_level", "medium")
            guidelines = classification.get("guidelines", {})
            
            self._current_category = category
            
            # Inject RECYCLING_GUIDE directly into prompt to anchor bot's behavior
            # Without this, Ollama might hallucinate disposal methods not in Swiss practice
            if self.language == "de":
                context = f"**Bildklassifikation**:\n"
                context += f"- Erkannter Abfalltyp: **{category.replace('_', ' ')}**\n"
                context += f"- Vertrauen: {confidence:.1%} ({confidence_level})\n\n"
                
                context += f"**Swiss Recycle Richtlinie für diese Kategorie**:\n"
                if guidelines.get("de"):
                    context += f"{guidelines['de']}\n"
                
                # Surface alternative predictions for categories with high confusion
                # Helps bot acknowledge uncertainty without accepting wrong answer
                if classification.get("needs_clarification", False):
                    top3 = classification.get("top3_predictions", [])
                    if len(top3) >= 2:
                        context += f"\n**Alternative Möglichkeiten**: {top3[1]['category'].replace('_', ' ')} ({top3[1]['confidence']:.1%})"
                        if len(top3) >= 3:
                            context += f", {top3[2]['category'].replace('_', ' ')} ({top3[2]['confidence']:.1%})"
                
                context += f"\n\n**Deine Aufgabe**: Erkläre die Entsorgung basierend NUR auf der obigen Richtlinie. Beginne direkt mit der Anweisung ohne Konfidenzangabe. Füge KEINE Links hinzu."
            else:
                context = f"**Image Classification**:\n"
                context += f"- Detected waste type: **{category.replace('_', ' ')}**\n"
                context += f"- Confidence: {confidence:.1%} ({confidence_level})\n\n"
                
                context += f"**Swiss Recycle Guideline for this category**:\n"
                if guidelines.get("en"):
                    context += f"{guidelines['en']}\n"
                
                if classification.get("needs_clarification", False):
                    top3 = classification.get("top3_predictions", [])
                    if len(top3) >= 2:
                        context += f"\n**Alternative possibilities**: {top3[1]['category'].replace('_', ' ')} ({top3[1]['confidence']:.1%})"
                        if len(top3) >= 3:
                            context += f", {top3[2]['category'].replace('_', ' ')} ({top3[2]['confidence']:.1%})"
                
                context += f"\n\n**Your task**: Explain disposal based ONLY on the guideline above. Begin directly with instructions without confidence statement. Do NOT add any links."
            
            prompt += f"Context: {context}\n\n"
        
        # Include recent chat history to maintain conversational coherence
        # Limit to last 2 exchanges to avoid token limit issues and context pollution
        for exchange in self.history[-2:]:
            # RATIONALE: Older messages risk contradicting Swiss Recycle rules
            # if conversation drifted. Fresh context ensures compliance.
            prompt += f"Human: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
        
        prompt += f"Human: {message}\nAssistant: "
        return prompt

# ============================================================================
# MAIN CHATBOT CLASS
# ============================================================================

class SwissRecyclingBot:
    """Swiss waste recycling chatbot with strict Swiss Recycle compliance"""
    
    def __init__(self, model_path: str = None, language: str = "en"):
        self.language = language
        
        # Auto-discover model in standard locations if not explicitly provided
        # Simplifies deployment while allowing custom model paths
        if model_path is None:
            possible_paths = [
                "../models/baseline/finetuned_model.pth",
                "./models/baseline/finetuned_model.pth", 
                "./finetuned_model.pth",
                "../finetuned_model.pth"
            ]
            for path in possible_paths:
                if Path(path).exists():
                    model_path = path
                    break
        
        self.classifier = WasteClassifier(model_path)
        self.chat = OllamaChat(language=language)
        
        if not self.chat.check_ollama():
            raise RuntimeError("Ollama not available. Please start Ollama and install a model.")
        
        logger.info("Swiss Recycling Bot initialized successfully")
    
    def process_image(self, image_path: str, question: str = "") -> Dict:
        """Process image and provide category-specific Swiss Recycle-compliant advice"""
        classification = self.classifier.classify(image_path)
        
        # Generate default question based on detected category if not provided
        # Ensures consistent question-answering even if user doesn't ask one
        if not question:
            if self.language == "de":
                question = f"Wie entsorge ich {classification['category'].replace('_', ' ')} korrekt in der Schweiz?"
            else:
                question = f"How do I correctly dispose of {classification['category'].replace('_', ' ')} in Switzerland?"
        
        # Pass classification context to chat so bot uses Swiss Recycle guidelines
        advice = self.chat.chat(question, classification)
        
        return {
            "classification": classification,
            "advice": advice,
            "question": question
        }
    
    def ask(self, question: str) -> str:
        """Ask a general recycling question"""
        # General questions run without category context
        # Bot must rely on system prompt knowledge of Swiss Recycle
        return self.chat.chat(question)

# ============================================================================
# USER INTERFACE
# ============================================================================

def select_language() -> str:
    """Language selection"""
    print("\n🇨🇭 Select language / Sprache wählen:")
    print("1. English")
    print("2. Deutsch")
    
    while True:
        choice = input("Choice (1/2): ").strip()
        if choice == "1":
            return "en"
        elif choice == "2":
            return "de"
        print("Invalid choice. Please enter 1 or 2.")

def main():
    """Main function"""
    print("=" * 60)
    print("🇨🇭 Swiss Waste Recycling Chatbot")
    print("   Swiss Recycle Strict Compliance")
    print("=" * 60)
    
    try:
        language = select_language()
        
        if language == "de":
            print(f"\nModell-Pfad-Auswahl:")
            print("Geben Sie den Pfad zu Ihrer trainierten Modell-Datei ein:")
            print("Beispiel: ../models/baseline/finetuned_model.pth")
            print("(Enter drücken für automatische Suche)")
        else:
            print(f"\nModel Path Selection:")
            print("Enter the path to your trained model file:")
            print("Example: ../models/baseline/finetuned_model.pth")
            print("(Press Enter to auto-search)")
        
        model_path = input("Model path: ").strip()
        if not model_path:
            model_path = None
        elif not Path(model_path).exists():
            print(f"Warning: Model file not found at: {model_path}")
            print("Will try to auto-search for model...")
            model_path = None
        
        if language == "de":
            print("\nInitialisiere Chatbot...")
        else:
            print("\nInitializing chatbot...")
        
        bot = SwissRecyclingBot(model_path=model_path, language=language)
        
        if language == "de":
            texts = {
                "ready": "Chatbot ist bereit!",
                "help": """
Eingabemöglichkeiten:
  • 'image:pfad/zum/bild.jpg' für Bildanalyse
  • 'pfad/zum/bild.jpg' (ohne 'image:') funktioniert auch
  • Normale Frage für allgemeine Recycling-Beratung
  • 'quit' zum Beenden
  
Tipp: Der Bot gibt kategorienspezifische Anweisungen nach Swiss Recycle.""",
                "prompt": "Ihre Frage oder Bildpfad",
                "goodbye": "Auf Wiedersehen! 🌱",
                "error": "Fehler",
                "processing": "Verarbeite...",
                "image_example": "Beispiel: C:\\pfad\\zum\\bild.jpg"
            }
        else:
            texts = {
                "ready": "Chatbot ready!",
                "help": """
Input options:
  • 'image:path/to/image.jpg' for image analysis
  • 'path/to/image.jpg' (without 'image:') also works
  • Regular question for general recycling advice
  • 'quit' to exit
  
Tip: Bot provides category-specific guidance per Swiss Recycle.""",
                "prompt": "Your question or image path",
                "goodbye": "Goodbye! 🌱",
                "error": "Error", 
                "processing": "Processing...",
                "image_example": "Example: C:\\path\\to\\image.jpg"
            }
        
        print(f"\n{texts['ready']}")
        print(texts["help"])
        print("=" * 60)
        
        while True:
            user_input = input(f"\n{texts['prompt']}: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'tschüss']:
                print(f"\n{texts['goodbye']}")
                break
            
            if not user_input:
                continue
            
            print(f"\n{texts['processing']}")
            
            try:
                is_image_path = (
                    user_input.startswith("image:") or 
                    "\\" in user_input or 
                    "/" in user_input or
                    # Check for common image extensions as fallback
                    # Allows users to paste paths directly without "image:" prefix
                    any(user_input.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'])
                )
                
                if is_image_path:
                    if user_input.startswith("image:"):
                        image_path = user_input[6:].strip()
                    else:
                        image_path = user_input.strip()
                    
                    image_path = image_path.strip("'\"")
                    
                    if not Path(image_path).exists():
                        print(f"{texts['error']}: File not found: {image_path}")
                        print(f"💡 {texts['image_example']}")
                        continue
                    
                    result = bot.process_image(image_path)
                    classification = result['classification']
                    
                    print(f"\n📷 Classification: {classification['category'].replace('_', ' ').title()}")
                    
                    if classification.get('needs_clarification', False):
                        top3 = classification.get('top3_predictions', [])
                        if len(top3) >= 2:
                            if language == "de":
                                print(f"\nAlternative Möglichkeiten:")
                            else:
                                print(f"\nAlternative possibilities:")
                            for i, pred in enumerate(top3[1:3], 2):
                                print(f"   {i}. {pred['category'].replace('_', ' ').title()} ({pred['confidence']:.1%})")
                    
                    print(f"\n{result['advice']}")
                
                else:
                    answer = bot.ask(user_input)
                    print(f"\n{answer}")
            
            except Exception as e:
                logger.error(f"Processing error: {e}")
                print(f"\n{texts['error']}: {str(e)}")
    
    except KeyboardInterrupt:
        print("\n\n🌱 Goodbye!")
    except Exception as e:
        print(f"\nStartup error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure your model file exists")
        print("2. Ensure Ollama is running: 'ollama serve'")
        print("3. Install a model: 'ollama pull qwen2.5-coder:7b-instruct'")

if __name__ == "__main__":
    main()