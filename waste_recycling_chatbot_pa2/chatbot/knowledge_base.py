#!/usr/bin/env python3
"""Knowledge base and classifier for the Swiss waste recycling chatbot.

Provides the RECYCLING_GUIDE disposal data (17 waste categories, aligned with
Swiss Recycle guidelines) and the WasteClassifier image model (MobileNetV3).

Authoritative source: https://swissrecycle.ch/de/wertstoffe-wissen/recycling-in-der-schweiz
"""

import logging
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for Swiss waste recycling chatbot"""
    # Complete waste categories (17 categories from dataset)
    WASTE_CATEGORIES = [
        'aluminium',
        'brown_glass',
        'cardboard',
        'composite_carton',
        'green_glass',
        'hazardous_waste_(battery)',
        'metal',
        'non_waste',
        'organic_waste',
        'paper',
        'pet',
        'plastic',
        'plastic_aluminium',
        'residual_waste',
        'rigid_plastic_container',
        'white_glass',
        'white_glass_metal'
    ]

# ============================================================================
# SWISS RECYCLING GUIDELINES, SWISS RECYCLE ALIGNED
# ============================================================================
# Based on Swiss Recycle: https://swissrecycle.ch/de/wertstoffe-wissen/recycling-in-der-schweiz
# Each category has explicit flags controlling which disposal channels to mention.

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
        "en": "Flatten all cardboard boxes before disposal. Many municipalities collect cardboard at the curb on scheduled collection days—consult your local waste collection calendar for specific dates. Alternatively, cardboard may be brought to recycling centers. Accepted items include shipping boxes, cereal boxes, egg cartons, and folded paper carrier bags (without plastic coating). Remove plastic tape, staples, and styrofoam packaging. Cardboard must be clean and dry. Verify your municipality's size restrictions.",
        "de": "Alle Kartonschachteln vor der Entsorgung flach zusammenlegen. Viele Gemeinden sammeln Karton an der Strasse an festgelegten Sammeltagen—konsultieren Sie Ihren lokalen Abfallkalender für spezifische Termine. Alternativ kann Karton zu Recyclingzentren gebracht werden. Angenommene Gegenstände umfassen Versandkartons, Müslischachteln, Eierkartons und gefaltete Papiertragtaschen (ohne Kunststoffbeschichtung). Plastikklebeband, Heftklammern und Styroporverpackungen entfernen. Karton muss sauber und trocken sein. Grössenbeschränkungen Ihrer Gemeinde prüfen."
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
    "non_waste": {
        "allow_curbside": False,
        "primary_channels": ["not_applicable"],
        "en": "The image does not appear to show a waste item. Please upload a clear photo of the item you want to dispose of, ideally with the object centered and visible.",
        "de": "Das Bild scheint keinen Abfallgegenstand zu zeigen. Bitte lade ein klares Foto des Gegenstands hoch, den du entsorgen möchtest, idealerweise zentriert und gut sichtbar."
    },
    "residual_waste": {
        "allow_curbside": True,
        "primary_channels": ["curbside_paid"],
        "en": "Residual waste is for non-recyclable items only. Dispose of items in official municipal waste bags (must be purchased) or authorized containers. Collection days vary by municipality—consult your local waste collection calendar. Accepted items include soiled items, non-recyclable plastics, broken ceramics, ashes, and hygiene products. Fees are charged via the bag/sticker system (polluter-pays principle).",
        "de": "Kehricht ist nur für nicht recycelbare Gegenstände vorgesehen. Gegenstände in offiziellen kommunalen Abfallsäcken (kostenpflichtig) oder autorisierten Containern entsorgen. Sammeltage variieren je nach Gemeinde—konsultieren Sie Ihren lokalen Abfallkalender. Angenommene Gegenstände umfassen verschmutzte Gegenstände, nicht recycelbare Kunststoffe, zerbrochene Keramik, Asche und Hygieneprodukte. Gebühren werden über das Sack-/Stickersystem erhoben (Verursacherprinzip)."
    },
    # Source: Swiss Recycle (swissrecycle.ch/wertstoffe/leuchtmittel),
    # SENS eRecycling (took over SLRS in 2021), legal basis: VREG.
    # Incandescent/halogen: residual waste. LED/CFL/fluorescent: return to retailers or SENS.
    "incandescent_lamp": {
        "allow_curbside": True,
        "primary_channels": ["curbside_paid"],
        "en": "Incandescent and halogen light bulbs belong in the residual waste (Kehricht/Hausmüll) — they are non-hazardous and the metal filament cannot be separated from the glass. Do not place in glass containers. Dispose of in the official waste bag, wrapped carefully to prevent breakage.",
        "de": "Glüh- und Halogenlampen gehören in den Kehricht (Hausmüll) – sie sind schadstofffrei und der Metallwendel lässt sich nicht vom Glas trennen. Nicht in den Glascontainer. Im kostenpflichtigen Kehrichtsack entsorgen, gut verpackt zum Schutz vor Glasbruch.",
    },
    "lamp_special_disposal": {
        "allow_curbside": False,
        "primary_channels": ["shop_takeback", "special_collection"],
        "en": "LED lamps, energy-saving lamps (compact fluorescent lamps), and fluorescent tubes must NOT be disposed of in residual waste. Return them free of charge to any retailer or a SENS eRecycling collection point (mandatory take-back under VREG). Energy-saving lamps contain mercury — do not break them. Do not place in glass containers.",
        "de": "LED-Lampen, Energiesparlampen (Kompaktleuchtstofflampen) und Leuchtstoffröhren dürfen NICHT in den Hausmüll. Kostenlose Rückgabe im Verkaufsgeschäft oder bei einer SENS-eRecycling-Sammelstelle (gesetzliche Rücknahmepflicht nach VREG). Energiesparlampen enthalten Quecksilber – nicht zerbrechen lassen. Nicht in den Glascontainer.",
    },
    # Source: Swiss Recycle (swissrecycle.ch/de/wertstoffe-wissen/wertstoffe/oel), supplemented by BAFU.
    # Waste oil uses separate collection, not classified as hazardous waste (hazardous = fuels/paints/solvents).
    # Motor oil: collection points and garages/retailers. Never in residual waste or drains.
    "waste_oil": {
        "allow_curbside": False,
        "primary_channels": ["recycling_center", "shop_takeback"],
        "en": "Used motor oil, gearbox oil, lubricating oil, and cooking/frying oil must be brought to designated waste-oil collection points (recycling centres / Entsorgungshof). Motor oil can additionally be returned free of charge at garages and retail outlets that sell oil (e.g. hardware stores), in typical household quantities. Do not dispose of in residual waste and never pour down the drain.",
        "de": "Altöl (Motoren-, Getriebe-, Schmieröl) sowie Speise- und Frittieröl gehören in die Altölsammlung an Sammelstellen und Entsorgungshöfen. Motorenöl kann zusätzlich bei Garagen und Verkaufsstellen (z.B. Fach-/Baumärkte, die Öle verkaufen) in haushaltsüblichen Mengen kostenlos abgegeben werden. Nicht in den Kehricht und niemals in die Kanalisation oder den Abfluss.",
    },
    # Source: INOBAT (inobat.ch), Swiss battery recycling organisation; BAFU guidelines for Li-ion
    # fire hazards. Swollen/leaking batteries require special handling distinct from normal
    # battery return. The image classifier cannot distinguish damaged from intact batteries,
    # so this entry is reached via the text path only.
    "damaged_battery": {
        "allow_curbside": False,
        "primary_channels": ["recycling_center", "special_collection"],
        "en": "WARNING: A swollen, leaking, or damaged lithium-ion battery is a fire hazard. Do NOT put it in regular battery collection boxes or residual waste. Keep it away from flammable materials. Store it cool and dry, ideally in a non-combustible container (e.g. a box with sand). Bring it to a recycling centre (Entsorgungshof) or a hazardous-waste collection point that accepts damaged batteries — call ahead to confirm. Do not puncture or crush the battery.",
        "de": "WARNUNG: Ein aufgeblähter, auslaufender oder beschädigter Lithium-Ionen-Akku ist brandgefährlich. NICHT in den normalen Batteriesammelbehälter im Laden und NICHT in den Hausmüll geben. Von brennbaren Materialien fernhalten. Kühl und trocken, möglichst in einem nicht brennbaren Behälter (z.B. mit Sand) lagern. Zur Abgabe an einen Entsorgungshof oder eine Sammelstelle bringen, die beschädigte Akkus / Gefahrgut annimmt – im Zweifel vorher anrufen. Akku nicht beschädigen oder durchstechen.",
    },
    # Sources: Migros / 20min (blue eco-receipts marketed as recyclable, physical print process),
    # INGEDE / VKU (caution: pigments can affect paper recycling quality; small amounts are fine).
    # White/classic receipts: residual waste. Blue/eco receipts: paper recycling possible, residual waste if in doubt.
    "thermal_receipt": {
        "allow_curbside": False,
        "primary_channels": ["residual_waste"],
        "en": "Classic (white) till receipts are typically printed on thermal paper and must be disposed of in residual waste (Kehricht) — not in paper recycling. The thermal coating (which may contain bisphenol) interferes with the paper recycling process.",
        "de": "Klassische (weisse) Kassenzettel bestehen meist aus Thermopapier und gehören in den Kehricht – nicht ins Altpapier. Die Thermobeschichtung (mögliche Bisphenol-Rückstände) stört das Papierrecycling.",
    },
    "eco_receipt": {
        "allow_curbside": False,
        "primary_channels": ["paper_recycling", "residual_waste"],
        "en": "Blue eco-receipts (e.g. Migros) are phenol-free and marketed by the retailer as suitable for paper recycling (physical rather than chemical print process). Note: recycling associations (INGEDE/VKU) advise caution in large quantities as the black pigments can affect paper recycling quality; small amounts in paper recycling are unproblematic. If in doubt, dispose of in residual waste (Kehricht) as the safe option.",
        "de": "Blaue Öko-Bons (z.B. Migros) sind phenolfrei und werden vom Händler als altpapier-recyclingfähig beworben (physikalischer statt chemischer Druckprozess). Hinweis: Recyclingverbände (INGEDE/VKU) raten bei grossen Mengen zur Vorsicht – Farbpigmente können das Papierrecycling stören; einzelne Bons im Altpapier sind unproblematisch. Im Zweifel: Kehricht als sichere Variante.",
    },
    # Source: Swiss Recycle (swissrecycle.ch/de/wertstoffe-wissen/wertstoffe/sonderabfall).
    # All aerosol cans are classified as hazardous waste due to propellant residues (explosion hazard when compacted).
    # Cantonal rules vary slightly (e.g. empty nitrous oxide cans), but the umbrella body guideline is hazardous waste.
    "aerosol_can": {
        "allow_curbside": False,
        "primary_channels": ["special_collection", "shop_takeback"],
        "en": "All aerosol cans (hairspray, deodorant, shaving foam, whipped cream, and other spray cans) must be disposed of as hazardous waste (Sonderabfall) — return them to a staffed collection point, a mobile hazardous-waste collection day, or a retail outlet with take-back. Do NOT put aerosol cans in the aluminium/metal collection or in residual waste: pressurised containers with propellant residues are an explosion hazard.",
        "de": "Alle Spraydosen (Haarspray, Deo, Rasierschaum, Schlagrahm/Sprührahm und andere Druckdosen) gehören in den Sonderabfall – Abgabe bei bedienter Sammelstelle, Sonderabfall-Mobil oder Verkaufsstelle mit Rücknahme. NICHT in die Alu-/Metallsammlung und NICHT in den Kehricht: Druckbehälter mit Treibgasresten sind explosionsgefährlich.",
    },
}

# ============================================================================
# CONFIDENCE THRESHOLDS
# ============================================================================

class ConfidenceLevel:
    """Confidence thresholds for classification"""
    HIGH = 0.70
    MEDIUM = 0.35
    LOW = 0.20

# ============================================================================
# IMAGE CLASSIFIER
# ============================================================================

# MobileNetV3-based image classifier for 17 waste categories
class WasteClassifier:
    """Waste image classifier using MobileNetV3"""

    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categories = Config.WASTE_CATEGORIES

        # Standard ImageNet normalization for transfer learning compatibility
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.model = self._create_model(model_path)
        logger.info(f"Classifier loaded on {self.device}")

    def _create_model(self, model_path: str):
        """Create and load the classification model"""
        model = models.mobilenet_v3_large(weights=None)

        # Architecture must match finetuned_model.pth exactly: any difference
        # causes load_state_dict to raise RuntimeError and falls back to random weights.
        model.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(0.5),
            nn.Linear(1280, 512),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(512, len(self.categories))
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

            logger.info(f"Top 3 predictions: {top3_predictions}")

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
