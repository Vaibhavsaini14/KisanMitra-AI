"""
rag_engine.py — RAG Knowledge Base with FAISS
Stores Rajasthan crop advisory data as vector embeddings.
Falls back to keyword matching if sentence-transformers unavailable.
"""

import json
import numpy as np
from pathlib import Path

# ── Rajasthan Crop Knowledge Base ─────────────────────────────────────────────
CROP_KNOWLEDGE = [
    {
        "id": "bajra-downy-mildew",
        "crop": "Bajra",
        "problem_type": "disease",
        "keywords": "bajra pearl millet yellow grey powder downy mildew leaves stunted growth",
        "content": """
BAJRA — DOWNY MILDEW (Sclerospora graminicola)
Symptoms: Gray-white powder under leaves, yellow patches on upper surface, stunted/bushy growth, no panicle formation.
Cause: Fungal pathogen, spreads in humid conditions, soil-borne.
Treatment: Metalaxyl 35% + Mancozeb 64% WP at 2.5g per litre of water. Spray 2-3 times at 10-day intervals.
Seed Treatment: Metalaxyl 6g per kg seed before sowing to prevent next season.
Organic: Trichoderma viride 4g per kg seed treatment OR neem oil 5ml/litre foliar spray.
Varieties: HHB 67 Improved, Rajasthan Composite 1, Pusa 415 — all downy mildew resistant.
Source: ICAR-CAZRI Jodhpur Advisory Bulletin, Rajasthan Agriculture Dept. Kharif 2024
"""
    },
    {
        "id": "bajra-ergot",
        "crop": "Bajra",
        "problem_type": "disease",
        "keywords": "bajra pearl millet ergot sweet sour smell dark seeds honeydew sticky",
        "content": """
BAJRA — ERGOT (Claviceps fusiformis)
Symptoms: Pink/cream sticky honeydew secretion from florets, later hard dark sclerotia replace grain.
Cause: Fungal. Spreads via infected seed and humid conditions during flowering.
Treatment: NO curative spray. Remove and destroy infected panicles immediately. Do NOT feed ergot-infected grain to animals.
Seed Treatment: Soak seed in 10% salt solution (1kg salt in 10L water) for 10 min, remove floating seeds, dry and sow.
Prevention: Avoid late sowing (increases vulnerability), use disease-free certified seed.
Organic: Salt water seed treatment is the primary organic method.
Source: ICAR Research Station Jodhpur Bulletin
"""
    },
    {
        "id": "bajra-stem-borer",
        "crop": "Bajra",
        "problem_type": "pest",
        "keywords": "bajra millet stem borer dead heart shoot holes larvae caterpillar",
        "content": """
BAJRA — SHOOT FLY & STEM BORER
Shoot Fly (Atherigona approximata): Dead hearts in young plants (before knee-height).
Stem Borer (Chilo partellus): Holes in leaves, dead hearts in older plants, frass inside stem.
Treatment (Shoot Fly): Seed treatment with Imidacloprid 600 FS at 9ml per kg seed.
Treatment (Stem Borer): Carbofuran 3G granules at 8 kg per acre, apply in leaf whorls. OR Chlorpyrifos 20EC at 2ml per litre spray.
Organic: Beauveria bassiana 1L per acre spray in evening. Neem cake 200kg per acre at sowing.
Timing: Most critical period is 2-4 weeks after germination.
Source: ICAR All India Coordinated Research Project on Pearl Millet
"""
    },
    {
        "id": "mustard-aphid",
        "crop": "Mustard",
        "problem_type": "pest",
        "keywords": "mustard sarson aphid black small insects sucking leaves yellowing kide",
        "content": """
MUSTARD — MUSTARD APHID (Lipaphis erysimi)
Symptoms: Clusters of small black/greenish insects on leaves, buds, and pods. Leaves curl. Honey dew causes sooty mold. Severe yield loss up to 80% if unchecked.
Treatment: Imidacloprid 17.8 SL at 150ml per acre (most effective). OR Dimethoate 30 EC at 1ml per litre water.
Timing: Spray when 10-15 aphids per plant detected, usually January-February in Rajasthan.
Organic: Neem oil 5ml per litre water spray in evening. Yellow sticky traps for monitoring.
Threshold: Do not spray if natural enemies (ladybird beetles) are present in good numbers — they control aphids naturally.
Varieties: Pusa Bold, RH-30, Bio-902 show moderate aphid resistance.
Source: ICAR-DRMR Bharatpur Advisory, Rajasthan Agriculture Dept. Rabi Bulletin
"""
    },
    {
        "id": "mustard-white-rust",
        "crop": "Mustard",
        "problem_type": "disease",
        "keywords": "mustard white rust blisters white pustules leaves stalks stem",
        "content": """
MUSTARD — WHITE RUST (Albugo candida)
Symptoms: White shiny blisters/pustules on leaves, stems, and pods. Infected pods become distorted (staghead). 
Treatment: Mancozeb 75 WP at 2g per litre, spray 2 times at 15-day intervals. OR Metalaxyl + Mancozeb 2.5g per litre.
Prevention: Sow on time (Oct 1-20 in western Rajasthan), avoid late sowing, maintain proper plant spacing.
Organic: Copper oxychloride 3g per litre as alternative.
Varieties: RH-8812, Rajasthan varieties RN-393 — less susceptible.
Source: ICAR-DRMR Rabi Advisory, Rajasthan Agriculture Department
"""
    },
    {
        "id": "wheat-yellow-rust",
        "crop": "Wheat",
        "problem_type": "disease",
        "keywords": "wheat gehun yellow rust orange stripe patti dhariyan rust disease",
        "content": """
WHEAT — YELLOW RUST / STRIPE RUST (Puccinia striiformis)
Symptoms: Bright yellow/orange stripes running along leaf veins. URGENT — spreads rapidly in cool humid weather (Jan-Feb in Rajasthan).
Treatment: Propiconazole 25 EC at 1ml per litre water — spray IMMEDIATELY on first appearance. Second spray after 15 days if needed.
Also effective: Tebuconazole 250 EW at 1ml per litre.
Timing: Do not delay even 1 week — rust can destroy entire crop in 3 weeks.
Organic: No effective organic cure for rust — chemical treatment is mandatory for this disease.
Varieties: HD-2781, Raj-4120, K-307, PBW-343 — rust resistant varieties for Rajasthan.
Water: Stop overhead irrigation as it spreads spores. Use drip or furrow irrigation only.
Source: Wheat Research Station Durgapura, Jaipur · ICAR-IIWBR Karnal Advisory
"""
    },
    {
        "id": "cumin-wilt",
        "crop": "Cumin",
        "problem_type": "disease",
        "keywords": "cumin jeera wilt paghal dying collapse entire plant fusarium ulta paila",
        "content": """
CUMIN (JEERA) — WILT (Fusarium oxysporum f.sp. cumini)
Symptoms: Entire plant suddenly wilts and collapses. Roots show brown discoloration inside. Most destructive cumin disease — entire field can be lost in 2-3 weeks.
Treatment: Carbendazim 50 WP at 1g per litre — soil drench near roots immediately. Also spray foliage.
Seed Treatment: Carbendazim 2g + Thiram 2g per kg seed before sowing (MANDATORY in Jodhpur belt).
Soil Treatment: Trichoderma viride 2.5 kg per acre mixed in compost, apply at sowing time.
Prevention: Crop rotation — do NOT grow cumin in same field for 3 consecutive years. Avoid waterlogging.
Varieties: RZ-19, RZ-209, GC-4 (Gujarat Cumin-4, adapted to Rajasthan), Jodhpur Jeera-1.
Water: Cumin is VERY sensitive to moisture — use ONLY light irrigations. Drip irrigation strongly recommended.
Source: ICAR-NRCSS Ajmer, Rajasthan Agriculture Dept. Spice Crop Advisory
"""
    },
    {
        "id": "cumin-blight",
        "crop": "Cumin",
        "problem_type": "disease",
        "keywords": "cumin jeera blight dark spots leaves dying tips burning",
        "content": """
CUMIN (JEERA) — BLIGHT (Alternaria burnsii / Ramularia cumin)
Symptoms: Small dark brown spots on leaves and stems that rapidly enlarge. Affected parts die from tips. In severe cases entire crop appears burnt.
Treatment: Mancozeb 75 WP at 2g per litre. Or Carbendazim 50 WP + Mancozeb 75 WP mix. Spray 2-3 times at 10-day intervals.
Prevention: Do not apply overhead irrigation, use drip only. Optimal sowing time Nov 15-Dec 15 in Jodhpur.
Organic: Copper oxychloride 3g per litre as partial organic option.
Source: ICAR-NRCSS Ajmer, Directorate of Spices Development
"""
    },
    {
        "id": "moong-yellow-mosaic",
        "crop": "Moong",
        "problem_type": "disease",
        "keywords": "moong green gram yellow mosaic virus YMV yellow patches leaves mottling",
        "content": """
MOONG / MOTH BEAN — YELLOW MOSAIC VIRUS (YMV)
Symptoms: Bright yellow and green patches on leaves, mottled appearance. Affected plants give no yield. Virus spread by whitefly vector.
Treatment: NO cure for virus. Remove and destroy infected plants immediately to prevent spread. Control whitefly vector: Imidacloprid 17.8 SL 0.5ml per litre OR Thiamethoxam 25 WG 0.3g per litre.
Yellow Sticky Traps: Place 20 per acre to catch whiteflies and monitor infestation.
Varieties: YMV-resistant varieties: PDM-139 (Samrat), MH-2-15, Pusa Vishal.
Organic: Neem oil 5ml per litre spray to control whitefly. Yellow sticky traps.
Source: ICAR-IIPR Kanpur, Rajasthan Agriculture Dept.
"""
    },
    {
        "id": "groundnut-tikka",
        "crop": "Groundnut",
        "problem_type": "disease",
        "keywords": "groundnut mungfali tikka leaf spot dark spots circular lesions",
        "content": """
GROUNDNUT — TIKKA / LEAF SPOT (Cercospora arachidicola)
Symptoms: Circular dark brown/black spots on leaves with yellow halo. Severe infection causes early defoliation reducing yield by 50%.
Treatment: Chlorothalonil 75 WP at 2g per litre, start spraying at 30 days after germination, repeat at 15-day intervals (3-4 sprays total).
Organic: Tebuconazole or copper-based fungicide as partial organic options.
Soil: Ensure gypsum application at pod initiation stage for calcium nutrition.
Varieties: For Rajasthan's sandy soils — GG-20, ICGV-86031.
Source: ICAR-RCER Patna, Rajasthan Agriculture Dept. Kharif Bulletin
"""
    },
    {
        "id": "cotton-bollworm",
        "crop": "Cotton",
        "problem_type": "pest",
        "keywords": "cotton kapas bollworm boll damaged holes caterpillar fruit dropping",
        "content": """
COTTON — AMERICAN BOLLWORM (Helicoverpa armigera)
Symptoms: Circular holes in bolls, shed squares/bolls, larvae inside bolls with frass.
Treatment: Spinosad 45 SC at 0.3ml per litre OR Indoxacarb 14.5 SC at 1ml per litre. Emamectin Benzoate 5 SG at 0.4g per litre for severe infestation.
Biological: Bt Kurstaki spray 1.5kg per acre (most effective for young larvae).
Pheromone Traps: Place 5 per acre for monitoring — spray when 5 moths per trap per night caught.
Bolls Not Opening: Could also be moisture stress — check soil moisture, avoid water stress at boll development stage.
Source: ICAR-CICR Nagpur, Cotton Corporation of India Advisory
"""
    },
    {
        "id": "water-jodhpur-barmer",
        "crop": "all",
        "problem_type": "water",
        "keywords": "water irrigation Jodhpur Barmer Jaisalmer Bikaner groundwater scarce arid",
        "content": """
WATER ADVISORY — WESTERN ARID ZONE (Jodhpur, Barmer, Jaisalmer, Bikaner, Nagaur, Pali)
Groundwater Status: CRITICAL — average depletion of 0.5-1.5m per year. Many blocks in red zone.
Mandatory Practices:
- Shift to DRIP IRRIGATION for Jeera, Coriander, Vegetables (saves 40-60% water).
- Sprinkler system for Bajra, Moong, Moth Bean (saves 25-35%).
- Avoid flood/furrow irrigation for spice crops completely.
- Irrigate at CRI stage and flowering only for wheat equivalent crops.
- Mulching with crop residue reduces evaporation by 30%.
Scheme: PM Krishi Sinchayee Yojana — 55-75% subsidy on drip/sprinkler installation. Apply at agriculture office.
Source: CGWB Rajasthan Report 2024, CAZRI Jodhpur Water Management Division
"""
    },
    {
        "id": "water-hanumangarh-ganganagar",
        "crop": "all",
        "problem_type": "water",
        "keywords": "water irrigation Hanumangarh Sri Ganganagar canal IGNP",
        "content": """
WATER ADVISORY — NORTHERN CANAL ZONE (Hanumangarh, Sri Ganganagar)
Irrigation Source: IGNP (Indira Gandhi Nahar Pariyojana) canal system.
Schedule: Follow canal department roster. Canal water available in rotation — store water when available.
Wheat: 4-6 irrigations. CRI (crown root initiation at 20-25 DAS) is most critical stage.
Mustard: 2-3 irrigations at rosette and pod-filling stages.
Salinity: Some areas have saline water — check EC. Use gypsum soil amendment.
Source: Rajasthan Canal Department, ICAR-CSSRI Karnal Advisory for Canal Zone
"""
    },
    {
        "id": "organic-general",
        "crop": "all",
        "problem_type": "organic",
        "keywords": "organic natural solution jeevamrit panchagavya neem jaivik kheti",
        "content": """
ORGANIC / NATURAL FARMING SOLUTIONS FOR RAJASTHAN
Jeevamrit (General immunity booster):
- 10kg fresh cow dung + 10L cow urine + 2kg jaggery + 2kg gram flour + handful of native soil
- Mix in 200L water, ferment 48 hours in shade, dilute 1:10 with water, use as soil drench or foliar spray weekly.
Panchagavya (5% solution): Mix 5 cow products, ferment, dilute — spray fortnightly for plant immunity.
Neem Oil Spray: 5ml neem oil + 2ml liquid soap per litre water — spray in evening for pests.
Neem Cake: 200-250kg per acre as basal dose controls soil-borne pests and adds nutrients.
Trichoderma viride: 2.5kg per acre mixed in 50kg FYM, apply at sowing — controls Fusarium, Pythium diseases.
Beauveria bassiana: 1L per acre spray in evening — biological control for caterpillars, stem borers.
Source: National Centre of Organic Farming, ICAR Guidelines for Organic Farming
"""
    },
]

# ── RAG Engine ─────────────────────────────────────────────────────────────────
class CropRAG:
    """
    RAG engine using FAISS for vector similarity search.
    Falls back to keyword matching when sentence-transformers unavailable.
    """

    def __init__(self):
        self.knowledge = CROP_KNOWLEDGE
        self.index = None
        self._use_faiss = False

    def build_index(self):
        """Build FAISS vector index from crop knowledge base."""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss

            print("[RAG] Loading SentenceTransformer model...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

            texts = [
                f"{k['crop']} {k['keywords']} {k['content']}"
                for k in self.knowledge
            ]
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            dim = embeddings.shape[1]

            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings.astype(np.float32))
            self._use_faiss = True
            print(f"[RAG] FAISS index built with {len(self.knowledge)} knowledge entries.")

        except ImportError:
            print("[RAG] FAISS/sentence-transformers not installed. Using keyword fallback.")
            self._use_faiss = False

    def retrieve(self, crop: str, district: str, problem: str, top_k: int = 3) -> str:
        """
        Retrieve relevant knowledge chunks for a given crop query.
        Returns concatenated context string for the LLM.
        """
        if self._use_faiss:
            return self._faiss_retrieve(crop, district, problem, top_k)
        return self._keyword_retrieve(crop, district, problem, top_k)

    def _faiss_retrieve(self, crop: str, district: str, problem: str, top_k: int) -> str:
        import faiss
        query = f"{crop} {district} {problem}"
        qvec = self._model.encode([query], normalize_embeddings=True).astype(np.float32)
        distances, indices = self.index.search(qvec, min(top_k, len(self.knowledge)))
        results = [self.knowledge[i]["content"] for i in indices[0] if i < len(self.knowledge)]
        return "\n\n---\n\n".join(results)

    def _keyword_retrieve(self, crop: str, district: str, problem: str, top_k: int) -> str:
        """Keyword-based fallback when FAISS is unavailable."""
        query_words = set((crop + " " + district + " " + problem).lower().split())
        scores = []

        for entry in self.knowledge:
            kw_words = set(entry["keywords"].lower().split())
            crop_match = entry["crop"].lower() in crop.lower() or crop.lower() in entry["crop"].lower()
            overlap = len(query_words & kw_words)
            score = overlap + (5 if crop_match else 0)

            # Water advisory bonus for arid districts
            arid = ["jodhpur", "barmer", "jaisalmer", "bikaner", "nagaur", "pali", "churu", "sikar"]
            if district.lower() in arid and "water" in entry["problem_type"]:
                score += 3

            scores.append((score, entry))

        scores.sort(key=lambda x: -x[0])
        results = [e["content"] for _, e in scores[:top_k] if _[0] > 0]

        if not results:
            # Return general organic knowledge as fallback
            results = [self.knowledge[-2]["content"]]

        return "\n\n---\n\n".join(results)
