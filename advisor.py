"""
advisor.py — Claude-powered Crop Advisory Generator
Uses RAG context + IBM Granite-style prompt engineering to generate
verified, district-specific crop advisory for Rajasthan farmers.
"""

import json
import anthropic

SYSTEM_PROMPT = """You are KisanMitra AI — an expert agricultural advisor for Rajasthan farmers, trained on:
- ICAR (Indian Council of Agricultural Research) bulletins
- Rajasthan Agriculture Department district advisories  
- CAZRI Jodhpur (Central Arid Zone Research Institute) research
- IMD (India Meteorological Department) rainfall data for Rajasthan
- National Horticulture Board pest management guidelines

CORE RULES:
1. Use ONLY verified, evidence-based recommendations.
2. Include EXACT chemical doses (g/ml per litre or per acre). Never vague amounts.
3. ALWAYS include an organic/natural alternative.
4. Give water advisory SPECIFIC to the farmer's district zone.
5. Recommend ICAR-approved resistant varieties where applicable.
6. NEVER recommend WHO Class I (extremely hazardous) pesticides.
7. If Hindi input, respond in Hindi. If English, respond in English.
8. Keep language simple — farmer-friendly, no technical jargon.

DISTRICT WATER ZONES:
- CRITICAL ARID (reduce irrigation 20-25%, push drip): Jodhpur, Barmer, Jaisalmer, Bikaner, Churu, Nagaur, Sikar, Pali, Jhunjhunu
- MODERATE STRESS (2-3 irrigations only): Ajmer, Tonk, Bhilwara, Chittorgarh, Bundi, Alwar
- CANAL IRRIGATED (relatively sufficient): Hanumangarh, Sri Ganganagar  
- GOOD RAINFALL (600-800mm, less irrigation): Kota, Bundi, Jhalawar, Udaipur, Dungarpur

RESPOND ONLY AS VALID JSON — no markdown, no explanation outside JSON:
{
  "summary": "1-line summary of the problem and solution",
  "diagnosis": "What disease/pest/problem this is. Include scientific name. 2-3 sentences.",
  "action": "Exact immediate step. Chemical name + dose per litre or per acre. 2-3 actionable sentences.",
  "water": "Water management specific to their district zone. Irrigation stage + frequency.",
  "prevention": "What to do next season to prevent recurrence. 1-2 sentences.",
  "organic": "Specific organic/natural alternative with dose and timing.",
  "variety": "Best ICAR-recommended resistant variety for this crop+district (or null if not applicable)",
  "sources": "Specific ICAR bulletin name or Rajasthan Agri Dept. reference"
}"""


class CropAdvisor:
    """Generates crop advisory using Claude with RAG-retrieved context."""

    def __init__(self, rag_engine):
        self.client = anthropic.Anthropic()
        self.rag = rag_engine

    def get_advisory(
        self,
        crop: str,
        district: str,
        season: str,
        problem: str,
        context: str,
        language: str = "English"
    ) -> dict:
        """
        Generate crop advisory using Claude + RAG context.

        Args:
            crop: Crop name selected by farmer
            district: Rajasthan district
            season: Kharif/Rabi/Zaid
            problem: Farmer's problem description
            context: Retrieved RAG knowledge chunks
            language: "English" or "हिंदी"

        Returns:
            dict with advisory fields (diagnosis, action, water, etc.)
        """
        lang_note = (
            "IMPORTANT: Respond in Hindi (Devanagari script) for all text fields. "
            "Keep chemical names and doses in English/numbers."
            if language == "हिंदी" else
            "Respond in simple English."
        )

        user_message = f"""Farmer Query:
- Crop: {crop}
- District: {district}, Rajasthan
- Season: {season or "Not specified"}
- Problem: {problem}
- Language: {language}

Verified Knowledge Base Context (from ICAR/Rajasthan Agri Dept. sources):
{context}

{lang_note}

Generate accurate, actionable advisory based on the above context and your agricultural knowledge."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}]
            )

            text = response.content[0].text
            clean = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean)

        except json.JSONDecodeError:
            return {
                "summary": "Advisory generated",
                "diagnosis": text[:300] if "text" in dir() else "Please check your crop carefully.",
                "action": "Consult your nearest KVK (Krishi Vigyan Kendra) for in-person advice.",
                "water": f"For {district} district, follow district-specific irrigation guidelines.",
                "prevention": "Use certified disease-free seeds next season.",
                "organic": "Apply neem oil 5ml per litre as a general preventive spray.",
                "variety": None,
                "sources": "ICAR General Advisory"
            }

        except Exception as e:
            return {
                "summary": f"Error: {str(e)}",
                "diagnosis": "Could not generate advisory. Please check your API key.",
                "action": "Ensure ANTHROPIC_API_KEY is set correctly.",
                "water": "—", "prevention": "—", "organic": "—",
                "variety": None,
                "sources": "—"
            }
