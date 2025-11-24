import re
import logging
from typing import Optional, Dict, Any
from openai import OpenAI
from db_connector import PostgresConnector

logger = logging.getLogger(__name__)

class TerminologyMapper:

    def __init__(self, db: PostgresConnector, llm: OpenAI, llm_model: str):
        self.db = db
        self.llm = llm
        self.llm_model = llm_model

    def map_term(self, user_query: str) -> Optional[Dict[str, Any]]:
        ql = (user_query or "").strip()
        if not ql:
            return None

        norm = re.sub(r"\s+", " ", ql).strip()

        base = norm

        base = base.replace("당화 혈색소", "당화혈색소")

        try:
            prompt = f'임상질문에서 핵심 의학 용어 한 개만 영어로 출력. 질문: "{ql}"'
            r = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=16,
                timeout=10,
            )
            content = (r.choices[0].message.content or "").strip()
            if content and len(content) <= 128:
                base = content
        except Exception as e:
            logger.warning(f"LLM term extract 실패: {e}")

        base = base.strip().strip('"').strip("'")

        return {"base_term": base or "Unknown", "specific": False}

    def _guess_domain_by_terminology(self, term: Optional[str]) -> Optional[str]:
        if not term:
            return None
        t = term.lower().strip()

        hba1c_keys = ["hba1c", "a1c", "glycated hemoglobin", "glycohemoglobin", "당화혈색소", "헤모글로빈a1c"]
        if any(k in t for k in hba1c_keys):
            return "Measurement"

        death_keys = ["death", "mortality", "사망", "사망률"]
        if any(k in t for k in death_keys):
            return "Death"

        return None

    def _extract_term_and_specificity(self, user_query: str) -> Optional[Dict]:
        prompt = f"""
        You are a clinical terminology helper.
        Extract ONE **English** clinical term and whether it is a **specific subtype**.
        Rules:
        - If the question mentions a specific subtype (e.g., "Lewy body dementia", "type 2 diabetes mellitus", "vascular dementia"),
        set "specific": true and return that phrase as "base_term".
        - Otherwise collapse to the **base/generic disorder** (e.g., "Hypertension", "Diabetes mellitus", "Dementia") and set "specific": false.
        Return ONLY JSON:
        {{"base_term": "...", "specific": true|false}}
        Question: "{user_query}"
        """
        r = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0, max_tokens=64, timeout=20
        )
        txt = (r.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            logger.error("LLM JSON parse fail: %s", txt); return None
        try:
            data = json.loads(m.group(0))
            if not isinstance(data, dict) or "base_term" not in data or "specific" not in data:
                return None
            data["base_term"] = str(data["base_term"]).strip()
            data["specific"] = bool(data["specific"])
            return data
        except Exception as e:
            logger.error("JSON decode error: %s", e); return None

    def _get_domain_spec(self, domain: str) -> dict:
        return DOMAIN_SPEC[domain].copy()