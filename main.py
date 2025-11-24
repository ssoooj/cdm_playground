import os
import re
import json
import yaml
import logging
import pandas as pd
from functools import lru_cache
from typing import Dict, Any, Tuple, Optional, Union
from datetime import date, datetime
from openai import OpenAI

from db_connector import PostgresConnector
from terminology_mapper_default import TerminologyMapper
from anchor_strategy import load_anchor_strategy, AutoClampAnchor
from sql_builder import SQLBuilder
from domain_spec import DOMAIN_SPEC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
print(BASE_DIR, CONFIG_PATH)

def _force_domain_override(base_term: str) -> str | None:
    t = (base_term or "").lower()
    # HbA1c → Measurement
    hba1c_keys = ["hba1c", "a1c", "glycated hemoglobin", "glycohemoglobin", "당화혈색소", "헤모글로빈a1c"]
    if any(k in t for k in hba1c_keys):
        return "Measurement"
    # Death/Mortality → Death
    death_keys = ["death", "mortality", "사망", "사망률"]
    if any(k in t for k in death_keys):
        return "Death"
    return None

class CDMRagSystem:
    def __init__(self, config_path: str = CONFIG_PATH):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"설정 파일({config_path})을 찾을 수 없습니다.")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        db_params = config.get("postgresql")
        if not db_params:
            raise ValueError("설정 파일에 'postgresql' 섹션이 없습니다.")
        self.db_connector = PostgresConnector(db_params)

        llm_config = config.get("llm")
        if not llm_config:
            raise ValueError("설정 파일에 'llm' 섹션이 없습니다.")
        provider = llm_config["provider"]
        provider_cfg = llm_config[provider]

        self.llm = OpenAI(
            base_url=provider_cfg["api_base"],
            api_key=provider_cfg["api_key"],
        )
        self.llm_model = provider_cfg["model"]
        self.llm_params = provider_cfg.get("parameters", {}) or {}

        self.term_mapper = TerminologyMapper(self.db_connector, self.llm, self.llm_model)
        self.anchor_base = load_anchor_strategy(config)                   # 진짜 앵커( resolve 제공 )
        self.auto_anchor = AutoClampAnchor(base=self.anchor_base, lookback_days=365*5)
        self.anchor_strategy = self.anchor_base                           # SQLBuilder 에는 base만 넘긴다
        self.sql_builder = SQLBuilder(
            self.db_connector,
            self.anchor_strategy,
            DOMAIN_SPEC,
            self.llm,
            self.llm_params
        )
        self.default_timeout = 30
        self.default_max_tokens_small = 500
        self.default_max_tokens_medium = 1000
        self.default_max_tokens_large = 3000

        logger.info(f"시스템 초기화 완료: DB 및 LLM({provider}) 클라이언트가 준비되었습니다.")

    def resolve_domain(self, intent: int, user_query: str, base_term: Optional[str] = None) -> str:
        q = user_query.lower()
        proc_keywords_ko = ["수술", "시술", "이식", "삽입", "제거", "절제", "봉합", "성형", "우회", "치환"]
        proc_keywords_en = ["surgery", "procedure", "transplant", "ectomy", "otomy", "plasty", "bypass", "insertion", "removal", "repair", "arthroplasty", "stent"]
        keyword_proc = any(k in user_query for k in proc_keywords_ko) or any(k in q for k in proc_keywords_en)

        if keyword_proc or intent == 4:
            domain_heur = "Procedure"
        elif "약" in user_query or "처방" in user_query or "drug" in q or "medication" in q:
            domain_heur = "Drug"
        elif "검사" in user_query or "lab" in q or "loinc" in q or "수치" in user_query:
            domain_heur = "Measurement"
        elif "관찰" in user_query or "observation" in q:
            domain_heur = "Observation"
        elif "기기" in user_query or "device" in q:
            domain_heur = "Device"
        elif ("사망" in user_query) or ("사망률" in user_query) or ("death" in q) or ("mortality" in q):
            domain_heur = "Death"
        else:
            domain_heur = "Condition"

        guessed = self.term_mapper._guess_domain_by_terminology(base_term) if base_term else None

        if keyword_proc:
            return "Procedure"

        if guessed:
            return guessed

        return domain_heur

    def _chat(self, prompt: str, *, max_tokens: int = 10000, temperature: float = 0.0, timeout: Optional[int] = None):
        return self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout or self.default_timeout,
        )

    @lru_cache(maxsize=128)
    def get_intent_from_llm(self, query: str) -> int:
        prompt = f"""
        너는 임상 데이터 관련 질문의 의도를 파악하는 전문가야.
        다음 질문의 의도를 분석 후 아래 옵션 중 가장 적합한 것의 '번호'만 대답해.

        질문: "{query}"

        옵션:
        1. 특정 질환(예: 고혈압, 당뇨병)을 가진 환자의 인구통계학적 정보 관련
        2. 특정 질환 환자의 약물 관련
        3. 특정 질환 환자의 진료 기록(방문/입원) 관련
        4. 특정 질환 환자의 시술/검사 관련
        5. OMOP CDM 특정 개념 정보
        """
        try:
            r = self._chat(prompt, max_tokens=8, temperature=0.0, timeout=10)
            content = r.choices[0].message.content or ""
            nums = re.findall(r"\d+", content)
            return int(nums[0]) if nums else 5
        except Exception as e:
            logger.error(f"LLM 의도 파악 오류: {e}")
            return 5

    def extract_conditions_from_query(self, user_query: str) -> Dict[str, Any]:
        prompt = f"""
        너는 사용자의 자연어 질문을 분석하여 SQL 쿼리 생성을 위한 조건들을 JSON 객체로 변환하는 에이전트야.

        [작업 지시]
        - 다음 [사용자 질문]을 분석하여 아래 [필드 설명]에 따라 JSON 객체를 생성해.
        - 해당 정보가 없으면 필드는 생략 가능.
        - 다른 설명 없이 오직 JSON 객체만 출력하세요. 코드 블록 사용 금지.

        [필드 설명]
        - "gender": 'MALE' 또는 'FEMALE'
        - "age": 정수
        - "age_comparison": '>=', '<=', '=', '>', '<' 중 하나
        - "aggregation": 'count' 또는 'select'
        - "within_years": 정수 (예: 최근 2년 → 2)
        - "within_months": 정수 (예: 최근 6개월 → 6)
        - "within_days": 정수 (예: 최근 30일 → 30)
        - "date_from": 'YYYY-MM-DD' (이날 이후)
        - "date_to": 'YYYY-MM-DD' (이날 이전)

        [예시]
        - "최근 2년 내 패혈증 남자 60세 이상 몇 명?" ->
        {{"gender":"MALE","age":60,"age_comparison":">=","aggregation":"count","within_years":2}}
        - "2022년 이후 진단받은 여자 당뇨 환자 수" ->
        {{"gender":"FEMALE","aggregation":"count","date_from":"2022-01-01"}}
        - "지난 6개월 고혈압 환자 목록" ->
        {{"aggregation":"select","within_months":6}}

        ---
        [사용자 질문]
        "{user_query}"

        [응답]
        """
        try:
            r = self._chat(
                prompt,
                max_tokens=self.default_max_tokens_small,
                temperature=0.0,
                timeout=20,
            )
            content = r.choices[0].message.content or ""
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if not m:
                logger.error(f"조건 JSON 추출 실패: {content}")
                return {}
            conditions = json.loads(m.group(0))
            logger.info(f"질문에서 추출된 조건: {conditions}")
            return conditions
        except Exception as e:
            logger.error(f"조건 추출 LLM 오류: {e}")
            return {}
        
    def process_query(self, user_query: str) -> str:
        try:
            logger.info("STEP1 map_term start")
            mapped = self.term_mapper.map_term(user_query)
            logger.info("STEP1 map_term done: %s", mapped)
            if not mapped:
                return "질문에서 핵심 의학 용어를 안정적으로 매핑하지 못함. 용어를 바꿔 다시 시도하시오."

            base_term = mapped["base_term"]
            specific = bool(mapped.get("specific", True))

            logger.info("STEP1.5 intent/domain resolve")
            _domain_fix = _force_domain_override(base_term)
            if _domain_fix:
                domain = _domain_fix
            intent = self.get_intent_from_llm(user_query)
            domain = self.resolve_domain(intent, user_query, base_term=base_term)
            logger.info("domain=%s (intent=%s)", domain, intent)

            logger.info("STEP2 extract_conditions start")
            conditions = self.extract_conditions_from_query(user_query)
            logger.info("STEP2 extract_conditions done: %s", conditions)

            def _run_once(_domain: str, _term: str, _specific: bool):
                targets_cte_sql, targets_cte_params = self.sql_builder._seed_and_closure_cte(_domain, _term, _specific)

                user_date_from = (conditions or {}).get("date_from")

                anchor_date_str, effective_date_from = self.auto_anchor.resolve_with_clamp(
                    db=self.db_connector,
                    domain=_domain,
                    base_term=_term,
                    user_date_from=user_date_from,
                    targets_cte_sql=targets_cte_sql,
                    targets_cte_params=targets_cte_params,
                )

                if effective_date_from and not user_date_from:
                    cond = dict(conditions or {})
                    cond["date_from"] = effective_date_from
                else:
                    cond = dict(conditions or {})

                sql_query, params = self.sql_builder.build_dynamic_sql(
                    domain=_domain,
                    base_term=_term,
                    specific=_specific,
                    conditions=cond,
                )
                result_df = self.db_connector.execute_query(sql_query, params)

                total = None
                if "patient_count" in result_df.columns and not result_df.empty:
                    try:
                        total = int(result_df.iloc[0]["patient_count"])
                    except Exception:
                        total = None

                try:
                    cte, cparams = self.sql_builder._seed_and_closure_cte(_domain, _term, _specific)
                    df_t = self.db_connector.execute_query(f"""{cte}
                        SELECT COUNT(*) AS n FROM targets;""", cparams)
                    n_targets = int(df_t.iloc[0]["n"]) if not df_t.empty else 0

                    spec = self.sql_builder._get_domain_spec(_domain)
                    tbl, fld = spec["table"], spec["concept_field"]
                    df_o = self.db_connector.execute_query(f"""{cte}
                        SELECT COUNT(*) AS n_rows, COUNT(DISTINCT person_id) AS n_persons
                        FROM {tbl}
                        WHERE {fld} IN (SELECT concept_id FROM targets);""", cparams)
                    n_rows = int(df_o.iloc[0]["n_rows"]) if not df_o.empty else 0
                    n_persons = int(df_o.iloc[0]["n_persons"]) if not df_o.empty else 0

                    logger.info(
                        "DEBUG targets: domain=%s term='%s' specific=%s → targets=%d, occ_rows=%d, occ_persons=%d",
                        _domain, _term, _specific, n_targets, n_rows, n_persons
                    )
                except Exception as de:
                    logger.warning("DEBUG targets 실패: %s", de)

                return total, sql_query, result_df
            
            quick_topn = None
            if domain == "Drug":
                m1 = re.search(r'(?i)\btop\s*(\d{1,3})\b', user_query)
                m2 = re.search(r'상위\s*(\d{1,3})', user_query)
                if ('top' in user_query.lower()) or ('상위' in user_query):
                    try:
                        quick_topn = int((m1 or m2).group(1)) if (m1 or m2) else 10
                    except Exception:
                        quick_topn = 10

            if quick_topn:
                logger.info("FAST PATH: Drug TopN by patients detected (topn=%s)", quick_topn)
                sql_query, params = self.sql_builder.build_drug_topn_by_patients(
                    base_term=base_term,
                    specific=specific,
                    topn=quick_topn,
                    conditions=conditions
                )
                result_df = self.db_connector.execute_query(sql_query, params)
                logger.info("STEP4 execute_sql (fast path) done: %s rows", 0 if result_df is None else len(result_df))
                text = self.sql_builder.interpret_result(result_df, user_query, sql_query)
                return text, sql_query

            logger.info("STEP3 build_sql & STEP4 execute_sql (primary)")
            total_n, sql_query, result_df = _run_once(domain, base_term, specific)
            logger.info("STEP4 execute_sql done (primary): %s", total_n)

            if (total_n == 0) and specific:
                logger.info("0명 결과 → descendants 확장 폴백 재시도 (specific=False)")
                total_n, sql_query, result_df = _run_once(domain, base_term, False)
                logger.info("STEP4 execute_sql (fallback A) done: %s", total_n)

            if (total_n == 0) and (domain == "Measurement"):
                low = (base_term or "").lower()
                is_sbp = ("systolic blood pressure" in low) or ("sbp" in low) or ("8480-6" in low)
                if is_sbp:
                    logger.info("0명 결과 → SBP Observation 규칙 개념 폴백 재시도")

                    try_days = []
                    if "within_days" in conditions:
                        try_days.append(int(conditions["within_days"]))
                    try_days += [180, 365, None]  # 이미 90이면 180→365→무제한

                    obs_n = 0
                    obs_sql = None
                    obs_df = None

                    for d in try_days:
                        cond2 = dict(conditions)  # shallow copy
                        cond2.pop("within_months", None)
                        cond2.pop("within_years", None)
                        if d is None:
                            cond2.pop("within_days", None)
                            log_sfx = "no-date"
                        else:
                            cond2["within_days"] = d
                            log_sfx = f"{d}d"

                        logger.info(f"SBP Observation 폴백 시도({log_sfx})")
                        sql_try, params_try = self.sql_builder.build_bp_observation_sql(cond2)
                        df_try = self.db_connector.execute_query(sql_try, params_try)

                        if "patient_count" in df_try.columns and not df_try.empty:
                            try:
                                n_try = int(df_try.iloc[0]["patient_count"])
                            except Exception:
                                n_try = 0
                        else:
                            n_try = 0

                        logger.info(f"SBP Observation 폴백 결과({log_sfx}): {n_try}")
                        if n_try > 0:
                            obs_n = n_try
                            obs_sql = sql_try
                            obs_df = df_try
                            break

                    if obs_n > 0:
                        total_n = obs_n
                        sql_query = obs_sql
                        result_df = obs_df
                        domain = "Observation"
                        base_term = "Blood pressure (rule concepts)"
                        logger.info("SBP Observation 폴백 성공: %d명", obs_n)

            if (total_n == 0) and (domain == "Procedure"):
                for vt in self.sql_builder._transplant_variants(base_term):
                    if vt == base_term:
                        continue
                    logger.info("0명 결과 → 용어 변형 재시도: '%s'", vt)
                    n_try, sql_try, df_try = _run_once(domain, vt, False)
                    if n_try and n_try > 0:
                        base_term = vt
                        total_n, sql_query, result_df = n_try, sql_try, df_try
                        logger.info("용어 변형 성공: '%s' → %d명", vt, n_try)
                        break

            if (total_n == 0) and (domain == "Procedure"):
                logger.info("0명 결과 → Condition 도메인 폴백 재시도")
                n_try, sql_try, df_try = _run_once("Condition", base_term, False)
                if n_try and n_try > 0:
                    domain = "Condition"
                    total_n, sql_query, result_df = n_try, sql_try, df_try
                    logger.info("Condition 폴백 성공: %d명", n_try)

            logger.info("STEP5 interpret_result start")

            if total_n is not None:
                label_by_domain = {
                    "Condition": "질환(용어)",
                    "Procedure": "시술/수술(용어)",
                    "Drug": "약물(용어)",
                    "Measurement": "검사(용어)",
                    "Observation": "관찰(용어)",
                    "Device": "의료기기(용어)",
                    "Death": "사망(이벤트)"
                }

                extras: Dict[str, Any] = {}

                def _safe_extras(fn, *args, **kwargs):
                    try:
                        return fn(*args, **kwargs)
                    except Exception as ee:
                        logger.warning("extras(%s) 계산 실패: %s", domain, ee)
                        return {}

                if total_n and total_n > 0:
                    if domain == "Condition":
                        extras = _safe_extras(self.sql_builder.compute_extras_condition,
                                              base_term, specific, conditions, total_n)
                    elif domain == "Procedure":
                        extras = _safe_extras(self.sql_builder.compute_extras_procedure,
                                              base_term, specific, conditions, total_n)
                    elif domain == "Drug":
                        extras = _safe_extras(self.sql_builder.compute_extras_drug,
                                              base_term, specific, conditions, total_n)
                    elif domain == "Measurement":
                        extras = _safe_extras(self.sql_builder.compute_extras_measurement_numeric,
                                              base_term, specific, conditions, total_n)
                    elif domain == "Observation":
                        extras = _safe_extras(self.sql_builder.compute_extras_observation,
                                              base_term, specific, conditions, total_n)
                    elif domain == "Device":
                        extras = _safe_extras(self.sql_builder.compute_extras_device,
                                              base_term, specific, conditions, total_n)

                def _fmt_pct(x):
                    return f"{x:.2f}%" if x is not None else "N/A"

                def _to_numeric_for_front(extras: Dict[str, Any],
                                          conditions: Dict[str, Any],
                                          total_n: int) -> Dict[str, Any]:
                    v = (extras or {}).get("value", {})
                    summ = v.get("summary", {})
                    hist = v.get("histogram", [])

                    labels, counts = [], []
                    if hist:
                        if "bucket" in hist[0]:
                            labels = [str(h.get("bucket")) for h in hist]
                            counts = [int(h.get("n") or 0) for h in hist]
                        elif "bin_low" in hist[0] and "bin_high" in hist[0]:
                            def _fmt(x):
                                try:
                                    return f"{float(x):.2f}"
                                except Exception:
                                    return str(x)
                            labels = [f"{_fmt(h.get('bin_low'))}–{_fmt(h.get('bin_high'))}" for h in hist]
                            counts = [int(h.get("n") or 0) for h in hist]

                    unit_top = [{"unit": u.get("unit"), "n": int(u.get("count", 0))}
                                for u in v.get("unit_candidates", [])]

                    quantiles = {
                        "min":   summ.get("min"),
                        "p50":   summ.get("median"),
                        "mean":  summ.get("mean"),
                        "p95":   summ.get("p95"),
                        "p99":   summ.get("p99"),
                        "max":   summ.get("max"),
                    }

                    return {
                        "value_strategy": (conditions.get("value_strategy") or "latest"),
                        "outlier": conditions.get("outlier"),
                        "count": total_n,
                        "quantiles": quantiles,
                        "histogram": {"labels": labels, "counts": counts},
                        "unit_top": unit_top,
                    }

                analytics = {"base_term": base_term, "total_n": total_n, "domain": domain}
                analytics.update(extras or {})
                if domain == "Measurement" and extras:
                    analytics["numeric"] = _to_numeric_for_front(extras, conditions, total_n)

                try:
                    ql = (user_query or "").lower()
                    wants_hba1c = ("hba1c" in ql) or ("당화혈색소" in ql) or ("당화 혈색소" in ql)
                    if wants_hba1c and domain in ("Condition", "Drug"):
                        sql_h, params_h = self.sql_builder.build_hba1c_summary_for_condition(base_term, specific, conditions)
                        df_h = self.db_connector.execute_query(sql_h, params_h)
                        if not df_h.empty:
                            hv = df_h.iloc[0].to_dict()
                            # 숫자 포맷
                            mean = hv.get("mean")
                            median = hv.get("median")
                            n_vals = int(hv.get("n") or 0)
                            # 결과 텍스트 구성에 사용할 수 있도록 extras에 덧붙임
                            if "value" not in analytics:
                                analytics["value"] = {}
                            analytics["value"]["hba1c_summary"] = {
                                "n": n_vals,
                                "mean": float(mean) if mean is not None else None,
                                "median": float(median) if median is not None else None,
                                "p05": float(hv.get("p05")) if hv.get("p05") is not None else None,
                                "p25": float(hv.get("p25")) if hv.get("p25") is not None else None,
                                "p75": float(hv.get("p75")) if hv.get("p75") is not None else None,
                                "p95": float(hv.get("p95")) if hv.get("p95") is not None else None,
                            }
                except Exception as _e:
                    logger.warning("HbA1c 요약 계산 실패: %s", _e)

                lines = []
                lines.append('결과 요약')
                lines.append(f'{label_by_domain.get(domain, "개념(용어)")} : {base_term}')
                lines.append(f'환자 수: {total_n:,}명')

                if extras:
                    if extras.get("pct_of_all") is not None and extras.get("base_all") is not None:
                        lines.append(f'전체 환자 대비: {_fmt_pct(extras["pct_of_all"])} (모수 {extras["base_all"]:,}명)')
                    if extras.get("pct_of_base_age_gender") is not None and extras.get("base_age_gender") is not None:
                        lines.append(f'질문 조건(성별/나이) 모수 대비: {_fmt_pct(extras["pct_of_base_age_gender"])} (모수 {extras["base_age_gender"]:,}명)')
                    lines.append('')

                    rec = extras.get("recency", {})
                    if rec:
                        lines.append('최근 활동(중복 허용)')
                        lines.append(f'- 365일: {rec.get("n365",{}).get("n",0):,}명 ({_fmt_pct(rec.get("n365",{}).get("pct"))})')
                        lines.append(f'- 180일: {rec.get("n180",{}).get("n",0):,}명 ({_fmt_pct(rec.get("n180",{}).get("pct"))})')
                        lines.append(f'-  90일: {rec.get("n90",{}).get("n",0):,}명 ({_fmt_pct(rec.get("n90",{}).get("pct"))})')
                        lines.append(f'-  30일: {rec.get("n30",{}).get("n",0):,}명 ({_fmt_pct(rec.get("n30",{}).get("pct"))})')
                        lines.append('')

                    ab = extras.get("age_buckets", [])
                    if ab:
                        lines.append('연령대 분포')
                        for i in ab:
                            lines.append(f'- {i["bucket"]}: {i["n"]:,}명 ({_fmt_pct(i["pct"])})')

                    gd = extras.get("gender_breakdown", [])
                    if gd:
                        lines.append('')
                        lines.append('성별 분포')
                        for i in gd:
                            lines.append(f'- {i["gender"]}: {i["n"]:,}명 ({_fmt_pct(i["pct"])})')

                    tops = extras.get("top_concepts", [])
                    if tops:
                        lines.append('')
                        title_map = {
                            "Condition": "상위 진단 개념(환자 수 기준)",
                            "Procedure": "상위 시술/수술 개념(환자 수 기준)",
                            "Drug": "상위 약물 개념(환자 수 기준)",
                        }
                        lines.append(title_map.get(domain, "상위 개념(환자 수 기준)"))
                        for t in tops:
                            lines.append(f'- {t["concept_name"]}: {t["n"]:,}명 ({_fmt_pct(t["pct"])})')

                    lines.append("")
                    lines.append(f"<!--CDM_METRICS:{json.dumps({'base_term': base_term, 'metrics': extras, 'total_n': total_n, 'domain': domain}, ensure_ascii=False)}-->")

                return {
                    "answer": "\n".join(lines),
                    "meta": { "analytics": analytics }
                }

            return self.sql_builder.interpret_result(result_df, user_query, sql_query)

        except Exception as e:
            logger.error("질문 처리 중 오류: %s", e, exc_info=True)
            return f"오류 발생: {e}"

def main():
    cfg_path = CONFIG_PATH
    rag_system = CDMRagSystem(config_path=cfg_path)
    rag_system.db_connector.connect()

    print("CDM RAG 시스템입니다. 질문을 입력해주세요. (종료는 'quit')", flush=True)
    try:
        while True:
            user_query = input("질문: ").strip()
            if user_query.lower() in {"quit", "exit"}:
                break
            print("[debug] got user input:", user_query, flush=True)

            answer = rag_system.process_query(user_query)
            print("\n[답변]\n", answer, "\n", flush=True)
    finally:
        rag_system.db_connector.close()
        print("시스템 종료.", flush=True)

if __name__ == "__main__":
    main()