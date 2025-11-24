import logging
import re
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, Any, Tuple, Optional, List
from domain_spec import DOMAIN_SPEC

logger = logging.getLogger(__name__)

class SQLBuilder:
    def __init__(self, db_connector, anchor_strategy, DOMAIN_SPEC, llm=None, llm_params=None):
        self.db_connector = db_connector
        self.anchor_strategy = anchor_strategy
        self.domain_spec = DOMAIN_SPEC
        self.llm = llm
        self.llm_params = llm_params or {}
        self.default_max_tokens_large = 3000

    def _seed_and_closure_cte(self, domain: str, base_term: str, specific: bool) -> tuple[str, tuple]:
        spec = self._get_domain_spec(domain)
        std_vocs = spec.get("standard", [])
        src_vocs = list({*spec.get("sources", []), *std_vocs})
        rel_whitelist = spec.get("maps_to", ["Maps to"])
        closure_vocabs = spec.get("closure_vocabs", std_vocs)

        xwalk_rels = spec.get("xwalk_rels", ["Maps to", "Maps to value", "Concept same as"])

        like = f"%{base_term}%"

        force_expand_descendants = (domain == "Procedure")

        cte = f"""
        WITH seed AS (
            SELECT DISTINCT c.concept_id, c.standard_concept, c.vocabulary_id
            FROM concept c
            LEFT JOIN concept_synonym s ON s.concept_id = c.concept_id
            WHERE c.domain_id = %s
            AND c.vocabulary_id = ANY(%s)
            AND (c.concept_name ILIKE %s OR s.concept_synonym_name ILIKE %s)
            AND COALESCE(c.invalid_reason,'') = ''
        ),
        seed_std AS (
            SELECT concept_id
            FROM seed
            WHERE standard_concept = 'S'
            AND vocabulary_id = ANY(%s)
        ),
        mapped_raw AS (
            SELECT cr.concept_id_1, cr.concept_id_2
            FROM seed x
            JOIN concept_relationship cr
            ON cr.concept_id_1 = x.concept_id
            WHERE cr.relationship_id = ANY(%s)
            AND COALESCE(cr.invalid_reason,'') = ''
        ),
        mapped_std AS (
            SELECT DISTINCT mr.concept_id_2 AS concept_id
            FROM mapped_raw mr
            JOIN concept c2 ON c2.concept_id = mr.concept_id_2
            WHERE c2.standard_concept = 'S'
            AND c2.domain_id = %s
            AND c2.vocabulary_id = ANY(%s)
            AND COALESCE(c2.invalid_reason,'') = ''
            AND (
                    c2.concept_name ILIKE %s
                OR EXISTS (
                        SELECT 1 FROM concept_synonym s2
                        WHERE s2.concept_id = c2.concept_id
                        AND s2.concept_synonym_name ILIKE %s
                    )
            )
        ),
        seed_union AS (
            SELECT concept_id FROM seed_std
            UNION
            SELECT concept_id FROM mapped_std
        ),
        closure AS (
            {(
                "SELECT concept_id AS descendant_concept_id FROM seed_union"
                if (specific and not force_expand_descendants) else
                f"""
                SELECT su.concept_id AS descendant_concept_id
                FROM seed_union su
                UNION
                SELECT ca.descendant_concept_id
                FROM concept_ancestor ca
                JOIN concept d ON d.concept_id = ca.descendant_concept_id
                WHERE ca.ancestor_concept_id IN (SELECT concept_id FROM seed_union)
                AND d.vocabulary_id = ANY(%s)
                """
            )}
        ),
        xwalk_fwd AS (
            SELECT DISTINCT cr.concept_id_2 AS concept_id
            FROM concept_relationship cr
            JOIN concept c2 ON c2.concept_id = cr.concept_id_2
            WHERE cr.concept_id_1 IN (SELECT descendant_concept_id FROM closure)
            AND cr.relationship_id = ANY(%s)
            AND COALESCE(cr.invalid_reason,'') = ''
            AND c2.standard_concept = 'S'
            AND c2.domain_id = %s
        ),
        xwalk_rev AS (
            SELECT DISTINCT cr.concept_id_1 AS concept_id
            FROM concept_relationship cr
            JOIN concept c1 ON c1.concept_id = cr.concept_id_1
            WHERE cr.concept_id_2 IN (SELECT descendant_concept_id FROM closure)
            AND cr.relationship_id = ANY(%s)
            AND COALESCE(cr.invalid_reason,'') = ''
            AND c1.standard_concept = 'S'
            AND c1.domain_id = %s
        ),
        targets AS (
            SELECT descendant_concept_id AS concept_id FROM closure
            UNION
            SELECT concept_id FROM xwalk_fwd
            UNION
            SELECT concept_id FROM xwalk_rev
        )
        """
        params = (
            # seed
            domain, src_vocs, like, like,
            # seed_std
            std_vocs,
            # mapped_raw
            rel_whitelist,
            # mapped_std
            domain, std_vocs, like, like
        )
        if not (specific and not force_expand_descendants):
            params = params + (closure_vocabs,)

        params = params + (xwalk_rels, domain, xwalk_rels, domain)

        return cte, params
    
    def build_dynamic_sql(self, domain: str, base_term: str, specific: bool, conditions: Dict[str, Any]) -> tuple[str, tuple]:
        spec = self._get_domain_spec(domain)
        table = spec["table"]
        concept_field = spec["concept_field"]
        date_field = spec.get("date_field")

        if domain == "Death" or spec.get("concept_field") in (None, "",):
            table = spec["table"]            # "death"
            date_field = spec["date_field"]  # "death_date"

            ga_parts, ga_params = self._where_parts_from_conditions(conditions)
            where_sql_person = (" AND " + " AND ".join(ga_parts)) if ga_parts else ""

            anchor_val, _ = self.anchor_strategy.resolve(
                self.db_connector,
                domain="Death",
                base_term=base_term,
                where_sql_person=where_sql_person,
                where_params_person=tuple(ga_params),
            )
            anchor_cte = "anchor AS (SELECT %s::date AS d)"

            date_parts, date_params = self._date_where_parts("d", date_field, conditions, anchor_expr="(SELECT d FROM anchor)")

            where_parts = ga_parts + date_parts
            where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

            sql = f"""
            WITH {anchor_cte}
            SELECT COUNT(DISTINCT p.person_id) AS patient_count
            FROM person p
            JOIN {table} d ON d.person_id = p.person_id
            {where_sql};
            """.strip()

            params = (anchor_val,) + tuple(ga_params + date_params)
            return sql, params

        cte, cte_params = self._seed_and_closure_cte(domain, base_term, specific)
        where_clauses: list[str] = [f"{table}.{concept_field} IN (SELECT concept_id FROM targets)"]

        ga_parts, ga_params = self._where_parts_from_conditions(conditions)
        if ga_parts:
            where_clauses.extend(ga_parts)

        def _pos_int(v):
            try:
                v = int(v)
                return v if v >= 0 else None
            except Exception:
                return None

        need_anchor = any(_pos_int(conditions.get(k)) is not None for k in ("within_years", "within_months", "within_days"))

        anchor_cte = ""
        anchor_val = None
        if need_anchor:
            where_sql_person = (" AND " + " AND ".join(ga_parts)) if ga_parts else ""
            anchor_val, _ = self.anchor_strategy.resolve(
                self.db_connector,
                domain=domain,
                base_term=base_term,
                where_sql_person=where_sql_person,
                where_params_person=tuple(ga_params),
            )
            anchor_cte = ",\nanchor AS (SELECT %s::date AS d)\n"

        date_params: list[Any] = []
        if date_field:
            anchor_expr = "(SELECT d FROM anchor)" if need_anchor else "CURRENT_DATE"
            date_parts, abs_params = self._date_where_parts(table, date_field, conditions, anchor_expr=anchor_expr)
            if date_parts:
                where_clauses.extend(date_parts)
                date_params.extend(abs_params)

        select_sql = ("SELECT COUNT(DISTINCT p.person_id) AS patient_count"
                    if conditions.get("aggregation") == "count"
                    else "SELECT DISTINCT p.person_id, p.gender_concept_id, p.year_of_birth")

        sql = f"""
        {cte}{anchor_cte}
        {select_sql}
        FROM person p
        JOIN {table} ON p.person_id = {table}.person_id
        WHERE {' AND '.join(where_clauses)};
        """

        params: list[Any] = list(cte_params)
        if need_anchor:
            params.append(anchor_val)
        params.extend(ga_params)
        params.extend(date_params)

        return sql, tuple(params)
    
    def _where_parts_from_conditions(self, conditions: Dict[str, Any]) -> tuple[list, list]:
        parts, params = [], []
        gmap = self._gender_map()
        if (g := conditions.get("gender")) in gmap:
            parts.append("p.gender_concept_id = %s")
            params.append(gmap[g])
        if conditions.get("age") and conditions.get("age_comparison") in {">=", "<=", "=", ">", "<"}:
            parts.append(f"{self._age_sql_str()} {conditions['age_comparison']} %s")
            params.append(int(conditions["age"]))
        return parts, params
    
    def _date_where_parts(self, table_alias: str, date_field: str, conditions: Dict[str, Any], anchor_expr: str = "CURRENT_DATE") -> tuple[list, list]:
        parts: list[str] = []
        params: list[Any] = []

        if not date_field:
            return parts, params

        date_from = conditions.get("date_from")
        date_to = conditions.get("date_to")
        if date_from:
            parts.append(f"{table_alias}.{date_field} >= %s")
            params.append(str(date_from))
        if date_to:
            parts.append(f"{table_alias}.{date_field} <= %s")
            params.append(str(date_to))

        def _pos_int(v):
            try:
                v = int(v)
                return v if v >= 0 else None
            except Exception:
                return None

        wy = _pos_int(conditions.get("within_years"))
        wm = _pos_int(conditions.get("within_months"))
        wd = _pos_int(conditions.get("within_days"))

        if wy is not None:
            parts.append(f"{table_alias}.{date_field} >= {anchor_expr} - INTERVAL '{wy} years'")
        if wm is not None:
            parts.append(f"{table_alias}.{date_field} >= {anchor_expr} - INTERVAL '{wm} months'")
        if wd is not None:
            parts.append(f"{table_alias}.{date_field} >= {anchor_expr} - INTERVAL '{wd} days'")

        return parts, params
    
    def _closure_cte_for_condition(self, base_term: str) -> tuple[str, tuple]:
        like = f"%{base_term}%"
        cte = """
        WITH matched AS (
        SELECT DISTINCT c.concept_id
        FROM concept c
        LEFT JOIN concept_synonym s ON s.concept_id = c.concept_id
        WHERE c.standard_concept = 'S'
            AND c.domain_id = 'Condition'
            AND c.vocabulary_id = 'SNOMED'
            AND (c.concept_name ILIKE %s OR s.concept_synonym_name ILIKE %s)
        ),
        closure AS (
        SELECT m.concept_id AS descendant_concept_id FROM matched m
        UNION
        SELECT ca.descendant_concept_id
        FROM concept_ancestor ca
        WHERE ca.ancestor_concept_id IN (SELECT concept_id FROM matched)
        )
        """
        return cte, (like, like)
    
    def _age_sql_str(self) -> str:
        return """
        (
        CASE
            WHEN p.birth_datetime IS NOT NULL THEN
            EXTRACT(YEAR FROM AGE(CURRENT_DATE, p.birth_datetime))
            ELSE
            EXTRACT(YEAR FROM AGE(
                CURRENT_DATE,
                MAKE_DATE(p.year_of_birth, COALESCE(p.month_of_birth, 6), COALESCE(p.day_of_birth, 15))
            ))
        END
        )
        """

    def _gender_map(self) -> dict:
        return {"MALE": 8507, "FEMALE": 8532}
    
    def build_bp_observation_sql(self, conditions: Dict[str, Any]) -> tuple[str, tuple]:
        spec = self._get_domain_spec("Observation")
        table = spec["table"]                  # observation
        concept_field = spec["concept_field"]  # observation_concept_id
        date_field = spec["date_field"]        # observation_date

        ga_parts, ga_params = self._where_parts_from_conditions(conditions)
        where_sql_person = (" AND " + " AND ".join(ga_parts)) if ga_parts else ""

        anchor_val, _ = self.anchor_strategy.resolve(
            self.db_connector,
            domain="Observation",
            base_term="SBP_RULE",
            where_sql_person=where_sql_person,
            where_params_person=tuple(ga_params),
        )
        anchor_cte = "anchor AS (SELECT %s::date AS d)"

        date_parts, date_params = self._date_where_parts(table, date_field, conditions, anchor_expr="(SELECT d FROM anchor)")

        target_ids = (2617875, 2617876)

        where_parts = [f"{table}.{concept_field} = ANY(%s)"] + ga_parts + date_parts
        where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

        sql = f"""
        WITH {anchor_cte}
        SELECT COUNT(DISTINCT p.person_id) AS patient_count
        FROM person p
        JOIN {table} ON p.person_id = {table}.person_id
        {where_sql};
        """.strip()

        params = (anchor_val, target_ids) + tuple(ga_params + date_params)
        return sql, params

    def _hba1c_concept_ids_sql(self) -> str:
        return """
        SELECT c.concept_id
        FROM concept c
        WHERE c.vocabulary_id = 'LOINC'
        AND c.standard_concept = 'S'
        AND COALESCE(c.invalid_reason,'') = ''
        AND (
            lower(c.concept_name) LIKE '%hemoglobin a1c%'
            OR c.concept_code IN ('4548-4','17856-6','59261-8')
        )
        """

    def build_hba1c_summary_for_condition(self, base_term: str, specific: bool, conditions: dict):
        hba1c_cte, h_params = self._seed_and_closure_cte("Measurement", "HbA1c", False)
        cohort_cte, c_params = self._seed_and_closure_cte("Condition", base_term, specific)

        sql = f"""
        {hba1c_cte}
        {cohort_cte}
        , cohort AS (
            SELECT DISTINCT p.person_id
            FROM person p
            JOIN condition_occurrence co ON co.person_id = p.person_id
            WHERE co.condition_concept_id IN (SELECT concept_id FROM targets)  -- from cohort_cte
        ),
        lab AS (
            SELECT m.person_id,
                m.measurement_date,
                m.value_as_number AS value_num,
                u.concept_name AS unit
            FROM measurement m
            LEFT JOIN concept u ON u.concept_id = m.unit_concept_id
            WHERE m.measurement_concept_id IN (SELECT concept_id FROM targets) -- from hba1c_cte
            AND m.value_as_number IS NOT NULL
        ),
        latest AS (
            SELECT l.person_id, l.value_num, COALESCE(l.unit, '%') AS unit
            FROM (
                SELECT person_id, value_num, unit,
                    ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY measurement_date DESC) AS rn
                FROM lab
            ) l
            WHERE l.rn = 1
        ),
        latest_clamped AS (
            SELECT person_id,
                LEAST(20.0, GREATEST(2.0, value_num)) AS value_num,  -- 2~20% 하드 클램프
                unit
            FROM latest
        )
        SELECT
            COUNT(*) AS n,
            AVG(value_num) AS mean,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY value_num) AS median,
            PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY value_num) AS p05,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value_num) AS p25,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value_num) AS p75,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value_num) AS p95
        FROM latest_clamped lc
        JOIN cohort c ON c.person_id = lc.person_id;
        """
        return sql, [*h_params, *c_params]
    
    def _recency_counts(
        self,
        base_term: str,
        where_sql_person: str, where_params_person: tuple,
        where_sql_occ: str,    where_params_occ: tuple,
        anchor_val: Optional[str] = None
    ) -> dict:
        if not anchor_val:
            anchor_val, _ = self.anchor_strategy.resolve(
                self.db_connector,
                domain="Condition",
                base_term=base_term,
                where_sql_person=where_sql_person,
                where_params_person=where_params_person,
            )
            if not anchor_val:
                anchor_val = datetime.today().date().isoformat()

        cte, cte_params = self._seed_and_closure_cte("Condition", base_term, specific=False)

        occ_sql = f"""
        {cte},
        anchor AS (SELECT %s::date AS d)
        SELECT
        COUNT(DISTINCT CASE WHEN co.condition_start_date >= (SELECT d FROM anchor) - INTERVAL '365 days' THEN p.person_id END) AS n365,
        COUNT(DISTINCT CASE WHEN co.condition_start_date >= (SELECT d FROM anchor) - INTERVAL '180 days' THEN p.person_id END) AS n180,
        COUNT(DISTINCT CASE WHEN co.condition_start_date >= (SELECT d FROM anchor) - INTERVAL '90  days' THEN p.person_id END) AS n90,
        COUNT(DISTINCT CASE WHEN co.condition_start_date >= (SELECT d FROM anchor) - INTERVAL '30  days' THEN p.person_id END) AS n30
        FROM person p
        JOIN condition_occurrence co ON p.person_id = co.person_id
        WHERE co.condition_concept_id IN (SELECT descendant_concept_id FROM closure)
        {where_sql_occ};
        """
        df = self.db_connector.execute_query(
            occ_sql,
            cte_params + (anchor_val,) + where_params_occ
        )
        row = df.iloc[0].to_dict() if not df.empty else {}
        return {k: int(row.get(k, 0) or 0) for k in ("n365","n180","n90","n30")}
    
    def compute_extras_condition(self, base_term: str, specific: bool, conditions: Dict[str, Any], total_n: int) -> Dict[str, Any]:
        extras: Dict[str, Any] = {
            "pct_of_all": None,
            "pct_of_base_age_gender": None,
            "base_all": None,
            "base_age_gender": None,
            "recency": {},
            "age_buckets": [],
            "gender_breakdown": [],
            "top_concepts": []
        }

        cte, cte_params = self._seed_and_closure_cte("Condition", base_term, specific=specific)

        where_parts_ga, where_params_ga = self._where_parts_from_conditions(conditions)

        where_sql_person = (" AND " + " AND ".join(where_parts_ga)) if where_parts_ga else ""
        anchor_val, _ = self.anchor_strategy.resolve(
            self.db_connector,
            domain="Condition",
            base_term=base_term,
            where_sql_person=where_sql_person,
            where_params_person=tuple(where_params_ga),
        )
        anchor_cte = ",\nanchor AS (SELECT %s::date AS d)\n"

        date_field = self._get_domain_spec("Condition")["date_field"]
        date_parts, date_params = self._date_where_parts("co", date_field, conditions, anchor_expr="(SELECT d FROM anchor)")

        where_parts_all = where_parts_ga + date_parts
        where_sql_all = (" AND " + " AND ".join(where_parts_all)) if where_parts_all else ""
        where_params_all = tuple(where_params_ga + date_params)

        df_all = self.db_connector.execute_query("SELECT COUNT(*) AS n FROM person;", ())
        base_all = int(df_all.iloc[0]["n"]) if not df_all.empty else 0
        extras["base_all"] = base_all
        extras["pct_of_all"] = (total_n / base_all * 100.0) if base_all > 0 else None

        if where_parts_ga:
            sql_bg = f"SELECT COUNT(*) AS n FROM person p WHERE {' AND '.join(where_parts_ga)};"
            df_bg = self.db_connector.execute_query(sql_bg, tuple(where_params_ga))
            base_age_gender = int(df_bg.iloc[0]["n"]) if not df_bg.empty else 0
            extras["base_age_gender"] = base_age_gender
            extras["pct_of_base_age_gender"] = (total_n / base_age_gender * 100.0) if base_age_gender > 0 else None

        rec_raw = self._recency_counts(
            base_term,
            where_sql_person, tuple(where_params_ga),
            where_sql_all,    where_params_all,
            anchor_val=anchor_val
        )

        extras["recency"] = {}
        for k in ["n365","n180","n90","n30"]:
            v = int(rec_raw.get(k, 0) or 0)
            extras["recency"][k] = {"n": v, "pct": (v / total_n * 100.0) if total_n > 0 else None}

        age_sql = self._age_sql_str()
        use_senior = (conditions.get("age_comparison") == ">=" and int(conditions.get("age", 0)) >= 70)
        if use_senior:
            bucket_case = f"""
            CASE
            WHEN {age_sql} BETWEEN 70 AND 74 THEN '70-74'
            WHEN {age_sql} BETWEEN 75 AND 79 THEN '75-79'
            WHEN {age_sql} BETWEEN 80 AND 84 THEN '80-84'
            WHEN {age_sql} >= 85 THEN '85+'
            END
            """
            order_case = """
            CASE b.bucket
            WHEN '70-74' THEN 70
            WHEN '75-79' THEN 75
            WHEN '80-84' THEN 80
            WHEN '85+'  THEN 85
            ELSE 999
            END
            """
        else:
            bucket_case = f"""
            CASE
            WHEN {age_sql} < 18 THEN '0-17'
            WHEN {age_sql} BETWEEN 18 AND 39 THEN '18-39'
            WHEN {age_sql} BETWEEN 40 AND 64 THEN '40-64'
            WHEN {age_sql} BETWEEN 65 AND 74 THEN '65-74'
            WHEN {age_sql} >= 75 THEN '75+'
            END
            """
            order_case = """
            CASE b.bucket
            WHEN '0-17'  THEN 0
            WHEN '18-39' THEN 18
            WHEN '40-64' THEN 40
            WHEN '65-74' THEN 65
            WHEN '75+'   THEN 75
            ELSE 999
            END
            """

        age_bucket_sql = f"""
        {cte}{anchor_cte}
        SELECT b.bucket, COUNT(DISTINCT b.person_id) AS n
        FROM (
        SELECT p.person_id,
                {bucket_case} AS bucket
        FROM person p
        JOIN condition_occurrence co ON p.person_id = co.person_id
        WHERE co.condition_concept_id IN (SELECT descendant_concept_id FROM closure)
        {where_sql_all}
        ) b
        WHERE b.bucket IS NOT NULL
        GROUP BY b.bucket
        ORDER BY {order_case}, b.bucket;
        """
        df_ab = self.db_connector.execute_query(age_bucket_sql, cte_params + (anchor_val,) + where_params_all)
        extras["age_buckets"] = [
            {"bucket": str(r["bucket"]), "n": int(r["n"]), "pct": (int(r["n"]) / total_n * 100.0) if total_n > 0 else None}
            for _, r in df_ab.iterrows()
        ]

        if not conditions.get("gender"):
            gender_dist_sql = f"""
            {cte}{anchor_cte}
            SELECT
            CASE p.gender_concept_id
                WHEN 8507 THEN 'MALE'
                WHEN 8532 THEN 'FEMALE'
                ELSE 'OTHER'
            END AS gender,
            COUNT(DISTINCT p.person_id) AS n
            FROM person p
            JOIN condition_occurrence co ON p.person_id = co.person_id
            WHERE co.condition_concept_id IN (SELECT descendant_concept_id FROM closure)
            {where_sql_all}
            GROUP BY 1
            ORDER BY 1;
            """
            df_g = self.db_connector.execute_query(gender_dist_sql, cte_params + (anchor_val,) + where_params_all)
            extras["gender_breakdown"] = [
                {"gender": str(r["gender"]), "n": int(r["n"]), "pct": (int(r["n"]) / total_n * 100.0) if total_n > 0 else None}
                for _, r in df_g.iterrows()
            ]

        top_concepts_sql = f"""
        {cte}{anchor_cte}
        SELECT c.concept_name, COUNT(DISTINCT p.person_id) AS n
        FROM person p
        JOIN condition_occurrence co ON p.person_id = co.person_id
        JOIN concept c ON c.concept_id = co.condition_concept_id
        WHERE co.condition_concept_id IN (SELECT descendant_concept_id FROM closure)
        {where_sql_all}
        GROUP BY c.concept_name
        ORDER BY n DESC, c.concept_name
        LIMIT 10;
        """
        df_top = self.db_connector.execute_query(top_concepts_sql, cte_params + (anchor_val,) + where_params_all)
        extras["top_concepts"] = [
            {"concept_name": str(r["concept_name"]), "n": int(r["n"]), "pct": (int(r["n"]) / total_n * 100.0) if total_n > 0 else None}
            for _, r in df_top.iterrows()
        ]

        return extras

    def compute_extras_procedure(self, base_term: str, specific: bool, conditions: Dict[str, Any], total_n: int) -> Dict[str, Any]:
        extras: Dict[str, Any] = {
            "pct_of_all": None,
            "pct_of_base_age_gender": None,
            "base_all": None,
            "base_age_gender": None,
            "recency": {},
            "age_buckets": [],
            "gender_breakdown": [],
            "top_concepts": []
        }

        spec = self._get_domain_spec("Procedure")
        table = spec["table"]                         # procedure_occurrence
        concept_field = spec["concept_field"]         # procedure_concept_id
        date_field = spec["date_field"]               # procedure_date

        cte, cte_params = self._seed_and_closure_cte("Procedure", base_term, specific=specific)

        ga_parts, ga_params = self._where_parts_from_conditions(conditions)
        where_sql_person = (" AND " + " AND ".join(ga_parts)) if ga_parts else ""

        anchor_val, _ = self.anchor_strategy.resolve(
            self.db_connector,
            domain="Procedure",
            base_term=base_term,
            where_sql_person=where_sql_person,
            where_params_person=tuple(ga_params),
        )
        anchor_cte = ",\nanchor AS (SELECT %s::date AS d)\n"

        date_parts, date_params = self._date_where_parts(table, date_field, conditions, anchor_expr="(SELECT d FROM anchor)")
        where_parts_all = [f"{table}.{concept_field} IN (SELECT descendant_concept_id FROM closure)"] + ga_parts + date_parts
        where_sql_all = (" AND " + " AND ".join(where_parts_all)) if where_parts_all else ""
        where_params_all = tuple(ga_params + date_params)

        df_all = self.db_connector.execute_query("SELECT COUNT(*) AS n FROM person;", ())
        base_all = int(df_all.iloc[0]["n"]) if not df_all.empty else 0
        extras["base_all"] = base_all
        extras["pct_of_all"] = (total_n / base_all * 100.0) if base_all > 0 else None

        if ga_parts:
            df_bg = self.db_connector.execute_query(f"SELECT COUNT(*) AS n FROM person p WHERE {' AND '.join(ga_parts)};", tuple(ga_params))
            base_bg = int(df_bg.iloc[0]["n"]) if not df_bg.empty else 0
            extras["base_age_gender"] = base_bg
            extras["pct_of_base_age_gender"] = (total_n / base_bg * 100.0) if base_bg > 0 else None

        recency_sql = f"""
        {cte}{anchor_cte}
        SELECT
          COUNT(DISTINCT CASE WHEN {table}.{date_field} BETWEEN (SELECT d FROM anchor) - INTERVAL '365 days' AND (SELECT d FROM anchor) THEN p.person_id END) AS n365,
          COUNT(DISTINCT CASE WHEN {table}.{date_field} BETWEEN (SELECT d FROM anchor) - INTERVAL '180 days' AND (SELECT d FROM anchor) THEN p.person_id END) AS n180,
          COUNT(DISTINCT CASE WHEN {table}.{date_field} BETWEEN (SELECT d FROM anchor) - INTERVAL '90  days' AND (SELECT d FROM anchor) THEN p.person_id END) AS n90,
          COUNT(DISTINCT CASE WHEN {table}.{date_field} BETWEEN (SELECT d FROM anchor) - INTERVAL '30  days' AND (SELECT d FROM anchor) THEN p.person_id END) AS n30
        FROM person p
        JOIN {table} ON p.person_id = {table}.person_id
        WHERE {table}.{concept_field} IN (SELECT descendant_concept_id FROM closure)
        {where_sql_all};
        """
        df_rec = self.db_connector.execute_query(recency_sql, cte_params + (anchor_val,) + where_params_all)
        row = df_rec.iloc[0].to_dict() if not df_rec.empty else {}
        for k in ("n365","n180","n90","n30"):
            v = int(row.get(k, 0) or 0)
            extras["recency"][k] = {"n": v, "pct": (v / total_n * 100.0) if total_n > 0 else None}

        age_sql = self._age_sql_str()
        age_bucket_sql = f"""
        {cte}{anchor_cte}
        SELECT b.bucket, COUNT(DISTINCT b.person_id) AS n
        FROM (
          SELECT p.person_id,
                 CASE
                   WHEN {age_sql} < 18 THEN '0-17'
                   WHEN {age_sql} BETWEEN 18 AND 39 THEN '18-39'
                   WHEN {age_sql} BETWEEN 40 AND 64 THEN '40-64'
                   WHEN {age_sql} BETWEEN 65 AND 74 THEN '65-74'
                   WHEN {age_sql} >= 75 THEN '75+'
                 END AS bucket
          FROM person p
          JOIN {table} ON p.person_id = {table}.person_id
          WHERE {table}.{concept_field} IN (SELECT descendant_concept_id FROM closure)
          {where_sql_all}
        ) b
        WHERE b.bucket IS NOT NULL
        GROUP BY b.bucket
        ORDER BY CASE b.bucket WHEN '0-17' THEN 0 WHEN '18-39' THEN 18 WHEN '40-64' THEN 40 WHEN '65-74' THEN 65 WHEN '75+' THEN 75 ELSE 999 END, b.bucket;
        """
        df_ab = self.db_connector.execute_query(age_bucket_sql, cte_params + (anchor_val,) + where_params_all)
        extras["age_buckets"] = [
            {"bucket": str(r["bucket"]), "n": int(r["n"]), "pct": (int(r["n"]) / total_n * 100.0) if total_n > 0 else None}
            for _, r in df_ab.iterrows()
        ]

        if not conditions.get("gender"):
            gender_sql = f"""
            {cte}{anchor_cte}
            SELECT CASE p.gender_concept_id WHEN 8507 THEN 'MALE' WHEN 8532 THEN 'FEMALE' ELSE 'OTHER' END AS gender,
                   COUNT(DISTINCT p.person_id) AS n
            FROM person p
            JOIN {table} ON p.person_id = {table}.person_id
            WHERE {table}.{concept_field} IN (SELECT descendant_concept_id FROM closure)
            {where_sql_all}
            GROUP BY 1 ORDER BY 1;
            """
            df_g = self.db_connector.execute_query(gender_sql, cte_params + (anchor_val,) + where_params_all)
            extras["gender_breakdown"] = [
                {"gender": str(r["gender"]), "n": int(r["n"]), "pct": (int(r["n"]) / total_n * 100.0) if total_n > 0 else None}
                for _, r in df_g.iterrows()
            ]

        top_sql = f"""
        {cte}{anchor_cte}
        SELECT c.concept_name, COUNT(DISTINCT p.person_id) AS n
        FROM person p
        JOIN {table} ON p.person_id = {table}.person_id
        JOIN concept c ON c.concept_id = {table}.{concept_field}
        WHERE {table}.{concept_field} IN (SELECT descendant_concept_id FROM closure)
        {where_sql_all}
        GROUP BY c.concept_name
        ORDER BY n DESC, c.concept_name
        LIMIT 10;
        """
        df_top = self.db_connector.execute_query(top_sql, cte_params + (anchor_val,) + where_params_all)
        extras["top_concepts"] = [
            {"concept_name": str(r["concept_name"]), "n": int(r["n"]), "pct": (int(r["n"]) / total_n * 100.0) if total_n > 0 else None}
            for _, r in df_top.iterrows()
        ]

        return extras
    
    def compute_extras_measurement_numeric(self, base_term: str, specific: bool, conditions: Dict[str, Any], total_n: int) -> Dict[str, Any]:
        extras = {
            "pct_of_all": None,
            "pct_of_base_age_gender": None,
            "base_all": None,
            "base_age_gender": None,
            "recency": {},
            "age_buckets": [],
            "gender_breakdown": [],
            "top_concepts": [],
            "value": {
                "unit_candidates": [],
                "summary": {},
                "histogram": []
            }
        }

        spec = self._get_domain_spec("Measurement")
        table = spec["table"]                         # measurement
        concept_field = spec["concept_field"]         # measurement_concept_id
        date_field = spec["date_field"]               # measurement_date

        cte, cte_params = self._seed_and_closure_cte("Measurement", base_term, specific=specific)

        ga_parts, ga_params = self._where_parts_from_conditions(conditions)
        where_sql_person = (" AND " + " AND ".join(ga_parts)) if ga_parts else ""

        anchor_val, _ = self.anchor_strategy.resolve(
            self.db_connector,
            domain="Measurement",
            base_term=base_term,
            where_sql_person=where_sql_person,
            where_params_person=tuple(ga_params),
        )
        anchor_cte = ",\nanchor AS (SELECT %s::date AS d)\n"

        date_parts, date_params = self._date_where_parts(table, date_field, conditions, anchor_expr="(SELECT d FROM anchor)")

        val_parts, val_params = [], []
        def _add(cond_sql, *ps):
            val_parts.append(cond_sql); val_params.extend(ps)

        if conditions.get("value_ge") is not None:
            _add(f"{table}.value_as_number >= %s", float(conditions["value_ge"]))
        if conditions.get("value_gt") is not None:
            _add(f"{table}.value_as_number > %s", float(conditions["value_gt"]))
        if conditions.get("value_le") is not None:
            _add(f"{table}.value_as_number <= %s", float(conditions["value_le"]))
        if conditions.get("value_lt") is not None:
            _add(f"{table}.value_as_number < %s", float(conditions["value_lt"]))
        if conditions.get("value_between"):
            low, high = conditions["value_between"][0], conditions["value_between"][1]
            _add(f"{table}.value_as_number BETWEEN %s AND %s", float(low), float(high))

        unit_join = ""
        unit_parts, unit_params = [], []
        if conditions.get("unit_concept_id") is not None:
            unit_parts.append(f"{table}.unit_concept_id = %s")
            unit_params.append(int(conditions["unit_concept_id"]))
        elif conditions.get("unit_like"):
            unit_join = " LEFT JOIN concept u ON u.concept_id = m.unit_concept_id "
            unit_parts.append("u.concept_name ILIKE %s")
            unit_params.append(f"%{conditions['unit_like']}%")

        where_parts_all = [
            f"{table}.{concept_field} IN (SELECT descendant_concept_id FROM closure)"
        ] + ga_parts + date_parts + val_parts + unit_parts
        where_sql_all = (" AND " + " AND ".join(where_parts_all)) if where_parts_all else ""
        where_params_all = tuple(ga_params + date_params + val_params + unit_params)

        df_all = self.db_connector.execute_query("SELECT COUNT(*) AS n FROM person;", ())
        base_all = int(df_all.iloc[0]["n"]) if not df_all.empty else 0
        extras["base_all"] = base_all
        extras["pct_of_all"] = (total_n / base_all * 100.0) if base_all > 0 else None

        if ga_parts:
            df_bg = self.db_connector.execute_query(
                f"SELECT COUNT(*) AS n FROM person p WHERE {' AND '.join(ga_parts)};", tuple(ga_params)
            )
            base_bg = int(df_bg.iloc[0]["n"]) if not df_bg.empty else 0
            extras["base_age_gender"] = base_bg
            extras["pct_of_base_age_gender"] = (total_n / base_bg * 100.0) if base_bg > 0 else None

        strategy = (conditions.get("value_strategy") or "latest").lower()
        # 허용: latest, first, min, max, mean
        if strategy not in {"latest", "first", "min", "max", "mean"}:
            strategy = "latest"

        outlier = conditions.get("outlier")
        iqr_k = None
        if isinstance(outlier, str):
            o = outlier.lower()
            if o == "iqr15":
                iqr_k = 1.5
            elif o == "iqr30":
                iqr_k = 3.0

        rep_sql_strategy = ""
        if strategy in {"latest", "first"}:
            order = "DESC" if strategy == "latest" else "ASC"
            rep_sql_strategy = f"""
              rep_rows AS (
                SELECT person_id, value_as_number AS val, {table}.{date_field} AS dt
                FROM (
                  SELECT p.person_id, m.value_as_number, m.{date_field},
                         ROW_NUMBER() OVER (PARTITION BY p.person_id ORDER BY m.{date_field} {order}, m.measurement_id {order}) AS rn
                  FROM person p
                  JOIN {table} m ON p.person_id = m.person_id
                  WHERE m.{concept_field} IN (SELECT descendant_concept_id FROM closure)
                  {where_sql_all}
                ) x
                WHERE rn = 1
              )
            """
        elif strategy in {"min", "max", "mean"}:
            agg = {"min": "MIN", "max": "MAX", "mean": "AVG"}[strategy]
            rep_sql_strategy = f"""
              rep_rows AS (
                SELECT p.person_id, {agg}(m.value_as_number) AS val, NULL::date AS dt
                FROM person p
                JOIN {table} m ON p.person_id = m.person_id
                WHERE m.{concept_field} IN (SELECT descendant_concept_id FROM closure)
                {where_sql_all}
                GROUP BY p.person_id
              )
            """

        iqr_sql = ""       # iqr CTE
        rep_final = "rep_rows"
        if iqr_k is not None:
            iqr_sql = f"""
              ,iqr AS (
                SELECT
                  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY val) AS q1,
                  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY val) AS q3
                FROM rep_rows
              ),
              rep_rows_filtered AS (
                SELECT r.*
                FROM rep_rows r, iqr
                WHERE r.val IS NOT NULL
                  AND r.val >= (iqr.q1 - {iqr_k} * (iqr.q3 - iqr.q1))
                  AND r.val <= (iqr.q3 + {iqr_k} * (iqr.q3 - iqr.q1))
              )
            """
            rep_final = "rep_rows_filtered"

        recency_sql = f"""
        {cte}{anchor_cte}
        WITH
          base AS (
            SELECT p.person_id, m.{date_field} AS dt
            FROM person p
            JOIN {table} m ON p.person_id = m.person_id
            WHERE m.{concept_field} IN (SELECT descendant_concept_id FROM closure)
            {where_sql_all}
          )
        SELECT
          COUNT(DISTINCT CASE WHEN dt BETWEEN (SELECT d FROM anchor) - INTERVAL '365 days' AND (SELECT d FROM anchor) THEN person_id END) AS n365,
          COUNT(DISTINCT CASE WHEN dt BETWEEN (SELECT d FROM anchor) - INTERVAL '180 days' AND (SELECT d FROM anchor) THEN person_id END) AS n180,
          COUNT(DISTINCT CASE WHEN dt BETWEEN (SELECT d FROM anchor) - INTERVAL '90  days' AND (SELECT d FROM anchor) THEN person_id END)  AS n90,
          COUNT(DISTINCT CASE WHEN dt BETWEEN (SELECT d FROM anchor) - INTERVAL '30  days' AND (SELECT d FROM anchor) THEN person_id END)  AS n30
        FROM base;
        """
        df_rec = self.db_connector.execute_query(recency_sql, cte_params + (anchor_val,) + where_params_all)
        row = df_rec.iloc[0].to_dict() if not df_rec.empty else {}
        for k in ("n365","n180","n90","n30"):
            v = int(row.get(k, 0) or 0)
            extras["recency"][k] = {"n": v, "pct": (v / total_n * 100.0) if total_n > 0 else None}

        summary_sql = f"""
        {cte}{anchor_cte}
        WITH
          {rep_sql_strategy}
          {iqr_sql}
        SELECT
          MIN(val) AS v_min,
          PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY val) AS v_p50,
          AVG(val) AS v_mean,
          PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY val) AS v_p95,
          PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY val) AS v_p99,
          PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY val) AS v_p05
        FROM {rep_final};
        """
        df_sum = self.db_connector.execute_query(summary_sql, cte_params + (anchor_val,))
        v05 = v95 = None
        if not df_sum.empty:
            r = df_sum.iloc[0].to_dict()
            extras["value"]["summary"] = {
                "min": float(r["v_min"]) if r["v_min"] is not None else None,
                "median": float(r["v_p50"]) if r["v_p50"] is not None else None,
                "mean": float(r["v_mean"]) if r["v_mean"] is not None else None,
                "p95": float(r["v_p95"]) if r["v_p95"] is not None else None,
                "p99": float(r["v_p99"]) if r["v_p99"] is not None else None,
            }
            v05 = float(r["v_p05"]) if r["v_p05"] is not None else None
            v95 = float(r["v_p95"]) if r["v_p95"] is not None else None

        qlow = (base_term or "").lower()
        is_hba1c = ("hba1c" in qlow) or ("glycated hemoglobin" in qlow) or ("4548-4" in qlow)
        is_sbp   = ("sbp" in qlow) or ("systolic blood pressure" in qlow) or ("8480-6" in qlow)

        if is_hba1c:
            hist_sql = f"""
            {cte}{anchor_cte}
            WITH
              {rep_sql_strategy}
              {iqr_sql}
            SELECT bucket, COUNT(*) AS n
            FROM (
              SELECT CASE
                  WHEN val < 5.7 THEN '<5.7'
                  WHEN val >= 5.7  AND val < 6.5 THEN '5.7–6.4'
                  WHEN val >= 6.5  AND val < 8.0 THEN '6.5–7.9'
                  WHEN val >= 8.0  AND val < 10.0 THEN '8.0–9.9'
                  WHEN val >= 10.0 THEN '≥10.0'
                END AS bucket
              FROM {rep_final}
            ) t
            WHERE bucket IS NOT NULL
            GROUP BY bucket
            ORDER BY MIN(
              CASE bucket
                WHEN '<5.7' THEN 1
                WHEN '5.7–6.4' THEN 2
                WHEN '6.5–7.9' THEN 3
                WHEN '8.0–9.9' THEN 4
                WHEN '≥10.0'  THEN 5
                ELSE 999
              END
            );
            """
            df_hist = self.db_connector.execute_query(hist_sql, cte_params + (anchor_val,))
            extras["value"]["histogram"] = [
                {"bucket": str(r["bucket"]), "n": int(r["n"]), "pct": (int(r["n"]) / total_n * 100.0) if total_n > 0 else None}
                for _, r in df_hist.iterrows()
            ]
        elif is_sbp:
            hist_sql = f"""
            {cte}{anchor_cte}
            WITH
              {rep_sql_strategy}
              {iqr_sql}
            SELECT bucket, COUNT(*) AS n
            FROM (
              SELECT CASE
                  WHEN val < 120 THEN '<120'
                  WHEN val >= 120 AND val < 130 THEN '120–129'
                  WHEN val >= 130 AND val < 140 THEN '130–139'
                  WHEN val >= 140 AND val < 160 THEN '140–159'
                  WHEN val >= 160 THEN '≥160'
                END AS bucket
              FROM {rep_final}
            ) t
            WHERE bucket IS NOT NULL
            GROUP BY bucket
            ORDER BY MIN(
              CASE bucket
                WHEN '<120'    THEN 1
                WHEN '120–129' THEN 2
                WHEN '130–139' THEN 3
                WHEN '140–159' THEN 4
                WHEN '≥160'    THEN 5
                ELSE 999
              END
            );
            """
            df_hist = self.db_connector.execute_query(hist_sql, cte_params + (anchor_val,))
            extras["value"]["histogram"] = [
                {"bucket": str(r["bucket"]), "n": int(r["n"]), "pct": (int(r["n"]) / total_n * 100.0) if total_n > 0 else None}
                for _, r in df_hist.iterrows()
            ]
        else:
            if v05 is not None and v95 is not None and v95 > v05:
                bin_count = 10
                hist_sql = f"""
                {cte}{anchor_cte}
                WITH
                  {rep_sql_strategy}
                  {iqr_sql}
                SELECT
                  MIN(val) AS bin_low,
                  MAX(val) AS bin_high,
                  COUNT(*) AS n,
                  WIDTH_BUCKET(val, %s, %s, {bin_count}) AS b
                FROM {rep_final}
                WHERE val BETWEEN %s AND %s
                GROUP BY b
                ORDER BY b;
                """
                df_hist = self.db_connector.execute_query(
                    hist_sql,
                    cte_params + (anchor_val,) + (v05, v95, v05, v95)
                )
                extras["value"]["histogram"] = [
                    {
                        "bin_low": float(r["bin_low"]) if r["bin_low"] is not None else None,
                        "bin_high": float(r["bin_high"]) if r["bin_high"] is not None else None,
                        "n": int(r["n"]),
                        "pct": (int(r["n"]) / total_n * 100.0) if total_n > 0 else None
                    }
                    for _, r in df_hist.iterrows()
                ]

        top_sql = f"""
        {cte}{anchor_cte}
        SELECT c.concept_name, COUNT(DISTINCT p.person_id) AS n
        FROM person p
        JOIN {table} m ON p.person_id = m.person_id
        JOIN concept c ON c.concept_id = m.{concept_field}
        WHERE m.{concept_field} IN (SELECT descendant_concept_id FROM closure)
        {where_sql_all}
        GROUP BY c.concept_name
        ORDER BY n DESC, c.concept_name
        LIMIT 10;
        """
        df_top = self.db_connector.execute_query(top_sql, cte_params + (anchor_val,) + where_params_all)
        extras["top_concepts"] = [
            {"concept_name": str(r["concept_name"]), "n": int(r["n"]), "pct": (int(r["n"]) / total_n * 100.0) if total_n > 0 else None}
            for _, r in df_top.iterrows()
        ]

        unit_top_sql = f"""
        {cte}{anchor_cte}
        SELECT COALESCE(u.concept_name, 'UNIT_CONCEPT_ID='||m.unit_concept_id::text) AS unit_label,
               COUNT(*) AS cnt
        FROM person p
        JOIN {table} m ON p.person_id = m.person_id
        LEFT JOIN concept u ON u.concept_id = m.unit_concept_id
        WHERE m.{concept_field} IN (SELECT descendant_concept_id FROM closure)
        {where_sql_all}
        GROUP BY unit_label
        ORDER BY cnt DESC
        LIMIT 5;
        """
        df_unit = self.db_connector.execute_query(unit_top_sql, cte_params + (anchor_val,) + where_params_all)
        extras["value"]["unit_candidates"] = [
            {"unit": str(r["unit_label"]), "count": int(r["cnt"])} for _, r in df_unit.iterrows()
        ]

        age_sql = self._age_sql_str()
        age_bucket_sql = f"""
        {cte}{anchor_cte}
        WITH
          {rep_sql_strategy}
          {iqr_sql}
        SELECT b.bucket, COUNT(*) AS n
        FROM (
          SELECT r.person_id,
                 CASE
                   WHEN {age_sql} < 18 THEN '0-17'
                   WHEN {age_sql} BETWEEN 18 AND 39 THEN '18-39'
                   WHEN {age_sql} BETWEEN 40 AND 64 THEN '40-64'
                   WHEN {age_sql} BETWEEN 65 AND 74 THEN '65-74'
                   WHEN {age_sql} >= 75 THEN '75+'
                 END AS bucket
          FROM {rep_final} r
          JOIN person p ON p.person_id = r.person_id
        ) b
        WHERE b.bucket IS NOT NULL
        GROUP BY b.bucket
        ORDER BY CASE b.bucket WHEN '0-17' THEN 0 WHEN '18-39' THEN 18 WHEN '40-64' THEN 40 WHEN '65-74' THEN 65 WHEN '75+' THEN 75 ELSE 999 END, b.bucket;
        """
        df_ab = self.db_connector.execute_query(age_bucket_sql, cte_params + (anchor_val,))
        extras["age_buckets"] = [
            {"bucket": str(r["bucket"]), "n": int(r["n"]), "pct": (int(r["n"]) / total_n * 100.0) if total_n > 0 else None}
            for _, r in df_ab.iterrows()
        ]

        if not conditions.get("gender"):
            gender_sql = f"""
            {cte}{anchor_cte}
            WITH
              {rep_sql_strategy}
              {iqr_sql}
            SELECT g.gender, COUNT(*) AS n
            FROM (
              SELECT r.person_id,
                     CASE p.gender_concept_id
                       WHEN 8507 THEN 'MALE'
                       WHEN 8532 THEN 'FEMALE'
                       ELSE 'OTHER'
                     END AS gender
              FROM {rep_final} r
              JOIN person p ON p.person_id = r.person_id
            ) g
            GROUP BY g.gender
            ORDER BY g.gender;
            """
            df_g = self.db_connector.execute_query(gender_sql, cte_params + (anchor_val,))
            extras["gender_breakdown"] = [
                {"gender": str(r["gender"]), "n": int(r["n"]), "pct": (int(r["n"]) / total_n * 100.0) if total_n > 0 else None}
                for _, r in df_g.iterrows()
            ]

        return extras

    def _compute_extras_generic(self, domain: str, base_term: str, specific: bool, conditions: Dict[str, Any], total_n: int) -> Dict[str, Any]:
        extras: Dict[str, Any] = {
            "pct_of_all": None,
            "pct_of_base_age_gender": None,
            "base_all": None,
            "base_age_gender": None,
            "recency": {},
            "age_buckets": [],
            "gender_breakdown": [],
            "top_concepts": []
        }

        spec = self._get_domain_spec(domain)
        table = spec["table"]
        concept_field = spec["concept_field"]
        date_field = spec.get("date_field")
        if not concept_field:
            return extras

        cte, cte_params = self._seed_and_closure_cte(domain, base_term, specific=specific)

        ga_parts, ga_params = self._where_parts_from_conditions(conditions)
        where_sql_person = (" AND " + " AND ".join(ga_parts)) if ga_parts else ""

        anchor_val, _ = self.anchor_strategy.resolve(
            self.db_connector,
            domain=domain,
            base_term=base_term,
            where_sql_person=where_sql_person,
            where_params_person=tuple(ga_params),
        )
        anchor_cte = ",\nanchor AS (SELECT %s::date AS d)\n"

        date_parts, date_params = self._date_where_parts(table, date_field, conditions, anchor_expr="(SELECT d FROM anchor)")
        where_parts_all = [f"{table}.{concept_field} IN (SELECT descendant_concept_id FROM closure)"] + ga_parts + date_parts
        where_sql_all = (" AND " + " AND ".join(where_parts_all)) if where_parts_all else ""
        where_params_all = tuple(ga_params + date_params)

        df_all = self.db_connector.execute_query("SELECT COUNT(*) AS n FROM person;", ())
        base_all = int(df_all.iloc[0]["n"]) if not df_all.empty else 0
        extras["base_all"] = base_all
        extras["pct_of_all"] = (total_n / base_all * 100.0) if base_all > 0 else None

        if ga_parts:
            df_bg = self.db_connector.execute_query(
                f"SELECT COUNT(*) AS n FROM person p WHERE {' AND '.join(ga_parts)};",
                tuple(ga_params)
            )
            base_bg = int(df_bg.iloc[0]["n"]) if not df_bg.empty else 0
            extras["base_age_gender"] = base_bg
            extras["pct_of_base_age_gender"] = (total_n / base_bg * 100.0) if base_bg > 0 else None

        recency_sql = f"""
        {cte}{anchor_cte}
        SELECT
          COUNT(DISTINCT CASE WHEN {table}.{date_field} BETWEEN (SELECT d FROM anchor) - INTERVAL '365 days' AND (SELECT d FROM anchor) THEN p.person_id END) AS n365,
          COUNT(DISTINCT CASE WHEN {table}.{date_field} BETWEEN (SELECT d FROM anchor) - INTERVAL '180 days' AND (SELECT d FROM anchor) THEN p.person_id END) AS n180,
          COUNT(DISTINCT CASE WHEN {table}.{date_field} BETWEEN (SELECT d FROM anchor) - INTERVAL '90  days' AND (SELECT d FROM anchor) THEN p.person_id END)  AS n90,
          COUNT(DISTINCT CASE WHEN {table}.{date_field} BETWEEN (SELECT d FROM anchor) - INTERVAL '30  days' AND (SELECT d FROM anchor) THEN p.person_id END)  AS n30
        FROM person p
        JOIN {table} ON p.person_id = {table}.person_id
        WHERE {table}.{concept_field} IN (SELECT descendant_concept_id FROM closure)
        {where_sql_all};
        """
        df_rec = self.db_connector.execute_query(recency_sql, cte_params + (anchor_val,) + where_params_all)
        row = df_rec.iloc[0].to_dict() if not df_rec.empty else {}
        for k in ("n365","n180","n90","n30"):
            v = int(row.get(k, 0) or 0)
            extras["recency"][k] = {"n": v, "pct": (v / total_n * 100.0) if total_n > 0 else None}

        age_sql = self._age_sql_str()
        age_bucket_sql = f"""
        {cte}{anchor_cte}
        SELECT b.bucket, COUNT(DISTINCT b.person_id) AS n
        FROM (
          SELECT p.person_id,
                 CASE
                   WHEN {age_sql} < 18 THEN '0-17'
                   WHEN {age_sql} BETWEEN 18 AND 39 THEN '18-39'
                   WHEN {age_sql} BETWEEN 40 AND 64 THEN '40-64'
                   WHEN {age_sql} BETWEEN 65 AND 74 THEN '65-74'
                   WHEN {age_sql} >= 75 THEN '75+'
                 END AS bucket
          FROM person p
          JOIN {table} ON p.person_id = {table}.person_id
          WHERE {table}.{concept_field} IN (SELECT descendant_concept_id FROM closure)
          {where_sql_all}
        ) b
        WHERE b.bucket IS NOT NULL
        GROUP BY b.bucket
        ORDER BY CASE b.bucket WHEN '0-17' THEN 0 WHEN '18-39' THEN 18 WHEN '40-64' THEN 40 WHEN '65-74' THEN 65 WHEN '75+' THEN 75 ELSE 999 END, b.bucket;
        """
        df_ab = self.db_connector.execute_query(age_bucket_sql, cte_params + (anchor_val,) + where_params_all)
        extras["age_buckets"] = [
            {"bucket": str(r["bucket"]), "n": int(r["n"]), "pct": (int(r["n"]) / total_n * 100.0) if total_n > 0 else None}
            for _, r in df_ab.iterrows()
        ]

        if not conditions.get("gender"):
            gender_sql = f"""
            {cte}{anchor_cte}
            SELECT CASE p.gender_concept_id WHEN 8507 THEN 'MALE' WHEN 8532 THEN 'FEMALE' ELSE 'OTHER' END AS gender,
                   COUNT(DISTINCT p.person_id) AS n
            FROM person p
            JOIN {table} ON p.person_id = {table}.person_id
            WHERE {table}.{concept_field} IN (SELECT descendant_concept_id FROM closure)
            {where_sql_all}
            GROUP BY 1 ORDER BY 1;
            """
            df_g = self.db_connector.execute_query(gender_sql, cte_params + (anchor_val,) + where_params_all)
            extras["gender_breakdown"] = [
                {"gender": str(r["gender"]), "n": int(r["n"]), "pct": (int(r["n"]) / total_n * 100.0) if total_n > 0 else None}
                for _, r in df_g.iterrows()
            ]

        top_sql = f"""
        {cte}{anchor_cte}
        SELECT c.concept_name, COUNT(DISTINCT p.person_id) AS n
        FROM person p
        JOIN {table} ON p.person_id = {table}.person_id
        JOIN concept c ON c.concept_id = {table}.{concept_field}
        WHERE {table}.{concept_field} IN (SELECT descendant_concept_id FROM closure)
        {where_sql_all}
        GROUP BY c.concept_name
        ORDER BY n DESC, c.concept_name
        LIMIT 10;
        """
        df_top = self.db_connector.execute_query(top_sql, cte_params + (anchor_val,) + where_params_all)
        extras["top_concepts"] = [
            {"concept_name": str(r["concept_name"]), "n": int(r["n"]), "pct": (int(r["n"]) / total_n * 100.0) if total_n > 0 else None}
            for _, r in df_top.iterrows()
        ]

        return extras
    
    def _build_death_count_sql(self, conditions: dict) -> tuple[str, list]:
        date_from = conditions.get("date_from")
        date_to   = conditions.get("date_to")

        wh = []
        params = []
        if date_from:
            wh.append("d.death_date >= %s")
            params.append(date_from)
        if date_to:
            wh.append("d.death_date <= %s")
            params.append(date_to)

        where_sql = ("WHERE " + " AND ".join(wh)) if wh else ""
        sql = f"""
            SELECT COUNT(DISTINCT d.person_id) AS patient_count
            FROM death d
            {where_sql};
        """
        return sql, params
    
    def build_drug_topn_by_patients(self, base_term: str, specific: bool, topn: int, conditions: Dict[str, Any]) -> tuple[str, tuple]:
        spec = self._get_domain_spec("Drug")
        table = spec["table"]                  # drug_exposure
        concept_field = spec["concept_field"]  # drug_concept_id
        date_field = spec["date_field"]        # drug_exposure_start_date

        cte, cte_params = self._seed_and_closure_cte("Drug", base_term, specific=specific)

        ga_parts, ga_params = self._where_parts_from_conditions(conditions)
        where_sql_person = (" AND " + " AND ".join(ga_parts)) if ga_parts else ""

        anchor_val, _ = self.anchor_strategy.resolve(
            self.db_connector,
            domain="Drug",
            base_term=base_term,
            where_sql_person=where_sql_person,
            where_params_person=tuple(ga_params),
        )
        anchor_cte = ",\nanchor AS (SELECT %s::date AS d)\n"

        date_parts, date_params = self._date_where_parts(table, date_field, conditions, anchor_expr="(SELECT d FROM anchor)")

        where_parts_all = [f"{table}.{concept_field} IN (SELECT concept_id FROM targets)"] + ga_parts + date_parts
        where_sql_all = (" AND " + " AND ".join(where_parts_all)) if where_parts_all else ""
        where_params_all = tuple(ga_params + date_params)

        sql = f"""
        {cte}{anchor_cte}
        SELECT
            c.concept_id,
            c.concept_name,
            COUNT(DISTINCT p.person_id) AS patients
        FROM person p
        JOIN {table} de
        ON p.person_id = de.person_id
        JOIN concept c
        ON c.concept_id = de.{concept_field}
        WHERE {where_sql_all}
        GROUP BY c.concept_id, c.concept_name
        ORDER BY patients DESC, c.concept_name
        LIMIT %s;
        """.strip()

        params = tuple(cte_params) + (anchor_val,) + where_params_all + (int(topn),)
        return sql, params

    def compute_extras_drug(self, base_term: str, specific: bool, conditions: Dict[str, Any], total_n: int) -> Dict[str, Any]:
        return self._compute_extras_generic("Drug", base_term, specific, conditions, total_n)

    def compute_extras_measurement(self, base_term: str, specific: bool, conditions: Dict[str, Any], total_n: int) -> Dict[str, Any]:
        return self._compute_extras_generic("Measurement", base_term, specific, conditions, total_n)

    def compute_extras_observation(self, base_term: str, specific: bool, conditions: Dict[str, Any], total_n: int) -> Dict[str, Any]:
        return self._compute_extras_generic("Observation", base_term, specific, conditions, total_n)

    def compute_extras_device(self, base_term: str, specific: bool, conditions: Dict[str, Any], total_n: int) -> Dict[str, Any]:
        return self._compute_extras_generic("Device", base_term, specific, conditions, total_n)
    
    def _get_domain_spec(self, domain: str) -> dict:
        return self.domain_spec[domain].copy()

    @staticmethod
    def _transplant_variants(_term: str) -> list[str]:
                t = (_term or "").strip()
                variants = {t}
                low = t.lower()

                if "transplant" in low or "transplantation" in low:
                    organ = re.sub(r"(?i)\b(transplantation|transplant)\b", "", t).strip()
                    if not organ:
                        organ = "liver"
                    variants.update({
                        f"{organ} transplantation",
                        f"Transplantation of {organ}",
                        f"{organ} transplant"
                    })
                    if organ.lower() == "liver":
                        variants.update({"Hepatic transplantation"})

                if "liver" in low:
                    variants.update({"liver transplantation", "transplantation of liver", "hepatic transplantation", "간 이식"})

                out = [v for v in variants if v]
                return [t] + sorted([v for v in out if v != t])
    
    def interpret_result(self, result_df, user_query: str, sql_query: str) -> str:
        try:
            import pandas as pd
        except Exception:
            pd = None

        if result_df is None or getattr(result_df, "empty", True):
            return "조건에 맞는 레코드 없음."

        n_rows = len(result_df)
        cols = list(result_df.columns)

        try:
            preview_df = result_df.head(50)
            preview_md = preview_df.to_markdown(index=False)
        except Exception:
            try:
                preview_md = "\n".join(
                    [", ".join(map(str, cols))] +
                    [", ".join("" if x is None else str(x) for x in row) for row in result_df.head(50).values]
                )
            except Exception:
                preview_md = "(미리보기 생성 실패)"

        lines = []
        lines.append("결과 표")
        lines.append(f"- 행 수: {n_rows:,}")
        lines.append(f"- 열: {', '.join(map(str, cols))}")
        lines.append("")
        lines.append(preview_md)

        return "\n".join(lines)