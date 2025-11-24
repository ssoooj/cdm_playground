import pandas as pd
import logging
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Optional, Any, Tuple, Dict
from domain_spec import DOMAIN_SPEC

logger = logging.getLogger(__name__)

class AnchorStrategy:
    def resolve(
        self,
        db: Any,
        *,
        domain: str,
        base_term: str,
        where_sql_person: str = "",
        where_params_person: tuple = (),
        fallback_today: bool = True,
    ) -> tuple[str, str]:
        raise NotImplementedError


class CurrentDateAnchor(AnchorStrategy):
    def resolve(self, db: Any, **kwargs) -> tuple[str, str]:
        today = date.today().isoformat()
        return (today, today)


class FixedDateAnchor(AnchorStrategy):
    def __init__(self, fixed_date: str):
        self.fixed_date = fixed_date

    def resolve(self, db: Any, **kwargs) -> tuple[str, str]:
        return (self.fixed_date, self.fixed_date)


class MaxDateInDataAnchor(AnchorStrategy):
    def resolve(
        self,
        db: Any,
        *,
        domain: str,
        base_term: str,
        where_sql_person: str = "",
        where_params_person: tuple = (),
        fallback_today: bool = True,
    ) -> tuple[str, str]:
        if domain not in DOMAIN_SPEC:
            today = date.today().isoformat()
            return (today, today)
        table = DOMAIN_SPEC[domain]["table"]
        date_field = DOMAIN_SPEC[domain]["date_field"]
        sql = f"SELECT MAX({date_field}) AS max_dt FROM {table};"
        df = db.execute_query(sql, ())
        if df.empty or pd.isna(df.iloc[0]["max_dt"]):
            today = date.today().isoformat()
            return (today, today)
        mx = str(df.iloc[0]["max_dt"])
        return (mx, mx)


def load_anchor_strategy(cfg: Dict) -> AnchorStrategy:
    anchor = (cfg or {}).get("anchor") or {}
    typ = anchor.get("type") or cfg.get("anchor_strategy") or "current_date"

    if typ == "current_date":
        return CurrentDateAnchor()
    if typ == "fixed_date":
        as_of = anchor.get("as_of") or cfg.get("anchor_fixed_date") or "2020-01-01"
        return FixedDateAnchor(as_of)
    if typ == "max_in_data":
        return MaxDateInDataAnchor()
    return CurrentDateAnchor()

def _calc_window_for_targets(
    db: Any,
    *,
    domain: str,
    targets_cte_sql: Optional[str] = None,
    targets_cte_params: tuple = (),
) -> tuple[Optional[date], Optional[date]]:
    if domain not in DOMAIN_SPEC:
        return (None, None)

    table = DOMAIN_SPEC[domain]["table"]
    date_col = DOMAIN_SPEC[domain]["date_field"]
    concept_col = DOMAIN_SPEC[domain]["concept_field"]

    if not table or not date_col or not concept_col:
        return (None, None)

    if targets_cte_sql:
        sql = f"""
        {targets_cte_sql}
        SELECT MIN({date_col}) AS min_dt, MAX({date_col}) AS max_dt
        FROM {table}
        WHERE {concept_col} IN (SELECT concept_id FROM targets);
        """
        df = db.execute_query(sql, targets_cte_params)
    else:
        sql = f"SELECT MIN({date_col}) AS min_dt, MAX({date_col}) AS max_dt FROM {table};"
        df = db.execute_query(sql, ())


    if df.empty:
        return (None, None)

    min_dt = df.iloc[0]["min_dt"]
    max_dt = df.iloc[0]["max_dt"]
    min_d = date.fromisoformat(str(min_dt)) if pd.notna(min_dt) else None
    max_d = date.fromisoformat(str(max_dt)) if pd.notna(max_dt) else None
    return (min_d, max_d)

@dataclass
class AutoClampAnchor:
    def __init__(self, base, lookback_days: int = 365*5):
        self.base = base
        self.lookback_days = lookback_days

    def resolve_with_clamp(
        self,
        db,
        domain: str,
        base_term: str,
        user_date_from: Optional[str],
        targets_cte_sql: Optional[str] = None,
        targets_cte_params: Optional[list] = None,
    ) -> Tuple[str, Optional[str]]:

        anchor_date_str, _ = self.base.resolve(db, domain=domain, base_term=base_term)

        if user_date_from:
            return anchor_date_str, user_date_from

        eff_from = None
        try:
            if targets_cte_sql:
                spec = self.base.domain_spec[domain]
                tbl = spec["table"]
                date_fld = spec["date_field"]

                sql = f"""
                {targets_cte_sql}
                SELECT MIN({date_fld}) AS min_dt, MAX({date_fld}) AS max_dt
                FROM {tbl}
                WHERE {spec['concept_field']} IN (SELECT concept_id FROM targets);
                """
                df = db.execute_query(sql, targets_cte_params or [])
                min_dt = df.iloc[0]["min_dt"] if not df.empty else None
                max_dt = df.iloc[0]["max_dt"] if not df.empty else None

                if max_dt:
                    anchor_date = date.fromisoformat(str(max_dt))
                    eff_from = (anchor_date - timedelta(days=self.lookback_days)).isoformat()
        except Exception as e:
            logger.warning(f"AutoClampAnchor 범위 계산 실패: {e}")

        if not eff_from:
            try:
                anchor = date.fromisoformat(anchor_date_str)
                eff_from = (anchor - timedelta(days=self.lookback_days)).isoformat()
            except Exception:
                eff_from = None

        return anchor_date_str, eff_from