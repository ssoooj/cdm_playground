from __future__ import annotations

import os
import re
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Set

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from main import CDMRagSystem
except Exception as e:
    CDMRagSystem = None  # type: ignore
    print("[server] WARN: main.CDMRagSystem import failed:", e)

try:
    from db_connector import PostgresConnector
except Exception as e:
    PostgresConnector = None  # type: ignore
    print("[server] WARN: db_connector import failed:", e)

try:
    from terminology_mapper_default import TerminologyMapper
except Exception:
    TerminologyMapper = None  # type: ignore

try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None  # type: ignore
    print("[server] WARN: openai package not installed:", e)

try:
    from neo4j import GraphDatabase
    from neo4j.graph import Node, Relationship
except Exception as e:
    GraphDatabase = None  # type: ignore
    print("[server] WARN: neo4j driver not installed:", e)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
log = logging.getLogger("server")

class AppConfig(BaseModel):
    postgresql: Optional[dict] = None
    llm: dict
    neo4j: Optional[dict] = None

def load_config() -> AppConfig:
    path = os.environ.get("CONFIG_PATH", "config.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.yaml not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(**raw)

class Clients:
    def __init__(self, cfg: AppConfig):
        if not OpenAI:
            raise RuntimeError("openai package is required. pip install openai>=1.0")
        provider = (cfg.llm or {}).get("provider", "lmstudio")
        provider_cfg: dict = cfg.llm.get(provider, {}) if cfg.llm else {}
        api_base = provider_cfg.get("api_base")
        api_key = provider_cfg.get("api_key", "not-needed")
        model = provider_cfg.get("model")
        params = provider_cfg.get("parameters", {}) or {}
        if not api_base or not model:
            raise RuntimeError("LLM config(api_base, model) missing. Check config.yaml.")
        self.llm_client = OpenAI(base_url=api_base, api_key=api_key)
        self.llm_model = model
        self.llm_params = {
            "temperature": params.get("temperature", 0.0),
            "max_tokens": params.get("max_tokens", 1024),
            "top_p": params.get("top_p", 1.0),
        }

        self.db = None
        if PostgresConnector and cfg.postgresql:
            try:
                self.db = PostgresConnector(cfg.postgresql)
                log.info("Postgres connected")
            except Exception as e:
                log.warning("Postgres connect failed. Will answer via LLM only: %s", e)

        self.neo4j = None
        neo_cfg = (cfg.neo4j or {}) if cfg.neo4j else {}
        enabled = str(os.getenv("NEO4J_ENABLED", str(neo_cfg.get("enabled", False)))).lower() == "true"

        if GraphDatabase and enabled and all(neo_cfg.get(k) for k in ("uri", "user", "password")):
            try:
                self.neo4j = GraphDatabase.driver(
                    neo_cfg["uri"], auth=(neo_cfg["user"], neo_cfg["password"])
                )
                with self.neo4j.session() as s:
                    s.run("RETURN 1").single()
                log.info("Neo4j connected (startup)")
            except Exception as e:
                self.neo4j = None
                log.warning("Neo4j connect failed on startup: %s", e)
        else:
            log.info("Neo4j disabled (startup)")

        self.term_mapper = None
        if TerminologyMapper:
            try:
                self.term_mapper = TerminologyMapper(self.db, self.llm_client, self.llm_model)  # type: ignore[arg-type]
            except Exception as e:
                log.warning("TerminologyMapper init failed: %s", e)

    def llm_chat(self, messages: List[Dict[str, str]]) -> str:
        resp = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=self.llm_params.get("temperature", 0.0),
            max_tokens=self.llm_params.get("max_tokens", 1024),
            top_p=self.llm_params.get("top_p", 1.0),
            stream=False,
        )
        return resp.choices[0].message.content or ""

    def try_simple_omop_answer(self, query: str) -> Optional[str]:
        if not self.db:
            return None
        gender_kw = None
        if re.search(r"남자|남성", query):
            gender_kw = "Male"
        elif re.search(r"여자|여성", query):
            gender_kw = "Female"
        m = re.search(r"(\d+)\s*~\s*(\d+)\s*세", query)
        if m:
            min_age = int(m.group(1)); max_age = int(m.group(2))
        else:
            m2 = re.search(r"(\d+)\s*대", query)
            if m2:
                base = int(m2.group(1)); min_age, max_age = base, base + 9
            else:
                m3 = re.search(r"(\d+)\s*세\s*이상", query)
                if m3:
                    min_age, max_age = int(m3.group(1)), 150
                else:
                    m4 = re.search(r"(\d+)\s*세\s*이하", query)
                    if m4:
                        min_age, max_age = 0, int(m4.group(1))
                    else:
                        min_age = max_age = None
        where = []
        params: list[Any] = []
        if min_age is not None and max_age is not None:
            where.append("(EXTRACT(YEAR FROM CURRENT_DATE) - p.year_of_birth) BETWEEN %s AND %s")
            params += [min_age, max_age]
        if gender_kw:
            where.append("p.gender_concept_id = (SELECT concept_id FROM concept WHERE lower(concept_name) = lower(%s) LIMIT 1)")
            params.append(gender_kw)
        if not where:
            return None
        sql = "SELECT COUNT(*) AS patient_count FROM person p WHERE " + " AND ".join(where) + ";"
        try:
            df = self.db.execute_query(sql, tuple(params))
            if not df.empty:
                cnt = int(df.iloc[0]["patient_count"])  # type: ignore[index]
                detail = []
                if min_age is not None and max_age is not None:
                    detail.append(f"나이 {min_age}~{max_age}세")
                if gender_kw:
                    detail.append(gender_kw)
                return f"{', '.join(detail)} 환자 수는 **{cnt}명**입니다."
        except Exception as e:
            log.warning("OMOP sample query failed: %s", e)
        return None

app = FastAPI(title="Pastel Chat API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_FILE = Path(os.environ.get("FRONTEND_FILE", "khdp_frontend.html")).resolve()

@app.get("/", response_class=HTMLResponse)
def root_page():
    if FRONTEND_FILE.exists():
        return HTMLResponse(FRONTEND_FILE.read_text(encoding="utf-8"))
    return HTMLResponse(
        """
        <html><body style="font-family: ui-sans-serif; padding:24px">
        <h2>Pastel Chat API</h2>
        <p>Frontend file not found.</p>
        <ul>
          <li>Place <code>pastel_chat_frontend_index.html</code> in project root, or</li>
          <li>Set env <code>FRONTEND_FILE=/abs/path/index.html</code></li>
        </ul>
        <p>API: <code>POST /ask</code></p>
        </body></html>
        """,
        status_code=200,
    )

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/graph/llm")
def graph_llm(req: GraphReq):
    if not CLIENTS:
        raise HTTPException(status_code=503, detail="server initializing")
    term = (req.term or "").strip()
    if not term:
        raise HTTPException(status_code=400, detail="term is required")

    sys = "You are a clinical knowledge graph generator."
    user = f'''Generate a concise JSON graph for the disease "{term}".
    Return ONLY valid JSON with keys "nodes" and "edges".
    - nodes: [{{ "id": "string", "label":"string", "group":"concept|diagnosis|symptom|drug|test|complication" }}]
    - edges: [{{ "source":"nodeId", "target":"nodeId", "label":"relation" }}]
    Keep it small (<= 30 nodes). Use English labels.'''

    try:
        txt = CLIENTS.llm_chat([
            {"role":"system","content":sys},
            {"role":"user","content":user}
        ])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    try:
        m = re.search(r"\{[\s\S]*\}\s*$", txt)
        j = json.loads(m.group(0) if m else txt)
    except Exception:
        raise HTTPException(status_code=500, detail="LLM JSON parse failed")

    def norm_group(g: Optional[str]) -> str:
        g = (g or "").lower()
        return g if g in {"drug","person","concept","diagnosis","symptom","test","complication"} else "concept"

    nodes_in = j.get("nodes") or []
    edges_in = j.get("edges") or []
    nodes = []
    node_ids = set()
    for n in nodes_in:
        nid = str(n.get("id") or n.get("label") or f"n_{len(nodes)}")
        node_ids.add(nid)
        nodes.append({
            "data": {
                "id": nid,
                "label": str(n.get("label") or n.get("id") or "Node"),
                "group": norm_group(n.get("group")),
            }
        })

    edges = []
    eid = 0
    for e in edges_in:
        s = str(e.get("source") or "")
        t = str(e.get("target") or "")
        if not s or not t or s not in node_ids or t not in node_ids:
            continue
        edges.append({
            "data": {
                "id": f"e_{eid}",
                "source": s,
                "target": t,
                "label": str(e.get("label") or "")
            }
        })
        eid += 1

    return {"nodes": nodes, "edges": edges}

class Neo4jConfig(BaseModel):
    uri: str
    user: str
    password: str

class Neo4jConnInline(BaseModel):
    uri: str
    user: str
    password: str

class JourneyReq(BaseModel):
    term: Optional[str] = None 
    icd: Optional[str] = None
    age_min: Optional[int] = None
    max_patients: int = 100
    limit: int = 800
    neo4j: Optional[Neo4jConnInline] = None

class GraphReq(BaseModel):
    term: str
    limit: int = 400
    seed_limit: int = 50
    neo4j: Optional[Neo4jConnInline] = None


@app.post("/config/neo4j")
def config_neo4j(cfg: Neo4jConfig):
    if not GraphDatabase:
        raise HTTPException(status_code=400, detail="neo4j driver not installed")
    try:
        driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
        with driver.session() as s:
            s.run("RETURN 1").single()
        if CLIENTS and getattr(CLIENTS, "neo4j", None):
            try:
                CLIENTS.neo4j.close()  # type: ignore
            except Exception:
                pass
        if CLIENTS:
            CLIENTS.neo4j = driver
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Neo4j connect error: {e}")

TEXT_PROP_MAP: Dict[str, List[str]] = {}
GRAPH_PROPS_CSV = os.environ.get(
    "GRAPH_PROPS_CSV",
    "/Users/sohyeon/Downloads/250618_property_전체.csv"
)

DISPLAY_KEY_PRIORITY = [
    "name", "label", "title", "description", "desc",
    "drug", "item", "text", "icd_code", "code"
]

def _parse_labels_cell(cell: str) -> List[str]:
    s = (cell or "").strip()
    if not s:
        return []
    s = s.strip("[]").strip()
    if not s:
        return []
    parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
    return [p for p in parts if p]

def _build_text_prop_map_from_csv(path: str) -> Dict[str, List[str]]:
    if not Path(path).exists():
        log.warning("Graph props CSV not found: %s", path)
        return {}
    mp: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels = _parse_labels_cell(row.get("nodeLabels", ""))
            key = (row.get("key") or "").strip()
            dtype = (row.get("dataType") or "").strip().upper()
            if not labels or not key:
                continue
            if "STRING" not in dtype:
                continue
            for lb in labels:
                mp.setdefault(lb, [])
                if key not in mp[lb]:
                    mp[lb].append(key)
    DROP = {"subject_id","hadm_id","icustay_id","stay_id",
            "charttime","starttime","endtime","intime","outtime"}
    for lb, keys in list(mp.items()):
        mp[lb] = [k for k in keys if k not in DROP]
    return mp

def _build_text_prop_map_from_db(driver) -> Dict[str, List[str]]:
    mp: Dict[str, List[str]] = {}
    try:
        with driver.session() as s:
            recs = s.run("CALL db.schema.nodeTypeProperties()").data()
            for r in recs:
                labels = r.get("nodeLabels") or r.get("nodeType") or []
                pname = r.get("propertyName")
                ptypes = r.get("propertyTypes") or []
                if not labels or not pname:
                    continue
                if any("STRING" in str(t).upper() for t in ptypes):
                    for lb in labels:
                        lb = lb.strip(":")
                        mp.setdefault(lb, [])
                        if pname not in mp[lb]:
                            mp[lb].append(pname)
    except Exception:
        mp = {}

    if not mp:
        try:
            with driver.session() as s:
                labels = [r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label").data()]
                for lb in labels:
                    ks = s.run(f"""
                        MATCH (n:`{lb}`)
                        WITH n LIMIT 50
                        WITH collect(n) AS ns
                        UNWIND ns AS n
                        UNWIND keys(n) AS k
                        RETURN collect(DISTINCT k) AS ks
                    """).single()
                    if ks and ks["ks"]:
                        mp[lb] = list(ks["ks"])
        except Exception:
            mp = {}

    DROP = {"subject_id","hadm_id","icustay_id","stay_id",
            "charttime","starttime","endtime","intime","outtime"}
    for lb, keys in list(mp.items()):
        mp[lb] = [k for k in keys if k not in DROP]

    return mp

def _pick_display_text(props: Dict[str, Any]) -> str:
    for k in DISPLAY_KEY_PRIORITY:
        if k in props and props[k] not in (None, ""):
            try:
                return str(props[k])
            except Exception:
                pass
    for k, v in props.items():
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in ("subject_id","hadm_id","icustay_id","stay_id","row_id"):
        if k in props:
            return f"{k}:{props[k]}"
    return "Node"

def _group_from_labels(labels: List[str]) -> str:
    L = [l.lower() for l in labels]
    if "patient" in L: return "person"
    if "admission" in L: return "admission"
    if "icustay" in L or "icu" in L: return "icu"
    if "diagnosis" in L or "condition" in L: return "diagnosis"
    if any(x in L for x in ["inputevent","outputevent","chartevent","procedureevent","ingredientevent"]): return "event"
    if "item" in L: return "concept"
    if "transfer" in L: return "transfer"
    return labels[0] if labels else "node"

def _cy_node_id(node: Node) -> str:
    labels = list(node.labels)
    prefix = labels[0] if labels else "N"
    return f"{prefix}:{node.id}"

def ensure_text_prop_map(driver) -> None:
    global TEXT_PROP_MAP
    if TEXT_PROP_MAP:
        return
    path = Path(GRAPH_PROPS_CSV) if GRAPH_PROPS_CSV else None
    try:
        if path and path.exists() and path.stat().st_size <= 5_000_000:
            TEXT_PROP_MAP = _build_text_prop_map_from_csv(str(path))
            if TEXT_PROP_MAP:
                log.info("Graph text-property map loaded from CSV for %d labels", len(TEXT_PROP_MAP))
                return
        else:
            if path and path.exists():
                log.warning("Graph props CSV too large; falling back to DB schema.")
    except Exception as e:
        log.warning("CSV load failed, falling back to DB schema: %s", e)

    try:
        if CLIENTS and getattr(CLIENTS, "neo4j", None):
            TEXT_PROP_MAP = _build_text_prop_map_from_db(CLIENTS.neo4j)  # type: ignore[arg-type]
            if TEXT_PROP_MAP:
                log.info("Graph text-property map loaded from DB for %d labels", len(TEXT_PROP_MAP))
                return
    except Exception as e:
        log.warning("DB schema load failed: %s", e)

    TEXT_PROP_MAP = {}
    log.warning("Using generic ANY(keys(n)) search for graph (no text-property map).")


class ChatMessage(BaseModel):
    role: str = Field(..., description="'system'|'user'|'assistant'")
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    answer: str
    meta: Dict[str, Any] = {}

import pandas as pd
from collections.abc import Mapping

MAX_TABLE_ROWS = 500  # 필요 시 조정

_TABLE_LIKE_KEYS = {
    "df","dataframe","table_df","table","rows","records","result_df","results_df","data","table_md"
}

_TABLE_ROW_RE = re.compile(r"^\|\s*[^|]+\s*(\|\s*[^|]+\s*)+\|\s*$", re.MULTILINE)
_HTML_TABLE_RE = re.compile(r"<table\b[\s\S]*?</table>", re.IGNORECASE)
SAFE_KEYS: Set[str] = {"meta", "analytics", "metrics", "_ui"}

def _strip_heavy_tables_in_dict(obj: dict, _seen: Optional[Set[int]] = None, _depth: int = 0) -> dict:
    MAX_DEPTH = 6
    if _seen is None:
        _seen = set()

    # 순환/깊이 가드
    oid = id(obj)
    if oid in _seen:
        return {"_ui": {"show_table": False}}
    _seen.add(oid)
    if _depth > MAX_DEPTH:
        return {"_ui": {"show_table": False}}

    obj = dict(obj)  # shallow copy
    obj.setdefault("_ui", {})
    obj["_ui"]["show_table"] = False

    def _process_value(k: str, v: Any):
        if k in SAFE_KEYS:
            return v
        if isinstance(v, pd.DataFrame):
            return None
        if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], Mapping)):
            return v if len(v) <= MAX_TABLE_ROWS else None
        if isinstance(v, Mapping):
            return _strip_heavy_tables_in_dict(dict(v), _seen=_seen, _depth=_depth + 1)
        return v

    for k in list(obj.keys()):
        if k in _TABLE_LIKE_KEYS:
            try:
                obj[k] = _process_value(k, obj[k])
            except RecursionError:
                obj[k] = None

    for k, v in list(obj.items()):
        try:
            obj[k] = _process_value(k, v)
        except RecursionError:
            obj[k] = None

    ans = obj.get("answer")
    if isinstance(ans, str):
        lines = ans.splitlines()
        kept = []
        in_table = False
        for ln in lines:
            if _TABLE_ROW_RE.match(ln):
                in_table = True
                continue
            else:
                if in_table:
                    in_table = False
                kept.append(ln)
        ans_clean = "\n".join(kept).strip()
        # HTML <table>…</table> 제거
        ans_clean = _HTML_TABLE_RE.sub("", ans_clean).strip()
        obj["answer"] = ans_clean

    return obj

def _strip_heavy_tables_any(res: Any) -> Any:
    try:
        if isinstance(res, dict):
            return _strip_heavy_tables_in_dict(res)
        if isinstance(res, str):
            lines = res.splitlines()
            kept = []
            for ln in lines:
                if _TABLE_ROW_RE.match(ln):
                    continue
                kept.append(ln)
            out = "\n".join(kept).strip()
            # HTML <table>…</table> 제거
            out = _HTML_TABLE_RE.sub("", out).strip()
            return out
        return res
    except RecursionError:
        if isinstance(res, dict):
            ans = res.get("answer") if isinstance(res.get("answer"), str) else ""
            ans = _HTML_TABLE_RE.sub("", ans or "")
            # 마크다운 표 라인도 제거
            kept = [ln for ln in (ans.splitlines() if ans else []) if not _TABLE_ROW_RE.match(ln)]
            return {"answer": "\n".join(kept).strip(), "_ui": {"show_table": False}}
        if isinstance(res, str):
            kept = [ln for ln in res.splitlines() if not _TABLE_ROW_RE.match(ln)]
            return _HTML_TABLE_RE.sub("", "\n".join(kept)).strip()
        return res

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask", response_model=ChatResponse)
def ask(req: ChatRequest):
    if not CLIENTS:
        raise HTTPException(status_code=503, detail="server initializing")
    messages = [m.model_dump() for m in req.messages]
    if not messages:
        raise HTTPException(status_code=400, detail="messages is empty")
    last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), None)

    if last_user and RAG:
        try:
            rag_answer = RAG.process_query(last_user)
            rag_answer = _strip_heavy_tables_any(rag_answer)

            if rag_answer:
                if isinstance(rag_answer, dict):
                    ans = rag_answer.get("answer", "")
                    meta = rag_answer.get("meta", {})
                    meta = {"source": "rag", **meta}
                    return ChatResponse(answer=ans, meta=meta)
                if not str(rag_answer).startswith("오류"):
                    return ChatResponse(answer=str(rag_answer), meta={"source": "rag"})
        except Exception as e:
            log.warning("RAG failed; fallback to LLM: %s", e)

    if last_user:
        omop = CLIENTS.try_simple_omop_answer(last_user)
        if omop:
            return ChatResponse(answer=omop, meta={"source": "omop"})

    sys_prompt = {
        "role": "system",
        "content": "You are a helpful assistant. Reply in the user's language. Keep answers concise and clear."
    }
    msgs = [sys_prompt] + [m for m in messages if m.get("role") != "system"]
    try:
        answer = CLIENTS.llm_chat(msgs)
        return ChatResponse(answer=answer, meta={"source": "llm", "model": CLIENTS.llm_model})
    except Exception as e:
        log.exception("LLM call failed")
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

@app.post("/graph/journey")
def graph_journey(req: JourneyReq):
    if not GraphDatabase:
        raise HTTPException(status_code=400, detail="neo4j driver not installed")

    driver = None
    close_after = False
    if req.neo4j:
        driver = GraphDatabase.driver(req.neo4j.uri, auth=(req.neo4j.user, req.neo4j.password))
        close_after = True
    elif CLIENTS and getattr(CLIENTS, "neo4j", None):
        driver = CLIENTS.neo4j
    else:
        log.info("graph_journey: Neo4j not configured -> return empty graph")
        return {"nodes": [], "edges": []}

    term = (req.term or "").strip()
    icd_raw = (req.icd or "").strip()
    icd_norm = icd_raw.upper().replace(".", "").replace(" ", "")
    use_icd = bool(icd_norm)
    use_term = bool(term)
    if not use_icd and not use_term:
        raise HTTPException(status_code=400, detail="Provide 'icd' or 'term'")

    icd_field = "diag.icd_code"

    cy = f"""
    WITH $icd_norm AS icd_norm, $use_icd AS use_icd, $use_term AS use_term, toLower($term) AS term_lc

    // 1) Diagnosis 히트
    MATCH (diag:Diagnosis)
    WHERE
    ($use_icd = true  AND toUpper(replace(replace(coalesce(diag.icd_code,''),'.',''),' ','')) STARTS WITH $icd_norm)
    OR
    ($use_term = true AND ANY(k IN keys(diag)
            WHERE toLower(toString(diag[k])) CONTAINS toLower($term)))

    // 2) Admission/Patient 연결
    MATCH (a:Admission)-[:HAS_DIAGNOSIS]-(diag)
    MATCH (p:Patient)-[:HAS_ADMISSION]-(a)

    // 3) 나이 계산 (있는 것만) → lenient
    WITH p, $age_min AS age_min
    WITH p, age_min,
        CASE
        WHEN p['age'] IS NOT NULL THEN toInteger(p['age'])
        WHEN p['dob'] IS NOT NULL THEN date().year - toInteger(substring(toString(p['dob']),0,4))
        WHEN p['year_of_birth'] IS NOT NULL THEN date().year - toInteger(p['year_of_birth'])
        WHEN p['yob'] IS NOT NULL THEN date().year - toInteger(p['yob'])
        ELSE NULL
        END AS age_val
    WHERE (age_min IS NULL OR age_val IS NULL OR age_val >= age_min)

    WITH DISTINCT p
    ORDER BY rand()
    LIMIT $max_patients

    // 4) 주변 hop 확장
    MATCH (p)-[r_adm:HAS_ADMISSION]-(a:Admission)
    OPTIONAL MATCH (a)-[r_diag:HAS_DIAGNOSIS]-(diag:Diagnosis)
    OPTIONAL MATCH (a)-[r_icu:HAS_ICU_STAY]-(icu:ICUStay)
    OPTIONAL MATCH (icu)-[r_event:HAS_CHART_EVENT|HAS_INPUT_EVENT|HAS_OUTPUT_EVENT|HAS_PROCEDURE_EVENT]-(event)
    OPTIONAL MATCH (event)-[r_def:HAS_DEFINITION]-(item:Item)
    RETURN p, r_adm, a, r_diag, diag, r_icu, icu, r_event, event, r_def, item
    LIMIT $limit
    """.strip()

    params = {
        "term": term,
        "icd_norm": icd_norm,
        "use_icd": use_icd,
        "use_term": use_term,
        "age_min": req.age_min if req.age_min is not None else None,
        "max_patients": req.max_patients,
        "limit": req.limit,
    }

    nodes, edges = {}, {}

    def nid(n):
        labels = list(n.labels)
        pref = labels[0] if labels else "N"
        return f"{pref}:{n.id}"

    def add_node(n):
        i = nid(n)
        if i in nodes:
            return
        labels = list(n.labels)
        props = dict(n)
        nodes[i] = {
            "data": {
                "id": i,
                "label": _pick_display_text(props),
                "group": _group_from_labels(labels),
                "_labels": labels,
            }
        }

    def add_edge(a, r, b):
        ai = nid(a); bi = nid(b)
        et = getattr(r, "type", "RELATED")
        ei = f"{ai}-{et}->{bi}"
        if ei in edges:
            return
        edges[ei] = {"data": {"id": ei, "source": ai, "target": bi, "label": et}}

    try:
        with driver.session() as s:
            log.info("journey params=%s", {k: v for k, v in params.items()})
            hit_cnt = s.run("""
                WITH $icd_norm AS icd_norm, $use_icd AS use_icd, $use_term AS use_term, toLower($term) AS term_lc
                MATCH (diag:Diagnosis)
                WHERE
                (use_icd = true  AND toUpper(replace(replace(coalesce(diag.icd_code,''),'.',''),' ','')) STARTS WITH icd_norm)
                OR
                (use_term = true AND ANY(k IN keys(diag)
                        WHERE toLower(toString(diag[k])) CONTAINS term_lc))
                RETURN count(diag) AS n
            """, parameters=params).single()["n"]
            log.info("journey diagnosis hits=%s", hit_cnt)

            recs = s.run(cy, parameters=params)  # ★ 핵심 수정 ★
            for row in recs:
                for key in ["p","a","diag","icu","event","item"]:
                    n = row.get(key)
                    if n is not None:
                        add_node(n)
                for a_key, r_key, b_key in [
                    ("p","r_adm","a"),
                    ("a","r_diag","diag"),
                    ("a","r_icu","icu"),
                    ("icu","r_event","event"),
                    ("event","r_def","item"),
                ]:
                    a = row.get(a_key); r = row.get(r_key); b = row.get(b_key)
                    if a is not None and r is not None and b is not None:
                        add_edge(a, r, b)
    except Exception as e:
        if close_after and driver:
            driver.close()
        raise HTTPException(status_code=500, detail=f"Cypher error: {e}") from e
    finally:
        if close_after and driver:
            driver.close()

    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


CLIENTS: Optional[Clients] = None
RAG: Optional[Any] = None  # CDMRagSystem | None

@app.on_event("startup")
def _startup():
    global CLIENTS, RAG, TEXT_PROP_MAP
    cfg = load_config()
    CLIENTS = Clients(cfg)

    if CDMRagSystem:
        try:
            RAG = CDMRagSystem(config_path=os.environ.get("CONFIG_PATH", "config.yaml"))
            if getattr(RAG, "db_connector", None):
                try:
                    RAG.db_connector.connect()
                    log.info("RAG DB connected")
                except Exception as e:
                    log.warning("RAG DB connect failed: %s", e)
        except Exception as e:
            RAG = None
            log.warning("CDMRagSystem init failed: %s", e)

    log.info("Server up. LLM=%s, PG=%s, RAG=%s, Neo4j=%s",
             CLIENTS.llm_model,
             bool(CLIENTS.db),
             bool(RAG),
             bool(getattr(CLIENTS, "neo4j", None)))


@app.on_event("shutdown")
def _shutdown():
    try:
        if RAG and getattr(RAG, "db_connector", None):
            RAG.db_connector.close()
            log.info("RAG DB closed")
    except Exception as e:
        log.warning("shutdown error: %s", e)
    try:
        if CLIENTS and getattr(CLIENTS, "neo4j", None):
            CLIENTS.neo4j.close()  # type: ignore[call-arg]
            log.info("Neo4j closed")
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
