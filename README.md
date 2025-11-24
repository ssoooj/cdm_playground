# cdm_playground

CDM Playground - Cohort Extraction  
Version: 1.5v  
Institution: Seoul National University Hospital – Healthcare AI Research Institute

## Overview

**CDM Playground** is a platform designed for cohort extraction, terminology mapping, and analytics on OMOP-CDM databases.  
It integrates LLMs with structured SQL query generation, enabling users to explore clinical data through natural language questions.

The project includes:

- A FastAPI backend serving the extraction and analytics logic  
- A web frontend for chat-style interaction and data visualization  
- Database connections for OMOP CDM (PostgreSQL)  
- Optional integration with Neo4j for graph-based exploration  

## Architecture

```text
├── main.py                        # Core logic (CDMRagSystem)
├── server.py                      # FastAPI server setup & API endpoints
├── db_connector.py                # PostgreSQL connection manager
├── sql_builder.py                 # Dynamic SQL query generator
├── anchor_strategy.py             # Date anchoring logic (e.g., current, fixed, max-in-data)
├── domain_spec.py                 # Domain metadata (Condition, Drug, Procedure, etc.)
├── terminology_mapper_default.py  # Terminology mapping via LLM
├── khdp_frontend.html             # Web-based user interface
├── config.yaml                    # Configuration (DB, LLM, Neo4j)
└── README.md                      # Documentation
````

## Features

### 1. Natural Language Query to SQL

Users can enter questions like:

> “How many male patients over 60 were diagnosed with hypertension in the past two years?”

The system uses LLM-based parsing and OMOP mappings to automatically generate equivalent SQL queries.

### 2. Anchor Strategies

Anchors determine the reference date for cohort selection:

* **CurrentDateAnchor**: uses today’s date
* **FixedDateAnchor**: uses a manually specified date
* **MaxDateInDataAnchor**: uses the most recent date available in data

### 3. Dynamic SQL Generation

The `SQLBuilder` class constructs complex SQL queries including:

* Condition / Procedure / Drug / Measurement cohorts
* Temporal filters (e.g., `within_years`, `date_from`, etc.)
* Demographic conditions (age, gender)

### 4. Terminology Mapping

The `TerminologyMapper` uses the LLM to identify and normalize clinical concepts (e.g., “혈압” → “Hypertension”), linking them to standard OMOP concepts such as **SNOMED** or **LOINC**.

### 5. Frontend Web Interface

A responsive chat-style interface (`khdp_frontend.html`) enables:

* Real-time LLM interaction
* Analytics visualization (Chart.js, Cytoscape.js)
* Settings management for backend URL, LLM, and Neo4j

## System Components

### Backend (FastAPI)

`server.py` exposes REST endpoints:

* `POST /ask` → Process user query
* `POST /graph/llm` → Generate a concept graph via LLM
* `GET /` → Serves the frontend interface

### Core Engine

`main.py` defines the `CDMRagSystem`, responsible for:

* Interpreting queries
* Extracting intent and conditions
* Mapping terms
* Generating and executing SQL

### Database Layer

`db_connector.py` handles PostgreSQL connectivity, providing:

* Connection management
* Query execution with parameter binding
* Error handling and logging

### Configuration

All credentials and runtime parameters are defined in `config.yaml`, including:

```yaml
postgresql:
  host: "localhost"
  port: 5432
  dbname: "omop"
  user: "cdm_user"
  password: "******"

llm:
  provider: "openai"
  openai:
    api_base: "https://api.openai.com/v1"
    api_key: "<YOUR_API_KEY>"
    model: "gpt-4o-mini"

neo4j:
  enabled: true
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "******"
```

## Frontend (KHDP CDM RAG Studio)

The frontend (`khdp_frontend.html`) includes:

* A chat interface for clinical question input
* Visualization panels for cohort statistics and graphs
* Settings UI to configure backend and LLM parameters

It uses:

* **Chart.js** for charts
* **Cytoscape.js** for graph visualization
* Modern CSS grid layouts for responsive design

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```
M
**Main dependencies:**

* `fastapi`
* `uvicorn`
* `openai`
* `psycopg2`
* `pandas`
* `pyyaml`
* `neo4j`

### 2. Configure Environment

Edit `config.yaml` with your PostgreSQL, LLM, and Neo4j credentials.

### 3. Launch the Server

```bash
uvicorn server:app --reload --port 8000
```

### 4. Access the Web App

Open in a browser:

* [http://localhost:8000](http://localhost:8000)

## Example Query Flow

1. **User Input**

   > “How many diabetic patients over 40 received insulin in the last 2 years?”

2. **Terminology Mapping**

   ```text
   base_term = "Diabetes mellitus"
   ```

3. **Domain Resolution**

   ```text
   domain = "Condition"
   ```

4. **SQL Generation**

   The system automatically constructs and executes a parameterized query against the OMOP-CDM database.

5. **Result Rendering**

   The frontend returns summary statistics and visualizations for the cohort.