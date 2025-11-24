DOMAIN_SPEC = {
        "Condition": {
            "standard": ["SNOMED"],
            "sources": ["ICD10CM","ICD9CM"],
            "table": "condition_occurrence",
            "concept_field": "condition_concept_id",
            "date_field": "condition_start_date"
            },
        "Procedure": {
            "standard": ["SNOMED","CPT4","ICD10PCS"],
            "sources": ["ICD9Proc"],
            "table": "procedure_occurrence",
            "concept_field": "procedure_concept_id",
            "date_field": "procedure_date"
            },
        "Drug": {
            "standard": ["RxNorm","RxNorm Extension"],
            "sources": ["NDC","ATC"],
            "table": "drug_exposure",
            "concept_field": "drug_concept_id",
            "date_field": "drug_exposure_start_date"
            },
        "Measurement": {
            "standard": ["LOINC"],
            "sources": [],
            "table": "measurement",
            "concept_field": "measurement_concept_id",
            "date_field": "measurement_date"
            },
        "Observation": {
            "standard": ["SNOMED","LOINC"],
            "sources": [],
            "table": "observation",
            "concept_field": "observation_concept_id",
            "date_field": "observation_date"
            },
        "Device": {
            "standard": ["SNOMED"],
            "sources": [],
            "table": "device_exposure",
            "concept_field": "device_concept_id",
            "date_field": "device_exposure_start_date"
            },
        "Death": {
            "standard": [],
            "sources":  [],
            "maps_to":  [],
            "closure_vocabs": [],
            "table": "death",
            "concept_field": None,
            "date_field": "death_date"
            },

    }