# =============================================================================
# MASTER SCRIPT: Fault-Tolerant LLM JSON Parser & Relational Schema Mapper
# Title: Automated Extraction & Standardization of Nanoparticle Formulations
# Description: This script ingests unstructured LLM text outputs and standardizes 
#              them into a strict 24-column relational dataset. It employs a 
#              hybrid JSON/Regex parsing engine to recover truncated outputs, 
#              strips markdown artifacts, and sanitizes Continuous Quality Attributes 
#              (CQAs) to ensure seamless downstream machine learning compatibility.
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import json
import re

# =============================================================================
# Step 1: Configuration & Schema Definition
# =============================================================================
# The standardized 24-column master database schema
TARGET_SCHEMA = [
    "Paper Name", "DOI", "Run ID", "Method", "Total Formulation Volume", 
    "API Name", "API Initial Amount", "Chitosan Molecular Weight/Type", 
    "Chitosan Initial Amount", "Alginate Initial Amount", "CS_ALG Ratio", 
    "Surfactant Type", "Surfactant Initial Amount", "Crosslinker Type", 
    "Crosslinker Initial Amount", "Stirring Speed", "Temperature", 
    "Sonication Time", "pH", "Particle Size", "Zeta Potential", 
    "Encapsulation Efficiency", "Data Extraction Remarks", "Data Extraction Thoughts"
]

# =============================================================================
# Step 2: Utility Functions for Data Sanitization
# =============================================================================
def clean_number(val):
    """Safely extracts numeric values; handles raw text strings and maps 'NaN' to None."""
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    if val_str.upper() in ["NAN", "NONE", "NULL", ""]:
        return None
    cleaned = re.sub(r'[^\d.-]', '', val_str)
    try:
        return float(cleaned) if cleaned not in ["", "-", "."] else None
    except ValueError:
        return None

def clean_cqa(val):
    """Cleans CQA strings by dropping standard deviations but preserving units (e.g., '220 nm')."""
    if pd.isna(val): 
        return None
    val_str = str(val).strip()
    if val_str.upper() in ["NAN", "NONE", "NULL", ""]:
        return None
    if "±" in val_str:
        val_str = val_str.split("±")[0].strip()
    return val_str

def clean_id(val):
    """Ensures Run IDs are strictly strings to prevent downstream type inference errors."""
    if pd.isna(val): 
        return None
    val_str = str(val).strip()
    if val_str.upper() in ["NAN", "NONE", ""]:
        return None
    return val_str

# =============================================================================
# Step 3: Data Ingestion & Setup
# =============================================================================
input_df = knio.input_tables[0].to_pandas()
all_extracted_rows = []

# Dynamically locate the output column from the upstream LLM node
response_col = 'Response' if 'Response' in input_df.columns else input_df.columns[-1]

# =============================================================================
# Step 4: Hybrid LLM Parsing Loop
# =============================================================================
for index, row in input_df.iterrows():
    raw_response = str(row[response_col]).strip()
    
    # --- 4A. Artifact Sanitization (Markdown Wrappers) ---
    if "```json" in raw_response:
        parts = raw_response.split("```json")
        if len(parts) > 1: 
            raw_response = parts[1].split("```")[0].strip()
    elif "```" in raw_response:
        parts = raw_response.split("```")
        if len(parts) > 1: 
            raw_response = parts[1].strip()

    header_string = "None | None | None | None"
    runs_list = []
    thoughts_string = "None"

    # --- 4B. Primary Parsing Strategy: Standard JSON Deserialization ---
    try:
        data = json.loads(raw_response)
        if 'skipped' in data or 'SKIPPED' in str(data).upper(): 
            continue
        header_string = data.get("HEADER", "None | None | None | None")
        runs_list = data.get("RUNS", [])
        thoughts_string = data.get("THOUGHTS", "None")
        
    # --- 4C. Fallback Parsing Strategy: Regex Token Recovery ---
    # Triggered when LLM token limits cause truncated/malformed JSON arrays
    except json.JSONDecodeError:
        if 'skipped' in raw_response.lower() or 'not relevant' in raw_response.lower():
            continue
            
        header_match = re.search(r'"HEADER"\s*:\s*"([^"]+)"', raw_response)
        if header_match: 
            header_string = header_match.group(1)
            
        thoughts_match = re.search(r'"THOUGHTS"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_response)
        if thoughts_match:
            thoughts_string = thoughts_match.group(1).replace('\\n', '\n')
            
        for line in raw_response.split('\n'):
            if line.count('|') >= 16:  
                clean_line = line.strip().strip('",')
                runs_list.append(clean_line)

    # =========================================================================
    # Step 5: Matrix Alignment & Unpacking
    # =========================================================================
    # Unpack 4-Part Header Constants (Safeguarded against pipe corruption in titles)
    header_parts = [part.strip() for part in header_string.split("|", 3)]
    while len(header_parts) < 4:
        header_parts.append("None")
        
    paper_name  = header_parts[0] if header_parts[0].upper() != "NAN" else None
    doi         = header_parts[1] if header_parts[1].upper() != "NAN" else None
    api_name    = header_parts[2] if header_parts[2].upper() != "NAN" else None
    method_type = header_parts[3] if header_parts[3].upper() != "NAN" else None

    if thoughts_string == "None" or str(thoughts_string).upper() in ["NAN", ""]:
        thoughts_string = None

    # Unpack 19-Value Piped Run Strings
    for run_string in runs_list:
        parts = [p.strip() for p in run_string.split("|")]
        
        # Enforce exact matrix structural alignment
        while len(parts) < 19: 
            parts.append("NaN")
        parts = parts[:19]  # Truncate hallucinated trailing elements
            
        # Standardize missing values
        parts = [None if str(p).upper() == "NAN" else p for p in parts]
            
        try:
            new_row = [
                paper_name,               # 1. Paper Name
                doi,                      # 2. DOI
                clean_id(parts[0]),       # 3. Run ID
                method_type,              # 4. Method
                parts[14],                # 5. Total Formulation Volume
                api_name,                 # 6. API Name
                parts[1],                 # 7. API Initial Amount
                parts[2],                 # 8. Chitosan Molecular Weight/Type
                parts[3],                 # 9. Chitosan Initial Amount
                parts[4],                 # 10. Alginate Initial Amount
                parts[5],                 # 11. CS_ALG Ratio
                parts[6],                 # 12. Surfactant Type
                parts[7],                 # 13. Surfactant Initial Amount
                parts[8],                 # 14. Crosslinker Type
                parts[9],                 # 15. Crosslinker Initial Amount
                parts[10],                # 16. Stirring Speed
                parts[11],                # 17. Temperature
                parts[12],                # 18. Sonication Time
                parts[13],                # 19. pH
                clean_cqa(parts[15]),     # 20. Particle Size
                clean_cqa(parts[16]),     # 21. Zeta Potential
                clean_cqa(parts[17]),     # 22. Encapsulation Efficiency
                parts[18],                # 23. Data Extraction Remarks
                thoughts_string           # 24. Data Extraction Thoughts
            ]
            all_extracted_rows.append(new_row)
        except Exception:
            # Bypass partial structural failures to maintain pipeline integrity
            pass 

# =============================================================================
# Step 6: Convert and Export to KNIME Node Output
# =============================================================================
output_df = pd.DataFrame(all_extracted_rows, columns=TARGET_SCHEMA)
knio.output_tables[0] = knio.Table.from_pandas(output_df)