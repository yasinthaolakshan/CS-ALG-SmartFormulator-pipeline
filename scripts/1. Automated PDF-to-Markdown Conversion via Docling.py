# =============================================================================
# MASTER SCRIPT: Automated PDF-to-Markdown Conversion via Docling
# Title: High-Fidelity Document Ingestion & Text Extraction Pipeline
# Description: This script processes academic PDFs, utilizing the Docling deep 
#              learning library to extract structural text and export it as 
#              LLM-ready Markdown. It integrates a PyTorch cuDNN hotfix for 
#              hardware stability within KNIME and automatically archives the 
#              parsed texts to a local directory for version control.
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import os
import torch

# --- Hardware Stability Hotfix ---
# Disabling cuDNN prevents specific memory allocation crashes when running 
# containerized deep learning models (like Docling) inside the KNIME environment.
torch.backends.cudnn.enabled = False

from docling.document_converter import DocumentConverter

# =============================================================================
# Step 1: Configuration & Directory Setup
# =============================================================================
# Target column containing the absolute paths to the PDF files
PATH_COLUMN_NAME = "Filepath" 

# Destination folder for the exported Markdown files (.md)
OUTPUT_FOLDER = r"/home/user1/CSALG_New/Docling_Markdown/" 
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =============================================================================
# Step 2: Data Ingestion & Model Initialization
# =============================================================================
input_df = knio.input_tables[0].to_pandas()
parsed_text_list = []

try:
    print("--- INITIALIZING DOCLING ENGINE ---")
    converter = DocumentConverter()
    print("Docling Deep Learning Models loaded successfully.")
    print("-----------------------------------\n")
except Exception as e:
    raise RuntimeError(f"CRITICAL ERROR: Failed to load Docling models. Details: {str(e)}")

# =============================================================================
# Step 3: Batch Document Processing Loop
# =============================================================================
for index, row in input_df.iterrows():
    pdf_path = row.get(PATH_COLUMN_NAME)
    
    # Validation safeguard against null or empty paths in the KNIME table
    if pd.isna(pdf_path) or not str(pdf_path).strip():
        parsed_text_list.append("Skipped: Null or empty filepath")
        continue
        
    try:
        print(f"[{index + 1}/{len(input_df)}] Extracting: {pdf_path}")
        
        # 1. Engage Deep Learning Document Conversion
        result = converter.convert(str(pdf_path))
        
        # 2. Extract structured Markdown representation
        md_text = result.document.export_to_markdown()
        parsed_text_list.append(md_text)
        
        # 3. Local System Archival (Preserving Linux and LLM-friendly formatting)
        base_name = os.path.basename(str(pdf_path)) 
        file_name_only = os.path.splitext(base_name)[0] 
        save_path = os.path.join(OUTPUT_FOLDER, f"{file_name_only}.md")
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(md_text)
            
    except Exception as e:
        print(f"!!! Failed extraction on {pdf_path}. Error: {str(e)}")
        parsed_text_list.append(f"Extraction Failed: {str(e)}")

# =============================================================================
# Step 4: Finalize Dataset & Export to KNIME Node
# =============================================================================
input_df['Docling_Markdown'] = parsed_text_list
knio.output_tables[0] = knio.Table.from_pandas(input_df)

print("\n--- BATCH EXTRACTION COMPLETE ---")