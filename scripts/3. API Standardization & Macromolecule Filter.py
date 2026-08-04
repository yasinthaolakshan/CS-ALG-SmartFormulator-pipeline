import knime.scripting.io as knio
import pandas as pd
import re

# =============================================================================
# Step 1: Data Ingestion
# =============================================================================
df = knio.input_tables[0].to_pandas()

# =============================================================================
# Step 2: API Standardization & Corrections
# =============================================================================
def standardize_turmeric(api_name):
    """
    Standardizes variations of Turmeric Oil to 'Turmerone' using Regex.
    """
    if pd.isna(api_name):
        return api_name
    return re.sub(r'(?i)Turmeric\s*Oil.*', 'Turmerone', str(api_name))

df['API Name'] = df['API Name'].apply(standardize_turmeric)

# Correct Favipiravir records: Move amount from Remarks to API Initial Amount
# We use case-insensitive matching to ensure all 'Favipiravir' rows are caught
favipiravir_mask = df['API Name'].astype(str).str.lower() == 'favipiravir'
df.loc[favipiravir_mask, 'API Initial Amount'] = df.loc[favipiravir_mask, 'Data Extraction Remarks']

# =============================================================================
# Step 3: Define Exclusion Criteria & Classify
# =============================================================================
# Define keywords identifying APIs that cannot be used (macromolecules, specific mixtures, etc.)
exclude_keywords = [
    'bsa', 'lysozyme', 'antisense', 'insulin', 'ovalbumin', 
    'protein', 'plasmid', 'rhbmp-2', 'streptokinase', 'nisin', 
    'nisaplin', 'polyphenol', 'con a complex', 'lutein', 'malathion', 'nan',
    'mangostins','ciprofloxacin hcl','tamoxifen citrate'
]

def classify_api(api_name):
    """
    Categorizes the API based on the expanded exclusion list.
    """
    if pd.isna(api_name) or str(api_name).strip().lower() == 'nan':
        return 'Excluded API'
        
    api_name_str = str(api_name).lower()
    if any(keyword in api_name_str for keyword in exclude_keywords):
        return 'Excluded API'
    
    return 'Retained Small Molecule'

df['API Class'] = df['API Name'].apply(classify_api)

# =============================================================================
# Step 4: Split Co-encapsulated APIs
# =============================================================================
# 4A. Split the 'API Name' column
api_name_split = df['API Name'].str.split(r'\s*\+\s*', n=1, expand=True)
df['API 1 Name'] = api_name_split[0]
df['API 2 Name'] = api_name_split[1] if api_name_split.shape[1] > 1 else None

# 4B. Split the 'API Initial Amount' column
api_amount_str = df['API Initial Amount'].astype(str)
api_amt_split = api_amount_str.str.split(r'\s*\+\s*', n=1, expand=True)

df['API 1 Initial Amount'] = api_amt_split[0].replace('nan', pd.NA)

if api_amt_split.shape[1] > 1:
    df['API 2 Initial Amount'] = api_amt_split[1].replace('nan', pd.NA)
else:
    df['API 2 Initial Amount'] = None

# =============================================================================
# Step 5: Data Filtering & Output
# =============================================================================
# Filter the DataFrame to KEEP ONLY the suitable small molecules
df_filtered = df[df['API Class'] == 'Retained Small Molecule'].copy()

# Output the finished, curated table back to the KNIME output port
knio.output_tables[0] = knio.Table.from_pandas(df_filtered)