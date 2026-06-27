# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io

app = FastAPI(title="Multimodal Cancer Survival Prognosis API")

# Strict CORS configuration for GitHub Codespaces proxying
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, # Must be False when using wildcard origin
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_genomic_file(file_contents: bytes) -> list:
    """Parses raw TSV bytes, computes log2(TPM + 1), and extracts top features."""
    try:
        # Wrap bytes in a StringIO buffer to read into pandas
        string_data = file_contents.decode("utf-8")
        data_buffer = io.StringIO(string_data)
        
        # TCGA files skip the first 6 rows of metadata
        df = pd.read_csv(
            data_buffer, 
            sep='\t', 
            skiprows=6, 
            names=['gene_id', 'gene_name', 'gene_type', 'unstranded', 
                   'stranded_first', 'stranded_second', 'tpm_unstranded', 
                   'fpkm_unstranded', 'fpkm_uq_unstranded']
        )
        
        # Filter for protein-coding rows
        coding_df = df[df['gene_type'] == 'protein_coding'].copy()
        if coding_df.empty:
            raise ValueError("No protein-coding genes found in the provided matrix.")
            
        # Transform exponential values to log scale
        coding_df['log2_tpm'] = np.log2(coding_df['tpm_unstranded'] + 1)
        
        # Isolate top 100 signature genes as our intermediate vector
        top_expressed = coding_df.sort_values(by='log2_tpm', ascending=False).head(100)
        
        # Convert array to standard Python float list for JSON serialization
        return top_expressed['log2_tpm'].values.tolist()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse genomic matrix: {str(e)}")

@app.post("/api/upload-genomics")
async def upload_genomics(file: UploadFile = File(...)):
    # Validate file extension
    if not file.filename.endswith('.tsv'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a TCGA .tsv expression matrix.")
        
    # Read raw incoming bytes asynchronously
    contents = await file.read()
    
    # Process the file using our bioinformatic transformation logic
    extracted_features = process_genomic_file(contents)
    
    # Calculate a proxy Hazard Ratio based on the real mean variance of the vector
    # This acts as our temporary placeholder until we load the PyTorch weights in Phase B
    mean_expression = np.mean(extracted_features)
    mock_hazard_ratio = float(round(max(0.5, (mean_expression / 4.5)), 2))
    
    # Map the hazard ratio to survival curve data points
    months = [0, 12, 24, 36, 48, 60]
    survival_decay = [round(float(np.exp(-0.01 * mock_hazard_ratio * t)) * 100, 1) for t in months]
    
    return {
        "status": "success",
        "filename": file.filename,
        "features_extracted": len(extracted_features),
        "hazard_ratio": mock_hazard_ratio,
        "risk_classification": "High Risk" if mock_hazard_ratio > 2.0 else "Low Risk",
        "survival_curve": {f"Month {m}": f"{s}%" for m, s in zip(months, survival_decay)}
    }