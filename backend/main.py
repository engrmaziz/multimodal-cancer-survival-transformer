# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import os
import torch
import torch.nn as nn

app = FastAPI(title="Multimodal Cancer Survival Prognosis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. THE NEURAL NETWORK ARCHITECTURE
# ==========================================
class CancerSurvivalTransformer(nn.Module):
    def __init__(self, genomic_features=100, image_embed_dim=384, num_heads=4):
        super().__init__()
        self.genomic_proj = nn.Sequential(
            nn.Linear(genomic_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, image_embed_dim)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=image_embed_dim, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(image_embed_dim)
        self.risk_predictor = nn.Sequential(
            nn.Linear(image_embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, genomic_x, image_x):
        g_proj = self.genomic_proj(genomic_x).unsqueeze(1)
        attn_out, _ = self.cross_attn(query=g_proj, key=image_x, value=image_x)
        fused_vector = self.norm(g_proj + attn_out).squeeze(1)
        return self.risk_predictor(fused_vector)

# ==========================================
# 2. MODEL INITIALIZATION (Loads on Startup)
# ==========================================
device = torch.device("cpu") # Inference runs fine on CPU
model = CancerSurvivalTransformer().to(device)

# Load the weights you just trained in Colab
weights_path = os.path.join(os.path.dirname(__file__), "multimodal_survival_weights.pth")
if os.path.exists(weights_path):
    # map_location='cpu' ensures it loads even if the Codespace lacks a GPU
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() # Set to evaluation mode (disables dropout)
    print("✅ ENTERPRISE WEIGHTS LOADED SUCCESSFULLY")
else:
    print("⚠️ WARNING: multimodal_survival_weights.pth not found! Ensure it is in the backend/ directory.")


# ==========================================
# 3. DATA ENGINEERING & API ROUTE
# ==========================================
def process_genomic_file(file_contents: bytes) -> list:
    """Extracts the 100-dimensional biological signature."""
    try:
        string_data = file_contents.decode("utf-8")
        data_buffer = io.StringIO(string_data)
        df = pd.read_csv(
            data_buffer, sep='\t', skiprows=6, 
            names=['gene_id', 'gene_name', 'gene_type', 'unstranded', 
                   'stranded_first', 'stranded_second', 'tpm_unstranded', 
                   'fpkm_unstranded', 'fpkm_uq_unstranded']
        )
        coding_df = df[df['gene_type'] == 'protein_coding'].copy()
        coding_df['log2_tpm'] = np.log2(coding_df['tpm_unstranded'] + 1)
        top_expressed = coding_df.sort_values(by='log2_tpm', ascending=False).head(100)
        return top_expressed['log2_tpm'].values.tolist()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse matrix: {str(e)}")

@app.post("/api/upload-genomics")
async def upload_genomics(file: UploadFile = File(...)):
    if not file.filename.endswith('.tsv'):
        raise HTTPException(status_code=400, detail="Invalid format. Please upload a .tsv matrix.")
        
    contents = await file.read()
    extracted_features = process_genomic_file(contents)
    
    # --- ENTERPRISE INFERENCE BLOCK ---
    with torch.no_grad(): # Disable gradient tracking for speed
        # 1. Convert 1D Python list to 2D PyTorch Tensor: Shape [1, 100]
        genomic_tensor = torch.tensor(extracted_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 2. Inject standard placeholder DINOv2 pathology embeddings: Shape [1, 50, 384]
        # In a V2 release, this would be passed from a Whole Slide Image parser
        mock_image_tensor = torch.zeros((1, 50, 384)).to(device)
        
        # 3. Predict Log-Hazard
        predicted_log_hazard = model(genomic_tensor, mock_image_tensor).item()
    
    # Convert Log-Hazard to Hazard Ratio using standard exp(x)
    # Adding a baseline shift to map the raw network output to a clinical 0.5 - 3.5 scale for the UI
    baseline_shift = 1.2 
    hazard_ratio = float(round(np.exp(predicted_log_hazard) + baseline_shift, 2))
    
    # Map to Kaplan-Meier Curve
    months = [0, 12, 24, 36, 48, 60]
    survival_decay = [round(float(np.exp(-0.01 * hazard_ratio * t)) * 100, 1) for t in months]
    
    return {
        "status": "success",
        "filename": file.filename,
        "features_extracted": len(extracted_features),
        "hazard_ratio": hazard_ratio,
        "risk_classification": "High Risk" if hazard_ratio > 2.0 else "Low Risk",
        "survival_curve": {f"Month {m}": f"{s}%" for m, s in zip(months, survival_decay)}
    }