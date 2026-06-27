# modules/models/transformer_fusion.py
import torch
import torch.nn as nn

class CrossAttentionFusionTransformer(nn.Module):
    def __init__(self, genomic_dim, embedding_dim=384, num_heads=4):
        super().__init__()
        # Project high-dim genomics/clinical data to the shared transformer dimension
        self.genomic_projector = nn.Sequential(
            nn.Linear(genomic_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim)
        )
        
        # Cross-Attention layer: Query from Genomics, Key/Value from Pathology Tiles
        self.cross_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Risk Predictor Layer (Outputs a single hazard risk ratio value)
        self.risk_layer = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, genomics, pathology_tiles):
        """
        genomics: [Batch, Genomic_Dim]
        pathology_tiles: [Batch, Num_Tiles, Embedding_Dim]
        """
        # 1. Project genomics to shared hidden space -> Shape: [Batch, 1, Embedding_Dim]
        h_gen = self.genomic_projector(genomics).unsqueeze(1)
        
        # 2. Apply Cross-Attention
        # Query: h_gen, Key: pathology_tiles, Value: pathology_tiles
        attn_output, attn_weights = self.cross_attn(
            query=h_gen, 
            key=pathology_tiles, 
            value=pathology_tiles
        )
        
        # Residual connection and layer normalization
        fused_vector = self.layer_norm(attn_output.squeeze(1) + h_gen.squeeze(1))
        
        # 3. Predict the continuous log-hazard ratio for Cox loss
        log_hazard_ratio = self.risk_layer(fused_vector)
        return log_hazard_ratio, attn_weights