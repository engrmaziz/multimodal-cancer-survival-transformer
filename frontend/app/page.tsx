// frontend/app/page.tsx
'use client';
import { useState } from 'react';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Strip trailing slashes from the API base address
  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUploadAndAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a raw genomics .tsv file first.");
      return;
    }

    setLoading(true);
    setError(null);

    // Prepare multi-part form data payload
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${apiBase}/api/upload-genomics`, {
        method: 'POST',
        body: formData, // No headers needed, the browser sets multi-part boundaries automatically
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Server responded with an error.");
      }

      const data = await response.json();
      setResults(data);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred connecting to the backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main style={{ padding: '40px', maxWidth: '800px', margin: '0 auto', fontFamily: 'sans-serif' }}>
      <h2>Multimodal Cancer Survival Prognosis Panel</h2>
      <p style={{ color: '#666' }}>Research Prototype for Deep Learning Cross-Attention Prediction Verification</p>
      
      <form onSubmit={handleUploadAndAnalyze} style={{ border: '1px solid #ccc', padding: '20px', borderRadius: '8px', marginBottom: '20px' }}>
        <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '10px' }}>
          Upload Patient RNA-Seq Matrix (.tsv format):
        </label>
        <input 
          type="file" 
          accept=".tsv" 
          onChange={handleFileChange} 
          style={{ display: 'block', marginBottom: '20px' }}
        />
        
        <button 
          type="submit" 
          disabled={loading}
          style={{ padding: '10px 20px', cursor: 'pointer', background: '#000', color: '#fff', border: 'none', borderRadius: '4px' }}
        >
          {loading ? "Processing Bioinformatic Vectors..." : "Upload & Generate Survival Analysis"}
        </button>
      </form>

      {error && <div style={{ color: 'red', marginBottom: '20px', fontWeight: 'bold' }}>⚠️ Error: {error}</div>}

      {results && (
        <div style={{ border: '1px solid #222', padding: '20px', borderRadius: '8px', background: '#fafafa' }}>
          <h3>Prognostic Output Metrics</h3>
          <p><strong>Analyzed File:</strong> {results.filename}</p>
          <p><strong>Extracted Coding Genes:</strong> {results.features_extracted}</p>
          <p><strong>Hazard Ratio:</strong> {results.hazard_ratio}</p>
          <p><strong>Risk Classification:</strong> {results.risk_classification}</p>
          
          <h4>Kaplan-Meier Survival Decay Vectors</h4>
          <ul>
            {Object.entries(results.survival_curve).map(([timeline, percentage]: any) => (
              <li key={timeline}><strong>{timeline}:</strong> {percentage}</li>
            ))}
          </ul>
        </div>
      )}
    </main>
  );
}