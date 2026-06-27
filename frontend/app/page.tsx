// frontend/app/page.tsx
"use client";

import React, { useState } from 'react';

export default function OncologyDashboard() {
  const [stage, setStage] = useState<number>(2);
  const [genomics, setGenomics] = useState<string>("0.2, -0.5, 1.1, 0.4");
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const triggerInference = async () => {
    setLoading(true);
    try {
      const numericVectors = genomics.split(',').map(v => parseFloat(v.trim()));
      
      // Read the environment variable dynamically
      const apiBase = 'https://reimagined-waddle-g4994j64jj6r3wqwp-8000.app.github.dev';
      
      const response = await fetch(`${apiBase}/api/predict-survival`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          clinical_stage: stage,
          genomic_expression_vector: numericVectors
        })
      });
      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error("Inference execution failure:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="p-8 max-w-5xl mx-auto space-y-6">
      <div className="border-b pb-4">
        <h1 className="text-3xl font-bold tracking-tight">Multimodal Cancer Survival Prognosis Panel</h1>
        <p className="text-gray-500">Research Prototype for Deep Learning Cross-Attention Prediction Verification</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Controls Layout Wrapper */}
        <div className="space-y-4 p-6 border rounded-xl bg-white shadow-sm">
          <h2 className="text-xl font-semibold">Patient Clinical Variables</h2>
          
          <div>
            <label className="block text-sm font-medium text-gray-700">Pathological Cancer Stage</label>
            <select 
              value={stage} 
              onChange={(e) => setStage(Number(e.target.value))}
              className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm bg-white"
            >
              <option value={1}>Stage I</option>
              <option value={2}>Stage II</option>
              <option value={3}>Stage III</option>
              <option value={4}>Stage IV</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">Target Gene Markers Expression Matrix (Comma Separated)</label>
            <input 
              type="text" 
              value={genomics} 
              onChange={(e) => setGenomics(e.target.value)}
              className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm font-mono text-sm"
            />
          </div>

          <button 
            onClick={triggerInference}
            disabled={loading}
            className="w-full bg-slate-900 text-white p-3 rounded-lg font-medium hover:bg-slate-800 transition-colors disabled:bg-slate-400"
          >
            {loading ? "Computing Transformer Hazard Ratios..." : "Generate Survival Curve Analysis"}
          </button>
        </div>

        {/* Results Metrics Block */}
        <div className="p-6 border rounded-xl bg-slate-50 flex flex-col justify-between">
          <h2 className="text-xl font-semibold mb-4">Prognostic Output Metrics</h2>
          {results ? (
            <div className="space-y-4 flex-1">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white p-4 rounded-lg border shadow-sm">
                  <span className="text-xs text-gray-500 block uppercase font-bold tracking-wider">Hazard Ratio</span>
                  <span className="text-2xl font-bold text-slate-800">{results.hazard_ratio}</span>
                </div>
                <div className="bg-white p-4 rounded-lg border shadow-sm">
                  <span className="text-xs text-gray-500 block uppercase font-bold tracking-wider">Classification</span>
                  <span className="text-lg font-bold text-slate-800">{results.risk_classification}</span>
                </div>
              </div>

              <div className="bg-white p-4 rounded-lg border shadow-sm flex-1">
                <span className="text-sm font-medium block mb-2 text-gray-700">Kaplan-Meier Survival Decay Vectors</span>
                <div className="space-y-1 font-mono text-xs text-gray-600">
                  {results.timeline_months?.map((m: number, idx: number) => (
                    <div key={m} className="flex justify-between border-b pb-1">
                      <span>Month {m}:</span>
                      <span className="font-bold">{(results.survival_probabilities[idx] * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-gray-400 text-center py-12 border-2 border-dashed rounded-lg flex-1 flex items-center justify-center">
              Awaiting model inputs to map risk timeline probability vector metrics.
            </div>
          )}
        </div>
      </div>
    </main>
  );
}