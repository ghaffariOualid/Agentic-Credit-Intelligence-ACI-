"use client";

import React, { useState } from "react";
import { Sliders, ShieldAlert, CheckCircle2, Sparkles } from "lucide-react";
import { RouterDecision, apiFetch } from "@/lib/api";

export function RouterView() {
  const [complexity, setComplexity] = useState(0.5);
  const [gpuUtil, setGpuUtil] = useState(0.5);
  const [vramUtil, setVramUtil] = useState(0.5);
  const [tokens, setTokens] = useState(4000);
  const [taskType, setTaskType] = useState("financial_extraction");
  const [decision, setDecision] = useState<RouterDecision | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function simulateRoute() {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch<RouterDecision>("/api/v1/router/predict", {
        method: "POST",
        body: JSON.stringify({
          task_type: taskType,
          context_tokens: tokens,
          estimated_complexity: complexity,
          gpu_utilization: gpuUtil,
          vram_utilization: vramUtil,
          queue_depth: Math.round(gpuUtil * 16),
          remaining_budget_usd: 0.85,
        }),
      });
      setDecision(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Backend FastAPI non joignable.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-5">
      <div className="bank-card p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-slate-600" />
            <h2 className="text-sm font-semibold text-slate-900">
              Évaluation de politique (`POST /api/v1/router/predict`)
            </h2>
          </div>
          <button onClick={simulateRoute} disabled={loading} className="btn-primary">
            <Sparkles className="w-3.5 h-3.5" />
            {loading ? "Calcul..." : "Évaluer la politique"}
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 text-xs">
          <div>
            <label className="font-medium text-slate-700 block mb-1">Tâche</label>
            <select
              value={taskType}
              onChange={(e) => setTaskType(e.target.value)}
              className="bank-input"
            >
              <option value="financial_extraction">financial_extraction</option>
              <option value="risk_analysis">risk_analysis</option>
              <option value="verification">verification</option>
              <option value="report_synthesis">report_synthesis</option>
            </select>
          </div>

          <div>
            <label className="font-medium text-slate-700 flex justify-between mb-1">
              <span>Complexité</span>
              <span className="font-semibold text-slate-900">{complexity.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0.1"
              max="0.95"
              step="0.05"
              value={complexity}
              onChange={(e) => setComplexity(parseFloat(e.target.value))}
              className="w-full accent-slate-900"
            />
          </div>

          <div>
            <label className="font-medium text-slate-700 flex justify-between mb-1">
              <span>Charge GPU</span>
              <span className="font-semibold text-slate-900">{Math.round(gpuUtil * 100)} %</span>
            </label>
            <input
              type="range"
              min="0.1"
              max="0.95"
              step="0.05"
              value={gpuUtil}
              onChange={(e) => setGpuUtil(parseFloat(e.target.value))}
              className="w-full accent-slate-900"
            />
          </div>

          <div>
            <label className="font-medium text-slate-700 flex justify-between mb-1">
              <span>VRAM T4</span>
              <span className="font-semibold text-slate-900">{Math.round(vramUtil * 100)} %</span>
            </label>
            <input
              type="range"
              min="0.2"
              max="0.95"
              step="0.05"
              value={vramUtil}
              onChange={(e) => setVramUtil(parseFloat(e.target.value))}
              className="w-full accent-slate-900"
            />
          </div>

          <div>
            <label className="font-medium text-slate-700 flex justify-between mb-1">
              <span>Contexte (tokens)</span>
              <span className="font-semibold text-slate-900">{tokens.toLocaleString("fr-FR")}</span>
            </label>
            <input
              type="range"
              min="1000"
              max="12000"
              step="500"
              value={tokens}
              onChange={(e) => setTokens(parseInt(e.target.value, 10))}
              className="w-full accent-slate-900"
            />
          </div>
        </div>
        {error && <p className="mt-3 text-xs text-rose-700 font-medium">{error}</p>}
      </div>

      {!decision ? (
        <div className="bank-card p-10 text-center text-sm text-slate-500">
          Ajustez les paramètres puis cliquez sur <strong>Évaluer la politique</strong> pour
          interroger le Decision Engine en direct.
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="bank-card p-5">
              <div className="text-xs text-slate-500">Modèle sélectionné ({decision.decision_id})</div>
              <div className="mt-2 flex items-center gap-2.5">
                <span className="text-3xl font-semibold uppercase text-slate-900">
                  {decision.action.selected_tier}
                </span>
                <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-medium">
                  Autorisée
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-2">Politique : {decision.policy_version}</p>
            </div>

            <div className="bank-card p-5 lg:col-span-2">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
                Justification
              </div>
              <p className="text-sm text-slate-800">{decision.rationale}</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {decision.constraints_applied.map((c, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg bg-slate-50 border border-slate-200 text-slate-600"
                  >
                    <ShieldAlert className="w-3.5 h-3.5 text-slate-500" />
                    {c}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="bank-card overflow-hidden">
            <div className="px-5 py-3.5 border-b border-slate-100 text-sm font-semibold text-slate-900">
              Comparaison des candidats (0,6Q − 0,15C − 0,15L − 0,1E)
            </div>
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50/80 text-[11px] uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="px-5 py-3">Tier</th>
                  <th className="px-5 py-3">Qualité prédite</th>
                  <th className="px-5 py-3">Latence T4</th>
                  <th className="px-5 py-3">Coût estimé</th>
                  <th className="px-5 py-3">Score</th>
                  <th className="px-5 py-3">Statut</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {decision.candidates.map((cand) => {
                  const chosen = cand.tier === decision.action.selected_tier;
                  return (
                    <tr key={cand.tier} className={chosen ? "bg-slate-50" : ""}>
                      <td className="px-5 py-3.5 font-semibold uppercase text-slate-900">{cand.tier}</td>
                      <td className="px-5 py-3.5 text-slate-700">{Math.round(cand.predicted_quality * 100)} %</td>
                      <td className="px-5 py-3.5 text-slate-700">{cand.predicted_latency_s.toFixed(1)} s</td>
                      <td className="px-5 py-3.5 text-slate-700">{cand.predicted_cost_usd.toFixed(2)} $</td>
                      <td className="px-5 py-3.5 font-semibold text-slate-900">
                        {cand.score.toFixed(3)}
                      </td>
                      <td className="px-5 py-3.5">
                        {!cand.allowed ? (
                          <span className="text-xs px-2.5 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200">
                            Masquée
                          </span>
                        ) : chosen ? (
                          <span className="inline-flex items-center gap-1 text-xs px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-semibold">
                            <CheckCircle2 className="w-3.5 h-3.5" /> Sélectionnée
                          </span>
                        ) : (
                          <span className="text-xs text-slate-500">Autorisée</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
