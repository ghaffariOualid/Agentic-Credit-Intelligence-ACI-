"use client";

import React from "react";
import { Cpu, Activity, Shield, BarChart3 } from "lucide-react";
import { HealthResponse, InferenceMetricsSnapshot } from "@/lib/api";

interface GpuInfraViewProps {
  health: HealthResponse | null;
  metrics: InferenceMetricsSnapshot | null;
}

export function GpuInfraView({ health, metrics }: GpuInfraViewProps) {
  const models = health?.models;
  const rows = [
    { tier: "small", model: models?.small || "Non configuré", calls: metrics?.requests_by_tier?.small ?? 0 },
    { tier: "medium", model: models?.medium || "Non configuré", calls: metrics?.requests_by_tier?.medium ?? 0 },
    { tier: "large", model: models?.large || "Non configuré", calls: metrics?.requests_by_tier?.large ?? 0 },
  ];

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bank-card p-5">
          <div className="flex justify-between text-xs text-slate-500">
            <span>Tunnel Colab T4</span>
            <Shield className="w-4 h-4 text-slate-400" />
          </div>
          <div className="text-2xl font-semibold text-slate-900 mt-2">
            {health ? (health.vllm_configured ? "Connecté" : "Non configuré") : "Hors ligne"}
          </div>
          <div className="text-xs text-slate-400 mt-1">Mode : {health?.inference_mode || "—"}</div>
        </div>

        <div className="bank-card p-5">
          <div className="flex justify-between text-xs text-slate-500">
            <span>Requêtes totales</span>
            <BarChart3 className="w-4 h-4 text-slate-400" />
          </div>
          <div className="text-2xl font-semibold text-slate-900 mt-2">
            {metrics?.total_requests ?? 0}
          </div>
          <div className="text-xs text-slate-400 mt-1">En cours : {metrics?.in_flight ?? 0}</div>
        </div>

        <div className="bank-card p-5">
          <div className="flex justify-between text-xs text-slate-500">
            <span>Latence moyenne</span>
            <Activity className="w-4 h-4 text-slate-400" />
          </div>
          <div className="text-2xl font-semibold text-slate-900 mt-2">
            {metrics && metrics.total_requests > 0
              ? `${(metrics.avg_latency_ms / 1000).toFixed(2)} s`
              : "—"}
          </div>
          <div className="text-xs text-slate-400 mt-1">Mesurée par la Gateway</div>
        </div>

        <div className="bank-card p-5">
          <div className="flex justify-between text-xs text-slate-500">
            <span>Fallbacks / Erreurs</span>
            <Cpu className="w-4 h-4 text-slate-400" />
          </div>
          <div className="text-2xl font-semibold text-slate-900 mt-2">
            {metrics?.fallback_count ?? 0} / {metrics?.error_count ?? 0}
          </div>
          <div className="text-xs text-slate-400 mt-1">Compteurs temps réel</div>
        </div>
      </div>

      <div className="bank-card overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-100 text-sm font-semibold text-slate-900">
          Modèles configurés & trafic réel par tier
        </div>
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-50/80 text-[11px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-5 py-3">Tier</th>
              <th className="px-5 py-3">Modèle résolu</th>
              <th className="px-5 py-3">Appels effectués</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map((r) => (
              <tr key={r.tier}>
                <td className="px-5 py-3.5 font-semibold uppercase text-slate-900">{r.tier}</td>
                <td className="px-5 py-3.5 font-mono text-xs text-slate-600">{r.model}</td>
                <td className="px-5 py-3.5 font-semibold text-slate-900">{r.calls}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
