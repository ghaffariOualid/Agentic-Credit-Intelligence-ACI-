"use client";

import React from "react";
import {
  ArrowRight,
  CheckCircle2,
  Clock,
  FileText,
  FolderPlus,
  Activity,
} from "lucide-react";
import { CreditCase, InferenceMetricsSnapshot, RunRecord, formatMAD } from "@/lib/api";

interface OverviewViewProps {
  cases: CreditCase[];
  runs: RunRecord[];
  inferenceMetrics: InferenceMetricsSnapshot | null;
  onSelectCase: (caseId: string) => void;
  onCreateFirstCase: () => void;
}

const STATUS_STYLE: Record<string, { label: string; cls: string }> = {
  needs_review: { label: "À vérifier", cls: "bg-amber-50 text-amber-800 border-amber-200" },
  completed: { label: "Validé", cls: "bg-emerald-50 text-emerald-700 border-emerald-200" },
  analyzing: { label: "En analyse", cls: "bg-slate-100 text-slate-700 border-slate-200" },
  draft: { label: "Brouillon", cls: "bg-slate-50 text-slate-600 border-slate-200" },
  failed: { label: "Échec", cls: "bg-rose-50 text-rose-700 border-rose-200" },
};

export function OverviewView({
  cases,
  runs,
  inferenceMetrics,
  onSelectCase,
  onCreateFirstCase,
}: OverviewViewProps) {
  const needsReview = cases.filter((c) => c.status === "needs_review").length;
  const completed = cases.filter((c) => c.status === "completed").length;
  const pending = cases.filter((c) => c.status === "analyzing" || c.status === "draft").length;
  const total = cases.length;

  const avgQuality =
    cases.filter((c) => c.extraction_quality != null).length > 0
      ? Math.round(
          (cases.reduce((acc, c) => acc + (c.extraction_quality ?? 0), 0) /
            cases.filter((c) => c.extraction_quality != null).length) *
            100
        )
      : null;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <div className="bank-card p-5">
          <div className="flex items-center justify-between text-xs font-medium text-slate-500">
            <span>Dossiers créés</span>
            <FileText className="w-4 h-4 text-slate-400" />
          </div>
          <div className="mt-2.5 flex items-baseline gap-2.5">
            <span className="text-3xl font-semibold tracking-tight text-slate-900">{total}</span>
            {needsReview > 0 && (
              <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-amber-50 text-amber-800 border border-amber-200">
                {needsReview} à vérifier
              </span>
            )}
          </div>
          <p className="mt-1 text-xs text-slate-400">Portefeuille actif</p>
        </div>

        <div className="bank-card p-5">
          <div className="flex items-center justify-between text-xs font-medium text-slate-500">
            <span>Exécutions LangGraph</span>
            <Activity className="w-4 h-4 text-slate-400" />
          </div>
          <div className="mt-2.5 flex items-baseline gap-2">
            <span className="text-3xl font-semibold tracking-tight text-slate-900">{runs.length}</span>
            <span className="text-xs text-slate-400">runs</span>
          </div>
          <p className="mt-1 text-xs text-slate-400">Analyses multi-agents</p>
        </div>

        <div className="bank-card p-5">
          <div className="flex items-center justify-between text-xs font-medium text-slate-500">
            <span>Qualité moyenne</span>
            <CheckCircle2 className="w-4 h-4 text-slate-400" />
          </div>
          <div className="mt-2.5">
            <span className="text-3xl font-semibold tracking-tight text-slate-900">
              {avgQuality !== null ? `${avgQuality} %` : "—"}
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-400">Extraction documentaire</p>
        </div>

        <div className="bank-card p-5">
          <div className="flex items-center justify-between text-xs font-medium text-slate-500">
            <span>Latence moyenne LLM</span>
            <Clock className="w-4 h-4 text-slate-400" />
          </div>
          <div className="mt-2.5">
            <span className="text-3xl font-semibold tracking-tight text-slate-900">
              {inferenceMetrics && inferenceMetrics.total_requests > 0
                ? `${(inferenceMetrics.avg_latency_ms / 1000).toFixed(2)} s`
                : "—"}
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-400">
            {inferenceMetrics?.total_requests ?? 0} requête(s) Gateway
          </p>
        </div>
      </div>

      {cases.length === 0 ? (
        <div className="bank-card p-12 text-center space-y-4">
          <div className="w-11 h-11 rounded-xl bg-slate-100 border border-slate-200 mx-auto flex items-center justify-center">
            <FolderPlus className="w-5 h-5 text-slate-700" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Aucun dossier enregistré
            </h2>
            <p className="text-sm text-slate-500 max-w-md mx-auto mt-1">
              Créez votre premier dossier en important vos pièces financières (PDF, Excel ou TXT)
              pour démarrer une analyse traçable.
            </p>
          </div>
          <button onClick={onCreateFirstCase} className="btn-primary">
            <FolderPlus className="w-4 h-4" />
            Créer mon premier dossier
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div className="bank-card overflow-hidden lg:col-span-2">
            <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-slate-900">Dossiers de crédit</h2>
                <p className="text-xs text-slate-500">
                  Sélectionnez un dossier pour lancer l&apos;analyse ou le contrôle humain
                </p>
              </div>
              <button onClick={onCreateFirstCase} className="btn-primary">
                <FolderPlus className="w-3.5 h-3.5" />
                Nouveau dossier
              </button>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="bg-slate-50/80 text-[11px] uppercase tracking-wider text-slate-500">
                  <tr>
                    <th className="px-5 py-3 font-semibold">Dossier</th>
                    <th className="px-5 py-3 font-semibold">Entreprise</th>
                    <th className="px-5 py-3 font-semibold">Financement</th>
                    <th className="px-5 py-3 font-semibold">Statut</th>
                    <th className="px-5 py-3 text-right font-semibold">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {cases.map((c) => {
                    const st = STATUS_STYLE[c.status] || STATUS_STYLE.draft;
                    return (
                      <tr
                        key={c.case_id}
                        onClick={() => onSelectCase(c.case_id)}
                        className="hover:bg-slate-50 cursor-pointer transition-colors"
                      >
                        <td className="px-5 py-3.5 font-semibold text-slate-900">{c.case_id}</td>
                        <td className="px-5 py-3.5">
                          <div className="font-medium text-slate-900">{c.company_name}</div>
                          <div className="text-xs text-slate-500">{c.sector}</div>
                        </td>
                        <td className="px-5 py-3.5 text-slate-700">
                          {formatMAD(c.requested_amount)} {c.currency} · {c.tenor_months} mois
                        </td>
                        <td className="px-5 py-3.5">
                          <span className={`inline-flex px-2.5 py-0.5 rounded-full text-xs font-medium border ${st.cls}`}>
                            {st.label}
                          </span>
                        </td>
                        <td className="px-5 py-3.5 text-right">
                          <span className="inline-flex items-center gap-1 text-xs font-semibold text-slate-700 hover:text-slate-950">
                            Ouvrir <ArrowRight className="w-3.5 h-3.5" />
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="bank-card p-5 flex flex-col justify-between">
            <div>
              <h2 className="text-sm font-semibold text-slate-900 mb-1">Répartition</h2>
              <p className="text-xs text-slate-500 mb-5">Statut de votre portefeuille</p>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="font-medium text-slate-700">Validés</span>
                    <span className="text-emerald-700 font-semibold">{completed}</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-emerald-600"
                      style={{ width: `${Math.round((completed / total) * 100)}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="font-medium text-slate-700">À vérifier (HITL)</span>
                    <span className="text-amber-700 font-semibold">{needsReview}</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-amber-500"
                      style={{ width: `${Math.round((needsReview / total) * 100)}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="font-medium text-slate-700">Brouillon / En cours</span>
                    <span className="text-slate-600 font-semibold">{pending}</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-slate-700"
                      style={{ width: `${Math.round((pending / total) * 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
            <div className="mt-5 pt-3 border-t border-slate-100 text-[11px] text-slate-400">
              Validation humaine obligatoire avant clôture du rapport.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
