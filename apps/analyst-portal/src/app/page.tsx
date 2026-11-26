"use client";

import React, { useEffect, useState } from "react";
import {
  LayoutDashboard,
  FolderKanban,
  GitBranch,
  Compass,
  FlaskConical,
  Server,
  ShieldCheck,
} from "lucide-react";
import {
  API_BASE,
  CreditCase,
  HealthResponse,
  InferenceMetricsSnapshot,
  RunRecord,
  TrainingJobRecord,
  apiFetch,
} from "@/lib/api";
import { TapisHeroBanner } from "@/components/TapisHeroBanner";
import { OverviewView } from "@/components/OverviewView";
import { CaseWorkspaceView } from "@/components/CaseWorkspaceView";
import { ExecutionsView } from "@/components/ExecutionsView";
import { RouterView } from "@/components/RouterView";
import { StudioRLView } from "@/components/StudioRLView";
import { GpuInfraView } from "@/components/GpuInfraView";

type TabId = "overview" | "cases" | "executions" | "router" | "studio" | "gpu";

const NAV_ITEMS: { id: TabId; label: string; group: "metier" | "ia"; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: "overview", label: "Vue générale", group: "metier", icon: LayoutDashboard },
  { id: "cases", label: "Dossiers de crédit", group: "metier", icon: FolderKanban },
  { id: "executions", label: "Exécutions agents", group: "metier", icon: GitBranch },
  { id: "router", label: "Routeur intelligent", group: "ia", icon: Compass },
  { id: "studio", label: "Studio RL", group: "ia", icon: FlaskConical },
  { id: "gpu", label: "Infrastructure GPU", group: "ia", icon: Server },
];

const BANNER_META: Record<TabId, { label: string; title: string; subtitle: string }> = {
  overview: {
    label: "Banque Entreprise · Pilotage",
    title: "Agentic Credit Intelligence",
    subtitle:
      "Vue consolidée de vos dossiers, exécutions multi-agents et métriques réelles d'inférence.",
  },
  cases: {
    label: "Poste Analyste · Contrôle HITL",
    title: "Instruction des dossiers de crédit",
    subtitle:
      "Importez vos pièces (PDF, Excel, TXT), lancez l'analyse déterministe et validez en HITL.",
  },
  executions: {
    label: "Runtime LangGraph · Observabilité",
    title: "Exécution multi-agents & traces",
    subtitle:
      "Suivez chaque étape réelle du graphe, le modèle routé sur T4 et les suspensions HITL.",
  },
  router: {
    label: "Decision Engine · Explicabilité",
    title: "Routeur intelligent de modèles",
    subtitle:
      "Évaluez la politique active Small / Medium / Large sous contraintes de qualité et VRAM T4.",
  },
  studio: {
    label: "Apprentissage Hors Production",
    title: "Studio RL & Entraînement",
    subtitle:
      "Entraînez et comparez vos propres politiques Supervisé, Bandit LinUCB et PPO.",
  },
  gpu: {
    label: "Serving Distribué · vLLM",
    title: "Infrastructure GPU Colab T4",
    subtitle:
      "Compteurs temps réel de la passerelle d'inférence et état de connexion Colab T4.",
  },
};

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<TabId>("overview");
  const [cases, setCases] = useState<CreditCase[]>([]);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [jobs, setJobs] = useState<TrainingJobRecord[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string>("");
  const [activeRun, setActiveRun] = useState<RunRecord | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [inferenceMetrics, setInferenceMetrics] = useState<InferenceMetricsSnapshot | null>(null);
  const [loading, setLoading] = useState(false);

  async function refreshAll() {
    try {
      const h = await apiFetch<HealthResponse>("/health");
      setHealth(h);
      const [caseList, runList, jobList, infMetrics] = await Promise.all([
        apiFetch<CreditCase[]>("/api/v1/cases"),
        apiFetch<RunRecord[]>("/api/v1/runs"),
        apiFetch<TrainingJobRecord[]>("/api/v1/training/jobs"),
        apiFetch<InferenceMetricsSnapshot>("/api/v1/inference/metrics"),
      ]);
      setCases(caseList);
      setRuns(runList);
      setJobs(jobList);
      setInferenceMetrics(infMetrics);
      if (caseList.length > 0 && !selectedCaseId) {
        setSelectedCaseId(caseList[0].case_id);
      }
    } catch {
      setHealth(null);
    }
  }

  useEffect(() => {
    refreshAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleAnalyzeCase(caseId: string) {
    setLoading(true);
    try {
      const run = await apiFetch<RunRecord>(`/api/v1/cases/${caseId}/analyze`, {
        method: "POST",
      });
      setActiveRun(run);
      await refreshAll();
    } finally {
      setLoading(false);
    }
  }

  async function handleUploadCase(formData: FormData) {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/cases/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Erreur import (${res.status}): ${text}`);
      }
      const created = (await res.json()) as CreditCase;
      setSelectedCaseId(created.case_id);
      setActiveRun(null);
      await refreshAll();
    } finally {
      setLoading(false);
    }
  }

  async function handleResumeHitl(
    runId: string,
    payload: {
      reviewer_id: string;
      action: "approve" | "correct" | "request_correction";
      comment: string;
      field_overrides: Record<string, number>;
    }
  ) {
    setLoading(true);
    try {
      const resumed = await apiFetch<RunRecord>(`/api/v1/runs/${runId}/resume`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setActiveRun(resumed);
      await refreshAll();
    } finally {
      setLoading(false);
    }
  }

  const meta = BANNER_META[activeTab];

  return (
    <div className="min-h-screen flex bg-[#F8FAFC]">
      {/* Sidebar claire institutionnelle (Private Banking / Linear aesthetic) */}
      <aside className="w-64 shrink-0 bg-white text-slate-900 flex flex-col justify-between p-5 border-r border-slate-200">
        <div>
          <div className="pb-5 mb-5 border-b border-slate-100">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-slate-900 flex items-center justify-center font-bold text-xs tracking-wider text-white">
                ACI
              </div>
              <div>
                <div className="font-semibold text-sm tracking-tight text-slate-900">
                  ACI Workspace
                </div>
                <div className="text-[11px] text-slate-400 font-medium">
                  Banque Entreprise
                </div>
              </div>
            </div>
          </div>

          <nav className="space-y-6">
            <div>
              <div className="px-3 mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                Analyse Crédit
              </div>
              <div className="space-y-1">
                {NAV_ITEMS.filter((i) => i.group === "metier").map((item) => {
                  const Icon = item.icon;
                  const active = activeTab === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => setActiveTab(item.id)}
                      className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-all cursor-pointer ${
                        active
                          ? "bg-slate-900 text-white shadow-xs"
                          : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
                      }`}
                    >
                      <Icon className={`w-4 h-4 ${active ? "text-white" : "text-slate-400"}`} />
                      {item.label}
                    </button>
                  );
                })}
              </div>
            </div>

            <div>
              <div className="px-3 mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                Intelligence & GPU
              </div>
              <div className="space-y-1">
                {NAV_ITEMS.filter((i) => i.group === "ia").map((item) => {
                  const Icon = item.icon;
                  const active = activeTab === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => setActiveTab(item.id)}
                      className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-all cursor-pointer ${
                        active
                          ? "bg-slate-900 text-white shadow-xs"
                          : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
                      }`}
                    >
                      <Icon className={`w-4 h-4 ${active ? "text-white" : "text-slate-400"}`} />
                      {item.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </nav>
        </div>

        <div className="pt-4 border-t border-slate-100 space-y-1.5">
          <div className="flex items-center gap-2 text-xs text-slate-600">
            <ShieldCheck className="w-4 h-4 text-[#9A4424] shrink-0" />
            <span>Assistance uniquement — aucune décision automatique</span>
          </div>
          <div className="text-[11px] text-slate-400">
            Analyste / AI Engineer · V1.0
          </div>
        </div>
      </aside>

      <main className="flex-1 p-6 lg:p-8 max-w-7xl mx-auto w-full">
        <TapisHeroBanner
          sectionLabel={meta.label}
          title={meta.title}
          subtitle={meta.subtitle}
          apiOnline={health?.status === "ok"}
          vllmConfigured={Boolean(health?.vllm_configured)}
        />

        {activeTab === "overview" && (
          <OverviewView
            cases={cases}
            runs={runs}
            inferenceMetrics={inferenceMetrics}
            onSelectCase={(id) => {
              setSelectedCaseId(id);
              setActiveTab("cases");
            }}
            onCreateFirstCase={() => setActiveTab("cases")}
          />
        )}

        {activeTab === "cases" && (
          <CaseWorkspaceView
            cases={cases}
            selectedCaseId={selectedCaseId}
            activeRun={activeRun}
            loading={loading}
            onSelectCase={setSelectedCaseId}
            onAnalyzeCase={handleAnalyzeCase}
            onUploadCase={handleUploadCase}
            onResumeHitl={handleResumeHitl}
            onOpenTrace={() => setActiveTab("executions")}
          />
        )}

        {activeTab === "executions" && (
          <ExecutionsView
            run={activeRun}
            onGoToHitl={() => {
              if (activeRun) setSelectedCaseId(activeRun.case_id);
              setActiveTab("cases");
            }}
          />
        )}

        {activeTab === "router" && <RouterView />}
        {activeTab === "studio" && (
          <StudioRLView
            jobs={jobs}
            onJobCreated={(job) => setJobs((prev) => [...prev, job])}
          />
        )}
        {activeTab === "gpu" && (
          <GpuInfraView health={health} metrics={inferenceMetrics} />
        )}
      </main>
    </div>
  );
}
