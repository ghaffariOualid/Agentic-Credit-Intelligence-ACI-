"use client";

import React, { useState } from "react";
import { CheckCircle2, PauseCircle, Clock, ArrowRight, GitBranch } from "lucide-react";
import { AgentStepEvent, RunRecord } from "@/lib/api";

interface ExecutionsViewProps {
  run: RunRecord | null;
  onGoToHitl: () => void;
}

const ALL_NODES = [
  "Document Agent",
  "Financial Agent",
  "Risk + Compliance",
  "Verification Agent",
  "Report Agent",
];

export function ExecutionsView({ run, onGoToHitl }: ExecutionsViewProps) {
  const [selectedIdx, setSelectedIdx] = useState(0);

  if (!run) {
    return (
      <div className="bank-card p-12 text-center space-y-3">
        <div className="w-11 h-11 rounded-xl bg-slate-100 border border-slate-200 mx-auto flex items-center justify-center">
          <GitBranch className="w-5 h-5 text-slate-700" />
        </div>
        <h2 className="text-base font-semibold text-slate-900">Aucune exécution enregistrée</h2>
        <p className="text-sm text-slate-500 max-w-md mx-auto">
          Importez un dossier dans <strong>Dossiers de crédit</strong> puis cliquez sur{" "}
          <strong>Lancer l&apos;analyse</strong> pour suivre la trace LangGraph ici.
        </p>
      </div>
    );
  }

  const safeIdx = Math.min(selectedIdx, Math.max(0, run.events.length - 1));
  const selectedEvent: AgentStepEvent | undefined =
    run.events[safeIdx] || run.events[run.events.length - 1];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
      <div className="bank-card p-6 lg:col-span-7">
        <div className="flex items-center justify-between mb-5">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Graphe d&apos;exécution — {run.run_id}
            </h2>
            <p className="text-xs text-slate-500">
              Dossier {run.case_id} · workflow {run.workflow_version} · politique{" "}
              <span className="font-semibold text-slate-800">{run.policy_version}</span>
            </p>
          </div>
          <span
            className={`px-3 py-1 rounded-full text-xs font-medium border ${
              run.status === "completed"
                ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                : run.status === "interrupted"
                ? "bg-amber-50 text-amber-800 border-amber-200"
                : "bg-slate-100 text-slate-700 border-slate-200"
            }`}
          >
            {run.status === "completed"
              ? "Terminé"
              : run.status === "interrupted"
              ? "Interruption HITL"
              : "En cours"}
          </span>
        </div>

        <div className="space-y-3">
          {ALL_NODES.map((nodeName, idx) => {
            const evIdx = run.events.findIndex((e) => e.node === nodeName);
            const ev = evIdx >= 0 ? run.events[evIdx] : null;
            const isSelected = evIdx >= 0 && evIdx === safeIdx;
            const status = ev?.status || "pending";
            const style =
              status === "completed"
                ? "bg-emerald-50/50 border-emerald-200/80"
                : status === "interrupted"
                ? "bg-amber-50/70 border-amber-200"
                : "bg-slate-50 border-slate-200 opacity-70";

            return (
              <div key={nodeName}>
                <button
                  disabled={!ev}
                  onClick={() => evIdx >= 0 && setSelectedIdx(evIdx)}
                  className={`w-full text-left p-4 rounded-xl border transition-all flex items-center justify-between cursor-pointer ${style} ${
                    isSelected ? "ring-2 ring-slate-900" : ""
                  }`}
                >
                  <div className="flex items-center gap-3">
                    {status === "completed" ? (
                      <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />
                    ) : status === "interrupted" ? (
                      <PauseCircle className="w-5 h-5 text-amber-600 shrink-0" />
                    ) : (
                      <Clock className="w-5 h-5 text-slate-400 shrink-0" />
                    )}
                    <div>
                      <div className="font-semibold text-sm text-slate-900">{nodeName}</div>
                      <div className="text-xs text-slate-500">
                        {ev
                          ? `${ev.tier} · ${(ev.latency_ms / 1000).toFixed(2)} s · ${ev.tokens} tokens`
                          : "En attente"}
                      </div>
                    </div>
                  </div>
                  <span className="text-xs font-medium text-slate-500">
                    {status === "completed"
                      ? "Terminé"
                      : status === "interrupted"
                      ? "Contrôle requis"
                      : "En attente"}
                  </span>
                </button>
                {idx < ALL_NODES.length - 1 && (
                  <div className="ml-6 h-3 w-px bg-slate-200" />
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="bank-card p-6 lg:col-span-5 flex flex-col justify-between">
        <div>
          <h3 className="text-sm font-semibold text-slate-900 mb-4">Inspecteur d&apos;étape</h3>
          {selectedEvent ? (
            <div className="space-y-4">
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-500">Agent</span>
                  <span className="font-semibold text-slate-900">{selectedEvent.node}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Modèle résolu</span>
                  <span className="font-mono text-slate-900">{selectedEvent.model}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Backend</span>
                  <span className="font-semibold text-slate-800">{selectedEvent.backend}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Politique active</span>
                  <span className="font-semibold text-slate-900">{selectedEvent.policy}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Latence / Tokens</span>
                  <span className="font-semibold text-slate-900">
                    {selectedEvent.latency_ms.toFixed(0)} ms · {selectedEvent.tokens} tok
                  </span>
                </div>
              </div>
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
                  Trace & motif
                </div>
                <p className="text-sm text-slate-700 leading-relaxed">{selectedEvent.message}</p>
              </div>
            </div>
          ) : (
            <p className="text-sm text-slate-500">Sélectionnez un nœud du graphe.</p>
          )}
        </div>

        {run.status === "interrupted" && (
          <button onClick={onGoToHitl} className="btn-primary mt-6 w-full py-2.5 text-sm">
            Ouvrir le contrôle humain (HITL)
            <ArrowRight className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}
