"use client";

import React, { useState } from "react";
import { Play, Loader2, Award, FlaskConical } from "lucide-react";
import { TrainingJobRecord, apiFetch } from "@/lib/api";

interface StudioRLViewProps {
  jobs: TrainingJobRecord[];
  onJobCreated: (job: TrainingJobRecord) => void;
}

export function StudioRLView({ jobs, onJobCreated }: StudioRLViewProps) {
  const [algo, setAlgo] = useState<"ppo" | "supervised" | "bandit">("ppo");
  const [episodes, setEpisodes] = useState(64);
  const [seed, setSeed] = useState(42);
  const [activate, setActivate] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const latest = jobs.length > 0 ? jobs[jobs.length - 1] : null;

  async function launchTraining() {
    setRunning(true);
    setError(null);
    try {
      const rec = await apiFetch<TrainingJobRecord>("/api/v1/training/jobs", {
        method: "POST",
        body: JSON.stringify({
          algorithm: algo,
          episodes,
          seed,
          activate_policy: activate,
        }),
      });
      onJobCreated(rec);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Backend FastAPI non joignable.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="space-y-5">
      <div className="bank-card p-5 space-y-3">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div className="flex flex-wrap items-end gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">Algorithme</label>
              <select
                value={algo}
                onChange={(e) => setAlgo(e.target.value as "ppo" | "supervised" | "bandit")}
                className="bank-input"
              >
                <option value="ppo">RL — PPO + Action Masking (PyTorch)</option>
                <option value="supervised">Supervisé — MLP PyTorch</option>
                <option value="bandit">Bandit contextuel — LinUCB</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">Épisodes</label>
              <input
                type="number"
                min={16}
                max={5000}
                value={episodes}
                onChange={(e) => setEpisodes(parseInt(e.target.value, 10) || 64)}
                className="bank-input w-28"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">Seed</label>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value, 10) || 42)}
                className="bank-input w-24"
              />
            </div>
            <label className="inline-flex items-center gap-2 text-xs text-slate-700 pb-2 cursor-pointer">
              <input
                type="checkbox"
                checked={activate}
                onChange={(e) => setActivate(e.target.checked)}
                className="accent-slate-900"
              />
              Activer la politique après validation
            </label>
          </div>

          <button onClick={launchTraining} disabled={running} className="btn-primary">
            {running ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
            Lancer un entraînement
          </button>
        </div>
        {error && <p className="text-xs text-rose-700 font-medium">{error}</p>}
      </div>

      {!latest ? (
        <div className="bank-card p-12 text-center space-y-3">
          <div className="w-11 h-11 rounded-xl bg-slate-100 border border-slate-200 mx-auto flex items-center justify-center">
            <FlaskConical className="w-5 h-5 text-slate-700" />
          </div>
          <h2 className="text-base font-semibold text-slate-900">Aucun entraînement exécuté</h2>
          <p className="text-sm text-slate-500 max-w-md mx-auto">
            Configurez un algorithme et un budget d&apos;épisodes ci-dessus, puis lancez votre
            première expérience hors production.
          </p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bank-card p-4">
              <div className="text-xs text-slate-500">Dernière expérience</div>
              <div className="text-2xl font-semibold text-slate-900 mt-1">{latest.job_id}</div>
              <div className="text-xs text-slate-500 mt-1 uppercase font-semibold">{latest.algorithm}</div>
            </div>
            <div className="bank-card p-4">
              <div className="text-xs text-slate-500">Reward moyen</div>
              <div className="text-2xl font-semibold text-slate-900 mt-1">
                {latest.mean_reward.toFixed(3)}
              </div>
              <div className="text-xs text-emerald-700 mt-1">Validation simulée T4</div>
            </div>
            <div className="bank-card p-4">
              <div className="text-xs text-slate-500">Qualité moyenne</div>
              <div className="text-2xl font-semibold text-slate-900 mt-1">
                {(latest.mean_quality * 100).toFixed(1)} %
              </div>
              <div className="text-xs text-slate-400 mt-1">{latest.episodes} épisodes</div>
            </div>
            <div className="bank-card p-4">
              <div className="text-xs text-slate-500">Coût relatif (vs Large)</div>
              <div className="text-2xl font-semibold text-emerald-700 mt-1">
                {latest.relative_cost.toFixed(2)}×
              </div>
              <div className="text-xs text-slate-400 mt-1">
                {latest.deployed ? "Politique active" : "Évaluation requise"}
              </div>
            </div>
          </div>

          <div className="bank-card overflow-hidden">
            <div className="px-5 py-3.5 border-b border-slate-100 flex items-center gap-2 text-sm font-semibold text-slate-900">
              <Award className="w-4 h-4 text-slate-500" />
              Historique de vos entraînements
            </div>
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50/80 text-[11px] uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="px-5 py-3">Run</th>
                  <th className="px-5 py-3">Méthode</th>
                  <th className="px-5 py-3">Politique</th>
                  <th className="px-5 py-3">Qualité</th>
                  <th className="px-5 py-3">Coût relatif</th>
                  <th className="px-5 py-3">Reward</th>
                  <th className="px-5 py-3">Statut</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {jobs.map((j) => (
                  <tr key={j.job_id}>
                    <td className="px-5 py-3 font-semibold text-slate-900">{j.job_id}</td>
                    <td className="px-5 py-3 uppercase text-xs font-semibold text-slate-700">{j.algorithm}</td>
                    <td className="px-5 py-3 font-mono text-xs text-slate-700">{j.policy_version}</td>
                    <td className="px-5 py-3 text-slate-700">{(j.mean_quality * 100).toFixed(1)} %</td>
                    <td className="px-5 py-3 font-semibold text-emerald-700">{j.relative_cost.toFixed(2)}</td>
                    <td className="px-5 py-3 font-semibold text-slate-900">{j.mean_reward.toFixed(3)}</td>
                    <td className="px-5 py-3">
                      <span
                        className={`px-2.5 py-0.5 rounded-full text-xs font-medium border ${
                          j.deployed
                            ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                            : "bg-slate-50 text-slate-600 border-slate-200"
                        }`}
                      >
                        {j.deployed ? "Active" : "À vérifier"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
