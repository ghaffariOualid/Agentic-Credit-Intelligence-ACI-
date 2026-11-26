"use client";

import React, { useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  FileText,
  Play,
  ShieldCheck,
  Loader2,
  Upload,
  X,
} from "lucide-react";
import { API_BASE, CreditCase, RunRecord, formatMAD } from "@/lib/api";

interface CaseWorkspaceViewProps {
  cases: CreditCase[];
  selectedCaseId: string;
  activeRun: RunRecord | null;
  loading: boolean;
  onSelectCase: (caseId: string) => void;
  onAnalyzeCase: (caseId: string) => Promise<void>;
  onUploadCase: (formData: FormData) => Promise<void>;
  onResumeHitl: (
    runId: string,
    payload: {
      reviewer_id: string;
      action: "approve" | "correct" | "request_correction";
      comment: string;
      field_overrides: Record<string, number>;
    }
  ) => Promise<void>;
  onOpenTrace: () => void;
}

export function CaseWorkspaceView({
  cases,
  selectedCaseId,
  activeRun,
  loading,
  onSelectCase,
  onAnalyzeCase,
  onUploadCase,
  onResumeHitl,
  onOpenTrace,
}: CaseWorkspaceViewProps) {
  const currentCase = cases.find((c) => c.case_id === selectedCaseId) || cases[0] || null;
  const runForCase = activeRun && currentCase && activeRun.case_id === currentCase.case_id ? activeRun : null;

  const [reviewerId, setReviewerId] = useState("");
  const [correctedDebt, setCorrectedDebt] = useState("");
  const [comment, setComment] = useState("");
  const [showUpload, setShowUpload] = useState(cases.length === 0);

  const [upCaseId, setUpCaseId] = useState("");
  const [upCompany, setUpCompany] = useState("");
  const [upSector, setUpSector] = useState("");
  const [upAmount, setUpAmount] = useState("");
  const [upCurrency, setUpCurrency] = useState("MAD");
  const [upTenor, setUpTenor] = useState("60");
  const [upFiles, setUpFiles] = useState<FileList | null>(null);
  const [upError, setUpError] = useState<string | null>(null);

  const metrics = runForCase?.metrics;
  const isInterrupted = runForCase?.status === "interrupted";
  const isCompleted = runForCase?.status === "completed";

  async function handleUploadSubmit(e: React.FormEvent) {
    e.preventDefault();
    setUpError(null);
    if (!upCaseId.trim() || !upCompany.trim() || !upAmount || !upFiles || upFiles.length === 0) {
      setUpError(
        "Veuillez remplir l'identifiant du dossier, l'entreprise, le montant et joindre au moins un fichier (PDF, XLSX ou TXT)."
      );
      return;
    }
    const fd = new FormData();
    fd.append("case_id", upCaseId.trim());
    fd.append("company_name", upCompany.trim());
    fd.append("sector", upSector.trim() || "Entreprise");
    fd.append("requested_amount", upAmount);
    fd.append("tenor_months", upTenor || "60");
    fd.append("currency", upCurrency);
    Array.from(upFiles).forEach((f) => fd.append("files", f));

    try {
      await onUploadCase(fd);
      setShowUpload(false);
      setUpCaseId("");
      setUpCompany("");
      setUpSector("");
      setUpAmount("");
      setUpFiles(null);
    } catch (err) {
      setUpError(err instanceof Error ? err.message : "Échec de l'importation.");
    }
  }

  return (
    <div className="space-y-5">
      <div className="bank-card p-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          {cases.length === 0 ? (
            <span className="text-xs text-slate-500">
              Aucun dossier créé — utilisez le formulaire ci-dessous pour importer vos pièces.
            </span>
          ) : (
            cases.map((c) => {
              const active = currentCase && c.case_id === currentCase.case_id;
              return (
                <button
                  key={c.case_id}
                  onClick={() => onSelectCase(c.case_id)}
                  className={`px-3.5 py-2 rounded-lg text-xs font-semibold transition-all cursor-pointer border ${
                    active
                      ? "bg-slate-900 text-white border-slate-900"
                      : "bg-slate-50 text-slate-700 border-slate-200 hover:border-slate-400"
                  }`}
                >
                  {c.case_id} · {c.company_name}
                </button>
              );
            })
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button onClick={() => setShowUpload((v) => !v)} className="btn-secondary">
            <Upload className="w-3.5 h-3.5 text-slate-600" />
            Nouveau dossier
          </button>
          {runForCase && (
            <button onClick={onOpenTrace} className="btn-secondary">
              Voir le graphe ({runForCase.run_id})
            </button>
          )}
          {currentCase && (
            <button
              onClick={() => onAnalyzeCase(currentCase.case_id)}
              disabled={loading}
              className="btn-primary"
            >
              {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
              Lancer l&apos;analyse
            </button>
          )}
        </div>
      </div>

      {(showUpload || cases.length === 0) && (
        <form onSubmit={handleUploadSubmit} className="bank-card p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">
                Créer un dossier & importer vos pièces (PDF, Excel, TXT)
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Renseignez le dossier et sélectionnez vos bilans, comptes de résultat ou tableaux de dettes.
              </p>
            </div>
            {cases.length > 0 && (
              <button
                type="button"
                onClick={() => setShowUpload(false)}
                className="p-1 text-slate-400 hover:text-slate-700 cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-6 gap-3.5 text-xs">
            <div>
              <label className="block font-medium text-slate-700 mb-1">ID Dossier *</label>
              <input
                value={upCaseId}
                onChange={(e) => setUpCaseId(e.target.value)}
                placeholder="Ex. CR-2026-01"
                className="bank-input"
                required
              />
            </div>
            <div className="md:col-span-2">
              <label className="block font-medium text-slate-700 mb-1">Entreprise *</label>
              <input
                value={upCompany}
                onChange={(e) => setUpCompany(e.target.value)}
                placeholder="Raison sociale"
                className="bank-input"
                required
              />
            </div>
            <div>
              <label className="block font-medium text-slate-700 mb-1">Secteur</label>
              <input
                value={upSector}
                onChange={(e) => setUpSector(e.target.value)}
                placeholder="Ex. Industrie"
                className="bank-input"
              />
            </div>
            <div>
              <label className="block font-medium text-slate-700 mb-1">Montant *</label>
              <div className="flex gap-1.5">
                <input
                  type="number"
                  value={upAmount}
                  onChange={(e) => setUpAmount(e.target.value)}
                  placeholder="2000000"
                  className="bank-input"
                  required
                />
                <select
                  value={upCurrency}
                  onChange={(e) => setUpCurrency(e.target.value)}
                  className="rounded-lg border border-slate-300 bg-white px-2 text-xs text-slate-900"
                >
                  <option value="MAD">MAD</option>
                  <option value="EUR">EUR</option>
                  <option value="USD">USD</option>
                </select>
              </div>
            </div>
            <div>
              <label className="block font-medium text-slate-700 mb-1">Durée (mois)</label>
              <input
                type="number"
                value={upTenor}
                onChange={(e) => setUpTenor(e.target.value)}
                className="bank-input"
                required
              />
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-100">
            <input
              type="file"
              multiple
              accept=".pdf,.xlsx,.txt,.csv"
              onChange={(e) => setUpFiles(e.target.files)}
              className="text-xs text-slate-600 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border file:border-slate-300 file:bg-slate-50 file:text-slate-800 file:font-semibold hover:file:bg-slate-100 file:cursor-pointer"
              required
            />
            <button type="submit" disabled={loading} className="btn-primary">
              <Upload className="w-3.5 h-3.5" />
              Importer & Créer le dossier
            </button>
          </div>
          {upError && <p className="text-xs text-rose-700 font-medium">{upError}</p>}
        </form>
      )}

      {currentCase && (
        <>
          <div className="bank-card p-5 flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2.5">
                <h2 className="text-lg font-semibold text-slate-900">
                  {currentCase.case_id} — {currentCase.company_name}
                </h2>
                <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-600 font-medium">
                  {currentCase.sector}
                </span>
              </div>
              <p className="text-sm text-slate-500 mt-1">
                {currentCase.facility_type} ·{" "}
                <strong className="text-slate-900">
                  {currentCase.requested_amount.toLocaleString("fr-FR")} {currentCase.currency}
                </strong>{" "}
                · {currentCase.tenor_months} mois
              </p>
            </div>
            <div className="text-right text-xs text-slate-500">
              <div>Gouvernance ACI</div>
              <div className="font-semibold text-slate-800">
                Décision finale réservée à l&apos;analyste
              </div>
            </div>
          </div>

          {isInterrupted && (
            <div className="rounded-xl border border-amber-200 bg-amber-50/90 p-4 flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-700 shrink-0 mt-0.5" />
              <div className="text-sm">
                <div className="font-semibold text-amber-900">
                  Contrôle humain requis — Exécution suspendue sur Verification Agent
                </div>
                <p className="text-amber-800 mt-0.5">
                  {runForCase.interrupt_payload?.suspension_reason ||
                    "Vérification manuelle requise avant génération du rapport."}
                </p>
              </div>
            </div>
          )}

          {isCompleted && (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50/80 p-4 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-emerald-700 shrink-0 mt-0.5" />
                <div className="text-sm">
                  <div className="font-semibold text-emerald-950">
                    Analyse validée — Rapport versionné v{runForCase.report?.version || 1} prêt
                  </div>
                  <p className="text-emerald-800 mt-0.5">{runForCase.report?.disclaimer}</p>
                </div>
              </div>
              <a
                href={`${API_BASE}/api/v1/cases/${currentCase.case_id}/report/pdf`}
                target="_blank"
                rel="noopener noreferrer"
                className="btn-primary"
              >
                <Download className="w-3.5 h-3.5" />
                Télécharger le rapport PDF
              </a>
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bank-card p-4">
              <div className="text-xs text-slate-500">Chiffre d&apos;affaires</div>
              <div className="text-2xl font-semibold text-slate-900 mt-1">
                {metrics ? `${formatMAD(metrics.revenue)} ${metrics.currency}` : "—"}
              </div>
              <div className="text-xs text-slate-400 mt-1">
                {metrics ? `Exercice ${metrics.fiscal_year}` : "Lancez l'analyse"}
              </div>
            </div>

            <div className="bank-card p-4">
              <div className="text-xs text-slate-500">Résultat net</div>
              <div className="text-2xl font-semibold text-slate-900 mt-1">
                {metrics ? `${formatMAD(metrics.net_income)} ${metrics.currency}` : "—"}
              </div>
              <div className="text-xs text-slate-400 mt-1">
                {metrics?.net_margin != null ? `Marge ${(metrics.net_margin * 100).toFixed(1)} %` : "—"}
              </div>
            </div>

            <div className="bank-card p-4">
              <div className="text-xs text-slate-500">Dette / Capitaux propres</div>
              <div className="text-2xl font-semibold text-slate-900 mt-1">
                {metrics?.debt_to_equity != null
                  ? metrics.debt_to_equity.toFixed(2).replace(".", ",")
                  : "—"}
              </div>
              <div className="text-xs text-slate-400 mt-1">Calcul déterministe</div>
            </div>

            <div className="bank-card p-4">
              <div className="text-xs text-slate-500">DSCR</div>
              <div className="text-2xl font-semibold text-slate-900 mt-1">
                {metrics?.dscr != null ? `${metrics.dscr.toFixed(2).replace(".", ",")}×` : "—"}
              </div>
              <div className="text-xs text-slate-400 mt-1">Seuil interne ≥ 1,20×</div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <div className="bank-card p-5">
              <h3 className="text-sm font-semibold text-slate-900 mb-1">
                Pièces importées ({currentCase.documents.length})
              </h3>
              <p className="text-xs text-slate-500 mb-4">
                Fichiers isolés pour le dossier {currentCase.case_id}
              </p>
              <div className="divide-y divide-slate-100">
                {currentCase.documents.map((doc) => (
                  <div key={doc.document_id} className="py-3 flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2.5">
                      {doc.filename.endsWith(".xlsx") ? (
                        <FileSpreadsheet className="w-4 h-4 text-emerald-700" />
                      ) : (
                        <FileText className="w-4 h-4 text-slate-500" />
                      )}
                      <div>
                        <div className="font-medium text-slate-900">{doc.filename}</div>
                        <div className="text-xs text-slate-400">
                          {doc.page_or_sheet_count} {doc.filename.endsWith(".xlsx") ? "feuilles" : "pages"}
                        </div>
                      </div>
                    </div>
                    <span className="px-2.5 py-0.5 rounded-full text-xs font-medium border bg-emerald-50 text-emerald-700 border-emerald-200">
                      {doc.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bank-card p-5 flex flex-col justify-between">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <ShieldCheck className="w-4 h-4 text-slate-700" />
                  <h3 className="text-sm font-semibold text-slate-900">
                    Contrôle humain (HITL) & reprise
                  </h3>
                </div>
                <p className="text-xs text-slate-500 mb-4">
                  Disponible lorsque le Verification Agent suspend le workflow.
                </p>

                <div className="space-y-3">
                  <div>
                    <label className="block text-xs font-medium text-slate-700 mb-1">
                      Identifiant analyste
                    </label>
                    <input
                      value={reviewerId}
                      onChange={(e) => setReviewerId(e.target.value)}
                      placeholder="Ex. oualid.analyste"
                      className="bank-input"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-slate-700 mb-1">
                      Correction optionnelle (`debt_schedule_total`, {currentCase.currency})
                    </label>
                    <input
                      type="number"
                      value={correctedDebt}
                      onChange={(e) => setCorrectedDebt(e.target.value)}
                      placeholder="Laisser vide si aucune correction chiffrée"
                      className="bank-input"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-slate-700 mb-1">
                      Commentaire d&apos;audit
                    </label>
                    <textarea
                      rows={2}
                      value={comment}
                      onChange={(e) => setComment(e.target.value)}
                      placeholder="Saisissez votre justification..."
                      className="bank-input"
                    />
                  </div>
                </div>
              </div>

              <div className="mt-5 flex flex-wrap gap-2 justify-end">
                <button
                  disabled={!isInterrupted || loading || !correctedDebt}
                  onClick={() =>
                    runForCase &&
                    onResumeHitl(runForCase.run_id, {
                      reviewer_id: reviewerId.trim() || "analyste",
                      action: "correct",
                      comment,
                      field_overrides: { debt_schedule_total: Number(correctedDebt) },
                    })
                  }
                  className="btn-secondary disabled:opacity-40"
                >
                  Corriger & Reprendre
                </button>
                <button
                  disabled={!isInterrupted || loading}
                  onClick={() =>
                    runForCase &&
                    onResumeHitl(runForCase.run_id, {
                      reviewer_id: reviewerId.trim() || "analyste",
                      action: "approve",
                      comment,
                      field_overrides: {},
                    })
                  }
                  className="btn-primary"
                >
                  Valider le rapport
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
