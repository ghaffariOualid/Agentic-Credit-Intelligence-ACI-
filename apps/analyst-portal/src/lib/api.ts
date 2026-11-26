/**
 * Client API typé vers le backend FastAPI ACI (http://127.0.0.1:8000).
 * Aucune donnée fictive : toutes les données proviennent des saisies utilisateur et de l'API.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_ACI_API_URL || "http://127.0.0.1:8000";

export interface DocumentRef {
  document_id: string;
  filename: string;
  doc_type: string;
  status: "pending" | "validated" | "needs_review" | "rejected";
  page_or_sheet_count: number;
}

export interface CreditCase {
  case_id: string;
  company_name: string;
  sector: string;
  facility_type: string;
  requested_amount: number;
  currency: string;
  tenor_months: number;
  status: "draft" | "analyzing" | "needs_review" | "completed" | "failed";
  extraction_quality?: number | null;
  documents: DocumentRef[];
}

export interface AgentStepEvent {
  node: string;
  status: "completed" | "interrupted" | "pending" | "failed";
  model: string;
  tier: string;
  backend: string;
  tokens: number;
  latency_ms: number;
  policy: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface FinancialMetrics {
  case_id: string;
  fiscal_year: string;
  currency: string;
  revenue: number;
  ebitda: number;
  net_income: number;
  total_debt: number;
  equity: number;
  net_debt: number;
  debt_to_equity: number | null;
  net_debt_to_ebitda: number | null;
  net_margin: number | null;
  current_ratio: number | null;
  dscr: number | null;
  debt_reconciliation_diff: number | null;
  is_debt_consistent: boolean;
  warnings: string[];
}

export interface RunRecord {
  run_id: string;
  case_id: string;
  workflow_version: string;
  policy_version: string;
  status: "running" | "interrupted" | "completed" | "failed";
  events: AgentStepEvent[];
  metrics: FinancialMetrics | null;
  verification: {
    passed: boolean;
    blocking_errors: string[];
    requires_hitl: boolean;
    suspension_reason: string | null;
  } | null;
  interrupt_payload: {
    node?: string;
    suspension_reason?: string;
    blocking_errors?: string[];
    action_required?: string;
  } | null;
  report: {
    version: number;
    disclaimer: string;
  } | null;
}

export interface RouterCandidate {
  tier: "small" | "medium" | "large";
  predicted_quality: number;
  predicted_latency_s: number;
  predicted_cost_usd: number;
  score: number;
  allowed: boolean;
}

export interface RouterDecision {
  decision_id: string;
  policy_version: string;
  action: {
    selected_tier: "small" | "medium" | "large";
    run_verification: boolean;
    masked_actions: string[];
  };
  candidates: RouterCandidate[];
  rationale: string;
  constraints_applied: string[];
}

export interface TrainingJobRecord {
  job_id: string;
  algorithm: "supervised" | "bandit" | "ppo";
  policy_version: string;
  episodes: number;
  mean_reward: number;
  mean_quality: number;
  mean_cost_usd: number;
  relative_cost: number;
  reward_curve: number[];
  artifact_path: string | null;
  deployed: boolean;
  status: "completed" | "needs_review";
}

export interface InferenceMetricsSnapshot {
  total_requests: number;
  requests_by_tier: Record<string, number>;
  fallback_count: number;
  error_count: number;
  avg_latency_ms: number;
  in_flight: number;
  last_error: string | null;
}

export interface HealthResponse {
  status: string;
  cases_loaded: number;
  runs_count: number;
  inference_mode: string;
  vllm_configured: boolean;
  models?: {
    small: string;
    medium: string;
    large: string;
  };
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export function formatMAD(value: number): string {
  if (Math.abs(value) >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2).replace(".", ",")} M`;
  }
  if (Math.abs(value) >= 1_000) {
    return `${Math.round(value / 1_000).toLocaleString("fr-FR")} k`;
  }
  return value.toLocaleString("fr-FR");
}
