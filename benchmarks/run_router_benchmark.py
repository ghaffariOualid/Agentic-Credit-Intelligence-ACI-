"""Benchmark comparatif multi-seeds : Static vs Bandit vs Supervised vs PPO (Page 14).

Protocole expérimental conforme au cahier des charges ACI V1.0 :
- Même corpus / simulateur T4 (16 Go VRAM), mêmes ressources, charge reproductible.
- Plusieurs seeds pour mesurer la moyenne et l'écart-type (variabilité).
- Comparaison à qualité équivalente : coût GPU relatif, latence moyenne & P95, reward moyen.

Exécution :
    python -m benchmarks.run_router_benchmark
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from intelligence.decision_model.bandit import train_bandit_router
from intelligence.decision_model.features import TIER_LIST
from intelligence.decision_model.supervised import train_supervised_router
from intelligence.reinforcement_learning.env import ACIRoutingEnv
from intelligence.reinforcement_learning.ppo import PPOConfig, train_ppo_router
from services.contracts.intelligence import RouterState


@dataclass
class StrategyBenchmarkResult:
    strategy: str
    mean_quality: float
    std_quality: float
    mean_cost_usd: float
    std_cost_usd: float
    relative_cost_vs_static_large: float
    mean_latency_s: float
    p95_latency_s: float
    mean_reward: float
    std_reward: float


def _eval_detailed(
    policy_fn: Callable[[RouterState, np.ndarray], int],
    eval_episodes: int,
    seed: int,
) -> dict[str, float]:
    env = ACIRoutingEnv(seed=seed)
    rewards, qualities, costs, latencies = [], [], [], []
    for _ in range(eval_episodes):
        _, info = env.reset()
        done = False
        while not done:
            action = int(policy_fn(info["router_state"], info["action_mask"]))
            _, r, terminated, truncated, info = env.step(action)
            rewards.append(r)
            qualities.append(info["quality"])
            costs.append(info["cost_usd"])
            latencies.append(info["latency_s"])
            done = terminated or truncated

    return {
        "quality": float(np.mean(qualities)),
        "cost_usd": float(np.mean(costs)),
        "latency_s": float(np.mean(latencies)),
        "p95_latency_s": float(np.percentile(latencies, 95)),
        "reward": float(np.mean(rewards)),
    }


def run_benchmark(
    train_episodes: int = 120,
    eval_episodes: int = 40,
    seeds: tuple[int, ...] = (42, 101, 202),
    output_dir: Path | None = None,
) -> list[StrategyBenchmarkResult]:
    """Exécute la comparaison reproductible sur plusieurs seeds et exporte JSON + Markdown."""
    per_strategy: dict[str, list[dict[str, float]]] = {
        "Static-Small": [],
        "Static-Medium": [],
        "Static-Large": [],
        "Bandit (LinUCB)": [],
        "Supervised (PyTorch MLP)": [],
        "RL (PPO + Masking)": [],
    }

    for seed in seeds:
        eval_seed = seed + 5000

        # Baselines statiques (respectant toujours le masquage de sécurité)
        per_strategy["Static-Small"].append(
            _eval_detailed(lambda _s, m: 0 if m[0] else 1, eval_episodes, eval_seed)
        )
        per_strategy["Static-Medium"].append(
            _eval_detailed(lambda _s, _m: 1, eval_episodes, eval_seed)
        )
        per_strategy["Static-Large"].append(
            _eval_detailed(lambda _s, m: 2 if m[2] else 1, eval_episodes, eval_seed)
        )

        # Bandit contextuel LinUCB
        bandit, _ = train_bandit_router(num_episodes=train_episodes, seed=seed)
        per_strategy["Bandit (LinUCB)"].append(
            _eval_detailed(
                lambda s, _m, b=bandit: TIER_LIST.index(b.predict_tier(s)[0]),
                eval_episodes,
                eval_seed,
            )
        )

        # Routeur supervisé PyTorch
        sup_net, _ = train_supervised_router(num_episodes=train_episodes, epochs=15, seed=seed)
        per_strategy["Supervised (PyTorch MLP)"].append(
            _eval_detailed(
                lambda s, _m, m_net=sup_net: TIER_LIST.index(m_net.predict_tier(s)[0]),
                eval_episodes,
                eval_seed,
            )
        )

        # Politique RL PPO PyTorch avec action masking
        ppo_net, _ = train_ppo_router(
            config=PPOConfig(episodes=train_episodes, batch_episodes=16, seed=seed)
        )
        per_strategy["RL (PPO + Masking)"].append(
            _eval_detailed(
                lambda s, _m, p_net=ppo_net: TIER_LIST.index(p_net.predict_tier(s)[0]),
                eval_episodes,
                eval_seed,
            )
        )

    static_large_cost = float(np.mean([r["cost_usd"] for r in per_strategy["Static-Large"]]))

    results: list[StrategyBenchmarkResult] = []
    for name, runs in per_strategy.items():
        q = [r["quality"] for r in runs]
        c = [r["cost_usd"] for r in runs]
        lat = [r["latency_s"] for r in runs]
        p95 = [r["p95_latency_s"] for r in runs]
        rew = [r["reward"] for r in runs]
        mean_c = float(np.mean(c))
        results.append(
            StrategyBenchmarkResult(
                strategy=name,
                mean_quality=round(float(np.mean(q)), 4),
                std_quality=round(float(np.std(q)), 4),
                mean_cost_usd=round(mean_c, 4),
                std_cost_usd=round(float(np.std(c)), 4),
                relative_cost_vs_static_large=round(mean_c / static_large_cost, 4)
                if static_large_cost > 0
                else 1.0,
                mean_latency_s=round(float(np.mean(lat)), 3),
                p95_latency_s=round(float(np.mean(p95)), 3),
                mean_reward=round(float(np.mean(rew)), 4),
                std_reward=round(float(np.std(rew)), 4),
            )
        )

    target_dir = output_dir or (Path(__file__).resolve().parent / "results")
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / "router_benchmark.json"
    json_path.write_text(
        json.dumps(
            {
                "config": {
                    "gpu_profile": "nvidia-t4-16gb-colab",
                    "train_episodes": train_episodes,
                    "eval_episodes": eval_episodes,
                    "seeds": list(seeds),
                },
                "results": [asdict(r) for r in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    md_lines = [
        "# Benchmark comparatif du routeur ACI (profil GPU T4 16 Go)",
        "",
        f"- **Épisodes d'entraînement** : {train_episodes} | **Évaluation** : {eval_episodes} épisodes × {len(seeds)} seeds `{list(seeds)}`",
        "",
        "| Stratégie | Qualité (moy ± σ) | Coût ($) | Coût relatif (vs Large) | Latence moy (s) | Latence P95 (s) | Reward (moy ± σ) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        md_lines.append(
            f"| {r.strategy} | {r.mean_quality:.3f} ± {r.std_quality:.3f} | "
            f"{r.mean_cost_usd:.3f} | {r.relative_cost_vs_static_large:.2f} | "
            f"{r.mean_latency_s:.2f} | {r.p95_latency_s:.2f} | "
            f"{r.mean_reward:.3f} ± {r.std_reward:.3f} |"
        )
    md_text = "\n".join(md_lines) + "\n"
    (target_dir / "router_benchmark.md").write_text(md_text, encoding="utf-8")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark comparatif du routeur ACI (T4).")
    parser.add_argument("--train-episodes", type=int, default=120)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202])
    args = parser.parse_args()

    results = run_benchmark(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        seeds=tuple(args.seeds),
    )
    print("\n=== Résultats Benchmark ACI (Static vs Bandit vs Supervised vs PPO) ===")
    for r in results:
        print(
            f"{r.strategy:<26} | Q={r.mean_quality:.3f}±{r.std_quality:.3f} | "
            f"Coût rel={r.relative_cost_vs_static_large:.2f} | P95={r.p95_latency_s:.2f}s | "
            f"Reward={r.mean_reward:.3f}±{r.std_reward:.3f}"
        )


if __name__ == "__main__":
    main()
