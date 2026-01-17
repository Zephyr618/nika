import argparse
import csv
import os

from nika.net_env.net_env_pool import get_net_env_instance
from scripts.step1_net_env_start import start_net_env
from scripts.step2_failure_inject import inject_failure
from scripts.step3_agent_run import start_agent
from scripts.step4_result_eval import eval_results

cur_dir = os.path.dirname(os.path.abspath(__file__))


def run_single_benchmark(
    problem: str,
    scenario: str,
    topo_size: int,
    agent_type: str,
    backend_model: str,
    max_steps: int,
    judge_model: str,
    destroy_env: bool,
):
    """
    Run a single benchmark case.

    Args:
        problem: Name of the failure/problem to inject
        scenario: Network scenario name
        topo_size: Topology size
        agent_type: Agent type (e.g., react)
        backend_model: LLM backend model
        max_steps: Maximum agent steps
        judge_model: Model used for evaluation
        destroy_env: Whether to destroy the network environment after evaluation
    """

    print(f"Running benchmark for Problem: {problem}, Scenario: {scenario}, Topo Size: {topo_size}")

    # Step 1: Start network environment (always redeploy for single run)
    start_net_env(scenario, topo_size=topo_size, redeploy=True)

    # Step 2: Inject failure
    inject_failure(problem_names=[problem])

    # Step 3: Start agent
    start_agent(
        agent_type=agent_type,
        backend_model=backend_model,
        max_steps=max_steps,
    )

    # Step 4: Evaluate results
    eval_results(judge_model=judge_model, destroy_env=destroy_env)

    # Step 5: Cleanup environment if required
    if destroy_env:
        net_env = get_net_env_instance(scenario, topo_size=topo_size)
        if net_env.lab_exists():
            net_env.undeploy()


def run_benchmark_from_csv(
    benchmark_file: str,
    agent_type: str,
    backend_model: str,
    max_steps: int,
    judge_model: str,
    destroy_env: bool,
):
    """
    Run benchmark cases defined in a CSV file.

    The CSV file must contain the following columns:
    - problem
    - scenario
    - topo_size
    """

    with open(benchmark_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            run_single_benchmark(
                problem=row["problem"],
                scenario=row["scenario"],
                topo_size=int(row["topo_size"]),
                agent_type=agent_type,
                backend_model=backend_model,
                max_steps=max_steps,
                judge_model=judge_model,
                destroy_env=destroy_env,
            )


def main():
    """
    Entry point for the benchmark runner.
    Supports both single-case execution and CSV-based batch execution.
    """

    parser = argparse.ArgumentParser(description="Run network benchmark tests")

    # ===== Execution mode =====
    parser.add_argument(
        "--benchmark-csv",
        type=str,
        default=os.path.join(cur_dir, "benchmark.csv"),
        help="Path to benchmark CSV file",
    )

    parser.add_argument("--problem", type=str, help="Problem name")
    parser.add_argument("--scenario", type=str, help="Scenario name")
    parser.add_argument("--topo-size", type=str, help="Topology size")

    # ===== Agent configuration =====
    parser.add_argument("--agent-type", type=str, default="react")
    parser.add_argument("--backend-model", type=str, default="deepseek-chat")
    parser.add_argument("--max-steps", type=int, default=20)

    # ===== Evaluation configuration =====
    parser.add_argument("--judge-model", type=str, default="deepseek-chat")
    parser.add_argument(
        "--destroy-env",
        action="store_true",
        help="Destroy the network environment after evaluation",
    )

    args = parser.parse_args()

    # Determine execution mode
    if args.problem and args.scenario and args.topo_size:
        # Single benchmark execution
        run_single_benchmark(
            problem=args.problem,
            scenario=args.scenario,
            topo_size=args.topo_size,
            agent_type=args.agent_type,
            backend_model=args.backend_model,
            max_steps=args.max_steps,
            judge_model=args.judge_model,
            destroy_env=args.destroy_env,
        )
    else:
        # CSV-based batch execution
        run_benchmark_from_csv(
            benchmark_file=args.benchmark_csv,
            agent_type=args.agent_type,
            backend_model=args.backend_model,
            max_steps=args.max_steps,
            judge_model=args.judge_model,
            destroy_env=args.destroy_env,
        )


if __name__ == "__main__":
    main()
