"""Discovery pipeline demo — mine correlation hypotheses, then evaluate one statistically."""

import random

from cds.hypothesis import EvaluationData, HypothesisEvaluator, mine_correlations


def main() -> None:
    # --- Synthetic dataset: one strong relationship plus pure-noise columns ---
    rng = random.Random(42)
    n_rows = 120
    dose = [rng.uniform(0.0, 10.0) for _ in range(n_rows)]
    response = [2.0 * d + rng.gauss(0.0, 1.5) for d in dose]
    data: dict[str, list[float]] = {
        "dose": dose,
        "response": response,
        "lab_temperature": [rng.gauss(21.0, 1.0) for _ in range(n_rows)],
        "instrument_drift": [rng.gauss(0.0, 2.0) for _ in range(n_rows)],
    }

    print("=== Mining correlation hypotheses ===")
    mined = mine_correlations(data)
    print(f"{len(mined)} hypothesis(ies) survived the |r| >= 0.3 and p < 0.05 filters:\n")
    for hit in mined:
        print(
            f"[{hit.strength}] {hit.feature_a} ~ {hit.feature_b}: "
            f"r={hit.correlation:+.4f}, p={hit.p_value:.3e}"
        )
        print(f"  {hit.hypothesis.statement}")
        print(f"  confidence: {hit.hypothesis.confidence:.2f}\n")

    # --- Evaluate the strongest mined hypothesis ---
    print("=== Evaluating the strongest mined hypothesis ===")
    print("The evaluator runs classical tests (t-test/ANOVA/chi-square), so the")
    print("association is translated into a group comparison: mean response below")
    print("vs above the median dose.")
    top = mined[0]
    doses = data[top.feature_a]
    responses = data[top.feature_b]
    median_dose = sorted(doses)[len(doses) // 2]
    low_group = [r for d, r in zip(doses, responses) if d <= median_dose]
    high_group = [r for d, r in zip(doses, responses) if d > median_dose]

    evaluator = HypothesisEvaluator(alpha=0.05)
    payload: EvaluationData = {"groups": [low_group, high_group]}
    result = evaluator.evaluate(top.hypothesis, payload)
    print(f"Test          : {result.test_name}")
    print(f"Statistic     : {result.statistic:.4f}")
    print(f"p-value       : {result.p_value:.3e}")
    print(f"Effect size   : {result.effect_size_label} = {result.effect_size:.3f}")
    print(f"Significant   : {result.is_significant}")
    print(f"Conclusion    : {result.conclusion}")
    print(f"Hypothesis status after evaluation: {top.hypothesis.status.value}")


if __name__ == "__main__":
    main()
