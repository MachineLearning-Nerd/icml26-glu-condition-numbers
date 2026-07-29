"""Independent checker for serialized real-architecture NTK evidence."""

from __future__ import annotations


def check(result: dict) -> tuple[bool, list[str]]:
    failures = []
    summaries = result["paired_summaries"]
    if len(summaries) != 6:
        failures.append(f"expected six architecture/activation summaries, found {len(summaries)}")
    for summary in summaries:
        label = f"{summary['architecture']}:{'/'.join(summary['pair'])}"
        if summary["seed_total"] != 4:
            failures.append(f"{label}: incomplete seed set")
        if summary["seed_wins"] < 3:
            failures.append(f"{label}: GLU wins only {summary['seed_wins']}/4 seeds")
        if summary["geometric_mean_ratio_glu_over_non"] >= 1.0:
            failures.append(f"{label}: geometric mean does not favor GLU")

    by_key = {
        (row["architecture"], row["seed"], row["activation"]): row
        for row in result["rows"]
    }
    for architecture in ("vit", "gpt2"):
        for seed in (41, 42, 43, 44):
            for non_name, glu_name in (("relu", "reglu"), ("gelu", "geglu"), ("silu", "swiglu")):
                non = by_key[(architecture, seed, non_name)]
                glu = by_key[(architecture, seed, glu_name)]
                relative_difference = abs(
                    non["parameter_count"] - glu["parameter_count"]
                ) / non["parameter_count"]
                if relative_difference > 0.005:
                    failures.append(
                        f"{architecture} seed {seed} {non_name}/{glu_name}: "
                        f"FFN parameter mismatch {relative_difference:.3%}"
                    )

    control = result["negative_control"]
    if control["ratio"] != 1.0 or control["expected_reduction"] is not False:
        failures.append("no-change negative control did not fail for the intended reason")
    return not failures, failures
