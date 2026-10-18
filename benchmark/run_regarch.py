from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

import regarch.gene_regulatory_network as grn_module
from regarch.cell_fate_map import FateMap, load_fate_map_topology, parse_edge_dict
from regarch.gene_regulatory_network import GRNInference


def _odds_scale(probabilities: pd.Series, multiplier: float) -> pd.Series:
    """Scale probability odds without changing their rank or clipping."""
    if multiplier <= 0:
        raise ValueError(f"odds multiplier must be positive; got {multiplier}")
    values = probabilities.astype(float)
    if ((values < 0) | (values > 1)).any():
        raise ValueError("prior_score values must lie in [0, 1]")
    numerator = multiplier * values
    return numerator / (1.0 - values + numerator)


def _write_alt_prior(
    prior_path: Path,
    output_path: Path,
    root_odds_multiplier: float = 1.0,
) -> tuple[float, float]:
    prior = pd.read_csv(prior_path, sep="\t")
    required = {"node", "regulator", "target", "prior_score"}
    missing = required - set(prior)
    if missing:
        raise ValueError(f"{prior_path} is missing columns: {sorted(missing)}")
    if "Root" not in set(prior["node"]):
        root = (
            prior.groupby(["regulator", "target"], as_index=False)["prior_score"]
            .mean()
            .assign(node="Root")
        )
        prior = pd.concat([prior, root], ignore_index=True)
    root_mask = prior["node"].eq("Root")
    if not root_mask.any():
        raise ValueError(f"{prior_path} does not define a Root prior")
    root_mean_before = float(prior.loc[root_mask, "prior_score"].mean())
    prior.loc[root_mask, "prior_score"] = _odds_scale(
        prior.loc[root_mask, "prior_score"],
        root_odds_multiplier,
    )
    root_mean_after = float(prior.loc[root_mask, "prior_score"].mean())
    with output_path.open("w", encoding="utf-8") as handle:
        for row in prior.itertuples(index=False):
            handle.write(
                f"{row.regulator},{row.target},{float(row.prior_score):.8g},{row.node}\n"
            )
    return root_mean_before, root_mean_after


def _uniform_root_probability(config: dict, requested: float | None) -> float:
    probability = float(
        config["grn"].get("density", 0.15) if requested is None else requested
    )
    if not 0 < probability < 1:
        raise ValueError(
            "uniform root-prior probability must lie strictly between 0 and 1"
        )
    return probability


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RegArch benchmark variants.")
    parser.add_argument("replicate_dir", type=Path)
    parser.add_argument(
        "--variant",
        choices=["full", "no_fate_map", "no_atac", "wrong_fate_map"],
        default="full",
    )
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--prior-file", type=Path)
    parser.add_argument(
        "--fate-map-file",
        type=Path,
        help="Optional inference fate map; RNA, ATAC, and truth remain unchanged.",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--lineage-weight", type=float, default=0.5)
    parser.add_argument("--atac-weight", type=float, default=1.0)
    parser.add_argument("--likelihood-weight", type=float, default=1.0)
    parser.add_argument(
        "--lineage-model",
        choices=["ctmc_signed", "ctmc_presence", "alt"],
        default="ctmc_signed",
        help=(
            "Use the simulator-matched signed CTMC, its unsigned presence "
            "marginal, or the original symmetric prior."
        ),
    )
    parser.add_argument(
        "--root-prior",
        choices=["atac", "uniform"],
        default="atac",
        help=(
            "Use an edge-specific mean-leaf ATAC prior or an edge-agnostic "
            "root prevalence prior."
        ),
    )
    parser.add_argument(
        "--root-prior-probability",
        type=float,
        default=None,
        help=(
            "Uniform root-edge probability. If omitted with --root-prior "
            "uniform, use grn.density from the frozen simulation config."
        ),
    )
    parser.add_argument(
        "--root-prior-odds-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiply only the edge-specific Root-prior odds; 1 leaves the "
            "ATAC prior unchanged."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=["L-BFGS-B", "SLSQP", "COBYLA"],
        default="L-BFGS-B",
    )
    parser.add_argument("--optimizer-maxfun", type=int, default=60000)
    parser.add_argument("--gain-rate", type=float)
    parser.add_argument("--loss-rate", type=float)
    parser.add_argument("--sign-switch-rate", type=float)
    parser.add_argument("--positive-fraction", type=float)
    parser.add_argument("--edge-temperature", type=float)
    parser.add_argument("--sign-temperature", type=float)
    args = parser.parse_args()

    inputs = args.replicate_dir / "method_inputs" / "RegArch"
    run_name = args.run_name or args.variant
    work_dir = args.replicate_dir / "method_outputs" / "raw" / f"RegArch_{run_name}"
    work_dir.mkdir(parents=True, exist_ok=True)
    expression_path = inputs / "expression_data.csv"
    atac_path = inputs / "atac_data.csv"
    fate_path = inputs / "fate_map_edges.csv"
    if args.fate_map_file:
        fate_path = args.fate_map_file

    grn_module.OPTIMIZE_METHOD = args.optimizer
    grn_module.OPTIMIZE_MAX_FUNCTION_EVALUATIONS = args.optimizer_maxfun
    grn_module.LINEAGE_TRANSITION_MODEL = args.lineage_model
    grn_module.LINEAGE_ROOT_PRIOR_MODE = args.root_prior
    grn_module.LINEAGE_PRIOR_WEIGHT = args.lineage_weight
    grn_module.ATAC_PRIOR_WEIGHT = args.atac_weight
    grn_module.LIKELIHOOD_WEIGHT = args.likelihood_weight
    config = json.loads((args.replicate_dir / "config.json").read_text())
    grn_module.LINEAGE_GAIN_RATE = float(
        config["grn"]["gain_rate"] if args.gain_rate is None else args.gain_rate
    )
    grn_module.LINEAGE_LOSS_RATE = float(
        config["grn"]["loss_rate"] if args.loss_rate is None else args.loss_rate
    )
    grn_module.LINEAGE_SIGN_SWITCH_RATE = float(
        config["grn"].get("sign_switch_rate", 0.0)
        if args.sign_switch_rate is None
        else args.sign_switch_rate
    )
    grn_module.LINEAGE_POSITIVE_FRACTION = float(
        config["grn"].get("positive_fraction", 0.5)
        if args.positive_fraction is None
        else args.positive_fraction
    )
    if args.edge_temperature is not None:
        grn_module.LINEAGE_EDGE_TEMPERATURE = args.edge_temperature
    if args.sign_temperature is not None:
        grn_module.LINEAGE_SIGN_TEMPERATURE = args.sign_temperature
    for name, value in {
        "gain rate": grn_module.LINEAGE_GAIN_RATE,
        "loss rate": grn_module.LINEAGE_LOSS_RATE,
        "sign-switch rate": grn_module.LINEAGE_SIGN_SWITCH_RATE,
        "edge temperature": grn_module.LINEAGE_EDGE_TEMPERATURE,
        "sign temperature": grn_module.LINEAGE_SIGN_TEMPERATURE,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive; got {value}")
    if not 0 < grn_module.LINEAGE_POSITIVE_FRACTION < 1:
        raise ValueError("positive fraction must lie strictly between 0 and 1")
    if args.root_prior_odds_multiplier <= 0:
        raise ValueError("root-prior odds multiplier must be positive")
    if args.root_prior == "uniform" and args.root_prior_odds_multiplier != 1.0:
        raise ValueError(
            "root-prior odds scaling is defined for edge-specific ATAC priors, "
            "not uniform priors"
        )
    if args.root_prior == "uniform":
        grn_module.LINEAGE_ROOT_PRIOR_PROBABILITY = _uniform_root_probability(
            config,
            args.root_prior_probability,
        )
    elif args.root_prior_probability is not None:
        raise ValueError(
            "--root-prior-probability is only meaningful with --root-prior uniform"
        )
    if args.variant == "no_fate_map":
        grn_module.LINEAGE_PRIOR_WEIGHT = 0
    root_prior_mean_before = None
    root_prior_mean_after = None
    if args.prior_file or args.root_prior_odds_multiplier != 1.0:
        custom_prior = work_dir / "custom_atac_data.csv"
        source_prior = args.prior_file or inputs / "atac_prior_tf_target.tsv"
        root_prior_mean_before, root_prior_mean_after = _write_alt_prior(
            source_prior,
            custom_prior,
            root_odds_multiplier=args.root_prior_odds_multiplier,
        )
        atac_path = custom_prior
    if args.variant == "no_atac":
        grn_module.ATAC_PRIOR_WEIGHT = 0.0
        grn_module.LINEAGE_ROOT_PRIOR_MODE = "uniform"
        grn_module.LINEAGE_ROOT_PRIOR_PROBABILITY = _uniform_root_probability(
            config,
            args.root_prior_probability,
        )
        prior = pd.read_csv(inputs / "atac_prior_tf_target.tsv", sep="\t")
        prior["prior_score"] = 0.0
        zero_prior = work_dir / "atac_data.csv"
        prior.to_csv(work_dir / "atac_prior_tf_target.tsv", sep="\t", index=False)
        _write_alt_prior(work_dir / "atac_prior_tf_target.tsv", zero_prior)
        atac_path = zero_prior
    if args.variant == "wrong_fate_map":
        fate_path = (
            args.replicate_dir
            / "ablations"
            / "wrong_fate_map"
            / "wrong_fate_map_edges.csv"
        )

    run_config = {
        "variant": args.variant,
        "run_name": run_name,
        "optimizer": args.optimizer,
        "optimizer_maxfun": args.optimizer_maxfun,
        "lineage_model": args.lineage_model,
        "lineage_weight": grn_module.LINEAGE_PRIOR_WEIGHT,
        "atac_weight": grn_module.ATAC_PRIOR_WEIGHT,
        "likelihood_weight": grn_module.LIKELIHOOD_WEIGHT,
        "lineage_gain_rate": grn_module.LINEAGE_GAIN_RATE,
        "lineage_loss_rate": grn_module.LINEAGE_LOSS_RATE,
        "lineage_sign_switch_rate": grn_module.LINEAGE_SIGN_SWITCH_RATE,
        "lineage_positive_fraction": grn_module.LINEAGE_POSITIVE_FRACTION,
        "lineage_edge_temperature": grn_module.LINEAGE_EDGE_TEMPERATURE,
        "lineage_sign_temperature": grn_module.LINEAGE_SIGN_TEMPERATURE,
        "root_prior": grn_module.LINEAGE_ROOT_PRIOR_MODE,
        "root_prior_probability": grn_module.LINEAGE_ROOT_PRIOR_PROBABILITY,
        "root_prior_odds_multiplier": args.root_prior_odds_multiplier,
        "root_prior_mean_before": root_prior_mean_before,
        "root_prior_mean_after": root_prior_mean_after,
        "prior_file": str(args.prior_file) if args.prior_file else None,
        "fate_map_file": str(fate_path),
        "input_sha256": {
            "expression": _sha256(expression_path),
            "atac": _sha256(atac_path),
            "fate_map": _sha256(fate_path),
        },
    }
    (work_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    edge_dict = load_fate_map_topology(fate_path)
    fate_map = FateMap(parse_edge_dict(edge_dict))
    inference = GRNInference(
        str(atac_path),
        str(expression_path),
        fate_map,
        str(work_dir / "inferred_grns"),
    )
    inference.infer_grn(args.processes)


if __name__ == "__main__":
    main()
