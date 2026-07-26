from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.t1_sector_relative_volatility.block_bootstrap import (
        block_bootstrap,
    )
    from experiments.t1_sector_relative_volatility.build_t1_target import (
        build_t1_target,
        manual_validation_sample,
    )
    from experiments.t1_sector_relative_volatility.config import ExperimentConfig
    from experiments.t1_sector_relative_volatility.cross_sectional_analysis import (
        correlation_and_effective_sample,
        ranking_portfolio,
        ranking_portfolio_summary,
        target_diagnostics,
    )
    from experiments.t1_sector_relative_volatility.evaluate_daily import (
        evaluate_overlapping,
    )
    from experiments.t1_sector_relative_volatility.evaluate_non_overlapping import (
        evaluate_non_overlapping,
    )
    from experiments.t1_sector_relative_volatility.inspect_project import (
        inspect_artifacts,
        write_structure_report,
    )
    from experiments.t1_sector_relative_volatility.plot_results import (
        generate_figures,
    )
    from experiments.t1_sector_relative_volatility.prepare_dataset import (
        prepare_dataset,
    )
    from experiments.t1_sector_relative_volatility.report import (
        generate_final_report,
        ticker_comparison_table,
    )
    from experiments.t1_sector_relative_volatility.split_purged import (
        assign_non_overlapping_offsets,
        build_purged_splits,
    )
    from experiments.t1_sector_relative_volatility.statistical_tests import (
        hac_dm_results,
        paired_loss_differences,
    )
    from experiments.t1_sector_relative_volatility.train_models import train_models
    from experiments.t1_sector_relative_volatility.utils import (
        configure_logging,
        set_seed,
        stage_message,
        write_json,
        write_yaml,
    )
    from experiments.t1_sector_relative_volatility.validate_target import (
        validate_t1_target,
    )
else:
    from .block_bootstrap import block_bootstrap
    from .build_t1_target import build_t1_target, manual_validation_sample
    from .config import ExperimentConfig
    from .cross_sectional_analysis import (
        correlation_and_effective_sample,
        ranking_portfolio,
        ranking_portfolio_summary,
        target_diagnostics,
    )
    from .evaluate_daily import evaluate_overlapping
    from .evaluate_non_overlapping import evaluate_non_overlapping
    from .inspect_project import inspect_artifacts, write_structure_report
    from .plot_results import generate_figures
    from .prepare_dataset import prepare_dataset
    from .report import generate_final_report, ticker_comparison_table
    from .split_purged import assign_non_overlapping_offsets, build_purged_splits
    from .statistical_tests import (
        hac_dm_results,
        paired_loss_differences,
    )
    from .train_models import train_models
    from .utils import (
        configure_logging,
        set_seed,
        stage_message,
        write_json,
        write_yaml,
    )
    from .validate_target import validate_t1_target


TOTAL_STAGES = 8


def _runtime_information() -> dict[str, Any]:
    information: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "gpu_name": None,
        "cuda_version": None,
        "pytorch_version": None,
    }
    try:
        import torch

        information["pytorch_version"] = torch.__version__
        information["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            information["device"] = "cuda"
            information["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return information


def run(config: ExperimentConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)
    logger = configure_logging(output)
    set_seed(config.seeds[0])
    write_yaml(output / "config_used.yaml", config.as_serializable_dict())
    runtime = _runtime_information()
    write_json(output / "runtime.json", runtime)

    stage_message(1, TOTAL_STAGES, "Inspect project")
    data_audit, market = inspect_artifacts(config)
    mapping_path = write_structure_report(config, data_audit)
    logger.info("Project mapping written to %s", mapping_path)

    stage_message(2, TOTAL_STAGES, "Build T1 target")
    target, target_audit = build_t1_target(market, config)
    target_directory = output / "data"
    target_directory.mkdir(exist_ok=True)
    target.to_parquet(target_directory / "t1_target.parquet", index=False)
    manual_validation_sample(target, 30).to_csv(
        output / "target_validation_sample.csv", index=False
    )
    data_audit["target_construction"] = target_audit
    write_json(output / "data_audit.json", data_audit)
    logger.info(
        "T1 target: %d rows, %d dates",
        len(target),
        target["date"].nunique(),
    )

    stage_message(3, TOTAL_STAGES, "Validate leakage")
    validation_checks, validation_summary = validate_t1_target(target, config)
    validation_checks.to_csv(output / "target_validation.csv", index=False)
    write_json(output / "target_validation_summary.json", validation_summary)

    stage_message(4, TOTAL_STAGES, "Prepare purged splits")
    prepared, feature_groups, preparation_audit = prepare_dataset(target, config)
    purged, split_summary = build_purged_splits(prepared, config)
    purged = assign_non_overlapping_offsets(purged, stride=config.eval_stride)
    purged.to_parquet(target_directory / "prepared_t1_dataset.parquet", index=False)
    write_json(output / "split_summary.json", split_summary)
    write_json(output / "feature_audit.json", preparation_audit)
    for row in split_summary["splits"]:
        logger.info(
            "%s: %s -> %s | %d dates | %d rows | outcomes %s -> %s",
            row["split"],
            row["start_date"],
            row["end_date"],
            row["dates"],
            row["rows"],
            row["outcome_start"],
            row["outcome_end"],
        )

    stage_message(5, TOTAL_STAGES, "Train models")
    training = train_models(purged, feature_groups, config)
    training.selection.to_csv(output / "model_selection_validation.csv", index=False)
    predictions = training.predictions.sort_values(
        ["split", "model_name", "seed", "date", "ticker"]
    ).reset_index(drop=True)
    predictions[predictions["split"].eq("validation")].to_csv(
        output / "predictions" / "validation_predictions.csv", index=False
    )
    predictions[predictions["split"].eq("test")].to_csv(
        output / "predictions" / "test_predictions.csv", index=False
    )

    stage_message(6, TOTAL_STAGES, "Non-overlapping evaluation")
    overlapping, ticker_metrics_long, daily_ic = evaluate_overlapping(
        predictions, config
    )
    offsets, non_overlapping = evaluate_non_overlapping(predictions, config)
    ticker_comparison = ticker_comparison_table(predictions)
    overlapping.to_csv(output / "model_metrics_overlapping.csv", index=False)
    non_overlapping.to_csv(
        output / "model_metrics_non_overlapping.csv", index=False
    )
    offsets.to_csv(output / "metrics_by_offset.csv", index=False)
    ticker_comparison.to_csv(output / "metrics_by_ticker.csv", index=False)
    ticker_metrics_long.to_csv(output / "metrics_by_ticker_long.csv", index=False)
    daily_ic.to_csv(output / "daily_cross_sectional_ic.csv", index=False)

    stage_message(7, TOTAL_STAGES, "Statistical tests")
    paired_rows, paired_daily = paired_loss_differences(predictions, config)
    hac = hac_dm_results(paired_daily, config)
    bootstrap = block_bootstrap(paired_rows, config)
    before, after, effective = correlation_and_effective_sample(target, config)
    target_distribution, acf_table = target_diagnostics(target, paired_daily)
    ranking = ranking_portfolio(predictions, config)
    ranking_summary = ranking_portfolio_summary(ranking, config)

    paired_rows.to_csv(output / "paired_loss_difference.csv", index=False)
    paired_daily.to_csv(output / "paired_daily_loss_difference.csv", index=False)
    hac.to_csv(output / "hac_dm_results.csv", index=False)
    bootstrap.to_csv(output / "bootstrap_results.csv", index=False)
    effective.to_csv(output / "effective_sample_size.csv", index=False)
    before.to_csv(output / "correlation_before_demean.csv")
    after.to_csv(output / "correlation_after_demean.csv")
    target_distribution.to_csv(output / "target_diagnostics.csv", index=False)
    acf_table.to_csv(output / "acf_diagnostics.csv", index=False)
    ranking.to_csv(output / "ranking_portfolio_results.csv", index=False)
    ranking_summary.to_csv(
        output / "ranking_portfolio_bootstrap.csv", index=False
    )

    stage_message(8, TOTAL_STAGES, "Generate report")
    figure_paths = generate_figures(
        target=target,
        predictions=predictions,
        overlapping=overlapping,
        offsets=offsets,
        daily_ic=daily_ic,
        paired_daily=paired_daily,
        bootstrap=bootstrap,
        acf_table=acf_table,
        correlation_before=before,
        correlation_after=after,
        output_directory=output / "figures",
    )
    decision, reasons = generate_final_report(
        config=config,
        data_audit=data_audit,
        target_audit=target_audit,
        split_summary=split_summary,
        target_distribution=target_distribution,
        effective_sample=effective,
        overlapping=overlapping,
        non_overlapping=non_overlapping,
        offsets=offsets,
        bootstrap=bootstrap,
        hac=hac,
        ticker_table=ticker_comparison,
        output_path=output / "final_report.md",
    )
    elapsed = time.perf_counter() - started
    manifest = {
        "status": "completed",
        "mode": "debug" if config.debug else "full",
        "decision": decision,
        "decision_reasons": reasons,
        "elapsed_seconds": elapsed,
        "output_directory": str(output.resolve()),
        "figure_count": len(figure_paths),
        "runtime": runtime,
        "source_directories_modified": False,
    }
    write_json(output / "run_manifest.json", manifest)
    logger.info(
        "Completed T1 experiment in %.1f seconds | decision=%s | output=%s",
        elapsed,
        decision,
        output.resolve(),
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Five-day sector-relative volatility semantic-news experiment"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.py",
        help="Documented config path; Python dataclass defaults are used.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--embargo-days", type=int, default=0)
    parser.add_argument(
        "--semantic-representation", choices=["R2", "R3"], default="R3"
    )
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if not args.config.exists():
        raise FileNotFoundError(f"Config path does not exist: {args.config}")
    kwargs: dict[str, Any] = {
        "debug": args.debug,
        "embargo_days": args.embargo_days,
        "semantic_representation": args.semantic_representation,
    }
    if args.bootstrap is not None:
        kwargs["n_bootstrap"] = args.bootstrap
        kwargs["debug_bootstrap"] = args.bootstrap
    if args.output_directory is not None:
        kwargs["output_directory"] = args.output_directory
    if args.seeds:
        kwargs["seeds"] = args.seeds
    return ExperimentConfig(**kwargs)


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    result = run(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
