"""
Fairness Analysis for Marketing Data
Uses AI Fairness 360 (aif360) to detect and analyze potential biases in the data
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in scalar divide",
)

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from data_loader import SyntheticDataLoader
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from constants import REGIONAL_VARIATIONS, STATE_TO_REGION


class MarketingFairness:
    def __init__(self, test_mode: bool = False):
        self.loader = SyntheticDataLoader(test_mode=test_mode)
        self.output_dir = Path("fairness_outputs")
        self.output_dir.mkdir(exist_ok=True)

        # Define what we want to analyze - will be populated based on available data
        self.analyses = {}

        # Add logging configuration
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def _configure_analyses(self, datasets):
        """Configure analyses based on available data"""
        if "transactions" in datasets:
            df = datasets["transactions"]
            regions = df["region"].unique() if "region" in df.columns else []
            channels = df["channel"].unique() if "channel" in df.columns else []

            # Only add region analysis if we have multiple regions
            region_analysis = {}
            if len(regions) > 1:
                # Split regions into privileged/unprivileged based on REGIONAL_VARIATIONS
                region_mult = {
                    r: REGIONAL_VARIATIONS[STATE_TO_REGION.get(r, r)][
                        "baseline_multiplier"
                    ]
                    for r in regions
                    if STATE_TO_REGION.get(r, r) in REGIONAL_VARIATIONS
                }
                if region_mult:
                    # Sort regions by multiplier and split into privileged/unprivileged
                    sorted_regions = sorted(
                        region_mult.items(), key=lambda x: x[1], reverse=True
                    )
                    mid = len(sorted_regions) // 2
                    region_analysis = {
                        "privileged": [r[0] for r in sorted_regions[:mid]],
                        "unprivileged": [r[0] for r in sorted_regions[mid:]],
                    }

            self.analyses["transactions"] = {
                "protected_attributes": {
                    "age_group": {
                        "privileged": ["35-54"],
                        "unprivileged": ["18-34", "55+"],
                    },
                    **({"region": region_analysis} if region_analysis else {}),
                    **(
                        {
                            "channel": {
                                "privileged": ["desktop"],
                                "unprivileged": [c for c in channels if c != "desktop"],
                            }
                        }
                        if "desktop" in channels
                        else {}
                    ),
                },
                "outcomes": {
                    "transaction_value": lambda x: x > x.median(),
                    "num_items": lambda x: x > x.median(),
                    "buying_frequency": lambda x: x == "high",
                },
            }

        if "loyalties" in datasets:
            df = datasets["loyalties"]
            tiers = df["tier"].unique() if "tier" in df.columns else []

            # Only include tier analysis if we have multiple tiers
            tier_analysis = {}
            if len(tiers) > 1:
                # Sort tiers alphabetically and use higher tiers as privileged
                sorted_tiers = sorted(tiers)
                mid = len(sorted_tiers) // 2
                tier_analysis = {
                    "privileged": sorted_tiers[mid:],
                    "unprivileged": sorted_tiers[:mid],
                }

            self.analyses["loyalties"] = {
                "protected_attributes": {
                    "age_group": {
                        "privileged": ["35-54"],
                        "unprivileged": ["18-34", "55+"],
                    },
                    **({"tier": tier_analysis} if tier_analysis else {}),
                },
                "outcomes": {
                    "points_balance": lambda x: x > x.median(),
                    "redemption_rate": lambda x: x > x.median(),
                    "lifetime_points": lambda x: x > x.median(),
                },
            }

        if "engagements" in datasets:
            df = datasets["engagements"]
            campaign_types = (
                df["campaign_type"].unique() if "campaign_type" in df.columns else []
            )

            # Only include campaign_type analysis if we have the right types
            digital_social = [
                ct for ct in campaign_types if ct in ["digital", "social"]
            ]
            traditional = [
                ct for ct in campaign_types if ct in ["print", "traditional"]
            ]

            if digital_social and traditional:
                self.analyses["engagements"] = {
                    "protected_attributes": {
                        "campaign_type": {
                            "privileged": digital_social,
                            "unprivileged": traditional,
                        }
                    },
                    "outcomes": {
                        "engagement_rate": lambda x: x > x.median(),
                        "conversion_rate": lambda x: x > x.median(),
                        "clicks": lambda x: x > x.median(),
                    },
                }

    def analyze_fairness(self):
        """Generate fairness analysis for each dataset"""
        datasets = self.loader.load_all()

        # Configure analyses based on available data
        self._configure_analyses(datasets)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"fairness_analysis_{timestamp}"
        report_dir.mkdir(exist_ok=True)

        fairness_results = {}

        # Analyze each dataset we care about
        for dataset_name, analysis_config in self.analyses.items():
            if dataset_name not in datasets:
                self.logger.warning(
                    f"Dataset '{dataset_name}' not found in loaded data"
                )
                continue

            df = datasets[dataset_name]
            dataset_results = {}

            # For each demographic we want to check
            for attr_name, attr_config in analysis_config[
                "protected_attributes"
            ].items():
                if attr_name not in df.columns:
                    self.logger.warning(
                        f"Protected attribute '{attr_name}' not found in {dataset_name} dataset"
                    )
                    continue

                # Validate that the configured groups exist in the data
                existing_groups = set(df[attr_name].unique())
                existing_privileged = set(attr_config["privileged"]) & existing_groups
                existing_unprivileged = (
                    set(attr_config["unprivileged"]) & existing_groups
                )
                missing_privileged = set(attr_config["privileged"]) - existing_groups
                missing_unprivileged = (
                    set(attr_config["unprivileged"]) - existing_groups
                )

                if missing_privileged or missing_unprivileged:
                    self.logger.warning(
                        f"Missing groups in {dataset_name}.{attr_name}:\n"
                        f"  Missing privileged groups: {missing_privileged or 'None'}\n"
                        f"  Missing unprivileged groups: {missing_unprivileged or 'None'}\n"
                        f"  Available groups: {existing_groups}"
                    )

                if not existing_privileged or not existing_unprivileged:
                    self.logger.warning(
                        f"Skipping {attr_name} analysis - insufficient groups for comparison"
                    )
                    continue

                # Update config with only existing groups
                attr_config["privileged"] = list(existing_privileged)
                attr_config["unprivileged"] = list(existing_unprivileged)

                attr_results = {}
                # For each outcome we want to measure
                for outcome_name, outcome_func in analysis_config["outcomes"].items():
                    if outcome_name not in df.columns:
                        continue

                    try:
                        # Create binary labels
                        favorable_outcome = outcome_func(df[outcome_name])
                        protected_binary = (
                            df[attr_name].isin(attr_config["privileged"]).astype(int)
                        )

                        # Create dataset for fairness metrics
                        dataset = BinaryLabelDataset(
                            df=pd.DataFrame(
                                {
                                    "label": favorable_outcome.astype(int),
                                    attr_name: protected_binary,
                                }
                            ),
                            label_names=["label"],
                            protected_attribute_names=[attr_name],
                            privileged_protected_attributes=[[1]],
                            unprivileged_protected_attributes=[[0]],
                        )

                        # Calculate metrics
                        metrics = BinaryLabelDatasetMetric(
                            dataset,
                            unprivileged_groups=[{attr_name: 0}],
                            privileged_groups=[{attr_name: 1}],
                        )

                        attr_results[outcome_name] = {
                            "disparate_impact": metrics.disparate_impact(),
                            "statistical_parity_difference": metrics.statistical_parity_difference(),
                        }

                    except Exception as e:
                        logging.error(
                            f"Error analyzing {outcome_name} for {attr_name}: {str(e)}"
                        )

                if attr_results:
                    dataset_results[attr_name] = attr_results

            if dataset_results:
                fairness_results[dataset_name] = dataset_results

        # Save results
        self._save_results(fairness_results, report_dir, datasets)
        return report_dir

    def _save_results(self, results: dict, report_dir: Path, datasets: dict):
        """Save fairness analysis results"""
        for group_name, group_results in results.items():
            df = datasets[group_name]
            output_path = report_dir / f"{group_name}_fairness.md"

            with open(output_path, "w") as f:
                f.write(f"# Fairness Analysis: {group_name}\n\n")
                f.write(
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )
                f.write(f"Total records analyzed: {len(df):,}\n\n")

                for attr, attr_results in group_results.items():
                    f.write(f"## Protected Attribute: {attr}\n\n")

                    # Add group distribution
                    f.write("### Group Distribution\n")
                    f.write("| Group | Count | Percentage |\n")
                    f.write("|-------|--------|------------|\n")
                    total = len(df)
                    for group in df[attr].unique():
                        count = len(df[df[attr] == group])
                        pct = (count / total) * 100
                        f.write(f"| {group} | {count:,} | {pct:.1f}% |\n")
                    f.write("\n")

                    for outcome, metrics in attr_results.items():
                        f.write(f"### Outcome: {outcome}\n\n")
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        for metric, value in metrics.items():
                            f.write(
                                f"| {metric.replace('_', ' ').title()} | {value:.3f} |\n"
                            )
                        f.write("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate marketing data fairness analysis"
    )
    parser.add_argument("--test", action="store_true", help="Use test data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    fairness = MarketingFairness(test_mode=args.test)
    report_dir = fairness.analyze_fairness()
    print(f"\nFairness analysis has been generated in: {report_dir}")
