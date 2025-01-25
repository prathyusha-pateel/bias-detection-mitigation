from generate import SyntheticDataGenerator
from detection import BiasDetectionPipeline
from mitigation import BiasMitigationPipeline
from ai_analyzer import AIAnalyzer
from output import Output
from typing import Dict, Any, Optional
import warnings
from sklearn.exceptions import ConvergenceWarning
import logging  # Suppress specific warnings

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    # Code that triggers FutureWarning

# Add these specific warning filters
warnings.filterwarnings("ignore", message=".*functorch into PyTorch.*")
warnings.filterwarnings("ignore", message=".*We've integrated functorch.*")

# Your existing warning filters
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="aif360")


class BiasPipeline:
    def __init__(
        self,
        n_samples: int = 200000,
        output_type: str = "both",
        output_dir: str = "bias_pipeline_output",
    ):
        """Initialize the BiasPipeline."""
        self.n_samples = n_samples
        self.output = Output(output_type=output_type, output_dir=output_dir)

        # Configure logging
        logging.basicConfig(
            filename=f"{output_dir}/pipeline.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        self.generator = SyntheticDataGenerator(self.output)
        self.detector: Optional[BiasDetectionPipeline] = None

        # Calculate Average_Basket_Size
        self.frequency_multiplier = {
            "Weekly": 52,
            "Bi-Weekly": 26,
            "Monthly": 12,
            "Quarterly": 4,
        }

        # Update feature lists to match available columns
        self.features = [
            "Age",
            "Weekly_Earnings",
            "Income_Level",
            "Education_Level",
            "Employment_Status",
            "Region",
        ]
        self.sensitive_features = ["Gender", "Race", "Ethnicity"]
        self.target_column = "Average_Basket_Size"

        self.mitigator: Optional[BiasMitigationPipeline] = None
        self.data = None
        self.ai_analyzer = AIAnalyzer(self.output)

        # Track pipeline state
        self.state = {
            "data_generated": False,
            "bias_detected": False,
            "bias_mitigated": False,
            "report_generated": False,
        }

    def generate_data(self) -> None:
        """Generate synthetic data and analyze it."""
        self.output.write("# 1. Data Generation Report\n")
        try:
            logging.info("Starting data generation")
            self.data = self.generator.generate_synthetic_consumer_data(self.n_samples)

            # Calculate Average_Basket_Size
            self.data["Average_Basket_Size"] = self.data.apply(
                lambda x: x["Weekly_Earnings"]
                * self.frequency_multiplier.get(x["Buying_Frequency"], 12)
                / 52,
                axis=1,
            )

            # Ensure the target column is binary
            median_basket_size = self.data[self.target_column].median()
            self.data[self.target_column] = (
                self.data[self.target_column] > median_basket_size
            ).astype(int)

            self.generator.save_data()
            generation_summary = self.generator.analyze_data()
            self.output.write(generation_summary)

            data_gen_analysis = self.ai_analyzer.analyze_data_generation(
                generation_summary
            )
            self.output.write("\n## AI Analysis of Data Generation\n")
            self.output.write(data_gen_analysis)

            self.output.save_to_markdown("1_data_generation_report.md")
            self.output.clear_buffer()

            self.state["data_generated"] = True
            logging.info("Data generation completed successfully")
        except Exception as e:
            logging.error(f"Error in data generation: {str(e)}")
            self.output.write(f"Error in data generation: {str(e)}")
            raise

    def detect_bias(self) -> None:
        """Detect bias in the generated data."""
        if not self.state["data_generated"]:
            raise ValueError("Must generate data before detecting bias")

        self.output.write("# 2. Bias Detection Report\n")
        try:
            logging.info("Starting bias detection")
            self.detector = BiasDetectionPipeline(
                data=self.data,
                target_column=self.target_column,
                features=self.features,
                sensitive_features=self.sensitive_features,
                output=self.output,
            )
            detection_results = self.detector.run_detection()

            self.output.write("\n## Supervised Model Results\n")
            self._write_detection_results(detection_results["supervised"])

            self.output.write("\n## Unsupervised Model Results\n")
            self._write_detection_results(detection_results["unsupervised"])

            detection_analysis = self.ai_analyzer.analyze_bias_detection(
                detection_results
            )
            self.output.write("\n## AI Analysis of Bias Detection\n")
            self.output.write(detection_analysis)

            self.output.save_to_markdown("2_bias_detection_report.md")
            self.output.clear_buffer()

            self.state["bias_detected"] = True
            logging.info("Bias detection completed successfully")
        except Exception as e:
            logging.error(f"Error in bias detection: {str(e)}")
            self.output.write(f"Error in bias detection: {str(e)}")
            raise

    def _write_detection_results(self, results: Dict[str, Any]) -> None:
        """
        Write detection results to the output.

        Args:
            results (Dict[str, Any]): Detection results to write.
        """
        for metric, value in results.get("overall_metrics", {}).items():
            self.output.write(f"{metric}: {value:.4f}\n")

        self.output.write("\n### Metrics by Group\n")
        for metric, groups in results.get("metrics_by_group", {}).items():
            self.output.write(f"{metric}:\n")
            for group, value in groups.items():
                self.output.write(f"  {group}: {value:.4f}\n")

        if "demographic_parity_difference" in results:
            self.output.write(
                f"\nDemographic Parity Difference: {results['demographic_parity_difference']:.4f}\n"
            )
        if "equalized_odds_difference" in results:
            self.output.write(
                f"Equalized Odds Difference: {results['equalized_odds_difference']:.4f}\n"
            )

    def mitigate_bias(self) -> None:
        """
        Mitigate bias in the data.
        """
        self.output.write("# 3. Bias Mitigation Report\n")
        try:
            self.mitigator = BiasMitigationPipeline(
                data=self.data,
                target_column=self.target_column,
                features=self.features,
                sensitive_features=self.sensitive_features,
                output=self.output,
            )

            mitigation_results = self.mitigator.run_mitigation()

            for method, results in mitigation_results.items():
                self.output.write(f"\n## Bias Mitigation Results ({method}):\n")
                self._write_mitigation_results(results)

            self.output.save_to_markdown("3_bias_mitigation_report.md")

            mitigation_analysis = self.ai_analyzer.analyze_bias_mitigation(
                mitigation_results
            )
            self.output.write("\n## AI Analysis of Bias Mitigation\n")
            self.output.write(mitigation_analysis)

            self.output.clear_buffer()
        except Exception as e:
            self.output.write(f"Error in bias mitigation: {str(e)}")

    def _write_mitigation_results(self, results: Dict[str, Any]) -> None:
        """
        Write mitigation results to the output.

        Args:
            results (Dict[str, Any]): Mitigation results to write.
        """
        self.output.write("Mitigated Metrics by Group:\n")
        for metric, values in results.get("mitigated_metrics_by_group", {}).items():
            self.output.write(f"\n{metric}:\n")
            for group, value in values.items():
                self.output.write(f"  {group}: {value:.4f}\n")

        self.output.write("\nOverall Mitigated Metrics:\n")
        for metric, value in results.get("overall_mitigated_metrics", {}).items():
            self.output.write(f"{metric}: {value:.4f}\n")

        if "mitigated_demographic_parity_difference" in results:
            self.output.write(
                f"\nMitigated Demographic Parity Difference: {results['mitigated_demographic_parity_difference']:.4f}"
            )
        if "mitigated_equalized_odds_difference" in results:
            self.output.write(
                f"\nMitigated Equalized Odds Difference: {results['mitigated_equalized_odds_difference']:.4f}"
            )

    def generate_final_report(self) -> None:
        """
        Generate the final report summarizing all steps of the pipeline.
        """
        self.output.write("# 4. Final Pipeline Report\n")
        try:
            # Fetch bias detection and mitigation results
            detection_results = self.detector.run_detection()
            mitigation_results = self.mitigator.run_mitigation()

            self.output.write("## Bias Detection and Mitigation Summary\n")

            for model_type in ["supervised", "unsupervised"]:
                self.output.write(f"\n### {model_type.capitalize()} Model\n")

                if model_type == "supervised":
                    original_dpd = detection_results[model_type].get(
                        "demographic_parity_difference"
                    )

                    # Use the first mitigation method's results for comparison
                    first_method = next(iter(mitigation_results))
                    mitigated_dpd = mitigation_results[first_method].get(
                        "mitigated_demographic_parity_difference"
                    )

                    if original_dpd is not None:
                        self.output.write(
                            f"Original Demographic Parity Difference: {original_dpd:.4f}\n"
                        )

                        if mitigated_dpd is not None:
                            self.output.write(
                                f"Mitigated Demographic Parity Difference: {mitigated_dpd:.4f}\n"
                            )
                            absolute_reduction = abs(original_dpd - mitigated_dpd)
                            relative_reduction = (
                                (1 - mitigated_dpd / original_dpd) * 100
                                if original_dpd != 0
                                else 0
                            )
                            self.output.write(
                                f"Absolute Reduction in Bias: {absolute_reduction:.4f}\n"
                            )
                            self.output.write(
                                f"Relative Reduction in Bias: {relative_reduction:.2f}%\n"
                            )
                        else:
                            self.output.write(
                                "Mitigated Demographic Parity Difference: Not available\n"
                            )
                    else:
                        self.output.write(
                            "Demographic Parity Difference: Not available\n"
                        )
                else:
                    self.output.write(
                        "Demographic Parity Difference: Not applicable for unsupervised model\n"
                    )
                    self.output.write("Mitigation not applied to unsupervised model.\n")

            # Save the final summary to markdown
            self.output.save_to_markdown("4_final_pipeline_report.md")

            # AI Analysis of the final report
            final_analysis = self.ai_analyzer.analyze_final_report(
                detection_results, mitigation_results
            )
            self.output.write("\n## AI Analysis of Final Results\n")
            self.output.write(final_analysis)

            # Ensure buffer is cleared after the report is saved
            self.output.clear_buffer()

        except Exception as e:
            self.output.write(f"Error in generating final report: {str(e)}")

    def run_pipeline(self) -> None:
        """Run the entire bias detection and mitigation pipeline."""
        logging.info("Starting pipeline execution")
        try:
            self.generate_data()
            self.detect_bias()
            self.mitigate_bias()
            self.generate_final_report()
            logging.info("Pipeline execution completed successfully")
        except Exception as e:
            logging.error(f"Pipeline execution failed: {str(e)}")
            self.output.write(f"Error in pipeline execution: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        pipeline = BiasPipeline(
            n_samples=10000, output_type="both", output_dir="bias_pipeline_output"
        )
        pipeline.run_pipeline()
    except Exception as e:
        print(f"An error occurred in the main pipeline: {str(e)}")
