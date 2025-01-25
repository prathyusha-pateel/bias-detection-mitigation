from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Union
from pathlib import Path
from output import Output
from config import DataProcessingError
import json
import pandas as pd
import numpy as np

load_dotenv()


class AIAnalyzerError(DataProcessingError):
    """Base exception class for AI analyzer errors."""

    pass


class AIAnalyzer:
    def __init__(self, output: Output):
        """Initialize the AIAnalyzer with OpenAI client."""
        self.output = output
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.max_tokens = 2000  # Reduced from 4000
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables."
            )

    def _truncate_text(
        self, text: str, max_chars: int = 1500
    ) -> str:  # Reduced from 3000
        """Truncate text to a maximum number of characters while keeping it meaningful."""
        if len(text) <= max_chars:
            return text

        # Try to truncate at a natural break point
        truncated = text[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > 0:
            truncated = truncated[: last_period + 1]

        return truncated + "\n[Content truncated...]"

    def _summarize_dict(
        self, data: Dict[str, Any], max_items: int = 3
    ) -> Dict[str, Any]:  # Reduced from 5
        """Summarize a dictionary by keeping only the most important items."""
        if len(data) <= max_items:
            return data

        # Prioritize important metrics
        priority_metrics = [
            "demographic_parity_difference",
            "equalized_odds_difference",
            "selection_rate",
            "false_positive_rate",
            "precision",
        ]

        summary = {}
        added_items = 0

        # First add priority items
        for key, value in data.items():
            if added_items >= max_items:
                break
            if any(metric in str(key).lower() for metric in priority_metrics):
                if isinstance(value, pd.Series):
                    summary[key] = value.head(3).to_dict()  # Only take first 3 items
                elif isinstance(value, np.ndarray):
                    summary[key] = value[:3].tolist()  # Only take first 3 items
                elif isinstance(value, dict):
                    summary[key] = {
                        k: v for k, v in list(value.items())[:3]
                    }  # Only take first 3 items
                else:
                    summary[key] = value
                added_items += 1

        # Fill remaining slots with other items if needed
        for key, value in data.items():
            if added_items >= max_items:
                break
            if key not in summary:
                if isinstance(value, (pd.Series, np.ndarray, dict)):
                    summary[key] = "[truncated]"
                else:
                    summary[key] = value
                added_items += 1

        if len(data) > max_items:
            summary["[truncated]"] = f"{len(data) - max_items} more items"

        return summary

    def _prepare_data_for_analysis(
        self, data: Union[str, Dict[str, Any], pd.Series]
    ) -> str:
        """Prepare data for analysis by truncating or summarizing as needed."""
        if isinstance(data, str):
            return self._truncate_text(data)

        if isinstance(data, pd.Series):
            return data.to_dict()

        if isinstance(data, dict):
            summarized_dict = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    summarized_dict[key] = self._summarize_dict(value)
                elif isinstance(value, pd.Series):
                    summarized_dict[key] = value.to_dict()
                elif isinstance(value, np.ndarray):
                    summarized_dict[key] = value.tolist()
                else:
                    summarized_dict[key] = value
            return json.dumps(summarized_dict, indent=2, default=str)

        return str(data)

    def analyze_data_generation(self, generation_summary: str) -> str:
        """Analyze data generation results using AI."""
        if not generation_summary:
            raise AIAnalyzerError("generation_summary cannot be empty")

        prepared_summary = self._prepare_data_for_analysis(generation_summary)
        prompt = f"""
        Analyze this summary of generated synthetic consumer data:

        {prepared_summary}

        Provide a concise analysis covering:
        1. Key data distribution insights
        2. Notable demographic patterns
        3. Potential marketing implications
        4. Top recommendations
        """

        return self._get_analysis(prompt)

    def analyze_bias_detection(self, detection_results: Dict[str, Any]) -> str:
        """Analyze bias detection results using AI."""
        if not detection_results:
            raise AIAnalyzerError("detection_results cannot be empty")

        prepared_results = self._prepare_data_for_analysis(detection_results)
        prompt = f"""
        Analyze these key bias detection findings:

        {prepared_results}

        Provide a focused analysis of:
        1. Most significant fairness metrics
        2. Key biases identified
        3. Critical implications
        4. Priority recommendations
        """
        return self._get_analysis(prompt)

    def analyze_bias_mitigation(self, mitigation_results: Dict[str, Any]) -> str:
        """Analyze bias mitigation results using AI."""
        if not mitigation_results:
            raise AIAnalyzerError("mitigation_results cannot be empty")

        prepared_results = self._prepare_data_for_analysis(mitigation_results)
        prompt = f"""
        Analyze these bias mitigation results:

        {prepared_results}

        Focus on:
        1. Effectiveness of mitigation
        2. Key trade-offs observed
        3. Remaining challenges
        4. Implementation recommendations
        """
        return self._get_analysis(prompt)

    def analyze_final_report(
        self, detection_results: Dict[str, Any], mitigation_results: Dict[str, Any]
    ) -> str:
        """Analyze overall pipeline results using AI."""
        if not detection_results or not mitigation_results:
            raise AIAnalyzerError("Both detection and mitigation results are required")

        # Only keep the most important results
        key_detection_metrics = {
            "demographic_parity_difference": detection_results.get(
                "supervised", {}
            ).get("demographic_parity_difference"),
            "equalized_odds_difference": detection_results.get("supervised", {}).get(
                "equalized_odds_difference"
            ),
            "overall_metrics": detection_results.get("supervised", {}).get(
                "overall_metrics", {}
            ),
        }

        key_mitigation_metrics = {}
        for method, results in mitigation_results.items():
            key_mitigation_metrics[method] = {
                "mitigated_demographic_parity_difference": results.get(
                    "mitigated_demographic_parity_difference"
                ),
                "mitigated_equalized_odds_difference": results.get(
                    "mitigated_equalized_odds_difference"
                ),
            }

        prepared_detection = self._prepare_data_for_analysis(key_detection_metrics)
        prepared_mitigation = self._prepare_data_for_analysis(key_mitigation_metrics)

        prompt = f"""
        Analyze these key metrics:

        Detection:
        {prepared_detection}

        Mitigation:
        {prepared_mitigation}

        Provide a very concise summary of:
        1. Key findings
        2. Main improvements
        3. Remaining issues
        4. Next steps
        """
        return self._get_analysis(prompt)

    def _get_analysis(self, prompt: str) -> str:
        """Get AI-generated analysis from the OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI expert providing concise analysis on bias detection and mitigation in marketing AI systems.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                n=1,
                stop=None,
                temperature=0.7,
            )

            if not response.choices:
                raise AIAnalyzerError("No response received from OpenAI API")

            return response.choices[0].message.content.strip()

        except Exception as e:
            error_message = f"Error in generating analysis: {str(e)}"
            self.output.write(error_message)
            raise AIAnalyzerError(error_message) from e

    def save_analysis(self, analysis: str, filename: str) -> None:
        """Save AI-generated analysis to a file."""
        if not analysis:
            raise AIAnalyzerError("Analysis content cannot be empty")

        try:
            self.output.write(analysis)
            self.output.save_to_markdown(filename)
        except Exception as e:
            raise AIAnalyzerError(f"Failed to save analysis: {str(e)}") from e


def test_analyzer() -> None:
    """Test the AIAnalyzer functionality."""
    try:
        output = Output(output_type="both", output_dir="test_output")
        analyzer = AIAnalyzer(output)

        # Test data
        sample_generation_summary = "Test generation summary"
        sample_detection_results = {"supervised": {"metrics": "test"}}
        sample_mitigation_results = {"method1": {"results": "test"}}

        # Run tests
        gen_analysis = analyzer.analyze_data_generation(sample_generation_summary)
        analyzer.save_analysis(gen_analysis, "test_generation_analysis.md")

        det_analysis = analyzer.analyze_bias_detection(sample_detection_results)
        analyzer.save_analysis(det_analysis, "test_detection_analysis.md")

        mit_analysis = analyzer.analyze_bias_mitigation(sample_mitigation_results)
        analyzer.save_analysis(mit_analysis, "test_mitigation_analysis.md")

        final_analysis = analyzer.analyze_final_report(
            sample_detection_results, sample_mitigation_results
        )
        analyzer.save_analysis(final_analysis, "test_final_analysis.md")

    except Exception as e:
        print(f"Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_analyzer()
