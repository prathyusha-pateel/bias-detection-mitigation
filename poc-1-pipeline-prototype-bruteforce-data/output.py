# File: output.py

import os
import pandas as pd
from typing import Union, List, Dict, Any
import json
import csv
import logging


class Output:
    def __init__(self, output_type: str = "both", output_dir: str = "output"):
        """
        Initialize the Output class.

        Args:
            output_type (str): Type of output. Can be "stdout", "markdown", "both", "csv", or "json".
            output_dir (str): Directory to save output files.
        """
        self.output_type = output_type
        self.output_dir = output_dir
        self.buffer: List[str] = []
        self.ensure_output_directory()
        self.setup_logging()

    def setup_logging(self) -> None:
        """Set up logging configuration."""
        try:
            logging.basicConfig(
                filename=os.path.join(self.output_dir, "output.log"),
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )
        except Exception as e:
            print(f"Error setting up logging: {e}")

    def ensure_output_directory(self) -> None:
        """Ensure the output directory exists."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logging.error(f"Error creating output directory: {e}")
            raise

    def write(self, message: str) -> None:
        """
        Write a message to the output.

        Args:
            message (str): The message to write.
        """
        if self.output_type in ["stdout", "both"]:
            print(message)
        self.buffer.append(message)
        logging.info(message)

    def save_to_markdown(self, filename: str = "report.md") -> None:
        """
        Save the buffered output to a markdown file.

        Args:
            filename (str): Name of the markdown file to save.
        """
        if self.output_type in ["markdown", "both"]:
            try:
                full_path = os.path.join(self.output_dir, filename)
                with open(full_path, "w") as f:
                    f.write("\n".join(self.buffer))
                logging.info(f"Report saved to {full_path}")
            except IOError as e:
                logging.error(f"Error saving markdown file: {e}")
                raise

    def save_to_csv(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        filename: str = "data.csv",
    ) -> None:
        """
        Save data to a CSV file.

        Args:
            data (Union[pd.DataFrame, List[Dict[str, Any]]]): Data to save.
            filename (str): Name of the CSV file to save.
        """
        if self.output_type in ["csv", "both"]:
            try:
                full_path = os.path.join(self.output_dir, filename)
                if isinstance(data, pd.DataFrame):
                    data.to_csv(full_path, index=False)
                else:
                    with open(full_path, "w", newline="") as csvfile:
                        if data:
                            fieldnames = data[0].keys()
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                            for row in data:
                                writer.writerow(row)
                logging.info(f"Data saved to {full_path}")
            except (IOError, pd.errors.EmptyDataError) as e:
                logging.error(f"Error saving CSV file: {e}")
                raise

    def save_to_json(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        filename: str = "data.json",
    ) -> None:
        """
        Save data to a JSON file.

        Args:
            data (Union[Dict[str, Any], List[Dict[str, Any]]]): Data to save.
            filename (str): Name of the JSON file to save.
        """
        if self.output_type in ["json", "both"]:
            try:
                full_path = os.path.join(self.output_dir, filename)
                with open(full_path, "w") as json_file:
                    json.dump(data, json_file, indent=2)
                logging.info(f"Data saved to {full_path}")
            except IOError as e:
                logging.error(f"Error saving JSON file: {e}")
                raise

    def clear_buffer(self) -> None:
        """Clear the output buffer."""
        self.buffer = []

    def get_buffer(self) -> str:
        """
        Get the contents of the output buffer.

        Returns:
            str: The contents of the output buffer.
        """
        return "\n".join(self.buffer)

    def set_output_type(self, output_type: str) -> None:
        """
        Set the output type.

        Args:
            output_type (str): The type of output to use.

        Raises:
            ValueError: If an invalid output type is provided.
        """
        if output_type in ["stdout", "markdown", "both", "csv", "json"]:
            self.output_type = output_type
        else:
            error_msg = "Invalid output type. Choose 'stdout', 'markdown', 'csv', 'json', or 'both'."
            logging.error(error_msg)
            raise ValueError(error_msg)

    def set_output_directory(self, output_dir: str) -> None:
        """
        Set the output directory.

        Args:
            output_dir (str): The directory to use for output files.
        """
        self.output_dir = output_dir
        self.ensure_output_directory()


if __name__ == "__main__":
    # Example usage
    try:
        output = Output(output_type="both", output_dir="test_output")

        output.write("This is a test message.")
        output.write("This is another test message.")

        output.save_to_markdown("test_report.md")

        test_data_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        output.save_to_csv(test_data_df, "test_data_df.csv")

        test_data_list = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]
        output.save_to_csv(test_data_list, "test_data_list.csv")

        test_data_json = {
            "employees": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
                {"name": "Charlie", "age": 35},
            ]
        }
        output.save_to_json(test_data_json, "test_data.json")

        print(output.get_buffer())

    except Exception as e:
        print(f"An error occurred: {str(e)}")
