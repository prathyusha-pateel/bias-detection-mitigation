from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from faker import Faker
import random
from tqdm import tqdm
from output import Output
from config import (
    EmploymentStatus,
    IncomeLevel,
    EducationLevel,
    Race,
    Ethnicity,
    Gender,
    BuyingFrequency,
    AGE_DISTRIBUTION,
    GENDER_DISTRIBUTION,
    RACE_DISTRIBUTION,
    HISPANIC_DISTRIBUTION,
    REGIONS,
    REGION_WEIGHTS,
    EDUCATION_DISTRIBUTION,
    EMPLOYMENT_STATUS_DISTRIBUTION,
    EMPLOYMENT_RATE_BY_RACE,
    UNEMPLOYMENT_RATE_BY_RACE,
    EDUCATION_BY_RACE,
    INCOME_DISTRIBUTION,
    INCOME_RANGES,
    MEDIAN_WEEKLY_EARNINGS,
    LABOR_FORCE_PARTICIPATION,
    EDUCATION_EMPLOYMENT_ADJUSTMENT,
    AGE_EMPLOYMENT_ADJUSTMENT,
    DEMOGRAPHIC_EARNINGS_ADJUSTMENTS,
    PRODUCT_CATEGORIES,
    COMMUNICATION_CHANNELS,
    ENGAGEMENT_PREFERENCES,
    VALIDATION_RANGES,
    DataProcessingError,
)


class GenerationError(DataProcessingError):
    """Base exception class for data generation errors."""

    pass


class SyntheticDataGenerator:
    def __init__(self, output: Output):
        """Initialize the SyntheticDataGenerator."""
        self.fake = Faker()
        self.output = output
        self.synthetic_consumer_data: Optional[pd.DataFrame] = None
        self._validate_distributions()

    def _validate_distributions(self) -> None:
        """Validate probability distributions sum to 1."""
        distributions = {
            "AGE_DISTRIBUTION": AGE_DISTRIBUTION,
            "GENDER_DISTRIBUTION": GENDER_DISTRIBUTION,
            "RACE_DISTRIBUTION": RACE_DISTRIBUTION,
            "HISPANIC_DISTRIBUTION": HISPANIC_DISTRIBUTION,
            "INCOME_DISTRIBUTION": INCOME_DISTRIBUTION,
            "REGION_WEIGHTS": REGION_WEIGHTS,
        }

        for name, dist in distributions.items():
            if not np.isclose(sum(dist.values()), 1.0, rtol=1e-5):
                raise GenerationError(f"{name} probabilities do not sum to 1")

    def generate_age(self) -> int:
        """Generate age based on distribution."""
        try:
            age_range = np.random.choice(
                list(AGE_DISTRIBUTION.keys()), p=list(AGE_DISTRIBUTION.values())
            )

            # Handle special case for "65+" range
            if age_range == "65+":
                return random.randint(65, 90)  # Set reasonable upper limit

            # Handle normal ranges like "20-24"
            start, end = map(int, age_range.split("-"))
            return random.randint(start, end)

        except Exception as e:
            raise GenerationError(f"Age generation failed: {str(e)}")

    def get_age_group(self, age: int) -> str:
        """Get the age group for a given age."""
        if not isinstance(age, (int, np.integer)):
            raise GenerationError("Age must be an integer")

        if age <= 19:
            return "16-19"
        elif age <= 24:
            return "20-24"
        elif age <= 34:
            return "25-34"
        elif age <= 44:
            return "35-44"
        elif age <= 54:
            return "45-54"
        elif age <= 64:
            return "55-64"
        else:
            return "65+"

    def generate_demographics(self) -> Dict[str, Any]:
        """Generate basic demographic characteristics."""
        try:
            race = np.random.choice(
                list(RACE_DISTRIBUTION.keys()), p=list(RACE_DISTRIBUTION.values())
            )
            ethnicity = np.random.choice(
                list(HISPANIC_DISTRIBUTION.keys()),
                p=list(HISPANIC_DISTRIBUTION.values()),
            )
            gender = np.random.choice(
                list(GENDER_DISTRIBUTION.keys()), p=list(GENDER_DISTRIBUTION.values())
            )
            age = self.generate_age()

            return {"race": race, "ethnicity": ethnicity, "gender": gender, "age": age}
        except Exception as e:
            raise GenerationError(f"Demographics generation failed: {str(e)}")

    def generate_education_level(
        self, age: int, race: str, ethnicity: str
    ) -> EducationLevel:
        """Generate education level based on demographics."""
        try:
            if age < 25:
                probs = {
                    EducationLevel.LESS_THAN_HIGH_SCHOOL: 0.15,
                    EducationLevel.HIGH_SCHOOL: 0.40,
                    EducationLevel.SOME_COLLEGE: 0.35,
                    EducationLevel.BACHELOR: 0.09,
                    EducationLevel.ADVANCED: 0.01,
                }
            else:
                dist = (
                    EDUCATION_BY_RACE.get(race)
                    or EDUCATION_BY_RACE.get(
                        "Hispanic or Latino"
                        if ethnicity == "Hispanic or Latino"
                        else None
                    )
                    or EDUCATION_BY_RACE["White"]
                )

                probs = {
                    EducationLevel.LESS_THAN_HIGH_SCHOOL: dist["Less than high school"],
                    EducationLevel.HIGH_SCHOOL: dist["High School"],
                    EducationLevel.SOME_COLLEGE: dist["Some College/Associate"],
                    EducationLevel.BACHELOR: dist["Bachelor"],
                    EducationLevel.ADVANCED: dist["Advanced"],
                }

            total = sum(probs.values())
            normalized_probs = [p / total for p in probs.values()]

            return np.random.choice(list(probs.keys()), p=normalized_probs)
        except Exception as e:
            raise GenerationError(f"Education level generation failed: {str(e)}")

    def generate_employment_status(
        self,
        age: int,
        education: EducationLevel,
        race: str,
        gender: str,
        labor_force_rate: float,
    ) -> EmploymentStatus:
        """Generate employment status based on demographics."""
        try:
            if random.random() > labor_force_rate:
                return EmploymentStatus.NOT_IN_LABOR_FORCE

            base_unemployment = UNEMPLOYMENT_RATE_BY_RACE.get(
                race, UNEMPLOYMENT_RATE_BY_RACE["White"]
            )
            edu_adj = EDUCATION_EMPLOYMENT_ADJUSTMENT[education.value]
            age_adj = AGE_EMPLOYMENT_ADJUSTMENT[self.get_age_group(age)]

            unemployment_prob = base_unemployment / (edu_adj * age_adj)

            return (
                EmploymentStatus.UNEMPLOYED
                if random.random() < unemployment_prob
                else EmploymentStatus.EMPLOYED
            )
        except Exception as e:
            raise GenerationError(f"Employment status generation failed: {str(e)}")

    def generate_income_and_earnings(
        self,
        age: int,
        education: EducationLevel,
        employment: EmploymentStatus,
        race: str,
        gender: str,
        ethnicity: str,
    ) -> Tuple[str, float]:
        """Generate income level and earnings based on demographics."""
        try:
            if employment != EmploymentStatus.EMPLOYED:
                return "Low", 0.0

            base_earnings = MEDIAN_WEEKLY_EARNINGS.get(race, {}).get(
                gender, MEDIAN_WEEKLY_EARNINGS["White"]["Male"]
            )

            edu_adjustment = DEMOGRAPHIC_EARNINGS_ADJUSTMENTS["Education"][
                education.value
            ].get(
                race,
                DEMOGRAPHIC_EARNINGS_ADJUSTMENTS["Education"][education.value]["White"],
            )

            age_adj = AGE_EMPLOYMENT_ADJUSTMENT[self.get_age_group(age)]
            ethnicity_adj = 0.95 if ethnicity == "Hispanic or Latino" else 1.0

            adjusted_earnings = base_earnings * edu_adjustment * age_adj * ethnicity_adj
            variation_range = (
                (0.85, 1.15) if education.value == "Advanced" else (0.90, 1.10)
            )
            adjusted_earnings *= random.uniform(*variation_range)

            if adjusted_earnings < INCOME_RANGES["Low"][1]:
                income_level = "Low"
            elif adjusted_earnings < INCOME_RANGES["Medium"][1]:
                income_level = "Medium"
            else:
                income_level = "High"

            return income_level, round(adjusted_earnings, 2)
        except Exception as e:
            raise GenerationError(f"Income generation failed: {str(e)}")

    def generate_shopping_preferences(
        self,
        age: int,
        income_level: IncomeLevel,
        region: str,
        employment: EmploymentStatus,
        race: str,
        ethnicity: str,
        gender: str,
    ) -> Dict[str, Any]:
        """Generate shopping preferences based on demographics."""
        try:
            n_products = (
                random.randint(2, 4)
                if income_level == IncomeLevel.LOW
                else (
                    random.randint(3, 5)
                    if income_level == IncomeLevel.MEDIUM
                    else random.randint(3, 6)
                )
            )

            n_channels = (
                random.randint(1, 3)
                if age > 65
                else random.randint(2, 4) if age > 45 else random.randint(2, 5)
            )

            n_engagements = (
                random.randint(3, 7)
                if employment == EmploymentStatus.EMPLOYED
                else random.randint(2, 4)
            )

            product_prefs = PRODUCT_CATEGORIES.copy()
            if region == "South":
                product_prefs.extend(["Condiments", "Meat Products"])
            elif region == "West":
                product_prefs.extend(["Plant-Based Foods", "Snacks"])
            if ethnicity == "Hispanic or Latino":
                product_prefs.extend(["Hispanic Foods", "Spices"])

            channel_prefs = COMMUNICATION_CHANNELS.copy()
            if age > 50:
                channel_prefs.extend(["TV Commercials", "Print Ads"])
            elif age < 30:
                channel_prefs.extend(["Social Media", "Mobile App Notifications"])

            engagement_prefs = ENGAGEMENT_PREFERENCES.copy()
            if income_level == IncomeLevel.HIGH:
                engagement_prefs.extend(
                    ["Loyalty Programs", "Personalized Product Recommendations"]
                )
            elif income_level == IncomeLevel.LOW:
                engagement_prefs.extend(["Coupons", "Seasonal Promotions"])

            return {
                "Preferred_Product_Categories": ",".join(
                    random.sample(product_prefs, k=min(n_products, len(product_prefs)))
                ),
                "Preferred_Communication_Channels": ",".join(
                    random.sample(channel_prefs, k=min(n_channels, len(channel_prefs)))
                ),
                "Engagement_Preferences": ",".join(
                    random.sample(
                        engagement_prefs, k=min(n_engagements, len(engagement_prefs))
                    )
                ),
                "Buying_Frequency": self.generate_buying_frequency(
                    age, income_level, employment
                ).value,
            }
        except Exception as e:
            raise GenerationError(f"Shopping preferences generation failed: {str(e)}")

    def generate_buying_frequency(
        self, age: int, income_level: IncomeLevel, employment_status: EmploymentStatus
    ) -> BuyingFrequency:
        """Generate buying frequency based on demographics."""
        try:
            if employment_status != EmploymentStatus.EMPLOYED:
                base_prob = [0.1, 0.2, 0.5, 0.2]
            else:
                if income_level == IncomeLevel.HIGH:
                    base_prob = [0.3, 0.4, 0.2, 0.1]
                elif income_level == IncomeLevel.LOW:
                    base_prob = [0.1, 0.2, 0.5, 0.2]
                else:
                    base_prob = [0.2, 0.3, 0.4, 0.1]

                if age > 50:
                    base_prob = [
                        base_prob[0] + 0.1,
                        base_prob[1],
                        base_prob[2] - 0.05,
                        base_prob[3] - 0.05,
                    ]
                elif age < 30:
                    base_prob = [
                        base_prob[0] - 0.05,
                        base_prob[1] - 0.05,
                        base_prob[2] + 0.1,
                        base_prob[3],
                    ]

            total = sum(base_prob)
            normalized_prob = [p / total for p in base_prob]

            return np.random.choice(list(BuyingFrequency), p=normalized_prob)
        except Exception as e:
            raise GenerationError(f"Buying frequency generation failed: {str(e)}")

    def generate_synthetic_consumer_data(self, n_samples: int = 200000) -> pd.DataFrame:
        """Generate synthetic consumer data."""
        if n_samples <= 0:
            raise GenerationError("n_samples must be positive")

        data = []
        try:
            for _ in tqdm(range(n_samples), desc="Generating synthetic data"):
                demographics = self.generate_demographics()
                education = self.generate_education_level(
                    demographics["age"], demographics["race"], demographics["ethnicity"]
                )

                labor_force_rate = LABOR_FORCE_PARTICIPATION.get(
                    (
                        demographics["race"]
                        if demographics["race"] != "Other"
                        else "White"
                    ),
                    {},
                ).get(
                    (
                        demographics["gender"]
                        if demographics["gender"] != "Other"
                        else "Male"
                    ),
                    0.5,
                )

                employment = self.generate_employment_status(
                    demographics["age"],
                    education,
                    demographics["race"],
                    demographics["gender"],
                    labor_force_rate,
                )

                income_level, weekly_earnings = self.generate_income_and_earnings(
                    demographics["age"],
                    education,
                    employment,
                    demographics["race"],
                    demographics["gender"],
                    demographics["ethnicity"],
                )

                region = np.random.choice(
                    list(REGION_WEIGHTS.keys()), p=list(REGION_WEIGHTS.values())
                )

                consumer = {
                    "Consumer_ID": self.fake.uuid4(),
                    "Age": demographics["age"],
                    "Gender": demographics["gender"],
                    "Race": demographics["race"],
                    "Ethnicity": demographics["ethnicity"],
                    "Education_Level": education.value,
                    "Employment_Status": employment.value,
                    "Income_Level": income_level,
                    "Weekly_Earnings": weekly_earnings,
                    "Region": region,
                    "State": np.random.choice(REGIONS[region]),
                    "City": self.fake.city(),
                    "ZIP_Code": self.fake.zipcode(),
                }

                consumer.update(
                    self.generate_shopping_preferences(
                        demographics["age"],
                        IncomeLevel(income_level),
                        region,
                        employment,
                        demographics["race"],
                        demographics["ethnicity"],
                        demographics["gender"],
                    )
                )

                data.append(consumer)

            self.synthetic_consumer_data = pd.DataFrame(data)
            return self.synthetic_consumer_data

        except Exception as e:
            raise GenerationError(f"Data generation failed: {str(e)}")

    def analyze_data(self) -> str:
        """Analyze the generated synthetic data and return a summary."""
        if self.synthetic_consumer_data is None:
            raise GenerationError("No data to analyze. Please generate data first.")

        analysis = []
        try:
            # 1. Demographic Distribution Analysis
            analysis.append("## 1. Demographic Distributions")

            analysis.append("\n### Age Distribution:")
            age_dist = pd.cut(
                self.synthetic_consumer_data["Age"],
                bins=[0, 19, 24, 34, 44, 54, 64, 100],
                labels=["16-19", "20-24", "25-34", "35-44", "45-54", "55-64", "65+"],
            ).value_counts(normalize=True)
            analysis.append(age_dist.to_string())

            analysis.append("\n### Race Distribution:")
            race_dist = self.synthetic_consumer_data["Race"].value_counts(
                normalize=True
            )
            analysis.append(race_dist.to_string())

            analysis.append("\n### Ethnicity Distribution:")
            eth_dist = self.synthetic_consumer_data["Ethnicity"].value_counts(
                normalize=True
            )
            analysis.append(eth_dist.to_string())

            analysis.append("\n### Gender Distribution:")
            gender_dist = self.synthetic_consumer_data["Gender"].value_counts(
                normalize=True
            )
            analysis.append(gender_dist.to_string())

            # 2. Employment and Education Analysis
            analysis.append("\n## 2. Employment and Education Analysis")

            analysis.append("\n### Employment Status by Race:")
            emp_by_race = pd.crosstab(
                self.synthetic_consumer_data["Race"],
                self.synthetic_consumer_data["Employment_Status"],
                normalize="index",
            )
            analysis.append(emp_by_race.to_string())

            analysis.append("\n### Education Level by Race:")
            edu_by_race = pd.crosstab(
                self.synthetic_consumer_data["Race"],
                self.synthetic_consumer_data["Education_Level"],
                normalize="index",
            )
            analysis.append(edu_by_race.to_string())

            # 3. Income and Earnings Analysis
            analysis.append("\n## 3. Income and Earnings Analysis")

            analysis.append("\n### Median Weekly Earnings by Race and Gender:")
            earnings_by_demo = self.synthetic_consumer_data[
                self.synthetic_consumer_data["Employment_Status"] == "Employed"
            ].pivot_table(
                values="Weekly_Earnings",
                index="Race",
                columns="Gender",
                aggfunc="median",
            )
            analysis.append(earnings_by_demo.to_string())

            analysis.append("\n### Income Level Distribution by Race:")
            income_by_race = pd.crosstab(
                self.synthetic_consumer_data["Race"],
                self.synthetic_consumer_data["Income_Level"],
                normalize="index",
            )
            analysis.append(income_by_race.to_string())

            # 4. Regional Analysis
            analysis.append("\n## 4. Regional Analysis")

            analysis.append("\n### Race Distribution by Region:")
            race_by_region = pd.crosstab(
                self.synthetic_consumer_data["Region"],
                self.synthetic_consumer_data["Race"],
                normalize="index",
            )
            analysis.append(race_by_region.to_string())

            analysis.append("\n### Median Earnings by Region:")
            earnings_by_region = (
                self.synthetic_consumer_data[
                    self.synthetic_consumer_data["Employment_Status"] == "Employed"
                ]
                .groupby("Region")["Weekly_Earnings"]
                .median()
            )
            analysis.append(earnings_by_region.to_string())

            # 5. Shopping Behavior Analysis
            analysis.append("\n## 5. Shopping Behavior Analysis")

            analysis.append("\n### Buying Frequency by Income Level:")
            buying_freq = pd.crosstab(
                self.synthetic_consumer_data["Income_Level"],
                self.synthetic_consumer_data["Buying_Frequency"],
                normalize="index",
            )
            analysis.append(buying_freq.to_string())

            # 6. Validation Against Expected Distributions
            analysis.append("\n## 6. Validation Analysis")

            # Employment rate validation
            synthetic_emp_rates = (
                self.synthetic_consumer_data[
                    self.synthetic_consumer_data["Employment_Status"] == "Employed"
                ]
                .groupby("Race")
                .size()
                / self.synthetic_consumer_data.groupby("Race").size()
            )

            analysis.append("\n### Employment Rate Comparison:")
            analysis.append("Race\tSynthetic\tActual\tDifference")
            for race in EMPLOYMENT_RATE_BY_RACE.keys():
                if race in synthetic_emp_rates:
                    diff = synthetic_emp_rates[race] - EMPLOYMENT_RATE_BY_RACE[race]
                    analysis.append(
                        f"{race}\t{synthetic_emp_rates[race]:.3f}\t"
                        f"{EMPLOYMENT_RATE_BY_RACE[race]:.3f}\t{diff:.3f}"
                    )

            return "\n".join(analysis)

        except Exception as e:
            raise GenerationError(f"Data analysis failed: {str(e)}")

    def save_data(self, filename: str = "synthetic_consumer_data.csv") -> None:
        """Save the generated synthetic data to a CSV file."""
        if self.synthetic_consumer_data is None:
            raise GenerationError("No data to save. Please generate data first.")

        try:
            self.output.save_to_csv(self.synthetic_consumer_data, filename)
            self.output.write(f"Data saved successfully to {filename}")
        except Exception as e:
            raise GenerationError(f"Failed to save data: {str(e)}")

    def validate_distributions(self) -> Dict[str, float]:
        """Validate the generated data distributions against expected ranges."""
        if self.synthetic_consumer_data is None:
            raise GenerationError("No data to validate. Please generate data first.")

        try:
            metrics = {}

            # Employment status distribution
            emp_status_dist = self.synthetic_consumer_data[
                "Employment_Status"
            ].value_counts(normalize=True)
            metrics["employment_status_diff"] = max(
                abs(
                    emp_status_dist.get(status, 0)
                    - EMPLOYMENT_STATUS_DISTRIBUTION[status]
                )
                for status in EMPLOYMENT_STATUS_DISTRIBUTION.keys()
            )

            # Education distribution
            edu_dist = self.synthetic_consumer_data["Education_Level"].value_counts(
                normalize=True
            )
            metrics["education_diff"] = max(
                abs(edu_dist.get(level, 0) - EDUCATION_DISTRIBUTION[level])
                for level in EDUCATION_DISTRIBUTION.keys()
            )

            # Income distribution
            income_dist = self.synthetic_consumer_data["Income_Level"].value_counts(
                normalize=True
            )
            metrics["income_diff"] = max(
                abs(income_dist.get(level, 0) - INCOME_DISTRIBUTION[level])
                for level in INCOME_DISTRIBUTION.keys()
            )

            # Age distribution
            age_mean = self.synthetic_consumer_data["Age"].mean()
            metrics["age_mean"] = age_mean

            # Gender ratio
            gender_dist = self.synthetic_consumer_data["Gender"].value_counts(
                normalize=True
            )
            metrics["gender_ratio"] = gender_dist.get("Female", 0)

            # Regional distribution
            region_dist = self.synthetic_consumer_data["Region"].value_counts(
                normalize=True
            )
            metrics["regional_distribution"] = max(region_dist)

            # Education rate
            high_edu_rate = self.synthetic_consumer_data[
                self.synthetic_consumer_data["Education_Level"].isin(
                    ["Bachelor", "Advanced"]
                )
            ].shape[0] / len(self.synthetic_consumer_data)
            metrics["high_education_rate"] = high_edu_rate

            # Income distribution skew
            income_mapping = {"Low": 0, "Medium": 1, "High": 2}
            income_numeric = self.synthetic_consumer_data["Income_Level"].map(
                income_mapping
            )
            metrics["income_distribution_skew"] = income_numeric.skew()

            # Validate against ranges
            for metric, (min_val, max_val) in VALIDATION_RANGES.items():
                if metric in metrics and not min_val <= metrics[metric] <= max_val:
                    self.output.write(
                        f"Warning: {metric} ({metrics[metric]:.3f}) outside expected range [{min_val}, {max_val}]"
                    )

            return metrics

        except Exception as e:
            raise GenerationError(f"Distribution validation failed: {str(e)}")


def test_generator() -> None:
    """Test the SyntheticDataGenerator functionality."""
    try:
        output = Output(output_type="both", output_dir="test_output")
        generator = SyntheticDataGenerator(output)

        # Generate test data
        data = generator.generate_synthetic_consumer_data(n_samples=1000)
        generator.save_data("test_synthetic_data.csv")

        # Analyze and validate
        analysis = generator.analyze_data()
        validation_metrics = generator.validate_distributions()

        # Write results
        output.write(analysis)
        output.write("\n## Validation Metrics")
        for metric, value in validation_metrics.items():
            output.write(f"{metric}: {value:.4f}")

        output.save_to_markdown("test_analysis.md")

        print("All tests completed successfully")

    except Exception as e:
        print(f"Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_generator()
