A/B Testing Statistical Framework

This project provides a comprehensive, production-ready Python framework for designing, analyzing, and interpreting A/B tests with statistical rigor. It includes a core logic module, a demonstration notebook, and an interactive Streamlit dashboard.

This framework is designed for data analysts, product managers, and marketers who want to move beyond simple p-value calculations and implement a robust testing culture that avoids common statistical pitfalls.

1. Business Problem Solved

Companies frequently run A/B tests incorrectly, leading to costly errors:

False Positives (Type I Errors): Stopping tests early ("peeking") or running multiple comparisons inflates the false positive rate, leading companies to launch features that have no real effect.

False Negatives (Type II Errors): Ignoring statistical power and running underpowered tests (small sample sizes) causes companies to miss real opportunities, concluding "no effect" when one actually existed.

Significance Confusion: Teams often confuse statistical significance (p < 0.05) with practical significance (is the lift big enough to matter?).

This framework solves these problems by providing tools for:

Correct Test Design: A SampleSizeCalculator to ensure tests are properly powered.

Robust Analysis: A HypothesisTester that provides p-values and, more importantly, confidence intervals.

Multiple Test Correction: A MultipleTesting module to control errors when testing multiple variants or metrics.

Clear Interpretation: A Visualizer and Utils class to provide plain-English interpretations that distinguish between statistical and practical significance.

2. Core Features

Sample Size Calculator: Calculates required sample size per variant based on baseline rate, MDE, alpha, and power.

A/B Test Analyzer: Performs two-proportion z-tests and Chi-Square tests.

Confidence Intervals: Calculates the 95% CI for the difference in proportions, which is often more useful than a p-value.

Effect Size Calculation: Computes Cohen's h to measure practical significance.

Multiple Testing Correction: Implements Bonferroni (FWER) and Benjamini-Hochberg (FDR) corrections.

Interactive Dashboard: A Streamlit app to use all these tools in a user-friendly UI.

Simulation Lab: A notebook and app tab for simulating test data to understand statistical concepts like power and p-values.

3. Tech Stack

Core Logic: Python 3.8+, NumPy, SciPy, Statsmodels

Data Handling: Pandas

Dashboard: Streamlit

Visualization: Plotly, Matplotlib, Seaborn

4. Project Structure & How to Run

Place the generated files in the following structure:

ab_testing_project/
├── ab_testing_framework.py     # <-- The core logic module
├── ab_testing_demo.ipynb       # <-- The demo notebook
├── streamlit_app.py            # <-- The Streamlit dashboard
├── requirements.txt            # <-- Python packages
└── README.md                   # <-- This file


Step 1: Install Requirements

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt


Step 2: Run the Demo Notebook

Launch Jupyter and open ab_testing_demo.ipynb to see 12+ scenarios and understand the framework's logic.

jupyter notebook ab_testing_demo.ipynb


Step 3: Launch the Streamlit App

Run the following command in your terminal. The app will open in your browser.

streamlit run streamlit_app.py


5. Statistical Concepts Covered

Null (H0) vs. Alternative (H1) Hypothesis: The core of testing. H0 = "no difference", H1 = "is a difference".

Type I Error (Alpha): The (low) probability of a False Positive. Set by alpha (e.g., 0.05).

Type II Error (Beta): The probability of a False Negative.

Statistical Power (1 - Beta): The (high) probability of detecting a real effect. The standard is 0.80.

P-Value: The probability of observing your data (or more extreme) if the null hypothesis is true. It is NOT "the probability the null is true".

Confidence Interval: The range of plausible values for the true effect. If the 95% CI is [+1%, +3%], we are 95% confident the true lift is between 1% and 3%.

Practical Significance (MDE): The minimum lift that the business cares about. A lift of 0.1% might be statistically significant but not practically significant.

Multiple Comparison Problem: Your chance of a false positive increases with every test you run. This requires correction.