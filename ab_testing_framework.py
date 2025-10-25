# -*- coding: utf-8 -*-
"""
A/B Testing Statistical Framework

This module provides a set of classes and functions to design, analyze,
and interpret A/B tests with statistical rigor.

Classes:
- SampleSizeCalculator: For determining required sample sizes.
- HypothesisTester: For running statistical tests (z-test, chi2).
- EffectSizeCalculator: For calculating practical significance (Cohen's h).
- MultipleTesting: For applying corrections (Bonferroni, FDR).
- Visualizer: For plotting results.
- Utils: For data generation and validation.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from statsmodels.stats.multitest import fdrcorrection
import plotly.graph_objects as go

class SampleSizeCalculator:
    """
    Calculates the required sample size for an A/B test.

    Methods:
    - calculate_sample_size: Computes n per variant.
    """
    def __init__(self, alpha=0.05, power=0.80):
        if not (0 < alpha < 1):
            raise ValueError("Alpha (significance level) must be between 0 and 1.")
        if not (0 < power < 1):
            raise ValueError("Power must be between 0 and 1.")
        self.alpha = alpha
        self.power = power
        
    def calculate_sample_size(self, baseline_rate, mde_relative=None, mde_absolute=None):
        """
        Calculates the required sample size per variant for a two-proportion z-test.

        Either mde_relative (e.g., 10% uplift) or mde_absolute (e.g., 2% lift) must be provided.

        Args:
            baseline_rate (float): The conversion rate of the control group (e.g., 0.10).
            mde_relative (float, optional): The minimum detectable effect as a relative 
                                            percentage (e.g., 0.10 for a 10% lift).
            mde_absolute (float, optional): The minimum detectable effect as an absolute 
                                            difference (e.g., 0.02 for 2% lift).

        Returns:
            int: The required sample size per variant.
        """
        if not (0 < baseline_rate < 1):
            raise ValueError("Baseline rate must be between 0 and 1.")
            
        if mde_relative is not None:
            if mde_relative <= 0:
                raise ValueError("Relative MDE must be greater than 0.")
            effect_size = baseline_rate * mde_relative
        elif mde_absolute is not None:
            if mde_absolute <= 0:
                raise ValueError("Absolute MDE must be greater than 0.")
            effect_size = mde_absolute
        else:
            raise ValueError("Either 'mde_relative' or 'mde_absolute' must be provided.")

        # Use statsmodels to calculate effect size (Cohen's h)
        # We need p1 and p2 to calculate Cohen's h
        p1 = baseline_rate
        p2 = baseline_rate + effect_size
        
        # Calculate effect size using Cohen's h for proportions
        es = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

        analysis = NormalIndPower()
        try:
            sample_size = analysis.solve_power(
                effect_size=es,
                alpha=self.alpha,
                power=self.power,
                ratio=1.0,  # Assuming equal sample sizes
                alternative='two-sided'
            )
        except ValueError as e:
            print(f"Error during power calculation: {e}")
            return -1

        return int(np.ceil(sample_size))

class HypothesisTester:
    """
    Performs hypothesis tests for A/B testing.
    """
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def proportion_z_test(self, conversions_a, total_a, conversions_b, total_b):
        """
        Performs a two-proportion z-test.

        Args:
            conversions_a (int): Number of conversions for variant A (control).
            total_a (int): Total sample size for variant A.
            conversions_b (int): Number of conversions for variant B (treatment).
            total_b (int): Total sample size for variant B.

        Returns:
            dict: A dictionary containing the z-statistic, p-value, and significance.
        """
        if total_a == 0 or total_b == 0:
            raise ValueError("Total sample sizes cannot be zero.")
            
        count = np.array([conversions_b, conversions_a])
        nobs = np.array([total_b, total_a])
        
        z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
        
        is_significant = p_value < self.alpha
        
        return {
            'z_stat': z_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'alpha': self.alpha
        }

    def confidence_interval(self, conversions_a, total_a, conversions_b, total_b):
        """
        Calculates the confidence interval for the difference between two proportions.

        Args:
            conversions_a (int): Number of conversions for variant A (control).
            total_a (int): Total sample size for variant A.
            conversions_b (int): Number of conversions for variant B (treatment).
            total_b (int): Total sample size for variant B.

        Returns:
            tuple: (lower_bound, upper_bound) of the confidence interval.
        """
        if total_a == 0 or total_b == 0:
            raise ValueError("Total sample sizes cannot be zero.")

        # Note: statsmodels order is (treatment, control) for confint
        low, upp = confint_proportions_2indep(
            count1=conversions_b, nobs1=total_b,
            count2=conversions_a, nobs2=total_a,
            alpha=self.alpha,
            method='agresti-caffo' # Robust method for CIs
        )
        return low, upp

    def chi_square_test(self, conversions_a, total_a, conversions_b, total_b):
        """
        Performs a Chi-Square test of independence.

        Args:
            conversions_a (int): Number of conversions for variant A (control).
            total_a (int): Total sample size for variant A.
            conversions_b (int): Number of conversions for variant B (treatment).
            total_b (int): Total sample size for variant B.

        Returns:
            dict: A dictionary containing the chi2-statistic, p-value, and significance.
        """
        non_conversions_a = total_a - conversions_a
        non_conversions_b = total_b - conversions_b

        contingency_table = np.array([
            [conversions_b, non_conversions_b],  # Treatment
            [conversions_a, non_conversions_a]   # Control
        ])

        try:
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table, correction=True)
            is_significant = p_value < self.alpha
            return {
                'chi2_stat': chi2,
                'p_value': p_value,
                'is_significant': is_significant,
                'alpha': self.alpha
            }
        except ValueError as e:
            print(f"Error during Chi-Square test: {e}")
            return None


class EffectSizeCalculator:
    """
    Calculates practical significance (effect size).
    """
    def cohens_h(self, rate_a, rate_b):
        """
        Calculates Cohen's h for the difference between two proportions.
        
        Guidelines:
        - 0.20: Small effect
        - 0.50: Medium effect
        - 0.80: Large effect

        Args:
            rate_a (float): Conversion rate for variant A (e.g., 0.10).
            rate_b (float): Conversion rate for variant B (e.g., 0.12).

        Returns:
            float: The Cohen's h value.
        """
        if not (0 <= rate_a <= 1) or not (0 <= rate_b <= 1):
            raise ValueError("Conversion rates must be between 0 and 1.")
            
        # Transform proportions to angles
        phi_a = 2 * np.arcsin(np.sqrt(rate_a))
        phi_b = 2 * np.arcsin(np.sqrt(rate_b))
        
        return phi_b - phi_a

class MultipleTesting:
    """
    Applies corrections for multiple comparisons.
    """
    def __init__(self, p_values, alpha=0.05):
        self.p_values = p_values
        self.alpha = alpha
        self.n_tests = len(p_values)
    
    def bonferroni_correction(self):
        """
        Applies Bonferroni correction.
        
        Returns:
            tuple: (corrected_p_values, significant_results_mask)
        """
        corrected_alpha = self.alpha / self.n_tests
        corrected_p_values = [min(p * self.n_tests, 1.0) for p in self.p_values]
        is_significant = [p < self.alpha for p in corrected_p_values]
        
        return {
            'method': 'Bonferroni',
            'corrected_alpha': corrected_alpha,
            'corrected_p_values': corrected_p_values,
            'is_significant': is_significant
        }

    def benjamini_hochberg(self):
        """
        Applies Benjamini-Hochberg (FDR) correction.

        Returns:
            tuple: (corrected_p_values, significant_results_mask)
        """
        is_significant, corrected_p_values, _, _ = fdrcorrection(
            self.p_values, 
            alpha=self.alpha, 
            method='indep'
        )
        
        return {
            'method': 'Benjamini-Hochberg (FDR)',
            'corrected_p_values': corrected_p_values,
            'is_significant': list(is_significant)
        }

class Visualizer:
    """
    Provides static methods for plotting A/B test results.
    """
    @staticmethod
    def plot_confidence_interval(ci_low, ci_high, rate_a, rate_b, mde_abs=0):
        """
        Plots the confidence interval for the difference in proportions.
        """
        diff = rate_b - rate_a
        
        fig = go.Figure()

        # Add CI bar
        fig.add_trace(go.Bar(
            x=[diff],
            y=['Difference (B - A)'],
            orientation='h',
            error_x=dict(
                type='data',
                symmetric=False,
                array=[ci_high - diff],
                arrayminus=[diff - ci_low]
            ),
            marker_color='blue',
            name='95% Confidence Interval'
        ))

        # Add line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="red", name="No Effect (0.0)")

        # Add MDE line if provided
        if mde_abs > 0:
             fig.add_vline(x=mde_abs, line_dash="dot", line_color="green", name=f"MDE (+{mde_abs*100:.1f}%)")
             fig.add_vline(x=-mde_abs, line_dash="dot", line_color="green", name=f"MDE (-{mde_abs*100:.1f}%)")
        
        fig.update_layout(
            title='Confidence Interval of the Difference (Treatment - Control)',
            xaxis_title='Difference in Conversion Rate',
            xaxis_tickformat=',.2%'
        )
        return fig

    @staticmethod
    def plot_power_curve(baseline_rate, mde, alpha=0.05, max_n=50000):
        """
        Plots statistical power as a function of sample size.
        """
        sample_sizes = np.linspace(100, max_n, 50)
        powers = []
        
        p1 = baseline_rate
        p2 = baseline_rate + mde
        es = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

        analysis = NormalIndPower()
        for n in sample_sizes:
            power = analysis.power(
                effect_size=es,
                nobs1=n,
                alpha=alpha,
                ratio=1.0,
                alternative='two-sided'
            )
            powers.append(power)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample_sizes,
            y=powers,
            mode='lines+markers',
            name='Power'
        ))
        
        fig.add_hline(y=0.80, line_dash="dash", line_color="red", name="Target Power (0.80)")
        
        fig.update_layout(
            title='Power vs. Sample Size',
            xaxis_title='Sample Size (per variant)',
            yaxis_title='Statistical Power'
        )
        return fig

class Utils:
    """
    Utility functions for generating data and validating inputs.
    """
    @staticmethod
    def generate_synthetic_data(n, conversion_rate):
        """
        Generates synthetic binary conversion data.

        Args:
            n (int): Sample size.
            conversion_rate (float): The true conversion rate.

        Returns:
            tuple: (conversions, total)
        """
        if n <= 0:
            raise ValueError("Sample size (n) must be > 0.")
        if not (0 <= conversion_rate <= 1):
             raise ValueError("Conversion rate must be between 0 and 1.")
             
        data = np.random.binomial(1, conversion_rate, n)
        conversions = np.sum(data)
        return conversions, n

    @staticmethod
    def interpret_results(test_results, ci, practical_mde_abs):
        """
        Provides a plain English interpretation of the test results.
        """
        is_significant = test_results['is_significant']
        p_value = test_results['p_value']
        ci_low, ci_high = ci
        
        # Check statistical significance
        if is_significant:
            stat_sig_msg = f"**Statistically Significant (p={p_value:.4f}).** We are >{ (1-test_results['alpha'])*100 :.0f}% confident the observed change is not due to random chance."
            color = "green"
        else:
            stat_sig_msg = f"**Not Statistically Significant (p={p_value:.4f}).** We cannot conclude the observed change is due to the test."
            color = "red"
            
        # Check practical significance
        if ci_low > practical_mde_abs:
            prac_sig_msg = f"**Practically Significant (Positive).** The entire 95% CI ({ci_low*100:.2f}% to {ci_high*100:.2f}%) is above the MDE of {practical_mde_abs*100:.2f}%."
            if not is_significant: color = "orange" # Edge case: practically sig but not stat sig (underpowered)
        elif ci_high < -practical_mde_abs:
            prac_sig_msg = f"**Practically Significant (Negative).** The entire 95% CI ({ci_low*100:.2f}% to {ci_high*100:.2f}%) is below the negative MDE."
            if not is_significant: color = "orange"
        elif ci_low < 0 and ci_high > 0:
            prac_sig_msg = "**Not Practically Significant.** The 95% CI includes 0, meaning no effect is a plausible outcome."
        else:
             prac_sig_msg = "**Inconclusive (Practicality).** The 95% CI is either fully within the MDE bounds or overlaps 0, but not the MDE."
             if is_significant: color = "orange" # Statistically sig, but practically insignificant

        return stat_sig_msg, prac_sig_msg, color
