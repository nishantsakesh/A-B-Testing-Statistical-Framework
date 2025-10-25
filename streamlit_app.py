import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO

# Assuming the framework is in a file named ab_testing_framework.py
# in the same directory or in PYTHONPATH
try:
    from ab_testing_framework import (
        SampleSizeCalculator,
        HypothesisTester,
        EffectSizeCalculator,
        MultipleTesting,
        Visualizer,
        Utils
    )
except ImportError:
    st.error("FATAL ERROR: Could not find 'ab_testing_framework.py'. Make sure it's in the same directory.")
    st.stop()

# --- Page Config ---
st.set_page_config(
    page_title="A/B Testing Framework",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Instantiate Tools ---
# We instantiate them with default values, which can be changed by user inputs
ALPHA = 0.05
POWER = 0.80
calculator = SampleSizeCalculator(alpha=ALPHA, power=POWER)
tester = HypothesisTester(alpha=ALPHA)
effect_calc = EffectSizeCalculator()
viz = Visualizer()
utils = Utils()


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "1. Sample Size Calculator", 
    "2. A/B Test Analyzer", 
    "3. Simulation Lab", 
    "4. Multiple Testing Correction", 
    "5. Learning Resources (Pitfalls)"
])

st.sidebar.markdown("---")
st.sidebar.title("Global Settings")
global_alpha = st.sidebar.slider("Significance Level (Alpha)", 0.01, 0.20, ALPHA, 0.01)

# Update tester alpha
tester.alpha = global_alpha
calculator.alpha = global_alpha


# --- Page 1: Sample Size Calculator ---
if page.startswith("1."):
    st.title("1. Sample Size Calculator")
    st.markdown("Determine the sample size needed *before* you start your test.")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Test Parameters")
        baseline_rate = st.number_input("Baseline Conversion Rate (e.g., 0.10 for 10%)", 0.001, 0.999, 0.10, 0.01, format="%.3f")
        mde_type = st.radio("Minimum Detectable Effect (MDE) Type", ["Relative", "Absolute"])
        
        mde_rel = None
        mde_abs = None

        if mde_type == "Relative":
            mde_rel_perc = st.number_input("Relative MDE (e.g., 10 for 10% lift)", 0.1, 100.0, 10.0, 0.5)
            mde_rel = mde_rel_perc / 100.0
            st.markdown(f"This means detecting a change from **{baseline_rate:.2%}** to **{baseline_rate * (1 + mde_rel):.2%}**.")
            mde_abs = baseline_rate * mde_rel
        else:
            mde_abs_perc = st.number_input("Absolute MDE (e.g., 2 for 2% lift)", 0.1, 50.0, 2.0, 0.1)
            mde_abs = mde_abs_perc / 100.0
            st.markdown(f"This means detecting a change from **{baseline_rate:.2%}** to **{baseline_rate + mde_abs:.2%}**.")
        
        power = st.slider("Statistical Power (1 - Beta)", 0.50, 0.99, POWER, 0.01)
        calculator.power = power

        if st.button("Calculate Sample Size", type="primary"):
            try:
                sample_size = calculator.calculate_sample_size(
                    baseline_rate=baseline_rate,
                    mde_absolute=mde_abs
                )
                st.success(f"**Required Sample Size per Variant: `{sample_size:,.0f}`**")
                st.markdown(f"You need a total of **`{sample_size*2:,.0f}`** users to run this test with **{power*100:.0f}%** power and **{global_alpha*100:.0f}%** significance level.")
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        st.subheader("Power vs. Sample Size Curve")
        try:
            max_n = 50000
            if 'sample_size' in locals() and sample_size > 0:
                max_n = max(50000, sample_size * 2) # Adjust plot range if needed
                
            fig = viz.plot_power_curve(baseline_rate, mde_abs, global_alpha, max_n=max_n)
            
            if 'sample_size' in locals() and sample_size > 0:
                fig.add_vline(x=sample_size, line_dash="solid", line_color="blue", name=f"Required N ({sample_size:,.0f})")

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate plot: {e}")


# --- Page 2: A/B Test Analyzer ---
elif page.startswith("2."):
    st.title("2. A/B Test Analyzer")
    st.markdown("Upload your test results (or enter manually) to see the analysis.")
    st.markdown("---")

    # Data Input Method
    input_method = st.radio("How to input data?", ["Enter Manually", "Upload CSV"], horizontal=True)
    
    c_a, n_a, c_b, n_b = 0, 0, 0, 0

    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        st.info("CSV must have columns: `variant` (A or B) and `converted` (1 or 0).")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.dataframe(data.head())
            
            try:
                summary = data.groupby('variant')['converted'].agg(['sum', 'count']).reset_index()
                
                c_a = summary.loc[summary['variant'] == 'A', 'sum'].values[0]
                n_a = summary.loc[summary['variant'] == 'A', 'count'].values[0]
                c_b = summary.loc[summary['variant'] == 'B', 'sum'].values[0]
                n_b = summary.loc[summary['variant'] == 'B', 'count'].values[0]
                
                st.markdown("Data loaded from CSV:")
                st.markdown(f"- **Control (A):** `{c_a}` conversions / `{n_a}` users")
                st.markdown(f"- **Treatment (B):** `{c_b}` conversions / `{n_b}` users")

            except Exception as e:
                st.error(f"Error processing CSV. Make sure columns are 'variant' (A/B) and 'converted' (1/0). Error: {e}")
                
    else: # Manual Input
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Control (A)")
            n_a = st.number_input("Total Users (A)", min_value=1, value=10000, step=100)
            c_a = st.number_input("Conversions (A)", min_value=0, value=1000, step=10)
        
        with col2:
            st.subheader("Treatment (B)")
            n_b = st.number_input("Total Users (B)", min_value=1, value=10000, step=100)
            c_b = st.number_input("Conversions (B)", min_value=0, value=1050, step=10)

    st.markdown("---")
    
    # Practical Significance Input
    st.subheader("Practical Significance (MDE)")
    mde_abs_perc_analysis = st.number_input("What is your Minimum Detectable Effect? (Absolute %, e.g., 1.0)", 0.0, 20.0, 1.0, 0.1)
    mde_abs_analysis = mde_abs_perc_analysis / 100.0
    
    if st.button("Analyze Test Results", type="primary"):
        if n_a <= 0 or n_b <= 0 or c_a > n_a or c_b > n_b:
            st.error("Invalid inputs. Conversions cannot be greater than total users, and users must be > 0.")
        else:
            rate_a = c_a / n_a
            rate_b = c_b / n_b
            
            # --- Run Analysis ---
            try:
                test_results = tester.proportion_z_test(c_a, n_a, c_b, n_b)
                ci = tester.confidence_interval(c_a, n_a, c_b, n_b)
                effect_size = effect_calc.cohens_h(rate_a, rate_b)
                stat_msg, prac_msg, color = utils.interpret_results(test_results, ci, mde_abs_analysis)
                
                # --- Display Results ---
                st.subheader("Test Results")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Control CVR", f"{rate_a:.2%}")
                col2.metric("Treatment CVR", f"{rate_b:.2%}")
                col3.metric("Observed Lift", f"{rate_b - rate_a:+.2%}")
                
                st.markdown(f"**95% Confidence Interval:** `[{ci[0]:.2%}, {ci[1]:.2%}]`")
                st.markdown(f"**P-Value:** `{test_results['p_value']:.4f}`")
                
                # --- Interpretation ---
                st.subheader("Interpretation")
                if color == "green":
                    st.success(f"**Result: {stat_msg} {prac_msg}**")
                elif color == "red":
                    st.error(f"**Result: {stat_msg} {prac_msg}**")
                else: # Orange
                    st.warning(f"**Result: {stat_msg} {prac_msg}**")

                # --- Visualization ---
                st.subheader("Confidence Interval Visualization")
                fig = viz.plot_confidence_interval(ci[0], ci[1], rate_a, rate_b, mde_abs=mde_abs_analysis)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

# --- Page 3: Simulation Lab ---
elif page.startswith("3."):
    st.title("3. Simulation Lab")
    st.markdown("Simulate test data to understand statistical concepts.")
    
    st.subheader("Simulation Parameters")
    col1, col2, col3 = st.columns(3)
    sim_n_a = col1.number_input("Sample Size (A)", 100, 100000, 1000)
    sim_n_b = col2.number_input("Sample Size (B)", 100, 100000, 1000)
    sim_rate_a = col3.number_input("True CVR (A)", 0.0, 1.0, 0.10, 0.01)
    sim_rate_b = col3.number_input("True CVR (B)", 0.0, 1.0, 0.10, 0.01) # Default to no effect
    
    sim_mde_perc = st.number_input("MDE for analysis (%)", 0.1, 20.0, 1.0, 0.1)
    sim_mde = sim_mde_perc / 100.0
    
    if st.button("Run Simulation", type="primary"):
        c_a, n_a = utils.generate_synthetic_data(sim_n_a, sim_rate_a)
        c_b, n_b = utils.generate_synthetic_data(sim_n_b, sim_rate_b)
        
        st.markdown(f"### Simulated Data (True Rates: A={sim_rate_a:.2%}, B={sim_rate_b:.2%})")
        
        # --- Run Analysis on Simulated Data ---
        try:
            rate_a = c_a / n_a
            rate_b = c_b / n_b
            
            test_results = tester.proportion_z_test(c_a, n_a, c_b, n_b)
            ci = tester.confidence_interval(c_a, n_a, c_b, n_b)
            stat_msg, prac_msg, color = utils.interpret_results(test_results, ci, sim_mde)
            
            # --- Display Results ---
            st.subheader("Simulation Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Observed Control CVR", f"{rate_a:.2%}")
            col2.metric("Observed Treatment CVR", f"{rate_b:.2%}")
            col3.metric("Observed Lift", f"{rate_b - rate_a:+.2%}")
            
            st.markdown(f"**95% Confidence Interval:** `[{ci[0]:.2%}, {ci[1]:.2%}]`")
            st.markdown(f"**P-Value:** `{test_results['p_value']:.4f}`")
            
            if color == "green":
                st.success(f"**Result: {stat_msg} {prac_msg}**")
            elif color == "red":
                st.error(f"**Result: {stat_msg} {prac_msg}**")
            else: # Orange
                st.warning(f"**Result: {stat_msg} {prac_msg}**")
                
            fig = viz.plot_confidence_interval(ci[0], ci[1], rate_a, rate_b, mde_abs=sim_mde)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"An error occurred during simulation analysis: {e}")

# --- Page 4: Multiple Testing Correction ---
elif page.startswith("4."):
    st.title("4. Multiple Testing Correction")
    st.markdown("Analyze results from A/B/C... tests. This helps control the Family-Wise Error Rate (FWER) or False Discovery Rate (FDR).")
    
    p_value_input = st.text_area("Enter p-values (comma-separated)", "0.01, 0.04, 0.06, 0.3")
    
    if st.button("Calculate Corrected P-Values", type="primary"):
        try:
            p_values_list = [float(p.strip()) for p in p_value_input.split(',')]
            if not all(0 <= p <= 1 for p in p_values_list):
                st.error("All p-values must be between 0 and 1.")
            else:
                corrector = MultipleTesting(p_values_list, alpha=global_alpha)
                
                bonferroni = corrector.bonferroni_correction()
                fdr = corrector.benjamini_hochberg()
                
                results_df = pd.DataFrame({
                    'Original P-Value': p_values_list,
                    'Bonferroni Corrected P': bonferroni['corrected_p_values'],
                    'Bonferroni Significant': bonferroni['is_significant'],
                    'FDR (B-H) Corrected P': fdr['corrected_p_values'],
                    'FDR Significant': fdr['is_significant']
                })
                
                st.subheader("Correction Results")
                st.dataframe(results_df.style.format({
                    'Original P-Value': '{:.4f}',
                    'Bonferroni Corrected P': '{:.4f}',
                    'FDR (B-H) Corrected P': '{:.4f}'
                }))
                
                st.markdown(f"**Bonferroni Corrected Alpha:** `{bonferroni['corrected_alpha']:.4f}` (Original p-values must be less than this)")
                st.markdown(f"**FDR controlled at:** `{global_alpha*100:.0f}%`")
                
        except Exception as e:
            st.error(f"Error parsing p-values. Make sure they are comma-separated numbers. Error: {e}")


# --- Page 5: Learning Resources ---
elif page.startswith("5."):
    st.title("5. Learning Resources & Common Pitfalls")
    st.markdown("The most common mistakes in A/B testing and how to avoid them.")
    
    st.subheader("🚫 Pitfall 1: Peeking (Early Stopping)")
    st.markdown("""
    **What it is:** Continuously monitoring a test's p-value and stopping it as soon as it crosses the significance threshold (e.g., p < 0.05).
    
    **Why it's bad:** This *dramatically* increases your **Type I Error (False Positive)** rate. If you peek 20 times, your chance of seeing a false positive can be > 40%, not 5%!
    
    **How to fix it:** Use the **Sample Size Calculator**! Determine your sample size *before* the test and commit to running it until you reach that size. Do not analyze results until the test is complete. (The exception is advanced sequential testing, which is complex to implement correctly).
    """)
    
    st.subheader("🚫 Pitfall 2: Ignoring Statistical Power (Small Samples)")
    st.markdown("""
    **What it is:** Running a test with too small a sample size.
    
    **Why it's bad:** The test is **underpowered**, meaning it has a low chance of detecting a *real* effect. This leads to a high **Type II Error (False Negative)** rate. You conclude "no effect" when there actually was one, and you miss a good opportunity.
    
    **How to fix it:** Use the **Sample Size Calculator**. Ensure your test has at least 80% power to detect your Minimum Detectable Effect (MDE).
    """)
    
    st.subheader("🚫 Pitfall 3: Confusing Statistical vs. Practical Significance")
    st.markdown("""
    **What it is:** A test is "statistically significant" (p < 0.05), so you launch it, even if the lift is tiny (e.g., 0.1%).
    
    **Why it's bad:** A 0.1% lift might not be worth the engineering cost or business complexity. With *huge* sample sizes (e.g., 1 million users), even tiny, meaningless effects will become statistically significant.
    
    **How to fix it:** Always look at the **Confidence Interval** and compare it to your **Minimum Detectable Effect (MDE)**. 
    - **Good:** The entire CI is above your MDE (e.g., CI is `[+1.5%, +2.5%]` and MDE is 1%).
    - **Bad:** The entire CI is below your MDE (e.g., CI is `[+0.1%, +0.3%]` and MDE is 1%). This is statistically significant, but *not* practically significant.
    """)
    
    st.subheader("🚫 Pitfall 4: Ignoring the Multiple Comparisons Problem")
    st.markdown("""
    **What it is:** Running an A/B/C/D test and just checking if any p-value is < 0.05. Or, testing 5 different metrics (CVR, AOV, CTR, etc.) and claiming victory if *any* of them is significant.
    
    **Why it's bad:** This inflates your Type I (False Positive) error rate. If you run 4 tests (A vs B, A vs C, A vs D), your true "family-wise" alpha is `1 - (1 - 0.05)^3 = 14.3%`, not 5%!
    
    **How to fix it:** Use the **Multiple Testing Correction** tab!
    - **Bonferroni:** Very strict, good for a few comparisons. Divides your alpha (0.05 / 3 = 0.0167).
    - **Benjamini-Hochberg (FDR):** Less strict, good for many comparisons. It controls the "False Discovery Rate" (e.g., no more than 5% of your "significant" results are false positives).
    """)
