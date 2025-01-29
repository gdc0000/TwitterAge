# updated_eda.py

import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import shapiro, spearmanr, pearsonr
import statsmodels.formula.api as smf
import io
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration
DEFAULT_DATA_PATH = 'data/output.csv'  # Relative path

# Authentication Function
def authenticate(password):
    """Check if the entered password matches the stored password."""
    return password == st.secrets["credentials"]["password"]

# Login Interface
def login():
    """Display a login form and authenticate the user."""
    if 'authentication_status' not in st.session_state:
        st.session_state.authentication_status = False

    if not st.session_state.authentication_status:
        st.title("🔒 Login Required")
        password = st.text_input("Enter Password", type="password")
        if st.button("Login"):
            if authenticate(password):
                st.session_state.authentication_status = True
                st.success("✅ Authentication successful!")
                st.experimental_rerun()
            else:
                st.error("❌ Incorrect password. Please try again.")

# Data Loading and Filtering Functions
@st.cache_data
def load_data(file):
    """
    Load the dataset and identify all columns ending with '_word_perc'.
    """
    try:
        df = pd.read_csv(file)
        # Identify all columns that end with '_word_perc'
        perc_cols = [col for col in df.columns if col.endswith('_word_perc')]
        required_cols = ['age', 'botScore', 'n_tokens', 'gender'] + perc_cols
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            st.error(f"Missing columns in the data: {missing_cols}")
            st.stop()
        # Drop rows with missing values in required columns
        df = df[required_cols].dropna()
        return df, perc_cols
    except FileNotFoundError:
        st.error(f"Data file not found at path: {DEFAULT_DATA_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        st.stop()

@st.cache_data
def filter_data(df, bot_min, bot_max, selected_genders, lower_percentile, upper_percentile):
    """
    Apply filters based on botScore, gender, and text length percentiles.
    """
    # Apply botScore and gender filters
    filtered_df = df[
        (df['botScore'] >= bot_min) &
        (df['botScore'] <= bot_max) &
        (df['gender'].isin(selected_genders))
    ]

    # Apply n_tokens percentile-based trimming
    lower_bound = filtered_df['n_tokens'].quantile(lower_percentile / 100)
    upper_bound = filtered_df['n_tokens'].quantile(upper_percentile / 100)
    filtered_df = filtered_df[
        (filtered_df['n_tokens'] >= lower_bound) &
        (filtered_df['n_tokens'] <= upper_bound)
    ]

    return filtered_df

def main():
    # Check authentication
    if not st.session_state.get('authentication_status'):
        login()
        return  # Stop further execution until authenticated

    st.set_page_config(page_title="Twitter EDA Dashboard", layout="wide")

    # Header Section: Project Overview and Contributors
    with st.container():
        st.title("Age and Motivation: Analysis of Twitter Data")
        st.markdown("""
        **Contributors**  
        Ewa Szumowska, Gabriele Di Cicco, and Gabriela Czarnek  
        Centre for Social Cognitive Studies, Institute of Psychology,  
        Jagiellonian University in Kraków, Poland  
        
        **Project Description**  
        This project examines the relationship between age and individuals' levels of energy (arousal) and motivation (locomotion and assessment). Additionally, it analyzes the relationship between age and proxies for extreme, norm-violating behavior, such as the use of curse words and expressions of anger. The analysis utilizes Twitter data to explore these dynamics and test predefined hypotheses.
        
        [Associated Project on OSF](https://osf.io/d2cyb) | [Internet Archive Link](https://archive.org/details/osf-registrations-g5sym-v1) | [Registration DOI](https://doi.org/10.17605/OSF.IO/G5SYM)
        """)

    # File uploader for data
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"], help="Ensure the CSV contains the required columns.")

    if uploaded_file:
        df, perc_cols = load_data(uploaded_file)
    else:
        # Attempt to load default data
        try:
            df, perc_cols = load_data(DEFAULT_DATA_PATH)
        except:
            st.warning("Please upload a CSV file to proceed.")
            st.stop()

    # Initialize session state for filters
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = df.copy()

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")

        # botScore Slider
        bot_min = float(df['botScore'].min())
        bot_max = float(df['botScore'].max())
        bot_score_slider = st.slider(
            "Filter by botScore:",
            min_value=bot_min,
            max_value=bot_max,
            value=(bot_min, bot_max),
            step=(bot_max - bot_min) / 100  # Adjust step as needed
        )

        # Gender Multiselect
        gender_options = sorted(df['gender'].unique().tolist())
        selected_genders = st.multiselect(
            "Select Gender Categories to Include:",
            options=gender_options,
            default=gender_options  # All selected by default
        )

        # Text Length Filter (n_tokens) - Percentile-based trimming
        st.markdown("### Text Length Filter")
        st.markdown("Exclude extreme text lengths based on word count percentiles.")

        col_lower, col_upper = st.columns(2)
        with col_lower:
            lower_percentile = st.number_input(
                "Lower Percentile (%)",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=1.0,
                help="Define the lower percentile to exclude short texts."
            )
        with col_upper:
            upper_percentile = st.number_input(
                "Upper Percentile (%)",
                min_value=0.0,
                max_value=100.0,
                value=95.0,
                step=1.0,
                help="Define the upper percentile to exclude long texts."
            )

        # Ensure lower_percentile is less than upper_percentile
        if lower_percentile >= upper_percentile:
            st.error("Lower percentile must be less than upper percentile.")

        # Apply Filters Button
        apply_filters = st.button("Apply Filters")

        # Additional controls
        show_significant = st.checkbox("Show only significant correlations (p < 0.05)", True)
        hist_log_scale = st.checkbox("Log scale for histograms", False)
        selected_corr_var = st.selectbox("Select variable for correlation plot:", perc_cols)

    # Apply filters when button is clicked
    if apply_filters:
        if lower_percentile >= upper_percentile:
            st.warning("Please set the lower percentile less than the upper percentile.")
        else:
            st.session_state.filtered_df = filter_data(
                df,
                bot_min=bot_score_slider[0],
                bot_max=bot_score_slider[1],
                selected_genders=selected_genders,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile
            )

    # Use the filtered data
    filtered_df = st.session_state.filtered_df

    # Check if filtered_df is empty
    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your filters.")
        st.stop()

    # Descriptive Statistics
    with st.expander("Descriptive Statistics", expanded=True):
        st.header("Sample Descriptive Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("N", f"{filtered_df.shape[0]}")

        with col2:
            mean_age = filtered_df['age'].mean()
            st.metric("Mean Age", f"{mean_age:.2f}")

        with col3:
            sd_age = filtered_df['age'].std()
            st.metric("SD Age", f"{sd_age:.2f}")

        with col4:
            age_min = filtered_df['age'].min()
            age_max = filtered_df['age'].max()
            st.metric("Age Range", f"{age_min} - {age_max}")

        with col5:
            gender_counts = filtered_df['gender'].value_counts()
            gender_dist = ", ".join([f"{k}: {v}" for k, v in gender_counts.items()])
            st.text(f"Gender Distribution:\n{gender_dist}")

        st.markdown("---")

        st.header("Data Distributions")

        # Create a grid of histograms
        num_cols = 3
        cols = st.columns(num_cols)
        variables = ['age', 'botScore', 'n_tokens'] + perc_cols

        for idx, var in enumerate(variables):
            with cols[idx % num_cols]:
                fig = px.histogram(
                    filtered_df,
                    x=var,
                    nbins=50,
                    title=f'{var} Distribution',
                    log_y=hist_log_scale
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # Normality Analysis
    with st.expander("Normality Tests"):
        st.header("Shapiro-Wilk Normality Results")

        @st.cache_data
        def compute_normality(df, perc_cols):
            normality_results = []
            for var in perc_cols:
                try:
                    stat, p = shapiro(df[var])
                    normality_results.append({
                        'Variable': var,
                        'W-statistic': stat,
                        'p-value': p,
                        'Normal (α=0.05)': p > 0.05
                    })
                except Exception as e:
                    normality_results.append({
                        'Variable': var,
                        'W-statistic': None,
                        'p-value': None,
                        'Normal (α=0.05)': None,
                        'Error': str(e)
                    })
            return pd.DataFrame(normality_results)

        norm_df = compute_normality(filtered_df, perc_cols)
        st.dataframe(
            norm_df.style.format({'W-statistic': '{:.3f}', 'p-value': '{:.4e}'})
                .applymap(
                    lambda x: 'color: green' if x == True else 'color: red' if x == False else '',
                    subset=['Normal (α=0.05)']
                ),
            use_container_width=True
        )

    # Correlation Analysis
    with st.expander("Correlation Analysis"):
        st.header("Correlation with Age")

        @st.cache_data
        def compute_correlations(df, perc_cols):
            corr_results = []
            for var in perc_cols:
                try:
                    # Spearman Correlation
                    spearman_corr, spearman_pval = spearmanr(df['age'], df[var])
                    # Pearson Correlation
                    pearson_corr, pearson_pval = pearsonr(df['age'], df[var])
                    corr_results.append({
                        'Variable': var,
                        'Pearson r': pearson_corr,
                        'Pearson p-value': pearson_pval,
                        'Spearman ρ': spearman_corr,
                        'Spearman p-value': spearman_pval
                    })
                except Exception as e:
                    corr_results.append({
                        'Variable': var,
                        'Pearson r': None,
                        'Pearson p-value': None,
                        'Spearman ρ': None,
                        'Spearman p-value': None,
                        'Error': str(e)
                    })
            return pd.DataFrame(corr_results)

        corr_df = compute_correlations(filtered_df, perc_cols)

        if show_significant:
            corr_df = corr_df[
                (corr_df['Pearson p-value'] < 0.05) |
                (corr_df['Spearman p-value'] < 0.05)
            ]

        st.subheader("Correlation Table")
        st.dataframe(
            corr_df.style.format({
                'Pearson r': '{:.3f}',
                'Pearson p-value': '{:.4e}',
                'Spearman ρ': '{:.3f}',
                'Spearman p-value': '{:.4e}'
            }),
            use_container_width=True
        )

        st.markdown("---")

        # Correlation Heatmap (Using Spearman ρ)
        if not corr_df.empty and 'Spearman ρ' in corr_df.columns:
            corr_matrix = corr_df.set_index('Variable')[['Spearman ρ']]
            max_abs = max(corr_matrix['Spearman ρ'].abs().max(), 0.1)  # Ensure minimum range

            fig = px.imshow(
                corr_matrix.T,
                color_continuous_scale='RdBu',
                zmin=-max_abs,
                zmax=max_abs,
                title="Spearman Correlation Heatmap",
                labels=dict(color="Spearman ρ")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No significant correlations found or missing 'Spearman ρ' data.")

        st.markdown("---")

        # Detailed Correlation Plot
        st.subheader(f"Age vs {selected_corr_var}")
        try:
            fig = px.scatter(
                filtered_df,
                x='age',
                y=selected_corr_var,
                trendline="lowess",
                hover_data=['botScore'],
                color='botScore',
                color_continuous_scale='Bluered',
                title=f"Age vs {selected_corr_var}"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating scatter plot: {e}")

    # Multiple Regression Analysis
    with st.expander("Multiple Regression Analysis", expanded=True):
        st.header("Multiple Regression Models")
        st.markdown("""
        **Instructions**:
        - Enter your regression model formulas below, one per line, using the following syntax:
            ```
            dependent_variable ~ independent_variable1 + independent_variable2 + ...
            ```
        - After entering the models, click the **Run Regression** button to execute and view the results.
        
        **Example**:
            ```
            Arousal_word_perc ~ age + botScore + gender
            Motivation_word_perc ~ age + n_tokens + gender
            ```
        """)

        # Text area for user to input multiple regression formulas
        regression_syntax = st.text_area(
            "Enter Regression Model Formulas",
            height=150,
            value="""
# Example Models
Arousal_word_perc ~ age + botScore + gender
Motivation_word_perc ~ age + n_tokens + gender
""",
            help="Define each regression model on a separate line using the syntax: dependent ~ independent1 + independent2 + ..."
        )

        # Run Regression button
        run_regression = st.button("Run Regression")

        if run_regression:
            if not regression_syntax.strip():
                st.error("Please enter at least one regression model formula.")
            else:
                # Split the input into separate lines
                models = [line.strip() for line in regression_syntax.strip().split('\n') if line.strip() and not line.strip().startswith('#')]

                if not models:
                    st.error("No valid regression models found. Ensure each model is on a separate line and follows the syntax.")
                else:
                    for idx, model_formula in enumerate(models, start=1):
                        st.subheader(f"Model {idx}: {model_formula}")
                        try:
                            # Fit the regression model using statsmodels
                            model = smf.ols(formula=model_formula, data=filtered_df).fit()

                            # Display the regression summary
                            st.text(model.summary())
                            st.markdown("---")
                        except Exception as e:
                            st.error(f"Error in Model {idx}: {e}")
                            st.markdown("---")

if __name__ == "__main__":
    main()
