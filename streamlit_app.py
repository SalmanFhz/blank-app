import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from datetime import timedelta
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, kruskal
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import poisson
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import holidays
import itertools


# Set page config and custom theme
st.set_page_config(
    page_title="Dashboard Pelatihan Dicoding Salman Fadhilurrohman",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.dicoding.com/academies/555',
        'Report a bug': "https://github.com/yourusername/yourrepository/issues",
        'About': "# This is a dashboard created for Dicoding course project."
    }
)


# Set the style and color palette
sns.set_style("whitegrid")
sns.set_palette("deep")

# sns.set(style='whitegrid')

def create_pengunjung_df(df):
    pengunjung_df = df.resample(rule='D', on='dteday').agg({
        "hr": "nunique",
        "cnt": "sum"
    }).reset_index()
    
    pengunjung_df.rename(columns={
        "hr": "order_count",
        "cnt": "total_visitors"
    }, inplace=True)
    
    return pengunjung_df

#load data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['dteday'] = pd.to_datetime(df['dteday'])
    df.sort_values(by="dteday", inplace=True)
    return df.reset_index(drop=True)

# Load data
all_df = load_data("hour.csv")


# Convert dteday to datetime
all_df['dteday'] = pd.to_datetime(all_df['dteday'])
all_df.sort_values(by="dteday", inplace=True)
all_df.reset_index(drop=True, inplace=True)

# Get minimum and maximum dates
min_date = all_df["dteday"].min()
max_date = all_df["dteday"].max()

# Sidebar setup
with st.sidebar:
    st.image("https://www.itcilo.org/sites/default/files/styles/fullscreen_image/public/courses/cover-images/A9017179.jpeg?h=3de5bcb3&itok=G0R4J3RO")
    
    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value=min_date.date(),
        max_value=max_date.date(),
        value=[min_date.date(), max_date.date()]
    )

st.header('Dashboard Bike Sharing Analysis ﾟ ⋆ ﾟ ☂︎ ⋆ ﾟﾟ ⋆ ')



# Generate daily aggregation for visitors
pengunjung_harian_df = create_pengunjung_df(all_df)  # Call the function to create the daily dataframe

# Display Forecast Table
# st.subheader('Tabel Peramalan Pengunjung 30 Hari ke Depan')
# future_forecast_table = future_df[['dteday', 'predicted_visitors']].rename(
#     columns={'dteday': 'Tanggal', 'predicted_visitors': 'Prediksi Pengunjung'}
# )
# st.write(future_forecast_table)


st.subheader('Monthly Visitor Count')
# Modify the function to group by month and sum the cnt values for total visitors
def create_pengunjung_bulanan_df(df):
    pengunjung_bulanan_df = df.resample(rule='M', on='dteday').agg({
        "cnt": "sum"
    }).reset_index()
    
    pengunjung_bulanan_df.rename(columns={
        "cnt": "total_visitors"
    }, inplace=True)
    
    return pengunjung_bulanan_df

# Create the monthly data for visitors
pengunjung_bulanan_df = create_pengunjung_bulanan_df(all_df)

# Monthly Visitor Section
# st.subheader('Monthly Visitors')
col1, col2 = st.columns(2)

# Monthly Visitors and Metrics
with col1:
    total_monthly_visits = pengunjung_bulanan_df['total_visitors'].sum()
    st.metric("Total Monthly Visits", value=total_monthly_visits)

with col2:
    avg_monthly_visits = pengunjung_bulanan_df['total_visitors'].mean()
    st.metric("Average Monthly Visits", value=int(avg_monthly_visits))

# Plot Monthly Visitor Counts with Data Labels
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    pengunjung_bulanan_df["dteday"],
    pengunjung_bulanan_df["total_visitors"],
    marker='o',
    linewidth=2,
    color="#5c4219",
    label="Monthly Visitor Count"
)

# Adding data labels
for i, value in enumerate(pengunjung_bulanan_df["total_visitors"]):
    ax.text(
        pengunjung_bulanan_df["dteday"].iloc[i],
        value + 0.5,
        str(value),
        color='black',
        fontsize=10,
        ha='center'
    )

# Chart settings
ax.set_title("Monthly Visitor Count", fontsize=18)
ax.set_xlabel("Month", fontsize=14)
ax.set_ylabel("Total Visitors", fontsize=14)
ax.legend()
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12, rotation=45)

# Display the chart in Streamlit
st.pyplot(fig)


# Load data
# df = pd.read_csv("hour.csv")
all_df['dteday'] = pd.to_datetime(all_df['dteday'])

# Mapping untuk season dan weathersit
season_mapping = {
    1: 'Spring', 
    2: 'Summer',
    3: 'Fall', 
    4: 'Winter'
}

weather_mapping = {
    1: 'Clear/Partly Cloudy',
    2: 'Mist/Cloudy',
    3: 'Light Rain/Snow',
    4: 'Heavy Rain/Snow'
}

all_df['season_name'] = all_df['season'].map(season_mapping)
all_df['weather_name'] = all_df['weathersit'].map(weather_mapping)

st.title('Analysis of the Effect of Weather on the Number of Bicycles Rent')

# 1. Analisis berdasarkan Musim
st.header('1. Season Effect')

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Box plot musim
sns.boxplot(data=all_df, x='season_name', y='cnt', ax=ax1)
ax1.set_title('Distribution of Rent Amount per Season')
ax1.set_xlabel('Season')
ax1.set_ylabel('Rent Amount')
ax1.tick_params(axis='x', rotation=45)

st.pyplot(fig1)

# ANOVA test untuk musim
season_groups = [group['cnt'].values for name, group in all_df.groupby('season')]
f_stat, p_val = stats.f_oneway(*season_groups)
st.write(f"ANOVA Test for Season: F-statistic = {f_stat:.2f}, p-value = {p_val:.10f}")

# 2. Analisis berdasarkan Kondisi Cuaca
st.header('2. Impact of Weather Conditions (Weather Situation)')

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))

# Box plot kondisi cuaca
sns.boxplot(data=all_df, x='weather_name', y='cnt', ax=ax3)
ax3.set_title('Distribution of Rent Amounts per Weather Condition')
ax3.set_xlabel('Weather Condition')
ax3.set_ylabel('Rent Amount')
ax3.tick_params(axis='x', rotation=45)

# Bar plot rata-rata peminjaman per kondisi cuaca
weather_avg = all_df.groupby('weather_name')['cnt'].mean().sort_values(ascending=False)
weather_avg.plot(kind='bar', ax=ax4)
ax4.set_title('Rata-rata Peminjaman per Kondisi Cuaca')
ax4.set_xlabel('Kondisi Cuaca')
ax4.set_ylabel('Rata-rata Peminjaman')
ax4.tick_params(axis='x', rotation=45)

st.pyplot(fig2)

# 3. Analisis Korelasi dengan Faktor Cuaca Numerik
st.header('3. Correlation with Weather Factors')

# Heatmap korelasi
weather_factors = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
correlation_matrix = all_df[weather_factors].corr()

fig3 = plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap: Weather Factors vs Rentals')
st.pyplot(fig3)

# 4. Analisis Detail per Faktor Numerik
st.header('4. Detailed Analysis per Factor')

fig4, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

# Temperature
sns.scatterplot(data=all_df, x='temp', y='cnt', ax=axes[0])
axes[0].set_title('Temperature vs Rentals')
axes[0].set_xlabel('Normalized Temperature')
axes[0].set_ylabel('Number of Rentals')

# Feels-like Temperature
sns.scatterplot(data=all_df, x='atemp', y='cnt', ax=axes[1])
axes[1].set_title('Feels-like Temperature vs Rentals')
axes[1].set_xlabel('Normalized Feels-like Temperature')
axes[1].set_ylabel('Number of Rentals')

# Humidity
sns.scatterplot(data=all_df, x='hum', y='cnt', ax=axes[2])
axes[2].set_title('Humidity vs Rentals')
axes[2].set_xlabel('Normalized Humidity')
axes[2].set_ylabel('Number of Rentals')

# Wind Speed
sns.scatterplot(data=all_df, x='windspeed', y='cnt', ax=axes[3])
axes[3].set_title('Wind Speed vs Rentals')
axes[3].set_xlabel('Normalized Wind Speed')
axes[3].set_ylabel('Number of Rentals')

st.pyplot(fig4)

# 5. Ringkasan Statistik
st.header('5. Summary')

# Korelasi dengan cnt
correlations = all_df[weather_factors].corr()['cnt'].sort_values(ascending=False)
st.write("Correlation between count and rent bike:")
st.write(correlations)

# Statistik deskriptif per musim
st.write("\nStatistical description per season:")
st.write(all_df.groupby('season_name')['cnt'].describe())

# Statistik deskriptif per kondisi cuaca
st.write("\nDescriptive statistics per weather condition:")
st.write(all_df.groupby('weather_name')['cnt'].describe())

# 6. Kesimpulan
st.header('6. Conclution')


st.write("""
Based on the results of the analysis, the following results was:

1. **Seasonal Effect:**
   - Fall and Summer have the highest loan amounts
   - Winter and Spring have the lowest loan amounts
   - The difference between each season was significant at statistik (p-value < 0.05)

2. **Weather Conditions Effect:**
   - Partly Cloudy have the highest loan amounts
   - Heavy Rain/Snow Winter and Spring have the lowest loan amounts
   - There is a significant drop in renting bike when the weather deteriorates

3. **Korelasi Faktor Cuaca:**
   - Temperature (temp) dan feels-like temperature (atemp) has a strong positive correlation with the number of loans
   - Humidity (hum) has a weak negative correlation with the number of loans
   - Wind speed (windspeed) has a weak negative correlation with the number of loans

4. **Recomendation:**
   - Increase bicycle availability in Fall and Summer seasons
   - Adjust the number of bicycles available based on the weather forecast
   - Pay special attention to bike maintenance during extreme weather seasons
""")

# 1. Data Loading and Preprocessing
main_df = all_df[
    (all_df["dteday"].dt.date >= start_date) &
    (all_df["dteday"].dt.date <= end_date)
]

# Generate daily aggregation
daily_df = main_df.resample('D', on='dteday').agg({
    'cnt': 'sum'
}).reset_index()
data = daily_df.set_index('dteday')['cnt']

st.header('Time Series Analysis ﾟ ⋆ ﾟ ☂︎  ﾟ ⋆ ﾟ ⋆ ')

# 2. Stationarity Analysis Functions
def check_stationarity(data, title=""):
    """
    Perform ADF test and create stationarity plots
    """
    # Perform ADF test
    result = adfuller(data)
    
    st.subheader("Augmented Dickey-Fuller Test Results")
    
    # Print ADF test results
    adf_output = pd.Series({
        'Test Statistic': result[0],
        'p-value': result[1],
        '1% Critical Value': result[4]['1%'],
        '5% Critical Value': result[4]['5%'],
        '10% Critical Value': result[4]['10%']
    })
    
    st.write(adf_output)
    
    # Interpret results
    if result[1] < 0.05:
        st.success("Data is stationary (reject H0)")
    else:
        st.warning("Data is non-stationary (fail to reject H0)")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Time series plot
    ax1.plot(data)
    ax1.set_title(f'Time Series Plot: {title}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Rentals')
    ax1.grid(True)
    
    # Rolling statistics
    rolling_mean = data.rolling(window=7).mean()
    rolling_std = data.rolling(window=7).std()
    
    ax2.plot(data, label='Original')
    ax2.plot(rolling_mean, label='Rolling Mean')
    ax2.plot(rolling_std, label='Rolling Std')
    ax2.set_title('Rolling Statistics')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    return result[1] < 0.05

def perform_decomposition(data):
    """
    Perform seasonal decomposition
    """
    decomposition = seasonal_decompose(data, period=7)
    
    # Plot decomposition
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
    
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    ax1.grid(True)
    
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    ax2.grid(True)
    
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    ax3.grid(True)
    
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    ax4.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    return decomposition

# 3. Perform Stationarity Analysis
st.subheader("Original Data Analysis")
is_stationary = check_stationarity(data, "Original Data")

# If data is not stationary, perform differencing
if not is_stationary:
    st.subheader("First Difference Analysis")
    diff_data = data.diff().dropna()
    is_stationary_diff = check_stationarity(diff_data, "First Difference")
    
    # If still not stationary, try seasonal differencing
    if not is_stationary_diff:
        st.subheader("Seasonal Difference Analysis")
        seasonal_diff = diff_data.diff(7).dropna()  # 7 for weekly seasonality
        is_stationary_seasonal = check_stationarity(seasonal_diff, "Seasonal Difference")

# 4. Perform Decomposition
st.subheader("Seasonal Decomposition")
decomp = perform_decomposition(data)

# 5. SARIMA Model Functions
def fit_sarima_model(data):
    """
    Fit SARIMA model with parameters based on stationarity analysis
    """
    # Adjust these parameters based on the ADF test results
    if is_stationary:
        order = (1, 0, 1)
        seasonal_order = (1, 0, 1, 7)
    elif is_stationary_diff:
        order = (1, 1, 1)
        seasonal_order = (1, 0, 1, 7)
    else:
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 7)
    
    model = SARIMAX(data,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    return model.fit()

def evaluate_model(actual, predicted):
    """
    Calculate evaluation metrics
    """
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }

# 6. Fit Model and Make Predictions
st.subheader("Fitting SARIMA Model")
with st.spinner('Fitting SARIMA Model...'):
    model_results = fit_sarima_model(data)
    predictions = model_results.get_prediction(start=data.index[0])
    predicted_mean = predictions.predicted_mean
    pred_conf = predictions.conf_int()

# 7. Display Results
# Model summary
with st.expander("View Model Summary"):
    st.text(str(model_results.summary()))

# Calculate and display metrics
metrics = evaluate_model(data, predicted_mean)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("MAE", f"{metrics['MAE']:.2f}")
with col2:
    st.metric("MSE", f"{metrics['MSE']:.2f}")
with col3:
    st.metric("RMSE", f"{metrics['RMSE']:.2f}")

# 8. Plot Results
# st.subheader('Bike Sharing Demand Analysis with SARIMA')
fig, ax = plt.subplots(figsize=(16, 8))

# Plot actual values
ax.plot(data.index, data, label='Actual', color='blue')

# Plot predicted values
ax.plot(predicted_mean.index, predicted_mean, label='Predicted', color='red', linestyle='--')

# Plot confidence intervals
ax.fill_between(pred_conf.index,
                pred_conf.iloc[:, 0],
                pred_conf.iloc[:, 1],
                color='red',
                alpha=0.1,
                label='95% Confidence Interval')

# Customize plot
ax.set_title("Bike Sharing Demand: Actual vs Predicted (SARIMA)", fontsize=20)
ax.set_xlabel("Date", fontsize=15)
ax.set_ylabel("Number of Rentals", fontsize=15)
ax.legend(fontsize=12)
ax.grid(True)
plt.xticks(rotation=45)

# Display plot
st.pyplot(fig)

data = pd.DataFrame(all_df)  # Replace with your data loading method
column = 'cnt'

# Function for differencing analysis
def perform_differencing_analysis(data, column='cnt', max_diff=2):
    fig, axes = plt.subplots(max_diff + 1, 2, figsize=(15, 4*(max_diff+1)))
    
    # Original series analysis
    series = data[column].copy()
    result = adfuller(series)
    
    # Plot original series
    axes[0, 0].plot(series)
    axes[0, 0].set_title(f'Original Series (ADF p-value: {result[1]:.4f})')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    
    # Plot ACF of original series
    pd.plotting.autocorrelation_plot(series, ax=axes[0, 1])
    axes[0, 1].set_title('ACF of Original Series')
    
    # Perform differencing
    for d in range(1, max_diff + 1):
        diff_series = series.diff(d).dropna()
        result = adfuller(diff_series)
        
        # Plot differenced series
        axes[d, 0].plot(diff_series)
        axes[d, 0].set_title(f'{d}-Order Difference (ADF p-value: {result[1]:.4f})')
        axes[d, 0].set_xlabel('Time')
        axes[d, 0].set_ylabel('Value')
        
        # Plot ACF of differenced series
        pd.plotting.autocorrelation_plot(diff_series, ax=axes[d, 1])
        axes[d, 1].set_title(f'ACF of {d}-Order Difference')
    
    plt.tight_layout()
    return fig

st.write (""" Dikarenakan tingkat akurasi time series masih kurang akurat dimana Mean Absolute Error ada diangka 638.94 dan Root Mean Squared Error sebesar 918.56
          maka perlu dilakukan diferencing analisis, adf test, dan fitting ke model Sarima yang baru.
          """)

# Streamlit App
st.title("Time Series Analysis")

# Input options
max_diff = st.sidebar.slider("Select maximum differencing order", 1, 5, 2)
forecast_steps = st.sidebar.slider("Forecast Steps", 1, 24, 12)

# Perform differencing analysis
st.subheader("Differencing Analysis")
fig_diff = perform_differencing_analysis(data, column=column, max_diff=max_diff)
st.pyplot(fig_diff)

# Stationarity Tests (ADF)
st.subheader("ADF Stationarity Test Results")
for d in range(3):
    if d == 0:
        series = data[column]
    else:
        series = data[column].diff(d).dropna()
    
    result = adfuller(series)
    st.write(f"\n{d}-Order Difference" if d > 0 else "Original Series:")
    st.write(f"ADF Statistic: {result[0]:.4f}")
    st.write(f"p-value: {result[1]:.4f}")
    st.write("Critical values:")
    for key, value in result[4].items():
        st.write(f"\t{key}: {value:.4f}")

# SARIMA Model Fitting
st.subheader("SARIMA Model Fitting")
order = (1, 0, 1)
seasonal_order = (1, 0, 1, 12)

model = SARIMAX(data[column], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
sarima_fit = model.fit(disp=False)

# Display Model Summary
st.write(sarima_fit.summary())

# Diagnostic Plots
st.subheader("SARIMA Diagnostic Plots")
fig_diag = sarima_fit.plot_diagnostics(figsize=(15, 8))
st.pyplot(fig_diag)

# Forecasting
st.subheader("Forecast Results")
forecast = sarima_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Evaluation Metrics
st.subheader("Evaluation Metrics")
# Select data to compare forecasted values
actual = data[column][-forecast_steps:]  # Adjust for available data if shorter
if len(actual) >= forecast_steps:
    actual = actual[-forecast_steps:]  # Adjust to match forecast steps if necessary

# Calculate MAE, MSE, RMSE
mae = mean_absolute_error(actual, forecast_mean)
mse = mean_squared_error(actual, forecast_mean)
rmse = np.sqrt(mse)

# Display Metrics
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

# Plot actual vs predicted
fig_forecast = plt.figure(figsize=(15, 6))
plt.plot(data.index, data[column], label='Actual', color='blue')
plt.plot(forecast_mean.index, forecast_mean, label='Predicted', color='red')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Actual vs Predicted Bike Rentals')
plt.xlabel('Date')
plt.ylabel('Number of Rentals')
plt.legend()
st.pyplot(fig_forecast)

actual = all_df['cnt'][-forecast_steps:]
predicted = forecast_mean

st.write("**Nilai Prediksi:**")
for idx, value in zip(predicted.index, predicted):
    st.write(f"Date: {idx}, Predicted: {value:.2f}")





def create_descriptive_stats(df):
    holiday_stats = df.groupby('holiday')[['casual', 'registered', 'cnt']].describe()
    workingday_stats = df.groupby('workingday')[['casual', 'registered', 'cnt']].describe()
    return holiday_stats, workingday_stats

def create_boxplots(df):
    fig = plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 3, 1)
    sns.boxplot(data=df, x='holiday', y='casual')
    plt.title("Casual Users on Holiday vs Non-Holiday")
    
    plt.subplot(1, 3, 2)
    sns.boxplot(data=df, x='holiday', y='registered')
    plt.title("Registered Users on Holiday vs Non-Holiday")
    
    plt.subplot(1, 3, 3)
    sns.boxplot(data=df, x='holiday', y='cnt')
    plt.title("Total Rentals on Holiday vs Non-Holiday")
    
    plt.tight_layout()
    return fig

def perform_statistical_tests(df):
    results = []
    
    # Holiday tests
    for column in ['casual', 'registered', 'cnt']:
        holiday_data = df[df['holiday'] == 1][column]
        non_holiday_data = df[df['holiday'] == 0][column]
        
        stat, p_val = ttest_ind(holiday_data, non_holiday_data)
        results.append({
            'test_type': 'Holiday',
            'variable': column,
            'p_value': p_val,
            'significant': p_val < 0.05
        })
    
    # Workingday tests
    for column in ['casual', 'registered', 'cnt']:
        workingday_data = df[df['workingday'] == 1][column]
        non_workingday_data = df[df['workingday'] == 0][column]
        
        stat, p_val = ttest_ind(workingday_data, non_workingday_data)
        results.append({
            'test_type': 'Workingday',
            'variable': column,
            'p_value': p_val,
            'significant': p_val < 0.05
        })
    
    # Weekday tests
    for column in ['casual', 'registered', 'cnt']:
        groups = [df[df['weekday'] == i][column] for i in range(7)]
        stat, p_val = f_oneway(*groups)
        results.append({
            'test_type': 'Weekday',
            'variable': column,
            'p_value': p_val,
            'significant': p_val < 0.05
        })
    
    return pd.DataFrame(results)

def run_regression(df):
    X = df[['holiday', 'workingday', 'weekday']]
    X = sm.add_constant(X)
    nb_model = sm.GLM(df['cnt'], X, family=sm.families.NegativeBinomial())
    return nb_model.fit()

def main():
    st.title('Bike Rental Analysis with Negative Binomial Regression')
    
    try:
        # Menggunakan all_df yang sudah ada
        df = all_df.copy()
        
        # Memastikan format tanggal
        df['dteday'] = pd.to_datetime(df['dteday'])
        df.sort_values(by="dteday", inplace=True)
        df = df.reset_index(drop=True)
        
        # st.success('Data ready for analysis!')
        
        # Show data preview
        st.subheader('Data Preview')
        st.dataframe(df.head())
        
        # Sidebar
        st.sidebar.header('Analysis Options')
        show_descriptive = st.sidebar.checkbox('Show Descriptive Statistics', True)
        show_plots = st.sidebar.checkbox('Show Box Plots', True)
        show_tests = st.sidebar.checkbox('Show Statistical Tests', True)
        show_regression = st.sidebar.checkbox('Show Regression Results', True)
        
        # Main content
        if show_descriptive:
            st.header('Descriptive Statistics')
            holiday_stats, workingday_stats = create_descriptive_stats(df)
            
            st.subheader('Holiday Statistics')
            st.dataframe(holiday_stats)
            
            st.subheader('Working Day Statistics')
            st.dataframe(workingday_stats)
        
        if show_plots:
            st.header('Box Plots')
            fig = create_boxplots(df)
            st.pyplot(fig)
        
        if show_tests:
            st.header('Statistical Tests Results')
            test_results = perform_statistical_tests(df)
            st.dataframe(test_results)
        
        if show_regression:
            st.header('Negative Binomial Regression Results')
            regression_results = run_regression(df)
            st.text(regression_results.summary().as_text())
            st.write("""
                     Based on the results of the Negative Binomial Regression analysis, the following results was:
                     1. Coefficients Interpretation:
                        - const (Intercept): 5.1794
                          This is the base log count of rentals when all predictors are zero. Exponentiating this gives the baseline rental count.
                        - holiday: -0.1430
                          A negative coefficient, indicating that bike rentals tend to decrease on holidays. Specifically, the coefficient suggests that rentals on holidays are about 𝑒 − 0.1430 ≈ 0.87 e−0.1430≈0.87 times the rentals on non-holidays, or about a 13% decrease.
                        - weekday: 0.0113
                          A small positive effect, implying a slight increase in rentals per additional weekday. Each day closer to the weekend shows a minor increase in rentals, though the effect is modest.
                     2. Statistical Significance:
                        - All predictors (holiday, workingday, and weekday) are statistically significant (p-values < 0.05), indicating that they contribute meaningfully to explaining the variation in bike rentals.
                     3. Model Fit:
                        - Pseudo R-squared (CS) of 0.001983 suggests a small proportion of variance explained by this model, indicating that other variables not included in this model likely have a substantial impact on bike rentals.
                     4. Conclution:
                        - Holidays are associated with a decrease in bike rentals.
                        - Working days are associated with a slight increase in rentals, likely because more people use bikes for commuting.
                        - Weekdays contribute a very minor but positive increase in rentals.
                     """)
            
    except Exception as e:
        st.error(f'Error processing data: {str(e)}')

if __name__ == '__main__':
    main()
    

    

# Footer
st.write('Built by Salman Fadhilurrohman © 2024 Dicoding Submission')

