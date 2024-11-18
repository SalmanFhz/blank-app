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
from statsmodels.tsa.statespace.sarimax import SARIMAX


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

# def load_and_preprocess(df, start_date, end_date):
#     """Load and preprocess the data"""
#     main_df = df[
#         (df["dteday"].dt.date >= start_date) &
#         (df["dteday"].dt.date <= end_date)
#     ]
    
#     daily_df = main_df.resample('D', on='dteday').agg({
#         'cnt': 'sum'
#     }).reset_index()
#     return daily_df.set_index('dteday')['cnt']

# def calculate_moving_averages(data):
#     """Calculate different moving averages"""
#     ma7 = data.rolling(window=7, center=True).mean()
#     ma30 = data.rolling(window=30, center=True).mean()
#     return ma7, ma30

# def perform_decomposition(data):
#     """Perform seasonal decomposition"""
#     decomposition = seasonal_decompose(data, period=7, extrapolate_trend='freq')
    
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
    
#     decomposition.observed.plot(ax=ax1)
#     ax1.set_title('Observed')
#     ax1.grid(True)
    
#     decomposition.trend.plot(ax=ax2)
#     ax2.set_title('Trend')
#     ax2.grid(True)
    
#     decomposition.seasonal.plot(ax=ax3)
#     ax3.set_title('Seasonal')
#     ax3.grid(True)
    
#     decomposition.resid.plot(ax=ax4)
#     ax4.set_title('Residual')
#     ax4.grid(True)
    
#     plt.tight_layout()
#     return fig, decomposition

# def calculate_confidence_interval(data, forecast, days_ahead):
#     """
#     Calculate more realistic confidence intervals based on historical error
#     """
#     # Calculate historical prediction error based on moving average
#     ma7 = data.rolling(window=7).mean()
#     historical_errors = (data - ma7).dropna()
    
#     # Error typically increases with forecast horizon
#     error_growth_factor = np.sqrt(days_ahead)
    
#     # Calculate error margins based on historical errors
#     error_margin = historical_errors.std() * error_growth_factor
    
#     # Scale down the margin for more realistic bounds
#     scaling_factor = 0.5  # Adjust this value to control interval width
#     error_margin = error_margin * scaling_factor
    
#     lower_bound = forecast - error_margin
#     upper_bound = forecast + error_margin
    
#     # Ensure lower bound isn't negative
#     lower_bound = max(0, lower_bound)
    
#     return lower_bound, upper_bound

# def make_simple_forecast(data, forecast_steps):
#     """Make simple forecast using historical averages and trends"""
#     # Calculate average daily change over the last 30 days
#     last_30_days = data[-30:]
#     daily_changes = last_30_days.diff()
#     avg_daily_change = daily_changes.mean()
    
#     # Calculate weekly pattern
#     weekly_pattern = data.groupby(data.index.dayofweek).mean()
    
#     # Get the last value
#     last_value = data.iloc[-1]
    
#     # Create forecast dates
#     last_date = data.index[-1]
#     forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
#                                  periods=forecast_steps, 
#                                  freq='D')
    
#     # Generate forecasts
#     forecasts = []
#     current_value = last_value
    
#     for date in forecast_dates:
#         # Add trend
#         current_value += avg_daily_change
        
#         # Add weekly pattern adjustment
#         day_of_week = date.dayofweek
#         weekly_adjustment = weekly_pattern[day_of_week] - weekly_pattern.mean()
        
#         # Calculate forecast
#         forecast_value = current_value + weekly_adjustment
        
#         # Ensure no negative values
#         forecast_value = max(0, forecast_value)
        
#         forecasts.append(forecast_value)
    
#     return pd.Series(forecasts, index=forecast_dates)

# def plot_results(data, ma7, ma30, forecast, confidence_intervals):
#     """Plot the results"""
#     fig, ax = plt.subplots(figsize=(15, 8))
    
#     # Plot actual values and moving averages
#     ax.plot(data.index, data, label='Actual', alpha=0.5)
#     ax.plot(ma7.index, ma7, label='7-day MA', linewidth=2)
#     ax.plot(ma30.index, ma30, label='30-day MA', linewidth=2)
    
#     # Plot forecast and confidence intervals
#     if forecast is not None:
#         ax.plot(forecast.index, forecast, label='Forecast', 
#                 color='red', linestyle='--')
        
#         # Plot confidence intervals
#         lower_bounds, upper_bounds = zip(*confidence_intervals)
#         ax.fill_between(forecast.index, lower_bounds, upper_bounds, 
#                        color='red', alpha=0.1, label='95% Confidence Interval')
    
#     ax.set_title('Bike Sharing Demand Analysis', fontsize=16)
#     ax.set_xlabel('Date', fontsize=12)
#     ax.set_ylabel('Number of Rentals', fontsize=12)
#     ax.legend()
#     ax.grid(True)
    
#     return fig

# # Main Streamlit app
# st.title('Simple Time Series Analysis 🚲')

# # Load data
# data = load_and_preprocess(all_df, start_date, end_date)

# # Calculate moving averages
# ma7, ma30 = calculate_moving_averages(data)

# # Display basic statistics
# st.subheader('Basic Statistics')
# col1, col2, col3, col4 = st.columns(4)
# with col1:
#     st.metric("Average Daily Rentals", f"{data.mean():.0f}")
# with col2:
#     st.metric("Maximum Rentals", f"{data.max():.0f}")
# with col3:
#     st.metric("Minimum Rentals", f"{data.min():.0f}")
# with col4:
#     st.metric("Standard Deviation", f"{data.std():.0f}")

# # Weekly pattern analysis
# st.subheader('Weekly Pattern Analysis')
# weekly_avg = data.groupby(data.index.dayofweek).mean()
# weekly_std = data.groupby(data.index.dayofweek).std()
# days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# fig_weekly = plt.figure(figsize=(12, 6))
# plt.bar(days, weekly_avg, yerr=weekly_std, capsize=5)
# plt.title('Average Daily Rentals by Day of Week')
# plt.ylabel('Number of Rentals')
# plt.xticks(rotation=45)
# st.pyplot(fig_weekly)

# # Perform decomposition
# st.subheader('Seasonal Decomposition')
# decomp_fig, decomposition = perform_decomposition(data)
# st.pyplot(decomp_fig)

# # Make forecast
# forecast_steps = st.slider('Forecast Days', 1, 30, 7)
# forecast = make_simple_forecast(data, forecast_steps)

# # Calculate confidence intervals for all forecast points
# confidence_intervals = [
#     calculate_confidence_interval(data, value, i+1) 
#     for i, value in enumerate(forecast)
# ]

# # Plot results
# st.subheader('Time Series Analysis')
# results_fig = plot_results(data, ma7, ma30, forecast, confidence_intervals)
# st.pyplot(results_fig)

# # Display forecast values
# st.subheader('Forecast Values')
# for i, (date, value) in enumerate(forecast.items(), 1):
#     lower_bound, upper_bound = confidence_intervals[i-1]
#     interval_width = upper_bound - lower_bound
    
#     st.write(f"""
#     **{date.strftime('%A, %B %d, %Y')}**
#     - Predicted Rentals: {value:.0f}
#     - Likely Range: {lower_bound:.0f} to {upper_bound:.0f} rentals
#     - Interval Width: ±{(interval_width/2):.0f} rentals from predicted value
#     """)

# # Display additional insights
# st.subheader('Additional Insights')
# best_day = days[weekly_avg.argmax()]
# worst_day = days[weekly_avg.argmin()]
# best_value = weekly_avg.max()
# worst_value = weekly_avg.min()

# st.write(f"""
# - Days with the highest average rentals: {best_day} ({best_value:.0f} rentals)
# - Days with the lowest average rentals: {worst_day} ({worst_value:.0f} rentals)
# - Variasi harian: {(weekly_avg.max() - weekly_avg.min()):.0f} rentals
# """)

# # Calculate and display trend
# recent_trend = data[-30:].mean() - data[-60:-30].mean()
# st.write(f"""
# Trends of the last 30 days: {'Down' if recent_trend > 0 else 'Increase'} in the amount of {abs(recent_trend):.0f} rentals
# """)




@st.cache_data
def load_data():
    """Memuat data dari file CSV dan mengembalikannya sebagai DataFrame."""
    try:
        df = pd.read_csv("hour.csv")
        if 'cnt' not in data.columns:
            raise ValueError("Kolom 'cnt' tidak ditemukan di dataset.")
        return df
    except Exception as e:
        raise ValueError(f"Gagal memuat data: {str(e)}")


def create_descriptive_stats(df):
    holiday_stats = df.groupby('holiday')[['casual', 'registered', 'cnt', 'temp', 'atemp', 'hum']].describe()
    workingday_stats = df.groupby('workingday')[['casual', 'registered', 'cnt', 'temp', 'atemp', 'hum']].describe()
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
    for column in ['casual', 'registered', 'cnt', 'temp', 'atemp', 'hum']:
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
    for column in ['casual', 'registered', 'cnt', 'temp', 'atemp', 'hum']:
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
    for column in ['casual', 'registered', 'cnt', 'temp', 'atemp', 'hum']:
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
    X = df[['holiday', 'workingday', 'weekday', 'temp', 'atemp', 'hum']]
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
                     Faktor yang meningkatkan peminjaman sepeda (koefisien positif):
                     1. atemp (feels-like temperature) memiliki pengaruh paling besar (+2.2185)
                        Saat orang merasa suhu nyaman, peminjaman sepeda meningkat signifikan
                        Ini masuk akal karena orang cenderung bersepeda saat cuaca terasa nyaman
                     2. workingday (+0.1149) Ada peningkatan peminjaman pada hari kerja Menunjukkan sepeda banyak digunakan untuk komuting/transportasi ke tempat kerja
                     3. weekday (+0.0079) Ada sedikit peningkatan seiring hari dalam minggu Efeknya kecil tapi masih signifikan
                     Faktor yang menurunkan peminjaman sepeda (koefisien negatif):
                     1. hum (humidity) memiliki efek negatif besar (-1.5127) Semakin lembab udara, semakin sedikit peminjaman Orang cenderung menghindari bersepeda saat kelembaban tinggi
                     2. holiday (-0.1279) Peminjaman menurun saat hari libur Mendukung temuan bahwa sepeda lebih banyak digunakan untuk komuting
                     """)
            
    except Exception as e:
        st.error(f'Error processing data: {str(e)}')

if __name__ == '__main__':
    main()


#Forecasting
# Fungsi untuk memuat data
@st.cache_data
def load_data():
    """Memuat data dari file CSV dan mengembalikannya sebagai DataFrame."""
    try:
        data = pd.read_csv("hour.csv")
        if 'cnt' not in data.columns:
            raise ValueError("Kolom 'cnt' tidak ditemukan di dataset.")
        return data
    except Exception as e:
        raise ValueError(f"Gagal memuat data: {str(e)}")

# Fungsi untuk membuat plot
def create_plot(actual_values, predicted_values):
    """Membuat plot perbandingan data aktual dan prediksi."""
    plt.figure(figsize=(12, 6))
    hours = range(24)
    
    plt.plot(hours, actual_values, 'b-o', label='Aktual (24 jam terakhir)', linewidth=2)
    plt.plot(hours, predicted_values, 'r--o', label='Prediksi (24 jam ke depan)', linewidth=2)
    
    plt.title('Perbandingan Data Aktual vs Prediksi (24 Jam)')
    plt.xlabel('Jam ke-')
    plt.ylabel('Jumlah Peminjaman Sepeda')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(hours)
    plt.tight_layout()
    
    return plt

# Fungsi utama
def main():
    st.title("Prediksi Jumlah Peminjaman Sepeda dengan SARIMA")
    
    # Load data
    try:
        data = load_data()
        st.success("Data berhasil dimuat!")
    except Exception as e:
        st.error(f"Error saat memuat data: {str(e)}")
        return
    
    # Tampilkan data
    st.subheader("Preview Data")
    st.dataframe(data.tail(10))  # Hanya tampilkan 10 data terakhir
    
    # Parameter SARIMA default
    p, d, q = 1, 1, 1  # ARIMA parameters
    P, D, Q, s = 0, 1, 1, 24  # Seasonal ARIMA parameters
    
    # Lakukan prediksi secara otomatis
    with st.spinner("Sedang melakukan prediksi..."):
        try:
            # Siapkan data time series
            ts_data = data['cnt'].values[-240:]  # Hanya gunakan 240 data terakhir
            
            # Fit model SARIMA
            model = SARIMAX(ts_data,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, s),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            
            results = model.fit(disp=False)
            
            # Ambil data aktual 24 jam terakhir
            actual_values = ts_data[-24:]
            
            # Prediksi 24 jam ke depan
            forecast = results.get_forecast(steps=24)
            predicted_values = forecast.predicted_mean
            
            # Hitung metrik evaluasi
            mae = np.mean(np.abs(actual_values - predicted_values))
            mse = np.mean((actual_values - predicted_values) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
            
            # Tampilkan metrik
            st.subheader("Metrik Evaluasi")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE", f"{mae:.2f}")
                st.metric("MSE", f"{mse:.2f}")
            with col2:
                st.metric("RMSE", f"{rmse:.2f}")
                st.metric("MAPE", f"{mape:.2f}%")
            
            # Tampilkan plot
            st.subheader("Visualisasi Prediksi")
            fig = create_plot(actual_values, predicted_values)
            st.pyplot(fig)
            
            # Download hasil prediksi
            predictions_df = pd.DataFrame({
                'Jam': range(24),
                'Aktual': actual_values,
                'Prediksi': predicted_values
            })
            
            st.download_button(
                label="Download Hasil Prediksi (CSV)",
                data=predictions_df.to_csv(index=False).encode('utf-8'),
                file_name='hasil_prediksi_sarima.csv',
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"Error saat melakukan prediksi: {str(e)}")
        finally:
            plt.close()
        st.write(""" 
            Insight:
            1. Alokasi Sepeda: Pastikan stok sepeda maksimal pada jam 7-9 dan 15-17 Bisa mengurangi jumlah sepeda yang tersedia pada jam 2-4
            2. Maintenance: Jadwalkan pemeliharaan pada periode volume rendah (dini hari)
            3. Staff: Tingkatkan staff support pada peak hours Kurangi staff pada periode volume rendah
        """)

if __name__ == "__main__":
    main()
    
 #anomali detection   
def detect_anomalies_iqr(df, column='cnt', multiplier=1.5):
    """Detect anomalies using the IQR method, with additional check for low-activity anomalies below the mean."""
    Q1 = df[column].quantile(0.25)  # First quartile (25th percentile)
    Q3 = df[column].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range (IQR)
    mean_value = df[column].mean()  # Mean of the column

    # Define the bounds for detecting anomalies
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    # Copy the dataframe and create a column to indicate anomalies
    results = df.copy()
    results['is_anomaly'] = False
    results.loc[results[column] > upper_bound, 'is_anomaly'] = True
    results.loc[(results[column] < lower_bound) & (results[column] < mean_value), 'is_anomaly'] = True

    # Label the type of anomaly (high or low)
    results['anomaly_type'] = 'normal'
    results.loc[results[column] > upper_bound, 'anomaly_type'] = 'high_activity'
    results.loc[(results[column] < lower_bound) & (results[column] < mean_value), 'anomaly_type'] = 'low_activity'

    # Add bounds columns for reference
    results['lower_bound'] = lower_bound
    results['upper_bound'] = upper_bound

    return results

##baruuuu
def create_anomaly_visualizations(all_anomalies):
    st.title('Visualisasi Data Anomali Peminjaman')
    
    # Create two columns for the visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Scatter Plot Anomali')
        
        # Create scatter plot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(all_anomalies['hr'], all_anomalies['cnt'],
                   s=100, c='red', alpha=0.6, label='Anomali')
        ax1.set_xlabel('Jam')
        ax1.set_ylabel('Jumlah Peminjaman')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend()
        
        # Display the plot
        st.pyplot(fig1)
        
    with col2:
        st.subheader('Box Plot Distribusi per Jam')
        
        # Create box plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=all_anomalies, x='hr', y='cnt', ax=ax2)
        ax2.set_title('Distribusi Jumlah Peminjaman Anomali per Jam')
        ax2.set_xlabel('Jam')
        ax2.set_ylabel('Jumlah Peminjaman')
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # Display the plot
        st.pyplot(fig2)
    
    # Add some statistics
    st.subheader('Statistik Anomali')
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric('Total Anomali', len(all_anomalies))
    with col4:
        st.metric('Rata-rata Peminjaman', f"{all_anomalies['cnt'].mean():.2f}")
    with col5:
        st.metric('Maksimum Peminjaman', all_anomalies['cnt'].max())
    
    # Add interactive data table
    st.subheader('Data Anomali')
    st.dataframe(all_anomalies)

# Example usage
if __name__ == "__main__":
    # Add file uploader
    st.sidebar.title('Upload Data')
    uploaded_file = st.sidebar.file_uploader("Upload file CSV anomali", type=['csv'])
    
    if uploaded_file is not None:
        # Read the uploaded file
        all_anomalies = pd.read_csv(uploaded_file)
        create_anomaly_visualizations(all_anomalies)
    else:
        st.info('Silakan upload file CSV yang berisi data anomali')


def main():
    # Title
    st.title("🚲 Bike Rental Anomaly Detection")

    # Sidebar controls
    st.sidebar.header("Settings")
    multiplier = st.sidebar.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1,
                                 help="Adjust sensitivity of anomaly detection. Lower values detect more anomalies.")

    # Load the real data
    # Replace 'bike_rentals.csv' with the path to your actual dataset
    all_df = pd.read_csv('hour.csv')
    
    # Detect anomalies
    results = detect_anomalies_iqr(all_df, multiplier=multiplier)
    
    # Display summary metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data", len(results))
    with col2:
        st.metric("Total Anomalies", results['is_anomaly'].sum())
    with col3:
        st.metric("High Activity Anomalies", (results['anomaly_type'] == 'high_activity').sum())
    with col4:
        st.metric("Low Activity Anomalies", (results['anomaly_type'] == 'low_activity').sum())
    
    # Create visualization
    st.subheader("Rental Pattern and Anomalies")
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot normal points
    normal_data = results[~results['is_anomaly']]
    ax.scatter(normal_data.index, normal_data['cnt'], color='blue', label='Normal', alpha=0.5, s=20)
    
    # Plot high anomalies
    high_anomalies = results[results['anomaly_type'] == 'high_activity']
    ax.scatter(high_anomalies.index, high_anomalies['cnt'], color='red', label='High Activity Anomaly', marker='*', s=100)
    
    # Plot low anomalies
    low_anomalies = results[results['anomaly_type'] == 'low_activity']
    ax.scatter(low_anomalies.index, low_anomalies['cnt'], color='orange', label='Low Activity Anomaly', marker='*', s=100)
    
    # Plot bounds
    ax.axhline(y=results['upper_bound'].iloc[0], color='red', linestyle='--', label='Upper Bound')
    ax.axhline(y=results['lower_bound'].iloc[0], color='orange', linestyle='--', label='Lower Bound')
    
    plt.title('Bike Rentals Over Time with Anomalies')
    plt.xlabel('Data Point')
    plt.ylabel('Number of Rentals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Display plot
    st.pyplot(fig)
    st.write("""pada jam 8, 11-19 sering terjadi anomali jumlah peminjaman dimana anomali tersebut termasuk pada anomali high atau lonjakan pengunjung, dapat diupayakan bahwa sepeda ready pada jam tersebut sehingga dapat mengatasi lonjakan signifikan pada peminjaman sepeda.""")
    
    # Display anomaly details in tabs
    tab1, tab2 = st.tabs(["High Activity Anomalies", "Low Activity Anomalies"])
    
    with tab1:
        st.dataframe(
            high_anomalies[['dteday', 'hr', 'cnt', 'weathersit', 'temp', 'hum']]
            .sort_values('cnt', ascending=False)
        )
    
    with tab2:
        st.dataframe(
            low_anomalies[['dteday', 'hr', 'cnt', 'weathersit', 'temp', 'hum']]
            .sort_values('cnt', ascending=True)
        )
        

if __name__ == "__main__":
    main()
    


# Footer
st.write('Built by Salman Fadhilurrohman © 2024 Dicoding Submission')

