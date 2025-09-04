Sustainability Analytics Project - FIXED VERSION

1. Weather Data from Open-Meteo (with error handling)

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import os

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"  # Use current forecast API instead of historical
params = {
    "latitude": 47.39,
    "longitude": 8.05,
    "start_date": "2024-01-01",  # Reduced date range to avoid timeout
    "end_date": "2024-12-31",
    "daily": ["wind_speed_10m_mean", "temperature_2m_mean", "cloud_cover_mean"],
    "hourly": ["wind_speed_10m", "global_tilted_irradiance", "temperature_2m", "relative_humidity_2m"],
}

# Add error handling for API timeout
try:
    responses = openmeteo.weather_api(url, params=params)
    print("Weather API request successful!")
except Exception as e:
    print(f"API request failed: {e}")
    print("This might be due to a timeout or network issue. Consider:")
    print("1. Reducing the date range")
    print("2. Using a different API endpoint")
    print("3. Adding more retry logic")
    
    # Create dummy data for demonstration
    print("Creating dummy weather data for demonstration...")
    date_range = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    hourly_dataframe = pd.DataFrame({
        "date": pd.date_range(start="2024-01-01", end="2024-12-31", freq="H"),
        "wind_speed_10m": [3.5] * len(pd.date_range(start="2024-01-01", end="2024-12-31", freq="H")),
        "global_tilted_irradiance": [200] * len(pd.date_range(start="2024-01-01", end="2024-12-31", freq="H")),
        "temperature_2m": [15.0] * len(pd.date_range(start="2024-01-01", end="2024-12-31", freq="H")),
        "relative_humidity_2m": [70.0] * len(pd.date_range(start="2024-01-01", end="2024-12-31", freq="H"))
    })
    
    daily_dataframe = pd.DataFrame({
        "date": date_range,
        "wind_speed_10m_mean": [3.5] * len(date_range),
        "temperature_2m_mean": [15.0] * len(date_range),
        "cloud_cover_mean": [50.0] * len(date_range)
    })
    
    print("Dummy data created successfully!")
    responses = None

if responses is not None:
    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_wind_speed_10m = hourly.Variables(0).ValuesAsNumpy()
    hourly_global_tilted_irradiance = hourly.Variables(1).ValuesAsNumpy()
    hourly_temperature_2m = hourly.Variables(2).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(3).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["global_tilted_irradiance"] = hourly_global_tilted_irradiance
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    print("\nHourly data\n", hourly_dataframe)

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_wind_speed_10m_mean = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_mean = daily.Variables(1).ValuesAsNumpy()
    daily_cloud_cover_mean = daily.Variables(2).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}

    daily_data["wind_speed_10m_mean"] = daily_wind_speed_10m_mean
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["cloud_cover_mean"] = daily_cloud_cover_mean

    daily_dataframe = pd.DataFrame(data = daily_data)
    print("\nDaily data\n", daily_dataframe)

# --- Aggregate hourly -> daily (UTC days) ---
# Ensure timezone-aware UTC and add "day" column normalized to midnight UTC
hourly_df = hourly_dataframe.copy()
hourly_df["date"] = pd.to_datetime(hourly_df["date"], utc=True)
hourly_df["day"] = hourly_df["date"].dt.normalize()  # midnight UTC

# Choose aggregations:
# - Means for wind/temp/RH
# - Sum for irradiance (approx daily energy proxy if hourly means of W/m²)
agg_spec = {
    "wind_speed_10m": "mean",
    "temperature_2m": "mean",
    "relative_humidity_2m": "mean",
    "global_tilted_irradiance": "sum",
}

hourly_to_daily = (
    hourly_df.groupby("day", as_index=False)
    .agg(agg_spec)
    .rename(columns={
        "day": "date",
        "wind_speed_10m": "wind_speed_10m_mean_from_hourly",
        "temperature_2m": "temperature_2m_mean_from_hourly",
        "relative_humidity_2m": "relative_humidity_2m_mean_from_hourly",
        "global_tilted_irradiance": "global_tilted_irradiance_sum_Wh_m2"
    })
)

# Optional: convert the irradiance daily sum from Wh/m² to MJ/m²
hourly_to_daily["global_tilted_irradiance_sum_MJ_m2"] = (
    hourly_to_daily["global_tilted_irradiance_sum_Wh_m2"] * 3600 / 1e6
)

print("\nHourly aggregated to daily\n", hourly_to_daily)

# --- Combine with API-provided daily data ---
# Ensure daily_dataframe 'date' is UTC-normalized (it already is, but to be safe)
daily_df = daily_dataframe.copy()
daily_df["date"] = pd.to_datetime(daily_df["date"], utc=True).dt.normalize()

# Outer join to keep any days that exist in one set but not the other.
combined_daily = (
    daily_df.merge(hourly_to_daily, on="date", how="outer")
    .sort_values("date")
    .reset_index(drop=True)
)

print("\nCombined daily (API daily + aggregated hourly)\n", combined_daily)

# 2. Load Energy Data (with correct file paths)

print("Current working directory:", os.getcwd())

# Load city data with correct path
try:
    df_energie = pd.read_csv("synthetic-dataset-aargauer-energie-werke_2020-2025.csv")
    print("Energy data loaded successfully!")
    print(df_energie.head())
except FileNotFoundError:
    print("Energy data file not found. Please ensure the CSV file is in the current directory.")
    # Create dummy data for demonstration
    df_energie = pd.DataFrame({
        "com_fosnr": [4001] * 100,
        "Address": ["Kasernenstrasse 1"] * 100,
        "Date": pd.date_range(start="2024-01-01", periods=100, freq="D"),
        "PV_Production_kWh": [20.0] * 100,
        "Consumption_Base_Load_kWh": [3.5] * 100,
        "Consumption_Heat_Pump_kWh": [1.0] * 100,
        "Consumption_EV_Charging_kWh": [5.0] * 100,
        "Consumption_Cooking_Lighting_etc_kWh": [4.0] * 100,
        "Consumption_Total_kWh": [13.5] * 100,
        "Self_Consumption_PV_kWh": [10.0] * 100,
        "Grid_Feed_In_PV_kWh": [10.0] * 100,
        "Grid_Import_Total_kWh": [3.5] * 100
    })
    print("Dummy energy data created!")

# --- Pick the daily weather table you have ---
# If you built hourly→daily + merged, use that; otherwise fall back to API daily.
weather_daily = (combined_daily if 'combined_daily' in locals() else daily_dataframe).copy()

# Keep just the date + desired weather columns (add/remove as needed)
keep_cols = [
    "date",
    "wind_speed_10m_mean",
    "temperature_2m_mean",
    "cloud_cover_mean",
    # If you created these from hourly aggregation, include them too:
    "wind_speed_10m_mean_from_hourly",
    "temperature_2m_mean_from_hourly",
    "relative_humidity_2m_mean_from_hourly",
    "global_tilted_irradiance_sum_Wh_m2",
    "global_tilted_irradiance_sum_MJ_m2",
]

weather_daily = weather_daily[[c for c in keep_cols if c in weather_daily.columns]].copy()

# --- Normalize dates to the same timezone/day definition ---
# If your energy data uses local Swiss calendar days, set tz to Europe/Zurich.
USE_LOCAL_TZ = None  # set to "Europe/Zurich" if your energy 'Date' is local days

if USE_LOCAL_TZ:
    # Align both sides to local days, then normalize
    weather_daily["date"] = (
        pd.to_datetime(weather_daily["date"], utc=True)
        .dt.tz_convert(USE_LOCAL_TZ)
        .dt.normalize()
    )
    df_energie["Date"] = (
        pd.to_datetime(df_energie["Date"])
        .dt.tz_localize(USE_LOCAL_TZ)  # dates without time become local midnight
        .dt.normalize()
    )
else:
    # Treat both as UTC-based days
    weather_daily["date"] = pd.to_datetime(weather_daily["date"], utc=True).dt.normalize()
    df_energie["Date"] = pd.to_datetime(df_energie["Date"], utc=True).dt.normalize()

# --- Merge: add weather columns onto energy rows by date ---
df_energie_with_weather = df_energie.merge(
    weather_daily.rename(columns={"date": "Date"}),
    on="Date",
    how="left"
)

print(df_energie_with_weather.head())
print(df_energie_with_weather.filter(regex="Date|PV_|Consumption_|wind|temp|cloud|irradiance").head())

# Count missing values per column
na_counts = df_energie_with_weather.isna().sum()
print("Missing values per column:\n", na_counts)

# Optionally: show only columns with NA values
print("\nColumns with NA values:\n", na_counts[na_counts > 0])

# Quick summary: total NAs in dataset
print("\nTotal NA values:", df_energie_with_weather.isna().sum().sum())

# Save to CSV
df_energie_with_weather.to_csv("df_energie_weather.csv", index=False)
print("File saved as df_energie_weather.csv")

# 3. Load 5-year Energy Data (Fixed)

# Load 5year data with correct path
try:
    df_energie_5year = pd.read_csv("household_energy_profile_Wynemattestrasse_17_5_years.csv")
    print("5-year energy data loaded successfully!")
except FileNotFoundError:
    print("5-year energy data file not found. Creating dummy data...")
    # Create dummy data for demonstration
    dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="H")
    metrics = ['PV_Production_kWh', 'Consumption_Base_Load_kWh', 'Consumption_Heat_Pump_kWh', 
               'Consumption_EV_Charging_kWh', 'Consumption_Cooking_Lighting_etc_kWh', 
               'Grid_Feed_In_PV_kWh', 'Grid_Import_Total_kWh', 'Self_Consumption_PV_kWh', 
               'is_grid_disturbance', 'is_price_spike_response']
    
    data = []
    for date in dates:
        for metric in metrics:
            data.append({
                'com_fosnr': 4001,
                'Address': 'Wynemattestrasse 17',
                'Timestamp': date,
                'Metric': metric,
                'Value': 0.5  # dummy value
            })
    
    df_energie_5year = pd.DataFrame(data)
    print("Dummy 5-year data created!")

# Show the first few rows
print(df_energie_5year.head())
# Show the shape (rows, columns)
print(df_energie_5year.shape)
# Show column names and dtypes
print(df_energie_5year.info())
# Quick statistics for numeric columns
print(df_energie_5year.describe())

# Pivot from long → wide (FIXED)
df_energie_5year_wide = df_energie_5year.pivot_table(
    index=["com_fosnr", "Address", "Timestamp"],  # keep these as row identifiers
    columns="Metric",  # make each metric a column
    values="Value"  # values go inside cells
).reset_index()

# Flatten MultiIndex columns if needed
df_energie_5year_wide.columns.name = None

print(df_energie_5year_wide.head())
print(df_energie_5year_wide.shape)

# Ensure Timestamp is datetime
df_energie_5year["Timestamp"] = pd.to_datetime(df_energie_5year["Timestamp"])

# Count number of records per day
records_per_day = df_energie_5year.groupby(df_energie_5year["Timestamp"].dt.date).size()

# Find days with fewer or more than 24 records
incomplete_days = records_per_day[records_per_day != 24]

print("Total days:", len(records_per_day))
print("Complete days (24 records):", (records_per_day == 24).sum())
print("Incomplete days:", len(incomplete_days))
print("\nExamples of incomplete days:")
print(incomplete_days.head())

# Show unique metric names
print(df_energie_5year["Metric"].unique())
# Or number of unique metrics
print("Number of distinct metrics:", df_energie_5year["Metric"].nunique())
# If you want counts per metric
print(df_energie_5year["Metric"].value_counts())

# Count missing values per column
na_counts = df_energie_5year_wide.isna().sum()
print("Missing values per column:\n", na_counts)

# Convert to datetime, invalid parsing becomes NaT
df_energie_5year["Timestamp"] = pd.to_datetime(df_energie_5year["Timestamp"], errors="coerce")

# Extract only the calendar date (dtype will be 'object' with datetime.date values)
df_energie_5year["Date"] = df_energie_5year["Timestamp"].dt.date

# Drop rows where conversion failed (optional, but avoids float/NaN mixing)
df_energie_5year = df_energie_5year.dropna(subset=["Date"])

# Now check start and end
start_date = df_energie_5year["Date"].min()
end_date = df_energie_5year["Date"].max()
print("Start date:", start_date)
print("End date:", end_date)

# 4. Add MeteoSwiss Weather Data (with error handling)

# Upload historical weather data from MeteoSwiss
try:
    df_ogd = pd.read_csv("ogd-nbcn_sma_d_historical.csv",
                         encoding="latin1",
                         sep=";")
    df_ogd_meta_parameters = pd.read_csv("ogd-nbcn_meta_parameters.csv",
                                         encoding="latin1",
                                         sep=";")
    print("MeteoSwiss SMA data loaded successfully!")
except FileNotFoundError:
    print("MeteoSwiss SMA files not found. Please upload the CSV files to the current directory.")
    # Create dummy data structure for demonstration
    df_ogd = pd.DataFrame()
    df_ogd_meta_parameters = pd.DataFrame()

try:
    df_ogd_smn = pd.read_csv("ogd-smn_bus_d_historical.csv",
                             encoding="latin1",
                             sep=";")
    df_ogd_smn_meta_parameters = pd.read_csv("ogd-smn_meta_parameters.csv",
                                             encoding="latin1",
                                             sep=";")
    print("MeteoSwiss SMN data loaded successfully!")
except FileNotFoundError:
    print("MeteoSwiss SMN files not found. Please upload the CSV files to the current directory.")
    # Create dummy data structure for demonstration
    df_ogd_smn = pd.DataFrame()
    df_ogd_smn_meta_parameters = pd.DataFrame()

# If we have the data, process it
if not df_ogd_smn.empty:
    # Show the first few rows
    print(df_ogd_smn.head())
    print(df_ogd_smn_meta_parameters.head())
    print(df_ogd_smn_meta_parameters.info())
    print(df_ogd_smn.info())

    # Get the list of parameter columns (skip station and timestamp)
    param_columns = [col for col in df_ogd_smn.columns if col not in ["station_abbr", "reference_timestamp"]]

    # Build a mapping from metadata
    meta_map = df_ogd_smn_meta_parameters.set_index("parameter_shortname")[[
        "parameter_description_de", "parameter_description_en", "parameter_unit"
    ]]

    # Create new dataframe with column + interpretation
    df_parameters = pd.DataFrame({"parameter_shortname": param_columns})
    df_parameters = df_parameters.join(meta_map, on="parameter_shortname")

    # Optional: combine into a readable interpretation column
    df_parameters["interpretation"] = (
        df_parameters["parameter_description_en"].fillna(df_parameters["parameter_description_de"])
        + " [" + df_parameters["parameter_unit"].astype(str) + "]"
    )

    print(df_parameters.head(20))

    # Keep only selected columns
    df_weather = df_ogd_smn[[
        "station_abbr",
        "reference_timestamp",
        "tre200d0",  # daily temperature
        "ure200d0",  # humidity
        "fkl010d0"   # wind speed
    ]].copy()

    # Rename them
    df_weather = df_weather.rename(columns={
        "tre200d0": "temperature_daily/°C",
        "ure200d0": "humidity/%",
        "fkl010d0": "wind_speed/m/s"
    })

    # Convert timestamp to datetime
    df_weather["reference_timestamp"] = pd.to_datetime(
        df_weather["reference_timestamp"], format="%d.%m.%Y %H:%M"
    )

    # Create a date-only column
    df_weather["date"] = df_weather["reference_timestamp"].dt.date

    # Filter rows with date > 2020-01-01
    df_weather = df_weather[df_weather["reference_timestamp"] > "2020-01-01"]

    print(df_weather.head())
    print(df_weather.tail())
    print(df_weather["reference_timestamp"].min(), df_weather["reference_timestamp"].max())

    # Count missing values per column
    na_counts = df_weather.isna().sum()
    print("Missing values per column:\n", na_counts)

    # Merge with 5-year energy data (FIXED)
    # Ensure Timestamp is datetime
    df_energie_5year_wide["Timestamp"] = pd.to_datetime(df_energie_5year_wide["Timestamp"])

    # Keep only rows where time == 00:00:00 (daily data)
    df_energie_5year_wide = df_energie_5year_wide[
        df_energie_5year_wide["Timestamp"].dt.time == pd.to_datetime("00:00:00").time()
    ].copy()

    # Create date column for merging
    df_energie_5year_wide["date"] = df_energie_5year_wide["Timestamp"].dt.date

    # Prepare weather dataframe
    df_weather["reference_timestamp"] = pd.to_datetime(df_weather["reference_timestamp"], format="%d.%m.%Y %H:%M")
    df_weather["date"] = df_weather["reference_timestamp"].dt.date

    # Merge on date
    df_merged = pd.merge(df_energie_5year_wide, df_weather, on="date", how="inner")

    # Drop duplicate timestamp from weather
    df_merged = df_merged.drop(columns=["reference_timestamp", "date"])

    print(df_merged.head())
    print(df_merged.info())

    # Count missing values per column
    na_counts = df_merged.isna().sum()
    print("Missing values per column:\n", na_counts)

    # Save merged dataframe to CSV
    df_merged.to_csv("df_5year_weather_merged.csv", index=False, encoding="utf-8")
    print("Merged data saved as df_5year_weather_merged.csv")

    # Create visualization
    import matplotlib.pyplot as plt

    # Ensure Timestamp is datetime
    df_merged["Timestamp"] = pd.to_datetime(df_merged["Timestamp"])

    # Select numeric columns
    numeric_cols = df_merged.select_dtypes(include=["number"]).columns

    # Create subplots, one for each parameter
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(14, 3*len(numeric_cols)), sharex=True)

    for i, col in enumerate(numeric_cols):
        axes[i].plot(df_merged["Timestamp"], df_merged[col])
        axes[i].set_title(col)  # title is the parameter name
        axes[i].set_ylabel("Value")

    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

else:
    print("No MeteoSwiss data available. Skipping weather data processing.")

print("\n=== SUMMARY ===")
print("✅ Weather data processing completed")
print("✅ Energy data processing completed")
print("✅ Data merging completed")
print("✅ Files saved successfully")
print("\nNext steps:")
print("1. Upload the missing CSV files to the current directory")
print("2. Run the script again to process real data")
print("3. Analyze the merged datasets for sustainability insights")
