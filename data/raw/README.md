# Smart Water Consumption Predictor

## Dataset Documentation & Source Investigation

### 1. Investigation of Public Water Consumption Datasets
During the preliminary research phase of this project, several public water consumption datasets were evaluated for their feasibility in residential daily forecasting:

1. **City of Barcelona Open Data (Domestic Water Consumption by Household & Neighborhood)**
   - *Source*: Open Data BCN (Ajuntament de Barcelona)
   - *Licence*: Open Data Commons Open Database License (ODbL)
   - *Granularity*: Aggregated quarterly/annual volume ($m^3$) per postal district.
   - *Limitation*: Insufficient temporal resolution (quarterly instead of daily) for responsive next-day residential predictive modeling.

2. **Smart Home Datasets (e.g., Kaggle / Pecan Street Dataport)**
   - *Source*: Pecan Street Research Institute (Dataport) / Kaggle Smart Home Energy & Water
   - *Licence*: Non-commercial research license / restricted access requiring university credentials.
   - *Limitation*: Requires proprietary sign-ups and API keys; heavy focus on electrical energy meters with water flow channels sparsely sampled or intermittent.

3. **WHO & Bureau of Indian Standards (BIS IS 1172: 1993) Benchmarks**
   - *Standard*: Code of Basic Requirements for Water Supply, Drainage and Sanitation (IS 1172).
   - *Benchmark*: Recommended basic domestic water supply for communities with full flushing systems is **135 to 150 Litres per capita per day (LPCD)**.
   - *Standard End-Use Breakdown*:
     - Bathing & Personal Hygiene: ~35% (45–55 L/day)
     - Toilet Flushing: ~22% (30 L/day)
     - Washing of clothes (Laundry): ~15% (20 L/day)
     - Kitchen & Cooking: ~15% (20 L/day)
     - House cleaning, outdoor & other: ~13% (15–20 L/day)

### 2. Demonstration Dataset Generation Rationale
To provide a reliable, reproducible, and leak-free foundation for this 6-week internship project without dependency on proprietary paid portals or coarse quarterly municipal records, we implemented a dedicated data generator:
- **File**: `data/generate_dataset.py`
- **Output**: `data/processed/water_consumption_daily.csv`
- **Designation**: **Calibrated Synthetic Demonstration Dataset**
- **Timeframe**: 730 consecutive days (2024-01-01 to 2025-12-31)
- **Features Included**:
  - `date`: YYYY-MM-DD
  - `household_size`: Integer (e.g., 4 members)
  - `temperature_c`: Daily average ambient temperature (°C)
  - `rainfall_mm`: Daily precipitation (mm)
  - `is_weekend`: Binary flag (0 = Weekday, 1 = Weekend)
  - `season`: Categorical (Winter, Spring/Pre-Summer, Summer, Monsoon, Autumn)
  - `guests_present`: Integer (number of additional visitors)
  - `usage_kitchen_l`: Litres
  - `usage_bathroom_l`: Litres
  - `usage_laundry_l`: Litres
  - `usage_toilet_l`: Litres
  - `usage_outdoor_l`: Litres
  - `total_consumption_l`: Total daily water consumption in litres (Target variable)

### 3. Important Transparency Notice
*This dataset is generated for demonstration and educational purposes. It is calibrated against empirical hydrological and municipal engineering standards (BIS 1172 / WHO). It must not be cited as empirical sensor readings from an active municipal smart meter.*
