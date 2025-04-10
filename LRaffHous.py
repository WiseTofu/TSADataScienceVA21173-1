import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Read the data
df = pd.read_csv('Affordable_Housing.csv')
df.columns = [col.upper() for col in df.columns]

# Create label encoder for ward values
ward_encoder = LabelEncoder()
df['WARD_ENCODED'] = ward_encoder.fit_transform(df['MAR_WARD'])

# Historical median income data (DC)
AMI_2022 = 142300  # Updated 2022 AMI
AMI_2023 = 152100  # Updated 2023 AMI
AMI_2024 = 154700  # Updated 2024 AMI

# Calculate growth rates
growth_rate_2023 = (AMI_2023 - AMI_2022) / AMI_2022
growth_rate_2024 = (AMI_2024 - AMI_2023) / AMI_2023

# AMI levels and corresponding columns
ami_columns = {
    '0-30%': ('AFFORDABLE_UNITS_AT_0_30_AMI', 0.3),
    '31-50%': ('AFFORDABLE_UNITS_AT_31_50_AMI', 0.5),
    '51-60%': ('AFFORDABLE_UNITS_AT_51_60_AMI', 0.6),
    '61-80%': ('AFFORDABLE_UNITS_AT_61_80_AMI', 0.8),
    '81%+': ('AFFORDABLE_UNITS_AT_81_AMI', 1.0)
}

# Prepare data for decision tree using individual projects
X = []  # Features: [income_level, ami_year, growth_rate, ward_encoded, total_units]
y = []  # Target: units available

# Collect data points from each project
for _, project in df.iterrows():
    for year, ami, growth_rate in [(2022, AMI_2022, 0), 
                                 (2023, AMI_2023, growth_rate_2023),
                                 (2024, AMI_2024, growth_rate_2024)]:
        for level, (column, percentage) in ami_columns.items():
            income = ami * percentage
            units = project[column] if not pd.isna(project[column]) else 0
            X.append([
                income,
                year,
                growth_rate,
                project['WARD_ENCODED'],  # Use encoded ward values
                project['TOTAL_AFFORDABLE_UNITS']
            ])
            y.append(units)

X = np.array(X)
y = np.array(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train decision tree
tree_model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5
)
tree_model.fit(X_scaled, y)

# Analysis output
print("\nAffordable Housing Availability Analysis")
print("=" * 50)
print(f"AMI Growth Rate (2023-2024): {growth_rate_2024:.1%}")
print("\nPredicted Housing Availability Trends by Income Level:")
print("-" * 50)

# Analyze each AMI level
for level, (column, percentage) in ami_columns.items():
    current_projects = df[df[column] > 0]
    current_units = current_projects[column].sum()
    
    # Make predictions for each existing project
    predictions = []
    for _, project in current_projects.iterrows():
        features = [
            AMI_2024 * percentage,
            2024,
            growth_rate_2024,
            project['WARD_ENCODED'],  # Use encoded ward values
            project['TOTAL_AFFORDABLE_UNITS']
        ]
        pred = tree_model.predict(scaler.transform([features]))[0]
        predictions.append(pred)
    
    total_prediction = sum(predictions)
    change = total_prediction - current_units
    trend = "🔼" if change > 0 else "🔽" if change < 0 else "➡️"
    
    print(f"\nIncome Level: {level}")
    print(f"Income Threshold: ${AMI_2024 * percentage:,.2f}")
    print(f"Current Units: {current_units:.0f}")
    print(f"Predicted Units: {total_prediction:.0f}")
    print(f"Trend: {trend}")
    if current_units > 0:
        print(f"Expected Change: {abs(change):.0f} units ({change/current_units:+.1%})")

# Model performance metrics
y_pred = tree_model.predict(X_scaled)
print("\nModel Performance:")
print("-" * 50)
print(f"R² Score: {r2_score(y, y_pred):.3f}")
print(f"Number of Projects Analyzed: {len(df)}")
print(f"Total Data Points: {len(X)}")