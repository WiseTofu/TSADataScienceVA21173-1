import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import datetime
import os
import sys

# Read the CSV file
df = pd.read_csv('Affordable_Housing.csv')

# Calculate total units by AMI level for each ward
def analyze_ward_ami():
    # Create summary by ward
    ward_summary = df.groupby('MAR_WARD').agg({
        'TOTAL_AFFORDABLE_UNITS': 'sum',
        'AFFORDABLE_UNITS_AT_0_30_AMI': 'sum', 
        'AFFORDABLE_UNITS_AT_31_50_AMI': 'sum',
        'AFFORDABLE_UNITS_AT_51_60_AMI': 'sum',
        'AFFORDABLE_UNITS_AT_61_80_AMI': 'sum',
        'AFFORDABLE_UNITS_AT_81_AMI': 'sum'
    }).round(2)
    
    # Calculate percentages
    ami_columns = ['AFFORDABLE_UNITS_AT_0_30_AMI', 'AFFORDABLE_UNITS_AT_31_50_AMI',
                   'AFFORDABLE_UNITS_AT_51_60_AMI', 'AFFORDABLE_UNITS_AT_61_80_AMI', 
                   'AFFORDABLE_UNITS_AT_81_AMI']
    
    for col in ami_columns:
        ward_summary[f'{col}_PCT'] = (ward_summary[col] / ward_summary['TOTAL_AFFORDABLE_UNITS'] * 100).round(1)
        
    return ward_summary

def plot_ami_distribution():
    ward_summary = analyze_ward_ami()
    
    # Create a stacked bar chart
    ami_pct_cols = [col for col in ward_summary.columns if col.endswith('_PCT')]
    ax = ward_summary[ami_pct_cols].plot(kind='bar', stacked=True, figsize=(12, 6))
    
    plt.title('Distribution of Affordable Units by AMI Level Across Wards')
    plt.xlabel('Ward')
    plt.ylabel('Percentage of Units')
    plt.legend(title='AMI Level', bbox_to_anchor=(1.05, 1), labels=[
        '0-30% AMI',
        '31-50% AMI',
        '51-60% AMI',
        '61-80% AMI',
        '81%+ AMI'
    ])
    plt.tight_layout()
    plt.show()

def plot_total_units():
    ward_summary = analyze_ward_ami()
    
    # Create bar chart of total units
    plt.figure(figsize=(10, 6))
    ward_summary['TOTAL_AFFORDABLE_UNITS'].plot(kind='bar')
    plt.title('Total Affordable Units by Ward')
    plt.xlabel('Ward')
    plt.ylabel('Number of Units')
    plt.tight_layout()
    plt.show()

def print_ward_statistics():
    ward_summary = analyze_ward_ami()
    
    print("\nAffordable Housing Analysis by Ward:")
    print("====================================")
    
    for ward in ward_summary.index:
        print(f"\n{ward}:")
        print(f"Total Units: {ward_summary.loc[ward, 'TOTAL_AFFORDABLE_UNITS']:.0f}")
        print("AMI Distribution:")
        print(f"  0-30%:  {ward_summary.loc[ward, 'AFFORDABLE_UNITS_AT_0_30_AMI_PCT']}%")
        print(f"  31-50%: {ward_summary.loc[ward, 'AFFORDABLE_UNITS_AT_31_50_AMI_PCT']}%")
        print(f"  51-60%: {ward_summary.loc[ward, 'AFFORDABLE_UNITS_AT_51_60_AMI_PCT']}%")
        print(f"  61-80%: {ward_summary.loc[ward, 'AFFORDABLE_UNITS_AT_61_80_AMI_PCT']}%")
        print(f"  81%+:   {ward_summary.loc[ward, 'AFFORDABLE_UNITS_AT_81_AMI_PCT']}%")

def analyze_segregation():
    # Prepare data for clustering
    features = ['AFFORDABLE_UNITS_AT_0_30_AMI', 'AFFORDABLE_UNITS_AT_31_50_AMI',
                'AFFORDABLE_UNITS_AT_51_60_AMI', 'AFFORDABLE_UNITS_AT_61_80_AMI',
                'AFFORDABLE_UNITS_AT_81_AMI', 'LATITUDE', 'LONGITUDE']
    
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze segregation patterns
    cluster_stats = df.groupby(['MAR_WARD', 'Cluster']).size().unstack(fill_value=0)
    segregation_index = cluster_stats.var(axis=1) / cluster_stats.mean(axis=1)
    
    print("\nSegregation Analysis:")
    print("====================")
    print("Segregation Index by Ward (higher values indicate more segregation):")
    print(segregation_index.round(2))

def calculate_affordability():
    # DC AMI for 2024 (approximate)
    dc_ami_2024 = 142300
    
    ami_levels = {
        '0-30': dc_ami_2024 * 0.30,
        '31-50': dc_ami_2024 * 0.50,
        '51-60': dc_ami_2024 * 0.60,
        '61-80': dc_ami_2024 * 0.80,
        '81+': dc_ami_2024 * 0.81
    }
    
    print("\nAffordability Analysis (2024):")
    print("==============================")
    print(f"DC Area Median Income (AMI): ${dc_ami_2024:,.2f}")
    print("\nRequired Annual Income by AMI Level:")
    for level, income in ami_levels.items():
        print(f"AMI {level}%: ${income:,.2f}")
        
    # Calculate average monthly housing cost (30% of income)
    print("\nMaximum Affordable Monthly Housing Cost (30% of income):")
    for level, income in ami_levels.items():
        monthly_housing = income * 0.3 / 12
        print(f"AMI {level}%: ${monthly_housing:,.2f}")

def predict_housing_trends():
    print("\nHousing Predictions Analysis:")
    print("===========================")
    
    # Prepare features for prediction
    features = ['LATITUDE', 'LONGITUDE', 
                'AFFORDABLE_UNITS_AT_0_30_AMI',
                'AFFORDABLE_UNITS_AT_31_50_AMI',
                'AFFORDABLE_UNITS_AT_51_60_AMI',
                'AFFORDABLE_UNITS_AT_61_80_AMI']
    
    X = df[features].fillna(0)
    y = df['TOTAL_AFFORDABLE_UNITS']
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Predict future trends by ward
    ward_predictions = {}
    for ward in df['MAR_WARD'].unique():
        ward_data = df[df['MAR_WARD'] == ward][features].mean().values.reshape(1, -1)
        predicted_units = model.predict(ward_data)[0]
        ward_predictions[ward] = predicted_units
    
    print("\nPredicted Housing Units by Ward (Next Year):")
    for ward, prediction in ward_predictions.items():
        print(f"{ward}: {prediction:.0f} units")
    
    # Feature importance analysis
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': abs(model.coef_)
    })
    importance = importance.sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    for _, row in importance.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.2f}")

def analyze_housing_patterns():
    # Clustering analysis for housing patterns
    features = ['LATITUDE', 'LONGITUDE', 'TOTAL_AFFORDABLE_UNITS']
    X = df[features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Housing_Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters
    cluster_stats = df.groupby('Housing_Cluster').agg({
        'TOTAL_AFFORDABLE_UNITS': 'mean',
        'MAR_WARD': lambda x: x.value_counts().index[0]
    }).round(2)
    
    print("\nHousing Pattern Analysis:")
    print("=======================")
    print("Cluster Statistics:")
    print(cluster_stats)

if __name__ == "__main__":    
        print("Starting affordable housing analysis...")
        # Generate all analyses
        plot_ami_distribution()
        plot_total_units()
        print_ward_statistics()
        analyze_segregation()
        calculate_affordability()
        predict_housing_trends()
        analyze_housing_patterns()
        print("\nAnalysis completed successfully!")