import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('Affordable_Housing.csv')
df.columns = [col.upper() for col in df.columns]

# Set style for better visualization
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Calculate AMI percentages and totals
ami_columns = [
    'AFFORDABLE_UNITS_AT_0_30_AMI',
    'AFFORDABLE_UNITS_AT_31_50_AMI',
    'AFFORDABLE_UNITS_AT_51_60_AMI',
    'AFFORDABLE_UNITS_AT_61_80_AMI',
    'AFFORDABLE_UNITS_AT_81_AMI'
]

# Calculate statistics
ami_by_ward = df.groupby('MAR_WARD')[ami_columns].sum()
ami_percentages = ami_by_ward.div(ami_by_ward.sum(axis=1), axis=0) * 100
ami_percentages.columns = [
    'Affordable @ 0-30 AMI',
    'Affordable @ 31-50 AMI', 
    'Affordable @ 51-60 AMI',
    'Affordable @ 61-80 AMI',
    'Affordable @ 81+ AMI'
]
total_units = df.groupby('MAR_WARD')['TOTAL_AFFORDABLE_UNITS'].sum()

# Print statistics
print("\nPercentage Distribution of Affordable Units by AMI Level in Each Ward:")
print("=" * 70)
print(ami_percentages.round(1).to_string())
print("\nTotal Number of Affordable Units in Each Ward:")
print("=" * 40)
print(total_units.to_string())
print("\n")

# Plot without percentage labels
ami_percentages.plot(kind='bar', stacked=True, ax=ax1)
ax1.set_title('Distribution of Affordable Units by AMI Level and Ward (%)')
ax1.set_xlabel('Ward')
ax1.set_ylabel('Percentage of Units')
ax1.legend(title='AMI Level', 
          labels=['0-30%', '31-50%', '51-60%', '61-80%', '81%+'],
          bbox_to_anchor=(1.05, 1))

# Second plot
affordability = df.groupby('MAR_WARD')[['TOTAL_AFFORDABLE_UNITS']].sum()
affordability.plot(kind='bar', ax=ax2)
ax2.set_title('Total Affordable Units by Ward')
ax2.set_xlabel('Ward')
ax2.set_ylabel('Number of Units')

plt.tight_layout()
plt.show()