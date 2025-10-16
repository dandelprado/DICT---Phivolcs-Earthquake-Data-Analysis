import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub
from kagglehub import KaggleDatasetAdapter

DATASET = "bwandowando/philippine-earthquakes-from-phivolcs"
FILE_PATH = "phivolcs_earthquake_data.csv"

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    DATASET,
    FILE_PATH
)

df['Date_Time_PH'] = pd.to_datetime(df['Date_Time_PH'], errors='coerce')
df = df.dropna(subset=['Date_Time_PH'])

ilocos_norte = df[
    df['General_Location'].str.contains('Ilocos Norte', case=False, na=False)
].copy()

ilocos_norte['Year'] = ilocos_norte['Date_Time_PH'].dt.year
ilocos_norte['Month'] = ilocos_norte['Date_Time_PH'].dt.month

plt.figure(figsize=(14, 6))
yearly_count = ilocos_norte.groupby('Year').size()
monthly_count = ilocos_norte.groupby(['Year', 'Month']).size().reset_index(name='Count')
monthly_pivot = monthly_count.pivot(index='Year', columns='Month', values='Count').fillna(0)
sns.heatmap(monthly_pivot, cmap='YlOrRd', annot=True, fmt='.0f')
plt.title('Monthly Earthquake Counts in Ilocos Norte (Heatmap)')
plt.ylabel('Year')
plt.xlabel('Month')
plt.tight_layout()
# plt.savefig('monthly-earthquake-counts-ilocos-norte.png')
plt.show()
plt.close()

plt.figure(figsize=(14, 6))
yearly_count.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Number of Earthquakes per Year in Ilocos Norte')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.savefig('yearly-earthquake-counts-ilocos-norte.png')
plt.show()
plt.close()


def mag_category(mag):
    if mag < 4.0:
        return 'Minor'
    elif mag < 6.0:
        return 'Moderate'
    else:
        return 'Significant'


ilocos_norte['MagCategory'] = ilocos_norte['Magnitude'].apply(mag_category)

plt.figure(figsize=(10, 6))
plt.hist(ilocos_norte['Magnitude'], bins=20, color='teal', edgecolor='black', alpha=0.7)
plt.title('Distribution of Earthquake Magnitudes in Ilocos Norte')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
# plt.savefig('magnitude-distribution-ilocos-norte.png')
plt.show()
plt.close()

plt.figure(figsize=(12, 6))
mag_trend = ilocos_norte.groupby('Year')['Magnitude'].agg(['mean', 'count']).reset_index()
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(mag_trend['Year'], mag_trend['mean'], color='orange', marker='s', label='Average Magnitude')
ax2.plot(mag_trend['Year'], mag_trend['count'], color='blue', marker='o', alpha=0.5, label='Event Count')
ax1.set_ylabel('Average Magnitude', color='orange')
ax2.set_ylabel('Number of Events', color='blue')
ax1.set_xlabel('Year')
plt.title('Trend in Average Magnitude and Event Count Over Time')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
# plt.savefig('magnitude-eventcount-trend-ilocos-norte.png')
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
ilocos_norte['MagCategory'].value_counts().reindex(['Minor', 'Moderate', 'Significant']).plot(
    kind='bar', color=['green', 'orange', 'red'])
plt.title('Earthquake Magnitude Categories in Ilocos Norte')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
# plt.savefig('magnitude-category-counts-ilocos-norte.png')
plt.show()
plt.close()
