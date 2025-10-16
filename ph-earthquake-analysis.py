import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# --- Load latest dataset directly into a DataFrame (KaggleHub) ---
import kagglehub
from kagglehub import KaggleDatasetAdapter

DATASET = "bwandowando/philippine-earthquakes-from-phivolcs"
FILE_PATH = "phivolcs_earthquake_data.csv"  # update if the filename on Kaggle ever changes

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    DATASET,
    FILE_PATH
)

# --- Standardize datetime and derive time parts ---
df['Date_Time_PH'] = pd.to_datetime(df['Date_Time_PH'], errors='coerce')
df = df.dropna(subset=['Date_Time_PH'])
df['Year']  = df['Date_Time_PH'].dt.year
df['Month'] = df['Date_Time_PH'].dt.month

# Month labels
MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# =========================
# A) PHILIPPINES OVERVIEW
# =========================

# 1) PH total earthquakes per year
ph_yearly = df.groupby('Year').size()

plt.figure(figsize=(14, 6))
ph_yearly.plot(kind='bar', color='#7db1ff', edgecolor='black')
plt.title('Philippines: Total Earthquakes per Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
plt.close()

# 2) PH total earthquakes per year-month (heatmap)
ph_monthly = df.groupby(['Year', 'Month']).size().reset_index(name='Count')
ph_monthly_pivot = ph_monthly.pivot(index='Year', columns='Month', values='Count').fillna(0)

plt.figure(figsize=(14, 6))
sns.heatmap(ph_monthly_pivot, cmap='YlGnBu', annot=True, fmt='.0f')
plt.title('Philippines: Earthquake Counts per Year-Month')
plt.ylabel('Year')
plt.xlabel('Month')
plt.tight_layout()
plt.show()
plt.close()

# 3) PH highest magnitude per year-month (heatmap + table)
#    Sort so that ties pick the earliest event among the max magnitudes
agg_max = (
    df.sort_values(['Year', 'Month', 'Magnitude', 'Date_Time_PH'],
                   ascending=[True, True, False, True])
      .groupby(['Year', 'Month'])
      .agg(
          MaxMagnitude=('Magnitude', 'max'),
          MaxMagDate=('Date_Time_PH', 'first'),
          MaxMagLocation=('General_Location', 'first')
      )
      .reset_index()
)

# 3a) Heatmap of max magnitude by year-month with month labels
ph_max_pivot = agg_max.pivot(index='Year', columns='Month', values='MaxMagnitude').fillna(0)
ph_max_pivot.columns = MONTH_LABELS[:len(ph_max_pivot.columns)]

plt.figure(figsize=(14, 6))
sns.heatmap(ph_max_pivot, cmap='OrRd', annot=True, fmt='.1f', linewidths=.5, linecolor='white')
plt.title('Philippines: Highest Magnitude per Year-Month')
plt.ylabel('Year')
plt.xlabel('Month')
plt.tight_layout()
plt.show()
plt.close()

# 3b) Table of Year, Month, MaxMagnitude, Location, Date
agg_max_view = agg_max.copy()
agg_max_view['MonthName'] = agg_max_view['Month'].map({i+1: m for i, m in enumerate(MONTH_LABELS)})
agg_max_view['MaxMagDate'] = agg_max_view['MaxMagDate'].dt.strftime('%Y-%m-%d %H:%M')

# Sort using numeric Month (kept internally), then select display columns
display_table = agg_max_view.sort_values(['Year','Month'])[
    ['Year','MonthName','MaxMagnitude','MaxMagLocation','MaxMagDate']
].copy()

# Wrap long location names for the printed table
def wrap_text(s, width=40):
    if pd.isna(s):
        return ''
    return "\n".join(textwrap.wrap(str(s), width=width))

display_table['MaxMagLocation'] = display_table['MaxMagLocation'].apply(lambda s: wrap_text(s, width=40))

fig, ax = plt.subplots(figsize=(16, max(6, len(display_table)*0.32)))
ax.axis('off')
tbl = ax.table(
    cellText=display_table.values,
    colLabels=display_table.columns,
    loc='center',
    cellLoc='left',
    colLoc='left'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.2)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#f0f3f7')
plt.title('Philippines: Highest Magnitude by Year-Month (Location and Date)', pad=12)
plt.tight_layout()
plt.show()
plt.close()

# =========================
# B) ILOCOS NORTE FOCUS
# =========================

# Filter to Ilocos Norte
ilocos_norte = df[
    df['General_Location'].str.contains('Ilocos Norte', case=False, na=False)
].copy()

# Monthly counts (Ilocos Norte heatmap)
plt.figure(figsize=(14, 6))
ilocos_yearly_count = ilocos_norte.groupby('Year').size()
ilocos_monthly_count = ilocos_norte.groupby(['Year', 'Month']).size().reset_index(name='Count')
ilocos_monthly_pivot = ilocos_monthly_count.pivot(index='Year', columns='Month', values='Count').fillna(0)
sns.heatmap(ilocos_monthly_pivot, cmap='YlOrRd', annot=True, fmt='.0f')
plt.title('Monthly Earthquake Counts in Ilocos Norte (Heatmap)')
plt.ylabel('Year')
plt.xlabel('Month')
plt.tight_layout()
plt.show()
plt.close()

# Yearly counts (Ilocos Norte bar)
plt.figure(figsize=(14, 6))
ilocos_yearly_count.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Number of Earthquakes per Year in Ilocos Norte')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
plt.close()

# Magnitude categorization
def mag_category(mag: float) -> str:
    if mag < 4.0:
        return 'Minor'
    elif mag < 6.0:
        return 'Moderate'
    else:
        return 'Significant'

ilocos_norte['MagCategory'] = ilocos_norte['Magnitude'].apply(mag_category)

# Magnitude histogram (Ilocos Norte)
plt.figure(figsize=(10, 6))
plt.hist(ilocos_norte['Magnitude'], bins=20, color='teal', edgecolor='black', alpha=0.7)
plt.title('Distribution of Earthquake Magnitudes in Ilocos Norte')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

# Avg magnitude vs event count trend (Ilocos Norte)
plt.figure(figsize=(12, 6))
mag_trend = ilocos_norte.groupby('Year')['Magnitude'].agg(['mean', 'count']).reset_index()
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(mag_trend['Year'], mag_trend['mean'], color='orange', marker='s', label='Average Magnitude')
ax2.plot(mag_trend['Year'], mag_trend['count'], color='blue', marker='o', alpha=0.5, label='Event Count')
ax1.set_ylabel('Average Magnitude', color='orange')
ax2.set_ylabel('Number of Events', color='blue')
ax1.set_xlabel('Year')
plt.title('Ilocos Norte: Trend in Average Magnitude and Event Count')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()
plt.close()

# Magnitude categories (Ilocos Norte)
plt.figure(figsize=(8, 5))
ilocos_norte['MagCategory'].value_counts().reindex(['Minor', 'Moderate', 'Significant']).plot(
    kind='bar', color=['green','orange','red'])
plt.title('Ilocos Norte: Earthquake Magnitude Categories')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
plt.close()

print("Generated national overview and Ilocos Norte visualizations.")

