import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from pathlib import Path

# External data loader
import kagglehub
from kagglehub import KaggleDatasetAdapter

DATASET = "bwandowando/philippine-earthquakes-from-phivolcs"
FILE_PATH = "phivolcs_earthquake_data.csv"  # Update if Kaggle file name changes
DOCS = Path("docs")
DOCS.mkdir(exist_ok=True)

# Month labels for plots/tables
MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    DATASET,
    FILE_PATH
)

df['Date_Time_PH'] = pd.to_datetime(df['Date_Time_PH'], errors='coerce')
df = df.dropna(subset=['Date_Time_PH'])
df['Year'] = df['Date_Time_PH'].dt.year
df['Month'] = df['Date_Time_PH'].dt.month

# Philippines overview

ph_yearly = df.groupby('Year').size()
plt.figure(figsize=(14, 6))
ph_yearly.plot(kind='bar', color='#7db1ff', edgecolor='black')
plt.title('Philippines: Total Earthquakes per Year')
plt.xlabel('Year'); plt.ylabel('Count')
plt.xticks(rotation=45); plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(DOCS / 'ph-yearly-earthquake-counts.png', dpi=150, bbox_inches='tight')
plt.close()

ph_monthly = df.groupby(['Year', 'Month']).size().reset_index(name='Count')
ph_monthly_pivot = ph_monthly.pivot(index='Year', columns='Month', values='Count').fillna(0)
plt.figure(figsize=(14, 6))
sns.heatmap(ph_monthly_pivot, cmap='YlGnBu', annot=True, fmt='.0f')
plt.title('Philippines: Earthquake Counts per Year-Month')
plt.ylabel('Year'); plt.xlabel('Month')
plt.tight_layout()
plt.savefig(DOCS / 'ph-yearmonth-earthquake-counts-heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

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

ph_max_pivot = agg_max.pivot(index='Year', columns='Month', values='MaxMagnitude').fillna(0)
ph_max_pivot.columns = MONTH_LABELS[:len(ph_max_pivot.columns)]
plt.figure(figsize=(14, 6))
sns.heatmap(ph_max_pivot, cmap='OrRd', annot=True, fmt='.1f', linewidths=.5, linecolor='white')
plt.title('Philippines: Highest Magnitude per Year-Month')
plt.ylabel('Year'); plt.xlabel('Month')
plt.tight_layout()
plt.savefig(DOCS / 'ph-yearmonth-max-magnitude-heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

agg_max_view = agg_max.copy()
agg_max_view['MonthName'] = agg_max_view['Month'].map({i+1: m for i, m in enumerate(MONTH_LABELS)})
agg_max_view['MaxMagDate'] = agg_max_view['MaxMagDate'].dt.strftime('%Y-%m-%d %H:%M')
display_table = agg_max_view.sort_values(['Year','Month'])[
    ['Year','MonthName','MaxMagnitude','MaxMagLocation','MaxMagDate']
].copy()

def wrap_text(s, width=60):
    if pd.isna(s): return ''
    return "<br>".join(textwrap.wrap(str(s), width=width))

display_table_html = display_table.copy()
display_table_html['MaxMagLocation'] = display_table_html['MaxMagLocation'].apply(lambda s: wrap_text(s, width=60))
html_table = display_table_html.to_html(index=False, escape=False)
(DOCS / 'ph-yearmonth-max-magnitude-table.html').write_text(
    '<meta charset="utf-8">'
    '<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:1080px;margin:2rem auto;} '
    'table{border-collapse:collapse;width:100%;} th,td{border:1px solid #e5e7eb;padding:8px;vertical-align:top;} '
    'th{background:#f0f3f7;text-align:left;}</style>' + html_table,
    encoding='utf-8'
)

plt.figure(figsize=(10, 6))
plt.hist(df['Magnitude'].dropna(), bins=30, color='#0ea5e9', edgecolor='black', alpha=0.8)
plt.title('Philippines: Distribution of Earthquake Magnitudes')
plt.xlabel('Magnitude'); plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.3); plt.tight_layout()
plt.savefig(DOCS / 'ph-magnitude-distribution.png', dpi=150, bbox_inches='tight')
plt.close()

def mag_category(m: float) -> str:
    if m < 4.0: return 'Minor'
    elif m < 6.0: return 'Moderate'
    else: return 'Significant'
df['MagCategory_PH'] = df['Magnitude'].apply(mag_category)

plt.figure(figsize=(8, 5))
df['MagCategory_PH'].value_counts().reindex(['Minor','Moderate','Significant']).plot(
    kind='bar', color=['#22c55e','#f59e0b','#ef4444'])
plt.title('Philippines: Magnitude Categories')
plt.ylabel('Count'); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(DOCS / 'ph-magnitude-categories.png', dpi=150, bbox_inches='tight')
plt.close()

# Ilocos Norte focus

ilocos_norte = df[df['General_Location'].str.contains('Ilocos Norte', case=False, na=False)].copy()

plt.figure(figsize=(14, 6))
ilocos_yearly_count = ilocos_norte.groupby('Year').size()
ilocos_monthly_count = ilocos_norte.groupby(['Year', 'Month']).size().reset_index(name='Count')
ilocos_monthly_pivot = ilocos_monthly_count.pivot(index='Year', columns='Month', values='Count').fillna(0)
sns.heatmap(ilocos_monthly_pivot, cmap='YlOrRd', annot=True, fmt='.0f')
plt.title('Monthly Earthquake Counts in Ilocos Norte (Heatmap)')
plt.ylabel('Year'); plt.xlabel('Month'); plt.tight_layout()
plt.savefig(DOCS / 'ilocos-monthly-earthquake-counts.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(14, 6))
ilocos_yearly_count.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Number of Earthquakes per Year in Ilocos Norte')
plt.xlabel('Year'); plt.ylabel('Count')
plt.xticks(rotation=45); plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(DOCS / 'ilocos-yearly-earthquake-counts.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(ilocos_norte['Magnitude'].dropna(), bins=20, color='teal', edgecolor='black', alpha=0.7)
plt.title('Distribution of Earthquake Magnitudes in Ilocos Norte')
plt.xlabel('Magnitude'); plt.ylabel('Frequency'); plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(DOCS / 'ilocos-magnitude-distribution.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
trend = ilocos_norte.groupby('Year')['Magnitude'].agg(['mean','count']).reset_index()
ax1 = plt.gca(); ax2 = ax1.twinx()
ax1.plot(trend['Year'], trend['mean'], color='orange', marker='s', label='Average Magnitude')
ax2.plot(trend['Year'], trend['count'], color='blue', marker='o', alpha=0.5, label='Event Count')
ax1.set_ylabel('Average Magnitude', color='orange'); ax2.set_ylabel('Number of Events', color='blue')
ax1.set_xlabel('Year'); plt.title('Ilocos Norte: Trend in Average Magnitude and Event Count')
ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig(DOCS / 'ilocos-magnitude-eventcount-trend.png', dpi=150, bbox_inches='tight')
plt.close()

def mag_category_local(m: float) -> str:
    if m < 4.0: return 'Minor'
    elif m < 6.0: return 'Moderate'
    else: return 'Significant'
ilocos_norte['MagCategory'] = ilocos_norte['Magnitude'].apply(mag_category_local)

plt.figure(figsize=(8, 5))
ilocos_norte['MagCategory'].value_counts().reindex(['Minor','Moderate','Significant']).plot(
    kind='bar', color=['green','orange','red'])
plt.title('Ilocos Norte: Earthquake Magnitude Categories')
plt.ylabel('Count'); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(DOCS / 'ilocos-magnitude-categories.png', dpi=150, bbox_inches='tight')
plt.close()
