import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import kagglehub
from kagglehub import KaggleDatasetAdapter

DATASET = "bwandowando/philippine-earthquakes-from-phivolcs"
FILE_PATH = "phivolcs_earthquake_data.csv"
DOCS = Path("docs")
DOCS.mkdir(exist_ok=True)

DPI = 250
MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, DATASET, FILE_PATH)

df['Date_Time_PH'] = pd.to_datetime(df['Date_Time_PH'], errors='coerce')
df = df.dropna(subset=['Date_Time_PH'])
df['Year']  = df['Date_Time_PH'].dt.year
df['Month'] = df['Date_Time_PH'].dt.month

ph_yearly = df.groupby('Year').size()
plt.figure(figsize=(14, 6))
ph_yearly.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Philippines: Earthquakes per Year')
plt.xlabel('Year'); plt.ylabel('Count')
plt.xticks(rotation=45); plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(DOCS / 'ph-yearly-earthquake-counts.png', dpi=DPI, bbox_inches='tight')
plt.close()

ph_monthly = df.groupby(['Year','Month']).size().reset_index(name='Count')
ph_monthly_pivot = ph_monthly.pivot(index='Year', columns='Month', values='Count').fillna(0)
plt.figure(figsize=(14, 6))
sns.heatmap(ph_monthly_pivot, cmap='YlGnBu', annot=True, fmt='.0f')
plt.title('Philippines: Monthly Earthquake Count by Year')
plt.ylabel('Year'); plt.xlabel('Month')
plt.tight_layout()
plt.savefig(DOCS / 'ph-yearmonth-earthquake-counts-heatmap.png', dpi=DPI, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(df['Magnitude'].dropna(), bins=30, color='teal', edgecolor='black', alpha=0.7)
plt.title('Philippines: Magnitude Distribution')
plt.xlabel('Magnitude'); plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.3); plt.tight_layout()
plt.savefig(DOCS / 'ph-magnitude-distribution.png', dpi=DPI, bbox_inches='tight')
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
plt.savefig(DOCS / 'ph-magnitude-categories.png', dpi=DPI, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
ph_trend = df.groupby('Year')['Magnitude'].agg(['mean','count']).reset_index()
ax1 = plt.gca(); ax2 = ax1.twinx()
ax1.plot(ph_trend['Year'], ph_trend['mean'], color='orange', marker='s', label='Average Magnitude')
ax2.plot(ph_trend['Year'], ph_trend['count'], color='blue', marker='o', alpha=0.5, label='Event Count')
ax1.set_ylabel('Average Magnitude', color='orange')
ax2.set_ylabel('Number of Events', color='blue')
ax1.set_xlabel('Year'); plt.title('Philippines: Average Magnitude & Event Count')
ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig(DOCS / 'ph-magnitude-eventcount-trend.png', dpi=DPI, bbox_inches='tight')
plt.close()

agg_max = (
    df.sort_values(['Year','Month','Magnitude','Date_Time_PH'],
                   ascending=[True,True,False,True])
      .groupby(['Year','Month'])
      .agg(MaxMagnitude=('Magnitude','max'),
           MaxMagDate=('Date_Time_PH','first'),
           MaxMagLocation=('General_Location','first'))
      .reset_index()
)

ph_max_pivot = agg_max.pivot(index='Year', columns='Month', values='MaxMagnitude').fillna(0)
ph_max_pivot.columns = MONTH_LABELS[:len(ph_max_pivot.columns)]
plt.figure(figsize=(14, 6))
sns.heatmap(ph_max_pivot, cmap='OrRd', annot=True, fmt='.1f', linewidths=.5, linecolor='white')
plt.title('Philippines: Highest Magnitude per Year‑Month')
plt.ylabel('Year'); plt.xlabel('Month')
plt.tight_layout()
plt.savefig(DOCS / 'ph-yearmonth-max-magnitude-heatmap.png', dpi=DPI, bbox_inches='tight')
plt.close()

leaders_count = agg_max.groupby('MaxMagLocation').size().reset_index(name='Months as Monthly Maximum')
leaders_peak = (
    agg_max.sort_values(['MaxMagLocation','MaxMagnitude','MaxMagDate'], ascending=[True,False,True])
          .groupby('MaxMagLocation', as_index=False)
          .first()[['MaxMagLocation','MaxMagnitude','MaxMagDate']]
          .rename(columns={'MaxMagnitude':'Highest Monthly Magnitude','MaxMagDate':'When Highest Occurred'})
)
leaders_full = (
    leaders_count.merge(leaders_peak, on='MaxMagLocation', how='left')
                 .sort_values(['Months as Monthly Maximum','Highest Monthly Magnitude'], ascending=[False,False])
                 .reset_index(drop=True)
)
leaders = leaders_full.head(10).copy()
leaders['Highest Monthly Magnitude'] = leaders['Highest Monthly Magnitude'].round(2)
leaders['When Highest Occurred'] = pd.to_datetime(leaders['When Highest Occurred']).dt.strftime('%Y-%m-%d %H:%M')
leaders.rename(columns={'MaxMagLocation':'Top Location'}, inplace=True)
leaders_html = leaders[['Top Location','Months as Monthly Maximum','Highest Monthly Magnitude','When Highest Occurred']].to_html(index=False)

df['Hour'] = df['Date_Time_PH'].dt.hour
day_hours   = [6,7,8,9,10,11,12,13,14,15,16,17]
night_hours = [18,19,20,21,22,23,0,1,2,3,4,5]
hour_counts = df.groupby('Hour').size().reset_index(name='Count').sort_values('Hour')
day_count   = int(hour_counts.loc[hour_counts['Hour'].isin(day_hours), 'Count'].sum())
night_count = int(hour_counts.loc[hour_counts['Hour'].isin(night_hours), 'Count'].sum())
total_count = int(hour_counts['Count'].sum()) if not hour_counts.empty else 0
day_ratio   = (day_count/total_count) if total_count else 0
night_ratio = (night_count/total_count) if total_count else 0

felt = df[df['Magnitude'] >= 4.0].copy()
felt_hour_counts = (felt['Date_Time_PH'].dt.hour.value_counts()
                    .rename_axis('Hour').reset_index(name='Count').sort_values('Hour'))
felt_day   = int(felt_hour_counts.loc[felt_hour_counts['Hour'].isin(day_hours), 'Count'].sum()) if not felt_hour_counts.empty else 0
felt_night = int(felt_hour_counts.loc[felt_hour_counts['Hour'].isin(night_hours), 'Count'].sum()) if not felt_hour_counts.empty else 0
felt_total = felt_day + felt_night
felt_day_ratio   = (felt_day/felt_total) if felt_total else 0
felt_night_ratio = (felt_night/felt_total) if felt_total else 0

page_html = f"""
<meta charset="utf-8">
<style>
  :root {{
    --ink:#1f2937; --muted:#6b7280; --brand:#c0392b; --edge:#e5e7eb;
  }}
  body {{ font-family:Segoe UI,Arial,sans-serif; color:var(--ink); max-width:1100px; margin:2rem auto; }}
  h1 {{ margin:.2rem 0 1rem; font-size:1.6rem; color:#c0392b; text-align:center; }}
  h2 {{ margin:1.1rem 0 .5rem; font-size:1.15rem; color:#3778c2; }}
  p.lead {{ color:var(--muted); text-align:center; margin:.4rem 0 1.2rem; }}
  section {{ background:#fff; border:1px solid var(--edge); border-radius:10px; padding:1rem 1.2rem; box-shadow:0 2px 8px rgba(0,0,0,.04); margin:1rem 0; }}
  table {{ border-collapse:collapse; width:100%; }}
  th,td {{ border:1px solid var(--edge); padding:8px; vertical-align:top; }}
  th {{ background:#f0f3f7; text-align:left; }}
  .meta {{ color:var(--muted); font-size:.95rem; text-align:center; margin-bottom:.8rem; }}
  .pill {{ display:inline-block; background:#eef2ff; color:#3730a3; border:1px solid #c7d2fe; padding:.2rem .55rem; border-radius:999px; font-size:.85rem; margin:.15rem .3rem 0 0; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; }}
  @media (max-width:900px){{ .grid2 {{ grid-template-columns:1fr; }} }}
</style>

<h1>Top Earthquake Epicenters & Timing</h1>
<p class="lead">
  Philippine sites most frequently topping monthly maximum magnitude, the largest monthly max at each, and its timing. Felt events (≥4.0) are compared by time of day.
</p>
<div class="meta">
  <span class="pill">Source: PHIVOLCS via Kaggle</span>
</div>

<section>
  <h2>Most Frequent “Monthly Maximum” Locations</h2>
  {leaders_html}
  <p class="meta">Top 10 of {len(leaders_full):,} locations with one or more months as top epicenter.</p>
</section>

<section>
  <div class="grid2">
    <div>
      <h2>All Earthquakes: Day vs Night</h2>
      <table>
        <thead><tr><th>Part of Day</th><th>Hours</th><th>Count</th><th>Percent</th></tr></thead>
        <tbody>
          <tr><td>Daytime</td><td>06–17</td><td>{day_count:,}</td><td>{day_ratio:.1%}</td></tr>
          <tr><td>Nighttime</td><td>18–05</td><td>{night_count:,}</td><td>{night_ratio:.1%}</td></tr>
          <tr><td>Total</td><td>00–23</td><td>{total_count:,}</td><td>100%</td></tr>
        </tbody>
      </table>
    </div>
    <div>
      <h2>Felt Earthquakes (≥4.0): Day vs Night</h2>
      <table>
        <thead><tr><th>Part of Day</th><th>Hours</th><th>Count</th><th>Percent</th></tr></thead>
        <tbody>
          <tr><td>Daytime</td><td>06–17</td><td>{felt_day:,}</td><td>{felt_day_ratio:.1%}</td></tr>
          <tr><td>Nighttime</td><td>18–05</td><td>{felt_night:,}</td><td>{felt_night_ratio:.1%}</td></tr>
          <tr><td>Total</td><td>00–23</td><td>{felt_total:,}</td><td>100%</td></tr>
        </tbody>
      </table>
      <p class="meta">Felt = magnitude ≥ 4.0 as a simple proxy for perceptibility.</p>
    </div>
  </div>
</section>
"""

(DOCS / 'ph-yearmonth-max-magnitude-table.html').write_text(page_html, encoding='utf-8')

ilocos_norte = df[df['General_Location'].str.contains('Ilocos Norte', case=False, na=False)].copy()

plt.figure(figsize=(14, 6))
ilocos_yearly = ilocos_norte.groupby('Year').size()
ilocos_yearly.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Ilocos Norte: Earthquakes per Year')
plt.xlabel('Year'); plt.ylabel('Count')
plt.xticks(rotation=45); plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(DOCS / 'ilocos-yearly-earthquake-counts.png', dpi=DPI, bbox_inches='tight')
plt.close()

plt.figure(figsize=(14, 6))
ilocos_monthly = ilocos_norte.groupby(['Year','Month']).size().reset_index(name='Count')
ilocos_monthly_pivot = ilocos_monthly.pivot(index='Year', columns='Month', values='Count').fillna(0)
sns.heatmap(ilocos_monthly_pivot, cmap='YlGnBu', annot=True, fmt='.0f')
plt.title('Ilocos Norte: Monthly Earthquake Count by Year')
plt.ylabel('Year'); plt.xlabel('Month'); plt.tight_layout()
plt.savefig(DOCS / 'ilocos-monthly-earthquake-counts.png', dpi=DPI, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(ilocos_norte['Magnitude'].dropna(), bins=20, color='teal', edgecolor='black', alpha=0.7)
plt.title('Ilocos Norte: Magnitude Distribution')
plt.xlabel('Magnitude'); plt.ylabel('Frequency'); plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(DOCS / 'ilocos-magnitude-distribution.png', dpi=DPI, bbox_inches='tight')
plt.close()

def mag_category_local(m: float) -> str:
  if m < 4.0: return 'Minor'
  elif m < 6.0: return 'Moderate'
  else: return 'Significant'
ilocos_norte['MagCategory'] = ilocos_norte['Magnitude'].apply(mag_category_local)

plt.figure(figsize=(8, 5))
ilocos_norte['MagCategory'].value_counts().reindex(['Minor','Moderate','Significant']).plot(
    kind='bar', color=['#22c55e','#f59e0b','#ef4444'])
plt.title('Ilocos Norte: Magnitude Categories')
plt.ylabel('Count'); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(DOCS / 'ilocos-magnitude-categories.png', dpi=DPI, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
iloc_trend = ilocos_norte.groupby('Year')['Magnitude'].agg(['mean','count']).reset_index()
ax1 = plt.gca(); ax2 = ax1.twinx()
ax1.plot(iloc_trend['Year'], iloc_trend['mean'], color='orange', marker='s', label='Average Magnitude')
ax2.plot(iloc_trend['Year'], iloc_trend['count'], color='blue', marker='o', alpha=0.5, label='Event Count')
ax1.set_ylabel('Average Magnitude', color='orange'); ax2.set_ylabel('Number of Events', color='blue')
ax1.set_xlabel('Year'); plt.title('Ilocos Norte: Average Magnitude & Event Count')
ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig(DOCS / 'ilocos-magnitude-eventcount-trend.png', dpi=DPI, bbox_inches='tight')
plt.close()

if not ilocos_norte.empty:
    ilocos_agg_max = (
        ilocos_norte.sort_values(['Year','Month','Magnitude','Date_Time_PH'],
                                 ascending=[True,True,False,True])
          .groupby(['Year','Month'])
          .agg(MaxMagnitude=('Magnitude','max'),
               MaxMagDate=('Date_Time_PH','first'),
               MaxMagLocation=('General_Location','first'))
          .reset_index()
    )
    ilocos_max_pivot = ilocos_agg_max.pivot(index='Year', columns='Month', values='MaxMagnitude').fillna(0)
    ilocos_max_pivot.columns = MONTH_LABELS[:len(ilocos_max_pivot.columns)]
    plt.figure(figsize=(14, 6))
    sns.heatmap(ilocos_max_pivot, cmap='OrRd', annot=True, fmt='.1f', linewidths=.5, linecolor='white')
    plt.title('Ilocos Norte: Highest Magnitude per Year‑Month')
    plt.ylabel('Year'); plt.xlabel('Month')
    plt.tight_layout()
    plt.savefig(DOCS / 'ilocos-yearmonth-max-magnitude-heatmap.png', dpi=DPI, bbox_inches='tight')
    plt.close()

