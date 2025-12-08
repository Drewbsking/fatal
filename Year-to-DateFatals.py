import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# Resolve Excel file path relative to this script
script_dir = Path(__file__).resolve().parent
candidates = [
    script_dir / "Book1.xlsx",
    script_dir.parent / "Book1.xlsx",
    script_dir / "data" / "Book1.xlsx"
]

file_path = None
for p in candidates:
    if p.exists():
        file_path = p
        break

# Fallback: find the first .xlsx under the repo if not found in common locations
if file_path is None:
    matches = list(script_dir.glob("**/*.xlsx"))
    if matches:
        file_path = matches[0]
    else:
        raise FileNotFoundError("Could not find Book1.xlsx (searched common relative locations)")

# Load the Excel file
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, sheet_name="Crash Summaries")

# Process the data (same steps as before)
mask_valid = ~df['Date'].astype(str).str.contains('Total', case=False, na=False)
df_valid = df.loc[mask_valid].copy()
df_valid['Year'] = pd.to_datetime(df_valid['Date']).dt.year
filtered_data = df_valid[df_valid['Year'] >= 2015].copy()
filtered_data['Month'] = pd.to_datetime(filtered_data['Date']).dt.month

focus_year = 2025
all_months = pd.DataFrame([(year, month) for year in range(2015, focus_year + 1) for month in range(1, 13)],
                          columns=['Year', 'Month'])
monthly_fatalities = filtered_data.groupby(['Year', 'Month'])['Fatal Persons'].sum().reset_index()
last_month_by_year = filtered_data.groupby('Year')['Month'].max()
complete_data = pd.merge(all_months, monthly_fatalities, on=['Year', 'Month'], how='left')
complete_data['last_month_observed'] = complete_data['Year'].map(last_month_by_year)
# Keep only months up through the latest observed month per year so we do not prefill future months
complete_data = complete_data[complete_data['last_month_observed'].notna()]
complete_data = complete_data[complete_data['Month'] <= complete_data['last_month_observed']]
complete_data = complete_data.drop(columns='last_month_observed').fillna(0)
pivot_complete = complete_data.pivot(index='Month', columns='Year', values='Fatal Persons').cumsum()

# Plot
num_years = len(pivot_complete.columns)
color_map = plt.get_cmap('tab10', num_years)
colors = [color_map(i) for i in range(num_years)]
unique_markers = ['o', 's', 'D', '^', 'v', '*', 'P', 'X', 'H', '<', '>']
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.figure(figsize=(12, 7))
focus_column_key = None
for i, (year, color) in enumerate(zip(pivot_complete.columns, colors)):
    is_focus_year = int(year) == focus_year
    if is_focus_year:
        focus_column_key = year
    linestyle = '-' if is_focus_year else ':'
    marker_style = unique_markers[i % len(unique_markers)]
    line_width = 2.5 if is_focus_year else 1.0
    line = plt.plot(
        pivot_complete.index,
        pivot_complete[year],
        marker=marker_style,
        linestyle=linestyle,
        color=color,
        linewidth=line_width,
        label=int(year)
    )[0]
    if is_focus_year:
        # Add a faint drop shadow so the focus year stands out
        line.set_path_effects([
            path_effects.SimpleLineShadow(offset=(2, -2), alpha=0.4),
            path_effects.Normal()
        ])

focus_months = sorted(filtered_data.loc[filtered_data['Year'] == focus_year, 'Month'].unique())
if len(focus_months) >= 2 and focus_column_key is not None:
    focus_series = pivot_complete[focus_column_key]
    x_vals = np.array(focus_months)
    y_vals = focus_series.loc[x_vals].to_numpy()
    if len(np.unique(x_vals)) >= 2:
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        trend_y = slope * x_vals + intercept
        plt.plot(
            x_vals,
            trend_y,
            color='black',
            linestyle='--',
            linewidth=2.0,
            label=f'{focus_year} trend'
        )

# Aggregate all historical points and fit a single trend to them
hist_months = []
hist_values = []
for year in pivot_complete.columns:
    if int(year) == focus_year:
        continue
    months_available = filtered_data.loc[filtered_data['Year'] == int(year), 'Month'].unique()
    if len(months_available) == 0:
        continue
    series = pivot_complete[year]
    for month in months_available:
        hist_months.append(month)
        hist_values.append(series.loc[month])

if len(hist_months) >= 2 and len(np.unique(hist_months)) >= 2:
    hist_x = np.array(hist_months)
    hist_y = np.array(hist_values)
    slope, intercept = np.polyfit(hist_x, hist_y, 1)
    x_line = np.linspace(hist_x.min(), hist_x.max(), 100)
    y_line = slope * x_line + intercept
    plt.plot(
        x_line,
        y_line,
        color='gray',
        linestyle='--',
        linewidth=1.8,
        alpha=0.7,
        label='Historical trend'
    )

plt.xticks(ticks=range(1, 13), labels=month_labels)
plt.title('Year-to-Date Fatalities by Month (2015–2025): Traffic-Related Deaths')
plt.xlabel('Month')
plt.ylabel('Cumulative Fatalities (Persons)')
plt.legend(title='Year')
plt.grid(True)

# Save output so it can be viewed in headless environments
output_path = script_dir / "Year-to-DateFatals.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {output_path}")
plt.show()
