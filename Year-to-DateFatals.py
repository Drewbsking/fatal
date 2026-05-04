from __future__ import annotations

import io
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import vl_convert as vlc

st.set_page_config(
    page_title="Year-to-Date Traffic Fatalities",
    layout="wide",
    page_icon="📈",
)

DEFAULT_SHEET_NAME = "Crash Summaries"
DEFAULT_FILES = ("Fatals.csv", "Fatals.xlsx", "Fatals.xls")
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
CHART_WIDTH = 820  # ~8.5in at ~96dpi
CHART_HEIGHT = 550  # ~5.5in at ~100dpi
MARKER_SHAPES = [
    'circle',
    'square',
    'triangle-up',
    'triangle-down',
    'diamond',
    'cross',
    'wedge',
    'triangle-right',
    'triangle-left',
    'triangle',
    'diamond',
]
TABLEAU_COLORS = [
    '#4c78a8',
    '#f58518',
    '#e45756',
    '#72b7b2',
    '#54a24b',
    '#eeca3b',
    '#b279a2',
    '#ff9da6',
    '#9d755d',
    '#bab0ac',
]


def resolve_default_dataset() -> Path:
    """Return the first data file that matches the expected names."""
    script_dir = Path(__file__).resolve().parent
    candidate_dirs = [script_dir, script_dir.parent, script_dir / "data"]

    for directory in candidate_dirs:
        for name in DEFAULT_FILES:
            candidate = directory / name
            if candidate.exists():
                return candidate

    for pattern in ("**/*.xlsx", "**/*.xls", "**/*.csv"):
        matches = list(script_dir.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError("No spreadsheet or CSV file found. Upload a file to continue.")


def read_csv_with_fallback(source, *, is_buffer: bool) -> pd.DataFrame:
    """Read CSV content while trying multiple encodings."""
    encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'latin1', 'cp1252']
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            if is_buffer:
                source.seek(0)
            return pd.read_csv(source, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError, ValueError) as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise ValueError("Failed to read CSV file.")


@st.cache_data(show_spinner=False)
def load_dataframe(file_bytes: bytes | None, filename: str | None) -> tuple[pd.DataFrame, str]:
    """Load crash summary data either from an upload or a default file."""
    if file_bytes is not None and filename:
        buffer = io.BytesIO(file_bytes)
        if filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(buffer, sheet_name=DEFAULT_SHEET_NAME)
        else:
            df = read_csv_with_fallback(buffer, is_buffer=True)
        return df, filename

    path = resolve_default_dataset()
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=DEFAULT_SHEET_NAME)
    elif path.suffix.lower() == ".csv":
        df = read_csv_with_fallback(path, is_buffer=False)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return df, str(path)


def preprocess_crash_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw crash data and add Year/Month columns."""
    working = df.copy()
    working['Date'] = pd.to_datetime(working['Date'], errors='coerce')
    mask_valid = ~working['Date'].isna()
    mask_totals = ~working['Date'].astype(str).str.contains('Total', case=False, na=False)
    clean = working.loc[mask_valid & mask_totals].copy()
    clean['Year'] = clean['Date'].dt.year
    clean['Month'] = clean['Date'].dt.month
    clean = clean.dropna(subset=['Fatal Persons'])
    clean['Fatal Persons'] = pd.to_numeric(clean['Fatal Persons'], errors='coerce').fillna(0)
    return clean


def get_ytd_cutoff_month(data: pd.DataFrame, comparison_year: int) -> int:
    """Return the latest observed month for the comparison year."""
    today = pd.Timestamp.today()
    if int(comparison_year) == int(today.year):
        return max(1, int(today.month) - 1)

    year_months = data.loc[data['Year'] == comparison_year, 'Month'].dropna()
    if year_months.empty:
        return 12
    return int(year_months.max())


def format_latest_data_date(data: pd.DataFrame) -> str:
    """Return a display label for the newest crash date in the dataset."""
    latest_date = data['Date'].max()
    if pd.isna(latest_date):
        return "unknown"
    timestamp = pd.Timestamp(latest_date)
    return f"{timestamp:%b} {timestamp.day}, {timestamp.year}"


def build_cumulative_table(
    data: pd.DataFrame,
    start_year: int,
    end_year: int,
    cutoff_month: int | None = None,
) -> pd.DataFrame:
    """Aggregate fatalities by month and compute the cumulative totals."""
    if data.empty:
        return pd.DataFrame()

    subset = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)].copy()
    if subset.empty:
        return pd.DataFrame()

    all_months = pd.DataFrame(
        [(year, month) for year in range(start_year, end_year + 1) for month in range(1, 13)],
        columns=['Year', 'Month'],
    )
    monthly = subset.groupby(['Year', 'Month'])['Fatal Persons'].sum().reset_index()
    last_month_by_year = subset.groupby('Year')['Month'].max()

    merged = pd.merge(all_months, monthly, on=['Year', 'Month'], how='left')
    merged['last_month_observed'] = merged['Year'].map(last_month_by_year).fillna(0).astype(int)
    merged['max_month_allowed'] = merged['last_month_observed']
    if cutoff_month is not None:
        merged['max_month_allowed'] = int(cutoff_month)
    merged = merged[merged['Month'] <= merged['max_month_allowed']]
    merged = merged.drop(columns=['last_month_observed', 'max_month_allowed']).fillna(0)

    if merged.empty:
        return pd.DataFrame()

    pivot = merged.pivot(index='Month', columns='Year', values='Fatal Persons').cumsum()
    return pivot


def get_ytd_totals(pivot: pd.DataFrame) -> pd.Series:
    """Return YTD totals from the last available row in each year column."""
    if pivot.empty:
        return pd.Series(dtype=float)
    return pivot.ffill().iloc[-1].fillna(0)


def compute_monthly_average_from_pivot(pivot_complete: pd.DataFrame) -> pd.DataFrame:
    """Return average fatalities per month using the non-cumulative increments from the pivot."""
    if pivot_complete.empty:
        return pd.DataFrame()
    pivot_recent = pivot_complete.loc[:, [c for c in pivot_complete.columns if int(c) > 2008]]
    if pivot_recent.empty:
        pivot_recent = pivot_complete
    monthly_inc = pivot_recent.diff().fillna(pivot_recent)
    avg = monthly_inc.mean(axis=1).reset_index()
    avg.columns = ['Month', 'Average Fatalities']
    avg['MonthLabel'] = avg['Month'].apply(lambda idx: MONTH_LABELS[int(idx) - 1])
    avg['Average Fatalities'] = avg['Average Fatalities'].round(2)
    return avg


def create_average_bar_chart(avg_df: pd.DataFrame, subtitle: str) -> alt.Chart:
    """Build a bar chart showing average fatalities per month."""
    if avg_df.empty:
        return alt.Chart(pd.DataFrame({'MonthLabel': [], 'Average Fatalities': []}))
    chart = (
        alt.Chart(avg_df)
        .mark_bar(color='#d04b4b', opacity=0.85)
        .encode(
            x=alt.X(
                'MonthLabel:N',
                sort=MONTH_LABELS,
                title='Month',
                axis=alt.Axis(
                    labelColor='#000',
                    tickColor='#000',
                    titleColor='#000',
                    domainColor='#000',
                ),
            ),
            y=alt.Y(
                'Average Fatalities:Q',
                title='Average fatalities per month',
                axis=alt.Axis(labelColor='#000', tickColor='#000', domainColor='#000', titleColor='#000'),
            ),
            tooltip=[
                alt.Tooltip('MonthLabel:N', title='Month'),
                alt.Tooltip('Average Fatalities:Q', title='Average', format='.2f'),
            ],
        )
        .properties(
            width=min(CHART_WIDTH, 850),
            height=260,
            title=alt.TitleParams(text=subtitle, color='#000', fontSize=16, anchor='start'),
        )
        .configure_axis(labelColor='#000', titleColor='#000', tickColor='#000', domainColor='#000')
        .configure_view(stroke='transparent')
        .configure(background='white')
    )
    return chart


def create_ranking_bar_chart(year_totals: pd.Series, focus_year: int, title_text: str | None = None) -> alt.Chart:
    """Horizontal bar chart ranking years (least → most fatalities)."""
    if year_totals.empty:
        return alt.Chart(pd.DataFrame({'Year': [], 'Fatalities': []}))
    df = year_totals.sort_values().reset_index()
    df.columns = ['Year', 'Fatalities']
    df['Year'] = df['Year'].astype(int)
    df['Focus'] = df['Year'] == focus_year
    df['BarColor'] = np.where(df['Focus'], 'black', '#4c78a8')
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y('Year:N', sort=df['Year'].tolist(), title='Year'),
            x=alt.X('Fatalities:Q', title='Total fatalities'),
            color=alt.Color(
                'BarColor:N',
                scale=None,
                legend=None,
            ),
            tooltip=['Year', 'Fatalities'],
        )
        .properties(
            width=min(CHART_WIDTH, 850),
            height=280,
            title=alt.TitleParams(text=title_text or 'Year ranking (least → most fatalities)', color='#000', fontSize=16, anchor='start'),
        )
        .configure_axis(labelColor='#000', titleColor='#000', tickColor='#000', domainColor='#000')
        .configure_view(stroke='transparent')
        .configure(background='white')
    )
    return chart


def create_index_chart(index_df: pd.DataFrame, title_text: str | None = None) -> alt.Chart:
    """Bar chart showing observed vs expected index for the current year."""
    if index_df.empty:
        return alt.Chart(pd.DataFrame({'MonthLabel': [], 'IndexValue': []}))
    plot_df = index_df.copy()
    plot_df['BarColor'] = np.where(plot_df['IndexValue'] >= 1, '#d04b4b', '#3c7dcf')
    bars = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(
                'MonthLabel:N',
                sort=MONTH_LABELS,
                title='Month',
                axis=alt.Axis(labelColor='#000', tickColor='#000', domainColor='#000', titleColor='#000'),
            ),
            y=alt.Y(
                'IndexValue:Q',
                title='Index (observed / 5-year avg)',
                axis=alt.Axis(labelColor='#000', tickColor='#000', domainColor='#000', titleColor='#000'),
            ),
            color=alt.Color('BarColor:N', scale=None, legend=None),
            tooltip=[
                alt.Tooltip('MonthLabel:N', title='Month'),
                alt.Tooltip('Observed:Q', title='Observed', format=','),
                alt.Tooltip('Expected:Q', title='5-yr avg', format=',.2f'),
                alt.Tooltip('IndexValue:Q', title='Index', format='.2f'),
            ],
        )
    )
    baseline_rule = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(color='#000', strokeDash=[4, 4]).encode(y='y:Q')
    chart = alt.layer(bars, baseline_rule).properties(
        width=min(CHART_WIDTH, 850),
        height=260,
        title=alt.TitleParams(text=title_text or 'Current month index (observed vs 5-year avg)', color='#000', fontSize=16, anchor='start'),
    ).configure_axis(labelColor='#000', titleColor='#000', tickColor='#000', domainColor='#000'
    ).configure_view(stroke='transparent'
    ).configure(background='white')
    return chart


def create_rolling_chart(monthly_totals: pd.DataFrame, title_text: str | None = None) -> alt.Chart:
    """Line chart of 12-month rolling fatalities."""
    if monthly_totals.empty or 'Rolling12' not in monthly_totals:
        return alt.Chart(pd.DataFrame({'Date': [], 'Rolling12': []}))
    base_line = alt.Chart(monthly_totals).mark_line(color='#333').encode(
        x=alt.X('Date:T', title='Month', axis=alt.Axis(labelColor='#000', tickColor='#000', domainColor='#000', titleColor='#000')),
        y=alt.Y('Rolling12:Q', title='12-month rolling fatalities', axis=alt.Axis(labelColor='#000', tickColor='#000', domainColor='#000', titleColor='#000')),
        tooltip=[
            alt.Tooltip('Date:T', title='Month'),
            alt.Tooltip('Rolling12:Q', title='Rolling 12-mo', format=',.0f'),
        ],
    )

    may_points = monthly_totals[pd.to_datetime(monthly_totals['Date']).dt.month == 5]
    layers = [base_line]
    if not may_points.empty:
        markers = alt.Chart(may_points).mark_point(color='black', size=50).encode(
            x='Date:T',
            y='Rolling12:Q',
            tooltip=[
                alt.Tooltip('Date:T', title='Month'),
                alt.Tooltip('Rolling12:Q', title='Rolling 12-mo', format=',.0f'),
            ],
        )
        layers.append(markers)

    chart = alt.layer(*layers).properties(
        width=min(CHART_WIDTH, 850),
        height=260,
        title=alt.TitleParams(text=title_text or '12-month rolling fatalities', color='#000', fontSize=16, anchor='start'),
    ).configure_axis(labelColor='#000', titleColor='#000', tickColor='#000', domainColor='#000'
    ).configure_view(stroke='transparent'
    ).configure(background='white')
    return chart


def create_annual_totals_chart(annual_totals: pd.Series, title_text: str | None = None) -> alt.Chart:
    """Bar and trend chart for complete annual fatality totals."""
    if annual_totals.empty:
        return alt.Chart(pd.DataFrame({'Year': [], 'Fatalities': []}))

    df = annual_totals.sort_index().reset_index()
    df.columns = ['Year', 'Fatalities']
    df['Year'] = df['Year'].astype(int)
    df['YearLabel'] = df['Year'].astype(str)

    bars = alt.Chart(df).mark_bar(color='#4c78a8', opacity=0.85).encode(
        x=alt.X('YearLabel:N', title='Year', axis=alt.Axis(labelAngle=-45, labelColor='#000', tickColor='#000', titleColor='#000')),
        y=alt.Y('Fatalities:Q', title='Annual fatalities', axis=alt.Axis(labelColor='#000', tickColor='#000', titleColor='#000')),
        tooltip=[
            alt.Tooltip('YearLabel:N', title='Year'),
            alt.Tooltip('Fatalities:Q', title='Fatalities', format=','),
        ],
    )

    layers = [bars]
    if len(df) >= 2:
        slope, intercept = np.polyfit(df['Year'].to_numpy(), df['Fatalities'].to_numpy(), 1)
        trend_df = pd.DataFrame({
            'Year': df['Year'],
            'Fatalities': slope * df['Year'].to_numpy() + intercept,
        })
        trend_df['YearLabel'] = trend_df['Year'].astype(str)
        trend = alt.Chart(trend_df).mark_line(color='black', strokeDash=[6, 3], size=2).encode(
            x=alt.X('YearLabel:N', sort=df['YearLabel'].tolist(), title=None),
            y='Fatalities:Q',
        )
        layers.append(trend)

    chart = alt.layer(*layers).properties(
        width=min(CHART_WIDTH, 850),
        height=320,
        title=alt.TitleParams(text=title_text or 'Annual fatalities by complete year', color='#000', fontSize=16, anchor='start'),
    ).configure_axis(labelColor='#000', titleColor='#000', tickColor='#000', domainColor='#000'
    ).configure_view(stroke='transparent'
    ).configure(background='white')
    return chart


def create_altair_chart(
    pivot_complete: pd.DataFrame,
    history_pivot: pd.DataFrame,
    focus_year: int,
    show_focus_trend: bool,
    show_history_trend: bool,
    show_focus_labels: bool,
    show_history_labels: bool,
    title_text: str,
) -> alt.Chart:
    """Build the interactive Altair line chart and optional trend layers."""
    tidy = (
        pivot_complete.reset_index()
        .melt(id_vars='Month', var_name='Year', value_name='Fatal Persons')
        .dropna()
    )
    tidy['Year'] = tidy['Year'].astype(int)
    tidy['Month'] = tidy['Month'].astype(int)
    tidy['MonthLabel'] = tidy['Month'].apply(lambda idx: MONTH_LABELS[idx - 1])
    tidy['YearLabel'] = tidy['Year'].astype(str)
    tidy['Marker Shape'] = tidy['Year'].map(
        {year: MARKER_SHAPES[i % len(MARKER_SHAPES)] for i, year in enumerate(sorted(tidy['Year'].unique()))}
    )
    tidy['IsFocus'] = tidy['Year'] == focus_year
    tidy['LineDash'] = tidy['IsFocus'].map({True: 'solid', False: 'dashed'})
    tidy['LineSize'] = tidy['IsFocus'].map({True: 3.0, False: 1.5})
    tidy['LineOpacity'] = tidy['IsFocus'].map({True: 1.0, False: 0.5})
    tidy['PointOpacity'] = tidy['IsFocus'].map({True: 1.0, False: 0.8})
    sorted_years = sorted(tidy['Year'].unique())
    year_domain = [str(year) for year in sorted_years]
    year_range = [
        'black' if year == focus_year else TABLEAU_COLORS[i % len(TABLEAU_COLORS)]
        for i, year in enumerate(sorted_years)
    ]
    base = alt.Chart(tidy).encode(
        x=alt.X(
            'MonthLabel:N',
            title='Month',
            axis=alt.Axis(
                values=MONTH_LABELS,
                labelColor='#000',
                tickColor='#000',
                titleColor='#000',
            ),
            sort=MONTH_LABELS,
        ),
        y=alt.Y(
            'Fatal Persons:Q',
            title='Cumulative Fatalities (Persons)',
            axis=alt.Axis(
                labelColor='#000',
                tickColor='#000',
                titleColor='#000',
            ),
        ),
        tooltip=[
            alt.Tooltip('YearLabel:N', title='Year'),
            alt.Tooltip('MonthLabel:N', title='Month'),
            alt.Tooltip('Fatal Persons:Q', title='Fatalities', format=','),
        ],
    )

    line = base.mark_line().encode(
        detail=alt.Detail('YearLabel:N'),
        strokeDash=alt.StrokeDash(
            'LineDash:N',
            scale=alt.Scale(domain=['solid', 'dashed'], range=[[1, 0], [4, 4]]),
            legend=None,
        ),
        size=alt.Size('LineSize:Q', legend=None),
        opacity=alt.Opacity('LineOpacity:Q', legend=None),
        color=alt.Color(
            'YearLabel:N',
            scale=alt.Scale(domain=year_domain, range=year_range),
            legend=alt.Legend(
                title='Year',
                labelColor='#000',
                titleColor='#000',
                symbolType='stroke',
                symbolStrokeWidth=3,
                columns=2,
            ),
        ),
    )

    points = base.mark_point(
        filled=True,
        size=80,
        stroke='white',
        strokeWidth=0.8,
    ).encode(
        detail=alt.Detail('YearLabel:N'),
        opacity=alt.Opacity('PointOpacity:Q', legend=None),
        shape=alt.Shape('Marker Shape:N', legend=None),
        color=alt.Color('YearLabel:N', scale=alt.Scale(domain=year_domain, range=year_range), legend=None),
    )

    layers = [line, points]
    trend_frames = []

    if show_focus_trend and focus_year in pivot_complete.columns:
        focus_series = pivot_complete[focus_year].dropna()
        focus_months = sorted(int(month) for month in focus_series.index)
        if len(focus_months) >= 2 and len(focus_series) >= 2:
            x_vals = np.array(focus_months)
            y_vals = focus_series.loc[x_vals].to_numpy()
            if len(np.unique(x_vals)) >= 2:
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                intercept = -slope  # ensure line passes through (Jan, 0)
                x_plot = sorted(set([1] + focus_months))
                trend_y = slope * np.array(x_plot) + intercept
                trend_y = np.clip(trend_y, 0, None)
                focus_df = pd.DataFrame({'Month': x_plot, 'Fatal Persons': trend_y})
                focus_df['MonthLabel'] = focus_df['Month'].round().astype(int).apply(lambda idx: MONTH_LABELS[idx - 1])
                focus_df['Trend line'] = 'Current-year trend'
                trend_frames.append(focus_df)

    history_columns = [col for col in history_pivot.columns if int(col) != focus_year]
    if show_history_trend and len(history_columns) >= 1:
        hist_months = []
        hist_values = []
        for year in history_columns:
            series = history_pivot[year].dropna()
            if series.empty:
                continue
            for month, value in series.items():
                hist_months.append(month)
                hist_values.append(value)

        if len(hist_months) >= 2 and len(np.unique(hist_months)) >= 2:
            hist_x = np.array(hist_months)
            hist_y = np.array(hist_values)
            slope, intercept = np.polyfit(hist_x, hist_y, 1)
            intercept = -slope
            max_month = max(hist_x.max(), 12)
            x_line = np.linspace(1, max_month, 100)
            y_line = slope * x_line + intercept
            y_line = np.clip(y_line, 0, None)
            hist_df = pd.DataFrame({'Month': x_line, 'Fatal Persons': y_line})
            hist_df['MonthLabel'] = hist_df['Month'].round().astype(int).clip(1, 12).apply(lambda idx: MONTH_LABELS[idx - 1])
            hist_df = hist_df.drop_duplicates(subset=['MonthLabel'], keep='first')
            hist_df['Trend line'] = 'Historical trend'
            trend_frames.append(hist_df)

    if trend_frames:
        trend_df = pd.concat(trend_frames, ignore_index=True)
        trend_domain = [label for label in ['Current-year trend', 'Historical trend'] if label in trend_df['Trend line'].unique()]
        trend_color_map = {
            'Current-year trend': 'black',
            'Historical trend': '#555555',
        }
        trend_layer = alt.Chart(trend_df).mark_line(
            strokeDash=[6, 3],
            opacity=0.95,
            size=2,
        ).encode(
            detail=alt.Detail('Trend line:N'),
            x=alt.X('MonthLabel:N', title=None, sort=MONTH_LABELS),
            y='Fatal Persons:Q',
            color=alt.Color(
                'Trend line:N',
                scale=alt.Scale(domain=trend_domain, range=[trend_color_map[label] for label in trend_domain]),
                legend=alt.Legend(
                    title='Trend line',
                    labelColor='#000',
                    titleColor='#000',
                    symbolType='stroke',
                    symbolStrokeWidth=3,
                ),
            ),
        )
        layers.append(trend_layer)

    if show_focus_labels:
        focus_label_data = tidy[tidy['Year'] == focus_year].copy()
        if not focus_label_data.empty:
            focus_labels = alt.Chart(focus_label_data).mark_text(
                dy=-12,
                fontWeight='bold',
                fontSize=11,
                color='black',
            ).encode(
                detail=alt.Detail('YearLabel:N'),
                x=alt.X('MonthLabel:N', sort=MONTH_LABELS),
                y='Fatal Persons:Q',
                text=alt.Text('Fatal Persons:Q', format=','),
            )
            layers.append(focus_labels)

    if show_history_labels:
        history_label_data = tidy[tidy['Year'] != focus_year].copy()
        if not history_label_data.empty:
            history_labels = alt.Chart(history_label_data).mark_text(
                dy=-10,
                fontSize=10,
                color='#444',
            ).encode(
                detail=alt.Detail('YearLabel:N'),
                x=alt.X('MonthLabel:N', sort=MONTH_LABELS),
                y='Fatal Persons:Q',
                text=alt.Text('Fatal Persons:Q', format=','),
            )
            layers.append(history_labels)

    chart = alt.layer(*layers).resolve_scale(color='independent').properties(
        width=min(CHART_WIDTH, 850),
        height=min(CHART_HEIGHT, 520),
        title=alt.TitleParams(
            text=title_text,
            color='#000',
            fontSize=20,
            anchor='start',
        ),
    ).configure_view(
        stroke='transparent',
    ).configure_axis(
        gridColor='#d0d0d0',
        gridDash=[3, 3],
        labelColor='#000',
        titleColor='#000',
        tickColor='#000',
        domainColor='#000',
    ).configure_legend(
        titleColor='#000',
        labelColor='#000',
    ).configure(background='white')
    return chart


@st.cache_data(show_spinner=False)
def chart_spec_to_png_bytes(spec_json: str) -> bytes:
    """Convert a Vega-Lite spec into a PNG and cache the expensive render."""
    return vlc.vegalite_to_png(json.loads(spec_json), scale=1.0)


def chart_to_png_bytes(chart: alt.Chart) -> bytes:
    """Convert an Altair chart into a PNG for download."""
    spec = chart.to_dict()
    spec['width'] = CHART_WIDTH
    spec['height'] = CHART_HEIGHT
    spec_json = json.dumps(spec, sort_keys=True, separators=(',', ':'))
    return chart_spec_to_png_bytes(spec_json)


def render_year_over_year(processed: pd.DataFrame, data_source: str) -> None:
    """Render complete-year comparisons, excluding the partial current year."""
    current_year = pd.Timestamp.today().year
    complete_data = processed[processed['Year'] < current_year].copy()
    if complete_data.empty:
        st.warning(f"No complete years are available before {current_year}.")
        return

    complete_years = sorted(int(year) for year in complete_data['Year'].unique())
    min_year, max_year = complete_years[0], complete_years[-1]
    default_start = 2015 if min_year <= 2015 <= max_year else min_year
    with st.sidebar:
        st.caption(f"Source: `{Path(data_source).name}`")
        st.header("Complete Years")
        start_year, end_year = st.slider(
            "Year range",
            min_year,
            max_year,
            (default_start, max_year),
            key="yoy_year_range",
        )
        year_options = [year for year in complete_years if start_year <= year <= end_year]
        focus_year = st.selectbox(
            "Focus year (highlighted)",
            list(reversed(year_options)),
            index=0,
            key="yoy_focus_year",
        )
        show_history_trend = st.checkbox(
            "Show historical trend line for selected years",
            value=True,
            key="yoy_history_trend",
        )
        show_focus_trend = st.checkbox(
            "Show focus-year trend line",
            value=True,
            key="yoy_focus_trend",
        )

        st.header("Labels")
        label_mode = st.selectbox(
            "Value label mode",
            ["None", "Focus year", "Historical years", "All"],
            index=0,
            key="yoy_label_mode",
        )
        show_focus_labels = label_mode in {"Focus year", "All"}
        show_history_labels = label_mode in {"Historical years", "All"}

    complete_data = complete_data[(complete_data['Year'] >= start_year) & (complete_data['Year'] <= end_year)].copy()
    complete_years = sorted(int(year) for year in complete_data['Year'].unique())
    latest_complete_year = end_year

    st.markdown(
        f"Showing complete years **{start_year}–{end_year}**. "
        f"The partial current year ({current_year}) is excluded from this view. "
        f"Focus year: **{focus_year}**."
    )

    annual_totals = complete_data.groupby('Year')['Fatal Persons'].sum().sort_index()
    latest_total = int(annual_totals.loc[latest_complete_year])
    best_year = int(annual_totals.idxmin())
    best_value = int(annual_totals.min())
    worst_year = int(annual_totals.idxmax())
    worst_value = int(annual_totals.max())
    prior_total = annual_totals.get(latest_complete_year - 1)
    pct_change_prior = (
        ((latest_total - prior_total) / prior_total) * 100
        if prior_total and prior_total > 0
        else None
    )
    average_total = annual_totals.mean()
    pct_vs_average = ((latest_total - average_total) / average_total) * 100 if average_total > 0 else None

    col_latest, col_range, col_prior, col_avg = st.columns(4)
    col_latest.metric(f"Fatalities ({latest_complete_year})", f"{latest_total:,}")
    col_range.metric(
        "Best/Worst complete year",
        f"Best {best_year}: {best_value:,}",
        f"Worst {worst_year}: {worst_value:,}",
        delta_color="off",
    )
    col_prior.metric(
        "Latest vs prior year",
        f"{latest_total:,}",
        f"{pct_change_prior:+.1f}% vs {latest_complete_year - 1}" if pct_change_prior is not None else "N/A",
    )
    col_avg.metric(
        "Latest vs complete-year avg",
        f"{latest_total:,}",
        f"{pct_vs_average:+.1f}% vs avg" if pct_vs_average is not None else "N/A",
    )

    complete_pivot = build_cumulative_table(complete_data, start_year, end_year, cutoff_month=12)
    if not complete_pivot.empty:
        line_years = [year for year in complete_years if year in complete_pivot.columns]
        pivot_complete = complete_pivot.loc[:, line_years]
        line_chart = create_altair_chart(
            pivot_complete,
            pivot_complete,
            focus_year,
            show_focus_trend,
            show_history_trend,
            show_focus_labels,
            show_history_labels,
            title_text=f"Year-over-year cumulative fatalities ({start_year}–{end_year}) — Focus {focus_year}",
        )
        st.altair_chart(
            line_chart,
            use_container_width=False,
            key=(
                f"yoy_line_{start_year}_{end_year}_{focus_year}_"
                f"{int(show_focus_trend)}_{int(show_history_trend)}_"
                f"{int(show_focus_labels)}_{int(show_history_labels)}"
            ),
        )

    st.altair_chart(
        create_annual_totals_chart(annual_totals, f"Annual fatalities by complete year ({start_year}–{end_year})"),
        use_container_width=False,
    )
    st.altair_chart(
        create_ranking_bar_chart(annual_totals, latest_complete_year, f"Complete-year ranking ({start_year}–{end_year})"),
        use_container_width=False,
    )

    avg_df = compute_monthly_average_from_pivot(complete_pivot)
    if not avg_df.empty:
        st.altair_chart(
            create_average_bar_chart(avg_df, f"Average monthly fatalities across complete years ({start_year}–{end_year})"),
            use_container_width=False,
        )

    monthly_totals = (
        complete_data.groupby(['Year', 'Month'])['Fatal Persons']
        .sum()
        .reset_index()
        .sort_values(['Year', 'Month'])
    )
    monthly_totals['Date'] = pd.to_datetime(dict(year=monthly_totals['Year'], month=monthly_totals['Month'], day=1))
    monthly_totals['Rolling12'] = monthly_totals['Fatal Persons'].rolling(window=12).sum()
    st.altair_chart(
        create_rolling_chart(monthly_totals[['Date', 'Rolling12']].dropna(), f"12-month rolling fatalities through {latest_complete_year}"),
        use_container_width=False,
    )

    with st.expander("Show complete-year totals"):
        st.dataframe(
            annual_totals.rename("Fatalities").astype(int).reset_index(),
            use_container_width=True,
        )


def main():
    st.title("Year-to-Date Traffic Fatalities")
    st.caption("Visualize cumulative traffic-related fatalities by month and compare year-over-year trends.")

    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Optional data file",
            type=['csv', 'xls', 'xlsx'],
            help="If omitted, the app loads Fatals.csv from the repo.",
        )

    try:
        uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
        uploaded_name = uploaded_file.name if uploaded_file is not None else None
        df_raw, data_source = load_dataframe(uploaded_bytes, uploaded_name)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    processed = preprocess_crash_data(df_raw)
    processed = processed[processed['Year'] > 2008].copy()
    if processed.empty:
        st.error("No valid crash summaries after 2008 were detected in the provided file.")
        st.stop()

    with st.sidebar:
        st.header("View")
        view = st.radio(
            "Dashboard view",
            ["YTD Dashboard", "Year Over Year"],
            index=0,
            key="dashboard_view",
        )
    if view == "Year Over Year":
        render_year_over_year(processed, data_source)
        return

    years = sorted(int(year) for year in processed['Year'].unique())
    min_year, max_year = years[0], years[-1]
    default_start = 2015 if min_year <= 2015 <= max_year else min_year
    current_year = pd.Timestamp.today().year
    focus_year = current_year
    report_year = current_year
    report_cutoff_month = get_ytd_cutoff_month(processed, report_year)
    report_cutoff_label = MONTH_LABELS[report_cutoff_month - 1]
    latest_data_label = format_latest_data_date(processed)
    slider_max_year = max(max_year, current_year)
    default_end = min(slider_max_year, current_year)

    with st.sidebar:
        st.caption(f"Source: `{Path(data_source).name}`")
        st.header("Years")
        start_year, end_year = st.slider(
            "Year range",
            min_year,
            slider_max_year,
            (default_start, default_end),
        )
        show_history_trend = st.checkbox(
            "Show historical trend line for selected years",
            value=True,
            key="history_trend",
        )

        st.header("Current Year")
        st.caption(f"Highlighted year: **{focus_year}**")
        show_focus_trend = st.checkbox("Show current-year trend line", value=True, key="focus_trend")

        st.header("Labels")
        label_mode = st.selectbox(
            "Value label mode",
            ["None", "Current year", "Historical years", "All"],
            index=0,
        )
        show_focus_labels = label_mode in {"Current year", "All"}
        show_history_labels = label_mode in {"Historical years", "All"}

    slider_years = [year for year in years if start_year <= year <= end_year]
    display_years = sorted(set(slider_years + [focus_year]))
    pivot_span_start, pivot_span_end = display_years[0], display_years[-1]
    chart_cutoff_month = report_cutoff_month
    chart_cutoff_label = report_cutoff_label

    pivot_full = build_cumulative_table(processed, pivot_span_start, pivot_span_end, cutoff_month=chart_cutoff_month)
    if pivot_full.empty:
        st.warning("Not enough data to build the visualization. Try adjusting the selected years.")
        st.stop()
    pivot_complete = pivot_full.loc[:, [col for col in display_years if col in pivot_full.columns]]
    full_history_pivot = build_cumulative_table(processed, years[0], years[-1], cutoff_month=report_cutoff_month)
    if full_history_pivot.empty:
        full_history_pivot = pivot_complete.copy()
    report_history_pivot = build_cumulative_table(processed, years[0], report_year, cutoff_month=report_cutoff_month)
    if report_history_pivot.empty:
        report_history_pivot = pivot_complete.loc[:, [col for col in pivot_complete.columns if int(col) <= report_year]]

    displayed_text = ", ".join(str(year) for year in pivot_complete.columns)
    st.markdown(
        f"Showing {len(pivot_complete.columns)} year(s): **{displayed_text}** through **{chart_cutoff_label}** "
        f"(current year {focus_year}, slider {start_year}–{end_year})."
    )

    st.caption(f"Latest record in source data: **{latest_data_label}**.")

    history_pivot = pivot_complete if len(pivot_complete.columns) > 1 else full_history_pivot

    chart = create_altair_chart(
        pivot_complete,
        history_pivot,
        focus_year,
        show_focus_trend,
        show_history_trend,
        show_focus_labels,
        show_history_labels,
        title_text=f"Year-to-Date Fatalities ({start_year}–{end_year}) — Current Year {focus_year}",
    )
    chart_key = (
        f"ytd_chart_{start_year}_{end_year}_{focus_year}_{chart_cutoff_month}_"
        f"{int(show_focus_trend)}_{int(show_history_trend)}_"
        f"{int(show_focus_labels)}_{int(show_history_labels)}"
    )
    st.altair_chart(chart, use_container_width=False, key=chart_key)

    prepare_png = st.button("Prepare chart PNG", type="primary")
    if prepare_png:
        with st.spinner("Preparing PNG download..."):
            try:
                st.session_state["ytd_png_bytes"] = chart_to_png_bytes(chart)
                st.session_state["ytd_png_key"] = chart_key
            except Exception as exc:  # pragma: no cover - best effort
                st.warning(f"Unable to create image export: {exc}")

    if st.session_state.get("ytd_png_key") == chart_key and st.session_state.get("ytd_png_bytes"):
        st.download_button(
            "Download chart as PNG",
            data=st.session_state["ytd_png_bytes"],
            file_name="year_to_date_fatalities.png",
            mime="image/png",
            type="primary",
            on_click="ignore",
        )

    # From here down, use full-history YTD data with zero-fatality months carried through the cutoff.
    full_pivot = report_history_pivot if not report_history_pivot.empty else pivot_complete
    year_totals_ytd = get_ytd_totals(full_pivot)

    if not year_totals_ytd.empty:
        focus_total = int(year_totals_ytd.get(report_year, 0))
        best_year = int(year_totals_ytd.idxmin())
        best_value = int(year_totals_ytd.min())
        worst_year = int(year_totals_ytd.idxmax())
        worst_value = int(year_totals_ytd.max())

        prev_total = year_totals_ytd.get(report_year - 1)
        pct_change_prior = (
            ((focus_total - prev_total) / prev_total) * 100
            if prev_total and prev_total > 0
            else None
        )

        all_mean = year_totals_ytd.mean() if len(year_totals_ytd) > 0 else None
        pct_vs_all = (
            ((focus_total - all_mean) / all_mean) * 100
            if all_mean and all_mean > 0
            else None
        )

        st.markdown("---")
        st.subheader(f"Quick stats through {report_cutoff_label} (all years)")
        col_focus, col_best, col_prior, col_all = st.columns(4)
        col_focus.metric(f"YTD fatalities ({report_year})", f"{focus_total:,}")
        col_best.metric(
            f"Best/Worst through {report_cutoff_label}",
            f"Best {best_year}: {best_value:,}",
            f"Worst {worst_year}: {worst_value:,}",
            delta_color="off",
        )
        col_prior.metric(
            "Current vs prior year",
            f"{focus_total:,}",
            f"{pct_change_prior:+.1f}% vs {report_year - 1}" if pct_change_prior is not None else "N/A",
        )
        col_all.metric(
            "Current vs all-years avg",
            f"{focus_total:,}",
            f"{pct_vs_all:+.1f}% vs all-years avg" if pct_vs_all is not None else "N/A",
        )

    # Index chart using full YTD history (5-year avg baseline before the current year)
    safe_pivot = full_pivot
    focus_year_index = max(safe_pivot.columns.astype(int)) if not safe_pivot.empty else None
    monthly_inc_all = safe_pivot.diff().fillna(safe_pivot)
    prev_years = [y for y in monthly_inc_all.columns if int(y) < int(focus_year_index)] if focus_year_index else []
    prev_years = sorted(prev_years)[-5:]
    focus_monthly = monthly_inc_all[focus_year_index] if focus_year_index in monthly_inc_all.columns else pd.Series(dtype=float)
    baseline_monthly = monthly_inc_all[prev_years].mean(axis=1) if prev_years else pd.Series(0, index=monthly_inc_all.index)
    index_df = pd.DataFrame({
        'Month': monthly_inc_all.index,
        'Observed': focus_monthly.reindex(monthly_inc_all.index).fillna(0),
        'Expected': baseline_monthly.fillna(0),
    })
    index_df['MonthLabel'] = index_df['Month'].apply(lambda m: MONTH_LABELS[int(m) - 1])
    index_df['IndexValue'] = np.where(
        index_df['Expected'] > 0,
        index_df['Observed'] / index_df['Expected'],
        0,
    )
    index_chart = create_index_chart(index_df, title_text=f"Current month index ({report_year} vs 5-year avg through {report_cutoff_label})")
    st.altair_chart(index_chart, use_container_width=False)

    # Year ranking as bar chart (all years, same YTD cutoff)
    focus_for_ranking = report_year if not year_totals_ytd.empty else focus_year
    ranking_chart = create_ranking_bar_chart(year_totals_ytd, focus_for_ranking, title_text=f"Year ranking through {report_cutoff_label} (current year {report_year})")
    st.altair_chart(ranking_chart, use_container_width=False)

    # Average monthly bar chart (all years, same YTD cutoff)
    avg_df = compute_monthly_average_from_pivot(full_pivot)
    if not avg_df.empty:
        avg_chart = create_average_bar_chart(avg_df, f"Average monthly fatalities through {report_cutoff_label} across all years")
        st.altair_chart(avg_chart, use_container_width=False)

    # 12-month rolling fatalities
    monthly_totals = (
        processed.groupby(['Year', 'Month'])['Fatal Persons']
        .sum()
        .reset_index()
        .sort_values(['Year', 'Month'])
    )
    monthly_totals['Date'] = pd.to_datetime(dict(year=monthly_totals['Year'], month=monthly_totals['Month'], day=1))
    monthly_totals['Rolling12'] = monthly_totals['Fatal Persons'].rolling(window=12).sum()
    rolling_chart = create_rolling_chart(monthly_totals[['Date', 'Rolling12']].dropna(), title_text=f"12-month rolling fatalities (current year {report_year})")
    st.altair_chart(rolling_chart, use_container_width=False)

    with st.expander("Show cumulative table"):
        formatted = pivot_complete.round(0).fillna(0).astype(int)
        st.dataframe(formatted, use_container_width=True)


if __name__ == "__main__":
    main()
