from __future__ import annotations

import io
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
DEFAULT_FILES = ("Book1.xlsx", "Book1.xls", "Book1.csv")
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


def build_cumulative_table(data: pd.DataFrame, start_year: int, focus_year: int) -> pd.DataFrame:
    """Aggregate fatalities by month and compute the cumulative totals."""
    if data.empty:
        return pd.DataFrame()

    current_year_global = data['Year'].max()
    subset = data[(data['Year'] >= start_year) & (data['Year'] <= focus_year)].copy()
    if subset.empty:
        return pd.DataFrame()

    all_months = pd.DataFrame(
        [(year, month) for year in range(start_year, focus_year + 1) for month in range(1, 13)],
        columns=['Year', 'Month'],
    )
    monthly = subset.groupby(['Year', 'Month'])['Fatal Persons'].sum().reset_index()
    last_month_by_year = subset.groupby('Year')['Month'].max()

    merged = pd.merge(all_months, monthly, on=['Year', 'Month'], how='left')
    merged['last_month_observed'] = merged['Year'].map(last_month_by_year)
    merged['max_month_allowed'] = np.where(
        (merged['Year'] == current_year_global) & merged['last_month_observed'].notna(),
        merged['last_month_observed'],
        12,
    )
    merged = merged[merged['Month'] <= merged['max_month_allowed']]
    merged = merged.drop(columns=['last_month_observed', 'max_month_allowed']).fillna(0)

    if merged.empty:
        return pd.DataFrame()

    pivot = merged.pivot(index='Month', columns='Year', values='Fatal Persons').cumsum()
    return pivot


def create_altair_chart(
    pivot_complete: pd.DataFrame,
    filtered_data: pd.DataFrame,
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
    tidy['Marker Shape'] = tidy['Year'].map(
        {year: MARKER_SHAPES[i % len(MARKER_SHAPES)] for i, year in enumerate(sorted(tidy['Year'].unique()))}
    )
    base = alt.Chart(tidy).encode(
        x=alt.X(
            'Month:Q',
            title='Month',
            scale=alt.Scale(domain=[1, 12]),
            axis=alt.Axis(
                values=list(range(1, 13)),
                labelExpr="['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][datum.value - 1]",
                labelColor='#000',
                tickColor='#000',
                titleColor='#000',
            ),
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
            alt.Tooltip('Year:N'),
            alt.Tooltip('MonthLabel:N', title='Month'),
            alt.Tooltip('Fatal Persons:Q', title='Fatalities', format=','),
        ],
    )

    line = base.mark_line().encode(
        strokeDash=alt.condition(alt.datum.Year == focus_year, alt.value([1, 0]), alt.value([4, 4])),
        size=alt.condition(alt.datum.Year == focus_year, alt.value(3), alt.value(1.5)),
        opacity=alt.condition(alt.datum.Year == focus_year, alt.value(1), alt.value(0.5)),
        color=alt.condition(
            alt.datum.Year == focus_year,
            alt.value('black'),
            alt.Color(
                'Year:N',
                legend=alt.Legend(
                    title='Year',
                    labelColor='#000',
                    titleColor='#000',
                    symbolType='stroke',
                    symbolStrokeWidth=3,
                    columns=2,
                ),
                scale=alt.Scale(scheme='tableau10'),
            ),
        ),
    )

    points = base.mark_point(
        filled=True,
        size=80,
        stroke='white',
        strokeWidth=0.8,
    ).encode(
        opacity=alt.condition(alt.datum.Year == focus_year, alt.value(1), alt.value(0.8)),
        shape=alt.Shape('Marker Shape:N', legend=None),
        color=alt.condition(
            alt.datum.Year == focus_year,
            alt.value('black'),
            alt.Color('Year:N', legend=None, scale=alt.Scale(scheme='tableau10')),
        ),
    )

    layers = [line, points]

    if show_focus_trend and focus_year in pivot_complete.columns:
        focus_series = pivot_complete[focus_year].dropna()
        focus_months = sorted(filtered_data.loc[filtered_data['Year'] == focus_year, 'Month'].unique())
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
                focus_layer = alt.Chart(focus_df).mark_line(
                    strokeDash=[6, 3],
                    color='black',
                    size=2,
                ).encode(
                    x=alt.X('Month:Q', title=None),
                    y='Fatal Persons:Q',
                )
                layers.append(focus_layer)

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
            hist_layer = alt.Chart(hist_df).mark_line(
                strokeDash=[6, 3],
                color='#555555',
                opacity=0.95,
                size=2,
            ).encode(
                x=alt.X('Month:Q', title=None),
                y='Fatal Persons:Q',
            )
            layers.append(hist_layer)

    if show_focus_labels:
        focus_label_data = tidy[tidy['Year'] == focus_year].copy()
        if not focus_label_data.empty:
            focus_label_data['MonthQ'] = focus_label_data['Month'].astype(float)
            focus_labels = alt.Chart(focus_label_data).mark_text(
                dy=-12,
                fontWeight='bold',
                fontSize=11,
                color='black',
            ).encode(
                x='MonthQ:Q',
                y='Fatal Persons:Q',
                text=alt.Text('Fatal Persons:Q', format=','),
            )
            layers.append(focus_labels)

    if show_history_labels:
        history_label_data = tidy[tidy['Year'] != focus_year].copy()
        if not history_label_data.empty:
            history_label_data['MonthQ'] = history_label_data['Month'].astype(float)
            history_labels = alt.Chart(history_label_data).mark_text(
                dy=-10,
                fontSize=10,
                color='#444',
            ).encode(
                x='MonthQ:Q',
                y='Fatal Persons:Q',
                text=alt.Text('Fatal Persons:Q', format=','),
            )
            layers.append(history_labels)

    legend_df = pd.DataFrame({'LegendYear': [str(focus_year)], 'Month': [0], 'Fatal Persons': [0]})
    legend_layer = alt.Chart(legend_df).mark_point(opacity=0, size=0).encode(
        x='Month:Q',
        y='Fatal Persons:Q',
        color=alt.Color(
            'LegendYear:N',
            scale=alt.Scale(domain=[str(focus_year)], range=['black']),
            legend=alt.Legend(
                title='Year',
                labelColor='#000',
                titleColor='#000',
                symbolType='stroke',
                symbolStrokeWidth=3,
                symbolSize=200,
            ),
        ),
    )
    layers.append(legend_layer)

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
    ).configure_legend(
        titleColor='#000',
        labelColor='#000',
    ).configure(background='white').interactive()
    return chart


def chart_to_png_bytes(chart: alt.Chart) -> bytes | None:
    """Convert an Altair chart into a PNG for download."""
    try:
        spec = chart.to_dict()
        spec['width'] = CHART_WIDTH
        spec['height'] = CHART_HEIGHT
        png_bytes = vlc.vegalite_to_png(spec, scale=1.0)
        return png_bytes
    except Exception as exc:  # pragma: no cover - best effort
        st.warning(f"Unable to create image export: {exc}")
        return None


def main():
    st.title("Year-to-Date Traffic Fatalities")
    st.caption("Visualize cumulative traffic-related fatalities by month and compare year-over-year trends.")

    try:
        df_raw, data_source = load_dataframe(None, None)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    processed = preprocess_crash_data(df_raw)
    if processed.empty:
        st.error("No valid crash summaries were detected in the provided file.")
        st.stop()

    years = sorted(processed['Year'].unique())
    min_year, max_year = years[0], years[-1]
    default_start = 2015 if min_year <= 2015 <= max_year else min_year

    with st.sidebar:
        st.header("Years")
        start_year, end_year = st.slider(
            "Year range",
            min_year,
            max_year,
            (default_start, max_year),
        )
        show_history_trend = st.checkbox("Show historical trend line", value=True, key="history_trend")

        st.header("Focus")
        focus_options = list(reversed(years))
        focus_year = st.selectbox(
            "Focus year (highlighted)",
            focus_options,
            index=0,
        )
        show_focus_trend = st.checkbox("Show focus-year trend line", value=True, key="focus_trend")

        st.header("Labels")
        label_mode = st.selectbox(
            "Value label mode",
            ["None", "Focus year", "Historical years", "All"],
            index=0,
        )
        show_focus_labels = label_mode in {"Focus year", "All"}
        show_history_labels = label_mode in {"Historical years", "All"}

    slider_years = [year for year in years if start_year <= year <= end_year]
    display_years = sorted(set(slider_years + [focus_year]))
    pivot_span_start, pivot_span_end = display_years[0], display_years[-1]

    pivot_full = build_cumulative_table(processed, pivot_span_start, pivot_span_end)
    if pivot_full.empty:
        st.warning("Not enough data to build the visualization. Try adjusting the selected years.")
        st.stop()
    pivot_complete = pivot_full.loc[:, [col for col in display_years if col in pivot_full.columns]]
    full_history_pivot = build_cumulative_table(processed, years[0], years[-1])
    if full_history_pivot.empty:
        full_history_pivot = pivot_complete.copy()

    displayed_text = ", ".join(str(year) for year in pivot_complete.columns)
    st.markdown(
        f"Showing {len(pivot_complete.columns)} year(s): **{displayed_text}** (slider {start_year}–{end_year}, focus {focus_year})."
    )

    history_pivot = pivot_complete if len(pivot_complete.columns) > 1 else full_history_pivot

    if start_year <= 2008:
        st.warning("Data from 2008 and earlier reflects annual totals only (no month-by-month breakdown).")

    if 'chart_reset_token' not in st.session_state:
        st.session_state['chart_reset_token'] = 0
    if st.button("Reset zoom/view"):
        st.session_state['chart_reset_token'] += 1

    chart = create_altair_chart(
        pivot_complete,
        processed[processed['Year'].isin(display_years)],
        history_pivot,
        focus_year,
        show_focus_trend,
        show_history_trend,
        show_focus_labels,
        show_history_labels,
        title_text=f"Year-to-Date Fatalities ({start_year}–{end_year}) — Focus {focus_year}",
    )
    st.altair_chart(chart, use_container_width=False, key=f"line_chart_{st.session_state['chart_reset_token']}")
    png_bytes = chart_to_png_bytes(chart)
    if png_bytes:
        st.download_button(
            "Download chart as PNG",
            data=png_bytes,
            file_name="year_to_date_fatalities.png",
            mime="image/png",
            type="primary",
        )

    with st.expander("Show cumulative table"):
        formatted = pivot_complete.round(0).fillna(0).astype(int)
        st.dataframe(formatted, use_container_width=True)


if __name__ == "__main__":
    main()
