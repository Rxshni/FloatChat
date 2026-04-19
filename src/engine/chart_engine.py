import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── COLUMN METADATA ───────────────────────────────────────────────────────────
UNITS = {
    "avg_temp_celsius":  "°C",
    "avg_salinity_psu":  "PSU",
    "avg_doxy_umol_kg":  "µmol/kg",
    "avg_chla_mg_m3":    "mg/m³",
}

FULL_NAMES = {
    "avg_temp_celsius":  "Temperature",
    "avg_salinity_psu":  "Salinity",
    "avg_doxy_umol_kg":  "Dissolved Oxygen",
    "avg_chla_mg_m3":    "Chlorophyll-a",
}

MONTH_NAMES = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
    7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
}

DEPTH_ORDER = ["Surface", "Epipelagic", "Mesopelagic", "Bathypelagic"]

COLORS = {
    "Arabian Sea":   "#0EA5E9",   # ocean blue
    "Bay of Bengal": "#10B981",   # teal green
    "Surface":       "#F59E0B",
    "Epipelagic":    "#3B82F6",
    "Mesopelagic":   "#8B5CF6",
    "Bathypelagic":  "#1E293B",
}

# ── KEYWORD DETECTION ─────────────────────────────────────────────────────────

VISUALIZATION_TRIGGERS = [
    "show", "plot", "chart", "graph", "visualize", "visualise",
    "display", "draw", "render", "trend", "compare visually"
]

def should_visualize(question: str) -> bool:
    """Returns True if the user wants a chart."""
    q = question.lower()
    return any(trigger in q for trigger in VISUALIZATION_TRIGGERS)


def detect_chart_type(question: str, df: pd.DataFrame) -> str:
    q = question.lower()

    # Single value → metric card
    if len(df) == 1:
        return "metric"

    # ── Multi-variable (8 cols) — check FIRST before anything else ───────
    var_cols = [c for c in FULL_NAMES.keys() if c in df.columns]
    if len(var_cols) >= 2:
        return "multivar_line"

    # ── Heatmap: keyword OR depth × month structure ───────────────────────
    if any(w in q for w in ["heatmap", "heat map"]):
        return "heatmap"

    if ("depth_zone" in df.columns and "month" in df.columns and
            df["depth_zone"].nunique() > 1 and df["month"].nunique() > 1):
        return "heatmap"

    # Multiple depth zones, single time point → horizontal bar
    if "depth_zone" in df.columns and df["depth_zone"].nunique() > 1:
        if "month" in df.columns and df["month"].nunique() <= 2:
            if "year" in df.columns and df["year"].nunique() == 1:
                return "hbar"

    # Two regions across time → multiline comparison
    if "region_name" in df.columns and df["region_name"].nunique() == 2:
        if any(w in q for w in ["trend", "over", "month", "year", "compare"]):
            return "multiline"
        return "bar"

    # Time series → line chart
    if "month" in df.columns and df["month"].nunique() > 2:
        return "line"
    if "year" in df.columns and df["year"].nunique() > 2:
        return "line"

    return "bar"

def build_multivar_line(df: pd.DataFrame, question: str) -> go.Figure:
    """
    Multi-variable line chart — one subplot per variable (temp, sal, O2, chla).
    Used when SQL returns all four numeric columns.
    Each variable gets its own Y axis so different units don't clash.
    """
    var_cols = [c for c in FULL_NAMES.keys() if c in df.columns]
    
    if not var_cols:
        return build_line_chart(df, question)  # fallback

    # Build time label
    df = df.sort_values(["year", "month"])
    df["x_label"] = df["month_label"] + " " + df["year"].astype(str)

    # If multiple depth zones, plot one line per depth per variable
    color_col = None
    if "depth_zone" in df.columns and df["depth_zone"].nunique() > 1:
        color_col = "depth_zone"

    n_vars = len(var_cols)
    fig    = make_subplots(
        rows=n_vars, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            f"{FULL_NAMES[c]} ({UNITS.get(c,'')})" for c in var_cols
        ]
    )

    for row_idx, col in enumerate(var_cols, start=1):
        if color_col:
            for depth_val in df[color_col].cat.categories:
                sub = df[df[color_col] == depth_val]
                if sub[col].isna().all():
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=sub["x_label"],
                        y=sub[col],
                        mode="lines+markers",
                        name=str(depth_val),
                        line=dict(color=COLORS.get(str(depth_val))),
                        showlegend=(row_idx == 1),  # legend only on first subplot
                        legendgroup=str(depth_val)
                    ),
                    row=row_idx, col=1
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df["x_label"],
                    y=df[col],
                    mode="lines+markers",
                    name=FULL_NAMES[col],
                    line=dict(width=2.5),
                    showlegend=True
                ),
                row=row_idx, col=1
            )

    region = df["region_name"].iloc[0] if "region_name" in df.columns else ""
    fig.update_layout(
        height=220 * n_vars,
        title_text=f"Ocean Conditions — {region}",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Depth Zone",
        margin=dict(t=60, b=40, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ── SQL RESULT → DATAFRAME ────────────────────────────────────────────────────

def parse_sql_to_df(sql_data: str) -> pd.DataFrame | None:
    """
    Converts raw SQL result string (list of tuples) into a clean DataFrame.
    Handles the standard column order: region_name, depth_zone, year, month, value(s)
    """
    try:
        rows = eval(sql_data.split("\n[TRUNCATED")[0].strip())

        # ── PASTE DEBUG BLOCK 3 HERE ──────────────────────────────────────────
        print(f"   [Chart Parser] rows[:2]: {rows[:2] if rows else 'EMPTY'}")
        print(f"   [Chart Parser] n_cols: {len(rows[0]) if rows else 0}")
        # ─────────────────────────────────────────────────────────────────────

        if not isinstance(rows, list) or len(rows) == 0:
            return None

        # Detect number of columns
        n_cols = len(rows[0])

        if n_cols == 5:
            df = pd.DataFrame(rows, columns=[
                "region_name", "depth_zone", "year", "month", "value"
            ])
            df["value_col"] = "value"

        elif n_cols == 6:
            df = pd.DataFrame(rows, columns=[
                "region_name", "depth_zone", "year", "month", "value1", "value2"
            ])

        elif n_cols == 8:
            df = pd.DataFrame(rows, columns=[
                "region_name", "depth_zone", "year", "month",
                "avg_temp_celsius", "avg_salinity_psu",
                "avg_doxy_umol_kg", "avg_chla_mg_m3"
            ])
            # Convert all value columns to numeric
            for col in ["avg_temp_celsius","avg_salinity_psu",
                        "avg_doxy_umol_kg","avg_chla_mg_m3"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        else:
            # Generic fallback
            df = pd.DataFrame(rows)

        # Clean types
        for col in ["year", "month"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Add readable month labels
        if "month" in df.columns:
            df["month_label"] = df["month"].map(MONTH_NAMES)

        # Sort by depth zone properly
        if "depth_zone" in df.columns:
            df["depth_zone"] = pd.Categorical(
                df["depth_zone"], categories=DEPTH_ORDER, ordered=True
            )
            df = df.sort_values("depth_zone")

        return df

    except Exception as e:
        print(f"   [Chart] Could not parse SQL to DataFrame: {e}")
        return None


def detect_value_column(sql_query_hint: str, df: pd.DataFrame) -> str:
    """
    Figures out which column holds the actual data values.
    Checks the column names against known variable names.
    """
    for col in FULL_NAMES.keys():
        if col in df.columns:
            return col
    if "value" in df.columns:
        return "value"
    # Last numeric column as fallback
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    exclude      = ["year", "month"]
    candidates   = [c for c in numeric_cols if c not in exclude]
    return candidates[-1] if candidates else None


# ── CHART BUILDERS ────────────────────────────────────────────────────────────

def build_metric_card(df: pd.DataFrame, question: str) -> go.Figure:
    """Single value — big number display."""
    val_col  = detect_value_column("", df)
    value    = df[val_col].iloc[0] if val_col else "N/A"
    unit     = UNITS.get(val_col, "")
    label    = FULL_NAMES.get(val_col, val_col or "Value")

    region   = df["region_name"].iloc[0] if "region_name" in df.columns else ""
    depth    = df["depth_zone"].iloc[0]  if "depth_zone"  in df.columns else ""
    year     = int(df["year"].iloc[0])   if "year"        in df.columns else ""
    month    = MONTH_NAMES.get(int(df["month"].iloc[0]), "") if "month" in df.columns else ""

    fig = go.Figure(go.Indicator(
        mode  = "number",
        value = float(value) if value != "N/A" else 0,
        number = {"suffix": f" {unit}", "font": {"size": 64, "color": "#0EA5E9"}},
        title  = {"text": f"{label}<br><span style='font-size:0.7em;color:#64748B'>"
                           f"{region} · {depth} · {month} {year}</span>"}
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=60, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_line_chart(df: pd.DataFrame, question: str) -> go.Figure:
    """Time series — temperature/salinity over months or years."""
    val_col   = detect_value_column("", df)
    unit      = UNITS.get(val_col, "")
    var_label = FULL_NAMES.get(val_col, val_col or "Value")

    # X axis: prefer month label, else year
    if "month" in df.columns and df["month"].nunique() > 1:
        df = df.sort_values(["year", "month"])
        df["x_label"] = df["month_label"] + " " + df["year"].astype(str)
        x_col = "x_label"
    else:
        df = df.sort_values("year")
        x_col = "year"

    color_col = None
    if "region_name" in df.columns and df["region_name"].nunique() > 1:
        color_col = "region_name"
    elif "depth_zone" in df.columns and df["depth_zone"].nunique() > 1:
        color_col = "depth_zone"

    fig = px.line(
        df, x=x_col, y=val_col,
        color=color_col,
        markers=True,
        color_discrete_map=COLORS,
        labels={val_col: f"{var_label} ({unit})", x_col: ""},
        title=f"{var_label} Over Time",
        template="plotly_white"
    )
    fig.update_traces(line=dict(width=2.5), marker=dict(size=7))
    fig.update_layout(
        height=400,
        legend_title_text="",
        hovermode="x unified",
        margin=dict(t=50, b=40, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_bar_chart(df: pd.DataFrame, question: str) -> go.Figure:
    """Comparison bar chart — regions or depths side by side."""
    val_col   = detect_value_column("", df)
    unit      = UNITS.get(val_col, "")
    var_label = FULL_NAMES.get(val_col, val_col or "Value")

    # Group label
    if "region_name" in df.columns and df["region_name"].nunique() > 1:
        x_col     = "region_name"
        color_col = "region_name"
    elif "depth_zone" in df.columns and df["depth_zone"].nunique() > 1:
        x_col     = "depth_zone"
        color_col = "depth_zone"
    else:
        x_col     = df.columns[0]
        color_col = None

    fig = px.bar(
        df, x=x_col, y=val_col,
        color=color_col,
        color_discrete_map=COLORS,
        text=df[val_col].round(3).astype(str) + f" {unit}",
        labels={val_col: f"{var_label} ({unit})", x_col: ""},
        title=f"{var_label} Comparison",
        template="plotly_white"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(t=50, b=40, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_hbar_chart(df: pd.DataFrame, question: str) -> go.Figure:
    """Horizontal bar — depth zones ranked by value."""
    val_col   = detect_value_column("", df)
    unit      = UNITS.get(val_col, "")
    var_label = FULL_NAMES.get(val_col, val_col or "Value")

    df = df.sort_values(val_col, ascending=True)

    fig = px.bar(
        df, x=val_col, y="depth_zone",
        orientation="h",
        color="depth_zone",
        color_discrete_map=COLORS,
        text=df[val_col].round(3).astype(str) + f" {unit}",
        labels={val_col: f"{var_label} ({unit})", "depth_zone": "Depth Zone"},
        title=f"{var_label} by Depth Zone",
        template="plotly_white"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=350,
        showlegend=False,
        margin=dict(t=50, b=40, l=130, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_multiline_chart(df: pd.DataFrame, question: str) -> go.Figure:
    """Two regions on same chart for direct comparison over time."""
    val_col   = detect_value_column("", df)
    unit      = UNITS.get(val_col, "")
    var_label = FULL_NAMES.get(val_col, val_col or "Value")

    df = df.sort_values(["region_name", "year", "month"])
    df["x_label"] = df["month_label"] + " " + df["year"].astype(str)

    fig = px.line(
        df, x="x_label", y=val_col,
        color="region_name",
        markers=True,
        color_discrete_map=COLORS,
        labels={val_col: f"{var_label} ({unit})", "x_label": ""},
        title=f"{var_label} — Arabian Sea vs Bay of Bengal",
        template="plotly_white"
    )
    fig.update_traces(line=dict(width=2.5), marker=dict(size=7))
    fig.update_layout(
        height=420,
        legend_title_text="",
        hovermode="x unified",
        margin=dict(t=50, b=40, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_heatmap(df: pd.DataFrame, question: str) -> go.Figure:
    """Depth × month heatmap — shows seasonal + vertical structure."""
    val_col   = detect_value_column("", df)
    unit      = UNITS.get(val_col, "")
    var_label = FULL_NAMES.get(val_col, val_col or "Value")

    pivot = df.pivot_table(
        index="depth_zone", columns="month_label",
        values=val_col, aggfunc="mean"
    )

    # Reorder depth zones surface → bottom
    pivot = pivot.reindex([d for d in DEPTH_ORDER if d in pivot.index])

    # Reorder months
    month_order = [MONTH_NAMES[i] for i in range(1, 13) if MONTH_NAMES[i] in pivot.columns]
    pivot       = pivot[month_order]

    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlBu_r",
        labels={"color": f"{var_label} ({unit})"},
        title=f"{var_label} — Depth × Month Heatmap",
        template="plotly_white",
        aspect="auto"
    )
    fig.update_layout(
        height=380,
        margin=dict(t=50, b=40, l=130, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(title=unit)
    )
    return fig


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

def build_chart(sql_data: str, question: str) -> go.Figure | None:
    df = parse_sql_to_df(sql_data)
    if df is None or len(df) == 0:
        print("   [Chart] No data to chart.")
        return None

    val_col    = detect_value_column("", df)
    chart_type = detect_chart_type(question, df)
    print(f"   [Chart] Type: {chart_type} | Rows: {len(df)} | Cols: {list(df.columns)}")

    try:
        if chart_type == "metric":
            return build_metric_card(df, question)
        elif chart_type == "line":
            return build_line_chart(df, question)
        elif chart_type == "bar":
            return build_bar_chart(df, question)
        elif chart_type == "hbar":
            return build_hbar_chart(df, question)
        elif chart_type == "multiline":
            return build_multiline_chart(df, question)
        elif chart_type == "heatmap":
            return build_heatmap(df, question)
        elif chart_type == "multivar_line":          # ← ADD THIS
            return build_multivar_line(df, question)
        else:
            return build_bar_chart(df, question)
    except Exception as e:
        print(f"   [Chart] Build failed: {e}")
        return None