"""
ocean_map.py
============
Interactive ocean map for FloatChat.
- Arabian Sea and Bay of Bengal: colored polygon fill + rich hover card on hover
- Other major oceans: name only on hover, no data
- No stat cards, no bubbles
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()
_engine = create_engine(os.getenv("DATABASE_URL"))

# ── REGION POLYGON DEFINITIONS ────────────────────────────────────────────────
DATA_REGIONS = {
    "Arabian Sea": {
        "lons":   [55, 75, 75, 65, 60, 55, 55],
        "lats":   [25, 25,  8,  8,  8, 15, 25],
        "color":  "rgba(14, 165, 233, 0.35)",
        "border": "#0EA5E9",
    },
    "Bay of Bengal": {
        "lons":   [80, 95, 95, 80, 80],
        "lats":   [22, 22,  8,  8, 22],
        "color":  "rgba(16, 185, 129, 0.35)",
        "border": "#10B981",
    }
}

# Other oceans — invisible hover points, name only
OTHER_OCEANS = [
    {"name": "Pacific Ocean",     "lat":   0.0, "lon": -150.0},
    {"name": "Atlantic Ocean",    "lat":   0.0, "lon":  -30.0},
    {"name": "Southern Ocean",    "lat": -60.0, "lon":    0.0},
    {"name": "Arctic Ocean",      "lat":  85.0, "lon":    0.0},
    {"name": "Mediterranean Sea", "lat":  35.0, "lon":   18.0},
    {"name": "Red Sea",           "lat":  20.0, "lon":   38.0},
    {"name": "Persian Gulf",      "lat":  26.0, "lon":   52.0},
    {"name": "Indian Ocean",      "lat": -20.0, "lon":   80.0},
]

DEPTH_ORDER = ["Surface", "Epipelagic", "Mesopelagic", "Bathypelagic"]

VARIABLES = {
    "Temperature": {"col": "avg_temp_celsius",  "unit": "°C",      "colorscale": "RdYlBu_r"},
    "Salinity":    {"col": "avg_salinity_psu",  "unit": "PSU",     "colorscale": "Blues"},
    "Oxygen":      {"col": "avg_doxy_umol_kg",  "unit": "µmol/kg", "colorscale": "Viridis"},
    "Chlorophyll": {"col": "avg_chla_mg_m3",    "unit": "mg/m³",   "colorscale": "Greens"},
}

# ── DATA FETCHING ─────────────────────────────────────────────────────────────

def fetch_latest_data() -> pd.DataFrame:
    query = """
        SELECT region_name, depth_zone, year, month,
               avg_temp_celsius, avg_salinity_psu,
               avg_doxy_umol_kg, avg_chla_mg_m3
        FROM argo_indian_ocean
        WHERE (year, month) = (
            SELECT year, month
            FROM argo_indian_ocean
            ORDER BY year DESC, month DESC
            LIMIT 1
        )
        ORDER BY region_name, depth_zone;
    """
    try:
        return pd.read_sql(query, _engine)
    except Exception as e:
        print(f"   [Map] Data fetch failed: {e}")
        return pd.DataFrame()

# ── HOVER CARD BUILDER ────────────────────────────────────────────────────────

def build_hover_card(region: str, df: pd.DataFrame, variable_col: str) -> str:
    region_df = df[df["region_name"] == region].copy()
    if region_df.empty:
        return f"<b>{region}</b><br>No data available"

    year       = int(region_df["year"].iloc[0])
    month      = int(region_df["month"].iloc[0])
    month_name = pd.Timestamp(year=year, month=month, day=1).strftime("%B %Y")
    unit       = next((v["unit"] for v in VARIABLES.values() if v["col"] == variable_col), "")
    var_name   = next((k for k, v in VARIABLES.items() if v["col"] == variable_col), variable_col)

    lines = [
        f"<b>{region}</b>",
        f"<span style='color:#94A3B8'>Latest data: {month_name}</span>",
        "─────────────────────────",
        f"<b>{var_name} by Depth</b>",
    ]

    for depth in DEPTH_ORDER:
        row = region_df[region_df["depth_zone"] == depth]
        if row.empty:
            continue
        val     = row[variable_col].iloc[0]
        val_str = f"<b>{val:.3f} {unit}</b>" if not pd.isna(val) else "<i>N/A</i>"
        lines.append(f"  {depth:<15} {val_str}")

    # Surface snapshot of all other variables
    surface_row = region_df[region_df["depth_zone"] == "Surface"]
    if not surface_row.empty:
        lines.append("─────────────────────────")
        lines.append("<b>Surface — All Variables</b>")
        for vname, vmeta in VARIABLES.items():
            if vmeta["col"] == variable_col:
                continue
            val = surface_row[vmeta["col"]].iloc[0]
            if not pd.isna(val):
                lines.append(f"  {vname:<15} <b>{val:.3f} {vmeta['unit']}</b>")

    return "<br>".join(lines)

# ── MAP BUILDER ───────────────────────────────────────────────────────────────

def build_ocean_map(variable_name: str = "Temperature") -> go.Figure:
    var_meta     = VARIABLES[variable_name]
    variable_col = var_meta["col"]

    df  = fetch_latest_data()
    fig = go.Figure()

    # ── 1. Filled polygon for each data region ────────────────────────────────
    for region, meta in DATA_REGIONS.items():
        hover_text = build_hover_card(region, df, variable_col) \
                     if not df.empty else f"<b>{region}</b><br>No data available"

        hover_style = dict(
            bgcolor     = "#0F172A",
            bordercolor = meta["border"],
            font        = dict(color="white", size=12, family="monospace"),
            align       = "left",
            namelength  = 0,
        )

        # Polygon fill
        fig.add_trace(go.Scattergeo(
            lon        = meta["lons"],
            lat        = meta["lats"],
            mode       = "lines",
            fill       = "toself",
            fillcolor  = meta["color"],
            line       = dict(color=meta["border"], width=1.8),
            hoverinfo  = "text",
            hovertext  = hover_text,
            hoverlabel = hover_style,
            name       = region,
            showlegend = True,
        ))

        # Invisible center point — easier to hover on mobile/small screens
        center_lat = (max(meta["lats"]) + min(meta["lats"])) / 2
        center_lon = (max(meta["lons"]) + min(meta["lons"])) / 2

        fig.add_trace(go.Scattergeo(
            lon        = [center_lon],
            lat        = [center_lat],
            mode       = "markers",
            marker     = dict(size=30, opacity=0),   # large invisible target
            hoverinfo  = "text",
            hovertext  = hover_text,
            hoverlabel = hover_style,
            showlegend = False,
            name       = "",
        ))

    # ── 2. Other oceans — invisible points, name only ─────────────────────────
    fig.add_trace(go.Scattergeo(
        lon       = [o["lon"] for o in OTHER_OCEANS],
        lat       = [o["lat"] for o in OTHER_OCEANS],
        mode      = "markers",
        marker    = dict(size=40, opacity=0),        # large invisible hover target
        hoverinfo = "text",
        hovertext = [
            f"<b>{o['name']}</b><br>"
            f"<span style='color:#64748B'>No data available yet</span>"
            for o in OTHER_OCEANS
        ],
        hoverlabel = dict(
            bgcolor     = "#1E293B",
            bordercolor = "#334155",
            font        = dict(color="#94A3B8", size=11),
            namelength  = 0,
        ),
        showlegend = False,
        name       = "",
    ))

    # ── 3. Map geo styling ────────────────────────────────────────────────────
    fig.update_geos(
        projection_type  = "natural earth",
        center           = dict(lat=10, lon=70),
        projection_scale = 3.5,
        showland         = True,
        landcolor        = "#1E293B",
        showocean        = True,
        oceancolor       = "#0A1628",
        showlakes        = True,
        lakecolor        = "#0A1628",
        showrivers       = False,
        showcountries    = True,
        countrycolor     = "#334155",
        countrywidth     = 0.5,
        showcoastlines   = True,
        coastlinecolor   = "#475569",
        coastlinewidth   = 0.8,
        showframe        = False,
        bgcolor          = "#0A1628",
    )

    # Date label
    date_label = ""
    if not df.empty:
        year       = int(df["year"].iloc[0])
        month      = int(df["month"].iloc[0])
        date_label = pd.Timestamp(year=year, month=month, day=1).strftime("%B %Y")

    fig.update_layout(
        title = dict(
            text    = (
                f"🌊 {variable_name} — Arabian Sea & Bay of Bengal"
                f"<br><span style='font-size:0.75em; color:#64748B'>"
                f"Hover over highlighted regions for data · Latest: {date_label}</span>"
            ),
            x       = 0.5,
            xanchor = "center",
            font    = dict(size=15, color="white", family="Arial"),
        ),
        paper_bgcolor = "#0A1628",
        plot_bgcolor  = "#0A1628",
        height        = 560,
        margin        = dict(t=80, b=10, l=10, r=10),
        legend        = dict(
            bgcolor     = "rgba(15,23,42,0.8)",
            bordercolor = "#334155",
            borderwidth = 1,
            font        = dict(color="white", size=11),
            x           = 0.01,
            y           = 0.01,
        ),
        font = dict(color="white"),
    )

    return fig