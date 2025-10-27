import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(layout="wide", page_title="SHAP Map (Folium Version)", page_icon="🗺️")

DATA_FILE = "ocp_wiredown_model_data.gpkg"
ID_COLUMN = "ID"

# ============================================================================
# LOAD DATA
# ============================================================================
st.title("🗺️ SHAP Map Viewer (Folium Version)")

with st.spinner("Loading geospatial data..."):
    gdf = gpd.read_file(DATA_FILE)

    # Ensure CRS
    if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    st.success(f"Loaded {len(gdf):,} segments")

# ============================================================================
# FILTERS
# ============================================================================
st.sidebar.title("⚙️ Filters")

if "POF" in gdf.columns:
    pof_min, pof_max = gdf["POF"].min(), gdf["POF"].max()
    pof_range = st.sidebar.slider(
        "POF Range",
        min_value=float(pof_min),
        max_value=float(pof_max),
        value=(float(pof_min), float(pof_max)),
        step=0.001,
        format="%.3f"
    )
    gdf = gdf[(gdf["POF"] >= pof_range[0]) & (gdf["POF"] <= pof_range[1])]

st.sidebar.write(f"Displaying {len(gdf):,} segments")

# ============================================================================
# MAP CREATION
# ============================================================================

# Compute map center
center = [gdf.geometry.unary_union.centroid.y, gdf.geometry.unary_union.centroid.x]

# Option A: normal OSM tiles (works in most Streamlit setups)
use_tiles = st.sidebar.checkbox("Use OpenStreetMap tiles", value=True)

if use_tiles:
    m = folium.Map(location=center, zoom_start=10, tiles="OpenStreetMap")
else:
    # Option B: completely CSP-safe gray background (no network requests)
    m = folium.Map(location=center, zoom_start=10, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles=None,
        name="Gray background",
        attr="",
        overlay=False
    ).add_to(m)
    m.get_root().html.add_child(folium.Element("""
        <style>
        .leaflet-container {
            background: #d3d3d3;
        }
        </style>
    """))

# Add GeoJSON overlay for your geometries
folium.GeoJson(
    data=gdf.to_json(),
    name="Segments",
    tooltip=folium.GeoJsonTooltip(
        fields=[ID_COLUMN, "POF"] if "POF" in gdf.columns else [ID_COLUMN],
        aliases=["ID", "POF"],
        localize=True
    ),
    style_function=lambda feature: {
        "color": feature["properties"].get("color", "red"),
        "weight": 3,
        "opacity": 0.9,
        "fillColor": feature["properties"].get("color", "red"),
        "fillOpacity": 0.6,
    },
    highlight_function=lambda x: {"weight": 5, "color": "yellow"},
).add_to(m)

folium.LayerControl().add_to(m)

# ============================================================================
# DISPLAY MAP
# ============================================================================
st_folium(m, width=1200, height=700)
