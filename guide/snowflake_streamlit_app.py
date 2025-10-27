"""
Snowflake Native Streamlit App: SHAP Map - Single File Version
Reads model and data directly from working directory.
Click on map segments to see SHAP explanations.
"""

import streamlit as st
import json
import io
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import mapping
from xgboost import XGBClassifier
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    layout="wide",
    page_title="SHAP Map Analysis",
    page_icon="🗺️"
)

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_FILE = "xgboost_model_v3.json"
DATA_FILE = "ocp_wiredown_model_data.gpkg"
ID_COLUMN = "ID"

# ============================================================================
# COLOR UTILITIES
# ============================================================================
def get_color_for_pof(pof_value: float, pof_min: float, pof_max: float) -> str:
    """Map POF value to hex color"""
    if pof_value is None or pof_min >= pof_max:
        return "#808080"
    norm = colors.Normalize(vmin=pof_min, vmax=pof_max)
    cmap = plt.get_cmap('RdYlBu_r')
    rgba_color = cmap(norm(pof_value))
    return colors.to_hex(rgba_color)

def generate_colorbar(pof_min: float, pof_max: float) -> bytes:
    """Generate colorbar PNG for POF range"""
    fig, ax = plt.subplots(figsize=(6, 0.8), dpi=100)
    cmap = plt.get_cmap("RdYlBu_r")
    norm = colors.Normalize(vmin=pof_min, vmax=pof_max)
    gradient = np.linspace(pof_min, pof_max, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=cmap, norm=norm, 
              extent=[pof_min, pof_max, 0, 1])
    ax.set_xlim(pof_min, pof_max)
    ax.set_yticks([])
    ax.set_xlabel("POF (Probability of Failure)", fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ============================================================================
# SHAP VISUALIZATION
# ============================================================================
def generate_shap_plot(
    asset_id: str,
    gdf: gpd.GeoDataFrame,
    model: XGBClassifier,
    explainer: shap.TreeExplainer,
    feature_names: list,
    top_k: int = 20
) -> bytes:
    """Generate SHAP waterfall plot for an asset"""
    sel = gdf[gdf[ID_COLUMN].astype(str) == str(asset_id)]
    
    if sel.empty:
        raise ValueError(f"Asset {asset_id} not found")
    
    pof_value = float(sel.iloc[0]["POF"])
    X_row = sel.iloc[0:1][feature_names].astype(float)
    explanation = explainer(X_row)
    
    single_explanation = shap.Explanation(
        values=explanation.values[0],
        base_values=explanation.base_values[0],
        data=explanation.data[0],
        feature_names=feature_names,
    )
    
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(single_explanation, max_display=int(top_k), show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP Waterfall - {ID_COLUMN}: {asset_id} - POF: {pof_value:.4f}", 
                 fontsize=12)
    
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================
st.title("🗺️ SHAP Map Analysis - OCP Wiredown Model")

with st.spinner("Loading data and model..."):
    try:
        # Load model
        model = XGBClassifier()
        model.load_model(MODEL_FILE)
        
        # Get feature names from model
        feature_names = list(model.feature_names_in_)
        
        # Load data
        gdf = gpd.read_file(DATA_FILE)
        
        # Ensure EPSG:4326
        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)
        
        # Verify features exist
        missing = [f for f in feature_names if f not in gdf.columns]
        if missing:
            st.error(f"Missing features in data: {missing}")
            st.stop()
        
        # Compute POF
        X_all = gdf[feature_names].astype(float)
        gdf["POF"] = model.predict_proba(X_all)[:, 1].astype(float)
        
        pof_min = float(gdf["POF"].min())
        pof_max = float(gdf["POF"].max())
        
        # Create SHAP explainer
        sample_size = min(len(gdf), 1000)
        explainer = shap.TreeExplainer(
            model,
            data=gdf[feature_names].iloc[:sample_size],
            model_output="probability"
        )
        
        st.success(f"✅ Loaded {len(gdf):,} segments with POF range [{pof_min:.4f}, {pof_max:.4f}]")
        st.info(f"📊 Model has {len(feature_names)} features")
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.title("⚙️ Settings")

# POF filter
st.sidebar.markdown("### 📈 POF Filter")
pof_filter = st.sidebar.slider(
    "Select POF range",
    min_value=pof_min,
    max_value=pof_max,
    value=(pof_min, pof_max),
    step=0.001,
    format="%.3f"
)

# Display limit
max_display = st.sidebar.number_input(
    "Max segments to display",
    min_value=100,
    max_value=10000,
    value=2000,
    step=100
)

# Top-K SHAP features
top_k = st.sidebar.number_input(
    "Top-K SHAP features",
    min_value=5,
    max_value=50,
    value=20,
    step=1
)

# ============================================================================
# FILTER DATA
# ============================================================================
# Apply POF filter
filtered_gdf = gdf[
    (gdf["POF"] >= pof_filter[0]) & 
    (gdf["POF"] <= pof_filter[1])
].copy()

# Limit display count
if len(filtered_gdf) > max_display:
    filtered_gdf = filtered_gdf.nlargest(max_display, "POF")

st.info(f"📊 Displaying {len(filtered_gdf):,} segments (filtered from {len(gdf):,} total)")

# ============================================================================
# PREPARE GEOJSON FOR MAP
# ============================================================================
features = []
for _, row in filtered_gdf.iterrows():
    if row.geometry is None:
        continue
    
    pof_val = float(row["POF"])
    
    features.append({
        "type": "Feature",
        "geometry": mapping(row.geometry),
        "properties": {
            "id": str(row[ID_COLUMN]),
            "POF": pof_val,
            "color": get_color_for_pof(pof_val, pof_filter[0], pof_filter[1])
        }
    })

geojson = {
    "type": "FeatureCollection",
    "features": features
}

# Calculate center
bounds = filtered_gdf.total_bounds
center_lat = (bounds[1] + bounds[3]) / 2
center_lng = (bounds[0] + bounds[2]) / 2

# ============================================================================
# MAP HTML
# ============================================================================
MAP_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SHAP Map</title>
    <link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet">
    <style>
        body { margin:0; padding:0; }
        #map { position: absolute; top:0; bottom:0; width:100%; height:600px; }
        .maplibregl-popup-content { padding: 10px; }
    </style>
</head>
<body>
<div id="map"></div>
<script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
<script>
const GEOJSON_DATA = %%GEOJSON_DATA%%;
const CENTER = %%CENTER%%;

const map = new maplibregl.Map({
    container: 'map',
    style: {
        version: 8,
        sources: {
            'osm': {
                type: 'raster',
                tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
                tileSize: 256,
                attribution: '© OpenStreetMap'
            }
        },
        layers: [{ id: 'osm', type: 'raster', source: 'osm' }]
    },
    center: CENTER,
    zoom: 12
});

map.addControl(new maplibregl.NavigationControl());

let popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false });

map.on('load', function() {
    map.addSource('segments', {
        type: 'geojson',
        data: GEOJSON_DATA
    });

    // Line layer
    map.addLayer({
        id: 'segment-lines',
        type: 'line',
        source: 'segments',
        filter: ['==', ['geometry-type'], 'LineString'],
        paint: {
            'line-color': ['get', 'color'],
            'line-width': 4,
            'line-opacity': 0.8
        }
    });

    // Point layer
    map.addLayer({
        id: 'segment-points',
        type: 'circle',
        source: 'segments',
        filter: ['==', ['geometry-type'], 'Point'],
        paint: {
            'circle-color': ['get', 'color'],
            'circle-radius': 6,
            'circle-stroke-width': 2,
            'circle-stroke-color': '#ffffff'
        }
    });

    // Polygon layer
    map.addLayer({
        id: 'segment-polygons',
        type: 'fill',
        source: 'segments',
        filter: ['==', ['geometry-type'], 'Polygon'],
        paint: {
            'fill-color': ['get', 'color'],
            'fill-opacity': 0.7
        }
    });

    const layers = ['segment-lines', 'segment-points', 'segment-polygons'];
    
    layers.forEach(layerId => {
        map.on('mouseenter', layerId, function(e) {
            map.getCanvas().style.cursor = 'pointer';
            const f = e.features[0];
            const id = f.properties.id;
            const pof = Number(f.properties.POF).toFixed(4);
            
            popup.setLngLat(e.lngLat)
                 .setHTML(`<b>ID:</b> ${id}<br><b>POF:</b> ${pof}<br><i>Click for SHAP</i>`)
                 .addTo(map);
        });
        
        map.on('mouseleave', layerId, function() {
            map.getCanvas().style.cursor = '';
            popup.remove();
        });
        
        map.on('click', layerId, function(e) {
            const id = e.features[0].properties.id;
            window.parent.postMessage({ type: 'segment_clicked', segment_id: id }, '*');
        });
    });
});
</script>
</body>
</html>
"""

# Render map
map_html = (
    MAP_HTML
    .replace("%%GEOJSON_DATA%%", json.dumps(geojson))
    .replace("%%CENTER%%", f"[{center_lng}, {center_lat}]")
)

st.components.v1.html(map_html, height=650, scrolling=False)

# Display colorbar
st.markdown("### 🎨 POF Color Scale")
colorbar_png = generate_colorbar(pof_filter[0], pof_filter[1])
st.image(colorbar_png, use_container_width=False)

# ============================================================================
# SHAP ANALYSIS SECTION
# ============================================================================
st.markdown("---")
st.markdown("### 🔬 SHAP Analysis")

# Segment selector
selected_segment = st.selectbox(
    "Select segment for SHAP analysis",
    options=[""] + sorted(filtered_gdf[ID_COLUMN].astype(str).tolist()),
    help="Choose a segment to see its SHAP explanation"
)

if selected_segment:
    with st.spinner(f"Generating SHAP explanation for segment {selected_segment}..."):
        try:
            shap_png = generate_shap_plot(
                asset_id=selected_segment,
                gdf=gdf,
                model=model,
                explainer=explainer,
                feature_names=feature_names,
                top_k=top_k
            )
            st.image(shap_png, use_container_width=True)
            
            # Show segment details
            segment_data = gdf[gdf[ID_COLUMN].astype(str) == selected_segment].iloc[0]
            st.markdown("#### 📊 Segment Details")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Segment ID", selected_segment)
                st.metric("POF", f"{segment_data['POF']:.4f}")
            with col2:
                geom_type = segment_data.geometry.geom_type
                st.metric("Geometry Type", geom_type)
        
        except Exception as e:
            st.error(f"❌ Error generating SHAP plot: {e}")

# ============================================================================
# INSTRUCTIONS
# ============================================================================
with st.expander("📖 How to Use"):
    st.markdown("""
    ### Instructions
    
    1. **Explore the Map**: Pan and zoom to explore your segments
    2. **Hover**: Hover over segments to see ID and POF values
    3. **Click**: Click on any segment to generate its SHAP explanation
    4. **Filter**: Use the sidebar to filter segments by POF range
    5. **Select**: Use the dropdown below the map to analyze specific segments
    
    ### Understanding SHAP Plots
    
    - **Red bars**: Features that increase the probability of failure
    - **Blue bars**: Features that decrease the probability of failure
    - **Bar length**: Shows the magnitude of each feature's impact
    - **Base value**: The average prediction across all data
    - **Output value**: The final prediction for this specific segment
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>🚀 Snowflake Native Streamlit App • XGBoost + SHAP + Geospatial ML</p>
</div>
""", unsafe_allow_html=True)
