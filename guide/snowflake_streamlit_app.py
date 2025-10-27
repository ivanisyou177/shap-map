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
import base64

# Read raw files
with open(r"maplibre-gl.js", "rb") as f:
    js_code = f.read().decode("utf-8", errors="ignore")

with open(r"maplibre-gl.css", "rb") as f:
    css_code = f.read().decode("utf-8", errors="ignore")


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

# CC Status filter
st.sidebar.markdown("### 🔌 CC Status Filter")
cc_status_options = {
    "both": "Both (All)",
    0: "No CC (0)",
    1: "With CC (1)"
}
if "CC Status" in gdf.columns:
    cc_counts = gdf["CC Status"].value_counts().to_dict()
    st.sidebar.caption(f"Distribution: 0={cc_counts.get(0, 0):,}, 1={cc_counts.get(1, 0):,}")
    cc_status_filter = st.sidebar.selectbox(
        "Filter by CC Status",
        options=list(cc_status_options.keys()),
        format_func=lambda x: cc_status_options[x]
    )
else:
    cc_status_filter = "both"
    st.sidebar.caption("⚠️ CC Status column not found")

# Conductor Age filter
st.sidebar.markdown("### 📅 Conductor Age Filter")
if "CALCULATED_CONDUCTOR_AGE" in gdf.columns:
    age_data = gdf["CALCULATED_CONDUCTOR_AGE"].dropna()
    if len(age_data) > 0:
        age_min = float(age_data.min())
        age_max = float(age_data.max())
        age_filter = st.sidebar.slider(
            "Select age range (years)",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max),
            step=max(1.0, (age_max - age_min) / 100),
            format="%.1f"
        )
    else:
        age_filter = None
        st.sidebar.caption("⚠️ No age data available")
else:
    age_filter = None
    st.sidebar.caption("⚠️ Age column not found")

# Conductor Length filter
st.sidebar.markdown("### 📏 Conductor Length Filter")
if "CONDUCTOR_LENGTH_UDF" in gdf.columns:
    length_data = gdf["CONDUCTOR_LENGTH_UDF"].dropna()
    if len(length_data) > 0:
        length_min = float(length_data.min())
        length_max = float(length_data.max())
        length_filter = st.sidebar.slider(
            "Select length range",
            min_value=length_min,
            max_value=length_max,
            value=(length_min, length_max),
            step=max(0.1, (length_max - length_min) / 100),
            format="%.2f"
        )
    else:
        length_filter = None
        st.sidebar.caption("⚠️ No length data available")
else:
    length_filter = None
    st.sidebar.caption("⚠️ Length column not found")

# Conductor Diameter filter
st.sidebar.markdown("### ⭕ Conductor Diameter Filter")
if "CONDUCTOR_DIAMETER_UDF" in gdf.columns:
    diameter_data = gdf["CONDUCTOR_DIAMETER_UDF"].dropna()
    if len(diameter_data) > 0:
        diameter_min = float(diameter_data.min())
        diameter_max = float(diameter_data.max())
        diameter_filter = st.sidebar.slider(
            "Select diameter range",
            min_value=diameter_min,
            max_value=diameter_max,
            value=(diameter_min, diameter_max),
            step=max(0.001, (diameter_max - diameter_min) / 100),
            format="%.4f"
        )
    else:
        diameter_filter = None
        st.sidebar.caption("⚠️ No diameter data available")
else:
    diameter_filter = None
    st.sidebar.caption("⚠️ Diameter column not found")

st.sidebar.markdown("---")

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
# Start with full dataset
filtered_gdf = gdf.copy()

# Apply POF filter
filtered_gdf = filtered_gdf[
    (filtered_gdf["POF"] >= pof_filter[0]) & 
    (filtered_gdf["POF"] <= pof_filter[1])
]

# Apply CC Status filter
if cc_status_filter != "both" and "CC Status" in gdf.columns:
    filtered_gdf = filtered_gdf[filtered_gdf["CC Status"] == cc_status_filter]

# Apply Age filter
if age_filter is not None and "CALCULATED_CONDUCTOR_AGE" in gdf.columns:
    filtered_gdf = filtered_gdf[
        (filtered_gdf["CALCULATED_CONDUCTOR_AGE"] >= age_filter[0]) &
        (filtered_gdf["CALCULATED_CONDUCTOR_AGE"] <= age_filter[1])
    ]

# Apply Length filter
if length_filter is not None and "CONDUCTOR_LENGTH_UDF" in gdf.columns:
    filtered_gdf = filtered_gdf[
        (filtered_gdf["CONDUCTOR_LENGTH_UDF"] >= length_filter[0]) &
        (filtered_gdf["CONDUCTOR_LENGTH_UDF"] <= length_filter[1])
    ]

# Apply Diameter filter
if diameter_filter is not None and "CONDUCTOR_DIAMETER_UDF" in gdf.columns:
    filtered_gdf = filtered_gdf[
        (filtered_gdf["CONDUCTOR_DIAMETER_UDF"] >= diameter_filter[0]) &
        (filtered_gdf["CONDUCTOR_DIAMETER_UDF"] <= diameter_filter[1])
    ]

# Limit display count
if len(filtered_gdf) > max_display:
    filtered_gdf = filtered_gdf.nlargest(max_display, "POF")

st.info(f"📊 Displaying {len(filtered_gdf):,} segments (filtered from {len(gdf):,} total)")

if len(filtered_gdf) == 0:
    st.warning("⚠️ No segments match the current filters. Try adjusting the filter ranges.")
    st.stop()

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
if len(filtered_gdf) > 0:
    bounds = filtered_gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lng = (bounds[0] + bounds[2]) / 2
else:
    center_lat = 0
    center_lng = 0

st.info(f"🗺️ Map center: [{center_lat:.6f}, {center_lng:.6f}] | Features on map: {len(features):,}")

# ============================================================================
# MAP HTML - COMPACT VERSION WITH MAPLIBRE GL
# ============================================================================
st.markdown("### 🗺️ Interactive Map")

# Create compact GeoJSON string
geojson_str = json.dumps(geojson, separators=(',', ':'))

MAP_HTML = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<style>
  html, body {{
    height: 100%;
    width: 100%;
    margin: 0;
    padding: 0;
    overflow: hidden;
  }}
  #map {{
    position: absolute;
    top: 0;
    bottom: 0;
    width: 100%;
    height: 700px;
  }}
  .maplibregl-canvas {{
    outline: none;
  }}
</style>
</head>
<body>
<div id="map"></div>

<!-- Inline MapLibre JS -->
<script>
{js_code}
</script>

<script>
(function() {{
  const centerLat = {center_lat};
  const centerLng = {center_lng};
  const geojson = {geojson_str};

  // Initialize map with solid gray background
  const map = new maplibregl.Map({{
    container: 'map',
    style: {{
      version: 8,
      sources: {{}},
      layers: [
        {{
          id: "background",
          type: "background",
          paint: {{
            "background-color": "#d3d3d3"
          }}
        }}
      ]
    }},
    center: [centerLng, centerLat],
    zoom: 10
  }});

  map.addControl(new maplibregl.NavigationControl());

  map.on('load', () => {{
    // Add your GeoJSON data
    map.addSource('assets', {{
      type: 'geojson',
      data: geojson
    }});

    // Lines
    map.addLayer({{
      id: 'lines',
      type: 'line',
      source: 'assets',
      filter: ['in', ['geometry-type'], ['literal', ['LineString', 'MultiLineString']]],
      paint: {{
        'line-color': ['get', 'color'],
        'line-width': 3,
        'line-opacity': 0.9
      }}
    }});

    // Points
    map.addLayer({{
      id: 'points',
      type: 'circle',
      source: 'assets',
      filter: ['==', ['geometry-type'], 'Point'],
      paint: {{
        'circle-color': ['get', 'color'],
        'circle-radius': 6,
        'circle-stroke-width': 1,
        'circle-stroke-color': '#000'
      }}
    }});

    // Polygons
    map.addLayer({{
      id: 'polygons',
      type: 'fill',
      source: 'assets',
      filter: ['==', ['geometry-type'], 'Polygon'],
      paint: {{
        'fill-color': ['get', 'color'],
        'fill-opacity': 0.6
      }}
    }});

    // Polygon outlines
    map.addLayer({{
      id: 'polygon-outline',
      type: 'line',
      source: 'assets',
      filter: ['==', ['geometry-type'], 'Polygon'],
      paint: {{
        'line-color': '#333',
        'line-width': 2
      }}
    }});

    // Automatically fit bounds to your data
    const features = geojson.features || [];
    if (features.length > 0) {{
      const bounds = new maplibregl.LngLatBounds();
      features.forEach(f => {{
        const g = f.geometry;
        if (!g) return;
        if (g.type === 'Point') bounds.extend(g.coordinates);
        else if (g.type === 'LineString') g.coordinates.forEach(c => bounds.extend(c));
        else if (g.type === 'MultiLineString') g.coordinates.forEach(l => l.forEach(c => bounds.extend(c)));
        else if (g.type === 'Polygon') g.coordinates[0].forEach(c => bounds.extend(c));
      }});
      if (!bounds.isEmpty()) map.fitBounds(bounds, {{ padding: 50, maxZoom: 14 }});
    }}
  }});
}})();
</script>
</body>
</html>"""

st.components.v1.html(MAP_HTML, height=700, scrolling=False)

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
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Segment ID", selected_segment)
                st.metric("POF", f"{segment_data['POF']:.4f}")
            with col2:
                geom_type = segment_data.geometry.geom_type
                st.metric("Geometry Type", geom_type)
                if "CC Status" in segment_data.index:
                    cc_val = int(segment_data["CC Status"])
                    st.metric("CC Status", f"{cc_val} ({'With CC' if cc_val == 1 else 'No CC'})")
            with col3:
                if "CALCULATED_CONDUCTOR_AGE" in segment_data.index:
                    st.metric("Conductor Age", f"{segment_data['CALCULATED_CONDUCTOR_AGE']:.1f} years")
                if "CONDUCTOR_LENGTH_UDF" in segment_data.index:
                    st.metric("Conductor Length", f"{segment_data['CONDUCTOR_LENGTH_UDF']:.2f}")
                if "CONDUCTOR_DIAMETER_UDF" in segment_data.index:
                    st.metric("Conductor Diameter", f"{segment_data['CONDUCTOR_DIAMETER_UDF']:.4f}")
        
        except Exception as e:
            st.error(f"❌ Error generating SHAP plot: {e}")

# ============================================================================
# FILTER SUMMARY
# ============================================================================
st.markdown("---")
st.markdown("### 📋 Active Filters Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**POF Range**")
    st.write(f"{pof_filter[0]:.3f} - {pof_filter[1]:.3f}")

with col2:
    st.markdown("**CC Status**")
    st.write(cc_status_options[cc_status_filter])

with col3:
    st.markdown("**Conductor Age**")
    if age_filter:
        st.write(f"{age_filter[0]:.1f} - {age_filter[1]:.1f} years")
    else:
        st.write("N/A")

with col4:
    st.markdown("**Results**")
    st.write(f"{len(filtered_gdf):,} / {len(gdf):,} segments")

# ============================================================================
# INSTRUCTIONS
# ============================================================================
with st.expander("📖 How to Use"):
    st.markdown("""
    ### Instructions
    
    1. **Explore the Map**: Pan and zoom to explore your segments
    2. **Hover**: Hover over segments to see ID and POF values
    3. **Click**: Click on any segment to generate its SHAP explanation
    4. **Filter**: Use the sidebar filters to narrow down segments:
       - **POF Range**: Filter by probability of failure
       - **CC Status**: Filter by critical component status (0=No CC, 1=With CC)
       - **Conductor Age**: Filter by age in years
       - **Conductor Length**: Filter by physical length
       - **Conductor Diameter**: Filter by diameter dimension
    5. **Select**: Use the dropdown below the map to analyze specific segments
    
    ### Understanding SHAP Plots
    
    - **Red bars**: Features that increase the probability of failure
    - **Blue bars**: Features that decrease the probability of failure
    - **Bar length**: Shows the magnitude of each feature's impact
    - **Base value**: The average prediction across all data
    - **Output value**: The final prediction for this specific segment
    
    ### Color Coding
    
    - **Red**: High probability of failure
    - **Yellow**: Medium probability of failure
    - **Blue**: Low probability of failure
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>🚀 Snowflake Native Streamlit App • XGBoost + SHAP + Geospatial ML</p>
</div>
""", unsafe_allow_html=True)
