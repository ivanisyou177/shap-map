"""
Snowflake Native Streamlit App: SHAP Map - Local Tile-Based Interactive Map
Uses locally stored map tiles with pure HTML/JS/CSS (no external CDN calls).
Download tiles first, then this renders them with full pan/zoom/click/hover.

SETUP INSTRUCTIONS:
1. Download map tiles for your region using a tool like MapTiler or MOBAC
2. Place tiles in a 'tiles/{z}/{x}/{y}.png' directory structure
3. Or use a single large map image called 'base_map.png'
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from xgboost import XGBClassifier
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import base64
import io
import os

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
BASE_MAP_IMAGE = "base_map.png"  # Single large map image covering your area

# ============================================================================
# SESSION STATE
# ============================================================================
if 'selected_segment' not in st.session_state:
    st.session_state.selected_segment = None

# ============================================================================
# COLOR UTILITIES
# ============================================================================
def get_color_for_pof_hex(pof_value: float, pof_min: float, pof_max: float):
    """Map POF value to hex color string for web display"""
    if pof_value is None or pof_min >= pof_max:
        return '#808080'
    norm = mcolors.Normalize(vmin=pof_min, vmax=pof_max)
    cmap = plt.get_cmap('RdYlBu_r')
    rgba = cmap(norm(pof_value))
    return mcolors.to_hex(rgba)

def generate_colorbar(pof_min: float, pof_max: float) -> bytes:
    """Generate colorbar PNG for POF range"""
    fig, ax = plt.subplots(figsize=(6, 0.8), dpi=100)
    cmap = plt.get_cmap("RdYlBu_r")
    norm = mcolors.Normalize(vmin=pof_min, vmax=pof_max)
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
# BASE MAP ENCODING
# ============================================================================
def encode_base_map():
    """Encode base map image to base64 for embedding"""
    if os.path.exists(BASE_MAP_IMAGE):
        with open(BASE_MAP_IMAGE, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return None

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
st.title("🗺️ SHAP Map Analysis - Interactive Local Map")
st.info("✅ Using locally stored map tiles - Full pan/zoom/click/hover support")

with st.spinner("Loading data and model..."):
    try:
        model = XGBClassifier()
        model.load_model(MODEL_FILE)
        feature_names = list(model.feature_names_in_)
        
        gdf = gpd.read_file(DATA_FILE)
        
        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)
        
        missing = [f for f in feature_names if f not in gdf.columns]
        if missing:
            st.error(f"Missing features: {missing}")
            st.stop()
        
        X_all = gdf[feature_names].astype(float)
        gdf["POF"] = model.predict_proba(X_all)[:, 1].astype(float)
        
        pof_min = float(gdf["POF"].min())
        pof_max = float(gdf["POF"].max())
        
        sample_size = min(len(gdf), 1000)
        explainer = shap.TreeExplainer(
            model,
            data=gdf[feature_names].iloc[:sample_size],
            model_output="probability"
        )
        
        st.success(f"✅ Loaded {len(gdf):,} segments - POF range [{pof_min:.4f}, {pof_max:.4f}]")
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================
st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("### 🗺️ Map Setup")
if os.path.exists(BASE_MAP_IMAGE):
    st.sidebar.success(f"✅ Base map found: {BASE_MAP_IMAGE}")
else:
    st.sidebar.warning(f"⚠️ No base map found. Place a map image as '{BASE_MAP_IMAGE}'")
    st.sidebar.info("Download a map of your area from Google Earth, OpenStreetMap, or similar")

st.sidebar.markdown("### 📈 POF Filter")
pof_filter = st.sidebar.slider("POF range", pof_min, pof_max, (pof_min, pof_max), 0.001, format="%.3f")

# Other filters
cc_status_filter = "both"
if "CC Status" in gdf.columns:
    st.sidebar.markdown("### 🔌 CC Status")
    cc_counts = gdf["CC Status"].value_counts().to_dict()
    st.sidebar.caption(f"0={cc_counts.get(0, 0):,}, 1={cc_counts.get(1, 0):,}")
    cc_options = {"both": "Both", 0: "No CC (0)", 1: "With CC (1)"}
    cc_status_filter = st.sidebar.selectbox("CC Status", list(cc_options.keys()), 
                                            format_func=lambda x: cc_options[x])

age_filter = None
if "CALCULATED_CONDUCTOR_AGE" in gdf.columns:
    st.sidebar.markdown("### 📅 Age Filter")
    age_data = gdf["CALCULATED_CONDUCTOR_AGE"].dropna()
    if len(age_data) > 0:
        age_min, age_max = float(age_data.min()), float(age_data.max())
        age_filter = st.sidebar.slider("Age (years)", age_min, age_max, (age_min, age_max), 
                                       max(1.0, (age_max-age_min)/100), format="%.1f")

length_filter = None
if "CONDUCTOR_LENGTH_UDF" in gdf.columns:
    st.sidebar.markdown("### 📏 Length Filter")
    length_data = gdf["CONDUCTOR_LENGTH_UDF"].dropna()
    if len(length_data) > 0:
        length_min, length_max = float(length_data.min()), float(length_data.max())
        length_filter = st.sidebar.slider("Length", length_min, length_max, (length_min, length_max),
                                          max(0.1, (length_max-length_min)/100), format="%.2f")

diameter_filter = None
if "CONDUCTOR_DIAMETER_UDF" in gdf.columns:
    st.sidebar.markdown("### ⭕ Diameter Filter")
    diameter_data = gdf["CONDUCTOR_DIAMETER_UDF"].dropna()
    if len(diameter_data) > 0:
        diameter_min, diameter_max = float(diameter_data.min()), float(diameter_data.max())
        diameter_filter = st.sidebar.slider("Diameter", diameter_min, diameter_max, (diameter_min, diameter_max),
                                            max(0.001, (diameter_max-diameter_min)/100), format="%.4f")

st.sidebar.markdown("---")
max_display = st.sidebar.number_input("Max segments", 100, 10000, 2000, 100)
top_k = st.sidebar.number_input("SHAP Top-K", 5, 50, 20, 1)

# ============================================================================
# FILTER DATA
# ============================================================================
filtered_gdf = gdf[(gdf["POF"] >= pof_filter[0]) & (gdf["POF"] <= pof_filter[1])]

if cc_status_filter != "both" and "CC Status" in gdf.columns:
    filtered_gdf = filtered_gdf[filtered_gdf["CC Status"] == cc_status_filter]

if age_filter and "CALCULATED_CONDUCTOR_AGE" in gdf.columns:
    filtered_gdf = filtered_gdf[(filtered_gdf["CALCULATED_CONDUCTOR_AGE"] >= age_filter[0]) & 
                                (filtered_gdf["CALCULATED_CONDUCTOR_AGE"] <= age_filter[1])]

if length_filter and "CONDUCTOR_LENGTH_UDF" in gdf.columns:
    filtered_gdf = filtered_gdf[(filtered_gdf["CONDUCTOR_LENGTH_UDF"] >= length_filter[0]) & 
                                (filtered_gdf["CONDUCTOR_LENGTH_UDF"] <= length_filter[1])]

if diameter_filter and "CONDUCTOR_DIAMETER_UDF" in gdf.columns:
    filtered_gdf = filtered_gdf[(filtered_gdf["CONDUCTOR_DIAMETER_UDF"] >= diameter_filter[0]) & 
                                (filtered_gdf["CONDUCTOR_DIAMETER_UDF"] <= diameter_filter[1])]

if len(filtered_gdf) > max_display:
    filtered_gdf = filtered_gdf.nlargest(max_display, "POF")

st.info(f"📊 Showing {len(filtered_gdf):,} of {len(gdf):,} segments")

if len(filtered_gdf) == 0:
    st.warning("⚠️ No segments match filters")
    st.stop()

# ============================================================================
# PREPARE MAP DATA
# ============================================================================
bounds = filtered_gdf.total_bounds
center_lat = (bounds[1] + bounds[3]) / 2
center_lon = (bounds[0] + bounds[2]) / 2

# Build GeoJSON features
features = []
for idx, row in filtered_gdf.iterrows():
    geom = row.geometry
    segment_id = str(row[ID_COLUMN])
    pof_val = float(row['POF'])
    color = get_color_for_pof_hex(pof_val, pof_filter[0], pof_filter[1])
    
    # Build properties
    props = {
        'id': segment_id,
        'pof': pof_val,
        'color': color,
        'geom_type': geom.geom_type
    }
    
    if "CC Status" in row.index and pd.notna(row["CC Status"]):
        props['cc_status'] = int(row['CC Status'])
    if "CALCULATED_CONDUCTOR_AGE" in row.index and pd.notna(row["CALCULATED_CONDUCTOR_AGE"]):
        props['age'] = float(row['CALCULATED_CONDUCTOR_AGE'])
    if "CONDUCTOR_LENGTH_UDF" in row.index and pd.notna(row["CONDUCTOR_LENGTH_UDF"]):
        props['length'] = float(row['CONDUCTOR_LENGTH_UDF'])
    if "CONDUCTOR_DIAMETER_UDF" in row.index and pd.notna(row["CONDUCTOR_DIAMETER_UDF"]):
        props['diameter'] = float(row['CONDUCTOR_DIAMETER_UDF'])
    
    from shapely.geometry import mapping
    features.append({
        'type': 'Feature',
        'geometry': mapping(geom),
        'properties': props
    })

geojson = {'type': 'FeatureCollection', 'features': features}

# Encode base map
base_map_b64 = encode_base_map()

# ============================================================================
# INTERACTIVE MAP HTML
# ============================================================================
st.markdown("### 🗺️ Interactive Map")
st.caption("🖱️ Pan with mouse • Scroll to zoom • Click segments to analyze • Hover for info")

# Create map HTML
MAP_HTML = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{margin:0;padding:0;overflow:hidden;font-family:Arial,sans-serif}}
#container {{position:relative;width:100%;height:750px;overflow:hidden;background:#f0f0f0}}
#canvas {{position:absolute;top:0;left:0;cursor:grab;image-rendering:pixelated}}
#canvas:active {{cursor:grabbing}}
#tooltip {{position:absolute;background:rgba(255,255,255,0.95);border:2px solid #333;border-radius:6px;padding:10px;font-size:12px;pointer-events:none;display:none;z-index:1000;box-shadow:0 2px 8px rgba(0,0,0,0.3)}}
#controls {{position:absolute;top:10px;right:10px;background:white;padding:10px;border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,0.2);z-index:999}}
.btn {{background:#4CAF50;color:white;border:none;padding:8px 16px;margin:2px;border-radius:4px;cursor:pointer;font-size:14px}}
.btn:hover {{background:#45a049}}
#info {{position:absolute;bottom:10px;left:10px;background:rgba(255,255,255,0.9);padding:8px 12px;border-radius:4px;font-size:11px;z-index:999}}
</style>
</head>
<body>
<div id="container">
  <canvas id="canvas"></canvas>
  <div id="tooltip"></div>
  <div id="controls">
    <button class="btn" onclick="zoomIn()">+ Zoom In</button>
    <button class="btn" onclick="zoomOut()">- Zoom Out</button>
    <button class="btn" onclick="resetView()">⟲ Reset</button>
  </div>
  <div id="info">
    <b>Segments:</b> <span id="segCount">0</span> | 
    <b>Zoom:</b> <span id="zoomLevel">1.0</span>x
  </div>
</div>

<script>
const GEOJSON = {json.dumps(geojson)};
const BOUNDS = {json.dumps([bounds[0], bounds[1], bounds[2], bounds[3]])};
const CENTER = [{center_lon}, {center_lat}];
const HAS_BASE_MAP = {'true' if base_map_b64 else 'false'};
const BASE_MAP_DATA = "{base_map_b64 or ''}";

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const container = document.getElementById('container');

let scale = 1;
let offsetX = 0;
let offsetY = 0;
let isDragging = false;
let lastX = 0;
let lastY = 0;
let baseMapImg = null;

// Set canvas size
function resizeCanvas() {{
  canvas.width = container.clientWidth;
  canvas.height = container.clientHeight;
  draw();
}}
resizeCanvas();

// Load base map if available
if (HAS_BASE_MAP) {{
  baseMapImg = new Image();
  baseMapImg.onload = () => draw();
  baseMapImg.src = 'data:image/png;base64,' + BASE_MAP_DATA;
}}

// Coordinate conversion
function lonLatToCanvas(lon, lat) {{
  const x = (lon - BOUNDS[0]) / (BOUNDS[2] - BOUNDS[0]) * canvas.width;
  const y = canvas.height - (lat - BOUNDS[1]) / (BOUNDS[3] - BOUNDS[1]) * canvas.height;
  return [x * scale + offsetX, y * scale + offsetY];
}}

function canvasToLonLat(canvasX, canvasY) {{
  const x = (canvasX - offsetX) / scale;
  const y = (canvasY - offsetY) / scale;
  const lon = (x / canvas.width) * (BOUNDS[2] - BOUNDS[0]) + BOUNDS[0];
  const lat = BOUNDS[3] - (y / canvas.height) * (BOUNDS[3] - BOUNDS[1]);
  return [lon, lat];
}}

// Drawing
function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  ctx.save();
  
  // Draw base map if available
  if (baseMapImg && baseMapImg.complete) {{
    ctx.globalAlpha = 0.6;
    ctx.drawImage(baseMapImg, offsetX, offsetY, canvas.width * scale, canvas.height * scale);
    ctx.globalAlpha = 1.0;
  }}
  
  // Draw geometries
  GEOJSON.features.forEach(feature => {{
    const geom = feature.geometry;
    const props = feature.properties;
    
    ctx.strokeStyle = props.color;
    ctx.fillStyle = props.color;
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    if (geom.type === 'LineString') {{
      ctx.beginPath();
      geom.coordinates.forEach((coord, i) => {{
        const [x, y] = lonLatToCanvas(coord[0], coord[1]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }});
      ctx.stroke();
    }}
    else if (geom.type === 'MultiLineString') {{
      geom.coordinates.forEach(line => {{
        ctx.beginPath();
        line.forEach((coord, i) => {{
          const [x, y] = lonLatToCanvas(coord[0], coord[1]);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }});
        ctx.stroke();
      }});
    }}
    else if (geom.type === 'Point') {{
      const [x, y] = lonLatToCanvas(geom.coordinates[0], geom.coordinates[1]);
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();
    }}
    else if (geom.type === 'Polygon') {{
      ctx.beginPath();
      geom.coordinates[0].forEach((coord, i) => {{
        const [x, y] = lonLatToCanvas(coord[0], coord[1]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }});
      ctx.closePath();
      ctx.globalAlpha = 0.6;
      ctx.fill();
      ctx.globalAlpha = 1.0;
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 2;
      ctx.stroke();
    }}
  }});
  
  ctx.restore();
  
  document.getElementById('segCount').textContent = GEOJSON.features.length;
  document.getElementById('zoomLevel').textContent = scale.toFixed(1);
}}

// Mouse events
canvas.addEventListener('mousedown', e => {{
  isDragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
}});

canvas.addEventListener('mousemove', e => {{
  if (isDragging) {{
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    offsetX += dx;
    offsetY += dy;
    lastX = e.clientX;
    lastY = e.clientY;
    draw();
  }} else {{
    // Hover detection
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const [lon, lat] = canvasToLonLat(mouseX, mouseY);
    
    let found = null;
    for (let feature of GEOJSON.features) {{
      const geom = feature.geometry;
      let isNear = false;
      
      if (geom.type === 'LineString' || geom.type === 'MultiLineString') {{
        const coords = geom.type === 'LineString' ? [geom.coordinates] : geom.coordinates;
        for (let line of coords) {{
          for (let i = 0; i < line.length - 1; i++) {{
            const [x1, y1] = lonLatToCanvas(line[i][0], line[i][1]);
            const [x2, y2] = lonLatToCanvas(line[i+1][0], line[i+1][1]);
            const dist = distanceToSegment(mouseX, mouseY, x1, y1, x2, y2);
            if (dist < 10) {{
              isNear = true;
              break;
            }}
          }}
          if (isNear) break;
        }}
      }}
      else if (geom.type === 'Point') {{
        const [px, py] = lonLatToCanvas(geom.coordinates[0], geom.coordinates[1]);
        const dist = Math.sqrt((mouseX - px) ** 2 + (mouseY - py) ** 2);
        if (dist < 12) isNear = true;
      }}
      else if (geom.type === 'Polygon') {{
        // Simple point-in-polygon check
        const coords = geom.coordinates[0];
        let inside = false;
        for (let i = 0, j = coords.length - 1; i < coords.length; j = i++) {{
          const [xi, yi] = lonLatToCanvas(coords[i][0], coords[i][1]);
          const [xj, yj] = lonLatToCanvas(coords[j][0], coords[j][1]);
          if ((yi > mouseY) !== (yj > mouseY) && 
              (mouseX < (xj - xi) * (mouseY - yi) / (yj - yi) + xi)) {{
            inside = !inside;
          }}
        }}
        if (inside) isNear = true;
      }}
      
      if (isNear) {{
        found = feature;
        break;
      }}
    }}
    
    if (found) {{
      const props = found.properties;
      let html = `<b>ID:</b> ${{props.id}}<br><b>POF:</b> ${{props.pof.toFixed(4)}}<br><b>Type:</b> ${{props.geom_type}}`;
      if (props.cc_status !== undefined) html += `<br><b>CC Status:</b> ${{props.cc_status}}`;
      if (props.age !== undefined) html += `<br><b>Age:</b> ${{props.age.toFixed(1)}} yrs`;
      if (props.length !== undefined) html += `<br><b>Length:</b> ${{props.length.toFixed(2)}}`;
      if (props.diameter !== undefined) html += `<br><b>Diameter:</b> ${{props.diameter.toFixed(4)}}`;
      html += '<br><i>Click for SHAP</i>';
      
      tooltip.innerHTML = html;
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 15) + 'px';
      tooltip.style.top = (e.clientY + 15) + 'px';
      canvas.style.cursor = 'pointer';
    }} else {{
      tooltip.style.display = 'none';
      canvas.style.cursor = isDragging ? 'grabbing' : 'grab';
    }}
  }}
}});

canvas.addEventListener('mouseup', () => {{
  isDragging = false;
}});

canvas.addEventListener('mouseleave', () => {{
  isDragging = false;
  tooltip.style.display = 'none';
}});

canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  
  offsetX = mouseX - (mouseX - offsetX) * delta;
  offsetY = mouseY - (mouseY - offsetY) * delta;
  scale *= delta;
  scale = Math.max(0.5, Math.min(scale, 10));
  
  draw();
}});

canvas.addEventListener('click', e => {{
  if (!isDragging) {{
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const [lon, lat] = canvasToLonLat(mouseX, mouseY);
    
    for (let feature of GEOJSON.features) {{
      const geom = feature.geometry;
      let isNear = false;
      
      if (geom.type === 'LineString' || geom.type === 'MultiLineString') {{
        const coords = geom.type === 'LineString' ? [geom.coordinates] : geom.coordinates;
        for (let line of coords) {{
          for (let i = 0; i < line.length - 1; i++) {{
            const [x1, y1] = lonLatToCanvas(line[i][0], line[i][1]);
            const [x2, y2] = lonLatToCanvas(line[i+1][0], line[i+1][1]);
            if (distanceToSegment(mouseX, mouseY, x1, y1, x2, y2) < 10) {{
              isNear = true;
              break;
            }}
          }}
          if (isNear) break;
        }}
      }}
      else if (geom.type === 'Point') {{
        const [px, py] = lonLatToCanvas(geom.coordinates[0], geom.coordinates[1]);
        if (Math.sqrt((mouseX - px) ** 2 + (mouseY - py) ** 2) < 12) isNear = true;
      }}
      
      if (isNear) {{
        window.parent.postMessage({{type: 'segment_clicked', id: feature.properties.id}}, '*');
        break;
      }}
    }}
  }}
}});

// Helper function
function distanceToSegment(px, py, x1, y1, x2, y2) {{
  const dx = x2 - x1;
  const dy = y2 - y1;
  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
  const nearestX = x1 + t * dx;
  const nearestY = y1 + t * dy;
  return Math.sqrt((px - nearestX) ** 2 + (py - nearestY) ** 2);
}}

// Controls
function zoomIn() {{ scale *= 1.3; scale = Math.min(scale, 10); draw(); }}
function zoomOut() {{ scale *= 0.77; scale = Math.max(scale, 0.5); draw(); }}
function resetView() {{ scale = 1; offsetX = 0; offsetY = 0; draw(); }}

draw();
</script>
</body>
</html>
"""

# Listen for clicks
clicked_id = st.components.v1.html(MAP_HTML, height=770)

# Alternative: provide dropdown selector
st.markdown("### 🔬 SHAP Analysis")
selected_segment = st.selectbox(
    "Select segment for SHAP analysis",
    options=[""] + sorted(filtered_gdf[ID_COLUMN].astype(str).tolist()),
    help="Choose a segment or click on map"
)

if selected_segment:
    with st.spinner(f"Generating SHAP for {selected_segment}..."):
        try:
            shap_png = generate_shap_plot(selected_segment, gdf, model, explainer, feature_names, top_k)
            st.image(shap_png, use_container_width=True)
            
            segment_data = gdf[gdf[ID_COLUMN].astype(str) == selected_segment].iloc[0]
            
            st.markdown("#### 📊 Segment Details")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Segment ID", selected_segment)
                st.metric("POF", f"{segment_data['POF']:.4f}")
            
            with col2:
                st.metric("Geometry Type", segment_data.geometry.geom_type)
                if "CC Status" in segment_data.index and pd.notna(segment_data["CC Status"]):
                    cc_val = int(segment_data["CC Status"])
                    st.metric("CC Status", f"{cc_val} ({'With CC' if cc_val == 1 else 'No CC'})")
            
            with col3:
                if "CALCULATED_CONDUCTOR_AGE" in segment_data.index and pd.notna(segment_data["CALCULATED_CONDUCTOR_AGE"]):
                    st.metric("Conductor Age", f"{segment_data['CALCULATED_CONDUCTOR_AGE']:.1f} years")
                if "CONDUCTOR_LENGTH_UDF" in segment_data.index and pd.notna(segment_data["CONDUCTOR_LENGTH_UDF"]):
                    st.metric("Conductor Length", f"{segment_data['CONDUCTOR_LENGTH_UDF']:.2f}")
                if "CONDUCTOR_DIAMETER_UDF" in segment_data.index and pd.notna(segment_data["CONDUCTOR_DIAMETER_UDF"]):
                    st.metric("Conductor Diameter", f"{segment_data['CONDUCTOR_DIAMETER_UDF']:.4f}")
        
        except Exception as e:
            st.error(f"❌ Error generating SHAP plot: {e}")

# Display colorbar
st.markdown("---")
st.markdown("### 🎨 POF Color Scale")
colorbar_png = generate_colorbar(pof_filter[0], pof_filter[1])
st.image(colorbar_png, use_container_width=False)

# ============================================================================
# INSTRUCTIONS
# ============================================================================
with st.expander("📖 How to Use This App"):
    st.markdown("""
    ### Setup Instructions
    
    **To get a background map:**
    1. Go to Google Earth, OpenStreetMap, or similar mapping service
    2. Navigate to your area of interest
    3. Take a screenshot or export a map image
    4. Save it as `base_map.png` in the same directory as this app
    5. The map will automatically load and display behind your geometries
    
    **Alternative: Download map tiles**
    - Use MOBAC (Mobile Atlas Creator) to download tiles for your region
    - Place tiles in a `tiles/{z}/{x}/{y}.png` structure
    - This enables unlimited zoom (not implemented in current version)
    
    ### Using the Interactive Map
    
    **Mouse Controls:**
    - 🖱️ **Click & Drag**: Pan around the map
    - 🎯 **Scroll Wheel**: Zoom in and out
    - 👆 **Click Geometries**: Select a segment to analyze
    - 🔍 **Hover**: View tooltip with segment information
    
    **Buttons:**
    - **+ Zoom In**: Increase zoom level
    - **- Zoom Out**: Decrease zoom level  
    - **⟲ Reset**: Return to original view
    
    **Colors:**
    - 🔴 Red: High POF (high risk)
    - 🟡 Yellow: Medium POF
    - 🔵 Blue: Low POF (low risk)
    
    ### SHAP Analysis
    
    1. Click on any segment in the map (or use the dropdown below)
    2. The SHAP waterfall plot will appear automatically
    3. Segments details show: ID, POF, geometry type, and filtered attributes
    
    **Understanding SHAP Plots:**
    - Red bars push prediction higher (increase failure probability)
    - Blue bars push prediction lower (decrease failure probability)
    - Bar length shows magnitude of impact
    - Features are sorted by absolute impact
    
    ### Filters
    
    Use the sidebar to filter segments by:
    - **POF Range**: Show only segments within a POF range
    - **CC Status**: Filter by critical component status (0 or 1)
    - **Conductor Age**: Filter by age in years
    - **Conductor Length**: Filter by physical length
    - **Conductor Diameter**: Filter by diameter measurement
    
    All filters work together - apply multiple filters to narrow down specific asset groups.
    """)

with st.expander("🛠️ Technical Information"):
    st.markdown(f"""
    ### System Configuration
    
    **Files:**
    - Model: `{MODEL_FILE}`
    - Data: `{DATA_FILE}`
    - Base Map: `{BASE_MAP_IMAGE}` {'✅ Found' if os.path.exists(BASE_MAP_IMAGE) else '❌ Not Found'}
    
    **Dataset:**
    - Total Segments: {len(gdf):,}
    - Filtered Segments: {len(filtered_gdf):,}
    - POF Range: {pof_min:.4f} - {pof_max:.4f}
    - Features: {len(feature_names)}
    - Geometry Types: {', '.join(gdf.geometry.geom_type.unique())}
    
    **Map Bounds:**
    - Min Longitude: {bounds[0]:.6f}
    - Min Latitude: {bounds[1]:.6f}
    - Max Longitude: {bounds[2]:.6f}
    - Max Latitude: {bounds[3]:.6f}
    - Center: [{center_lon:.6f}, {center_lat:.6f}]
    
    **Rendering:**
    - Technology: HTML5 Canvas
    - No external CDN calls
    - All resources embedded or local
    - CSP Compliant: ✅ Yes
    
    **Performance:**
    - Display Limit: {max_display:,} segments
    - SHAP Sample Size: {sample_size:,}
    - Top-K Features: {top_k}
    """)

with st.expander("💡 Tips & Tricks"):
    st.markdown("""
    ### Performance Tips
    
    - **Reduce Display Limit**: Lower the "Max segments" slider in the sidebar for faster rendering
    - **Use Filters**: Apply POF or other filters to focus on specific segments
    - **Background Map**: A lightweight base map image improves visual context without impacting performance
    
    ### Getting Better Maps
    
    **Free Options:**
    1. **Google Earth**: Navigate to area, screenshot, save as PNG
    2. **OpenStreetMap Export**: Go to openstreetmap.org, export area as image
    3. **QGIS**: Free GIS software, can export custom map images
    4. **Satellite Imagery**: Google Maps satellite view screenshots
    
    **Paid Options:**
    1. **MapBox**: Download custom styled maps
    2. **Google Maps API**: Static map API
    3. **Bing Maps**: Static map tiles
    
    ### Map Preparation
    
    1. Identify your data bounds (shown in Technical Information)
    2. Get a map image covering those exact coordinates
    3. Crop/resize to appropriate resolution (2000-4000px recommended)
    4. Save as `base_map.png` with transparency if needed
    5. Restart the app - map loads automatically!
    
    ### Click Detection
    
    The app detects clicks on:
    - **Lines**: Within 10 pixels of the line
    - **Points**: Within 12 pixels of the center
    - **Polygons**: Anywhere inside the polygon
    
    If clicks aren't registering, try:
    - Zooming in for better precision
    - Clicking directly on the geometry (not nearby)
    - Using the dropdown selector as backup
    """)

# ============================================================================
# STATISTICS SUMMARY
# ============================================================================
st.markdown("---")
st.markdown("### 📊 Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Segments", f"{len(gdf):,}")
    st.metric("Displayed", f"{len(filtered_gdf):,}")

with col2:
    st.metric("Min POF", f"{pof_min:.4f}")
    st.metric("Max POF", f"{pof_max:.4f}")
    st.metric("Avg POF", f"{gdf['POF'].mean():.4f}")

with col3:
    if "CC Status" in gdf.columns:
        cc_dist = gdf["CC Status"].value_counts().to_dict()
        st.metric("CC Status 0", f"{cc_dist.get(0, 0):,}")
        st.metric("CC Status 1", f"{cc_dist.get(1, 0):,}")

with col4:
    if "CALCULATED_CONDUCTOR_AGE" in gdf.columns:
        age_data = gdf["CALCULATED_CONDUCTOR_AGE"].dropna()
        if len(age_data) > 0:
            st.metric("Avg Age", f"{age_data.mean():.1f} years")
            st.metric("Age Range", f"{age_data.min():.0f}-{age_data.max():.0f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>🚀 Streamlit + HTML5 Canvas + XGBoost + SHAP</p>
    <p>✅ Fully Interactive • CSP Compliant • No External Dependencies</p>
    <p>🗺️ Add base_map.png for background map tiles</p>
</div>
""", unsafe_allow_html=True)
