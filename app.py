# frontend/streamlit_app.py
"""
Streamlit frontend with multi-model support embedding a MapLibre map (WebGL).
- Model selection dropdown in sidebar
- Different visualization types: single-click (equipment) vs multi-click (structure)
- Dynamic configuration based on selected model
- File download functionality
- POF range and label filtering per model
"""
import os
import streamlit as st
import requests
import json
from datetime import datetime

st.set_page_config(layout="wide", page_title="Multi-Model Assets Map + SHAP")

# Sidebar configuration
st.sidebar.title("🚀 Multi-Model Configuration")

API_BASE = st.sidebar.text_input(
    "Backend API base URL", value=os.environ.get("API_BASE", "http://127.0.0.1:8000")
)

# Model selection section
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Model Selection")

# Initialize session state
if 'available_models' not in st.session_state:
    st.session_state.available_models = {}
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None
if 'model_config' not in st.session_state:
    st.session_state.model_config = {}

# Fetch available models
try:
    models_response = requests.get(f"{API_BASE}/models", timeout=5)
    if models_response.status_code == 200:
        models_data = models_response.json()
        st.session_state.available_models = models_data["available_models"]
        backend_current_model = models_data.get("current_model")
        
        if not st.session_state.current_model:
            st.session_state.current_model = backend_current_model
            
    else:
        st.sidebar.error(f"❌ Failed to fetch models: {models_response.status_code}")
except Exception as e:
    st.sidebar.error(f"❌ Cannot connect to backend: {e}")

# Model selection dropdown
if st.session_state.available_models:
    model_options = list(st.session_state.available_models.keys())
    current_idx = 0
    if st.session_state.current_model in model_options:
        current_idx = model_options.index(st.session_state.current_model)
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=model_options,
        index=current_idx,
        format_func=lambda x: st.session_state.available_models[x]["name"],
        help="Choose the model configuration to use"
    )
    
    # Show model description
    if selected_model:
        model_info = st.session_state.available_models[selected_model]
        st.sidebar.info(f"📝 {model_info['description']}")
        st.sidebar.info(f"🎯 Visualization: {model_info['visualization_type']}")
        
        # Switch model if different from current
        if selected_model != st.session_state.current_model:
            with st.spinner(f"Loading {model_info['name']}..."):
                try:
                    load_response = requests.post(f"{API_BASE}/models/{selected_model}/load", timeout=30)
                    if load_response.status_code == 200:
                        load_data = load_response.json()
                        st.sidebar.success(f"✅ {load_data['message']}")
                        st.sidebar.info(f"📊 Features loaded: {load_data['features_loaded']:,}")
                        st.session_state.current_model = selected_model
                        st.session_state.dataset_info = None  # Force refresh
                        st.rerun()
                    else:
                        error_text = load_response.text if hasattr(load_response, 'text') else 'Unknown error'
                        st.sidebar.error(f"❌ Model loading failed: {load_response.status_code} - {error_text}")
                except Exception as e:
                    st.sidebar.error(f"❌ Model loading error: {e}")
else:
    st.sidebar.warning("⚠️ No models available")

# Main configuration section
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Display Settings")

LIMIT = st.sidebar.number_input(
    "Max features per viewport", min_value=100, max_value=20000, value=5000, step=100
)
TOPK = st.sidebar.number_input(
    "Top-K SHAP features", min_value=5, max_value=200, value=20, step=1
)

# Test backend connection and get dataset info
dataset_info = st.session_state.dataset_info
current_model = st.session_state.current_model

if not dataset_info and current_model:
    try:
        health_response = requests.get(f"{API_BASE}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            
            # Get dataset info
            try:
                info_response = requests.get(f"{API_BASE}/info", timeout=10)
                if info_response.status_code == 200:
                    dataset_info = info_response.json()
                    st.session_state.dataset_info = dataset_info
                    
            except Exception as e:
                st.warning(f"⚠️ Dataset info error: {e}")
    except Exception as e:
        st.error(f"❌ Cannot connect to backend: {e}")

# Main title and info
st.title("🗺️ Multi-Model Assets Map + Dynamic SHAP Analysis")

if dataset_info:
    # Display current model info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Model", dataset_info.get('model_name', 'Unknown'))
        st.metric("Visualization Type", dataset_info.get('visualization_type', 'Unknown'))
    
    with col2:
        st.metric("Total Features", f"{dataset_info['total_count']:,}")
        if dataset_info.get('pof_range'):
            pof_min_global, pof_max_global = dataset_info['pof_range']
            st.metric("POF Range", f"{pof_min_global:.3f} - {pof_max_global:.3f}")
    
    with col3:
        geom_types = dataset_info.get('geometry_types', {})
        st.metric("Geometry Types", len(geom_types))
        for geom_type, count in geom_types.items():
            st.write(f"• {geom_type}: {count:,}")

    # Structure-specific info
    if dataset_info.get('visualization_type') == 'multi_click':
        if 'structures_count' in dataset_info:
            st.info(f"🏗️ **Structure Model**: {dataset_info['structures_count']:,} structures with "
                   f"{dataset_info['equipment_per_structure']['mean']:.1f} equipment on average "
                   f"(range: {dataset_info['equipment_per_structure']['min']}-{dataset_info['equipment_per_structure']['max']})")

    # Show label distribution
    if dataset_info.get('has_label_column', False) and 'label_distribution' in dataset_info:
        label_dist = dataset_info['label_distribution']
        failures = int(label_dist.get('1', 0))
        non_failures = int(label_dist.get('0', 0))
        total_labeled = failures + non_failures
        if total_labeled > 0:
            failure_rate = failures / total_labeled * 100
            st.success(f"📊 **Labels**: {failure_rate:.1f}% failure rate ({failures:,} failures, {non_failures:,} non-failures)")

    # Calculate and show center point for navigation
    if dataset_info.get('bounds'):
        bounds = dataset_info['bounds']
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lng = (bounds[0] + bounds[2]) / 2
        st.info(f"🎯 **Data center**: {center_lat:.6f}, {center_lng:.6f}")

# Filters section
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Data Filters")

# POF Range Filter
pof_min_filter, pof_max_filter = None, None
if dataset_info and dataset_info.get('pof_range'):
    global_pof_min, global_pof_max = dataset_info['pof_range']
    
    st.sidebar.markdown("#### 📈 POF Range Filter")
    pof_range = st.sidebar.slider(
        "Select POF range to display",
        min_value=float(global_pof_min),
        max_value=float(global_pof_max),
        value=(float(global_pof_min), float(global_pof_max)),
        step=0.001,
        format="%.3f",
        help="Only show geometries with POF values within this range"
    )
    pof_min_filter, pof_max_filter = pof_range

# Label Filter
label_filter = "both"
ID_COL = dataset_info.get('id_column', 'id') if dataset_info else 'id'
LABEL_COL = dataset_info.get('label_column', 'label') if dataset_info else 'label'

if dataset_info and dataset_info.get('has_label_column', False):
    st.sidebar.markdown("#### 🏷️ Label Filter")
    label_options = {
        "both": "Both (failures + non-failures)",
        "failures": "Failures only (label=1)",
        "non_failures": "Non-failures only (label=0)"
    }
    
    label_filter = st.sidebar.selectbox(
        f"Filter by {LABEL_COL} column",
        options=list(label_options.keys()),
        format_func=lambda x: label_options[x],
        help="Filter features by their failure status"
    )
else:
    st.sidebar.info(f"Label filtering not available")

# File download section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 File Downloads")

try:
    files_response = requests.get(f"{API_BASE}/files", timeout=5)
    if files_response.status_code == 200:
        files_data = files_response.json()
        files_list = files_data["files"]
        
        if files_list:
            selected_file = st.sidebar.selectbox(
                "Available Files",
                options=[f["name"] for f in files_list],
                format_func=lambda x: f"{x} ({next(f['size'] for f in files_list if f['name'] == x)} bytes)"
            )
            
            if st.sidebar.button("📥 Download Selected File"):
                try:
                    download_url = f"{API_BASE}/files/{selected_file}"
                    st.sidebar.markdown(f"[📥 Click here to download {selected_file}]({download_url})")
                except Exception as e:
                    st.sidebar.error(f"Download error: {e}")
        else:
            st.sidebar.info("No files available for download")
    else:
        st.sidebar.warning("⚠️ Could not fetch file list")
except Exception as e:
    st.sidebar.warning(f"⚠️ File listing error: {e}")

# Instructions based on model type
if dataset_info:
    viz_type = dataset_info.get('visualization_type', 'single_click')
    if viz_type == 'single_click':
        st.markdown("""
        **Equipment Model Instructions:**
        - Pan/zoom to explore the data - viewport loading keeps performance smooth
        - Hover over equipment to see ID and POF values
        - Click on any equipment to generate and view its SHAP explanation
        - Use filters to focus on specific POF ranges or failure types
        """)
    elif viz_type == 'multi_click':
        st.markdown("""
        **Structure Model Instructions:**
        - Pan/zoom to explore structures - each structure may contain multiple equipment
        - Hover over structures to see aggregate information
        - Click on a structure to see SHAP explanations for ALL equipment within that structure
        - Multiple SHAP plots will appear in a scrollable popup for comprehensive analysis
        - Use filters to focus on structures with specific characteristics
        """)
else:
    st.markdown("Select a model to see specific instructions for that visualization type.")

# --- Enhanced HTML/JS Template for Multi-Model Support ---
HTML_TEMPLATE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Multi-Model Assets Map</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet" />
  <style>
    body { margin:0; padding:0; font-family: Arial, sans-serif; }
    #map { 
      position: absolute; 
      top:0; 
      bottom:0; 
      width:100%; 
      height:100vh; 
    }
    .debug-panel {
      position: absolute;
      top: 10px;
      right: 10px;
      background: rgba(255,255,255,0.95);
      border: 1px solid #ccc;
      border-radius: 5px;
      padding: 10px;
      font-family: monospace;
      font-size: 11px;
      max-width: 320px;
      z-index: 1000;
    }
    .debug-panel h4 { margin: 0 0 5px 0; font-size: 12px; }
    .colorbar-panel {
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(255,255,255,0.95);
      border: 1px solid #ccc;
      border-radius: 8px;
      padding: 12px;
      z-index: 1000;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .colorbar-panel h4 { 
      margin: 0 0 8px 0; 
      font-size: 12px; 
      font-weight: bold;
      text-align: center;
    }
    .colorbar-panel img { 
      display: block; 
      max-width: 300px; 
      height: auto;
      border-radius: 4px;
    }
    .modal {
      position: fixed;
      z-index: 9999;
      left: 50%;
      top: 5%;
      transform: translateX(-50%);
      background: white;
      border-radius: 8px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.4);
      padding: 15px;
      max-width: 95%;
      max-height: 90vh;
      overflow-y: auto;
      min-width: 800px;
    }
    .modal img { 
      max-width: 100%; 
      height: auto; 
      display: block; 
      margin: 10px auto;
      border: 1px solid #ddd;
      border-radius: 4px;
    }
    .close-btn { 
      cursor: pointer; 
      float: right; 
      padding: 8px 12px; 
      border-radius: 4px; 
      border: 1px solid #ccc; 
      background: #f5f5f5;
      font-weight: bold;
      margin-bottom: 10px;
    }
    .close-btn:hover {
      background: #e5e5e5;
    }
    .loading {
      text-align: center;
      padding: 20px;
      font-style: italic;
      color: #666;
    }
    .maplibregl-popup-content {
      padding: 8px 12px;
      max-width: 250px;
    }
    .filter-info {
      font-size: 10px;
      color: #666;
      margin-top: 5px;
      text-align: center;
    }
    .model-info {
      font-size: 10px;
      color: #333;
      margin-top: 5px;
      text-align: center;
      font-weight: bold;
    }
    .equipment-section {
      margin: 20px 0;
      padding: 15px;
      border: 1px solid #ddd;
      border-radius: 8px;
      background: #f9f9f9;
    }
    .equipment-header {
      font-size: 16px;
      font-weight: bold;
      margin-bottom: 10px;
      color: #333;
    }
    .equipment-info {
      font-size: 12px;
      color: #666;
      margin-bottom: 15px;
    }
    .shap-container {
      text-align: center;
    }
    .multi-shap-header {
      text-align: center;
      margin-bottom: 20px;
      padding-bottom: 10px;
      border-bottom: 2px solid #ddd;
    }
    .multi-shap-header h3 {
      margin: 0;
      color: #333;
    }
    .multi-shap-header .structure-info {
      color: #666;
      font-size: 14px;
      margin-top: 5px;
    }
  </style>
</head>
<body>
<div id="map"></div>

<div id="colorbar-panel" class="colorbar-panel">
  <h4>POF Color Scale</h4>
  <img id="colorbar-img" src="" alt="Loading colorbar..." style="display: none;" />
  <div id="colorbar-loading">Loading...</div>
  <div class="filter-info" id="filter-info"></div>
  <div class="model-info" id="model-info"></div>
</div>

<div id="debug-panel" class="debug-panel">
  <h4>🔍 Debug Info</h4>
  <div id="debug-content">Initializing...</div>
</div>

<div id="modal" class="modal" style="display:none;">
  <button id="closeBtn" class="close-btn">✕ Close</button>
  <div style="clear:both;"></div>
  <div id="modalBody"></div>
</div>

<script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
<script>
const API_BASE = "%%API_BASE%%";
const LIMIT = %%LIMIT%%;
const TOPK = %%TOPK%%;
const ID_COL = "%%ID_COL%%";
const LABEL_COL = "%%LABEL_COL%%";
const POF_MIN_FILTER = %%POF_MIN_FILTER%%;
const POF_MAX_FILTER = %%POF_MAX_FILTER%%;
const LABEL_FILTER = "%%LABEL_FILTER%%";

console.log("🗺️ Initializing multi-model map with config:", {
    API_BASE,
    POF_RANGE: [POF_MIN_FILTER, POF_MAX_FILTER],
    LABEL_FILTER,
    ID_COL,
    LABEL_COL
});

// Current model info
let currentModelInfo = null;

function updateDebugPanel(content) {
    document.getElementById('debug-content').innerHTML = content;
}

function updateFilterInfo() {
    let filterText = "";
    if (POF_MIN_FILTER !== null && POF_MAX_FILTER !== null) {
        filterText += `POF: ${POF_MIN_FILTER.toFixed(3)}-${POF_MAX_FILTER.toFixed(3)}`;
    }
    if (LABEL_FILTER !== "both") {
        if (filterText) filterText += "<br>";
        filterText += `Labels: ${LABEL_FILTER.replace('_', ' ')}`;
    }
    document.getElementById('filter-info').innerHTML = filterText;
}

function updateModelInfo(modelName, vizType) {
    const modelText = modelName ? `${modelName}<br>${vizType}` : '';
    document.getElementById('model-info').innerHTML = modelText;
}

function loadColorbar() {
    const img = document.getElementById('colorbar-img');
    const loading = document.getElementById('colorbar-loading');
    
    let colorbarUrl = `${API_BASE}/colorbar.png?width=300&height=40`;
    if (POF_MIN_FILTER !== null && POF_MAX_FILTER !== null) {
        colorbarUrl += `&pof_min_range=${POF_MIN_FILTER}&pof_max_range=${POF_MAX_FILTER}`;
    }
    
    console.log("🎨 Loading colorbar:", colorbarUrl);
    
    img.onload = function() {
        loading.style.display = 'none';
        img.style.display = 'block';
    };
    
    img.onerror = function() {
        loading.innerHTML = 'Colorbar error';
        console.error("❌ Colorbar load failed");
    };
    
    img.src = colorbarUrl;
}

// Test API connection and get model info
Promise.all([
    fetch(`${API_BASE}/health`),
    fetch(`${API_BASE}/info`)
]).then(async ([healthResp, infoResp]) => {
    if (healthResp.ok && infoResp.ok) {
        const healthData = await healthResp.json();
        const infoData = await infoResp.json();
        
        currentModelInfo = {
            name: infoData.model_name,
            visualization_type: infoData.visualization_type,
            current_model: infoData.current_model
        };
        
        console.log("✅ API Connected:", healthData);
        console.log("📋 Model Info:", currentModelInfo);
        
        updateDebugPanel(`✅ ${currentModelInfo.name}<br>Viz: ${currentModelInfo.visualization_type}<br>Features: ${infoData.total_count.toLocaleString()}`);
        updateModelInfo(currentModelInfo.name, currentModelInfo.visualization_type);
        loadColorbar();
        updateFilterInfo();
    } else {
        console.error("❌ API Connection failed");
        updateDebugPanel(`❌ API Error<br>Health: ${healthResp.status}<br>Info: ${infoResp.status}`);
    }
}).catch(e => {
    console.error("❌ API Error:", e);
    updateDebugPanel(`❌ Connection Error:<br>${e.message}`);
});

// Initialize map
const map = new maplibregl.Map({
    container: 'map',
    style: {
        version: 8,
        sources: {
            'osm': {
                type: 'raster',
                tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
                tileSize: 256,
                attribution: '© OpenStreetMap contributors'
            }
        },
        layers: [{
            id: 'osm',
            type: 'raster',
            source: 'osm'
        }]
    },
    center: [-122.45, 37.75],
    zoom: 10
});

map.addControl(new maplibregl.NavigationControl());

let popup = new maplibregl.Popup({ 
    closeButton: false, 
    closeOnClick: false 
});

let isLoading = false;

// Fetch viewport data
async function loadViewportData() {
    if (isLoading) return;
    isLoading = true;
    
    const b = map.getBounds();
    const minx = b.getWest();
    const miny = b.getSouth();
    const maxx = b.getEast();
    const maxy = b.getNorth();
    
    const bboxInfo = `Bbox: [${minx.toFixed(4)}, ${miny.toFixed(4)}, ${maxx.toFixed(4)}, ${maxy.toFixed(4)}]`;
    updateDebugPanel(`🔥 Loading...<br>Zoom: ${map.getZoom().toFixed(1)}<br>${bboxInfo.substring(0,50)}...`);
    
    let url = `${API_BASE}/bbox?minx=${minx}&miny=${miny}&maxx=${maxx}&maxy=${maxy}&limit=${LIMIT}`;
    
    if (POF_MIN_FILTER !== null) url += `&pof_min_filter=${POF_MIN_FILTER}`;
    if (POF_MAX_FILTER !== null) url += `&pof_max_filter=${POF_MAX_FILTER}`;
    if (LABEL_FILTER !== "both") url += `&label_filter=${LABEL_FILTER}`;
    
    try {
        const resp = await fetch(url);
        if (!resp.ok) {
            const errorText = await resp.text();
            updateDebugPanel(`❌ API Error ${resp.status}<br>${errorText.substring(0,50)}...`);
            return;
        }
        
        const geojson = await resp.json();
        console.log("📦 Received GeoJSON:", geojson);
        
        const featureCount = geojson.features.length;
        let debugText = `📍 Features: ${featureCount}<br>Zoom: ${map.getZoom().toFixed(1)}<br>Center: [${map.getCenter().lng.toFixed(4)}, ${map.getCenter().lat.toFixed(4)}]`;
        
        if (currentModelInfo) {
            debugText += `<br>Model: ${currentModelInfo.current_model}`;
        }
        
        if (POF_MIN_FILTER !== null || POF_MAX_FILTER !== null) {
            debugText += `<br>POF: ${POF_MIN_FILTER?.toFixed(3) || 'min'}-${POF_MAX_FILTER?.toFixed(3) || 'max'}`;
        }
        if (LABEL_FILTER !== "both") {
            debugText += `<br>Label: ${LABEL_FILTER}`;
        }
        
        updateDebugPanel(debugText);

        // Update or create map source
        if (map.getSource('assets')) {
            map.getSource('assets').setData(geojson);
        } else {
            map.addSource('assets', {
                type: 'geojson',
                data: geojson
            });

            // Create layers for different geometry types
            // LineString layer (equipment model)
            map.addLayer({
                id: 'linestring-layer',
                type: 'line',
                source: 'assets',
                filter: ['==', ['geometry-type'], 'LineString'],
                paint: {
                    'line-color': ['get', 'color'],
                    'line-width': 5,
                    'line-opacity': 0.9
                }
            });

            // Point layer (structure model points)
            map.addLayer({
                id: 'point-layer',
                type: 'circle',
                source: 'assets',
                filter: ['==', ['geometry-type'], 'Point'],
                paint: {
                    'circle-color': ['get', 'color'],
                    'circle-radius': 6,
                    'circle-stroke-width': 2,
                    'circle-stroke-color': '#ffffff'
                }
            });

            // Polygon layer (structure model polygons)
            map.addLayer({
                id: 'polygon-fill-layer',
                type: 'fill',
                source: 'assets',
                filter: ['==', ['geometry-type'], 'Polygon'],
                paint: {
                    'fill-color': ['get', 'color'],
                    'fill-opacity': 0.6
                }
            });

            map.addLayer({
                id: 'polygon-stroke-layer',
                type: 'line',
                source: 'assets',
                filter: ['==', ['geometry-type'], 'Polygon'],
                paint: {
                    'line-color': '#333',
                    'line-width': 2
                }
            });

            // Mouse events for all layers
            const layers = ['linestring-layer', 'point-layer', 'polygon-fill-layer'];
            
            layers.forEach(layerId => {
                // Hover events
                map.on('mouseenter', layerId, function(e) {
                    map.getCanvas().style.cursor = 'pointer';
                    const f = e.features[0];
                    if (!f) return;
                    
                    const props = f.properties || {};
                    const id = props[ID_COL] || props.id || "n/a";
                    const pof = props.POF !== undefined && props.POF !== null ? Number(props.POF).toFixed(4) : "n/a";
                    const vizType = props.visualization_type || "single_click";
                    
                    let htmlContent = `<b>ID:</b> ${id}<br/><b>POF:</b> ${pof}<br/><b>Type:</b> ${f.geometry.type}`;
                    
                    // Add structure-specific info
                    if (vizType === "multi_click" && props.structure_id) {
                        htmlContent += `<br/><b>Structure:</b> ${props.structure_id}`;
                    }
                    
                    // Add label info if available
                    if (props[LABEL_COL] !== undefined) {
                        const labelText = props[LABEL_COL] == 1 ? "Failure" : "Non-failure";
                        htmlContent += `<br/><b>Label:</b> ${labelText}`;
                    }
                    
                    // Add interaction hint
                    if (vizType === "multi_click") {
                        htmlContent += `<br/><i>Click for all equipment SHAP plots</i>`;
                    } else {
                        htmlContent += `<br/><i>Click for SHAP explanation</i>`;
                    }
                    
                    popup.setLngLat(e.lngLat)
                        .setHTML(htmlContent)
                        .addTo(map);
                });
                
                map.on('mouseleave', layerId, function() {
                    map.getCanvas().style.cursor = '';
                    popup.remove();
                });
                
                // Click events - different behavior based on visualization type
                map.on('click', layerId, function(e) {
                    const f = e.features[0];
                    if (!f) return;
                    
                    const props = f.properties || {};
                    const idRaw = props[ID_COL] || props.id;
                    const vizType = props.visualization_type || "single_click";
                    
                    if (!idRaw) {
                        console.warn("⚠️ No ID found in feature properties:", props);
                        return;
                    }
                    
                    console.log(`🖱️ Clicked feature with ID: ${idRaw}, viz type: ${vizType}`);
                    
                    if (vizType === "multi_click") {
                        // Structure model - show multiple SHAP plots
                        const structureId = props.structure_id || idRaw;
                        showMultiShapForStructure(encodeURIComponent(String(structureId)));
                    } else {
                        // Equipment model - show single SHAP plot
                        showShapForAsset(encodeURIComponent(String(idRaw)));
                    }
                });
            });
        }
        
    } catch (err) {
        console.error("❌ Error loading viewport data:", err);
        updateDebugPanel(`❌ Load Error:<br>${err.message}`);
    } finally {
        isLoading = false;
    }
}

// Show single SHAP PNG modal (equipment model)
async function showShapForAsset(assetIdEncoded) {
    const modal = document.getElementById('modal');
    const body = document.getElementById('modalBody');
    const closeBtn = document.getElementById('closeBtn');
    
    closeBtn.onclick = function() { 
        modal.style.display = 'none'; 
        body.innerHTML = ''; 
    };
    
    modal.style.display = 'block';
    const decodedId = decodeURIComponent(assetIdEncoded);
    body.innerHTML = `<div class="loading">Loading SHAP explanation for asset <b>${decodedId}</b>...</div>`;
    
    const imgUrl = `${API_BASE}/asset/${assetIdEncoded}/shap.png?top_k=${TOPK}`;
    console.log("🖼️ Loading SHAP image from:", imgUrl);
    
    const img = new Image();
    img.onload = function() {
        body.innerHTML = `
            <h3>SHAP Explanation - Asset ${decodedId}</h3>
            <img src="${imgUrl}" alt="SHAP waterfall plot" />
        `;
    };
    img.onerror = function() {
        body.innerHTML = `
            <h3>Error</h3>
            <p>Failed to load SHAP explanation for asset ${decodedId}.</p>
            <p>Please check the console for more details.</p>
        `;
    };
    img.src = imgUrl;
}

// Show multiple SHAP PNGs modal (structure model)
async function showMultiShapForStructure(structureIdEncoded) {
    const modal = document.getElementById('modal');
    const body = document.getElementById('modalBody');
    const closeBtn = document.getElementById('closeBtn');
    
    closeBtn.onclick = function() { 
        modal.style.display = 'none'; 
        body.innerHTML = ''; 
    };
    
    modal.style.display = 'block';
    const decodedStructureId = decodeURIComponent(structureIdEncoded);
    
    body.innerHTML = `<div class="loading">Loading equipment list for structure <b>${decodedStructureId}</b>...</div>`;
    
    try {
        // Get list of equipment for this structure
        const equipmentUrl = `${API_BASE}/structure/${structureIdEncoded}/equipment`;
        console.log("📋 Fetching equipment list:", equipmentUrl);
        
        const resp = await fetch(equipmentUrl);
        if (!resp.ok) {
            throw new Error(`Failed to fetch equipment: ${resp.status}`);
        }
        
        const equipmentData = await resp.json();
        console.log("🔧 Equipment data:", equipmentData);
        
        const equipment = equipmentData.equipment;
        const structureId = equipmentData.structure_id;
        
        if (!equipment || equipment.length === 0) {
            body.innerHTML = `
                <div class="multi-shap-header">
                    <h3>Structure ${structureId}</h3>
                    <div class="structure-info">No equipment found</div>
                </div>
            `;
            return;
        }
        
        // Create header
        let html = `
            <div class="multi-shap-header">
                <h3>Structure ${structureId} - SHAP Analysis</h3>
                <div class="structure-info">
                    ${equipment.length} equipment units - Loading explanations...
                </div>
            </div>
        `;
        
        // Add loading placeholders for each equipment
        equipment.forEach(eq => {
            html += `
                <div class="equipment-section" id="equipment-${eq.equipment_id}">
                    <div class="equipment-header">Equipment ${eq.equipment_id}</div>
                    <div class="equipment-info">
                        POF: ${eq.POF !== null ? eq.POF.toFixed(4) : 'N/A'}
                        ${eq.label !== null ? ` | Label: ${eq.label == 1 ? 'Failure' : 'Non-failure'}` : ''}
                    </div>
                    <div class="shap-container">
                        <div class="loading">Loading SHAP explanation...</div>
                    </div>
                </div>
            `;
        });
        
        body.innerHTML = html;
        
        // Load SHAP plots for each equipment asynchronously
        equipment.forEach((eq, index) => {
            const equipmentId = eq.equipment_id;
            const encodedEquipmentId = encodeURIComponent(equipmentId);
            const imgUrl = `${API_BASE}/asset/${encodedEquipmentId}/shap.png?top_k=${TOPK}`;
            const container = document.getElementById(`equipment-${equipmentId}`);
            const shapContainer = container ? container.querySelector('.shap-container') : null;

            if (!shapContainer) return; // Defensive: container must exist

            // Show loading spinner before starting image load
            shapContainer.innerHTML = `<div class="loading">Loading SHAP explanation...</div>`;

            const img = new window.Image();
            img.onload = function() {
                // Only update if container still exists (modal not closed)
                if (document.getElementById(`equipment-${equipmentId}`)) {
                    shapContainer.innerHTML = `<img src="${imgUrl}" alt="SHAP plot for equipment ${equipmentId}" />`;
                }
            };
            img.onerror = function() {
                if (document.getElementById(`equipment-${equipmentId}`)) {
                    shapContainer.innerHTML = `<div style="color: #d62728; text-align: center; padding: 20px;">Failed to load SHAP plot for equipment ${equipmentId}</div>`;
                }
            };
            img.src = imgUrl;
        });
        
    } catch (error) {
        console.error("❌ Error loading structure equipment:", error);
        body.innerHTML = `
            <div class="multi-shap-header">
                <h3>Error</h3>
                <div class="structure-info">
                    Failed to load equipment for structure ${decodedStructureId}
                </div>
            </div>
            <p style="text-align: center; color: #d62728;">${error.message}</p>
        `;
    }
}

// Initialize map events
map.on('load', function() {
    console.log("Map loaded, loading initial viewport data...");
    updateDebugPanel("Map loaded<br>Loading data...");
    loadViewportData();
});

map.on('moveend', function() {
    console.log("Map moved, loading viewport data...");
    loadViewportData();
});

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('modal');
    if (event.target === modal) {
        modal.style.display = 'none';
        document.getElementById('modalBody').innerHTML = '';
    }
}

console.log("Multi-model map initialization complete");
</script>
</body>
</html>
"""

# Replace tokens safely
pof_min_js = "null" if pof_min_filter is None else str(pof_min_filter)
pof_max_js = "null" if pof_max_filter is None else str(pof_max_filter)

# Prepare bounds for JS
if dataset_info and dataset_info.get('bounds'):
    minx, miny, maxx, maxy = dataset_info['bounds']
    center_lng = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2
    bounds_js = json.dumps([minx, miny, maxx, maxy])
    center_js = f"[{center_lng}, {center_lat}]"
else:
    bounds_js = "null"
    center_js = "[-122.45, 37.75]"  # fallback

html = (
    HTML_TEMPLATE
    .replace("%%API_BASE%%", API_BASE)
    .replace("%%LIMIT%%", str(LIMIT))
    .replace("%%TOPK%%", str(TOPK))
    .replace("%%ID_COL%%", ID_COL)
    .replace("%%LABEL_COL%%", LABEL_COL)
    .replace("%%POF_MIN_FILTER%%", pof_min_js)
    .replace("%%POF_MAX_FILTER%%", pof_max_js)
    .replace("%%LABEL_FILTER%%", label_filter)
    .replace("%%BOUNDS%%", bounds_js)
    .replace("%%CENTER%%", center_js)
)

# Render HTML in Streamlit
st.components.v1.html(html, height=800, scrolling=False)

# Enhanced information sections below the map
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 Active Configuration")
    if dataset_info:
        st.write(f"**Model:** {dataset_info.get('model_name', 'Unknown')}")
        st.write(f"**Visualization:** {dataset_info.get('visualization_type', 'Unknown')}")
        if pof_min_filter is not None and pof_max_filter is not None:
            st.write(f"**POF Range:** {pof_min_filter:.3f} - {pof_max_filter:.3f}")
        else:
            st.write("**POF Range:** All values")
        
        label_display = {
            "both": "All data",
            "failures": "Failures only",
            "non_failures": "Non-failures only"
        }
        st.write(f"**Label Filter:** {label_display.get(label_filter, label_filter)}")
    else:
        st.write("No model loaded")

with col2:
    st.markdown("### 📊 Dataset Statistics")
    if dataset_info:
        total_count = dataset_info['total_count']
        st.write(f"**Total Features:** {total_count:,}")
        
        if dataset_info.get('pof_range'):
            global_min, global_max = dataset_info['pof_range']
            st.write(f"**POF Range:** {global_min:.3f} - {global_max:.3f}")
        
        # Show geometry distribution
        geom_types = dataset_info.get('geometry_types', {})
        st.write("**Geometries:**")
        for geom_type, count in geom_types.items():
            percentage = (count / total_count) * 100
            st.write(f"• {geom_type}: {count:,} ({percentage:.1f}%)")
        
        # Structure-specific stats
        if dataset_info.get('visualization_type') == 'multi_click' and 'structures_count' in dataset_info:
            st.write(f"**Structures:** {dataset_info['structures_count']:,}")
            eq_stats = dataset_info['equipment_per_structure']
            st.write(f"**Equipment/Structure:** {eq_stats['min']}-{eq_stats['max']} (avg: {eq_stats['mean']:.1f})")
    else:
        st.write("Loading statistics...")

with col3:
    st.markdown("### 🗺️ Map Instructions")
    if dataset_info:
        viz_type = dataset_info.get('visualization_type', 'single_click')
        if viz_type == 'single_click':
            st.write("**Equipment Model:**")
            st.write("• Hover: View POF & details")
            st.write("• Click: Single SHAP plot")
            st.write("• Each geometry = 1 equipment")
        elif viz_type == 'multi_click':
            st.write("**Structure Model:**")
            st.write("• Hover: Structure info")
            st.write("• Click: Multiple SHAP plots")
            st.write("• Each geometry = 1 structure")
            st.write("• Multiple equipment per structure")
    else:
        st.write("Select a model to see instructions")

# Expandable help sections
with st.expander("🔧 Model Configuration Help"):
    st.markdown("""
    ### Adding New Models
    
    To add a new model configuration, update the `MODEL_CONFIGS` dictionary in the backend:
    
    ```python
    "your_model_key": {
        "name": "Your Model Name",
        "description": "Description of what this model does",
        "model_path": "path/to/your/model.json",
        "data_path": "path/to/your/data.csv",
        "feature_names_json": "your_features.json",
        "feature_names_txt": "your_features.txt", 
        "id_col": "your_id_column",
        "label_col": "your_label_column",
        "geometry_type": "equipment" or "structure",
        "structure_id_col": "structure_column", # Only for structure type
        "geometry_col": "geometry_column_name", # Default: "geometry"
        "visualization_type": "single_click" or "multi_click"
    }
    ```
    
    **Geometry Types:**
    - `equipment`: One geometry per equipment ID (single SHAP per click)
    - `structure`: Multiple equipment per structure (multiple SHAPs per click)
    
    **Visualization Types:**
    - `single_click`: Shows one SHAP plot when clicked
    - `multi_click`: Shows multiple SHAP plots in scrollable modal
    """)

with st.expander("📁 File Management"):
    st.markdown("""
    ### File Downloads
    
    Use the sidebar "File Downloads" section to:
    - View all files in the backend directory
    - See file sizes and modification dates
    - Download any file directly to your computer
    
    This is useful for:
    - Downloading model files
    - Accessing data files
    - Getting feature lists
    - Retrieving logs or other generated files
    """)

with st.expander("🎨 Visualization Guide"):
    st.markdown("""
    ### Understanding the Visualizations
    
    **Color Coding:**
    - Blue/Purple: Low POF (lower failure probability)
    - Yellow/Red: High POF (higher failure probability)
    - Color scale adjusts based on your POF filter range
    
    **Equipment Model (Single Click):**
    - Each LineString represents one piece of equipment
    - Click shows SHAP waterfall for that specific equipment
    - Good for individual equipment analysis
    
    **Structure Model (Multi Click):**
    - Each Point/Polygon represents a structure
    - Structure may contain multiple equipment
    - Click shows SHAP waterfalls for ALL equipment in that structure
    - Plots are scrollable in the popup
    - Good for comprehensive structure analysis
    
    **SHAP Interpretation:**
    - Green bars: Features that increase failure probability
    - Red bars: Features that decrease failure probability  
    - Longer bars = stronger influence on prediction
    - Features are ranked by absolute impact
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>🚀 Multi-Model Platform: FastAPI + Streamlit + MapLibre GL + XGBoost + SHAP</p>
    <p>Dynamic model switching • Viewport loading • Scalable architecture</p>
</div>
""", unsafe_allow_html=True)