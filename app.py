# frontend/streamlit_app.py
"""
Streamlit frontend embedding a MapLibre map (WebGL). The map requests features
in the current viewport via GET /bbox on the backend.

- Shows tooltip on hover with ID + POF
- On click (LineString geometries) fetches /asset/{id}/shap.png and displays in a modal
- POF range filtering and label filtering
- Colorbar display on map
"""
import os
import streamlit as st
import requests

st.set_page_config(layout="wide", page_title="Assets map + on-demand SHAP")

st.title("Assets map — dynamic viewport loading + on-click SHAP (LineString geometries)")

# Sidebar configuration
API_BASE = st.sidebar.text_input(
    "Backend API base URL", value=os.environ.get("API_BASE", "http://127.0.0.1:8000")
)
ID_COL = st.sidebar.text_input(
    "ID column name", value=os.environ.get("ID_COL", "id")
)
LABEL_COL = st.sidebar.text_input(
    "Label column name", value=os.environ.get("LABEL_COL", "label"),
    help="Column name for failure/non-failure labels (0/1)"
)
LIMIT = st.sidebar.number_input(
    "Max features per viewport (server limit)", min_value=100, max_value=200000, value=5000, step=100
)
TOPK = st.sidebar.number_input(
    "Top-K SHAP features (server-side PNG)", min_value=5, max_value=200, value=20, step=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Filters")

# Initialize session state for filters
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None

# Test backend connection and get dataset info
dataset_info = None
try:
    health_response = requests.get(f"{API_BASE}/health", timeout=5)
    if health_response.status_code == 200:
        health_data = health_response.json()
        st.success(f"✅ Backend connected: {health_data}")
        
        # Get dataset info
        try:
            info_response = requests.get(f"{API_BASE}/info", timeout=10)
            if info_response.status_code == 200:
                dataset_info = info_response.json()
                st.session_state.dataset_info = dataset_info
                
                st.info(f"""
**📊 Dataset Info:**
- Total features: {dataset_info['total_count']:,}
- Geometry types: {dataset_info['geometry_types']}
- POF range: {dataset_info['pof_range'][0]:.4f} - {dataset_info['pof_range'][1]:.4f}
- Data bounds: {dataset_info['bounds'] if dataset_info['bounds'] else 'Unknown'}
- Has label column '{LABEL_COL}': {dataset_info.get('has_label_column', False)}
                """)
                
                if dataset_info.get('has_label_column', False) and 'label_distribution' in dataset_info:
                    st.write(f"**Label distribution:** {dataset_info['label_distribution']}")
                
                # Calculate center point for easy reference
                if dataset_info['bounds']:
                    bounds = dataset_info['bounds']
                    center_lat = (bounds[1] + bounds[3]) / 2
                    center_lng = (bounds[0] + bounds[2]) / 2
                    st.write(f"📍 **Data center point:** {center_lat:.6f}, {center_lng:.6f}")
                        
            else:
                error_text = info_response.text if hasattr(info_response, 'text') else 'Unknown error'
                st.warning(f"⚠️ Could not get dataset info: {info_response.status_code} - {error_text}")
        except Exception as e:
            st.warning(f"⚠️ Dataset info error: {e}")
    else:
        st.error(f"❌ Backend health check failed: {health_response.status_code}")
except Exception as e:
    st.error(f"❌ Cannot connect to backend: {e}")

# Use session state info if available
if not dataset_info and st.session_state.dataset_info:
    dataset_info = st.session_state.dataset_info

# POF Range Filter
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
else:
    pof_min_filter, pof_max_filter = None, None

# Label Filter
label_filter = "both"  # default
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
    st.sidebar.info(f"Label filtering not available (column '{LABEL_COL}' not found)")

st.markdown(
    """
    - Pan/zoom to change the viewport. The browser asks the backend for only the visible features.
    - Hover over LineString geometries to see ID and POF. Click on a LineString to generate and display the SHAP waterfall (on-demand).
    - Use the sidebar filters to narrow down the displayed data by POF range and failure status.
    """
)

# --- HTML / JS template ---
HTML_TEMPLATE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Equipment LineString Map</title>
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
      max-width: 300px;
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
      overflow: auto;
    }
    .modal img { 
      max-width: 100%; 
      height: auto; 
      display:block; 
      margin: 10px auto; 
    }
    .close-btn { 
      cursor: pointer; 
      float: right; 
      padding: 8px 12px; 
      border-radius:4px; 
      border:1px solid #ccc; 
      background:#f5f5f5;
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
      max-width: 200px;
    }
    .filter-info {
      font-size: 10px;
      color: #666;
      margin-top: 5px;
      text-align: center;
    }
  </style>
</head>
<body>
<div id="map"></div>

<div id="colorbar-panel" class="colorbar-panel">
  <h4>POF Color Scale Legend</h4>
  <img id="colorbar-img" src="" alt="Loading colorbar..." style="display: none;" />
  <div id="colorbar-loading">Loading...</div>
  <div class="filter-info" id="filter-info"></div>
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

console.log("🗺️ Initializing map with filters:", {
    API_BASE,
    POF_RANGE: [POF_MIN_FILTER, POF_MAX_FILTER],
    LABEL_FILTER
});

function updateDebugPanel(content) {
    document.getElementById('debug-content').innerHTML = content;
}

function updateFilterInfo() {
    let filterText = "";
    if (POF_MIN_FILTER !== null && POF_MAX_FILTER !== null) {
        filterText += `POF Range: ${POF_MIN_FILTER.toFixed(3)}-${POF_MAX_FILTER.toFixed(3)}`;
    }
    if (LABEL_FILTER !== "both") {
        if (filterText) filterText += "<br>";
        filterText += `Labels: ${LABEL_FILTER.replace('_', ' ')}`;
    }
    document.getElementById('filter-info').innerHTML = filterText;
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

// Test API connection immediately
fetch(`${API_BASE}/health`)
    .then(r => r.json())
    .then(data => {
        console.log("✅ API Health:", data);
        updateDebugPanel(`✅ API Connected<br>Model: ${data.model_loaded}<br>Data: ${data.data_loaded}`);
        loadColorbar();
        updateFilterInfo();
    })
    .catch(e => {
        console.error("❌ API Error:", e);
        updateDebugPanel(`❌ API Error:<br>${e.message}`);
    });

// Use OpenStreetMap style for better LineString visibility
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
    center: [-122.45, 37.75], // San Francisco default
    zoom: 10
});

map.addControl(new maplibregl.NavigationControl());

let popup = new maplibregl.Popup({ 
    closeButton: false, 
    closeOnClick: false 
});

// Track if we're loading data
let isLoading = false;
let lastFeatureCount = 0;

// --- fetch viewport GeoJSON from backend ---
async function loadViewportData() {
    if (isLoading) return;
    isLoading = true;
    
    console.log("📥 Loading viewport data...");
    const b = map.getBounds();
    const minx = b.getWest();
    const miny = b.getSouth();
    const maxx = b.getEast();
    const maxy = b.getNorth();
    
    const bboxInfo = `Bbox: [${minx.toFixed(4)}, ${miny.toFixed(4)}, ${maxx.toFixed(4)}, ${maxy.toFixed(4)}]`;
    updateDebugPanel(`📥 Loading...<br>Zoom: ${map.getZoom().toFixed(1)}<br>${bboxInfo}`);
    
    let url = `${API_BASE}/bbox?minx=${minx}&miny=${miny}&maxx=${maxx}&maxy=${maxy}&limit=${LIMIT}`;
    
    // Add filters
    if (POF_MIN_FILTER !== null) url += `&pof_min_filter=${POF_MIN_FILTER}`;
    if (POF_MAX_FILTER !== null) url += `&pof_max_filter=${POF_MAX_FILTER}`;
    if (LABEL_FILTER !== "both") url += `&label_filter=${LABEL_FILTER}`;
    
    console.log("🌐 Fetching:", url);
    
    try {
        const resp = await fetch(url);
        if (!resp.ok) {
            const errorText = await resp.text();
            console.error("❌ BBOX fetch failed:", resp.status, errorText);
            updateDebugPanel(`❌ API Error ${resp.status}<br>${errorText.substring(0,50)}...`);
            return;
        }
        const geojson = await resp.json();
        console.log("📦 Received GeoJSON:", geojson);
        
        lastFeatureCount = geojson.features.length;
        let debugText = `🔍 Features: ${lastFeatureCount}<br>Zoom: ${map.getZoom().toFixed(1)}<br>Center: [${map.getCenter().lng.toFixed(4)}, ${map.getCenter().lat.toFixed(4)}]`;
        
        // Add filter info to debug
        if (POF_MIN_FILTER !== null || POF_MAX_FILTER !== null) {
            debugText += `<br>POF Filter: ${POF_MIN_FILTER?.toFixed(3) || 'min'}-${POF_MAX_FILTER?.toFixed(3) || 'max'}`;
        }
        if (LABEL_FILTER !== "both") {
            debugText += `<br>Label: ${LABEL_FILTER}`;
        }
        
        updateDebugPanel(debugText);
        
        // Log detailed feature info
        if (geojson.features.length > 0) {
            console.log("📋 Feature details:");
            geojson.features.slice(0, 3).forEach((f, i) => {
                const props = f.properties || {};
                console.log(`Feature ${i}:`, {
                    type: f.geometry.type,
                    id: props[ID_COL],
                    POF: props.POF,
                    label: props[LABEL_COL],
                    coordLength: f.geometry.coordinates ? f.geometry.coordinates.length : 'N/A'
                });
                
                if (f.geometry.type === 'LineString') {
                    const coords = f.geometry.coordinates;
                    console.log(`  🔹 LineString: ${coords.length} points`);
                    console.log(`  🔹 First: [${coords[0][0].toFixed(6)}, ${coords[0][1].toFixed(6)}]`);
                    console.log(`  🔹 Last: [${coords[coords.length-1][0].toFixed(6)}, ${coords[coords.length-1][1].toFixed(6)}]`);
                }
            });
        } else {
            console.log("❌ No features in response - checking data bounds...");
            // Get debug info from server
            try {
                const debugResp = await fetch(`${API_BASE}/debug/sample`);
                if (debugResp.ok) {
                    const debugData = await debugResp.json();
                    console.log("🔍 Debug data:", debugData);
                    
                    const bounds = debugData.bounds;
                    if (bounds && bounds.length === 4) {
                        const [dbMinX, dbMinY, dbMaxX, dbMaxY] = bounds;
                        const dbCenterLng = (dbMinX + dbMaxX) / 2;
                        const dbCenterLat = (dbMinY + dbMaxY) / 2;
                        
                        updateDebugPanel(`❌ No features in view<br>DB has ${debugData.total_count} total<br>DB center: [${dbCenterLng.toFixed(4)}, ${dbCenterLat.toFixed(4)}]<br>DB bounds: [${bounds.map(x=>x.toFixed(3)).join(',')}]`);
                        
                        console.log(`🗺️ Current view center: [${map.getCenter().lng.toFixed(6)}, ${map.getCenter().lat.toFixed(6)}]`);
                        console.log(`🗄️ Data center: [${dbCenterLng.toFixed(6)}, ${dbCenterLat.toFixed(6)}]`);
                        
                        // Auto-fly to data if it's far away
                        const distance = Math.sqrt(Math.pow(map.getCenter().lng - dbCenterLng, 2) + Math.pow(map.getCenter().lat - dbCenterLat, 2));
                        console.log(`📏 Distance to data: ${distance.toFixed(6)} degrees`);
                        
                        if (distance > 0.1) { // If more than ~11km away
                            console.log("🚀 Auto-flying to data location...");
                            setTimeout(() => {
                                map.fitBounds([[dbMinX, dbMinY], [dbMaxX, dbMaxY]], {
                                    padding: 50,
                                    maxZoom: 15
                                });
                            }, 1000);
                        }
                    } else {
                        updateDebugPanel(`❌ No features in view<br>DB has ${debugData.total_count || 'unknown'} total<br>No bounds available`);
                    }
                } else {
                    console.error("🔍 Debug fetch failed:", debugResp.status, await debugResp.text());
                }
            } catch (e) {
                console.error("🔍 Debug fetch failed:", e);
            }
        }

        // Update or create map source
        if (map.getSource('assets')) {
            console.log("📄 Updating existing source");
            map.getSource('assets').setData(geojson);
        } else {
            console.log("🆕 Creating new source and layers");
            
            // Add the source
            map.addSource('assets', {
                type: 'geojson',
                data: geojson
            });

            // Add layer for LineString geometries with bold styling
            map.addLayer({
                id: 'linestring-layer',
                type: 'line',
                source: 'assets',
                filter: ['==', ['geometry-type'], 'LineString'],
                paint: {
                    'line-color': ['get', 'color'],  // Use color from backend
                    'line-width': 5,      // Thicker lines
                    'line-opacity': 0.9   // More opaque
                }
            });

            // Add layer for Point geometries (if any)
            map.addLayer({
                id: 'point-layer',
                type: 'circle',
                source: 'assets',
                filter: ['==', ['geometry-type'], 'Point'],
                paint: {
                    'circle-color': ['get', 'color'],
                    'circle-radius': 8,
                    'circle-stroke-width': 2,
                    'circle-stroke-color': '#ffffff'
                }
            });

            // Add layer for Polygon geometries (if any)
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
                    'line-width': 1
                }
            });

            console.log("✅ Layers created successfully");

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
                    
                    let htmlContent = `<b>ID:</b> ${id}<br/><b>POF:</b> ${pof}<br/><b>Type:</b> ${f.geometry.type}`;
                    
                    // Add label info if available
                    if (props[LABEL_COL] !== undefined) {
                        const labelText = props[LABEL_COL] == 1 ? "Failure" : "Non-failure";
                        htmlContent += `<br/><b>Label:</b> ${labelText}`;
                    }
                    
                    popup.setLngLat(e.lngLat)
                        .setHTML(htmlContent)
                        .addTo(map);
                });
                
                map.on('mouseleave', layerId, function() {
                    map.getCanvas().style.cursor = '';
                    popup.remove();
                });
                
                // Click events
                map.on('click', layerId, function(e) {
                    const f = e.features[0];
                    if (!f) return;
                    const props = f.properties || {};
                    const idRaw = props[ID_COL] || props.id;
                    if (!idRaw) {
                        console.warn("⚠️ No ID found in feature properties:", props);
                        return;
                    }
                    console.log("🖱️ Clicked feature with ID:", idRaw);
                    showShapForAsset(encodeURIComponent(String(idRaw)));
                });
            });
            
            console.log("🖱️ Mouse events attached to layers");
        }
        
        console.log("✅ Data loading completed");
        
    } catch (err) {
        console.error("❌ Error loading viewport data:", err);
        updateDebugPanel(`❌ Load Error:<br>${err.message}`);
    } finally {
        isLoading = false;
    }
}

// --- show SHAP PNG modal ---
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
    
    // Create image element with error handling
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

// Initialize map
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

console.log("Map initialization complete");
</script>
</body>
</html>"""

# Replace tokens safely
pof_min_js = "null" if pof_min_filter is None else str(pof_min_filter)
pof_max_js = "null" if pof_max_filter is None else str(pof_max_filter)

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
)

# Render HTML in Streamlit
st.components.v1.html(html, height=800, scrolling=False)

# Additional information and controls below the map
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 Current Filters")
    if pof_min_filter is not None and pof_max_filter is not None:
        st.write(f"**POF Range:** {pof_min_filter:.3f} - {pof_max_filter:.3f}")
    else:
        st.write("**POF Range:** All values")
    
    if dataset_info and dataset_info.get('has_label_column', False):
        label_display = {
            "both": "All data",
            "failures": "Failures only (label=1)",
            "non_failures": "Non-failures only (label=0)"
        }
        st.write(f"**Label Filter:** {label_display.get(label_filter, label_filter)}")
    else:
        st.write("**Label Filter:** Not available")

with col2:
    st.markdown("### 📊 Quick Stats")
    if dataset_info:
        total_count = dataset_info['total_count']
        st.write(f"**Total Features:** {total_count:,}")
        
        if dataset_info.get('pof_range'):
            global_min, global_max = dataset_info['pof_range']
            st.write(f"**Global POF Range:** {global_min:.3f} - {global_max:.3f}")
        
        if dataset_info.get('label_distribution'):
            failures = dataset_info['label_distribution'].get('1', 0)
            non_failures = dataset_info['label_distribution'].get('0', 0)
            if failures > 0 or non_failures > 0:
                failure_rate = failures / (failures + non_failures) * 100
                st.write(f"**Failure Rate:** {failure_rate:.1f}% ({failures:,}/{failures + non_failures:,})")
    else:
        st.write("Loading dataset statistics...")

with col3:
    st.markdown("### 🗺️ Map Controls")
    st.write("**Navigation:**")
    st.write("- Pan: Click and drag")
    st.write("- Zoom: Mouse wheel or +/- buttons")
    st.write("- Reset: Double-click")
    
    st.write("**Interactions:**")
    st.write("- Hover: View POF and ID")
    st.write("- Click: Generate SHAP explanation")

# Help section
with st.expander("ℹ️ Help & Instructions"):
    st.markdown("""
    ### How to Use This Application
    
    #### 🗺️ **Map Interaction**
    - **Pan and Zoom**: Navigate to explore different areas of your data
    - **Hover**: Move your mouse over geometries to see ID, POF value, and label information
    - **Click**: Click on any geometry to generate and view its SHAP explanation
    
    #### 🎛️ **Filtering Options**
    - **POF Range**: Use the slider to focus on specific probability ranges
        - Example: Set to 0.7-1.0 to see only high-risk assets
    - **Label Filter**: Choose what type of observations to display
        - *Failures*: Show only confirmed failures (label=1)
        - *Non-failures*: Show only non-failure cases (label=0)
        - *Both*: Display all observations
    
    #### 🎨 **Visual Elements**
    - **Colorbar**: Located in top-left corner, shows POF scale
        - Colors from dark purple (low POF) to bright yellow (high POF)
        - Scale adjusts automatically based on your POF filter
    - **Debug Panel**: Top-right corner shows current viewport statistics
    
    #### 🔍 **SHAP Explanations**
    - Click any geometry to see why the model made its prediction
    - Waterfall plot shows feature contributions to the POF score
    - Green bars increase probability, red bars decrease it
    
    #### ⚙️ **Configuration**
    - **API Base URL**: Change if your backend is running on a different port
    - **Column Names**: Adjust ID and Label column names to match your data
    - **Limits**: Control maximum features loaded and SHAP features displayed
    
    #### 🚨 **Troubleshooting**
    - If no data appears, check the debug panel for connection status
    - Large datasets may take longer to load - adjust the feature limit
    - SHAP explanations are cached for faster repeated access
    """)

# Footer with additional info
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>🚀 Powered by FastAPI + Streamlit + MapLibre GL + XGBoost + SHAP</p>
    <p>Dynamic viewport loading ensures smooth performance with large datasets</p>
</div>
""", unsafe_allow_html=True)