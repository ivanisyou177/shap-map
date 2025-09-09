# frontend/streamlit_app.py
"""
Streamlit frontend embedding a MapLibre map (WebGL). The map requests features
in the current viewport via GET /bbox on the backend.

- Shows tooltip on hover with ID + POF
- On click (LineString geometries) fetches /asset/{id}/shap.png and displays in a modal
"""
import os
import streamlit as st

st.set_page_config(layout="wide", page_title="Assets map + on-demand SHAP")

st.title("Assets map – dynamic viewport loading + on-click SHAP (LineString geometries)")

API_BASE = st.sidebar.text_input(
    "Backend API base URL", value=os.environ.get("API_BASE", "http://127.0.0.1:8000")
)
ID_COL = st.sidebar.text_input(
    "ID column name", value=os.environ.get("ID_COL", "id")
)
LIMIT = st.sidebar.number_input(
    "Max features per viewport (server limit)", min_value=100, max_value=200000, value=5000, step=100
)
TOPK = st.sidebar.number_input(
    "Top-K SHAP features (server-side PNG)", min_value=5, max_value=200, value=20, step=1
)

st.markdown(
    """
    - Pan/zoom to change the viewport. The browser asks the backend for only the visible features.
    - Hover over LineString geometries to see ID and POF. Click on a LineString to generate and display the SHAP waterfall (on-demand).
    """
)

# Test backend connection and get debug info
try:
    import requests
    health_response = requests.get(f"{API_BASE}/health", timeout=5)
    if health_response.status_code == 200:
        health_data = health_response.json()
        st.success(f"✅ Backend connected: {health_data}")
        
        # Get debug info
        try:
            # Try basic debug first
            debug_response = requests.get(f"{API_BASE}/debug/basic", timeout=5)
            if debug_response.status_code == 200:
                basic_data = debug_response.json()
                st.write("**🔧 Basic Debug:**", basic_data)
            
            # Then try full debug
            debug_response = requests.get(f"{API_BASE}/debug/sample", timeout=10)
            if debug_response.status_code == 200:
                debug_data = debug_response.json()
                st.info(f"""
**🔍 Debug Info:**
- Total features in DB: {debug_data['total_count']}
- Geometry types: {debug_data['geometry_types']}
- POF range: {debug_data['pof_range'][0]:.4f} - {debug_data['pof_range'][1]:.4f}
- Data bounds: {debug_data['bounds'] if debug_data['bounds'] else 'Unknown'}
- Sample IDs: {debug_data['sample_ids'][:5]}
                """)
                
                # Calculate center point for easy reference
                if debug_data['bounds']:
                    bounds = debug_data['bounds']
                    center_lat = (bounds[1] + bounds[3]) / 2
                    center_lng = (bounds[0] + bounds[2]) / 2
                    st.write(f"📍 **Data center point:** {center_lat:.6f}, {center_lng:.6f}")
                    
                    # Show sample feature for debugging
                    if debug_data['sample_features']['features']:
                        sample_feature = debug_data['sample_features']['features'][0]
                        st.write(f"**Sample feature:** ID={sample_feature['properties']['id']}, Type={sample_feature['geometry']['type']}")
                        if sample_feature['geometry']['type'] == 'LineString':
                            coords = sample_feature['geometry']['coordinates']
                            st.write(f"LineString: {len(coords)} points from [{coords[0][0]:.6f}, {coords[0][1]:.6f}] to [{coords[-1][0]:.6f}, {coords[-1][1]:.6f}]")
                        
            else:
                error_text = debug_response.text if hasattr(debug_response, 'text') else 'Unknown error'
                st.warning(f"⚠️ Could not get debug info: {debug_response.status_code} - {error_text}")
        except Exception as e:
            st.warning(f"⚠️ Debug info error: {e}")
    else:
        st.error(f"❌ Backend health check failed: {health_response.status_code}")
except Exception as e:
    st.error(f"❌ Cannot connect to backend: {e}")

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
  </style>
</head>
<body>
<div id="map"></div>
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

console.log("🗺️ Initializing map with API_BASE:", API_BASE);

function updateDebugPanel(content) {
    document.getElementById('debug-content').innerHTML = content;
}

// Test API connection immediately
fetch(`${API_BASE}/health`)
    .then(r => r.json())
    .then(data => {
        console.log("✅ API Health:", data);
        updateDebugPanel(`✅ API Connected<br>Model: ${data.model_loaded}<br>Data: ${data.data_loaded}`);
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
    
    console.log("🔄 Loading viewport data...");
    const b = map.getBounds();
    const minx = b.getWest();
    const miny = b.getSouth();
    const maxx = b.getEast();
    const maxy = b.getNorth();
    
    const bboxInfo = `Bbox: [${minx.toFixed(4)}, ${miny.toFixed(4)}, ${maxx.toFixed(4)}, ${maxy.toFixed(4)}]`;
    updateDebugPanel(`🔄 Loading...<br>Zoom: ${map.getZoom().toFixed(1)}<br>${bboxInfo}`);
    
    const url = `${API_BASE}/bbox?minx=${minx}&miny=${miny}&maxx=${maxx}&maxy=${maxy}&limit=${LIMIT}`;
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
        updateDebugPanel(`🔍 Features: ${lastFeatureCount}<br>Zoom: ${map.getZoom().toFixed(1)}<br>Center: [${map.getCenter().lng.toFixed(4)}, ${map.getCenter().lat.toFixed(4)}]`);
        
        // Log detailed feature info
        if (geojson.features.length > 0) {
            console.log("📋 Feature details:");
            geojson.features.slice(0, 3).forEach((f, i) => {
                console.log(`Feature ${i}:`, {
                    type: f.geometry.type,
                    id: f.properties[ID_COL],
                    POF: f.properties.POF,
                    coordLength: f.geometry.coordinates ? f.geometry.coordinates.length : 'N/A'
                });
                
                if (f.geometry.type === 'LineString') {
                    const coords = f.geometry.coordinates;
                    console.log(`  🔍 LineString: ${coords.length} points`);
                    console.log(`  🔍 First: [${coords[0][0].toFixed(6)}, ${coords[0][1].toFixed(6)}]`);
                    console.log(`  🔍 Last: [${coords[coords.length-1][0].toFixed(6)}, ${coords[coords.length-1][1].toFixed(6)}]`);
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
            console.log("🔄 Updating existing source");
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
                    'line-color': [
                        'interpolate',
                        ['linear'],
                        ['get', 'POF'],
                        0, '#00ff00',     // Green for low POF
                        0.5, '#ffff00',   // Yellow for medium POF
                        1, '#ff0000'      // Red for high POF
                    ],
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
                    'circle-color': [
                        'interpolate',
                        ['linear'],
                        ['get', 'POF'],
                        0, '#00ff00',
                        0.5, '#ffff00',
                        1, '#ff0000'
                    ],
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
                    'fill-color': [
                        'interpolate',
                        ['linear'],
                        ['get', 'POF'],
                        0, '#00ff00',
                        0.5, '#ffff00',
                        1, '#ff0000'
                    ],
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
                    
                    popup.setLngLat(e.lngLat)
                        .setHTML(`<b>ID:</b> ${id}<br/><b>POF:</b> ${pof}<br/><b>Type:</b> ${f.geometry.type}`)
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
    console.log("🗺️ Map loaded, loading initial viewport data...");
    updateDebugPanel("🗺️ Map loaded<br>Loading data...");
    loadViewportData();
});

map.on('moveend', function() {
    console.log("🗺️ Map moved, loading viewport data...");
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

console.log("✅ Map initialization complete");
</script>
</body>
</html>"""

# replace tokens safely
html = (
    HTML_TEMPLATE
    .replace("%%API_BASE%%", API_BASE)
    .replace("%%LIMIT%%", str(LIMIT))
    .replace("%%TOPK%%", str(TOPK))
    .replace("%%ID_COL%%", ID_COL)
)

# render HTML in Streamlit
st.components.v1.html(html, height=800, scrolling=False)