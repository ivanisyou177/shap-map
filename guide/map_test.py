"""
Snowflake Native Streamlit App: SHAP Map - Completely Offline Version
NO external resources - works with strict CSP policies.
Uses only matplotlib for visualization.
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
import matplotlib.colors as colors
import io

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
def get_color_for_pof(pof_value: float, pof_min: float, pof_max: float):
    """Map POF value to RGB color"""
    if pof_value is None or pof_min >= pof_max:
        return (0.5, 0.5, 0.5)
    norm = colors.Normalize(vmin=pof_min, vmax=pof_max)
    cmap = plt.get_cmap('RdYlBu_r')
    return cmap(norm(pof_value))

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
# MAP GENERATION
# ============================================================================
def create_map_image(gdf_plot, pof_min_val, pof_max_val, figsize=(18, 14)):
    """Create static map using only matplotlib - NO external resources"""
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    
    # Set background
    ax.set_facecolor('#e8e8e8')
    fig.patch.set_facecolor('white')
    
    # Get colormap
    norm = colors.Normalize(vmin=pof_min_val, vmax=pof_max_val)
    cmap = plt.get_cmap('RdYlBu_r')
    
    # Plot each geometry
    for idx, row in gdf_plot.iterrows():
        geom = row.geometry
        pof_val = row['POF']
        color = cmap(norm(pof_val))
        
        if geom.geom_type == 'LineString':
            x, y = geom.xy
            ax.plot(x, y, color=color, linewidth=4, alpha=0.85, 
                   solid_capstyle='round', zorder=2)
        
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                x, y = line.xy
                ax.plot(x, y, color=color, linewidth=4, alpha=0.85, 
                       solid_capstyle='round', zorder=2)
        
        elif geom.geom_type == 'Point':
            ax.plot(geom.x, geom.y, 'o', color=color, markersize=10, 
                   markeredgecolor='white', markeredgewidth=2, 
                   alpha=0.9, zorder=3)
        
        elif geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            ax.fill(x, y, color=color, alpha=0.65, edgecolor='#333', 
                   linewidth=1.5, zorder=1)
    
    # Set bounds with margin
    bounds = gdf_plot.total_bounds
    margin_x = (bounds[2] - bounds[0]) * 0.02
    margin_y = (bounds[3] - bounds[1]) * 0.02
    ax.set_xlim(bounds[0] - margin_x, bounds[2] + margin_x)
    ax.set_ylim(bounds[1] - margin_y, bounds[3] + margin_y)
    ax.set_aspect('equal')
    
    # Styling
    ax.set_xlabel('Longitude', fontsize=13, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=13, fontweight='bold')
    ax.set_title(f'Asset Map - {len(gdf_plot):,} Segments Displayed', 
                fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='#666')
    ax.tick_params(labelsize=10)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', 
                        pad=0.08, fraction=0.046, aspect=40)
    cbar.set_label('POF (Probability of Failure)', 
                   fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add statistics text
    stats_text = (
        f"Min POF: {pof_min_val:.4f}\n"
        f"Max POF: {pof_max_val:.4f}\n"
        f"Segments: {len(gdf_plot):,}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    return fig

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
st.info("⚠️ Offline Mode - No external map tiles (CSP compliant)")

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
    step=100,
    help="Limit for performance - show top N by POF"
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
# DISPLAY MAP
# ============================================================================
st.markdown("### 🗺️ Static Map View")

# Calculate bounds for info
bounds = filtered_gdf.total_bounds
center_lat = (bounds[1] + bounds[3]) / 2
center_lng = (bounds[0] + bounds[2]) / 2

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Map Center", f"{center_lat:.4f}, {center_lng:.4f}")
with col2:
    st.metric("Longitude Range", f"{bounds[0]:.4f} to {bounds[2]:.4f}")
with col3:
    st.metric("Latitude Range", f"{bounds[1]:.4f} to {bounds[3]:.4f}")

# Generate and display main map
with st.spinner("Rendering map..."):
    fig_main = create_map_image(filtered_gdf, pof_filter[0], pof_filter[1])
    st.pyplot(fig_main)
    plt.close(fig_main)

# Display colorbar
st.markdown("### 🎨 POF Color Scale")
colorbar_png = generate_colorbar(pof_filter[0], pof_filter[1])
st.image(colorbar_png, use_container_width=False)

# ============================================================================
# ZOOM CONTROLS
# ============================================================================
st.markdown("---")
st.markdown("### 🔍 Zoom Controls")

col1, col2 = st.columns(2)

with col1:
    lon_range = st.slider(
        "Longitude Range",
        min_value=float(bounds[0]),
        max_value=float(bounds[2]),
        value=(float(bounds[0]), float(bounds[2])),
        step=(bounds[2] - bounds[0]) / 100,
        format="%.5f",
        key="lon_zoom"
    )

with col2:
    lat_range = st.slider(
        "Latitude Range",
        min_value=float(bounds[1]),
        max_value=float(bounds[3]),
        value=(float(bounds[1]), float(bounds[3])),
        step=(bounds[3] - bounds[1]) / 100,
        format="%.5f",
        key="lat_zoom"
    )

# Apply zoom if changed
if (lon_range[0] > bounds[0] or lon_range[1] < bounds[2] or 
    lat_range[0] > bounds[1] or lat_range[1] < bounds[3]):
    
    zoomed_gdf = filtered_gdf.cx[lon_range[0]:lon_range[1], lat_range[0]:lat_range[1]]
    
    if len(zoomed_gdf) > 0:
        st.markdown("#### 🔎 Zoomed View")
        st.info(f"Showing {len(zoomed_gdf):,} of {len(filtered_gdf):,} segments in zoomed area")
        
        with st.spinner("Rendering zoomed map..."):
            fig_zoom = create_map_image(zoomed_gdf, pof_filter[0], pof_filter[1], figsize=(18, 12))
            st.pyplot(fig_zoom)
            plt.close(fig_zoom)
    else:
        st.warning("No segments in selected zoom area")

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
# INSTRUCTIONS
# ============================================================================
with st.expander("📖 How to Use"):
    st.markdown("""
    ### Instructions
    
    1. **View the Map**: The static map shows all filtered segments
    2. **Adjust Filters**: Use the sidebar to filter by POF, CC Status, age, length, and diameter
    3. **Zoom In**: Use the longitude/latitude sliders to zoom into specific areas
    4. **Select Segments**: Use the dropdown to analyze specific segments with SHAP
    
    ### Understanding the Map
    
    - **Colors**: Red (high POF) to Blue (low POF)
    - **Geometry Types**: Lines, points, and polygons all supported
    - **Zoom View**: Adjusting zoom sliders creates a new detailed view below
    
    ### Understanding SHAP Plots
    
    - **Red bars**: Features that increase the probability of failure
    - **Blue bars**: Features that decrease the probability of failure
    - **Bar length**: Shows the magnitude of each feature's impact
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>🚀 Offline Streamlit App • XGBoost + SHAP + Matplotlib</p>
    <p>✅ CSP Compliant - No External Resources</p>
</div>
""", unsafe_allow_html=True)
