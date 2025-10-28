"""
Snowflake Native Streamlit App: SHAP Map - Fully Interactive Offline Version
Uses Plotly for interactive maps - NO external resources (CSP compliant).
Click on geometries to see SHAP plots, hover for tooltips.
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
import plotly.graph_objects as go
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
    """Map POF value to RGB color string"""
    if pof_value is None or pof_min >= pof_max:
        return 'rgb(128, 128, 128)'
    norm = mcolors.Normalize(vmin=pof_min, vmax=pof_max)
    cmap = plt.get_cmap('RdYlBu_r')
    rgba = cmap(norm(pof_value))
    return f'rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})'

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
# INTERACTIVE MAP GENERATION
# ============================================================================
def create_interactive_map(gdf_plot, pof_min_val, pof_max_val):
    """Create fully interactive map using Plotly - NO external resources"""
    
    fig = go.Figure()
    
    # Process each geometry and add to map
    for idx, row in gdf_plot.iterrows():
        geom = row.geometry
        segment_id = str(row[ID_COLUMN])
        pof_val = float(row['POF'])
        color = get_color_for_pof(pof_val, pof_min_val, pof_max_val)
        
        # Build hover text
        hover_text = f"<b>ID:</b> {segment_id}<br>"
        hover_text += f"<b>POF:</b> {pof_val:.4f}<br>"
        hover_text += f"<b>Type:</b> {geom.geom_type}<br>"
        
        if "CC Status" in row.index and pd.notna(row["CC Status"]):
            hover_text += f"<b>CC Status:</b> {int(row['CC Status'])}<br>"
        if "CALCULATED_CONDUCTOR_AGE" in row.index and pd.notna(row["CALCULATED_CONDUCTOR_AGE"]):
            hover_text += f"<b>Age:</b> {row['CALCULATED_CONDUCTOR_AGE']:.1f} years<br>"
        if "CONDUCTOR_LENGTH_UDF" in row.index and pd.notna(row["CONDUCTOR_LENGTH_UDF"]):
            hover_text += f"<b>Length:</b> {row['CONDUCTOR_LENGTH_UDF']:.2f}<br>"
        if "CONDUCTOR_DIAMETER_UDF" in row.index and pd.notna(row["CONDUCTOR_DIAMETER_UDF"]):
            hover_text += f"<b>Diameter:</b> {row['CONDUCTOR_DIAMETER_UDF']:.4f}<br>"
        
        hover_text += "<i>Click to see SHAP plot</i>"
        
        # Handle different geometry types
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            
            fig.add_trace(go.Scattermapbox(
                lon=lons,
                lat=lats,
                mode='lines',
                line=dict(width=4, color=color),
                hovertext=hover_text,
                hoverinfo='text',
                name=segment_id,
                showlegend=False,
                customdata=[segment_id] * len(lons)
            ))
        
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = list(line.coords)
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                
                fig.add_trace(go.Scattermapbox(
                    lon=lons,
                    lat=lats,
                    mode='lines',
                    line=dict(width=4, color=color),
                    hovertext=hover_text,
                    hoverinfo='text',
                    name=segment_id,
                    showlegend=False,
                    customdata=[segment_id] * len(lons)
                ))
        
        elif geom.geom_type == 'Point':
            fig.add_trace(go.Scattermapbox(
                lon=[geom.x],
                lat=[geom.y],
                mode='markers',
                marker=dict(size=10, color=color),
                hovertext=hover_text,
                hoverinfo='text',
                name=segment_id,
                showlegend=False,
                customdata=[segment_id]
            ))
        
        elif geom.geom_type == 'Polygon':
            coords = list(geom.exterior.coords)
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            
            fig.add_trace(go.Scattermapbox(
                lon=lons,
                lat=lats,
                mode='lines',
                fill='toself',
                fillcolor=color,
                line=dict(width=2, color='rgb(50,50,50)'),
                hovertext=hover_text,
                hoverinfo='text',
                name=segment_id,
                showlegend=False,
                customdata=[segment_id] * len(lons)
            ))
    
    # Calculate center and bounds
    bounds = gdf_plot.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Configure map layout - using basic style without external tiles
    fig.update_layout(
        mapbox=dict(
            style="white-bg",  # Basic style, no external tiles
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(l=0, r=0, t=30, b=0),
        height=700,
        title=dict(
            text=f"Asset Map - {len(gdf_plot):,} Segments (Click on segments for SHAP analysis)",
            x=0.5,
            xanchor='center'
        )
    )
    
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
st.info("✅ Fully Interactive Mode - Using Plotly (CSP compliant)")

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
# DISPLAY INTERACTIVE MAP
# ============================================================================
st.markdown("### 🗺️ Interactive Map")
st.caption("💡 Hover over segments to see details • Click on segments to generate SHAP plots below")

# Generate and display interactive map
with st.spinner("Generating interactive map..."):
    fig = create_interactive_map(filtered_gdf, pof_filter[0], pof_filter[1])
    
    # Display with Plotly - captures click events
    selected_data = st.plotly_chart(fig, use_container_width=True, key="main_map")

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
# INSTRUCTIONS
# ============================================================================
with st.expander("📖 How to Use"):
    st.markdown("""
    ### Instructions
    
    1. **Explore the Map**: 
       - Use mouse to pan and zoom the interactive map
       - Hover over segments to see tooltips with ID, POF, and other details
       - Use the zoom controls in the top-right of the map
    
    2. **Select Segments**: 
       - Use the dropdown below the map to select a segment
       - SHAP waterfall plot will appear automatically
    
    3. **Adjust Filters**: Use the sidebar to filter segments by:
       - POF range
       - CC Status
       - Conductor Age
       - Conductor Length  
       - Conductor Diameter
    
    ### Understanding the Visualization
    
    - **Map Colors**: Red (high POF/risk) → Yellow → Blue (low POF/risk)
    - **Geometry Types**: Lines (conductors), Points, Polygons all supported
    - **Interactive Controls**: Pan, zoom, hover all work natively
    
    ### Understanding SHAP Plots
    
    - **Red bars**: Features increasing failure probability
    - **Blue bars**: Features decreasing failure probability
    - **Bar length**: Magnitude of feature impact on prediction
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>🚀 Interactive Streamlit App • XGBoost + SHAP + Plotly</p>
    <p>✅ CSP Compliant - No External Map Tiles or Resources</p>
</div>
""", unsafe_allow_html=True)
