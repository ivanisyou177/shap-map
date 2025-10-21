"""
Snowflake Native Streamlit App: SHAP Map with Multi-Model Support
Fully integrated geospatial ML application running on Snowflake compute.

Features:
- Multi-model support (equipment and structure models)
- Dynamic viewport loading with spatial filtering
- SHAP explanations generated on-demand
- POF range, label, CC Status, and conductor length filtering
- All data and models loaded from Snowflake stages
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import json
import io
import tempfile
import os
from typing import Optional, Dict, Any, List, Tuple
from functools import lru_cache

# Snowflake
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F

# Data processing
import pandas as pd
import geopandas as gpd
import numpy as np

# Geospatial
from shapely.geometry import box, shape as geom_shape, mapping
from shapely import wkt

# ML
from xgboost import XGBClassifier
import shap

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    layout="wide",
    page_title="SHAP Map Analysis - Snowflake Native",
    page_icon="🗺️"
)

# ============================================================================
# SNOWFLAKE CONNECTION SETUP
# ============================================================================
@st.cache_resource
def get_snowflake_session():
    """Create and cache Snowflake session"""
    try:
        # Try to get current session (when running in Snowflake)
        session = st.connection("snowflake").session()
        return session
    except:
        # Fallback for local testing with secrets
        if "snowflake" in st.secrets:
            connection_parameters = {
                "account": st.secrets["snowflake"]["account"],
                "user": st.secrets["snowflake"]["user"],
                "password": st.secrets["snowflake"]["password"],
                "role": st.secrets["snowflake"]["role"],
                "warehouse": st.secrets["snowflake"]["warehouse"],
                "database": st.secrets["snowflake"]["database"],
                "schema": st.secrets["snowflake"]["schema"],
            }
            return Session.builder.configs(connection_parameters).create()
        else:
            st.error("❌ No Snowflake connection available. Please configure secrets.toml")
            st.stop()

# Initialize session
session = get_snowflake_session()

# Configure stage paths (UPDATE THESE WITH YOUR ACTUAL VALUES)
STAGE_NAME = "ML_ASSETS"
DATABASE_NAME = st.secrets.get("snowflake", {}).get("database", "SHAP_MAP_DB")
SCHEMA_NAME = st.secrets.get("snowflake", {}).get("schema", "GEOSPATIAL_ML")
STAGE_PATH = f"@{DATABASE_NAME}.{SCHEMA_NAME}.{STAGE_NAME}"

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================
MODEL_CONFIGS = {
    "equipment": {
        "name": "OCP Wiredown Model",
        "description": "OCP Wiredown Model SHAP analysis",
        "model_path": f"{STAGE_PATH}/models/xgboost_model_v3.json",
        "data_path": f"{STAGE_PATH}/data/ocp_wiredown_model_data.gpkg",
        "feature_names_path": f"{STAGE_PATH}/models/feature_names.json",
        "id_col": "ID",
        "label_col": "HAS_WIREDOWN",
        "geometry_type": "equipment",
        "structure_id_col": None,
        "visualization_type": "single_click"
    },
    "structure": {
        "name": "Transformer Model",
        "description": "Transformer Model SHAP analysis",
        "model_path": f"{STAGE_PATH}/models/transformer_model.bst",
        "data_path": f"{STAGE_PATH}/data/transformer_model_data.gpkg",
        "feature_names_path": f"{STAGE_PATH}/models/structure_feature_names.json",
        "id_col": "Equipment",
        "label_col": "Model_GROUP_Group0",
        "geometry_type": "structure",
        "structure_id_col": "FLOC_FunctionalLocationID",
        "geometry_col": "geometry",
        "visualization_type": "multi_click"
    }
}

# ============================================================================
# FILE LOADING UTILITIES
# ============================================================================
def get_file_from_stage(stage_path: str) -> bytes:
    """Download file from Snowflake stage to memory"""
    try:
        # Use session to get file
        result = session.file.get(stage_path, "/tmp")
        # Read the downloaded file
        local_path = f"/tmp/{os.path.basename(stage_path)}"
        with open(local_path, 'rb') as f:
            content = f.read()
        # Clean up
        os.remove(local_path)
        return content
    except Exception as e:
        st.error(f"❌ Error loading file from {stage_path}: {e}")
        raise

def load_json_from_stage(stage_path: str) -> dict:
    """Load JSON file from Snowflake stage"""
    content = get_file_from_stage(stage_path)
    return json.loads(content.decode('utf-8'))

def load_geopackage_from_stage(stage_path: str) -> gpd.GeoDataFrame:
    """Load GeoPackage from Snowflake stage"""
    content = get_file_from_stage(stage_path)
    
    # Write to temporary file (geopandas needs file path)
    with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        gdf = gpd.read_file(tmp_path)
        return gdf
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ============================================================================
# GEOMETRY UTILITIES
# ============================================================================
def parse_geometry(val):
    """Parse geometry from WKT string, GeoJSON dict, or shapely geometry"""
    if val is None:
        return None
    if hasattr(val, "geom_type"):
        return val
    if isinstance(val, str):
        s = val.strip()
        try:
            return wkt.loads(s)
        except Exception:
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict) and "type" in parsed and "coordinates" in parsed:
                    return geom_shape(parsed)
            except Exception:
                raise ValueError(f"Unsupported geometry string format: {s[:80]}...")
    if isinstance(val, dict):
        try:
            return geom_shape(val)
        except Exception as e:
            raise ValueError(f"Invalid geometry dict: {e}")
    raise ValueError(f"Unsupported geometry type: {type(val)}")

def get_color_for_pof(pof_value: float, pof_min: float, pof_max: float) -> str:
    """Map POF value to hex color using perceptually uniform colormap"""
    if pof_value is None or pof_min is None or pof_max is None or pof_min >= pof_max:
        return "#808080"
    
    norm = colors.Normalize(vmin=pof_min, vmax=pof_max)
    cmap = plt.get_cmap('RdYlBu_r')
    rgba_color = cmap(norm(pof_value))
    return colors.to_hex(rgba_color)

# ============================================================================
# MODEL AND DATA LOADING
# ============================================================================
@st.cache_resource
def load_model_and_data(model_key: str):
    """Load model and data for specified configuration from Snowflake stage"""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}")
    
    config = MODEL_CONFIGS[model_key].copy()
    
    with st.spinner(f"Loading {config['name']} from Snowflake..."):
        # Load model
        st.write(f"📥 Loading model from {config['model_path']}")
        model_content = get_file_from_stage(config['model_path'])
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp.write(model_content)
            model_path = tmp.name
        
        try:
            model = XGBClassifier()
            model.load_model(model_path)
        finally:
            os.remove(model_path)
        
        # Load feature names
        st.write(f"📥 Loading feature names from {config['feature_names_path']}")
        feature_names = load_json_from_stage(config['feature_names_path'])
        if isinstance(feature_names, dict):
            feature_names = list(feature_names.values())
        
        # Load data
        st.write(f"📥 Loading data from {config['data_path']}")
        gdf = load_geopackage_from_stage(config['data_path'])
        
        # Reproject to EPSG:4326 if needed
        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            st.write(f"🔄 Reprojecting from {gdf.crs.to_string()} to EPSG:4326")
            gdf = gdf.to_crs(epsg=4326)
        
        # Verify features
        missing = [c for c in feature_names if c not in gdf.columns]
        if missing:
            raise RuntimeError(f"Dataset missing features: {missing[:10]}")
        
        # Compute POF
        st.write("🔮 Computing POF values...")
        X_all = gdf[feature_names].astype(float)
        gdf["POF"] = model.predict_proba(X_all)[:, 1].astype(float)
        
        # Store ranges
        pof_min = float(gdf["POF"].min())
        pof_max = float(gdf["POF"].max())
        
        # Conductor length range
        conductor_length_range = None
        if "CONDUCTOR_LENGTH_UDF" in gdf.columns:
            valid_lengths = gdf["CONDUCTOR_LENGTH_UDF"].dropna()
            if not valid_lengths.empty:
                conductor_length_range = (float(valid_lengths.min()), float(valid_lengths.max()))
        
        # Create spatial index
        _ = gdf.sindex
        
        # Create SHAP explainer
        st.write("🔬 Initializing SHAP explainer...")
        sample_size = min(int(len(gdf) * 0.7), 1000)
        explainer = shap.TreeExplainer(
            model,
            data=gdf[feature_names].iloc[:sample_size],
            model_output="probability"
        )
        
        st.success(f"✅ Loaded {len(gdf):,} records with POF range [{pof_min:.4f}, {pof_max:.4f}]")
        
        return {
            "config": config,
            "model": model,
            "explainer": explainer,
            "gdf": gdf,
            "feature_names": feature_names,
            "pof_range": (pof_min, pof_max),
            "conductor_length_range": conductor_length_range
        }

# ============================================================================
# DATA FILTERING AND QUERYING
# ============================================================================
def get_bbox_features(
    gdf: gpd.GeoDataFrame,
    minx: float, miny: float, maxx: float, maxy: float,
    config: dict,
    pof_range: Tuple[float, float],
    limit: int = 5000,
    pof_filter: Optional[Tuple[float, float]] = None,
    label_filter: str = "both",
    cc_status_filter: Optional[int] = None,
    conductor_length_filter: Optional[Tuple[float, float]] = None
) -> List[dict]:
    """Get features within bounding box with filters"""
    
    # Spatial filter
    bbox_geom = box(minx, miny, maxx, maxy)
    indices = list(gdf.sindex.intersection((minx, miny, maxx, maxy)))
    subset = gdf.iloc[indices].copy() if indices else gdf.copy()
    subset = subset[subset.geometry.intersects(bbox_geom)]
    
    if subset.empty:
        return []
    
    # POF filter
    if pof_filter:
        pof_min_f, pof_max_f = pof_filter
        if pof_min_f is not None:
            subset = subset[subset["POF"] >= pof_min_f]
        if pof_max_f is not None:
            subset = subset[subset["POF"] <= pof_max_f]
    
    # Label filter
    label_col = config["label_col"]
    if label_filter != "both" and label_col in subset.columns:
        if label_filter == "failures":
            subset = subset[subset[label_col] == 1]
        elif label_filter == "non_failures":
            subset = subset[subset[label_col] == 0]
    
    # CC Status filter
    if cc_status_filter is not None and "CC Status" in subset.columns:
        subset = subset[subset["CC Status"] == cc_status_filter]
    
    # Conductor length filter
    if conductor_length_filter and "CONDUCTOR_LENGTH_UDF" in subset.columns:
        len_min, len_max = conductor_length_filter
        if len_min is not None:
            subset = subset[subset["CONDUCTOR_LENGTH_UDF"] >= len_min]
        if len_max is not None:
            subset = subset[subset["CONDUCTOR_LENGTH_UDF"] <= len_max]
    
    if subset.empty:
        return []
    
    # Handle structure aggregation
    if config["geometry_type"] == "structure":
        structure_id_col = config["structure_id_col"]
        if structure_id_col in subset.columns:
            structure_groups = subset.groupby(structure_id_col).agg({
                "POF": "max",
                "geometry": "first",
                label_col: lambda x: (x == 1).any() if label_col in subset.columns else 0
            }).reset_index()
            subset = structure_groups
            subset[config["id_col"]] = subset[structure_id_col]
    
    # Limit results
    if len(subset) > limit:
        subset = subset.nlargest(limit, "POF")
    
    # Determine color range
    color_min, color_max = pof_filter if pof_filter else pof_range
    
    # Build features
    features = []
    id_col = config["id_col"]
    
    for _, row in subset.iterrows():
        if row.geometry is None:
            continue
        
        pof_val = float(row.get("POF", 0))
        
        props = {
            id_col: str(row.get(id_col)),
            "id": str(row.get(id_col)),
            "POF": pof_val,
            "color": get_color_for_pof(pof_val, color_min, color_max),
            "model_type": config["geometry_type"],
            "visualization_type": config["visualization_type"]
        }
        
        # Add structure info
        if config["geometry_type"] == "structure":
            structure_id_col = config["structure_id_col"]
            props["structure_id"] = str(row.get(structure_id_col, row.get(id_col)))
        
        # Add label
        if label_col in gdf.columns:
            props[label_col] = int(row.get(label_col, 0))
        
        # Add CC Status
        if "CC Status" in gdf.columns and "CC Status" in row.index:
            props["CC_Status"] = int(row.get("CC Status", 0))
        
        # Add conductor length
        if "CONDUCTOR_LENGTH_UDF" in gdf.columns and "CONDUCTOR_LENGTH_UDF" in row.index:
            conductor_val = row.get("CONDUCTOR_LENGTH_UDF")
            props["CONDUCTOR_LENGTH_UDF"] = float(conductor_val) if pd.notna(conductor_val) else None
        
        features.append({
            "type": "Feature",
            "geometry": mapping(row.geometry),
            "properties": props
        })
    
    return features

# ============================================================================
# SHAP VISUALIZATION
# ============================================================================
def generate_shap_plot(
    asset_id: str,
    gdf: gpd.GeoDataFrame,
    model,
    explainer,
    feature_names: List[str],
    config: dict,
    top_k: int = 20
) -> bytes:
    """Generate SHAP waterfall plot for an asset"""
    
    id_col = config["id_col"]
    sel = gdf[gdf[id_col].astype(str) == str(asset_id)]
    
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
    
    try:
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(single_explanation, max_display=int(top_k), show=False)
        fig = plt.gcf()
        
        title = f"SHAP waterfall - {id_col} {asset_id} - POF: {pof_value:.4f}"
        fig.suptitle(title, fontsize=12)
        
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    
    except Exception as e:
        # Fallback implementation
        vals = np.array(single_explanation.values).reshape(-1)
        base_val = float(single_explanation.base_values)
        k = min(int(top_k), max(1, len(vals)))
        idx = np.argsort(-np.abs(vals))[:k]
        
        chosen_vals = vals[idx]
        chosen_names = [feature_names[i] for i in idx]
        
        starts, ends = [], []
        cum = base_val
        for v in chosen_vals:
            starts.append(cum)
            cum += float(v)
            ends.append(cum)
        
        colors_bar = ['#d62728' if v >= 0 else "#2E94E7" for v in chosen_vals]
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        y_pos = np.arange(len(chosen_vals))[::-1]
        
        for i, (y, s, e, c) in enumerate(zip(y_pos, starts, ends, colors_bar)):
            ax.barh(y, e - s, left=s, height=0.7, color=c)
            ax.text(e + 0.005*(max(ends)-min(starts)+1), y, f"{chosen_vals[i]:+.3f}", va='center', fontsize=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(chosen_names, fontsize=10)
        ax.set_xlabel("SHAP contribution (to probability)")
        ax.set_title(f"SHAP waterfall - {id_col} {asset_id} - POF: {pof_value:.4f}")
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

def generate_colorbar(pof_min: float, pof_max: float, width: int = 300, height: int = 40) -> bytes:
    """Generate colorbar PNG for POF range"""
    
    if pof_min >= pof_max:
        raise ValueError(f"Invalid POF range: min ({pof_min}) >= max ({pof_max})")
    
    fig_width = max(4, width / 75)
    fig_height = max(0.6, height / 75)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    
    cmap = plt.get_cmap("RdYlBu_r")
    norm = colors.Normalize(vmin=pof_min, vmax=pof_max)
    
    gradient = np.linspace(pof_min, pof_max, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=cmap, norm=norm, extent=[pof_min, pof_max, 0, 1])
    
    ax.set_xlim(pof_min, pof_max)
    ax.set_yticks([])
    ax.set_xlabel("POF (Probability of Failure)", fontsize=10, fontweight='bold')
    
    if pof_max - pof_min < 0.01:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.4f}'))
    elif pof_max - pof_min < 0.1:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    
    ax.tick_params(axis='x', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, 
               transparent=False, facecolor='white', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()

# ============================================================================
# STREAMLIT UI
# ============================================================================

# Sidebar
st.sidebar.title("🚀 SHAP Map - Snowflake Native")
st.sidebar.markdown(f"**Connected to:** {DATABASE_NAME}.{SCHEMA_NAME}")
st.sidebar.markdown("---")

# Model selection
st.sidebar.markdown("### 🔧 Model Selection")

if 'current_model_key' not in st.session_state:
    st.session_state.current_model_key = "equipment"

selected_model = st.sidebar.selectbox(
    "Select Model",
    options=list(MODEL_CONFIGS.keys()),
    index=list(MODEL_CONFIGS.keys()).index(st.session_state.current_model_key),
    format_func=lambda x: MODEL_CONFIGS[x]["name"],
    key="model_selector"
)

# Load model if changed
if selected_model != st.session_state.current_model_key:
    st.session_state.current_model_key = selected_model
    # Clear cache to force reload
    st.cache_resource.clear()
    st.rerun()

# Load current model
try:
    model_data = load_model_and_data(selected_model)
    config = model_data["config"]
    gdf = model_data["gdf"]
    model = model_data["model"]
    explainer = model_data["explainer"]
    feature_names = model_data["feature_names"]
    pof_range = model_data["pof_range"]
    conductor_length_range = model_data["conductor_length_range"]
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Display settings
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Display Settings")

LIMIT = st.sidebar.number_input(
    "Max assets per viewport",
    min_value=100,
    max_value=50000,
    value=5000,
    step=100
)

TOPK = st.sidebar.number_input(
    "Top-K SHAP features",
    min_value=5,
    max_value=200,
    value=20,
    step=1
)

# Filters
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Data Filters")

# POF filter
st.sidebar.markdown("#### 📈 POF Range Filter")
pof_filter = st.sidebar.slider(
    "Select POF range",
    min_value=float(pof_range[0]),
    max_value=float(pof_range[1]),
    value=(float(pof_range[0]), float(pof_range[1])),
    step=0.001,
    format="%.3f"
)

# Label filter
label_col = config["label_col"]
if label_col in gdf.columns:
    st.sidebar.markdown("#### 🏷️ Label Filter")
    label_options = {
        "both": "Both (all data)",
        "failures": "Failures only (label=1)",
        "non_failures": "Non-failures only (label=0)"
    }
    label_filter = st.sidebar.selectbox(
        f"Filter by {label_col}",
        options=list(label_options.keys()),
        format_func=lambda x: label_options[x]
    )
else:
    label_filter = "both"

# CC Status filter
cc_status_filter = None
if "CC Status" in gdf.columns:
    st.sidebar.markdown("#### 🔌 CC Status Filter")
    cc_dist = gdf["CC Status"].value_counts().to_dict()
    st.sidebar.caption(f"Distribution: {cc_dist}")
    
    cc_options = {"both": "Both", 0: "Status 0", 1: "Status 1"}
    cc_selection = st.sidebar.selectbox(
        "Filter by CC Status",
        options=list(cc_options.keys()),
        format_func=lambda x: cc_options[x]
    )
    if cc_selection != "both":
        cc_status_filter = int(cc_selection)

# Conductor length filter
conductor_length_filter = None
if conductor_length_range:
    st.sidebar.markdown("#### 📏 Conductor Length Filter")
    st.sidebar.caption(f"Range: {conductor_length_range[0]:.2f} - {conductor_length_range[1]:.2f}")
    
    length_range = st.sidebar.slider(
        "Select length range",
        min_value=float(conductor_length_range[0]),
        max_value=float(conductor_length_range[1]),
        value=(float(conductor_length_range[0]), float(conductor_length_range[1])),
        step=(conductor_length_range[1] - conductor_length_range[0]) / 100,
        format="%.2f"
    )
    conductor_length_filter = length_range

# Main title
st.title("🗺️ SHAP Map - Snowflake Native Application")

# Model info
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Current Model", config["name"])

with col2:
    st.metric("Total Assets", f"{len(gdf):,}")
    st.metric("POF Range", f"{pof_range[0]:.3f} - {pof_range[1]:.3f}")

with col3:
    geom_types = gdf.geometry.geom_type.value_counts().to_dict()
    st.metric("Geometry Types", len(geom_types))
    for geom_type, count in geom_types.items():
        st.write(f"• {geom_type}: {count:,}")

# Structure-specific info
if config["visualization_type"] == "multi_click":
    structure_id_col = config["structure_id_col"]
    if structure_id_col in gdf.columns:
        structure_counts = gdf.groupby(structure_id_col).size()
        st.info(f"🏗️ **Structure Model**: {len(structure_counts):,} structures with "
               f"{structure_counts.mean():.1f} equipment on average "
               f"(range: {structure_counts.min()}-{structure_counts.max()})")

# Label distribution
if label_col in gdf.columns:
    label_dist = gdf[label_col].value_counts().to_dict()
    failures = int(label_dist.get(1, 0))
    non_failures = int(label_dist.get(0, 0))
    total_labeled = failures + non_failures
    if total_labeled > 0:
        failure_rate = failures / total_labeled * 100
        st.success(f"📊 **Labels**: {failure_rate:.1f}% failure rate ({failures:,} failures, {non_failures:,} non-failures)")

# Data center
bounds = gdf.total_bounds
center_lat = (bounds[1] + bounds[3]) / 2
center_lng = (bounds[0] + bounds[2]) / 2
st.info(f"🎯 **Data center**: {center_lat:.6f}, {center_lng:.6f}")

# Instructions
viz_type = config["visualization_type"]
if viz_type == "single_click":
    st.markdown("""
    **Equipment Model Instructions:**
    - Pan/zoom to explore - viewport loading keeps performance smooth
    - Hover over equipment to see ID and POF values
    - Click on any equipment to generate its SHAP explanation
    - Use filters to focus on specific POF ranges, failure types, CC Status, or conductor lengths
    """)
else:
    st.markdown("""
    **Structure Model Instructions:**
    - Pan/zoom to explore structures - each may contain multiple equipment
    - Hover over structures to see aggregate information
    - Click on a structure to see SHAP explanations for ALL equipment within it
    - Multiple SHAP plots will appear below the map
    - Use filters to focus on structures with specific characteristics
    """)

# ============================================================================
# MAP RENDERING
# ============================================================================

st.markdown("---")
st.subheader("🗺️ Interactive Map")

# Initialize session state for map interaction
if 'clicked_asset' not in st.session_state:
    st.session_state.clicked_asset = None
if 'map_bounds' not in st.session_state:
    # Default bounds
    st.session_state.map_bounds = {
        'minx': bounds[0],
        'miny': bounds[1],
        'maxx': bounds[2],
        'maxy': bounds[3]
    }

# Map HTML template
MAP_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SHAP Map</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet">
    <style>
        body { margin:0; padding:0; font-family: Arial, sans-serif; }
        #map { position: absolute; top:0; bottom:0; width:100%; height:600px; }
        .info-panel {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.95);
            border-radius: 5px;
            padding: 10px;
            font-size: 11px;
            max-width: 250px;
            z-index: 1000;
        }
        .maplibregl-popup-content { padding: 8px 12px; max-width: 250px; }
    </style>
</head>
<body>
<div id="map"></div>
<div id="info-panel" class="info-panel">
    <strong>Debug Info</strong><br>
    <div id="info-content">Loading...</div>
</div>

<script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
<script>
const GEOJSON_DATA = %%GEOJSON_DATA%%;
const CENTER = %%CENTER%%;
const MODEL_NAME = "%%MODEL_NAME%%";
const VIZ_TYPE = "%%VIZ_TYPE%%";

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
    zoom: 10
});

map.addControl(new maplibregl.NavigationControl());

let popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false });

map.on('load', function() {
    map.addSource('assets', {
        type: 'geojson',
        data: GEOJSON_DATA
    });

    // LineString layer
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

    // Point layer
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

    // Polygon layer
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

    const layers = ['linestring-layer', 'point-layer', 'polygon-fill-layer'];
    
    layers.forEach(layerId => {
        map.on('mouseenter', layerId, function(e) {
            map.getCanvas().style.cursor = 'pointer';
            const f = e.features[0];
            if (!f) return;
            
            const props = f.properties;
            const id = props.id || "n/a";
            const pof = props.POF !== undefined ? Number(props.POF).toFixed(4) : "n/a";
            
            let html = `<b>ID:</b> ${id}<br><b>POF:</b> ${pof}`;
            
            if (props.structure_id) {
                html += `<br><b>Structure:</b> ${props.structure_id}`;
            }
            
            html += `<br><i>Click for SHAP</i>`;
            
            popup.setLngLat(e.lngLat).setHTML(html).addTo(map);
        });
        
        map.on('mouseleave', layerId, function() {
            map.getCanvas().style.cursor = '';
            popup.remove();
        });
        
        map.on('click', layerId, function(e) {
            const f = e.features[0];
            if (!f) return;
            
            const id = f.properties.id || f.properties.ID;
            if (id) {
                window.parent.postMessage({
                    type: 'asset_clicked',
                    asset_id: id,
                    viz_type: VIZ_TYPE
                }, '*');
            }
        });
    });

    const featureCount = GEOJSON_DATA.features.length;
    document.getElementById('info-content').innerHTML = 
        `Model: ${MODEL_NAME}<br>Features: ${featureCount}<br>Zoom: ${map.getZoom().toFixed(1)}`;
});

map.on('moveend', function() {
    const center = map.getCenter();
    const zoom = map.getZoom();
    document.getElementById('info-content').innerHTML = 
        `Model: ${MODEL_NAME}<br>Features: ${GEOJSON_DATA.features.length}<br>Zoom: ${zoom.toFixed(1)}<br>Lat: ${center.lat.toFixed(4)}<br>Lng: ${center.lng.toFixed(4)}`;
});
</script>
</body>
</html>
"""

# Get features for current viewport
map_bounds = st.session_state.map_bounds
features = get_bbox_features(
    gdf=gdf,
    minx=map_bounds['minx'],
    miny=map_bounds['miny'],
    maxx=map_bounds['maxx'],
    maxy=map_bounds['maxy'],
    config=config,
    pof_range=pof_range,
    limit=LIMIT,
    pof_filter=pof_filter,
    label_filter=label_filter,
    cc_status_filter=cc_status_filter,
    conductor_length_filter=conductor_length_filter
)

geojson = {
    "type": "FeatureCollection",
    "features": features
}

# Render map
map_html = (
    MAP_HTML
    .replace("%%GEOJSON_DATA%%", json.dumps(geojson))
    .replace("%%CENTER%%", f"[{center_lng}, {center_lat}]")
    .replace("%%MODEL_NAME%%", config["name"])
    .replace("%%VIZ_TYPE%%", config["visualization_type"])
)

# Display map
st.components.v1.html(map_html, height=650, scrolling=False)

# Display colorbar
st.markdown("### 🎨 POF Color Scale")
colorbar_png = generate_colorbar(pof_filter[0], pof_filter[1])
st.image(colorbar_png, use_container_width=False)

# ============================================================================
# SHAP VISUALIZATION SECTION
# ============================================================================

st.markdown("---")
st.subheader("🔬 SHAP Explanations")

# Asset selection for SHAP
id_col = config["id_col"]
selected_asset = st.selectbox(
    f"Select {id_col} for SHAP analysis",
    options=[""] + sorted(gdf[id_col].astype(str).unique().tolist()),
    help="Choose an asset to generate its SHAP explanation"
)

if selected_asset:
    if config["visualization_type"] == "single_click":
        # Single SHAP plot
        with st.spinner(f"Generating SHAP explanation for {selected_asset}..."):
            try:
                shap_png = generate_shap_plot(
                    asset_id=selected_asset,
                    gdf=gdf,
                    model=model,
                    explainer=explainer,
                    feature_names=feature_names,
                    config=config,
                    top_k=TOPK
                )
                st.image(shap_png, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Failed to generate SHAP plot: {e}")
    
    else:
        # Multi-SHAP for structure
        structure_id_col = config["structure_id_col"]
        structure_equipment = gdf[gdf[structure_id_col].astype(str) == selected_asset]
        
        if not structure_equipment.empty:
            st.info(f"🏗️ Structure {selected_asset} contains {len(structure_equipment)} equipment units")
            
            for idx, (_, row) in enumerate(structure_equipment.iterrows()):
                equipment_id = str(row[id_col])
                equipment_pof = float(row["POF"])
                
                with st.expander(f"Equipment {equipment_id} - POF: {equipment_pof:.4f}", expanded=(idx < 3)):
                    try:
                        shap_png = generate_shap_plot(
                            asset_id=equipment_id,
                            gdf=gdf,
                            model=model,
                            explainer=explainer,
                            feature_names=feature_names,
                            config=config,
                            top_k=TOPK
                        )
                        st.image(shap_png, use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Failed to generate SHAP plot for {equipment_id}: {e}")
        else:
            st.warning(f"⚠️ No equipment found for structure {selected_asset}")

# ============================================================================
# STATISTICS AND INFORMATION
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 Active Configuration")
    st.write(f"**Model:** {config['name']}")
    st.write(f"**POF Filter:** {pof_filter[0]:.3f} - {pof_filter[1]:.3f}")
    st.write(f"**Label Filter:** {label_filter}")
    if cc_status_filter is not None:
        st.write(f"**CC Status:** {cc_status_filter}")
    if conductor_length_filter:
        st.write(f"**Length:** {conductor_length_filter[0]:.2f} - {conductor_length_filter[1]:.2f}")

with col2:
    st.markdown("### 📊 Dataset Statistics")
    st.write(f"**Total Assets:** {len(gdf):,}")
    st.write(f"**Visible Features:** {len(features):,}")
    st.write(f"**POF Range:** {pof_range[0]:.3f} - {pof_range[1]:.3f}")
    
    if config["visualization_type"] == "multi_click" and structure_id_col in gdf.columns:
        structure_counts = gdf.groupby(structure_id_col).size()
        st.write(f"**Structures:** {len(structure_counts):,}")

with col3:
    st.markdown("### 🗺️ Instructions")
    if viz_type == "single_click":
        st.write("**Equipment Model:**")
        st.write("• Hover: View details")
        st.write("• Click: Single SHAP")
        st.write("• Use dropdown below")
    else:
        st.write("**Structure Model:**")
        st.write("• Hover: Structure info")
        st.write("• Click: All equipment")
        st.write("• Use dropdown below")

# Expandable help sections
with st.expander("📖 User Guide"):
    st.markdown("""
    ### Using This Application
    
    **Model Selection:**
    - Use sidebar dropdown to switch between models
    - Each model has different data and visualization behavior
    
    **Filtering Data:**
    - POF Range: Adjust slider to filter by probability of failure
    - Label Filter: Show only failures, non-failures, or both
    - CC Status: Filter by condition code status (if available)
    - Conductor Length: Filter by physical length (if available)
    
    **Map Interaction:**
    - Pan and zoom to explore data
    - Hover over features for quick info
    - Click features to generate SHAP explanations
    - Use dropdown selector for specific asset analysis
    
    **SHAP Explanations:**
    - Red bars: Features increasing failure probability
    - Blue bars: Features decreasing failure probability
    - Bar length indicates feature importance
    - Top-K setting controls number of features shown
    """)

with st.expander("🔧 Technical Details"):
    st.markdown(f"""
    ### System Information
    
    **Snowflake Configuration:**
    - Database: {DATABASE_NAME}
    - Schema: {SCHEMA_NAME}
    - Stage: {STAGE_NAME}
    - Warehouse: {st.secrets.get('snowflake', {}).get('warehouse', 'N/A')}
    
    **Current Model:**
    - Name: {config['name']}
    - Type: {config['geometry_type']}
    - Visualization: {config['visualization_type']}
    - Records: {len(gdf):,}
    - Features: {len(feature_names)}
    
    **Performance:**
    - Viewport limit: {LIMIT:,} features
    - SHAP sample size: {min(int(len(gdf) * 0.7), 1000):,} records
    - Top-K features: {TOPK}
    
    **Data Columns:**
    - ID: {id_col}
    - Label: {label_col}
    {f"- Structure ID: {structure_id_col}" if structure_id_col else ""}
    - Has CC Status: {("CC Status" in gdf.columns)}
    - Has Conductor Length: {("CONDUCTOR_LENGTH_UDF" in gdf.columns)}
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>🚀 Snowflake Native Application: Streamlit + XGBoost + SHAP + Geopandas</p>
    <p>All computation runs on Snowflake warehouses • Data loaded from Snowflake stages</p>
</div>
""", unsafe_allow_html=True)