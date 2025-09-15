# backend/main.py
"""
FastAPI backend with multi-model support for dynamic viewport loading and on-demand SHAP PNGs.
- Supports multiple model configurations with different data sources and visualization types
- Equipment model: One geometry per equipment (LineString)
- Structure model: Multiple equipment per structure (Point/Polygon geometries)
- Dynamic model switching via API endpoints
- File download functionality for backend files
"""
import os
import io
import json
import glob
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

import pandas as pd
import geopandas as gpd
import shap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
from shapely.geometry import box, shape as geom_shape, mapping
from shapely import wkt
from xgboost import XGBClassifier

plt.switch_backend("Agg")  # headless rendering for PNGs

# --- Multi-Model Configuration ---
MODEL_CONFIGS = {
    "equipment": {
        "name": "Equipment Model",
        "description": "Individual equipment analysis with LineString geometries",
        "model_path": "xgb_model.json",
        "data_path": "equipment_data.csv",
        "feature_names_json": "feature_names.json",
        "feature_names_txt": "feature_names.txt",
        "id_col": "id",
        "label_col": "label",
        "geometry_type": "equipment",  # One geometry per ID
        "structure_id_col": None,  # Not applicable for equipment model
        "visualization_type": "single_click"  # Click shows single SHAP plot
    },
    "structure": {
        "name": "Structure Model", 
        "description": "Structure analysis with multiple equipment per structure",
        "model_path": "structure_xgb_model.json",
        "data_path": "structure_data.csv", 
        "feature_names_json": "structure_feature_names.json",
        "feature_names_txt": "structure_feature_names.txt",
        "id_col": "equipment_id",  # Equipment ID (unique)
        "label_col": "label",
        "geometry_type": "structure",  # Multiple equipment per structure
        "structure_id_col": "structure_id",  # Groups equipment by structure
        "geometry_col": "geometry",  # Geometry represents structure, not equipment
        "visualization_type": "multi_click"  # Click shows multiple SHAP plots
    }
}

# Global configuration
API_MAX_RETURN = int(os.environ.get("API_MAX_RETURN", "5000"))
CURRENT_MODEL_KEY = os.environ.get("DEFAULT_MODEL", "equipment")

# --- Globals ---
current_config: Dict[str, Any] = {}
model: Optional[XGBClassifier] = None
explainer = None
gdf: Optional[gpd.GeoDataFrame] = None
feature_names = None
pof_min: Optional[float] = None
pof_max: Optional[float] = None

# --- Helpers ---
def _parse_geometry(val):
    '''Parses a geometry from WKT string, GeoJSON dict, or returns if already a shapely geometry.'''
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

def _get_color_for_pof(pof_value: float, pof_range_min: Optional[float] = None, pof_range_max: Optional[float] = None) -> str:
    """Maps a POF value to a hex color string using a perceptually uniform colormap."""
    # Use provided range or global range
    min_val = pof_range_min if pof_range_min is not None else pof_min
    max_val = pof_range_max if pof_range_max is not None else pof_max
    
    if pof_value is None or min_val is None or max_val is None or min_val >= max_val:
        return "#808080"  # Default grey for invalid data or uniform POF

    # Normalize POF value to 0-1 range
    norm = colors.Normalize(vmin=min_val, vmax=max_val)
    cmap = plt.get_cmap('RdYlBu_r')  # Reversed for blue=low, red=high

    # Get RGBA color and convert to hex
    rgba_color = cmap(norm(pof_value))
    return colors.to_hex(rgba_color)

def load_model_and_data(model_key: str):
    """Load model and data for the specified configuration"""
    global current_config, model, explainer, gdf, feature_names, pof_min, pof_max
    
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}")
    
    current_config = MODEL_CONFIGS[model_key].copy()
    print(f"[load_model] Loading model configuration: {current_config['name']}")
    
    # Load model
    model_path = current_config["model_path"]
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at {model_path}")
    
    model = XGBClassifier()
    model.load_model(model_path)
    print(f"[load_model] Model loaded from {model_path}")

    # Feature names
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        feature_names = list(model.feature_names_in_)
    else:
        booster = model.get_booster()
        feature_names = list(booster.feature_names) if getattr(booster, "feature_names", None) else None
        if feature_names is None:
            feature_json = current_config["feature_names_json"]
            feature_txt = current_config["feature_names_txt"]
            if os.path.exists(feature_json):
                with open(feature_json, "r") as fh:
                    feature_names = json.load(fh)
            elif os.path.exists(feature_txt):
                with open(feature_txt, "r") as fh:
                    feature_names = [line.strip() for line in fh if line.strip()]
    if feature_names is None:
        raise RuntimeError("Could not determine model feature names.")

    # Load dataset
    data_path = current_config["data_path"]
    if not os.path.exists(data_path):
        raise RuntimeError(f"Data file not found at {data_path}")
        
    print(f"[load_model] Loading data from {data_path}")
    
    if data_path.lower().endswith((".parquet", ".pq")):
        df = pd.read_parquet(data_path)
    elif data_path.lower().endswith((".geojson", ".json")):
        gdf_temp = gpd.read_file(data_path)
        df = pd.DataFrame(gdf_temp.drop(columns=gdf_temp.geometry.name))
        df["geometry"] = gdf_temp.geometry
    else:
        df = pd.read_csv(data_path)

    # Handle geometry column based on model type
    geometry_col = current_config.get("geometry_col", "geometry")
    if geometry_col not in df.columns:
        raise RuntimeError(f"Input data must have a '{geometry_col}' column.")
    
    id_col = current_config["id_col"]
    if id_col not in df.columns:
        raise RuntimeError(f"Input data must have an '{id_col}' column.")

    # Convert geometries
    sample = df[geometry_col].iloc[0]
    if hasattr(sample, "geom_type"):
        geoms = df[geometry_col]
    else:
        print(f"[load_model] Converting {geometry_col} column to shapely geometries...")
        geoms = df[geometry_col].apply(_parse_geometry)

    # Create GeoDataFrame, assuming WGS84
    df_copy = df.copy()
    df_copy["geometry"] = geoms
    gdf = gpd.GeoDataFrame(df_copy, geometry="geometry", crs="EPSG:4326")

    # Ensure features exist
    missing = [c for c in feature_names if c not in gdf.columns]
    if missing:
        raise RuntimeError(f"Dataset missing features required by the model: {missing[:10]}")

    # Compute POF
    X_all = gdf[feature_names].astype(float)
    gdf["POF"] = model.predict_proba(X_all)[:, 1].astype(float)

    # Store POF range for color mapping
    if not gdf["POF"].empty:
        pof_min = float(gdf["POF"].min())
        pof_max = float(gdf["POF"].max())

    # Spatial index
    _ = gdf.sindex

    # SHAP explainer
    explainer = shap.TreeExplainer(model, data=gdf[feature_names].iloc[:int(len(gdf) * 0.7)], model_output="probability")

    print(f"[load_model] Model ready! {len(gdf)} rows loaded with geometries.")
    print(f"[load_model] POF range: {pof_min:.4f} - {pof_max:.4f}")
    
    label_col = current_config["label_col"]
    if label_col in gdf.columns:
        label_counts = gdf[label_col].value_counts().to_dict()
        print(f"[load_model] Label distribution: {label_counts}")
    else:
        print(f"[load_model] Warning: Label column '{label_col}' not found in data")

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    '''Load initial model and data on startup, cleanup on shutdown'''
    
    print(f"[startup] Available model configurations: {list(MODEL_CONFIGS.keys())}")
    print(f"[startup] Loading default model: {CURRENT_MODEL_KEY}")
    
    try:
        load_model_and_data(CURRENT_MODEL_KEY)
    except Exception as e:
        print(f"[startup] Failed to load default model: {e}")
        print("[startup] Server will start but model switching will be required")
    
    yield
    print("[shutdown] Cleaning up resources.")
    try:
        del model
        del explainer
        del gdf
    except Exception:
        pass

# --- App ---
app = FastAPI(lifespan=lifespan, title="Multi-Model Assets + SHAP API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Model Management Endpoints ---
@app.get("/models")
def list_models():
    """List all available model configurations"""
    return {
        "available_models": {
            key: {
                "name": config["name"],
                "description": config["description"],
                "visualization_type": config["visualization_type"]
            }
            for key, config in MODEL_CONFIGS.items()
        },
        "current_model": CURRENT_MODEL_KEY if gdf is not None else None
    }

@app.post("/models/{model_key}/load")
def load_model(model_key: str):
    """Switch to a different model configuration"""
    global CURRENT_MODEL_KEY
    
    if model_key not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
    
    try:
        load_model_and_data(model_key)
        CURRENT_MODEL_KEY = model_key
        return {
            "status": "success",
            "message": f"Successfully loaded model: {MODEL_CONFIGS[model_key]['name']}",
            "current_model": model_key,
            "features_loaded": len(gdf) if gdf is not None else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/files")
def list_files():
    """List all files in the backend directory"""
    try:
        files = []
        for file_path in glob.glob("*"):
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                files.append({
                    "name": file_path,
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

@app.get("/files/{filename}")
def download_file(filename: str):
    """Download a file from the backend directory"""
    # Security check - only allow files in current directory, no path traversal
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(filename, media_type='application/octet-stream', filename=filename)

# --- Data Endpoints ---
@app.get("/health")
def health():
    return {
        "status": "ok", 
        "data_loaded": gdf is not None, 
        "model_loaded": model is not None,
        "current_model": CURRENT_MODEL_KEY if gdf is not None else None,
        "model_config": current_config.get("name", "None") if current_config else "None"
    }

@app.get("/info")
def get_info():
    """Returns basic info about the current dataset including model configuration"""
    if gdf is None:
        raise HTTPException(status_code=500, detail="No model loaded")
    
    id_col = current_config.get("id_col", "id")
    label_col = current_config.get("label_col", "label")

    # Safely get bounds
    try:
        bounds = [float(b) for b in gdf.total_bounds]
        if len(bounds) != 4 or any(pd.isna(bounds)):
            bounds = None
    except Exception:
        bounds = None

    # Safely get geometry types
    try:
        geometry_types = {str(k): int(v) for k, v in gdf.geometry.geom_type.value_counts().items()}
    except Exception:
        geometry_types = {}

    info = {
        "current_model": CURRENT_MODEL_KEY,
        "model_name": current_config.get("name", ""),
        "visualization_type": current_config.get("visualization_type", ""),
        "total_count": int(len(gdf)),
        "pof_range": [float(pof_min), float(pof_max)] if pof_min is not None and pof_max is not None else None,
        "bounds": bounds,
        "geometry_types": geometry_types,
        "id_column": id_col,
        "label_column": label_col,
        "has_label_column": label_col in gdf.columns
    }

    # Add structure-specific info
    if current_config.get("geometry_type") == "structure":
        structure_id_col = current_config.get("structure_id_col")
        info["structure_id_column"] = structure_id_col
        if structure_id_col and structure_id_col in gdf.columns:
            try:
                structure_counts = gdf.groupby(structure_id_col).size()
                info["structures_count"] = int(len(structure_counts))
                info["equipment_per_structure"] = {
                    "min": int(structure_counts.min()),
                    "max": int(structure_counts.max()),
                    "mean": float(structure_counts.mean())
                }
            except Exception:
                info["structures_count"] = None
                info["equipment_per_structure"] = None

    # Add label distribution
    if label_col in gdf.columns:
        try:
            info["label_distribution"] = {str(k): int(v) for k, v in gdf[label_col].value_counts().items()}
        except Exception:
            info["label_distribution"] = None

    return JSONResponse(content=jsonable_encoder(info))

@app.get("/bbox")
def bbox_geojson(
    minx: float = Query(...),
    miny: float = Query(...),
    maxx: float = Query(...),
    maxy: float = Query(...),
    limit: Optional[int] = Query(None),
    topk_by_pof: Optional[bool] = Query(False),
    pof_min_filter: Optional[float] = Query(None),
    pof_max_filter: Optional[float] = Query(None),
    label_filter: Optional[str] = Query(None)
):
    if gdf is None:
        raise HTTPException(status_code=500, detail="No model loaded")
    if minx >= maxx or miny >= maxy:
        raise HTTPException(status_code=400, detail="Invalid bbox coordinates.")

    limit = limit or API_MAX_RETURN
    bbox_geom = box(minx, miny, maxx, maxy)
    indices = list(gdf.sindex.intersection((minx, miny, maxx, maxy))) if getattr(gdf, "sindex", None) else None
    subset = gdf.iloc[indices].copy() if indices else gdf.copy()
    subset = subset[subset.geometry.intersects(bbox_geom)]

    if subset.empty:
        return JSONResponse(content={"type": "FeatureCollection", "features": []})

    # Apply POF range filter
    if pof_min_filter is not None:
        subset = subset[subset["POF"] >= pof_min_filter]
    if pof_max_filter is not None:
        subset = subset[subset["POF"] <= pof_max_filter]

    # Apply label filter
    label_col = current_config["label_col"]
    if label_filter and label_col in subset.columns:
        if label_filter == "failures":
            subset = subset[subset[label_col] == 1]
        elif label_filter == "non_failures":
            subset = subset[subset[label_col] == 0]

    if subset.empty:
        return JSONResponse(content={"type": "FeatureCollection", "features": []})

    # For structure model, we need to aggregate by structure for visualization
    if current_config["geometry_type"] == "structure":
        structure_id_col = current_config["structure_id_col"]
        if structure_id_col in subset.columns:
            # Group by structure and aggregate POF (e.g., max or mean)
            structure_groups = subset.groupby(structure_id_col).agg({
                "POF": "max",  # Use max POF for structure color
                "geometry": "first",  # Structures share the same geometry
                label_col: lambda x: (x == 1).any() if label_col in subset.columns else 0  # Any failure in structure
            }).reset_index()
            
            subset = structure_groups
            subset[current_config["id_col"]] = subset[structure_id_col]  # Use structure ID as display ID

    if len(subset) > limit:
        subset = subset.nlargest(limit, "POF") if topk_by_pof else subset.sample(n=limit, random_state=42)

    # Use filtered POF range for color mapping if provided
    color_min = pof_min_filter if pof_min_filter is not None else pof_min
    color_max = pof_max_filter if pof_max_filter is not None else pof_max

    features = []
    id_col = current_config["id_col"]
    
    for _, row in subset.iterrows():
        if row.geometry is None:
            continue
        
        pof_val = row.get("POF")
        pof_val_float = float(pof_val) if pd.notna(pof_val) else 0.0
        
        props = {
            id_col: str(row.get(id_col)),
            "id": str(row.get(id_col)),
            "POF": pof_val_float,
            "color": _get_color_for_pof(pof_val_float, color_min, color_max),
            "model_type": current_config["geometry_type"],
            "visualization_type": current_config["visualization_type"]
        }
        
        # Add structure-specific properties
        if current_config["geometry_type"] == "structure":
            structure_id_col = current_config["structure_id_col"]
            props["structure_id"] = str(row.get(structure_id_col, row.get(id_col)))
        
        # Add label if available
        if label_col in gdf.columns:
            props[label_col] = int(row.get(label_col, 0))
        
        features.append({
            "type": "Feature",
            "geometry": mapping(row.geometry),
            "properties": props
        })

    content = {"type": "FeatureCollection", "features": features}
    return JSONResponse(content=jsonable_encoder(content))

# --- SHAP Endpoints ---
@lru_cache(maxsize=2048)
def _render_shap_png_cached(asset_id_str: str, top_k: int = 20) -> bytes:
    if gdf is None:
        raise RuntimeError("No model loaded")
    
    id_col = current_config["id_col"]
    sel = gdf[gdf[id_col].astype(str) == str(asset_id_str)]
    if sel.empty:
        raise KeyError(f"Asset {asset_id_str} not found")
    
    try:
        pof_value = float(sel.iloc[0]["POF"])
    except (KeyError, ValueError, TypeError):
        pof_value = None

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
        
        title = f"SHAP waterfall - {id_col} {asset_id_str}"
        if pof_value is not None:
            title += f" - POF: {pof_value:.4f}"
        fig.suptitle(title, fontsize=12)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"SHAP waterfall plot failed, using fallback. Error: {e}")
        
        # Fallback implementation (same as before)
        expl_item = single_explanation
        vals = np.array(expl_item.values).reshape(-1)
        base_val = float(expl_item.base_values)
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
        ax.set_title(f"SHAP waterfall - {id_col} {asset_id_str} - POF: {pof_value:.4f}" if pof_value is not None 
                    else f"SHAP waterfall - {id_col} {asset_id_str}")
            
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

@app.get("/asset/{asset_id}/shap.png")
def asset_shap_png(asset_id: str, top_k: int = Query(20, ge=1, le=200)):
    try:
        png = _render_shap_png_cached(str(asset_id), int(top_k))
    except KeyError:
        raise HTTPException(status_code=404, detail="Asset not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP rendering error: {e}")
    return Response(content=png, media_type="image/png")

@app.get("/structure/{structure_id}/equipment")
def get_structure_equipment(structure_id: str):
    """Get all equipment IDs for a given structure (for multi-SHAP visualization)"""
    if gdf is None:
        raise HTTPException(status_code=500, detail="No model loaded")
    
    if current_config["geometry_type"] != "structure":
        raise HTTPException(status_code=400, detail="Multi-equipment view only available for structure model")
    
    structure_id_col = current_config["structure_id_col"]
    id_col = current_config["id_col"]
    
    equipment = gdf[gdf[structure_id_col].astype(str) == str(structure_id)]
    if equipment.empty:
        raise HTTPException(status_code=404, detail="Structure not found")
    
    equipment_list = []
    for _, row in equipment.iterrows():
        equipment_list.append({
            "equipment_id": str(row[id_col]),
            "POF": float(row["POF"]) if pd.notna(row["POF"]) else None,
            "label": int(row.get(current_config["label_col"], 0)) if current_config["label_col"] in gdf.columns else None
        })
    
    return {
        "structure_id": structure_id,
        "equipment": equipment_list,
        "total_equipment": len(equipment_list)
    }

@app.get("/colorbar.png")
def colorbar_png(
    pof_min_range: Optional[float] = Query(None),
    pof_max_range: Optional[float] = Query(None),
    width: int = Query(300, ge=100, le=800),
    height: int = Query(40, ge=20, le=200)
):
    """Generate a horizontal colorbar PNG for POF values"""
    min_val = pof_min_range if pof_min_range is not None else pof_min
    max_val = pof_max_range if pof_max_range is not None else pof_max
    
    if min_val is None or max_val is None:
        raise HTTPException(status_code=500, detail=f"POF range not available: min={min_val}, max={max_val}")
    
    if min_val >= max_val:
        raise HTTPException(status_code=500, detail=f"Invalid POF range: min ({min_val}) >= max ({max_val})")

    try:
        fig_width = max(4, width / 75)
        fig_height = max(0.6, height / 75)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
        
        cmap = plt.get_cmap("RdYlBu_r")
        norm = colors.Normalize(vmin=min_val, vmax=max_val)
        
        gradient = np.linspace(min_val, max_val, 256).reshape(1, -1)
        im = ax.imshow(gradient, aspect='auto', cmap=cmap, norm=norm, extent=[min_val, max_val, 0, 1])
        
        ax.set_xlim(min_val, max_val)
        ax.set_yticks([])
        ax.set_xlabel("POF (Probability of Failure)", fontsize=10, fontweight='bold')
        
        if max_val - min_val < 0.01:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.4f}'))
        elif max_val - min_val < 0.1:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
        else:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', left=False, right=False)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(False)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, 
                   transparent=False, facecolor='white', edgecolor='none', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        
        return Response(content=buf.getvalue(), media_type="image/png")
        
    except Exception as e:
        plt.close('all')
        raise HTTPException(status_code=500, detail=f"Colorbar rendering error: {e}")

@app.get("/asset/{asset_id}/info")
def asset_info(asset_id: str):
    if gdf is None:
        raise HTTPException(status_code=500, detail="No model loaded")
        
    id_col = current_config["id_col"]
    sel = gdf[gdf[id_col].astype(str) == str(asset_id)]
    if sel.empty:
        raise HTTPException(status_code=404, detail="Asset not found")
    row = sel.iloc[0]

    pof_val = row.get("POF")
    pof_val_float = float(pof_val) if pd.notna(pof_val) else None

    content = {
        "id": str(row.get(id_col)),
        "POF": pof_val_float,
        "geometry_type": str(row.geometry.geom_type) if row.geometry else "None",
        "model_type": current_config["geometry_type"],
        "visualization_type": current_config["visualization_type"]
    }
    
    # Add label info if available
    label_col = current_config["label_col"]
    if label_col in gdf.columns:
        content[label_col] = int(row.get(label_col, 0))
    
    # Add structure info if applicable
    if current_config["geometry_type"] == "structure":
        structure_id_col = current_config["structure_id_col"]
        content["structure_id"] = str(row.get(structure_id_col))
    
    return JSONResponse(content=jsonable_encoder(content))