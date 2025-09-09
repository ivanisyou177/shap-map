# backend/main.py
"""
FastAPI backend for dynamic viewport loading and on-demand SHAP PNGs.
- Loads XGBoost model saved as JSON (model.json).
- Loads dataset (CSV / Parquet / GeoJSON) with columns:
    * ID column (default "id"; configurable via env ID_COL)
    * geometry column named "geometry" (WKT strings, GeoJSON dicts, or shapely geometries)
    * all model features (names must match model.feature_names_in_ or a provided feature list file)
- Computes POF = model.predict_proba(X)[:,1] on startup for all rows.
- /bbox?minx&miny&maxx&maxy&limit[&topk_by_pof] -> returns GeoJSON of features within viewport (at most `limit`)
- /asset/{asset_id}/shap.png?top_k=20 -> returns generated SHAP waterfall PNG for clicked asset (cached)
- /asset/{asset_id}/info -> returns JSON with id, POF, and geometry type
"""
import os
import io
import json
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, JSONResponse
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

# --- Configuration (override via environment variables) ---
MODEL_PATH = os.environ.get("MODEL_PATH", "xgb_model.json")
DATA_PATH = os.environ.get("DATA_PATH", "equipment_data.csv")
API_MAX_RETURN = int(os.environ.get("API_MAX_RETURN", "5000"))
FEATURE_NAMES_FILE_JSON = os.environ.get("FEATURE_NAMES_JSON", "feature_names.json")
FEATURE_NAMES_FILE_TXT = os.environ.get("FEATURE_NAMES_TXT", "feature_names.txt")

ID_COL = os.environ.get("ID_COL", "id")

# --- Globals ---
model: Optional[XGBClassifier] = None
explainer = None
gdf: Optional[gpd.GeoDataFrame] = None
feature_names = None
pof_min: Optional[float] = None
pof_max: Optional[float] = None

# --- Helpers ---
def _parse_geometry(val):
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

def _get_color_for_pof(pof_value: float) -> str:
    """Maps a POF value to a hex color string using a perceptually uniform colormap."""
    if pof_value is None or pof_min is None or pof_max is None or pof_min >= pof_max:
        return "#808080"  # Default grey for invalid data or uniform POF

    # Normalize POF value to 0-1 range
    norm = colors.Normalize(vmin=pof_min, vmax=pof_max)
    cmap = plt.get_cmap('viridis')

    # Get RGBA color and convert to hex
    rgba_color = cmap(norm(pof_value))
    return colors.to_hex(rgba_color)

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, explainer, gdf, feature_names, pof_min, pof_max
    print(f"[startup] Loading model from {MODEL_PATH}")
    print(f"[startup] Loading data from {DATA_PATH}")
    print(f"[startup] Using ID column: {ID_COL}")

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    # Feature names
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        feature_names = list(model.feature_names_in_)
    else:
        booster = model.get_booster()
        feature_names = list(booster.feature_names) if getattr(booster, "feature_names", None) else None
        if feature_names is None:
            if os.path.exists(FEATURE_NAMES_FILE_JSON):
                with open(FEATURE_NAMES_FILE_JSON, "r") as fh:
                    feature_names = json.load(fh)
            elif os.path.exists(FEATURE_NAMES_FILE_TXT):
                with open(FEATURE_NAMES_FILE_TXT, "r") as fh:
                    feature_names = [line.strip() for line in fh if line.strip()]
    if feature_names is None:
        raise RuntimeError("Could not determine model feature names.")

    # Load dataset
    if DATA_PATH.lower().endswith((".parquet", ".pq")):
        df = pd.read_parquet(DATA_PATH)
    elif DATA_PATH.lower().endswith((".geojson", ".json")):
        gdf_temp = gpd.read_file(DATA_PATH)
        df = pd.DataFrame(gdf_temp.drop(columns=gdf_temp.geometry.name))
        df["geometry"] = gdf_temp.geometry
    else:
        df = pd.read_csv(DATA_PATH)

    if "geometry" not in df.columns:
        raise RuntimeError("Input data must have a 'geometry' column.")
    if ID_COL not in df.columns:
        raise RuntimeError(f"Input data must have an '{ID_COL}' column.")

    # Convert geometries
    sample = df["geometry"].iloc[0]
    if hasattr(sample, "geom_type"):
        geoms = df["geometry"]
    else:
        print("[startup] Converting geometry column to shapely geometries...")
        geoms = df["geometry"].apply(_parse_geometry)

    gdf = gpd.GeoDataFrame(df.copy(), geometry=geoms, crs="EPSG:4326")

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

    # SHAP explainer (modify data to be representative of the model input)
    explainer = shap.TreeExplainer(model, data=gdf[feature_names], model_output="probability")

    print(f"[startup] Server ready! {len(gdf)} rows loaded with geometries.")
    yield
    print("[shutdown] Cleaning up resources.")
    try:
        del model
        del explainer
        del gdf
    except Exception:
        pass

# --- App ---
app = FastAPI(lifespan=lifespan, title="Assets + SHAP API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok", "data_loaded": gdf is not None, "model_loaded": model is not None}

@app.get("/debug/sample")
def debug_sample():
    if gdf is None:
        raise HTTPException(status_code=500, detail="Server not ready")

    features = []
    for _, row in gdf.head(3).iterrows():
        if row.geometry is None:
            continue
        
        pof_val = row.get("POF")
        pof_val_float = float(pof_val) if pd.notna(pof_val) else 0.0

        props = {
            ID_COL: str(row.get(ID_COL)),
            "id": str(row.get(ID_COL)),
            "POF": pof_val_float,
            "color": _get_color_for_pof(pof_val_float)
        }
        features.append({
            "type": "Feature",
            "geometry": mapping(row.geometry),
            "properties": props
        })

    # Explicitly convert numpy types to native python types for robust JSON serialization
    geom_types = {k: int(v) for k, v in gdf.geometry.geom_type.value_counts().items()}
    total_bounds = [float(b) for b in gdf.total_bounds]
    sample_ids = [str(row[ID_COL]) for _, row in gdf.head(5).iterrows()]

    content = {
        "total_count": len(gdf),
        "geometry_types": geom_types,
        "pof_range": [pof_min, pof_max],
        "bounds": total_bounds,
        "sample_ids": sample_ids,
        "sample_features": {
            "type": "FeatureCollection",
            "features": features
        }
    }
    return JSONResponse(content=jsonable_encoder(content))

@app.get("/bbox")
def bbox_geojson(
    minx: float = Query(...),
    miny: float = Query(...),
    maxx: float = Query(...),
    maxy: float = Query(...),
    limit: Optional[int] = Query(None),
    topk_by_pof: Optional[bool] = Query(False),
):
    if gdf is None:
        raise HTTPException(status_code=500, detail="Server not ready (data not loaded).")
    if minx >= maxx or miny >= maxy:
        raise HTTPException(status_code=400, detail="Invalid bbox coordinates.")

    limit = limit or API_MAX_RETURN
    bbox_geom = box(minx, miny, maxx, maxy)
    indices = list(gdf.sindex.intersection((minx, miny, maxx, maxy))) if getattr(gdf, "sindex", None) else None
    subset = gdf.iloc[indices].copy() if indices else gdf.copy()
    subset = subset[subset.geometry.intersects(bbox_geom)]

    if subset.empty:
        return JSONResponse(content={"type": "FeatureCollection", "features": []})

    if len(subset) > limit:
        subset = subset.nlargest(limit, "POF") if topk_by_pof else subset.sample(n=limit, random_state=42)

    features = []
    for _, row in subset.iterrows():
        if row.geometry is None:
            continue
        
        pof_val = row.get("POF")
        pof_val_float = float(pof_val) if pd.notna(pof_val) else 0.0
        
        props = {
            ID_COL: str(row.get(ID_COL)),
            "id": str(row.get(ID_COL)),
            "POF": pof_val_float,
            "color": _get_color_for_pof(pof_val_float)
        }
        features.append({
            "type": "Feature",
            "geometry": mapping(row.geometry),
            "properties": props
        })

    content = {"type": "FeatureCollection", "features": features}
    return JSONResponse(content=jsonable_encoder(content))

@app.get("/colorbar.png")
def colorbar_png():
    if pof_min is None or pof_max is None or pof_min >= pof_max:
        raise HTTPException(status_code=500, detail="POF range not initialized")

    try:
        fig, ax = plt.subplots(figsize=(4, 0.5), dpi=150)
        cmap = plt.get_cmap("viridis")
        norm = colors.Normalize(vmin=pof_min, vmax=pof_max)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        # Remove axis for cleaner colorbar
        ax.remove()
        cbar = fig.colorbar(
            sm,
            orientation="horizontal",
            fraction=0.5,
            pad=0.2
        )
        cbar.set_label("POF", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, transparent=True)
        plt.close(fig)
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Colorbar rendering error: {e}")

# --- LRU-cached SHAP PNG renderer ---
@lru_cache(maxsize=2048)
def _render_shap_png_cached(asset_id_str: str, top_k: int = 20) -> bytes:
    sel = gdf[gdf[ID_COL].astype(str) == str(asset_id_str)]
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
        
        title = f"SHAP waterfall – asset {asset_id_str}"
        if pof_value is not None:
            title += f" – POF: {pof_value:.4f}"
        fig.suptitle(title, fontsize=12)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"SHAP waterfall plot failed, resorting to fallback. Error: {e}")
        
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
            
        colors_bar = ["#2E94E7" if v >= 0 else '#d62728' for v in chosen_vals]
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        y_pos = np.arange(len(chosen_vals))[::-1]
        
        for i, (y, s, e, c) in enumerate(zip(y_pos, starts, ends, colors_bar)):
            ax.barh(y, e - s, left=s, height=0.7, color=c)
            ax.text(e + 0.005*(max(ends)-min(starts)+1), y, f"{chosen_vals[i]:+.3f}", va='center', fontsize=10)
            
        ax.set_yticks(y_pos)
        ax.set_yticklabels(chosen_names, fontsize=10)
        ax.set_xlabel("SHAP contribution (to probability)")
        
        title = f"SHAP waterfall – asset {asset_id_str} (top {k})"
        if pof_value is not None:
            ax.set_title(f"SHAP waterfall – asset {asset_id_str} – POF: {pof_value:.4f}")
        else:
            ax.set_title(f"SHAP waterfall – asset {asset_id_str} (top {k})")
            
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

@app.get("/asset/{asset_id}/info")
def asset_info(asset_id: str):
    sel = gdf[gdf[ID_COL].astype(str) == str(asset_id)]
    if sel.empty:
        raise HTTPException(status_code=404, detail="Asset not found")
    row = sel.iloc[0]

    pof_val = row.get("POF")
    pof_val_float = float(pof_val) if pd.notna(pof_val) else None

    content = {
        "id": str(row.get(ID_COL)),
        "POF": pof_val_float,
        "geometry_type": str(row.geometry.geom_type) if row.geometry else "None"
    }
    return JSONResponse(content=jsonable_encoder(content))