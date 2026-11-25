"""
AI 3D Reconstruction Studio & Virtual Room Stager — Milestone 3 Application.
Course: DMET 901 Computer Vision, German University in Cairo (GUC)
Team 9: Mohamed Ayman Awad, Noureldin Abdelrahman, Noureldeen Amr, Abdelrahman Ashry, Ahmed Hamdy Mostafa

Organizes our 3 reconstruction paradigms cleanly into two operational modes:
  1. Generalization (Unseen / Any Object): Pix2Vox++ (Volumetric DL) & AtlasNet (Spherical Surface Mesh DL)
  2. Pre-Computed / Already-Seen Scene (Classical Math): COLMAP (Known Poses & Blind SfM)
"""

import base64
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import streamlit as st
import torch

# Ensure project root & viewer helpers are importable
APP_DIR = Path(__file__).resolve().parent
TRACK_ROOT = APP_DIR.parent
REPO_ROOT = TRACK_ROOT.parent
sys.path.insert(0, str(TRACK_ROOT))
sys.path.insert(0, str(REPO_ROOT / "viewer"))

from src.data.voxelizer import normalize_point_cloud, points_to_voxels, voxels_to_mesh  # noqa: E402
from src.data.splits import HOME_DECOR_CATEGORIES  # noqa: E402
from src.models.loader import load_pix2vox_checkpoint, load_atlasnet_checkpoint  # noqa: E402

try:
    from export import gt_frame, score as viewer_score  # noqa: E402
except Exception:
    gt_frame = None
    viewer_score = None

st.set_page_config(
    page_title="Omni3D Multi-Paradigm 3D Reconstruction Studio",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------- Path & Checkpoint Discovery -----------------

def get_dataset_root() -> str:
    for candidate in [
        REPO_ROOT / "dataset",
        TRACK_ROOT / "dataset",
        Path("dataset"),
    ]:
        if (candidate / "renders").is_dir():
            return str(candidate.resolve())
    return str((REPO_ROOT / "dataset").resolve())


DATASET_ROOT = get_dataset_root()


def find_best_checkpoint(patterns: List[str]) -> Optional[str]:
    ckpt_dir = TRACK_ROOT / "outputs" / "checkpoints"
    for pat in patterns:
        p = ckpt_dir / pat
        if p.is_file():
            return str(p)
    return None


@st.cache_resource
def load_generalized_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pix2vox_ckpt = find_best_checkpoint([
        "pix2vox++_ms3_omni215_aug_best.pth",
        "pix2vox++_3views_all_best.pth",
        "pix2vox++_3views_home_decor_best.pth",
        "pix2vox_3views_home_decor_best.pth",
    ])
    pix2vox = load_pix2vox_checkpoint(pix2vox_ckpt, device=device, use_refiner=True)

    atlas_ckpt = find_best_checkpoint([
        "atlasnet_ms3_omni215_aug_best.pth",
        "atlasnet_3views_all_best.pth",
        "atlasnet_3views_home_decor_best.pth",
    ])
    atlasnet = load_atlasnet_checkpoint(atlas_ckpt, device=device)

    return pix2vox, pix2vox_ckpt, atlasnet, atlas_ckpt, device


@st.cache_data
def get_available_objects() -> Dict[str, List[str]]:
    renders_dir = Path(DATASET_ROOT) / "renders"
    if not renders_dir.is_dir():
        return {}
    obj_dict: Dict[str, List[str]] = {}
    for cat_dir in sorted(renders_dir.iterdir()):
        if cat_dir.is_dir():
            objs = sorted(d.name for d in cat_dir.iterdir() if d.is_dir())
            if objs:
                obj_dict[cat_dir.name] = objs
    return obj_dict


@st.cache_data
def load_splits_info() -> Optional[dict]:
    splits_path = Path(DATASET_ROOT) / "splits.json"
    if splits_path.is_file():
        return json.loads(splits_path.read_text())
    return None


@st.cache_data
def load_colmap_run(run_id: str) -> Optional[dict]:
    run_file = REPO_ROOT / "viewer" / "runs" / run_id / "run.json"
    if run_file.is_file():
        return json.loads(run_file.read_text())
    return None


def decode_viewer_points(run_id: str, run_meta: dict, obj_id: str) -> Optional[np.ndarray]:
    ent = run_meta.get("objects", {}).get(obj_id)
    if not ent or "offset" not in ent:
        return None
    cat = obj_id.split("/")[0]
    pts_file = REPO_ROOT / "viewer" / "runs" / run_id / "pts" / f"{cat}.json"
    if not pts_file.is_file():
        return None
    b64 = json.loads(pts_file.read_text())["b64"]
    raw = np.frombuffer(base64.b64decode(b64), dtype="<i2").reshape(-1, 3)
    off, length = ent["offset"]
    if length == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return raw[off : off + length].astype(np.float32) * float(run_meta["quant"])


# ----------------- Image & Metric Helpers -----------------

def preprocess_pil_images(imgs: List[Image.Image], device: torch.device) -> torch.Tensor:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensors = []
    for im in imgs:
        im_rgba = im.convert("RGBA").resize((224, 224), Image.Resampling.BILINEAR)
        bg = Image.new("RGBA", (224, 224), (255, 255, 255, 255))
        comp = Image.alpha_composite(bg, im_rgba).convert("RGB")
        arr = (np.array(comp, dtype=np.float32) / 255.0 - mean) / std
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float())
    return torch.stack(tensors, dim=0).unsqueeze(0).to(device)


def map_norm_to_viewer_frame(pts_norm: np.ndarray, gt_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Maps [-0.5, 0.5]^3 normalized predictions into GT bounding-sphere frame for exact viewer_score."""
    _, centroid, max_dist = normalize_point_cloud(gt_raw)
    c, r = gt_frame(gt_raw)
    raw_pred = (pts_norm.astype(np.float64) / 0.95) * (2.0 * max_dist) + centroid
    pred_shared = (raw_pred - c) / r
    gt_shared = (gt_raw.astype(np.float64) - c) / r
    return pred_shared, gt_shared


def plot_point_cloud_3d(
    points: np.ndarray,
    title: str = "3D Point Cloud",
    color: str = "#2563eb",
    cameras: Optional[np.ndarray] = None,
    cameras_used: Optional[List[bool]] = None,
    gt_overlay: Optional[np.ndarray] = None,
):
    traces = []
    if gt_overlay is not None and len(gt_overlay) > 0:
        traces.append(
            go.Scatter3d(
                x=gt_overlay[:, 0], y=gt_overlay[:, 2], z=gt_overlay[:, 1],
                mode="markers",
                name="Ground Truth",
                marker=dict(size=2.0, color="#94a3b8", opacity=0.35),
            )
        )
    if len(points) > 0:
        traces.append(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 2], z=points[:, 1],
                mode="markers",
                name="Prediction",
                marker=dict(size=2.6, color=color, opacity=0.90),
            )
        )
    if cameras is not None and len(cameras) > 0:
        cam_colors = ["#ef4444" if (cameras_used and u) else "#64748b" for u in (cameras_used or [True] * len(cameras))]
        traces.append(
            go.Scatter3d(
                x=cameras[:, 0], y=cameras[:, 2], z=cameras[:, 1],
                mode="markers",
                name="Camera Rig",
                marker=dict(size=4.5, color=cam_colors, symbol="diamond", opacity=0.85),
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=0, r=0, b=0, t=35),
        scene=dict(
            xaxis=dict(range=[-1.2, 1.2], showbackground=False),
            yaxis=dict(range=[-1.2, 1.2], showbackground=False),
            zaxis=dict(range=[-1.2, 1.2], showbackground=False),
            aspectmode="cube",
        ),
        height=460,
        showlegend=(gt_overlay is not None or cameras is not None),
    )
    return fig


def plot_mesh_3d(verts: np.ndarray, faces: np.ndarray, title: str = "3D Surface Mesh", color: str = "#7c3aed"):
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0], y=verts[:, 2], z=verts[:, 1],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color,
            opacity=0.95,
            flatshading=True,
            lighting=dict(ambient=0.45, diffuse=0.65, roughness=0.45, specular=0.25),
        )
    ])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=0, r=0, b=0, t=35),
        scene=dict(
            xaxis=dict(range=[-0.6, 0.6], showbackground=False),
            yaxis=dict(range=[-0.6, 0.6], showbackground=False),
            zaxis=dict(range=[-0.6, 0.6], showbackground=False),
            aspectmode="cube",
        ),
        height=460,
    )
    return fig


def create_obj_string(verts: np.ndarray, faces: np.ndarray) -> str:
    lines = ["# Omni3D Exported Surface Mesh\n"]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
    return "".join(lines)


# ----------------- Load Models & Data -----------------

pix2vox_model, pix2vox_ckpt, atlasnet_model, atlas_ckpt, device = load_generalized_models()
obj_dict = get_available_objects()
splits_info = load_splits_info()
colmap_known = load_colmap_run("bones__colmap__sparse_known")
colmap_sfm = load_colmap_run("bones__colmap__sparse_sfm")

# ----------------- Sidebar -----------------

st.sidebar.title("🧊 Omni3D Studio — MS3")
st.sidebar.markdown(
    """
    **DMET 901 — Computer Vision (GUC)**  
    **Team 9 (`home-omni3d`)**  
    ---
    **Best Model Per Category:**
    - 📐 **Mathematics**: **COLMAP** *(24-view Triangulation / SfM)*
    - 🧊 **Volumetric DL**: **Pix2Vox++** *(ResNet + Fusion + 3D U-Net)*
    - 🕸️ **Surface Mesh DL**: **AtlasNet** *(Spherical Patches + Attention)*
    """
)
st.sidebar.divider()
st.sidebar.caption(f"Device: `{device}`")
st.sidebar.caption(f"Pix2Vox++ ckpt: `{Path(pix2vox_ckpt).name if pix2vox_ckpt else 'untrained'}`")
st.sidebar.caption(f"AtlasNet ckpt: `{Path(atlas_ckpt).name if atlas_ckpt else 'untrained'}`")
st.sidebar.caption(f"Categories on disk: `{len(obj_dict)}`")


# ----------------- Main Navigation Tabs -----------------

tab_general, tab_precomputed, tab_arena, tab_room, tab_dataset = st.tabs([
    "🌍 Generalization Mode (Unseen / Any Object)",
    "📐 Pre-Computed / Known Scene (COLMAP Math)",
    "⚔️ 3-Category Benchmark Arena",
    "🏡 Virtual Room Stager",
    "📊 215-Category Dataset & Augmentation",
])


# ==============================================================================
# TAB 1: GENERALIZATION MODE (Unseen Objects & User Uploads)
# ==============================================================================
with tab_general:
    st.subheader("🌍 Generalization Mode — Reconstruct Any Unseen Object from 1–5 Photos")
    st.info(
        "**Use this tab when you want a model that generalizes to unseen objects or brand-new photos without calibrated camera poses.** "
        "**Pix2Vox++** (our best generalizing volumetric model, trained on 215 categories with image augmentation) and "
        "**AtlasNet** (spherical surface patch deformation) infer 3D structure directly from 1, 3, or 5 RGB viewpoints."
    )

    col_in, col_out = st.columns([1, 2])
    with col_in:
        input_mode = st.radio(
            "Input Source",
            ["OmniObject3D Dataset (Unseen / All Categories)", "Upload Your Own Image(s)"],
            horizontal=False,
        )
        model_choice = st.selectbox(
            "Select Generalizing Model",
            [
                "Pix2Vox++ (Volumetric DL — Best Generalization)",
                "AtlasNet (Spherical Surface Mesh DL)",
            ],
        )
        voxel_thresh = 0.40
        if "Pix2Vox++" in model_choice:
            voxel_thresh = st.slider("Voxel Occupancy Threshold", 0.20, 0.70, 0.40, 0.02)

        loaded_imgs: List[Image.Image] = []
        gt_raw_pc: Optional[np.ndarray] = None
        selected_label = "Uploaded Object"

        if input_mode.startswith("OmniObject3D"):
            scope_filter = st.selectbox(
                "Category / Split Filter",
                [
                    "Held-Out Unseen Test Split Only (Best Fair Check)",
                    "New 139 Extended Categories (Food / Toys / Everyday)",
                    "Home Decor Subset (18 Furniture Classes)",
                    "All 215 Categories",
                ],
            )
            test_lookup: Dict[str, List[str]] = {}
            if splits_info and scope_filter.startswith("Held-Out"):
                for s in splits_info["test"]:
                    test_lookup.setdefault(s["category"], []).append(s["object_id"])
                filtered_dict = {k: sorted(v) for k, v in sorted(test_lookup.items()) if k in obj_dict}
            elif scope_filter.startswith("New 139"):
                legacy_cats = set(colmap_known.get("categories", {}).keys()) if colmap_known else set(HOME_DECOR_CATEGORIES)
                filtered_dict = {k: v for k, v in obj_dict.items() if k not in legacy_cats}
            elif scope_filter.startswith("Home Decor"):
                filtered_dict = {k: obj_dict[k] for k in HOME_DECOR_CATEGORIES if k in obj_dict}
            else:
                filtered_dict = obj_dict

            cat_list = sorted(filtered_dict.keys())
            default_idx = cat_list.index("chair") if "chair" in cat_list else 0
            selected_cat = st.selectbox("Category", cat_list, index=default_idx)
            selected_obj = st.selectbox("Object Instance", filtered_dict.get(selected_cat, []))
            num_views_choice = st.radio("Input Viewpoints (K)", [1, 3, 5], index=1, horizontal=True)

            obj_dir = Path(DATASET_ROOT) / "renders" / selected_cat / selected_obj
            step = 24 // num_views_choice
            view_indices = [(i * step) % 24 for i in range(num_views_choice)]
            selected_label = f"{selected_cat}/{selected_obj}"

            img_cols = st.columns(num_views_choice)
            for idx, v_idx in enumerate(view_indices):
                vp = obj_dir / f"{v_idx:03d}.png"
                if vp.is_file():
                    im = Image.open(vp)
                    loaded_imgs.append(im)
                    with img_cols[idx]:
                        st.image(im, caption=f"View {v_idx:02d}", use_container_width=True)

            gt_path = Path(DATASET_ROOT) / "point_clouds" / selected_cat / f"{selected_obj}.npy"
            if gt_path.is_file():
                gt_raw_pc = np.load(gt_path)
        else:
            uploaded_files = st.file_uploader(
                "Upload 1 to 5 images of an object (PNG/JPG)",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                loaded_imgs = [Image.open(f) for f in uploaded_files[:5]]
                ucols = st.columns(len(loaded_imgs))
                for idx, im in enumerate(loaded_imgs):
                    with ucols[idx]:
                        st.image(im, caption=f"Upload {idx+1}", use_container_width=True)

    with col_out:
        if loaded_imgs:
            st.markdown(f"### Prediction for **{selected_label}**")
            batch_img = preprocess_pil_images(loaded_imgs, device)
            t0 = time.time()

            if "Pix2Vox++" in model_choice:
                with torch.no_grad():
                    out = pix2vox_model(batch_img)
                    vox = out["voxels"][0].cpu().numpy()
                elapsed_ms = (time.time() - t0) * 1000.0
                mesh_res = voxels_to_mesh(vox, threshold=voxel_thresh)

                # Compute live F@5% if GT is available
                metrics_str = ""
                if gt_raw_pc is not None and gt_frame and viewer_score:
                    idx = np.argwhere(vox >= voxel_thresh).astype(np.float64)
                    if len(idx) > 0:
                        pred_norm = (idx / 31.0) - 0.5
                        pred_s, gt_s = map_norm_to_viewer_frame(pred_norm, gt_raw_pc)
                        sc = viewer_score(pred_s, gt_s)
                        metrics_str = f" | Viewer **F@5%: {sc['f_5']:.3f}** | **CD-L1: {sc['chamfer_l1']:.3f}**"

                st.caption(f"⚡ Inferred in **{elapsed_ms:.1f} ms** on `{device}` | Views: **{len(loaded_imgs)}**{metrics_str}")
                t_pred, t_gt = st.tabs(["Pix2Vox++ 3D Mesh", "Ground Truth Point Cloud"])
                with t_pred:
                    if mesh_res is not None:
                        verts, faces = mesh_res
                        st.plotly_chart(plot_mesh_3d(verts, faces, title=f"Pix2Vox++ Mesh (th={voxel_thresh:.2f})", color="#7c3aed"), use_container_width=True)
                        st.download_button("💾 Download Mesh (.obj)", data=create_obj_string(verts, faces), file_name=f"{selected_label.replace('/', '_')}_pix2voxpp.obj")
                    else:
                        st.warning("Threshold too high — lower the slider to visualize occupied voxels.")
                with t_gt:
                    if gt_raw_pc is not None:
                        gt_norm, _, _ = normalize_point_cloud(gt_raw_pc)
                        st.plotly_chart(plot_point_cloud_3d(gt_norm, title="Laser Scanner GT (4,096 pts)", color="#10b981"), use_container_width=True)
                    else:
                        st.info("No Ground Truth 3D scan for custom uploaded photos — showing pure generalization!")
            else:
                with torch.no_grad():
                    out_pts = atlasnet_model(batch_img, num_points=4096)["points"][0].cpu().numpy()
                verts, faces = atlasnet_model.generate_mesh(batch_img, grid_res=13)
                elapsed_ms = (time.time() - t0) * 1000.0

                metrics_str = ""
                if gt_raw_pc is not None and gt_frame and viewer_score:
                    pred_s, gt_s = map_norm_to_viewer_frame(out_pts, gt_raw_pc)
                    sc = viewer_score(pred_s, gt_s)
                    metrics_str = f" | Viewer **F@5%: {sc['f_5']:.3f}** | **CD-L1: {sc['chamfer_l1']:.3f}**"

                st.caption(f"⚡ Inferred in **{elapsed_ms:.1f} ms** on `{device}` | Template: `{atlasnet_model.template}`{metrics_str}")
                t_mesh, t_pts, t_gt = st.tabs(["AtlasNet Surface Mesh", "4,096 Predicted Points", "Ground Truth"])
                with t_mesh:
                    st.plotly_chart(plot_mesh_3d(verts, faces, title=f"AtlasNet ({atlasnet_model.template}) Mesh", color="#2563eb"), use_container_width=True)
                    st.download_button("💾 Download Mesh (.obj)", data=create_obj_string(verts, faces), file_name=f"{selected_label.replace('/', '_')}_atlasnet.obj")
                with t_pts:
                    st.plotly_chart(plot_point_cloud_3d(out_pts, title="AtlasNet Sampled Cloud", color="#2563eb"), use_container_width=True)
                with t_gt:
                    if gt_raw_pc is not None:
                        gt_norm, _, _ = normalize_point_cloud(gt_raw_pc)
                        st.plotly_chart(plot_point_cloud_3d(gt_norm, title="Laser Scanner GT (4,096 pts)", color="#10b981"), use_container_width=True)
                    else:
                        st.info("No Ground Truth 3D scan for uploaded photos.")
        else:
            st.warning("Upload at least 1 image or pick a dataset object to reconstruct.")


# ==============================================================================
# TAB 2: PRE-COMPUTED / KNOWN SCENE (COLMAP Mathematics)
# ==============================================================================
with tab_precomputed:
    st.subheader("📐 Pre-Computed / Already-Seen Scene Geometry — COLMAP (Classical Mathematics)")
    st.warning(
        "**Use this tab when inspecting a scene that was already captured with 24 multi-view images and pre-computed mathematically.** "
        "**COLMAP** uses SIFT keypoint matching + Epipolar Geometry + Bundle Adjustment / Triangulation (`pycolmap`) rather than learned priors. "
        "It achieves **95.3% spatial precision** on textured surfaces, but cannot infer unseen backsides from 1–3 photos without dense multi-view coverage."
    )

    if colmap_known is None:
        st.error("Pre-computed COLMAP viewer runs not found under `viewer/runs/bones__colmap__sparse_known`.")
    else:
        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            colmap_mode = st.radio(
                "COLMAP Reconstruction Pipeline",
                [
                    "Known Camera Poses (24 GT Cameras + SIFT Triangulation — Best Math)",
                    "Blind Incremental SfM (Estimated Poses + Sim(3) Align)",
                ],
            )
            active_run_id = "bones__colmap__sparse_known" if colmap_mode.startswith("Known") else "bones__colmap__sparse_sfm"
            active_run = colmap_known if active_run_id == "bones__colmap__sparse_known" else colmap_sfm

            colmap_objs = sorted(active_run["objects"].keys())
            colmap_by_cat: Dict[str, List[str]] = {}
            for oid in colmap_objs:
                c, o = oid.split("/")
                colmap_by_cat.setdefault(c, []).append(oid)

            c_cats = sorted(colmap_by_cat.keys())
            sel_c_cat = st.selectbox("Select Pre-Computed Category (76 classes)", c_cats, index=c_cats.index("teddy_bear") if "teddy_bear" in c_cats else 0)
            sel_c_obj = st.selectbox("Select Pre-Computed Object", colmap_by_cat[sel_c_cat])

            show_gt_colmap = st.checkbox("Overlay Ground Truth Cloud (Grey)", value=True)
            show_cams = st.checkbox("Show 24 Camera Rig Positions", value=True)

            # Display reference sample photos
            obj_dir = Path(DATASET_ROOT) / "renders" / sel_c_obj
            if obj_dir.is_dir():
                st.markdown("**Sample Input Views (out of all 24 used by COLMAP):**")
                pcols = st.columns(4)
                for i, v in enumerate([0, 6, 12, 18]):
                    vp = obj_dir / f"{v:03d}.png"
                    if vp.is_file():
                        with pcols[i]:
                            st.image(str(vp), caption=f"v{v:02d}", use_container_width=True)

        with col_c2:
            ent = active_run["objects"][sel_c_obj]
            pts = decode_viewer_points(active_run_id, active_run, sel_c_obj)
            gt_shared = None
            gt_p = Path(DATASET_ROOT) / "point_clouds" / f"{sel_c_obj}.npy"
            if show_gt_colmap and gt_p.is_file() and gt_frame:
                gt_raw = np.load(gt_p)
                c, r = gt_frame(gt_raw)
                gt_shared = (gt_raw.astype(np.float64) - c) / r

            cams = np.array(ent["cameras"], dtype=np.float32) if (show_cams and "cameras" in ent) else None
            cams_used = ent.get("cameras_used") if show_cams else None

            metrics = ent.get("metrics", {})
            info = ent.get("info", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("F-Score@5%", f"{metrics.get('f_5', 0):.3f}")
            m2.metric("Precision@5%", f"{metrics.get('precision_5', 0):.1%}")
            m3.metric("Chamfer-L1", f"{metrics.get('chamfer_l1', 0):.3f}")
            m4.metric("Triangulated Pts", f"{metrics.get('n_points', 0)}")

            st.plotly_chart(
                plot_point_cloud_3d(
                    pts if pts is not None else np.zeros((0, 3)),
                    title=f"COLMAP ({active_run['name']}) — {sel_c_obj}",
                    color="#f59e0b",
                    cameras=cams,
                    cameras_used=cams_used,
                    gt_overlay=gt_shared,
                ),
                use_container_width=True,
            )
            if "excluded" in ent:
                st.warning(f"Excluded from scoring: {ent['excluded']}")
            st.caption(f"Run info: `{info}` | Why pre-computed? Mathematical SIFT triangulation needs all 24 views & texture to solve 3D rays.")


# ==============================================================================
# TAB 3: 3-CATEGORY BENCHMARK ARENA
# ==============================================================================
with tab_arena:
    st.subheader("⚔️ Side-by-Side Comparison: Best Model from Each Category")
    st.markdown(
        "Compare **Classical Math (COLMAP)**, **Volumetric DL (Pix2Vox++ Generalization)**, "
        "**Surface Mesh DL (AtlasNet)**, and **Laser Scanner Ground Truth** on the exact same object."
    )

    shared_cats = sorted(set(colmap_known["objects"].keys()) if colmap_known else [])
    cat_map: Dict[str, List[str]] = {}
    for oid in shared_cats:
        cat_map.setdefault(oid.split("/")[0], []).append(oid)

    if cat_map:
        ac1, ac2 = st.columns([1, 2])
        with ac1:
            a_cat = st.selectbox("Benchmark Category", sorted(cat_map.keys()), index=sorted(cat_map.keys()).index("chair") if "chair" in cat_map else 0, key="arena_c")
        with ac2:
            a_obj = st.selectbox("Benchmark Instance", cat_map[a_cat], key="arena_o")

        a_dir = Path(DATASET_ROOT) / "renders" / a_obj
        imgs_3 = [Image.open(a_dir / f"{i*8:03d}.png") for i in range(3) if (a_dir / f"{i*8:03d}.png").is_file()]
        if len(imgs_3) == 3:
            batch_img = preprocess_pil_images(imgs_3, device)
            gt_raw = np.load(Path(DATASET_ROOT) / "point_clouds" / f"{a_obj}.npy")
            c, r = gt_frame(gt_raw)
            gt_s = (gt_raw.astype(np.float64) - c) / r

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("#### 1. COLMAP (Math)")
                pts_col = decode_viewer_points("bones__colmap__sparse_known", colmap_known, a_obj)
                sc_c = colmap_known["objects"][a_obj].get("metrics", {})
                st.caption(f"24 views + poses | **F@5%: {sc_c.get('f_5', 0):.3f}**")
                st.plotly_chart(plot_point_cloud_3d(pts_col if pts_col is not None else np.zeros((0, 3)), title="COLMAP Known", color="#f59e0b"), use_container_width=True)

            with col2:
                st.markdown("#### 2. Pix2Vox++ (Volumetric)")
                with torch.no_grad():
                    vox = pix2vox_model(batch_img)["voxels"][0].cpu().numpy()
                mesh_p = voxels_to_mesh(vox, threshold=0.40)
                idx = np.argwhere(vox >= 0.40).astype(np.float64)
                f5_p = 0.0
                if len(idx) > 0 and viewer_score:
                    ps, gs = map_norm_to_viewer_frame((idx / 31.0) - 0.5, gt_raw)
                    f5_p = viewer_score(ps, gs)["f_5"]
                st.caption(f"3 views (General) | **F@5%: {f5_p:.3f}**")
                if mesh_p:
                    st.plotly_chart(plot_mesh_3d(mesh_p[0], mesh_p[1], title="Pix2Vox++ Mesh", color="#7c3aed"), use_container_width=True)

            with col3:
                st.markdown("#### 3. AtlasNet (Surface)")
                with torch.no_grad():
                    pts_a = atlasnet_model(batch_img, num_points=4096)["points"][0].cpu().numpy()
                va, fa = atlasnet_model.generate_mesh(batch_img, grid_res=12)
                ps_a, gs_a = map_norm_to_viewer_frame(pts_a, gt_raw)
                f5_a = viewer_score(ps_a, gs_a)["f_5"] if viewer_score else 0.0
                st.caption(f"3 views ({atlasnet_model.template}) | **F@5%: {f5_a:.3f}**")
                st.plotly_chart(plot_mesh_3d(va, fa, title="AtlasNet Mesh", color="#2563eb"), use_container_width=True)

            with col4:
                st.markdown("#### 4. Ground Truth")
                st.caption("4,096 Laser Scanner Points")
                st.plotly_chart(plot_point_cloud_3d(gt_s, title="Ground Truth", color="#10b981"), use_container_width=True)

    st.divider()
    st.markdown("### 📈 Evaluation Reports on Disk (Pre- vs Post-Augmentation)")
    eval_dir = TRACK_ROOT / "outputs" / "evaluations"
    reports = sorted(eval_dir.glob("*_report.md"))
    if reports:
        rep_names = [r.name for r in reports]
        sel_reps = st.multiselect("Select Evaluation Reports to Compare", rep_names, default=rep_names[-3:])
        for rn in sel_reps:
            with st.expander(f"📄 {rn}", expanded=True):
                st.markdown((eval_dir / rn).read_text())


# ==============================================================================
# TAB 4: VIRTUAL 3D ROOM STAGER
# ==============================================================================
with tab_room:
    st.subheader("🏡 Virtual 3D Room Stager (Real Generalized Reconstructions)")
    st.markdown("Reconstruct real furniture objects on the fly with **Pix2Vox++** and stage them interactively in 3D space.")

    c_ctrl, c_canvas = st.columns([1, 2])
    with c_ctrl:
        room_theme = st.selectbox("Floor Material", ["Hardwood Oak", "Polished Marble", "Minimalist Slate"])
        furniture_choices = [c for c in ["chair", "sofa", "table", "bed", "lamp", "light", "cabinet", "vase"] if c in obj_dict]
        item1_cat = st.selectbox("Furniture Item 1", furniture_choices, index=0)
        item1_obj = st.selectbox("Instance 1", obj_dict.get(item1_cat, []))
        pos1 = st.slider("Item 1 Position (X, Y)", -1.5, 1.5, (-0.6, 0.3), step=0.1)

        item2_cat = st.selectbox("Furniture Item 2", furniture_choices, index=min(1, len(furniture_choices) - 1))
        item2_obj = st.selectbox("Instance 2", obj_dict.get(item2_cat, []))
        pos2 = st.slider("Item 2 Position (X, Y)", -1.5, 1.5, (0.5, -0.3), step=0.1)

    with c_canvas:
        room_fig = go.Figure()
        floor_colors = {"Hardwood Oak": "#d4a373", "Polished Marble": "#e2e8f0", "Minimalist Slate": "#64748b"}
        room_fig.add_trace(
            go.Mesh3d(
                x=[-2, 2, 2, -2], y=[-2, -2, 2, 2], z=[0, 0, 0, 0],
                i=[0, 0], j=[1, 2], k=[2, 3],
                color=floor_colors[room_theme], opacity=0.9, name="Floor",
            )
        )

        def reconstruct_item(cat: str, obj: str):
            d = Path(DATASET_ROOT) / "renders" / cat / obj
            ims = [Image.open(d / f"{v:03d}.png") for v in (0, 8, 16) if (d / f"{v:03d}.png").is_file()]
            if not ims:
                return None
            t = preprocess_pil_images(ims, device)
            with torch.no_grad():
                vox = pix2vox_model(t)["voxels"][0].cpu().numpy()
            res = voxels_to_mesh(vox, threshold=0.38)
            return res

        for name, cat, obj, pos, clr, scale in [
            ("Item 1", item1_cat, item1_obj, pos1, "#7c3aed", 0.9),
            ("Item 2", item2_cat, item2_obj, pos2, "#2563eb", 1.0),
        ]:
            res = reconstruct_item(cat, obj)
            if res is not None:
                v, f = res
                z_min = float(v[:, 1].min())
                room_fig.add_trace(
                    go.Mesh3d(
                        x=v[:, 0] * scale + pos[0],
                        y=v[:, 2] * scale + pos[1],
                        z=(v[:, 1] - z_min) * scale,
                        i=f[:, 0], j=f[:, 1], k=f[:, 2],
                        color=clr, flatshading=True, name=f"{obj}",
                    )
                )

        room_fig.update_layout(
            title="Staged 3D Room (Pix2Vox++ Real Reconstructions)",
            scene=dict(
                xaxis=dict(range=[-2.2, 2.2], title="X (m)"),
                yaxis=dict(range=[-2.2, 2.2], title="Y (m)"),
                zaxis=dict(range=[0, 1.8], title="Height (m)"),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.45),
            ),
            height=540,
            margin=dict(l=0, r=0, b=0, t=35),
        )
        st.plotly_chart(room_fig, use_container_width=True)


# ==============================================================================
# TAB 5: DATASET & AUGMENTATION ANALYTICS
# ==============================================================================
with tab_dataset:
    st.subheader("📊 Full OmniObject3D Dataset (215 Categories) & Augmentation Pipeline")

    total_cats = len(obj_dict)
    total_objs = sum(len(v) for v in obj_dict.values())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Categories on Disk", f"{total_cats}")
    m2.metric("Scanned 3D Objects", f"{total_objs:,}")
    m3.metric("Multi-View Images", f"{total_objs * 24:,}")
    m4.metric("Point Cloud Resolution", "4,096 Pts / Obj")

    st.markdown(
        """
        ### 3D-Safe Augmentation & Long-Tail Balancing Strategy (MS3 Upgrade)
        1. **Random Background Domain Randomization**: Composites RGBA foregrounds over clean white (55%), muted solid colors (30%), and noisy gradients (15%) to eliminate white-background overfitting on real photographs.
        2. **Photometric Appearance Jitter**: Random brightness, contrast, saturation, hue shifts, subtle Gaussian blur, and sensor noise across views.
        3. **Scale & Framing Invariance**: Mild `RandomResizedCrop(scale=0.86..1.0)` preserves canonical [-0.5, 0.5]³ 3D pose alignment while preventing pixel-coordinate memorization.
        4. **Random View Order + Multi-K Training**: Viewpoints sampled across all 24 azimuth/elevation angles (`K ∈ {1, 2, 3, 5}`) train a single network to handle arbitrary view counts.
        5. **Class-Balanced Sampling ($w_c \\propto N_c^{-0.5}$)**: Prevents frequent classes (`bread`: 156, `box`: 134, `biscuit`: 118) from drowning rare furniture classes (`bed`: 2, `plug`: 2, `drum`: 2).
        """
    )

    if obj_dict:
        df_counts = pd.DataFrame([{"Category": k, "Objects": len(v)} for k, v in obj_dict.items()]).sort_values("Objects", ascending=False)
        top_n = st.slider("Top/Bottom Categories to Plot", 15, min(60, len(df_counts)), 30)
        fig_hist = go.Figure()
        top_df = pd.concat([df_counts.head(top_n // 2), df_counts.tail(top_n // 2)])
        fig_hist.add_trace(go.Bar(x=top_df["Category"], y=top_df["Objects"], marker_color="#3b82f6"))
        fig_hist.update_layout(title="Long-Tail Category Counts (Largest vs Smallest Classes)", xaxis_tickangle=-45, height=420)
        st.plotly_chart(fig_hist, use_container_width=True)
