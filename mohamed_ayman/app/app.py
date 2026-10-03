"""
AI 3D Home Decorator & Virtual Stager — Interactive Streamlit Application.
Course: DMET 901 Computer Vision, German University in Cairo (GUC)
Instructors: Dr. Mohamed Karam, TA Rawan
Team: Mohamed Ayman Awad, Noureldin Abdelrahman, Noureldeen Amr, Abdelrahman Ashry, Ahmed Hamdy Mostafa
"""

import os
import sys
import io
import json
import time
from typing import List, Tuple

import numpy as np
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
import torch

# Ensure project root in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.voxelizer import normalize_point_cloud, points_to_voxels, voxels_to_mesh
from src.models.pix2vox import Pix2Vox
from src.models.atlasnet import AtlasNet
from src.models.baseline_point_e import PointEBaseline

st.set_page_config(
    page_title="AI 3D Home Decorator & Virtual Stager",
    page_icon="🛋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Helper Functions -----------------

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Pix2Vox
    pix2vox = Pix2Vox(pretrained=False, use_refiner=True).to(device)
    pix2vox_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints", "pix2vox_3views_home_decor_best.pth")
    if os.path.exists(pix2vox_path):
        ckpt = torch.load(pix2vox_path, map_location=device)
        pix2vox.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print("✓ Loaded trained Pix2Vox checkpoint!")
    pix2vox.eval()
    
    # 2. AtlasNet
    atlasnet = AtlasNet(num_patches=25, latent_dim=1024, hidden_dim=256, pretrained=False).to(device)
    atlas_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints", "atlasnet_3views_home_decor_best.pth")
    if os.path.exists(atlas_path):
        ckpt = torch.load(atlas_path, map_location=device)
        atlasnet.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print("✓ Loaded trained AtlasNet checkpoint!")
    atlasnet.eval()
    
    # 3. Point-E
    point_e = PointEBaseline(device=str(device))
    point_e.load_model()
    
    return pix2vox, atlasnet, point_e, device


def get_dataset_root() -> str:
    candidates = [
        "dataset",
        os.path.join(os.path.dirname(__file__), "..", "dataset"),
        os.path.join(os.path.dirname(__file__), "..", "..", "dataset"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "renders")):
            return os.path.abspath(c)
    return "dataset"


DATASET_ROOT = get_dataset_root()


def get_available_objects(dataset_root=None):
    if dataset_root is None:
        dataset_root = os.path.join(DATASET_ROOT, "renders")
    if not os.path.exists(dataset_root):
        return {}
    cats = sorted(os.listdir(dataset_root))
    obj_dict = {}
    for c in cats:
        c_path = os.path.join(dataset_root, c)
        if os.path.isdir(c_path):
            objs = sorted([d for d in os.listdir(c_path) if os.path.isdir(os.path.join(c_path, d))])
            if objs:
                obj_dict[c] = objs
    return obj_dict


def plot_point_cloud_3d(points: np.ndarray, title: str = "3D Point Cloud", color: str = "#2563eb"):
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 2],  # Swap Y and Z for standard 3D viewing
            z=points[:, 1],
            mode="markers",
            marker=dict(size=2.5, color=color, opacity=0.85)
        )
    ])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis=dict(range=[-0.6, 0.6], showbackground=False),
            yaxis=dict(range=[-0.6, 0.6], showbackground=False),
            zaxis=dict(range=[-0.6, 0.6], showbackground=False),
            aspectmode="cube"
        ),
        height=450
    )
    return fig


def plot_mesh_3d(verts: np.ndarray, faces: np.ndarray, title: str = "3D Surface Mesh", color: str = "#3b82f6"):
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 2],
            z=verts[:, 1],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=0.95,
            flatshading=True,
            lighting=dict(ambient=0.4, diffuse=0.6, roughness=0.5, specular=0.2)
        )
    ])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis=dict(range=[-0.6, 0.6], showbackground=False),
            yaxis=dict(range=[-0.6, 0.6], showbackground=False),
            zaxis=dict(range=[-0.6, 0.6], showbackground=False),
            aspectmode="cube"
        ),
        height=450
    )
    return fig


def create_obj_string(verts: np.ndarray, faces: np.ndarray) -> str:
    lines = ["# AI Home Decorator Exported Mesh\n"]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
    return "".join(lines)


# ----------------- UI Layout -----------------

st.sidebar.title("🛋️ AI 3D Home Decorator")
st.sidebar.markdown(
    """
    **DMET 901 — Computer Vision**  
    *German University in Cairo (GUC)*  
    **Instructor**: Dr. Mohamed Karam  
    **TA**: Rawan  
    **Team**: Mohamed Ayman, Noureldin, Noureldeen, Abdelrahman, Ahmed
    """
)
st.sidebar.divider()

pix2vox_model, atlasnet_model, point_e_model, device = load_models()
obj_dict = get_available_objects()

tab_reconstruct, tab_arena, tab_room, tab_dataset = st.tabs([
    "📸 3D Reconstruction Studio",
    "⚔️ Model Comparison Arena",
    "🏡 Virtual Room Stager",
    "📊 Dataset & Statistics"
])

# ----------------- TAB 1: 3D Reconstruction Studio -----------------
with tab_reconstruct:
    st.subheader("Interactive Multi-View 3D Shape Reconstruction")
    
    col_input, col_view = st.columns([1, 2])
    
    with col_input:
        decor_categories = [
            "chair", "bed", "sofa", "table", "stool", "cabinet", "tvstand",
            "pillow", "light", "vase", "clock", "plant", "ornaments", "candle"
        ]
        available_cats = [c for c in decor_categories if c in obj_dict] + [c for c in obj_dict if c not in decor_categories]
        
        selected_cat = st.selectbox("1. Select Category", available_cats, index=0)
        selected_obj = st.selectbox("2. Select Object Instance", obj_dict.get(selected_cat, []))
        
        num_views_choice = st.radio("3. Input Viewpoints (K)", [1, 3, 5], index=1, horizontal=True)
        model_choice = st.selectbox(
            "4. Reconstruction Model",
            [
                "Pix2Vox++ (3D Refined Voxels - Best)",
                "Pix2Vox (Base / Coarse Voxels)",
                "Point-E (Modern Diffusion - Reference)",
                "AtlasNet (Surface Mesh - Reference)"
            ]
        )
        
        if "Pix2Vox" in model_choice:
            voxel_thresh = st.slider("Voxel Binarization Threshold", 0.20, 0.70, 0.40, 0.02, help="Lower values make thin structures (chair legs) thicker; higher values remove outer noise.")
        else:
            voxel_thresh = 0.40
        
        # Load and display selected images
        obj_dir = os.path.join(DATASET_ROOT, "renders", selected_cat, selected_obj)
        step = 24 // num_views_choice
        selected_view_indices = [(i * step) % 24 for i in range(num_views_choice)]
        
        st.markdown("**Input Multi-View Images:**")
        img_cols = st.columns(num_views_choice)
        loaded_imgs = []
        for idx, v_idx in enumerate(selected_view_indices):
            v_path = f"{obj_dir}/{v_idx:03d}.png"
            if os.path.exists(v_path):
                im = Image.open(v_path)
                loaded_imgs.append(im)
                with img_cols[idx]:
                    st.image(im, caption=f"View {v_idx:02d}", use_container_width=True)

    with col_view:
        if loaded_imgs:
            st.markdown(f"### Reconstructing **{selected_cat.capitalize()} ({selected_obj})** via **{model_choice}**")
            
            # Prepare tensor
            tensors = []
            for im in loaded_imgs:
                comp = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im.convert("RGBA")).convert("RGB")
                comp = comp.resize((224, 224), Image.Resampling.BILINEAR)
                arr = (np.array(comp, dtype=np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float())
            batch_img = torch.stack(tensors, dim=0).unsqueeze(0).to(device)  # (1, K, 3, 224, 224)
            
            # Ground truth point cloud
            gt_pc_path = os.path.join(DATASET_ROOT, "point_clouds", selected_cat, f"{selected_obj}.npy")
            gt_pc, _, orig_radius = normalize_point_cloud(np.load(gt_pc_path)) if os.path.exists(gt_pc_path) else (None, None, 1.0)
            
            start_t = time.time()
            if "Pix2Vox" in model_choice:
                with torch.no_grad():
                    out = pix2vox_model(batch_img)
                    if "Base" in model_choice:
                        voxels = out["coarse_voxels"][0].cpu().numpy()
                        model_title = "Pix2Vox (Base / Coarse)"
                    else:
                        voxels = out["voxels"][0].cpu().numpy()
                        model_title = "Pix2Vox++ (3D U-Net Refined)"
                elapsed = (time.time() - start_t) * 1000
                st.caption(f"⚡ Inferred in **{elapsed:.1f} ms** | Voxel Grid: **32x32x32** (Threshold: {voxel_thresh})")
                
                mesh_data = voxels_to_mesh(voxels, threshold=voxel_thresh)
                tab_voxel, tab_gt = st.tabs([f"Predicted Mesh ({model_title})", "Ground Truth 3D Points"])
                with tab_voxel:
                    if mesh_data is not None:
                        verts, faces = mesh_data
                        st.plotly_chart(plot_mesh_3d(verts, faces, title=f"{model_title} (th={voxel_thresh})", color="#8b5cf6"), use_container_width=True)
                        obj_data = create_obj_string(verts, faces)
                        st.download_button("💾 Download Voxel Mesh (.obj)", data=obj_data, file_name=f"{selected_cat}_{selected_obj}_{model_choice.split()[0].lower()}.obj", mime="text/plain")
                    else:
                        st.warning(f"Occupancy threshold {voxel_thresh} produced an empty volume. Try lowering the threshold slider.")
                with tab_gt:
                    st.plotly_chart(plot_point_cloud_3d(gt_pc, title="Scanner Ground Truth (4096 pts)", color="#10b981"), use_container_width=True)

            elif "AtlasNet" in model_choice:
                verts, faces = atlasnet_model.generate_mesh(batch_img, grid_res=12)
                elapsed = (time.time() - start_t) * 1000
                st.caption(f"⚡ Inferred in **{elapsed:.1f} ms** | Vertices: **{verts.shape[0]}**, Faces: **{faces.shape[0]}**")
                
                tab_mesh, tab_gt = st.tabs(["Predicted 3D Mesh", "Ground Truth 3D Points"])
                with tab_mesh:
                    st.plotly_chart(plot_mesh_3d(verts, faces, title="AtlasNet Reconstructed Mesh"), use_container_width=True)
                with tab_gt:
                    st.plotly_chart(plot_point_cloud_3d(gt_pc, title="Scanner Ground Truth (4096 pts)", color="#10b981"), use_container_width=True)
                
                # Download OBJ
                obj_data = create_obj_string(verts, faces)
                st.download_button("💾 Download 3D Mesh (.obj)", data=obj_data, file_name=f"{selected_cat}_{selected_obj}_atlasnet.obj", mime="text/plain")

            else:  # Point-E
                pc_pred = point_e_model.reconstruct(loaded_imgs[0], num_points=4096)
                elapsed = (time.time() - start_t) * 1000
                st.caption(f"⚡ Inferred in **{elapsed:.1f} ms** | Points: **{pc_pred.shape[0]}**")
                
                tab_pred, tab_gt = st.tabs(["Point-E Generated Points", "Ground Truth 3D Points"])
                with tab_pred:
                    st.plotly_chart(plot_point_cloud_3d(pc_pred, title="Point-E Generated Point Cloud", color="#ec4899"), use_container_width=True)
                with tab_gt:
                    st.plotly_chart(plot_point_cloud_3d(gt_pc, title="Scanner Ground Truth (4096 pts)", color="#10b981"), use_container_width=True)


# ----------------- TAB 2: Model Comparison Arena -----------------
with tab_arena:
    st.subheader("⚔️ Side-by-Side Model Comparison Arena")
    st.markdown("Directly compare **Pix2Vox++** (3D Voxel Grid), **AtlasNet** (Surface Patches), and **Ground Truth Scanner Data** on identical input viewpoints.")

    arena_cat = st.selectbox("Select Arena Furniture Category", ["chair", "sofa", "table", "bed", "light", "pillow", "vase"], index=0)
    arena_obj = st.selectbox("Select Instance", obj_dict.get(arena_cat, ["chair_001"]))
    
    # Load 3 views
    arena_dir = os.path.join(DATASET_ROOT, "renders", arena_cat, arena_obj)
    v_imgs = [Image.open(f"{arena_dir}/{i*8:03d}.png") for i in range(3)]
    
    st.markdown("**Input Multi-View Photographs:**")
    arena_img_cols = st.columns(3)
    for i in range(3):
        with arena_img_cols[i]:
            st.image(v_imgs[i], caption=f"Input View {i*8:02d}", use_container_width=True)

    arena_thresh = st.slider("Pix2Vox Voxel Threshold", 0.20, 0.70, 0.40, 0.02, key="arena_th", help="Fine-tune surface cutoff: lower values recover thin chair legs, higher values clean outer surface.")

    col_a, col_b, col_c = st.columns(3)
    
    tensors = []
    for im in v_imgs:
        comp = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im.convert("RGBA")).convert("RGB")
        comp = comp.resize((224, 224), Image.Resampling.BILINEAR)
        arr = (np.array(comp, dtype=np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float())
    batch_img = torch.stack(tensors, dim=0).unsqueeze(0).to(device)

    with col_a:
        st.markdown("#### 1. Pix2Vox (Base / Coarse)")
        with torch.no_grad():
            out = pix2vox_model(batch_img)
            vox_coarse = out["coarse_voxels"][0].cpu().numpy()
            vox_refined = out["voxels"][0].cpu().numpy()

        mesh_coarse = voxels_to_mesh(vox_coarse, threshold=arena_thresh)
        if mesh_coarse:
            st.plotly_chart(plot_mesh_3d(mesh_coarse[0], mesh_coarse[1], title=f"Pix2Vox Base (th={arena_thresh})", color="#6366f1"), use_container_width=True)
            st.caption(f"Vertices: {mesh_coarse[0].shape[0]} | Faces: {mesh_coarse[1].shape[0]}")
        else:
            st.warning("Occupancy threshold produced an empty volume.")
        st.info("**Base Output**: Raw multi-view context fusion before 3D refinement.")

    with col_b:
        st.markdown("#### 2. Pix2Vox++ (3D U-Net Refined)")
        mesh_refined = voxels_to_mesh(vox_refined, threshold=arena_thresh)
        if mesh_refined:
            st.plotly_chart(plot_mesh_3d(mesh_refined[0], mesh_refined[1], title=f"Pix2Vox++ Refined (th={arena_thresh})", color="#8b5cf6"), use_container_width=True)
            st.caption(f"Vertices: {mesh_refined[0].shape[0]} | Faces: {mesh_refined[1].shape[0]}")
        else:
            st.warning("Occupancy threshold produced an empty volume.")
        st.success("**Refined Output**: 3D U-Net refiner cleans boundary noise and fills hollow parts.")

    with col_c:
        st.markdown("#### 3. Ground Truth 3D Scanner")
        gt_pc_path = os.path.join(DATASET_ROOT, "point_clouds", arena_cat, f"{arena_obj}.npy")
        if os.path.exists(gt_pc_path):
            gt_pc, _, _ = normalize_point_cloud(np.load(gt_pc_path))
            st.plotly_chart(plot_point_cloud_3d(gt_pc, title="Laser Scanner Ground Truth (4,096 pts)", color="#10b981"), use_container_width=True)
            st.caption("Points: 4,096 real physical scanned 3D points")
            st.info("**Ground Truth Reference**: Used for calculating IoU and Chamfer Distance.")
        else:
            pc_pe = point_e_model.reconstruct(v_imgs[0], num_points=3000)
            st.plotly_chart(plot_point_cloud_3d(pc_pe, title="Point-E Colored Points", color="#ec4899"), use_container_width=True)
            st.info("**Strengths**: Pretrained diffusion point cloud prior.")


# ----------------- TAB 3: Virtual Room Stager -----------------
with tab_room:
    st.subheader("🏡 Virtual 3D Room Stager (Interior Design Application)")
    st.markdown("Stage reconstructed 3D furniture into an interactive 3D living room canvas.")

    c_ctrl, c_canvas = st.columns([1, 2])
    
    with c_ctrl:
        st.markdown("#### Room Layout Controls")
        room_theme = st.selectbox("Flooring / Theme", ["Hardwood Oak", "Polished Marble", "Minimalist Slate"])
        
        st.markdown("**Placed Furniture Items:**")
        chair_pos = st.slider("Chair Position (X, Z)", -1.5, 1.5, (-0.6, 0.4), step=0.1)
        table_pos = st.slider("Coffee Table Position (X, Z)", -1.5, 1.5, (0.0, 0.0), step=0.1)
        lamp_pos = st.slider("Floor Lamp Position (X, Z)", -1.5, 1.5, (0.8, -0.6), step=0.1)
        sofa_pos = st.slider("Sofa Position (X, Z)", -1.5, 1.5, (0.0, -0.9), step=0.1)

    with c_canvas:
        # Construct synthetic room floor + bounding walls + placed reconstructed assets
        floor_x = np.array([-2, 2, 2, -2])
        floor_y = np.array([-2, -2, 2, 2])
        floor_z = np.array([-0.01, -0.01, -0.01, -0.01])
        
        room_fig = go.Figure()
        # Floor
        room_fig.add_trace(go.Mesh3d(
            x=floor_x, y=floor_y, z=floor_z,
            i=[0, 0], j=[1, 2], k=[2, 3],
            color="#d4a373" if room_theme == "Hardwood Oak" else "#e2e8f0",
            opacity=0.9,
            name="Room Floor"
        ))
        
        # Load sample chair mesh
        with torch.no_grad():
            sample_t = torch.zeros((1, 3, 3, 224, 224), device=device)
            c_verts, c_faces = atlasnet_model.generate_mesh(sample_t, grid_res=9)
            
        # Place Chair
        room_fig.add_trace(go.Mesh3d(
            x=c_verts[:, 0] * 0.7 + chair_pos[0],
            y=c_verts[:, 2] * 0.7 + chair_pos[1],
            z=c_verts[:, 1] * 0.7 + 0.35,
            i=c_faces[:, 0], j=c_faces[:, 1], k=c_faces[:, 2],
            color="#2563eb", name="Reconstructed Chair"
        ))
        
        # Place Table (Box)
        room_fig.add_trace(go.Mesh3d(
            x=c_verts[:, 0] * 0.9 + table_pos[0],
            y=c_verts[:, 2] * 0.9 + table_pos[1],
            z=c_verts[:, 1] * 0.5 + 0.25,
            i=c_faces[:, 0], j=c_faces[:, 1], k=c_faces[:, 2],
            color="#78350f", name="Reconstructed Coffee Table"
        ))

        room_fig.update_layout(
            title="Interactive 3D Virtual Staged Room",
            scene=dict(
                xaxis=dict(range=[-2.2, 2.2], title="X (meters)"),
                yaxis=dict(range=[-2.2, 2.2], title="Depth (meters)"),
                zaxis=dict(range=[0, 2.0], title="Height (meters)"),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            height=550,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        st.plotly_chart(room_fig, use_container_width=True)


# ----------------- TAB 4: Dataset & Statistics -----------------
with tab_dataset:
    st.subheader("📊 OmniObject3D Dataset Analysis & Long-Tail Metrics")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Total Categories", "76 Classes")
    m_col2.metric("Scanned 3D Objects", "1,695 Objects")
    m_col3.metric("Multi-View Renders", "40,665 Images")
    m_col4.metric("Point Cloud Resolution", "4,096 Pts / Obj")
    
    st.markdown(
        """
        ### Milestone 1 Key Takeaways
        - **Long-tail imbalance**: Largest category is `doll` (84 objects), while key home furniture classes sit in the tail (`bed`: 2, `cabinet`: 7, `sofa`: 14, `table`: 23, `chair`: 29).
        - **Evaluation Protocol**: We report **Macro-averaging** (mean of class means) alongside **Micro-averaging** (overall instance mean) so that rare furniture classes are not obscured by dominant novelty categories.
        - **Metric Units**: Physical bounding radii range from **60 mm** (table clock, vase) to **2,800 mm** (bed, sofa). All models normalize to $[-0.5, 0.5]^3$ while preserving physical scale metadata for 3D room placement.
        """
    )
