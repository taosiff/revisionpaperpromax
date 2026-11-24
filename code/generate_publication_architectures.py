"""
code/generate_publication_architectures.py
=========================================
Publication-Grade 3D Scientific Architecture Figures (Figs 1, 2, 3, 4)
Mathematical and Logical Alignment with Q1 Journal Review Standards

Generates:
  - Fig 1: Overall Multimodal Framework (Rank Averaging & Dawes' Decision Fusion)
  - Fig 2: Module 1 — Structural Neuroimaging Pipeline (Med3D 3D-ResNet-18 + Train-Only PCA-5 + Balanced LR)
  - Fig 3: Module 2 — Behavioral Questionnaires Pipeline (9 Transition Markers -> PCA-3 Factor Orthogonalization & FT-ProtoNet)
  - Fig 4: Detailed Deep Feature-Tokenizer ProtoNet Architecture (Feature Tokenizer -> Pre-LN Multi-Head Attention -> Query Pooling -> S^15 Metric Head)
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIRS = [
    BASE_DIR / 'results',
    BASE_DIR / 'prev_submissions',
    BASE_DIR / 'REVISION_PAPER_PRO_MAX' / 'figures'
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# Typography settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#0F172A'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# -------------------------------------------------------------------------
# CORE 3D DRAWING HELPERS
# -------------------------------------------------------------------------
def draw_3d_block(ax, x, y, w, h, depth=0.35, angle=35,
                  face_col='#1F618D', top_col='#5DADE2', side_col='#154360',
                  edge_col='#0E2F44', lw=1.1, zorder=10):
    """Draws an isometric 3D block with top highlight and right shading."""
    rad = np.radians(angle)
    dx = depth * np.cos(rad)
    dy = depth * np.sin(rad)
    
    # Front face
    front = patches.Rectangle((x, y), w, h, facecolor=face_col,
                              edgecolor=edge_col, linewidth=lw, zorder=zorder)
    ax.add_patch(front)
    
    # Top face (Highlight)
    top_pts = np.array([[x, y + h], [x + w, y + h], [x + w + dx, y + h + dy], [x + dx, y + h + dy]])
    top = patches.Polygon(top_pts, closed=True, facecolor=top_col,
                          edgecolor=edge_col, linewidth=lw, zorder=zorder+1)
    ax.add_patch(top)
    
    # Right side face (Shading)
    side_pts = np.array([[x + w, y], [x + w + dx, y + dy], [x + w + dx, y + h + dy], [x + w, y + h]])
    side = patches.Polygon(side_pts, closed=True, facecolor=side_col,
                           edgecolor=edge_col, linewidth=lw, zorder=zorder+1)
    ax.add_patch(side)
    return dx, dy

def draw_translucent_cube(ax, x, y, w, h, depth=0.6, angle=30,
                          face_col='#E0F2FE', edge_col='#0284C7', alpha=0.30, lw=1.4, zorder=10):
    """Draws a translucent 3D isometric glass cuboid."""
    rad = np.radians(angle)
    dx = depth * np.cos(rad)
    dy = depth * np.sin(rad)
    
    # Back wireframe edges
    ax.plot([x + dx, x + w + dx], [y + dy, y + dy], color=edge_col, lw=1.1, linestyle=':', zorder=zorder)
    ax.plot([x + dx, x + dx], [y + dy, y + h + dy], color=edge_col, lw=1.1, linestyle=':', zorder=zorder)
    ax.plot([x, x + dx], [y, y + dy], color=edge_col, lw=1.1, linestyle=':', zorder=zorder)

    # Front face
    front = patches.Rectangle((x, y), w, h, facecolor=face_col,
                              edgecolor=edge_col, alpha=alpha, linewidth=lw, zorder=zorder+2)
    ax.add_patch(front)

    # Top face
    top = patches.Polygon([[x, y+h], [x+w, y+h], [x+w+dx, y+h+dy], [x+dx, y+h+dy]], closed=True,
                          facecolor=face_col, edgecolor=edge_col, alpha=alpha+0.15, linewidth=lw, zorder=zorder+3)
    ax.add_patch(top)

    # Right side face
    side = patches.Polygon([[x+w, y], [x+w+dx, y+dy], [x+w+dx, y+h+dy], [x+w, y+h]], closed=True,
                           facecolor=face_col, edgecolor=edge_col, alpha=alpha+0.10, linewidth=lw, zorder=zorder+3)
    ax.add_patch(side)
    return dx, dy

def draw_pill(ax, x, y, w, h, text='', face_col='#FFFFFF', edge_col='#94A3B8',
              text_col='#0F172A', font_size=8.5, font_weight='bold', zorder=20):
    """Draws a clean rounded card / pill container with centered text."""
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.14",
                                 facecolor=face_col, edgecolor=edge_col, linewidth=1.1, zorder=zorder)
    ax.add_patch(box)
    if text:
        ax.text(x + w/2.0, y + h/2.0, text, ha='center', va='center',
                color=text_col, fontsize=font_size, fontweight=font_weight, zorder=zorder+5)

def draw_addition_circle(ax, cx, cy, r=0.20, face_col='#DCFCE7', edge_col='#16A34A', text_col='#14532D', zorder=25):
    """Draws an explicit circular addition junction with a + sign."""
    ax.add_patch(patches.Circle((cx, cy), r, facecolor=face_col, edgecolor=edge_col, lw=1.5, zorder=zorder))
    ax.text(cx, cy, "+", ha='center', va='center', color=text_col, fontsize=12.0, fontweight='bold', zorder=zorder+2)

def draw_branch_dot(ax, cx, cy, r=0.065, col='#0F172A', zorder=25):
    """Draws a solid branch point dot."""
    ax.add_patch(patches.Circle((cx, cy), r, facecolor=col, edgecolor='none', zorder=zorder))

def draw_arrow(ax, x1, y1, x2, y2, color='#64748B', lw=1.6, badge='', badge_col='#0284C7', zorder=15):
    """Draws a crisp directional arrow with optional weight badge."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=14),
                zorder=zorder)
    if badge:
        bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bbox_p = dict(boxstyle="round,pad=0.25,rounding_size=0.2", facecolor=badge_col,
                      edgecolor='none', alpha=0.95)
        ax.text(bx, by, badge, ha='center', va='center', color='white',
                fontsize=8.0, fontweight='bold', bbox=bbox_p, zorder=zorder+5)

def draw_sphere(ax, cx, cy, r=0.28, base_col='#10B981', highlight_col='#6EE7B7', edge_col='#047857', zorder=30):
    """Draws a pseudo-3D shaded sphere with realistic radial lighting."""
    ax.add_patch(patches.Circle((cx, cy), r, facecolor=base_col, edgecolor=edge_col, lw=1.2, zorder=zorder))
    ax.add_patch(patches.Circle((cx - r*0.35, cy + r*0.35), r*0.45,
                                facecolor=highlight_col, edgecolor='none', alpha=0.75, zorder=zorder+1))
    ax.add_patch(patches.Circle((cx - r*0.40, cy + r*0.40), r*0.20,
                                facecolor='white', edgecolor='none', alpha=0.90, zorder=zorder+2))

# -------------------------------------------------------------------------
# VECTOR ICONS & AUTHENTIC SCIENTIFIC VISUALIZATIONS
# -------------------------------------------------------------------------
def draw_brain_vector(ax, cx, cy, s=0.26, col='white', zorder=35):
    """Draws a crisp anatomical brain vector silhouette."""
    for dx, dy in [(-s*0.35, 0), (s*0.35, 0), (-s*0.25, s*0.3), (s*0.25, s*0.3)]:
        ax.add_patch(patches.Circle((cx + dx, cy + dy), s*0.36, facecolor='none', edgecolor=col, lw=1.3, zorder=zorder))
    ax.plot([cx, cx], [cy - s*0.4, cy + s*0.5], color=col, lw=1.4, zorder=zorder)
    ax.plot([cx - s*0.28, cx - s*0.08], [cy + s*0.1, cy + s*0.18], color=col, lw=1.1, zorder=zorder)
    ax.plot([cx + s*0.08, cx + s*0.28], [cy + s*0.1, cy + s*0.18], color=col, lw=1.1, zorder=zorder)

def draw_clipboard_vector(ax, cx, cy, s=0.28, col='#0284C7', zorder=35):
    """Draws a crisp clinical clipboard vector with checklist."""
    w, h = s * 0.90, s * 1.25
    ax.add_patch(patches.Rectangle((cx - w/2, cy - h/2), w, h, facecolor='white', edgecolor=col, lw=1.4, zorder=zorder))
    ax.add_patch(patches.Rectangle((cx - w*0.3, cy + h*0.38), w*0.6, h*0.18, facecolor=col, edgecolor=col, lw=0.8, zorder=zorder+1))
    for dy in [-0.25, 0.05, 0.32]:
        ax.plot([cx - w*0.12, cx + w*0.32], [cy + dy*h*0.65, cy + dy*h*0.65], color='#64748B', lw=1.2, zorder=zorder+1)
        ax.plot([cx - w*0.35, cx - w*0.25, cx - w*0.18], [cy + dy*h*0.65, cy + dy*h*0.65 - 0.04, cy + dy*h*0.65 + 0.05],
                color='#10B981', lw=1.3, zorder=zorder+1)

def draw_user_vector(ax, cx, cy, s=0.26, col='white', zorder=35):
    """Draws a crisp participant / demographic avatar vector."""
    ax.add_patch(patches.Circle((cx, cy + s*0.38), s*0.30, facecolor='none', edgecolor=col, lw=1.3, zorder=zorder))
    ax.add_patch(patches.Arc((cx, cy - s*0.48), s*1.15, s*1.15, angle=0, theta1=35, theta2=145, edgecolor=col, lw=1.3, zorder=zorder))
    ax.plot([cx - s*0.35, cx - s*0.15, cx, cx + s*0.15, cx + s*0.35],
            [cy - s*0.2, cy - s*0.2, cy - s*0.05, cy - s*0.35, cy - s*0.2], color=col, lw=1.1, zorder=zorder)

def draw_scientific_mri_volume(ax, cx, cy, scale=1.0, zorder=25):
    """
    Renders an authentic, publication-grade 3D isometric volumetric MRI representation.
    Displays a 3D isometric bounding cube with an anatomical sagittal midline brain projection:
    - Bounding wireframe coordinate box representing the 91x109x91 MNI152 volume
    - Sagittal brain cross-section showing cerebrum, corpus callosum arch, cerebellum, brainstem
    - Zero facial/emoji resemblance.
    """
    # 3D bounding wireframe cube
    w, h, depth = 1.65 * scale, 1.45 * scale, 0.55 * scale
    angle_rad = np.radians(35)
    dx = depth * np.cos(angle_rad)
    dy = depth * np.sin(angle_rad)
    x0, y0 = cx - (w + dx)/2.0, cy - (h + dy)/2.0

    # Back wireframe edges
    ax.plot([x0 + dx, x0 + w + dx], [y0 + dy, y0 + dy], color='#0284C7', lw=0.9, linestyle=':', alpha=0.6, zorder=zorder)
    ax.plot([x0 + dx, x0 + dx], [y0 + dy, y0 + h + dy], color='#0284C7', lw=0.9, linestyle=':', alpha=0.6, zorder=zorder)
    ax.plot([x0, x0 + dx], [y0, y0 + dy], color='#0284C7', lw=0.9, linestyle=':', alpha=0.6, zorder=zorder)

    # Front face edges
    ax.plot([x0, x0 + w], [y0, y0], color='#38BDF8', lw=1.3, zorder=zorder+1)
    ax.plot([x0, x0], [y0, y0 + h], color='#38BDF8', lw=1.3, zorder=zorder+1)
    ax.plot([x0 + w, x0 + w], [y0, y0 + h], color='#38BDF8', lw=1.3, zorder=zorder+1)
    ax.plot([x0, x0 + w], [y0 + h, y0 + h], color='#38BDF8', lw=1.3, zorder=zorder+1)

    # Top face edges
    ax.plot([x0 + w, x0 + w + dx], [y0 + h, y0 + h + dy], color='#38BDF8', lw=1.3, zorder=zorder+1)
    ax.plot([x0, x0 + dx], [y0 + h, y0 + h + dy], color='#38BDF8', lw=1.3, zorder=zorder+1)
    ax.plot([x0 + dx, x0 + w + dx], [y0 + h + dy, y0 + h + dy], color='#38BDF8', lw=1.3, zorder=zorder+1)

    # Right side face edges
    ax.plot([x0 + w, x0 + w + dx], [y0, y0 + dy], color='#38BDF8', lw=1.3, zorder=zorder+1)
    ax.plot([x0 + w + dx, x0 + w + dx], [y0 + dy, y0 + h + dy], color='#38BDF8', lw=1.3, zorder=zorder+1)

    # Sagittal Brain Hemisphere Profile (Side View inside volume)
    # Frontal pole -> Parietal -> Occipital -> Cerebellum -> Brainstem
    sag_x = cx + np.array([-0.48, -0.38, -0.15, 0.15, 0.40, 0.46, 0.35, 0.30, 0.08, -0.05, -0.20, -0.38, -0.48]) * scale
    sag_y = cy + np.array([ 0.05,  0.30,  0.45, 0.42, 0.28, 0.05, -0.12, -0.28, -0.36, -0.24, -0.12, -0.04,  0.05]) * scale
    ax.fill(sag_x, sag_y, color='#1E3A8A', alpha=0.65, zorder=zorder+2)
    ax.plot(sag_x, sag_y, color='#60A5FA', lw=1.4, zorder=zorder+3)

    # Cerebellum contour (Posterior-inferior arbor vitae structure)
    cb_x = cx + np.array([0.18, 0.35, 0.38, 0.25, 0.15]) * scale
    cb_y = cy + np.array([-0.14, -0.18, -0.30, -0.34, -0.24]) * scale
    ax.fill(cb_x, cb_y, color='#1E40AF', alpha=0.80, zorder=zorder+4)
    ax.plot(cb_x, cb_y, color='#93C5FD', lw=1.2, zorder=zorder+5)

    # Corpus Callosum C-shaped arch (thick white matter tract)
    cc_t = np.linspace(0.2, np.pi - 0.2, 30)
    cc_x = cx + 0.18 * scale * np.cos(cc_t)
    cc_y = cy + 0.10 * scale * np.sin(cc_t) + 0.04 * scale
    ax.plot(cc_x, cc_y, color='#E0F2FE', lw=1.8, zorder=zorder+6)

    # Brainstem descending through foramen magnum
    bs_x = cx + np.array([0.04, 0.10, 0.06, -0.02]) * scale
    bs_y = cy + np.array([-0.08, -0.36, -0.42, -0.16]) * scale
    ax.fill(bs_x, bs_y, color='#0284C7', alpha=0.75, zorder=zorder+4)
    ax.plot(bs_x, bs_y, color='#BAE6FD', lw=1.1, zorder=zorder+5)

    # Coordinate labels
    ax.text(x0 - 0.06, y0 + h + 0.04, 'Z (Sup)', color='#38BDF8', fontsize=6.5, fontweight='bold', zorder=zorder+7)
    ax.text(x0 + w + dx - 0.08, y0 + dy - 0.12, 'Y (Ant)', color='#38BDF8', fontsize=6.5, fontweight='bold', zorder=zorder+7)
    ax.text(x0 + w - 0.08, y0 - 0.12, 'X (Lat)', color='#38BDF8', fontsize=6.5, fontweight='bold', zorder=zorder+7)


def save_figure_to_all_locations(fig, filename_base):
    """Saves high-res PNG and TIF to all necessary project folders."""
    for out_dir in OUT_DIRS:
        png_path = out_dir / f"{filename_base}.png"
        tif_path = out_dir / f"{filename_base}.tif"
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(tif_path, dpi=300, bbox_inches='tight')
    print(f"  [SAVED] {filename_base}.png & .tif copied to {len(OUT_DIRS)} destination directories.")


# =========================================================================
# FIGURE 1: OVERALL MULTIMODAL PREDICTION FRAMEWORK
# =========================================================================
def build_fig1():
    print("Generating Figure 1: Overall Multimodal Architecture Framework...")
    fig, ax = plt.subplots(figsize=(16.8, 9.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 16.8)
    ax.set_ylim(-0.4, 9.8)
    ax.axis('off')

    # Top Modules (3D Beveled Blocks)
    # Module 1 (Teal/Navy - Structural Neuroimaging)
    draw_3d_block(ax, 0.7, 7.3, 4.5, 1.45, depth=0.35, angle=35,
                  face_col='#1F618D', top_col='#5DADE2', side_col='#154360', edge_col='#0B2535', zorder=10)
    ax.add_patch(patches.Circle((1.35, 8.02), 0.38, facecolor='#154360', edgecolor='#5DADE2', lw=1.0, zorder=25))
    draw_brain_vector(ax, 1.35, 8.02, s=0.26, col='#EBF5FB', zorder=30)
    ax.text(2.0, 8.20, "Module 1: STRUCTURAL MRI", ha='left', va='center',
            color='white', fontsize=10.5, fontweight='bold', zorder=30)
    ax.text(2.0, 7.86, "Med3D ResNet-18 + Train-Only PCA-5", ha='left', va='center',
            color='#D4E6F1', fontsize=8.2, zorder=30)
    ax.text(2.0, 7.56, "Balanced Logistic Regression (C=0.1)", ha='left', va='center',
            color='#A9CCE3', fontsize=7.6, zorder=30)

    # Module 2 (Emerald Green - Behavioral Questionnaires)
    draw_3d_block(ax, 6.15, 7.3, 4.5, 1.45, depth=0.35, angle=35,
                  face_col='#239B56', top_col='#58D68D', side_col='#196F3D', edge_col='#0E4B24', zorder=10)
    ax.add_patch(patches.Circle((6.80, 8.02), 0.38, facecolor='#196F3D', edgecolor='#58D68D', lw=1.0, zorder=25))
    draw_clipboard_vector(ax, 6.80, 8.02, s=0.25, col='#FFFFFF', zorder=30)
    ax.text(7.45, 8.20, "Module 2: QUESTIONNAIRES", ha='left', va='center',
            color='white', fontsize=10.5, fontweight='bold', zorder=30)
    ax.text(7.45, 7.86, "9 Transition Markers + PCA-3 Orthogonalization", ha='left', va='center',
            color='#D1F2EB', fontsize=8.2, zorder=30)
    ax.text(7.45, 7.56, "Balanced L2-Logistic Regression (C=0.01)", ha='left', va='center',
            color='#A3E4D7', fontsize=7.6, zorder=30)

    # Module 3 (Warm Rust/Orange - Phenotypic & Demographics)
    draw_3d_block(ax, 11.6, 7.3, 4.5, 1.45, depth=0.35, angle=35,
                  face_col='#D35400', top_col='#F39C12', side_col='#935116', edge_col='#592804', zorder=10)
    ax.add_patch(patches.Circle((12.25, 8.02), 0.38, facecolor='#935116', edgecolor='#F39C12', lw=1.0, zorder=25))
    draw_user_vector(ax, 12.25, 8.02, s=0.25, col='#FEF5E7', zorder=30)
    ax.text(12.90, 8.20, "Module 3: PHENOTYPIC", ha='left', va='center',
            color='white', fontsize=10.5, fontweight='bold', zorder=30)
    ax.text(12.90, 7.86, "4 SES & Pubertal Maturation Predictors", ha='left', va='center',
            color='#FDEBD0', fontsize=8.2, zorder=30)
    ax.text(12.90, 7.56, "Balanced L2-Logistic Regression (C=0.01)", ha='left', va='center',
            color='#FAD7A0', fontsize=7.6, zorder=30)

    # Downward inference flow arrows
    draw_arrow(ax, 2.95, 7.3, 2.95, 6.25, color='#64748B', lw=1.8, zorder=12)
    draw_arrow(ax, 8.40, 7.3, 8.40, 6.25, color='#64748B', lw=1.8, zorder=12)
    draw_arrow(ax, 13.85, 7.3, 13.85, 6.25, color='#64748B', lw=1.8, zorder=12)
    ax.text(2.95, 6.75, "Sample-Independent Scoring", ha='center', va='center',
            color='#475569', fontsize=8.0, fontstyle='italic', zorder=25)
    ax.text(8.40, 6.75, "Sample-Independent Scoring", ha='center', va='center',
            color='#475569', fontsize=8.0, fontstyle='italic', zorder=25)
    ax.text(13.85, 6.75, "Sample-Independent Scoring", ha='center', va='center',
            color='#475569', fontsize=8.0, fontstyle='italic', zorder=25)

    # Middle Row: Unimodal Probability Slabs (Silver/Grey 3D Beveled Bars)
    draw_3d_block(ax, 1.15, 5.5, 3.6, 0.72, depth=0.25, angle=35,
                  face_col='#F1F5F9', top_col='#FFFFFF', side_col='#CBD5E1', edge_col='#94A3B8', zorder=10)
    ax.text(2.95, 5.86, r"$\mathbf{P}_{\mathrm{MRI}}\ (\mathrm{Test\ AUC:\ 0.7221})$", ha='center', va='center',
            color='#0F172A', fontsize=9.8, fontweight='bold', zorder=30)

    draw_3d_block(ax, 6.60, 5.5, 3.6, 0.72, depth=0.25, angle=35,
                  face_col='#F1F5F9', top_col='#FFFFFF', side_col='#CBD5E1', edge_col='#94A3B8', zorder=10)
    ax.text(8.40, 5.86, r"$\mathbf{P}_{\mathrm{Quest}}\ (\mathrm{Test\ AUC:\ 0.6260})$", ha='center', va='center',
            color='#0F172A', fontsize=9.8, fontweight='bold', zorder=30)

    draw_3d_block(ax, 12.05, 5.5, 3.6, 0.72, depth=0.25, angle=35,
                  face_col='#F1F5F9', top_col='#FFFFFF', side_col='#CBD5E1', edge_col='#94A3B8', zorder=10)
    ax.text(13.85, 5.86, r"$\mathbf{P}_{\mathrm{Pheno}}\ (\mathrm{Test\ AUC:\ 0.6870})$", ha='center', va='center',
            color='#0F172A', fontsize=9.8, fontweight='bold', zorder=30)

    # Clean diagonal arrows with prominent w = 1/3 weight badges
    draw_arrow(ax, 2.95, 5.5, 6.25, 4.15, color='#64748B', lw=1.8, badge='w = 1/3', badge_col='#1F618D', zorder=12)
    draw_arrow(ax, 8.40, 5.5, 8.40, 4.15, color='#64748B', lw=1.8, badge='w = 1/3', badge_col='#239B56', zorder=12)
    draw_arrow(ax, 13.85, 5.5, 10.55, 4.15, color='#64748B', lw=1.8, badge='w = 1/3', badge_col='#D35400', zorder=12)

    # Multimodal Decision-Level Fusion 3D Bar (Royal Purple)
    draw_3d_block(ax, 3.4, 2.85, 10.0, 1.25, depth=0.35, angle=35,
                  face_col='#6C3483', top_col='#A569BD', side_col='#4A235A', edge_col='#2C1337', zorder=10)
    ax.text(8.40, 3.70, "Primary Decision-Level Fusion (Zero-Parameter Ensembling)", ha='center', va='center',
            color='white', fontsize=11.6, fontweight='bold', zorder=30)
    ax.text(8.40, 3.32, r"$\mathbf{P}_{\mathrm{rank}} = \frac{1}{3} \mathrm{Rank}(\mathbf{P}_{\mathrm{MRI}}) + \frac{1}{3} \mathrm{Rank}(\mathbf{P}_{\mathrm{Quest}}) + \frac{1}{3} \mathrm{Rank}(\mathbf{P}_{\mathrm{Pheno}})\quad [\mathrm{Test\ AUC:\ 0.8091}]$",
            ha='center', va='center', color='#F5EEF8', fontsize=9.6, fontweight='bold', zorder=30)
    ax.text(8.40, 3.00, r"Dawes' Equal Average: $P_{\mathrm{Dawes}} = \frac{1}{3}(P_{\mathrm{MRI}} + P_{\mathrm{Quest}} + P_{\mathrm{Pheno}})\quad [\mathrm{Test\ AUC:\ 0.7792}]$",
            ha='center', va='center', color='#E8DAEF', fontsize=8.5, fontstyle='italic', zorder=30)

    # Downward arrow to Threshold Calibration
    draw_arrow(ax, 8.40, 2.85, 8.40, 2.15, color='#64748B', lw=1.6, zorder=12)
    ax.text(8.40, 2.50, "Dual Prospective Operating Points", ha='center', va='center',
            color='#64748B', fontsize=8.0, fontstyle='italic', zorder=25)

    # Threshold Optimization 3D Bar (Rich Gold / Amber)
    draw_3d_block(ax, 4.4, 1.45, 8.0, 0.70, depth=0.25, angle=35,
                  face_col='#D4AC0D', top_col='#F4D03F', side_col='#9A7D0A', edge_col='#4D3E05', zorder=10)
    ax.text(8.40, 1.88, r"Screening Mode ($\tau = 0.50$): 100.0% Sensitivity (7/7 Incident Cases Detected, 0 FN)",
            ha='center', va='center', color='white', fontsize=9.8, fontweight='bold', zorder=30)
    ax.text(8.40, 1.60, r"Diagnostic Mode ($\tau = 0.51$, Train-OOF Calibrated): 85.7% Sensitivity (6/7 Detected, Spec: 54.5%, $F_1 = 0.3158$)",
            ha='center', va='center', color='#FEF9E7', fontsize=8.2, zorder=30)

    # Downward arrow to Final Prediction
    draw_arrow(ax, 8.40, 1.45, 8.40, 0.85, color='#64748B', lw=1.6, zorder=12)

    # Final Clinical Decision 3D Bar (Teal / Deep Cyan)
    draw_3d_block(ax, 3.2, -0.20, 10.4, 1.05, depth=0.30, angle=35,
                  face_col='#148F77', top_col='#48C9B0', side_col='#0E6252', edge_col='#083E34', zorder=10)
    ax.text(8.40, 0.58, "Prospective Incident Internalizing Disorder Prediction (Held-Out Test Cohort, n=62)", ha='center', va='center',
            color='white', fontsize=11.2, fontweight='bold', zorder=30)
    ax.text(8.40, 0.28, r"$\mathrm{Multimodal\ Rank\ Averaging:\ ROC\text{-}AUC} = 0.8091\ [95\%\ \mathrm{CI}:\ 0.6597 - 0.9306] \mid \mathrm{Raw}\ p = 0.0020 \mid \mathrm{FDR}\ q = 0.0160$",
            ha='center', va='center', color='white', fontsize=8.8, fontweight='bold', zorder=30)
    ax.text(8.40, 0.02, r"$\mathrm{Screening}\ (\tau=0.50):\ \mathrm{Sens} = 100.0\%\ (7/7) \mid \mathrm{Spec} = 52.7\%\ (29/55) \mid \mathrm{NPV} = 100.0\% \mid \mathrm{Dawes\ AUC} = 0.7792$",
            ha='center', va='center', color='#E8F8F5', fontsize=8.2, zorder=30)

    save_figure_to_all_locations(fig, "Fig1_Overall_Framework")
    plt.close(fig)
    print("  [OK] Fig1_Overall_Framework successfully generated.")


# =========================================================================
# FIGURE 2: MODULE 1 — STRUCTURAL NEUROIMAGING PIPELINE
# =========================================================================
def build_fig2():
    print("Generating Figure 2: Module 1 Structural MRI Architecture (Scientific 3D MRI Volume)...")
    fig, ax = plt.subplots(figsize=(16.8, 9.0), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0.2, 9.2)
    ax.axis('off')

    # 1. INPUT RAW SCANS (Stacked dark cards on Left with Genuine Scientific 3D MRI Volume)
    draw_3d_block(ax, 0.35, 4.3, 2.3, 3.1, depth=0.25, angle=35,
                  face_col='#1E293B', top_col='#334155', side_col='#0F172A', edge_col='#020617', zorder=10)
    ax.text(1.50, 7.15, "Input Raw Scans\nT1-weighted MRI\n(91 x 109 x 91)", ha='center', va='center',
            color='white', fontsize=8.8, fontweight='bold', zorder=30)

    # Genuine scientific 3D volumetric MRI rendering (zero emoji/alien eyes!)
    draw_scientific_mri_volume(ax, cx=1.50, cy=5.60, scale=0.92, zorder=25)

    ax.text(1.50, 4.45, "N = 304 Subjects (152 Pairs)\nMNI152 2mm Standard Space", ha='center', va='center',
            color='#94A3B8', fontsize=7.2, zorder=30)

    draw_arrow(ax, 2.85, 5.80, 3.25, 5.80, color='#475569', lw=1.8, zorder=15)

    # 2. PREPROCESSING PIPELINE (Slate Blue Panel)
    panel_prep = patches.FancyBboxPatch((3.25, 0.8), 3.4, 8.0, boxstyle="round,pad=0.15,rounding_size=0.2",
                                        facecolor='#344955', edgecolor='#232F34', lw=1.4, zorder=5)
    ax.add_patch(panel_prep)
    ax.text(4.95, 8.32, "Standardized MRI Preprocessing\n(FSL / ANTs Equivalent)", ha='center', va='center',
            color='white', fontsize=9.8, fontweight='bold', zorder=25)

    prep_steps = [
        ("Skull Stripping (Otsu/BET)", 7.0, "skull"),
        ("N4 Bias Field Correction", 5.5, "pulse"),
        ("MNI152 Non-linear Registration", 4.0, "brain"),
        ("Intensity Z-score Normalization", 2.5, "chart")
    ]
    for title, py, itype in prep_steps:
        pill = patches.FancyBboxPatch((3.40, py - 0.52), 3.1, 1.05, boxstyle="round,pad=0.1,rounding_size=0.15",
                                      facecolor='#ECEFF1', edgecolor='#CFD8DC', lw=1.0, zorder=7)
        ax.add_patch(pill)
        if itype == 'skull':
            # Clean anatomical skull stripping icon: dashed outer skull line + solid brain circle
            ax.add_patch(patches.Circle((3.75, py), 0.22, facecolor='none', edgecolor='#EF4444', lw=1.3, linestyle='--', zorder=15))
            ax.add_patch(patches.Circle((3.75, py), 0.15, facecolor='#0284C7', edgecolor='#0369A1', lw=1.0, zorder=16))
            ax.plot([3.75, 3.75], [py - 0.11, py + 0.11], color='white', lw=1.1, zorder=17)
        elif itype == 'pulse':
            ax.plot([3.55, 3.65, 3.71, 3.78, 3.85, 3.95], [py, py, py+0.24, py-0.24, py+0.12, py],
                    color='#0284C7', lw=1.8, zorder=15)
        elif itype == 'brain':
            ax.add_patch(patches.Circle((3.75, py), 0.20, facecolor='none', edgecolor='#0D9488', lw=1.6, zorder=15))
            ax.plot([3.75, 3.75], [py-0.18, py+0.18], color='#0D9488', lw=1.2, zorder=16)
        elif itype == 'chart':
            ax.add_patch(patches.Rectangle((3.61, py-0.18), 0.08, 0.26, facecolor='#64748B', zorder=15))
            ax.add_patch(patches.Rectangle((3.72, py-0.18), 0.08, 0.38, facecolor='#64748B', zorder=15))
            ax.add_patch(patches.Rectangle((3.83, py-0.18), 0.08, 0.18, facecolor='#64748B', zorder=15))

        ax.text(4.18, py, title, ha='left', va='center', color='#1E293B', fontsize=8.0, fontweight='bold', zorder=25)

    draw_arrow(ax, 6.80, 5.80, 7.15, 5.80, color='#475569', lw=1.8, zorder=15)

    # 3. MED3D RESNET-18 BACKBONE (Large Center Panel)
    panel_cnn = patches.FancyBboxPatch((7.15, 0.8), 5.65, 8.0, boxstyle="round,pad=0.15,rounding_size=0.2",
                                       facecolor='#344955', edgecolor='#232F34', lw=1.4, zorder=5)
    ax.add_patch(panel_cnn)
    ax.text(9.98, 8.32, "Med3D 3D-ResNet-18 Backbone\n(Pretrained Volumetric Embeddings)", ha='center', va='center',
            color='white', fontsize=10.2, fontweight='bold', zorder=25)

    # Upper inner box for feature map tensor slabs
    inner_box = patches.FancyBboxPatch((7.35, 4.4), 5.25, 3.3, boxstyle="round,pad=0.1,rounding_size=0.15",
                                       facecolor='#ECEFF1', edgecolor='#CFD8DC', lw=1.0, zorder=6)
    ax.add_patch(inner_box)

    # 3D Feature Map Slabs
    slabs = [
        (7.55, 4.90, 0.38, 2.1, '#3B82F6', '#93C5FD', '#1D4ED8', '#1E3A8A', 'Conv1\n64ch'),
        (8.25, 5.05, 0.42, 1.7, '#06B6D4', '#67E8F9', '#0E7490', '#155E75', 'Res2\n128ch'),
        (9.00, 5.20, 0.46, 1.4, '#6366F1', '#A5B4FC', '#4338CA', '#312E81', 'Res3\n256ch'),
        (9.80, 5.35, 0.50, 1.1, '#8B5CF6', '#C4B5FD', '#6D28D9', '#4C1D95', 'Res4\n512ch')
    ]
    for sx, sy, sw, sh, fc, tc, sc, ec, tag in slabs:
        draw_3d_block(ax, sx, sy, sw, sh, depth=0.26, angle=35,
                      face_col=fc, top_col=tc, side_col=sc, edge_col=ec, zorder=10)
        ax.text(sx + sw/2, sy - 0.26, tag, ha='center', va='top', color='#334155', fontsize=6.8, fontweight='bold', zorder=25)

    # Arrow from Res4 to Global AvgPool
    draw_arrow(ax, 10.45, 5.9, 10.75, 5.9, color='#64748B', lw=1.4, zorder=15)

    # Global AvgPool pill
    draw_pill(ax, 10.80, 5.35, 0.90, 1.10, text="Global\nAvgPool\n& Flatten",
              face_col='#FFFFFF', edge_col='#0284C7', text_col='#0369A1', font_size=7.0, font_weight='bold')

    draw_arrow(ax, 11.73, 5.9, 12.00, 5.9, color='#64748B', lw=1.4, zorder=15)

    # 768-D segmented bar
    draw_3d_block(ax, 12.05, 4.85, 0.28, 2.2, depth=0.20, angle=35,
                  face_col='#334155', top_col='#64748B', side_col='#0F172A', edge_col='#020617', zorder=10)
    for cy in np.linspace(5.0, 6.8, 8):
        ax.plot([12.05, 12.33], [cy, cy], color='#94A3B8', lw=0.8, zorder=15)
    ax.text(12.05, 4.55, "768-D Vector", ha='center', va='top', color='#0F172A', fontsize=7.2, fontweight='bold', zorder=25)

    # Downward flow to PCA feature selection with badge
    draw_arrow(ax, 9.98, 4.4, 9.98, 3.55, color='#0284C7', lw=1.8, badge='Train-Only Fit', badge_col='#0284C7', zorder=15)

    # Lower inner box: Train-Only PCA
    fs_box = patches.FancyBboxPatch((7.35, 1.15), 5.25, 2.3, boxstyle="round,pad=0.1,rounding_size=0.15",
                                    facecolor='#ECEFF1', edgecolor='#CFD8DC', lw=1.0, zorder=6)
    ax.add_patch(fs_box)
    
    # Funnel Graphic on left
    funnel_pts = np.array([[7.50, 3.05], [8.10, 3.05], [7.88, 2.30], [7.72, 2.30]])
    funnel = patches.Polygon(funnel_pts, closed=True, facecolor='#E0F2FE', edgecolor='#0284C7', lw=1.8, zorder=15)
    ax.add_patch(funnel)
    ax.plot([7.80, 7.80], [2.30, 1.80], color='#0284C7', lw=2.0, zorder=15)
    ax.text(7.80, 1.55, "k = 5", ha='center', va='center', color='#0284C7', fontsize=8.0, fontweight='bold', zorder=25)

    # Explanatory text strictly on the right
    ax.text(8.30, 2.95, "Strict Train-Only PCA Reduction", ha='left', va='center',
            color='#0F172A', fontsize=9.2, fontweight='bold', zorder=25)
    ax.text(8.30, 2.50, r"- $\mathrm{StandardScaler}$ & $\mathrm{PCA}$ fit strictly on Train ($n=180$)" + "\n" + r"- 5 Orthogonal Components explaining 63.2% Variance",
            ha='left', va='center', color='#334155', fontsize=7.8, zorder=25)
    ax.text(8.30, 1.85, r"$\mathbf{z}_{\mathrm{mri}} \in \mathbb{R}^5$ (Zero Validation / Pool Contamination)", ha='left', va='center',
            color='#0369A1', fontsize=7.8, fontweight='bold', zorder=25)

    draw_arrow(ax, 12.98, 4.8, 13.35, 4.8, color='#475569', lw=1.8, zorder=15)

    # 4. SUPERVISED CLASSIFIER PANEL (Right)
    panel_clf = patches.FancyBboxPatch((13.35, 0.8), 3.2, 8.0, boxstyle="round,pad=0.15,rounding_size=0.2",
                                       facecolor='#344955', edgecolor='#232F34', lw=1.4, zorder=5)
    ax.add_patch(panel_clf)
    ax.text(14.95, 8.32, "Cost-Sensitive Metric Head\n(Zero Data Leakage)", ha='center', va='center',
            color='white', fontsize=10.2, fontweight='bold', zorder=25)

    # Model pill
    draw_pill(ax, 13.55, 6.00, 2.8, 1.5,
              text=r"$L_2$-Regularized" + "\nLogistic Regression\n" + r"$(C = 0.1, \mathrm{balanced})$",
              face_col='#ECEFF1', edge_col='#CFD8DC', text_col='#0F172A', font_size=8.0, font_weight='bold')
    
    draw_arrow(ax, 14.95, 6.00, 14.95, 5.10, color='#0284C7', lw=1.6, zorder=15)

    # Probability output card
    draw_pill(ax, 13.55, 4.00, 2.8, 1.1, text=r"Predicted Risk:" + "\n" + r"$\mathbf{P}_{\mathrm{MRI}} \in [0, 1]$",
              face_col='#E0F2FE', edge_col='#0284C7', text_col='#0369A1', font_size=9.0, font_weight='bold')

    draw_arrow(ax, 14.95, 4.00, 14.95, 3.05, color='#0284C7', lw=1.6, zorder=15)

    # Metrics dark pill (Corrected: Spec = 61.8% (34/55 controls correctly identified))
    draw_pill(ax, 13.55, 1.15, 2.8, 1.8,
              text="Held-Out Test Performance:\n" + r"$\mathrm{Test\ ROC\text{-}AUC} = 0.7221$" + "\n" + r"$\mathrm{PR\text{-}AUC} = 0.1923$" + "\n" + r"$\mathrm{Sensitivity} = 71.4\%\ (5/7)$" + "\n" + r"$\mathrm{Specificity} = 61.8\%\ (34/55)$" + "\n" + r"$\mathrm{NPV} = 94.4\% \mid \mathrm{Brier} = 0.2266$",
              face_col='#0F172A', edge_col='#334155', text_col='#38BDF8', font_size=7.6, font_weight='bold')

    save_figure_to_all_locations(fig, "Fig2_MRI_Preprocessing")
    plt.close(fig)
    print("  [OK] Fig2_MRI_Preprocessing successfully generated.")


# =========================================================================
# FIGURE 3: MODULE 2 — BEHAVIORAL QUESTIONNAIRES PIPELINE
# =========================================================================
def build_fig3():
    print("Generating Figure 3: Module 2 Questionnaire Pipeline (Logically & Mathematically Aligned)...")
    fig, ax = plt.subplots(figsize=(17.2, 9.4), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0.2, 9.4)
    ax.axis('off')

    # Top Title
    ax.text(8.60, 9.05, "Module 2: Longitudinal Anxiety Symptomatology Pipeline (Regularized & Few-Shot Framework)",
            ha='center', va='center', color='#0F172A', fontsize=13.5, fontweight='bold', zorder=30)
    ax.text(8.60, 8.72, r"Leakage-Free Preprocessing $\to$ 9 Transition Markers $\to$ PCA-3 Domain Orthogonalization / FT-ProtoNet Metric Space",
            ha='center', va='center', color='#475569', fontsize=9.0, fontstyle='italic', zorder=30)

    # -------------------------------------------------------------------------
    # ROW 1 (FORWARD INFERENCE PIPELINE: y in [3.30, 8.35])
    # -------------------------------------------------------------------------
    
    # 1. INPUT: 3D TRANSLUCENT GLASS ISOMETRIC BOX (Left)
    draw_translucent_cube(ax, 0.40, 3.80, 1.85, 3.80, depth=0.65, angle=32,
                          face_col='#E0F2FE', edge_col='#0284C7', alpha=0.32, lw=1.5, zorder=10)
    draw_clipboard_vector(ax, 1.35, 5.80, s=0.45, col='#0284C7', zorder=20)
    ax.text(1.35, 3.35, "Raw Assessments", ha='center', va='center',
            color='#0F172A', fontsize=10.5, fontweight='bold', zorder=30)
    ax.text(1.35, 2.95, r"109 Items $\in$ SCAS, SPHERE," + "\nSDQ, MFQ", ha='center', va='center',
            color='#64748B', fontsize=7.8, zorder=30)

    # Thick cyan arrow to Preprocessing
    draw_arrow(ax, 2.70, 5.70, 3.20, 5.70, color='#00A3C4', lw=2.2, zorder=15)

    # 2. PREPROCESSING & IMPUTATION (Rounded 3D Card)
    box_prep = patches.FancyBboxPatch((3.20, 3.50), 2.30, 4.40, boxstyle="round,pad=0.1,rounding_size=0.18",
                                      facecolor='#FFFFFF', edgecolor='#00A3C4', lw=1.5, zorder=12)
    ax.add_patch(box_prep)
    ax.text(4.35, 7.55, "Leakage-Free\nPreprocessing", ha='center', va='center',
            color='#0F172A', fontsize=10.0, fontweight='bold', zorder=25)
    ax.text(4.35, 6.70, r"$\mathbf{x}_{\mathrm{norm}} = \frac{\mathbf{x} - \hat{\mu}_{\mathrm{tr}}}{\hat{\sigma}_{\mathrm{tr}}}$",
            ha='center', va='center', color='#0284C7', fontsize=9.5, fontweight='bold', zorder=25)
    ax.text(4.35, 5.30, "- In-fold median impute\n- Train-only StandardScaler\n- Purged sum-score totals\n  (Zero proxy leakage)\n- Preserves raw variances\n  without cohort pooling",
            ha='center', va='center', color='#334155', fontsize=7.4, zorder=25)
    ax.text(4.35, 3.85, r"$\mathbf{x}_{\mathrm{norm}} \in \mathbb{R}^{109}$",
            ha='center', va='center', color='#00A3C4', fontsize=8.0, fontweight='bold', zorder=25)

    # Thick cyan arrow to Transition Feature Selection
    draw_arrow(ax, 5.70, 5.70, 6.20, 5.70, color='#00A3C4', lw=2.2, zorder=15)

    # 3. CLINICAL TRANSITION FEATURE SELECTION (Rounded 3D Card - Breaks Polarity Inversion)
    box_trans = patches.FancyBboxPatch((6.20, 3.50), 2.75, 4.40, boxstyle="round,pad=0.1,rounding_size=0.18",
                                       facecolor='#F0FDF4', edgecolor='#16A34A', lw=1.6, zorder=12)
    ax.add_patch(box_trans)
    ax.text(7.57, 7.55, "9 Transition Markers\n(Resolves Inversion)", ha='center', va='center',
            color='#15803D', fontsize=10.0, fontweight='bold', zorder=25)
    ax.text(7.57, 6.65, r"- Somatic Tension:" + "\n" + r"  $\mathrm{pSDQ03}, \mathrm{SPH07,08,22}$",
            ha='center', va='center', color='#1E293B', fontsize=7.4, zorder=25)
    ax.text(7.57, 5.80, r"- Social Avoidance:" + "\n" + r"  $\mathrm{pSCAS\_socpho}, \mathrm{pSCAS31}$",
            ha='center', va='center', color='#1E293B', fontsize=7.4, zorder=25)
    ax.text(7.57, 4.90, r"- Informant Discrepancy:" + "\n" + r"  $\Delta_{26} = \mathrm{pSCAS26} - \mathrm{SCAS26}$" + "\n" + r"  $\Delta_{\mathrm{worry}} = \mathrm{pSCAS31} - \mathrm{SCAS31}$",
            ha='center', va='center', color='#1E293B', fontsize=7.3, zorder=25)
    ax.text(7.57, 3.85, r"$\mathbf{x} \in \mathbb{R}^9\ (\mathrm{True\ Pos\ Orientation})$",
            ha='center', va='center', color='#15803D', fontsize=7.8, fontweight='bold', zorder=25)

    # Thick arrow into PCA-3 / FT-ProtoNet Dual Engine
    draw_arrow(ax, 9.15, 5.70, 9.65, 5.70, color='#16A34A', lw=2.4, zorder=15)

    # 4. PRIMARY CLINICAL MODEL & DEEP BASELINE (Large 3D Box)
    draw_3d_block(ax, 9.65, 3.50, 3.85, 4.40, depth=0.35, angle=35,
                  face_col='#FFFFFF', top_col='#F8FAFC', side_col='#E2E8F0', edge_col='#475569', lw=1.5, zorder=10)
    ax.text(11.57, 7.55, "Dual Modeling Framework", ha='center', va='center',
            color='#0F172A', fontsize=10.5, fontweight='bold', zorder=25)

    # Stage 1: Primary Model: PCA-3 Factor Orthogonalization
    draw_3d_block(ax, 9.85, 6.20, 3.45, 0.95, depth=0.16, angle=35,
                  face_col='#E0F2FE', top_col='#F0F9FF', side_col='#BAE6FD', edge_col='#0284C7', lw=1.0, zorder=15)
    ax.text(11.57, 6.75, "Primary Model: PCA-3 Factor Reduction", ha='center', va='center',
            color='#0369A1', fontsize=8.0, fontweight='bold', zorder=25)
    ax.text(11.57, 6.40, r"$\mathbf{z} = \mathrm{PCA}_3(\mathbf{x}) \in \mathbb{R}^3\ (61.4\%\ \mathrm{Variance})$",
            ha='center', va='center', color='#0284C7', fontsize=7.4, fontweight='bold', zorder=25)

    draw_arrow(ax, 11.57, 6.20, 11.57, 5.65, color='#0284C7', lw=1.4, zorder=20)

    # Stage 2: Regularized L2 Logistic Regression
    draw_3d_block(ax, 9.85, 4.70, 3.45, 0.95, depth=0.16, angle=35,
                  face_col='#F0FDF4', top_col='#F9FEF5', side_col='#DCFCE7', edge_col='#16A34A', lw=1.0, zorder=15)
    ax.text(11.57, 5.25, r"$L_2\text{-}\mathrm{Regularized\ Logistic\ Regression}$", ha='center', va='center',
            color='#15803D', fontsize=8.0, fontweight='bold', zorder=25)
    ax.text(11.57, 4.90, r"$C = 0.01\ (\mathrm{Strict\ Train\text{-}CV\ Tuned}),\ \mathrm{balanced}$",
            ha='center', va='center', color='#166534', fontsize=7.4, zorder=25)

    draw_arrow(ax, 11.57, 4.70, 11.57, 4.20, color='#16A34A', lw=1.4, zorder=20)

    # Stage 3: Baseline Deep FT-ProtoNet Specification
    draw_3d_block(ax, 9.85, 3.60, 3.45, 0.60, depth=0.16, angle=35,
                  face_col='#FEF9E7', top_col='#FFFDF7', side_col='#FEF3C7', edge_col='#D97706', lw=1.0, zorder=15)
    ax.text(11.57, 3.90, "Deep Baseline: FT-ProtoNet (16D Embeddings)", ha='center', va='center',
            color='#B45309', fontsize=7.3, fontweight='bold', zorder=25)

    # Thick arrow to Latent Metric Space
    draw_arrow(ax, 13.75, 5.70, 14.15, 5.70, color='#7C3AED', lw=2.4, zorder=25)

    # 5. TEST BENCHMARK & METRIC PERFORMANCE (Box on Right)
    box_metric = patches.FancyBboxPatch((14.15, 3.50), 2.85, 4.40, boxstyle="round,pad=0.1,rounding_size=0.18",
                                        facecolor='#FAF5FF', edgecolor='#9333EA', lw=1.6, zorder=12)
    ax.add_patch(box_metric)
    ax.text(15.57, 7.55, "Held-Out Test Results", ha='center', va='center',
            color='#7E22CE', fontsize=10.0, fontweight='bold', zorder=25)

    # Performance pill: Primary Model
    draw_pill(ax, 14.30, 5.70, 2.55, 1.45,
              text="Primary Regularized Model:\n" +
                   r"$\mathrm{Test\ ROC\text{-}AUC} = 0.6260$" + "\n" +
                   r"$\mathrm{Val\ AUC} = 0.6571\ (\Delta = +0.031)$" + "\n" +
                   r"$\mathrm{Sensitivity} = 71.4\%\ (5/7)$" + "\n" +
                   r"$\mathrm{Specificity} = 67.3\%\ (37/55)$",
              face_col='#FFFFFF', edge_col='#16A34A', text_col='#15803D', font_size=7.2, font_weight='bold')

    # Performance pill: Baseline FT-ProtoNet
    draw_pill(ax, 14.30, 4.30, 2.55, 1.25,
              text="FT-ProtoNet Deep Baseline:\n" +
                   r"$\mathrm{Val\ AUC} = 0.8052$" + "\n" +
                   r"$\mathrm{Test\ AUC} = 0.5429$" + "\n" +
                   r"$\mathrm{Overfit\ Gap} = +0.2623$",
              face_col='#FEF2F2', edge_col='#EF4444', text_col='#B91C1C', font_size=7.2, font_weight='bold')

    # Output prediction pill
    draw_pill(ax, 14.30, 3.65, 2.55, 0.50, text=r"$\mathbf{P}_{\mathrm{Quest}} \in [0, 1]\ (\mathrm{Test\ AUC:\ 0.6260})$",
              face_col='#FAF5FF', edge_col='#7C3AED', text_col='#581C87', font_size=7.6, font_weight='bold')

    # -------------------------------------------------------------------------
    # ROW 2: METHODOLOGICAL AUDIT & OVERFITTING PREVENTION (y in [0.60, 2.70])
    # -------------------------------------------------------------------------
    
    # Downward arrow from Right Box to Audit Cards
    ax.plot([15.57, 15.57], [3.50, 1.55], color='#7C3AED', lw=1.8, zorder=20)
    draw_arrow(ax, 15.57, 1.55, 14.05, 1.55, color='#7C3AED', lw=1.8, badge=r'Audit', badge_col='#7C3AED', zorder=20)

    # Card 3: Overfitting Elimination
    draw_pill(ax, 9.65, 0.65, 4.40, 1.80,
              text="Finite-Sample Regularization Principle:\n" +
                   r"- PCA-3 domain reduction eliminates collinearity" + "\n" +
                   r"- Train-CV tuned $L_2$ ($C=0.01$) penalizes over-weighting" + "\n" +
                   r"- Recovers Test AUC from 0.4961 $\to$ 0.6260" + "\n" +
                   r"- Eliminates deep net memorization gap (0.262 $\to$ 0.031)",
              face_col='#FAF5FF', edge_col='#9333EA', text_col='#581C87', font_size=7.3, font_weight='bold')

    # Arrow from Card 3 to Card 2
    draw_arrow(ax, 9.65, 1.55, 9.25, 1.55, color='#16A34A', lw=1.6, zorder=20)

    # Card 2: Clinical Construct Validity
    draw_pill(ax, 4.85, 0.65, 4.40, 1.80,
              text="Construct Validity of 9 Transition Markers:\n" +
                   "- Somatization (SPHERE) reflects vegetative autonomic tension\n" +
                   "- Social avoidance (SCAS) indexes behavioral inhibition\n" +
                   "- Informant discrepancy (pSCAS - SCAS) captures covert distress\n" +
                   "- All 9 markers selected a priori from clinical literature",
              face_col='#F0FDF4', edge_col='#16A34A', text_col='#15803D', font_size=7.2, font_weight='bold')

    # Arrow from Card 2 to Card 1
    draw_arrow(ax, 4.85, 1.55, 4.55, 1.55, color='#0284C7', lw=1.6, zorder=20)

    # Card 1: Longitudinal Inversion Audit
    draw_pill(ax, 0.40, 0.65, 4.15, 1.80,
              text="Longitudinal Orientation Audit:\n" +
                   r"$\Delta_{26} = \mathrm{pSCAS26} - \mathrm{SCAS26} > 0$" + "\n" +
                   r"$\Delta_{\mathrm{worry}} = \mathrm{pSCAS31} - \mathrm{SCAS31} > 0$" + "\n" +
                   "True prospective signal; zero post-hoc inversion\n" +
                   "Verified identical distributions across Train/Val/Test",
              face_col='#F8FAFC', edge_col='#0284C7', text_col='#0369A1', font_size=7.3, font_weight='bold')

    save_figure_to_all_locations(fig, "Fig3_Questionnaire_Pipeline")
    plt.close(fig)
    print("  [OK] Fig3_Questionnaire_Pipeline successfully generated.")


# =========================================================================
# FIGURE 4: DETAILED DEEP PROTONET ARCHITECTURE
# =========================================================================
def build_fig4():
    print("Generating Figure 4: Detailed Deep Feature-Tokenizer ProtoNet Architecture (Full Mathematical Typography)...")
    fig, ax = plt.subplots(figsize=(18.4, 9.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 18.4)
    ax.set_ylim(0.2, 10.0)
    ax.axis('off')

    # Main Title (Centered at Top)
    ax.text(9.20, 9.60, "Deep Feature-Tokenizer Prototypical Network (FT-ProtoNet) Micro-Architecture",
            ha='center', va='center', color='#0F172A', fontsize=14.0, fontweight='bold', zorder=30)
    ax.text(9.20, 9.28, r"Intra-Sample Pre-LN Multi-Head Self-Attention $\to$ Query Attention Pooling $\to$ Episodic Metric Prototypes ($\mathbb{S}^{15}$)",
            ha='center', va='center', color='#475569', fontsize=9.2, fontstyle='italic', zorder=30)

    # -------------------------------------------------------------------------
    # COLUMN 1: RAW INPUT VECTOR (3D Silver Pillar)
    # -------------------------------------------------------------------------
    draw_3d_block(ax, 0.40, 2.00, 0.85, 6.20, depth=0.25, angle=35,
                  face_col='#E2E8F0', top_col='#F8FAFC', side_col='#CBD5E1', edge_col='#94A3B8', zorder=10)
    ax.text(0.82, 5.10, r"$\mathbf{x} = [x_1, \dots, x_D]^T \in \mathbb{R}^D$", ha='center', va='center',
            color='#0F172A', fontsize=9.5, fontweight='bold', rotation=90, zorder=30)
    ax.text(0.82, 1.55, r"$D=9\ (\mathrm{M2})$" + "\n" + r"$D=4\ (\mathrm{M3})$", ha='center', va='center',
            color='#475569', fontsize=8.0, zorder=30)

    draw_arrow(ax, 1.45, 5.10, 1.95, 5.10, color='#0F172A', lw=1.8, zorder=15)

    # -------------------------------------------------------------------------
    # COLUMN 2: FEATURE TOKENIZER (3D Blue Column)
    # -------------------------------------------------------------------------
    panel_tok = patches.FancyBboxPatch((1.95, 0.80), 2.75, 8.20, boxstyle="round,pad=0.15,rounding_size=0.2",
                                       facecolor='#F0F9FF', edgecolor='#0284C7', lw=1.4, zorder=5)
    ax.add_patch(panel_tok)
    ax.text(3.32, 8.65, "Feature Tokenizer", ha='center', va='center',
            color='#0369A1', fontsize=10.5, fontweight='bold', zorder=25)
    ax.text(3.32, 8.28, r"$\mathbf{t}_i = x_i \cdot \mathbf{w}_i + \mathbf{b}_i + \mathbf{e}_i \in \mathbb{R}^{16}$", ha='center', va='center',
            color='#0284C7', fontsize=8.5, fontweight='bold', zorder=25)

    # 4 Token Slabs
    token_labels = [
        r"$\mathbf{t}_1 \in \mathbb{R}^{16}\ (\mathrm{Somatic})$",
        r"$\mathbf{t}_2 \in \mathbb{R}^{16}\ (\mathrm{Social})$",
        r"$\dots\ (\mathrm{Discrepancy})$",
        r"$\mathbf{t}_D \in \mathbb{R}^{16}\ (\mathrm{Item\ D})$"
    ]
    for idx, t_lbl in enumerate(token_labels):
        ty = 7.00 - idx * 1.35
        draw_3d_block(ax, 2.15, ty, 2.15, 0.70, depth=0.18, angle=35,
                      face_col='#FFFFFF', top_col='#F0F9FF', side_col='#BAE6FD', edge_col='#0284C7', zorder=10)
        ax.text(3.22, ty + 0.35, t_lbl, ha='center', va='center', color='#0369A1', fontsize=7.6, fontweight='bold', zorder=25)

    # Grouping bracket on right edge of token slabs
    ax.plot([4.35, 4.55], [7.35, 7.35], color='#0284C7', lw=1.4, zorder=15)
    ax.plot([4.55, 4.55], [2.95, 7.35], color='#0284C7', lw=1.4, zorder=15)
    ax.plot([4.35, 4.55], [2.95, 2.95], color='#0284C7', lw=1.4, zorder=15)

    ax.text(3.32, 1.45, r"$\mathbf{T} = [\mathbf{t}_1, \dots, \mathbf{t}_D]^T$" + "\n" + r"$\mathbf{T} \in \mathbb{R}^{D \times 16}$ (Token Matrix)", ha='center', va='center',
            color='#0369A1', fontsize=7.8, fontweight='bold', zorder=25)

    # Clean flow from Tokenizer into Transformer at the TOP of Column 3
    ax.plot([4.55, 5.05, 5.05, 5.65], [5.15, 5.15, 7.65, 7.65], color='#0284C7', lw=1.8, zorder=15)
    draw_arrow(ax, 5.45, 7.65, 5.75, 7.65, color='#0284C7', lw=1.8, zorder=15)
    bbox_tok_exit = dict(boxstyle="round,pad=0.25,rounding_size=0.18", facecolor='#0284C7', edgecolor='none')
    ax.text(5.35, 7.95, r"$\mathbf{T} \in \mathbb{R}^{D \times 16}$",
            ha='center', va='center', color='white', fontsize=7.6, fontweight='bold', bbox=bbox_tok_exit, zorder=25)

    # -------------------------------------------------------------------------
    # COLUMN 3: INTRA-SAMPLE MULTI-HEAD SELF-ATTENTION TRANSFORMER (Center Panel)
    # -------------------------------------------------------------------------
    panel_sa = patches.FancyBboxPatch((5.40, 0.80), 4.90, 8.20, boxstyle="round,pad=0.15,rounding_size=0.2",
                                      facecolor='#F0FDF4', edgecolor='#16A34A', lw=1.4, zorder=5)
    ax.add_patch(panel_sa)
    ax.text(7.85, 8.65, "Pre-LN Transformer Block", ha='center', va='center',
            color='#15803D', fontsize=10.5, fontweight='bold', zorder=25)
    ax.text(7.85, 8.30, r"$\mathrm{Intra\text{-}Sample\ Multi\text{-}Head\ Attention\ across\ D\ Features}$", ha='center', va='center',
            color='#166534', fontsize=7.8, fontstyle='italic', zorder=25)
    
    # Branch Point 1: Input split at entrance of bypass lane
    draw_branch_dot(ax, 5.75, 7.65, r=0.065, col='#14532D', zorder=25)

    # Main Branch: Extends from Branch Point 1 across to Centerline and drops into LayerNorm 1
    ax.plot([5.75, 7.85], [7.65, 7.65], color='#16A34A', lw=1.8, zorder=15)
    draw_arrow(ax, 7.85, 7.65, 7.85, 6.95, color='#16A34A', lw=1.5, zorder=15)

    # LayerNorm 1 Pill
    draw_pill(ax, 6.70, 6.95, 2.30, 0.45, text=r"$\tilde{\mathbf{T}} = \mathrm{LayerNorm}(\mathbf{T}) \in \mathbb{R}^{D \times 16}$",
              face_col='#FFFFFF', edge_col='#16A34A', text_col='#166534', font_size=7.4, font_weight='normal')

    # Multi-Head Self-Attention Pill
    draw_pill(ax, 6.05, 5.25, 3.60, 1.35,
              text="Multi-Head Attention (H = 4, d_k = 4):\n" +
                   r"$\mathbf{Q}, \mathbf{K}, \mathbf{V} = \tilde{\mathbf{T}} \mathbf{W}_{Q,K,V} \in \mathbb{R}^{D \times 16}$" + "\n" +
                   r"$\mathbf{A} = \mathrm{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V} \mathbf{W}_{\mathrm{proj}}$",
              face_col='#FFFFFF', edge_col='#16A34A', text_col='#166534', font_size=7.3, font_weight='normal')
    draw_arrow(ax, 7.85, 6.95, 7.85, 6.60, color='#16A34A', lw=1.5, zorder=15)

    # Arrow from MHA into Addition Node 1
    draw_arrow(ax, 7.85, 5.25, 7.85, 4.95, color='#16A34A', lw=1.5, zorder=15)

    # Residual Skip Connection 1
    ax.plot([5.75, 5.75], [7.65, 4.75], color='#059669', lw=1.8, zorder=20)
    ax.annotate('', xy=(7.62, 4.75), xytext=(5.75, 4.75),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color='#059669', mutation_scale=12), zorder=20)
    ax.text(5.62, 6.15, r"$\mathrm{Residual\ Skip\ 1}$", ha='center', va='center',
            color='#047857', fontsize=7.0, fontweight='bold', rotation=90, zorder=25)
    ax.text(6.65, 4.95, r"$+\mathbf{T}$", ha='center', va='bottom', color='#047857', fontsize=7.6, fontweight='bold', zorder=25)

    # Addition Node 1 (+)
    draw_addition_circle(ax, 7.85, 4.75, r=0.20, face_col='#DCFCE7', edge_col='#16A34A', text_col='#14532D', zorder=25)
    ax.text(8.20, 4.75, r"$\mathbf{H}' = \mathbf{T} + \mathbf{A} \in \mathbb{R}^{D \times 16}$", ha='left', va='center',
            color='#14532D', fontsize=7.8, fontweight='bold', zorder=25)

    # Intermediate line to Sub-layer 2
    ax.plot([7.85, 7.85], [4.55, 4.25], color='#16A34A', lw=1.6, zorder=15)

    # Branch Point 2 (Solid Dot)
    draw_branch_dot(ax, 7.85, 4.25, r=0.065, col='#14532D', zorder=25)

    # LayerNorm 2 Pill
    draw_pill(ax, 6.70, 3.65, 2.30, 0.45, text=r"$\tilde{\mathbf{H}}' = \mathrm{LayerNorm}(\mathbf{H}') \in \mathbb{R}^{D \times 16}$",
              face_col='#FFFFFF', edge_col='#16A34A', text_col='#166534', font_size=7.4, font_weight='normal')
    draw_arrow(ax, 7.85, 4.25, 7.85, 3.65, color='#16A34A', lw=1.5, zorder=15)

    # Feed-Forward Network Pill
    draw_pill(ax, 6.05, 1.95, 3.60, 1.35,
              text="Position-Wise FFN (d_ff = 64):\n" +
                   r"$\mathrm{FFN}(\tilde{\mathbf{H}}') = \mathbf{W}_2 \cdot \mathrm{GELU}(\mathbf{W}_1 \tilde{\mathbf{H}}' + \mathbf{b}_1) + \mathbf{b}_2$" + "\n" +
                   r"$(\mathrm{Dropout} = 0.15,\ \mathrm{Linear}\ 16 \to 64 \to 16)$",
              face_col='#FFFFFF', edge_col='#16A34A', text_col='#166534', font_size=7.2, font_weight='normal')
    draw_arrow(ax, 7.85, 3.65, 7.85, 3.30, color='#16A34A', lw=1.5, zorder=15)

    # Arrow from FFN into Addition Node 2
    draw_arrow(ax, 7.85, 1.95, 7.85, 1.65, color='#16A34A', lw=1.5, zorder=15)

    # Residual Skip Connection 2
    ax.plot([7.85, 5.75], [4.25, 4.25], color='#059669', lw=1.8, zorder=20)
    ax.plot([5.75, 5.75], [4.25, 1.45], color='#059669', lw=1.8, zorder=20)
    ax.annotate('', xy=(7.62, 1.45), xytext=(5.75, 1.45),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color='#059669', mutation_scale=12), zorder=20)
    ax.text(5.62, 2.85, r"$\mathrm{Residual\ Skip\ 2}$", ha='center', va='center',
            color='#047857', fontsize=7.0, fontweight='bold', rotation=90, zorder=25)
    ax.text(6.65, 1.65, r"$+\mathbf{H}'$", ha='center', va='bottom', color='#047857', fontsize=7.6, fontweight='bold', zorder=25)

    # Addition Node 2 (+)
    draw_addition_circle(ax, 7.85, 1.45, r=0.20, face_col='#DCFCE7', edge_col='#16A34A', text_col='#14532D', zorder=25)
    ax.text(8.20, 1.65, r"$\mathbf{H} = \mathbf{H}' + \mathrm{FFN} \in \mathbb{R}^{D \times 16}$", ha='left', va='bottom',
            color='#14532D', fontsize=7.8, fontweight='bold', zorder=25)

    # Clean Chaining Badge indicating Layer 1 -> Layer 2 (x2 Blocks)
    draw_pill(ax, 6.20, 0.90, 3.30, 0.40,
              text=r"$\mathbf{H}^{(2)} = \mathrm{Layer}_2(\mathrm{Layer}_1(\mathbf{T}))\quad (\times 2\ \mathrm{Blocks})$",
              face_col='#DCFCE7', edge_col='#16A34A', text_col='#14532D', font_size=7.4, font_weight='bold')

    # Exit Routing from Transformer Block to Column 4
    ax.plot([8.05, 10.55], [1.45, 1.45], color='#16A34A', lw=1.8, zorder=15)
    ax.plot([10.55, 10.55], [1.45, 7.15], color='#16A34A', lw=1.8, zorder=15)
    draw_arrow(ax, 10.55, 7.15, 10.80, 7.15, color='#16A34A', lw=1.8, zorder=15)
    
    bbox_h2 = dict(boxstyle="round,pad=0.25,rounding_size=0.18", facecolor='#16A34A', edgecolor='none')
    ax.text(10.55, 4.30, r"$\mathbf{H}^{(2)} \in \mathbb{R}^{D \times 16}$" + "\n" + r"$(\mathrm{Contextual\ Sequence})$",
            ha='center', va='center', color='white', fontsize=7.2, fontweight='bold', bbox=bbox_h2, rotation=90, zorder=25)

    # -------------------------------------------------------------------------
    # COLUMN 4: LEARNED ATTENTION POOLING & HYPERSPHERE PROJECTION
    # -------------------------------------------------------------------------
    panel_pool = patches.FancyBboxPatch((10.80, 0.80), 2.85, 8.20, boxstyle="round,pad=0.15,rounding_size=0.2",
                                        facecolor='#FFFBEB', edgecolor='#D97706', lw=1.4, zorder=5)
    ax.add_patch(panel_pool)
    ax.text(12.22, 8.65, "Attention Pooling\n& Hypersphere Projection", ha='center', va='center',
            color='#B45309', fontsize=10.0, fontweight='bold', zorder=25)

    # Stage A: Query-Based Feature Importance
    draw_pill(ax, 11.00, 4.85, 2.45, 3.10,
              text="Query Attention Pooling:\n\n" +
                   r"$\mathbf{q}_{\mathrm{pool}} \in \mathbb{R}^{16}\ (\mathrm{Learned})$" + "\n\n" +
                   r"$\alpha_i = \frac{\exp(\mathbf{h}_i \cdot \mathbf{q} / \sqrt{d})}{\sum_j \exp(\mathbf{h}_j \cdot \mathbf{q} / \sqrt{d})}$" + "\n\n" +
                   r"$\mathbf{h}_{\mathrm{pool}} = \sum_{i=1}^D \alpha_i \mathbf{h}_i \in \mathbb{R}^{16}$" + "\n" +
                   r"$(\mathrm{Sample\text{-}Level\ Invariance})$",
              face_col='#FFFFFF', edge_col='#D97706', text_col='#92400E', font_size=7.4, font_weight='normal')

    draw_arrow(ax, 12.22, 4.85, 12.22, 4.10, color='#D97706', lw=1.6, zorder=15)

    # Stage B: Hypersphere Projection Head
    draw_pill(ax, 11.00, 1.35, 2.45, 2.60,
              text="Projection Head & Norm:\n\n" +
                   r"$\tilde{\mathbf{z}} = \mathrm{MLP}(\mathbf{h}_{\mathrm{pool}}) \in \mathbb{R}^{16}$" + "\n" +
                   r"$(\mathrm{Linear} \to \mathrm{GELU} \to \mathrm{Linear})$" + "\n\n" +
                   r"$\mathbf{z} = \frac{\tilde{\mathbf{z}}}{\|\tilde{\mathbf{z}}\|_2} \in \mathbb{S}^{15}$" + "\n" +
                   r"$(\mathrm{Unit\ Hypersphere})$",
              face_col='#FEF3C7', edge_col='#D97706', text_col='#78350F', font_size=7.5, font_weight='normal')

    # Arrow into Metric Head
    draw_arrow(ax, 13.85, 2.65, 14.35, 2.65, color='#D97706', lw=1.8, badge=r'$\mathbf{z} \in \mathbb{S}^{15}$', badge_col='#D97706', zorder=20)

    # -------------------------------------------------------------------------
    # COLUMN 5: EPISODIC FEW-SHOT METRIC PROTOTYPE HEAD
    # -------------------------------------------------------------------------
    panel_head = patches.FancyBboxPatch((14.35, 0.80), 3.00, 8.20, boxstyle="round,pad=0.15,rounding_size=0.2",
                                        facecolor='#FAF5FF', edgecolor='#9333EA', lw=1.6, zorder=5)
    ax.add_patch(panel_head)
    ax.text(15.85, 8.65, "Episodic Few-Shot Metric Head", ha='center', va='center',
            color='#7E22CE', fontsize=10.2, fontweight='bold', zorder=25)

    # Prototype Shaded Spheres
    draw_sphere(ax, 14.85, 7.65, r=0.26, base_col='#3B82F6', highlight_col='#93C5FD', edge_col='#1D4ED8', zorder=25)
    ax.text(15.25, 7.65, r"$\mathbf{c}_0\ (\mathrm{Class\ 0:\ Control})$", ha='left', va='center',
            color='#1E40AF', fontsize=7.8, fontweight='bold', zorder=25)

    draw_sphere(ax, 14.85, 6.85, r=0.26, base_col='#EF4444', highlight_col='#FCA5A5', edge_col='#B91C1C', zorder=25)
    ax.text(15.25, 6.85, r"$\mathbf{c}_1\ (\mathrm{Class\ 1:\ Incident})$", ha='left', va='center',
            color='#991B1B', fontsize=7.8, fontweight='bold', zorder=25)

    # Prototype Definition Formula Pill
    draw_pill(ax, 14.50, 4.85, 2.70, 1.75,
              text="12-Shot Support Prototypes:\n\n" +
                   r"$\mathbf{c}_k^* = \frac{\sum_{i \in S_k^*} \mathbf{z}_i}{\|\sum_{i \in S_k^*} \mathbf{z}_i\|_2}\ (|S_0^*|=12, |S_1^*|=12)$" + "\n\n" +
                   r"Identical 12-shot prototype rule across" + "\n" +
                   r"training, val checkpointing & test inference",
              face_col='#FFFFFF', edge_col='#9333EA', text_col='#6B21A8', font_size=7.1, font_weight='normal')

    draw_arrow(ax, 15.85, 4.85, 15.85, 4.25, color='#9333EA', lw=1.4, zorder=20)

    # Cosine Scoring Pill
    draw_pill(ax, 14.50, 2.45, 2.70, 1.80,
              text=r"Cosine Scoring on Test Query $\mathbf{z}^*$:" + "\n\n" +
                   r"$s_k(\mathbf{z}^*) = \frac{\mathbf{z}^{*T} \mathbf{c}_k^*}{\tau} = \frac{\cos(\mathbf{z}^*, \mathbf{c}_k^*)}{\tau}$" + "\n\n" +
                   r"$P(y = 1 \mid \mathbf{x}^*) = \frac{\exp(s_1)}{\exp(s_0) + \exp(s_1)}$" + "\n" +
                   r"$(\tau \in [0.05, 1.0]\ \mathrm{learnable})$",
              face_col='#F3E8FF', edge_col='#9333EA', text_col='#581C87', font_size=7.2, font_weight='normal')

    # Episodic Optimization Loss Pill
    draw_pill(ax, 14.50, 0.90, 2.70, 1.35,
              text="Episodic Training Loss:\n" +
                   r"$\mathcal{L}_{\mathrm{meta}} = -\frac{1}{|Q|} \sum_{q \in Q} \log P(y_q \mid \mathbf{x}_q; \mathbf{c}_k)$" + "\n" +
                   r"Query $Q$: $12\ \mathrm{cases} : 144\ \mathrm{controls}\ (1:12)$" + "\n" +
                   r"Directly trains prototypes $\mathbf{c}_k^*$ used at test",
              face_col='#7E22CE', edge_col='#581C87', text_col='#FFFFFF', font_size=6.9, font_weight='bold')

    draw_arrow(ax, 17.15, 3.35, 17.55, 3.35, color='#0F172A', lw=1.8, zorder=20)

    # -------------------------------------------------------------------------
    # COLUMN 6: EXTRACTED FEATURES & OUTPUT PREDICTION (3D Silver Pillar)
    # -------------------------------------------------------------------------
    draw_3d_block(ax, 17.55, 2.00, 0.75, 6.20, depth=0.25, angle=35,
                  face_col='#E2E8F0', top_col='#F8FAFC', side_col='#CBD5E1', edge_col='#94A3B8', zorder=10)
    ax.text(17.92, 5.10, r"$\mathbf{P}_{\mathrm{Quest}} = P(y = 1 \mid \mathbf{x}) \in [0, 1]$", ha='center', va='center',
            color='#0F172A', fontsize=9.2, fontweight='bold', rotation=90, zorder=30)

    save_figure_to_all_locations(fig, "Fig4_Prototypical_Network")
    plt.close(fig)
    print("  [OK] Fig4_Prototypical_Network successfully generated.")


if __name__ == "__main__":
    print("=" * 70)
    print("REGENERATING ALL 4 PUBLICATION ARCHITECTURE FIGURES (300 DPI)")
    print("=" * 70)
    build_fig1()
    build_fig2()
    build_fig3()
    build_fig4()
    print("=" * 70)
    print("ALL 4 FIGURES SUCCESSFULLY REGENERATED IN ALL LOCATIONS")
    print("=" * 70)
