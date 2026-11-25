"""
code/generate_publication_architectures.py
=========================================
Publication-Grade 3D Scientific Architecture Figures (Figs 1, 2, 3, 4)
Mathematical and Logical Alignment with Q1 Journal Review Standards

Generates:
  - Fig 1: Overall Multimodal Framework (Rank Averaging & Dawes' Decision Fusion)
  - Fig 2: Module 1 — Structural Neuroimaging Pipeline (Med3D 3D-ResNet-18 + Train-Only PCA-5 + Balanced LR)
  - Fig 3: Module 2 — Longitudinal Anxiety Symptomatology Architecture (Optimized Emerald Design)
  - Fig 4: Detailed Deep Feature-Tokenizer ProtoNet Architecture (Intra-Sample Attention + Metric Head)
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIRS = [
    BASE_DIR / 'figures',
    BASE_DIR / 'results',
    BASE_DIR.parent / 'figures' if BASE_DIR.name == 'REVISION_PAPER_PRO_MAX' else BASE_DIR / 'REVISION_PAPER_PRO_MAX' / 'figures',
    BASE_DIR.parent / 'results' if BASE_DIR.name == 'REVISION_PAPER_PRO_MAX' else BASE_DIR / 'REVISION_PAPER_PRO_MAX' / 'results'
]
OUT_DIRS = [d for d in OUT_DIRS if d.parent.exists()]
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
    
    front = patches.Rectangle((x, y), w, h, facecolor=face_col,
                              edgecolor=edge_col, linewidth=lw, zorder=zorder)
    ax.add_patch(front)
    
    top_pts = np.array([[x, y + h], [x + w, y + h], [x + w + dx, y + h + dy], [x + dx, y + h + dy]])
    top = patches.Polygon(top_pts, closed=True, facecolor=top_col,
                          edgecolor=edge_col, linewidth=lw, zorder=zorder+1)
    ax.add_patch(top)
    
    side_pts = np.array([[x + w, y], [x + w + dx, y + dy], [x + w + dx, y + h + dy], [x + w, y + h]])
    side = patches.Polygon(side_pts, closed=True, facecolor=side_col,
                           edgecolor=edge_col, linewidth=lw, zorder=zorder+1)
    ax.add_patch(side)
    return dx, dy

def draw_pill(ax, x, y, w, h, text='', face_col='#FFFFFF', edge_col='#94A3B8',
              text_col='#0F172A', font_size=8.5, font_weight='bold', zorder=20):
    """Draws a clean rounded card container with centered text."""
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
    """Draws a crisp participant avatar vector."""
    ax.add_patch(patches.Circle((cx, cy + s*0.38), s*0.30, facecolor='none', edgecolor=col, lw=1.3, zorder=zorder))
    ax.add_patch(patches.Arc((cx, cy - s*0.48), s*1.15, s*1.15, angle=0, theta1=35, theta2=145, edgecolor=col, lw=1.3, zorder=zorder))
    ax.plot([cx - s*0.35, cx - s*0.15, cx, cx + s*0.15, cx + s*0.35],
            [cy - s*0.2, cy - s*0.2, cy - s*0.05, cy - s*0.35, cy - s*0.2], color=col, lw=1.1, zorder=zorder)

def draw_authentic_mri_display(ax, cx, cy, w=1.75, h=1.55, zorder=20):
    """
    Renders an authentic, publication-grade scientific anatomical T1w MRI cross-section
    with realistic tissue contrast (dark CSF ventricles, bright white matter tracts,
    intermediate cortical ribbon, and skull/scalp boundary).
    Completely eliminates any hand-drawn polygons, bean shapes, or emoji resemblance.
    """
    ny, nx = 120, 120
    y, x = np.mgrid[-1:1:ny*1j, -1:1:nx*1j]

    r_head = (x/0.78)**2 + (y/0.92)**2
    head_mask = r_head <= 1.0
    skull_mask = (r_head <= 1.0) & (r_head > 0.88)
    r_brain = (x/0.70)**2 + (y/0.82)**2
    cortex_mask = (r_brain <= 1.0) & (r_brain > 0.60)
    wm_mask = (r_brain <= 0.60)
    v_left = ((x + 0.14)/0.08)**2 + ((y - 0.05)/0.25)**2 <= 1.0
    v_right = ((x - 0.14)/0.08)**2 + ((y - 0.05)/0.25)**2 <= 1.0
    ventricles = v_left | v_right
    fissure = (np.abs(x) < 0.02) & (r_brain <= 1.0)

    img = np.zeros((ny, nx))
    img[head_mask] = 0.15
    img[skull_mask] = 0.65
    img[cortex_mask] = 0.55
    img[wm_mask] = 0.82
    img[ventricles] = 0.05
    img[fissure] = 0.20
    img_smooth = gaussian_filter(img, sigma=1.4)

    x0, x1 = cx - w/2.0, cx + w/2.0
    y0, y1 = cy - h/2.0, cy + h/2.0
    
    bezel = patches.FancyBboxPatch((x0 - 0.04, y0 - 0.04), w + 0.08, h + 0.08,
                                   boxstyle="round,pad=0.03,rounding_size=0.06",
                                   facecolor='#0B132B', edgecolor='#38BDF8', lw=1.2, zorder=zorder)
    ax.add_patch(bezel)
    
    ax.imshow(img_smooth, extent=[x0, x1, y0, y1], cmap='bone', origin='lower',
              vmin=0, vmax=1, zorder=zorder+1, aspect='auto')
    
    ax.plot([cx, cx], [y1 - 0.10, y1], color='#38BDF8', lw=1.0, zorder=zorder+2)
    ax.plot([cx, cx], [y0, y0 + 0.10], color='#38BDF8', lw=1.0, zorder=zorder+2)
    ax.plot([x0, x0 + 0.10], [cy, cy], color='#38BDF8', lw=1.0, zorder=zorder+2)
    ax.plot([x1 - 0.10, x1], [cy, cy], color='#38BDF8', lw=1.0, zorder=zorder+2)
    
    ax.text(cx, y1 + 0.03, 'S (Axial Slice)', ha='center', va='bottom', color='#38BDF8', fontsize=6.8, fontweight='bold', zorder=zorder+3)
    ax.text(x0 - 0.04, cy, 'R', ha='right', va='center', color='#38BDF8', fontsize=6.8, fontweight='bold', zorder=zorder+3)
    ax.text(x1 + 0.04, cy, 'L', ha='left', va='center', color='#38BDF8', fontsize=6.8, fontweight='bold', zorder=zorder+3)

def save_figure_to_all_locations(fig, filename_base):
    """Saves the given figure to all registered output directories at 300 DPI."""
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
    fig, ax = plt.subplots(figsize=(17.2, 9.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 17.2)
    ax.set_ylim(-0.4, 9.8)
    ax.axis('off')

    # Top Modules (3D Beveled Blocks) - Width expanded to 4.90 for zero text clipping!
    # Module 1 (Teal/Navy - Structural Neuroimaging)
    draw_3d_block(ax, 0.60, 7.3, 4.90, 1.45, depth=0.35, angle=35,
                  face_col='#1F618D', top_col='#5DADE2', side_col='#154360', edge_col='#0B2535', zorder=10)
    ax.add_patch(patches.Circle((1.25, 8.02), 0.38, facecolor='#154360', edgecolor='#5DADE2', lw=1.0, zorder=25))
    draw_brain_vector(ax, 1.25, 8.02, s=0.26, col='#EBF5FB', zorder=30)
    ax.text(1.85, 8.20, "Module 1: STRUCTURAL MRI", ha='left', va='center',
            color='white', fontsize=10.5, fontweight='bold', zorder=30)
    ax.text(1.85, 7.86, "Med3D ResNet-18 + Train-Only PCA-5", ha='left', va='center',
            color='#D4E6F1', fontsize=8.2, zorder=30)
    ax.text(1.85, 7.56, "Balanced Logistic Regression (C=0.1)", ha='left', va='center',
            color='#A9CCE3', fontsize=7.6, zorder=30)

    # Module 2 (Emerald Green - Behavioral Questionnaires)
    draw_3d_block(ax, 6.15, 7.3, 4.90, 1.45, depth=0.35, angle=35,
                  face_col='#239B56', top_col='#58D68D', side_col='#196F3D', edge_col='#0E4B24', zorder=10)
    ax.add_patch(patches.Circle((6.80, 8.02), 0.38, facecolor='#196F3D', edgecolor='#58D68D', lw=1.0, zorder=25))
    draw_clipboard_vector(ax, 6.80, 8.02, s=0.25, col='#FFFFFF', zorder=30)
    ax.text(7.40, 8.20, "Module 2: QUESTIONNAIRES", ha='left', va='center',
            color='white', fontsize=10.5, fontweight='bold', zorder=30)
    ax.text(7.40, 7.86, "9 Transition Markers + PCA-3 Orthogonalization", ha='left', va='center',
            color='#D1F2EB', fontsize=8.2, zorder=30)
    ax.text(7.40, 7.56, "Balanced L2-Logistic Regression (C=0.01)", ha='left', va='center',
            color='#A3E4D7', fontsize=7.6, zorder=30)

    # Module 3 (Warm Rust/Orange - Phenotypic & Demographics)
    draw_3d_block(ax, 11.70, 7.3, 4.90, 1.45, depth=0.35, angle=35,
                  face_col='#D35400', top_col='#F39C12', side_col='#935116', edge_col='#592804', zorder=10)
    ax.add_patch(patches.Circle((12.35, 8.02), 0.38, facecolor='#935116', edgecolor='#F39C12', lw=1.0, zorder=25))
    draw_user_vector(ax, 12.35, 8.02, s=0.25, col='#FEF5E7', zorder=30)
    ax.text(12.95, 8.20, "Module 3: PHENOTYPIC", ha='left', va='center',
            color='white', fontsize=10.5, fontweight='bold', zorder=30)
    ax.text(12.95, 7.86, "4 SES & Pubertal Maturation Predictors", ha='left', va='center',
            color='#FDEBD0', fontsize=8.2, zorder=30)
    ax.text(12.95, 7.56, "Balanced L2-Logistic Regression (C=0.01)", ha='left', va='center',
            color='#FAD7A0', fontsize=7.6, zorder=30)

    # Downward inference flow arrows
    draw_arrow(ax, 3.05, 7.3, 3.05, 6.25, color='#64748B', lw=1.8, zorder=12)
    draw_arrow(ax, 8.60, 7.3, 8.60, 6.25, color='#64748B', lw=1.8, zorder=12)
    draw_arrow(ax, 14.15, 7.3, 14.15, 6.25, color='#64748B', lw=1.8, zorder=12)
    ax.text(3.05, 6.75, "Sample-Independent Scoring", ha='center', va='center',
            color='#475569', fontsize=8.0, fontstyle='italic', zorder=25)
    ax.text(8.60, 6.75, "Sample-Independent Scoring", ha='center', va='center',
            color='#475569', fontsize=8.0, fontstyle='italic', zorder=25)
    ax.text(14.15, 6.75, "Sample-Independent Scoring", ha='center', va='center',
            color='#475569', fontsize=8.0, fontstyle='italic', zorder=25)

    # Middle Row: Unimodal Probability Slabs
    draw_3d_block(ax, 1.20, 5.5, 3.70, 0.72, depth=0.25, angle=35,
                  face_col='#F1F5F9', top_col='#FFFFFF', side_col='#CBD5E1', edge_col='#94A3B8', zorder=10)
    ax.text(3.05, 5.86, r"$\mathbf{P}_{\mathrm{MRI}}\ (\mathrm{Test\ AUC:\ 0.7221})$", ha='center', va='center',
            color='#0F172A', fontsize=9.8, fontweight='bold', zorder=30)

    draw_3d_block(ax, 6.75, 5.5, 3.70, 0.72, depth=0.25, angle=35,
                  face_col='#F1F5F9', top_col='#FFFFFF', side_col='#CBD5E1', edge_col='#94A3B8', zorder=10)
    ax.text(8.60, 5.86, r"$\mathbf{P}_{\mathrm{Quest}}\ (\mathrm{Test\ AUC:\ 0.6260})$", ha='center', va='center',
            color='#0F172A', fontsize=9.8, fontweight='bold', zorder=30)

    draw_3d_block(ax, 12.30, 5.5, 3.70, 0.72, depth=0.25, angle=35,
                  face_col='#F1F5F9', top_col='#FFFFFF', side_col='#CBD5E1', edge_col='#94A3B8', zorder=10)
    ax.text(14.15, 5.86, r"$\mathbf{P}_{\mathrm{Pheno}}\ (\mathrm{Test\ AUC:\ 0.6870})$", ha='center', va='center',
            color='#0F172A', fontsize=9.8, fontweight='bold', zorder=30)

    # Clean diagonal arrows with prominent w = 1/3 weight badges
    draw_arrow(ax, 3.05, 5.5, 6.45, 4.15, color='#64748B', lw=1.8, badge='w = 1/3', badge_col='#1F618D', zorder=12)
    draw_arrow(ax, 8.60, 5.5, 8.60, 4.15, color='#64748B', lw=1.8, badge='w = 1/3', badge_col='#239B56', zorder=12)
    draw_arrow(ax, 14.15, 5.5, 10.75, 4.15, color='#64748B', lw=1.8, badge='w = 1/3', badge_col='#D35400', zorder=12)

    # Multimodal Decision-Level Fusion 3D Bar (Royal Purple) - Expanded Width 11.20
    draw_3d_block(ax, 3.00, 2.85, 11.20, 1.25, depth=0.35, angle=35,
                  face_col='#6C3483', top_col='#A569BD', side_col='#4A235A', edge_col='#2C1337', zorder=10)
    ax.text(8.60, 3.70, "Primary Decision-Level Fusion (Zero-Parameter Ensembling)", ha='center', va='center',
            color='white', fontsize=11.6, fontweight='bold', zorder=30)
    ax.text(8.60, 3.32, r"$\mathbf{P}_{\mathrm{rank}} = \frac{1}{3} \mathrm{Rank}(\mathbf{P}_{\mathrm{MRI}}) + \frac{1}{3} \mathrm{Rank}(\mathbf{P}_{\mathrm{Quest}}) + \frac{1}{3} \mathrm{Rank}(\mathbf{P}_{\mathrm{Pheno}})\quad [\mathrm{Test\ AUC:\ 0.8091}]$",
            ha='center', va='center', color='#F5EEF8', fontsize=9.2, fontweight='bold', zorder=30)
    ax.text(8.60, 3.00, r"Dawes' Equal Average: $P_{\mathrm{Dawes}} = \frac{1}{3}(P_{\mathrm{MRI}} + P_{\mathrm{Quest}} + P_{\mathrm{Pheno}})\quad [\mathrm{Test\ AUC:\ 0.7792}]$",
            ha='center', va='center', color='#E8DAEF', fontsize=8.4, fontstyle='italic', zorder=30)

    # Downward arrow to Threshold Calibration
    draw_arrow(ax, 8.60, 2.85, 8.60, 2.15, color='#64748B', lw=1.6, zorder=12)
    ax.text(8.60, 2.50, "Dual Prospective Operating Points", ha='center', va='center',
            color='#64748B', fontsize=8.0, fontstyle='italic', zorder=25)

    # Threshold Optimization 3D Bar (Rich Gold / Amber) - EXPANDED WIDTH 11.20 (Zero Text Bleed!)
    draw_3d_block(ax, 3.00, 1.45, 11.20, 0.70, depth=0.25, angle=35,
                  face_col='#D4AC0D', top_col='#F4D03F', side_col='#9A7D0A', edge_col='#4D3E05', zorder=10)
    ax.text(8.60, 1.88, r"Validation-Derived Decision Thresholds (Strict Out-of-Fold Calibration)",
            ha='center', va='center', color='white', fontsize=10.0, fontweight='bold', zorder=30)
    ax.text(8.60, 1.60, r"Screening Mode ($\tau = 0.50$): 100% Sens (7/7, 0 FN) $\mid$ Diagnostic Mode ($\tau^* = 0.56$): 85.7% Sens, 63.6% Spec (35/55 non-trans)",
            ha='center', va='center', color='#FEF9E7', fontsize=8.2, zorder=30)

    # Downward arrow to Final Prediction
    draw_arrow(ax, 8.60, 1.45, 8.60, 0.85, color='#64748B', lw=1.6, zorder=12)

    # Final Clinical Decision 3D Bar (Teal / Deep Cyan) - EXPANDED WIDTH 13.00 (Zero Text Bleed!)
    draw_3d_block(ax, 2.10, -0.20, 13.00, 1.05, depth=0.30, angle=35,
                  face_col='#148F77', top_col='#48C9B0', side_col='#0E6252', edge_col='#083E34', zorder=10)
    ax.text(8.60, 0.58, "Prospective Incident Anxiety Disorder Prediction (Held-Out Test Cohort, n = 62)", ha='center', va='center',
            color='white', fontsize=11.2, fontweight='bold', zorder=30)
    ax.text(8.60, 0.28, r"$\mathrm{Multimodal\ Rank\ Averaging:\ ROC\text{-}AUC} = 0.8091\ [95\%\ \mathrm{CI}:\ 0.660 - 0.931] \mid \mathrm{Raw}\ p = 0.0020 \mid \mathrm{FDR}\ q = 0.0160$",
            ha='center', va='center', color='white', fontsize=8.6, fontweight='bold', zorder=30)
    ax.text(8.60, 0.02, r"$\mathrm{Screening}\ (\tau=0.50):\ \mathrm{Sens} = 100.0\%\ (7/7) \mid \mathrm{NPV} = 100.0\% \mid \mathrm{Diagnostic}\ (\tau^*=0.56):\ \mathrm{Spec} = 63.6\%\ (35/55) \mid \mathrm{NPV} = 97.2\% \mid F_1 = 0.364$",
            ha='center', va='center', color='#E8F8F5', fontsize=8.2, zorder=30)

    save_figure_to_all_locations(fig, "Fig1_Overall_Framework")
    plt.close(fig)
    print("  [OK] Fig1_Overall_Framework successfully generated.")


# =========================================================================
# FIGURE 2: MODULE 1 — STRUCTURAL NEUROIMAGING PIPELINE
# =========================================================================
def build_fig2():
    print("Generating Figure 2: Module 1 Structural MRI Architecture (Scientific MRI Display)...")
    fig, ax = plt.subplots(figsize=(16.8, 9.0), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0.2, 9.2)
    ax.axis('off')

    # 1. INPUT RAW SCANS (Stacked dark cards on Left with Authentic Scientific MRI Display)
    draw_3d_block(ax, 0.35, 4.3, 2.3, 3.1, depth=0.25, angle=35,
                  face_col='#1E293B', top_col='#334155', side_col='#0F172A', edge_col='#020617', zorder=10)
    ax.text(1.50, 7.15, "Input Raw Scans\nT1-weighted MRI\n(91 x 109 x 91)", ha='center', va='center',
            color='white', fontsize=8.8, fontweight='bold', zorder=30)

    # Authentic scientific grayscale anatomical MRI slice display (zero emoji/bean shapes!)
    draw_authentic_mri_display(ax, cx=1.50, cy=5.65, w=1.75, h=1.55, zorder=25)

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

    inner_box = patches.FancyBboxPatch((7.35, 4.4), 5.25, 3.3, boxstyle="round,pad=0.1,rounding_size=0.15",
                                       facecolor='#ECEFF1', edgecolor='#CFD8DC', lw=1.0, zorder=6)
    ax.add_patch(inner_box)

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

    draw_arrow(ax, 10.45, 5.9, 10.75, 5.9, color='#64748B', lw=1.4, zorder=15)

    draw_pill(ax, 10.80, 5.35, 0.90, 1.10, text="Global\nAvgPool\n& Flatten",
              face_col='#FFFFFF', edge_col='#0284C7', text_col='#0369A1', font_size=7.0, font_weight='bold')

    draw_arrow(ax, 11.80, 5.9, 12.15, 5.9, color='#64748B', lw=1.4, zorder=15)

    draw_pill(ax, 12.15, 5.30, 0.38, 1.20, text=r"$\mathbf{e} \in \mathbb{R}^{768}$",
              face_col='#E0F2FE', edge_col='#0284C7', text_col='#0369A1', font_size=7.2, font_weight='bold')

    draw_arrow(ax, 9.98, 4.4, 9.98, 3.55, color='#0284C7', lw=1.8, badge='Train-Only Fit', badge_col='#0284C7', zorder=15)

    fs_box = patches.FancyBboxPatch((7.35, 1.15), 5.25, 2.3, boxstyle="round,pad=0.1,rounding_size=0.15",
                                    facecolor='#ECEFF1', edgecolor='#CFD8DC', lw=1.0, zorder=6)
    ax.add_patch(fs_box)
    
    funnel_pts = np.array([[7.50, 3.05], [8.10, 3.05], [7.88, 2.30], [7.72, 2.30]])
    funnel = patches.Polygon(funnel_pts, closed=True, facecolor='#E0F2FE', edgecolor='#0284C7', lw=1.8, zorder=15)
    ax.add_patch(funnel)
    ax.plot([7.80, 7.80], [2.30, 1.80], color='#0284C7', lw=2.0, zorder=15)
    ax.text(7.80, 1.55, "k = 5", ha='center', va='center', color='#0284C7', fontsize=8.0, fontweight='bold', zorder=25)

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

    draw_pill(ax, 13.55, 6.00, 2.8, 1.5,
              text=r"$L_2$-Regularized" + "\nLogistic Regression\n" + r"$(C = 0.1, \mathrm{balanced})$",
              face_col='#ECEFF1', edge_col='#CFD8DC', text_col='#0F172A', font_size=8.0, font_weight='bold')
    
    draw_arrow(ax, 14.95, 6.00, 14.95, 5.10, color='#0284C7', lw=1.6, zorder=15)

    draw_pill(ax, 13.55, 4.00, 2.8, 1.1, text=r"Predicted Risk:" + "\n" + r"$\mathbf{P}_{\mathrm{MRI}} \in [0, 1]$",
              face_col='#E0F2FE', edge_col='#0284C7', text_col='#0369A1', font_size=9.0, font_weight='bold')

    draw_arrow(ax, 14.95, 4.00, 14.95, 3.05, color='#0284C7', lw=1.6, zorder=15)

    draw_pill(ax, 13.55, 1.15, 2.8, 1.8,
              text="Held-Out Test Performance:\n" + r"$\mathrm{Test\ ROC\text{-}AUC} = 0.7221$" + "\n" + r"$\mathrm{PR\text{-}AUC} = 0.1923$" + "\n" + r"$\mathrm{Sensitivity} = 71.4\%\ (5/7)$" + "\n" + r"$\mathrm{Specificity} = 61.8\%\ (34/55\ \mathrm{non\text{-}trans})$" + "\n" + r"$\mathrm{NPV} = 94.4\% \mid \mathrm{Brier} = 0.2266$",
              face_col='#0F172A', edge_col='#334155', text_col='#38BDF8', font_size=7.6, font_weight='bold')

    save_figure_to_all_locations(fig, "Fig2_MRI_Preprocessing")
    plt.close(fig)
    print("  [OK] Fig2_MRI_Preprocessing successfully generated.")


# =========================================================================
# FIGURE 3: MODULE 2 — LONGITUDINAL QUESTIONNAIRES ARCHITECTURE
# (Creative Scientific 5-Column Design, Fully Aligned with 02_questionnaire_module.py)
# =========================================================================
def build_fig3():
    print("Generating Figure 3: Module 2 Questionnaire Architecture (Creative Scientific 5-Column Design)...")
    fig, ax = plt.subplots(figsize=(18.4, 9.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 18.4)
    ax.set_ylim(0.2, 9.8)
    ax.axis('off')

    # Main Title (Centered at Top)
    ax.text(9.20, 9.55, "Module 2: Longitudinal Anxiety Symptomatology Architecture (Zero-Leakage PCA-3 Head)",
            ha='center', va='center', color='#0F172A', fontsize=14.0, fontweight='bold')
    ax.text(9.20, 9.24, r"Psychometric Intake ($\mathbb{R}^{109}$) $\to$ 9 Transition Markers $\to$ Train-Only PCA-3 Eigenspace ($\mathbb{R}^3$) $\to$ 5-Fold Family Group-CV Bagged Head",
            ha='center', va='center', color='#475569', fontsize=9.2, fontstyle='italic')

    # =========================================================================
    # COLUMN 1: PSYCHOMETRICS & IN-FOLD INTAKE (x in [0.45, 3.75], w = 3.30)
    # =========================================================================
    c1_box = patches.FancyBboxPatch((0.45, 0.80), 3.30, 8.20, boxstyle="round,pad=0.10,rounding_size=0.18",
                                    facecolor='#F8FAFC', edgecolor='#059669', lw=1.5, zorder=5)
    ax.add_patch(c1_box)

    draw_3d_block(ax, 0.55, 8.15, 3.10, 0.70, depth=0.18, face_col='#059669', top_col='#34D399', side_col='#065F46')
    ax.text(2.10, 8.50, "1. Psychometrics ($N=304$)", ha='center', va='center',
            color='white', fontsize=10.0, fontweight='bold', zorder=25)

    # Visual: Psychometric Response Matrix / Multi-Battery Heatmap (Spacious 2x2 Grid)
    ax.text(2.10, 7.85, r"$\mathbf{X}_{\mathrm{raw}} \in \mathbb{R}^{304 \times 109}$ (Item Responses)", ha='center', va='center',
            color='#065F46', fontsize=8.4, fontweight='bold', zorder=25)

    # 2x2 grid for psychometric batteries to eliminate horizontal text crowding
    batt_specs = [
        ("SCAS-Child (44)", 0.60, 7.15, 1.45, 0.40, '#0284C7', '#BAE6FD'),
        ("SCAS-Parent (38)", 2.15, 7.15, 1.45, 0.40, '#0D9488', '#99F6E4'),
        ("SPHERE-34 (34)", 0.60, 6.45, 1.45, 0.40, '#16A34A', '#BBF7D0'),
        ("SDQ / MFQ Scales", 2.15, 6.45, 1.45, 0.40, '#7C3AED', '#DDD6FE')
    ]
    for bname, bx, by, bw, bh, bcol_dark, bcol_light in batt_specs:
        b_card = patches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.03,rounding_size=0.06",
                                        facecolor=bcol_dark, edgecolor='none', zorder=15)
        ax.add_patch(b_card)
        ax.text(bx + bw/2.0, by + bh/2.0, bname, ha='center', va='center',
                color='white', fontsize=6.8, fontweight='bold', zorder=20)
        # Miniature item response indicator tiles underneath each battery
        np.random.seed(int(bx * 100 + by * 10))
        for r_tile in range(7):
            tile_x = bx + r_tile * (bw / 7.0) + 0.02
            tile_col = bcol_dark if np.random.rand() > 0.5 else bcol_light
            ax.add_patch(patches.Rectangle((tile_x, by - 0.16), bw/7.0 - 0.04, 0.11,
                                           facecolor=tile_col, edgecolor='#CBD5E1', lw=0.3, zorder=12))

    ax.text(2.10, 6.00, r"$N = 304$ Adolescents (152 Twin Pairs)" + "\n" + r"109 Item-Level Standardized Assessments",
            ha='center', va='center', color='#334155', fontsize=7.2, fontweight='bold', zorder=25)

    # In-Fold Preprocessing Container
    draw_pill(ax, 0.60, 1.05, 3.00, 4.60,
              text="Leakage-Free In-Fold Processing:\n\n" +
                   r"$\bullet$ In-Fold Median Imputation:" + "\n" +
                   r"  Imputation values ($\hat{\mu}_{\mathrm{tr}}$) derived" + "\n" +
                   r"  exclusively from training folds" + "\n\n" +
                   r"$\bullet$ Train-Only StandardScaler:" + "\n" +
                   r"  $\mathbf{x}_{\mathrm{norm}} = \frac{\mathbf{x} - \hat{\mu}_{\mathrm{tr}}}{\hat{\sigma}_{\mathrm{tr}}} \in \mathbb{R}^{109}$" + "\n" +
                   r"  (Zero test variance contamination)" + "\n\n" +
                   r"$\bullet$ Purged Sum-Scores:" + "\n" +
                   r"  Removes criterion proxy leakage" + "\n\n" +
                   r"$\bullet$ Family-Stratified Partition:" + "\n" +
                   r"  90 Train : 31 Val : 31 Test Families" + "\n" +
                   r"  (Twin pairs strictly unseparated)",
              face_col='#FFFFFF', edge_col='#A7F3D0', text_col='#065F46', font_size=7.4, font_weight='normal')

    draw_arrow(ax, 3.75, 5.20, 4.15, 5.20, color='#059669', lw=2.0, badge=r'$\mathbb{R}^{109} \to \mathbb{R}^9$', badge_col='#059669')

    # =========================================================================
    # COLUMN 2: 9 TRANSITION MARKERS (x in [4.15, 7.55], w = 3.40)
    # =========================================================================
    c2_box = patches.FancyBboxPatch((4.15, 0.80), 3.40, 8.20, boxstyle="round,pad=0.10,rounding_size=0.18",
                                    facecolor='#F0FDF4', edgecolor='#16A34A', lw=1.5, zorder=5)
    ax.add_patch(c2_box)

    draw_3d_block(ax, 4.25, 8.15, 3.20, 0.70, depth=0.18, face_col='#16A34A', top_col='#4ADE80', side_col='#15803D')
    ax.text(5.85, 8.50, r"2. Transition Feature Space ($\mathbb{R}^9$)", ha='center', va='center',
            color='white', fontsize=10.0, fontweight='bold', zorder=25)

    # Pod 1: Somatization
    p1_box = patches.FancyBboxPatch((4.30, 5.95), 3.10, 1.95, boxstyle="round,pad=0.08,rounding_size=0.12",
                                    facecolor='#FFFFFF', edgecolor='#16A34A', lw=1.2, zorder=10)
    ax.add_patch(p1_box)
    ax.add_patch(patches.Rectangle((4.30, 5.95), 0.12, 1.95, facecolor='#16A34A', edgecolor='none', zorder=12))
    ax.text(4.46, 7.65, "A. Somatization & Autonomic Fatigue", ha='left', va='center', color='#15803D', fontsize=8.2, fontweight='bold', zorder=20)
    ax.text(4.46, 6.75, r"$\bullet$ pSDQ03: Somatic complaints & headaches" + "\n" +
                        r"$\bullet$ SPHERE07: Mental exhaustion & fatigue" + "\n" +
                        r"$\bullet$ SPHERE08: Tension headaches & aches" + "\n" +
                        r"$\bullet$ SPHERE22: Muscular weakness & fatigue",
            ha='left', va='center', color='#1E293B', fontsize=6.8, zorder=20)

    # Pod 2: Social Fear
    p2_box = patches.FancyBboxPatch((4.30, 3.85), 3.10, 1.85, boxstyle="round,pad=0.08,rounding_size=0.12",
                                    facecolor='#FFFFFF', edgecolor='#0284C7', lw=1.2, zorder=10)
    ax.add_patch(p2_box)
    ax.add_patch(patches.Rectangle((4.30, 3.85), 0.12, 1.85, facecolor='#0284C7', edgecolor='none', zorder=12))
    ax.text(4.46, 5.45, "B. Social Evaluative Fear", ha='left', va='center', color='#0369A1', fontsize=8.2, fontweight='bold', zorder=20)
    ax.text(4.46, 4.65, r"$\bullet$ pSCAS_socpho_score: Parent social fear" + "\n" +
                        r"$\bullet$ SCAS_socpho_score: Child social fear" + "\n" +
                        r"$\bullet$ pSCAS31: Fear of negative evaluation",
            ha='left', va='center', color='#1E293B', fontsize=6.8, zorder=20)

    # Pod 3: Discrepancies
    p3_box = patches.FancyBboxPatch((4.30, 1.85), 3.10, 1.75, boxstyle="round,pad=0.08,rounding_size=0.12",
                                    facecolor='#FFFFFF', edgecolor='#D97706', lw=1.2, zorder=10)
    ax.add_patch(p3_box)
    ax.add_patch(patches.Rectangle((4.30, 1.85), 0.12, 1.75, facecolor='#D97706', edgecolor='none', zorder=12))
    ax.text(4.46, 3.35, "C. Cross-Informant Discrepancy", ha='left', va='center', color='#B45309', fontsize=8.2, fontweight='bold', zorder=20)
    ax.text(4.46, 2.55, r"$\bullet\ \Delta_{26} = \mathrm{pSCAS26} - \mathrm{SCAS26}$ (Somatic gap)" + "\n" +
                        r"$\bullet\ \Delta_{\mathrm{worry}} = \mathrm{pSCAS31} - \mathrm{SCAS31}$ (Worry gap)" + "\n" +
                        r"Captures masked subjective distress in youth",
            ha='left', va='center', color='#1E293B', fontsize=6.8, zorder=20)

    # Bottom Grounding Banner
    draw_pill(ax, 4.30, 1.05, 3.10, 0.65,
              text=r"Resolves Inversion: True Positive Gradient ($\mathbf{x} \in \mathbb{R}^9$)",
              face_col='#ECFDF5', edge_col='#059669', text_col='#065F46', font_size=7.4, font_weight='bold')

    draw_arrow(ax, 7.55, 5.20, 7.95, 5.20, color='#16A34A', lw=2.0, badge=r'$\mathbf{V}_3^T \mathbf{x}$', badge_col='#16A34A')

    # =========================================================================
    # COLUMN 3: TRAIN-ONLY PCA-3 EIGENSPACE (x in [7.95, 11.55], w = 3.60)
    # =========================================================================
    c3_box = patches.FancyBboxPatch((7.95, 0.80), 3.60, 8.20, boxstyle="round,pad=0.10,rounding_size=0.18",
                                    facecolor='#F0F9FF', edgecolor='#0284C7', lw=1.5, zorder=5)
    ax.add_patch(c3_box)

    draw_3d_block(ax, 8.05, 8.15, 3.40, 0.70, depth=0.18, face_col='#0284C7', top_col='#38BDF8', side_col='#0369A1')
    ax.text(9.75, 8.50, r"3. PCA-3 Eigenspace ($\mathbb{R}^3$)", ha='center', va='center',
            color='white', fontsize=10.0, fontweight='bold', zorder=25)

    # Visual: Authentic 3D Isometric Cartesian Eigenspace with Safe Boundaries
    orig_x, orig_y = 9.75, 6.20

    # Draw 3D axis arrows (compact geometry to guarantee zero border collision)
    ax.annotate('', xy=(orig_x + 0.65, orig_y - 0.35), xytext=(orig_x, orig_y),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color='#0284C7'), zorder=20)
    ax.text(orig_x + 0.70, orig_y - 0.35, r"$\mathbf{z}_1\ (34.5\%)$", ha='left', va='center',
            color='#0284C7', fontsize=7.2, fontweight='bold', zorder=25)

    ax.annotate('', xy=(orig_x, orig_y + 1.10), xytext=(orig_x, orig_y),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color='#10B981'), zorder=20)
    ax.text(orig_x, orig_y + 1.15, r"$\mathbf{z}_2\ (17.0\%)$", ha='center', va='bottom',
            color='#047857', fontsize=7.2, fontweight='bold', zorder=25)

    ax.annotate('', xy=(orig_x - 0.65, orig_y - 0.30), xytext=(orig_x, orig_y),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color='#8B5CF6'), zorder=20)
    ax.text(orig_x - 0.70, orig_y - 0.30, r"$\mathbf{z}_3\ (11.8\%)$", ha='right', va='center',
            color='#6D28D9', fontsize=7.2, fontweight='bold', zorder=25)

    # Draw 3D projected variance ellipsoids
    ell1 = patches.Ellipse((orig_x, orig_y), 1.60, 0.65, angle=-18, fill=True, facecolor='#BAE6FD', edgecolor='#0284C7', alpha=0.35, lw=1.2, zorder=12)
    ell2 = patches.Ellipse((orig_x, orig_y), 1.10, 0.45, angle=25, fill=True, facecolor='#A7F3D0', edgecolor='#10B981', alpha=0.35, lw=1.2, zorder=13)
    ax.add_patch(ell1)
    ax.add_patch(ell2)

    # Scatter sample projection points
    np.random.seed(99)
    pts_neg = np.random.randn(14, 2) * 0.35 + np.array([-0.15, -0.05])
    pts_pos = np.random.randn(5, 2) * 0.28 + np.array([0.45, 0.35])
    for px, py in pts_neg:
        ax.add_patch(patches.Circle((orig_x + px, orig_y + py), 0.045, facecolor='#3B82F6', edgecolor='white', lw=0.5, zorder=18))
    for px, py in pts_pos:
        ax.add_patch(patches.Circle((orig_x + px, orig_y + py), 0.055, facecolor='#EF4444', edgecolor='white', lw=0.7, zorder=19))

    ax.text(orig_x, 4.85, r"Orthogonal Projection: $\mathbf{z} = \mathbf{V}_3^T (\mathbf{x} - \hat{\mu}_{\mathrm{tr}}) \in \mathbb{R}^3$",
            ha='center', va='center', color='#0F172A', fontsize=7.6, fontweight='bold', zorder=25)

    # Mathematical Orthogonalization Pill
    draw_pill(ax, 8.10, 1.05, 3.30, 3.60,
              text="Train-Only PCA-3 Mathematical Rationale:\n\n" +
                   r"$\bullet$ Kaiser-Guttman Rule ($\lambda \geq 1.0$):" + "\n" +
                   r"  $\lambda_1 = 3.12$ (34.7% var: Somatosensory tension)" + "\n" +
                   r"  $\lambda_2 = 1.54$ (17.1% var: Informant discrepancy)" + "\n" +
                   r"  $\lambda_3 = 1.07$ (11.6% var: Evaluative fear)" + "\n\n" +
                   r"$\bullet$ Multicollinearity Elimination:" + "\n" +
                   r"  Raw inter-item redundancy ($r > 0.65$)" + "\n" +
                   r"  Condition number drops: $\kappa = 142.8 \to 1.0$" + "\n\n" +
                   r"$\bullet$ Retains 63.38% Total Train Variance" + "\n" +
                   r"$\bullet$ Zero validation or test data reuse",
              face_col='#FFFFFF', edge_col='#BAE6FD', text_col='#0369A1', font_size=7.4, font_weight='normal')

    draw_arrow(ax, 11.55, 5.20, 11.95, 5.20, color='#0284C7', lw=2.0, badge=r'$\mathbf{z} \in \mathbb{R}^3$', badge_col='#0284C7')

    # =========================================================================
    # COLUMN 4: 5-FOLD GROUP-CV BAGGED ENSEMBLE (x in [11.95, 15.15], w = 3.20)
    # =========================================================================
    c4_box = patches.FancyBboxPatch((11.95, 0.80), 3.20, 8.20, boxstyle="round,pad=0.10,rounding_size=0.18",
                                    facecolor='#FAF5FF', edgecolor='#7C3AED', lw=1.5, zorder=5)
    ax.add_patch(c4_box)

    draw_3d_block(ax, 12.05, 8.15, 3.00, 0.70, depth=0.18, face_col='#7C3AED', top_col='#A78BFA', side_col='#5B21B6')
    ax.text(13.55, 8.50, "4. 5-Fold Group-CV Ensemble", ha='center', va='center',
            color='white', fontsize=10.0, fontweight='bold', zorder=25)

    # Visual: 5 Parallel Model Tracks Converging into Averaging Operator
    tracks_y = [7.50, 6.85, 6.20, 5.55, 4.90]
    for idx, ty in enumerate(tracks_y):
        draw_pill(ax, 12.15, ty - 0.22, 1.85, 0.48,
                  text=f"Fold {idx+1}: w_{idx+1}, b_{idx+1} (C=0.01)",
                  face_col='#FFFFFF', edge_col='#C084FC', text_col='#6B21A8', font_size=7.2, font_weight='bold')
        ax.plot([14.00, 14.35], [ty, 6.20], color='#7C3AED', lw=1.2, zorder=15)

    # Central Averaging Aggregator Circle
    ax.add_patch(patches.Circle((14.50, 6.20), 0.28, facecolor='#7C3AED', edgecolor='#5B21B6', lw=1.4, zorder=20))
    ax.text(14.50, 6.20, r"$\frac{1}{5}\sum$", ha='center', va='center', color='white', fontsize=10.0, fontweight='bold', zorder=25)

    ax.text(13.55, 4.45, "Ensemble Aggregation:\n" + r"$\mathbf{P}_{\mathrm{Quest}} = \frac{1}{5}\sum_{k=1}^5 \sigma(\mathbf{w}_k^T \mathbf{z} + b_k)$",
            ha='center', va='center', color='#581C87', fontsize=7.6, fontweight='bold', zorder=25)

    # Stratification & Regularization Details Pill
    draw_pill(ax, 12.10, 1.05, 2.90, 3.10,
              text="Group-CV & Ensembling Rationale:\n\n" +
                   r"$\bullet$ Family-Cluster Stratification:" + "\n" +
                   r"  90 Family Clusters partitioned" + "\n" +
                   r"  (Co-twins strictly isolated)" + "\n\n" +
                   r"$\bullet$ Hyperparameter Optimization:" + "\n" +
                   r"  Best $C = 0.01$ (L2-penalty frozen)" + "\n" +
                   r"  (Train CV ROC-AUC = 0.5541)" + "\n\n" +
                   r"$\bullet$ Balanced Class Weighting:" + "\n" +
                   r"  Compensates ~1:6.5 class imbalance" + "\n\n" +
                   r"$\bullet$ Variance Stabilization:" + "\n" +
                   r"  Bagging dampens small $n=24$ event noise",
              face_col='#FFFFFF', edge_col='#DDD6FE', text_col='#4C1D95', font_size=7.3, font_weight='normal')

    draw_arrow(ax, 15.15, 5.20, 15.55, 5.20, color='#7C3AED', lw=2.0, badge=r'$\mathbf{P}_{\mathrm{Quest}}$', badge_col='#7C3AED')

    # =========================================================================
    # COLUMN 5: SIGMOID RISK FUNCTION & TEST EVALUATION (x in [15.55, 18.15], w = 2.60)
    # =========================================================================
    c5_box = patches.FancyBboxPatch((15.55, 0.80), 2.60, 8.20, boxstyle="round,pad=0.10,rounding_size=0.18",
                                    facecolor='#F8FAFC', edgecolor='#0F172A', lw=1.5, zorder=5)
    ax.add_patch(c5_box)

    draw_3d_block(ax, 15.65, 8.15, 2.40, 0.70, depth=0.18, face_col='#0F172A', top_col='#334155', side_col='#020617')
    ax.text(16.85, 8.50, "5. Risk Scoring & Test", ha='center', va='center',
            color='white', fontsize=10.0, fontweight='bold', zorder=25)

    # Visual: Real Plotted Sigmoid Logistic Activation Curve
    sig_ax_x0, sig_ax_y0 = 15.85, 5.70
    sig_w, sig_h = 2.05, 1.80

    # Sigmoid axes
    ax.plot([sig_ax_x0, sig_ax_x0 + sig_w], [sig_ax_y0, sig_ax_y0], color='#94A3B8', lw=1.0, zorder=12)
    ax.plot([sig_ax_x0, sig_ax_x0], [sig_ax_y0, sig_ax_y0 + sig_h], color='#94A3B8', lw=1.0, zorder=12)
    ax.text(sig_ax_x0 + sig_w, sig_ax_y0 - 0.12, r"$t$", ha='right', va='center', color='#64748B', fontsize=7.2, fontweight='bold', zorder=15)
    ax.text(sig_ax_x0 - 0.08, sig_ax_y0 + sig_h, r"$P$", ha='right', va='center', color='#64748B', fontsize=7.2, fontweight='bold', zorder=15)

    # Plot sigmoid curve
    t_vals = np.linspace(-3.5, 3.5, 50)
    sig_vals = 1.0 / (1.0 + np.exp(-t_vals))
    curve_x = sig_ax_x0 + (t_vals + 3.5) / 7.0 * sig_w
    curve_y = sig_ax_y0 + sig_vals * sig_h
    ax.plot(curve_x, curve_y, color='#16A34A', lw=2.2, zorder=18)

    # Decision threshold tau = 0.50 line (Natural logistic operating point)
    tau_y = sig_ax_y0 + 0.50 * sig_h
    ax.plot([sig_ax_x0, sig_ax_x0 + sig_w], [tau_y, tau_y], color='#D97706', lw=1.2, linestyle='--', zorder=19)
    ax.text(sig_ax_x0 + sig_w - 0.05, tau_y + 0.10, r"$\tau = 0.50$ (Screening Cutoff)", ha='right', va='bottom', color='#D97706', fontsize=6.8, fontweight='bold', zorder=22)

    # Points along curve
    ax.plot([sig_ax_x0 + 0.45, sig_ax_x0 + 0.75], [sig_ax_y0 + 0.22, sig_ax_y0 + 0.38], 'o', color='#3B82F6', markersize=5, zorder=20)
    ax.text(sig_ax_x0 + 0.40, sig_ax_y0 + 0.10, "Non-Trans (0)", ha='left', va='center', color='#1E40AF', fontsize=6.2, fontweight='bold', zorder=22)

    ax.plot([sig_ax_x0 + 1.45, sig_ax_x0 + 1.70], [sig_ax_y0 + 1.35, sig_ax_y0 + 1.55], 's', color='#EF4444', markersize=5, zorder=20)
    ax.text(sig_ax_x0 + 1.60, sig_ax_y0 + 1.70, "Incident (1)", ha='center', va='bottom', color='#B91C1C', fontsize=6.2, fontweight='bold', zorder=22)

    ax.text(16.85, 5.45, r"Risk: $\mathbf{P}_{\mathrm{Quest}} = \sigma(\mathbf{w}^T \mathbf{z} + b) \in [0, 1]$",
            ha='center', va='center', color='#0F172A', fontsize=7.6, fontweight='bold', zorder=25)

    # Performance Metrics Pill
    draw_pill(ax, 15.70, 2.75, 2.30, 2.45,
              text="Held-Out Prospective\nTest Performance ($n=62$):\n\n" +
                   r"$\mathrm{Test\ ROC\text{-}AUC} = 0.6260$" + "\n" +
                   r"$\mathrm{Sensitivity} = 71.4\%\ (5/7)$" + "\n" +
                   r"$\mathrm{Specificity} = 67.3\%\ (37/55)$" + "\n" +
                   r"$\mathrm{NPV} = 94.9\% \mid \tau = 0.50$" + "\n" +
                   r"$\mathrm{Brier\ Score} = 0.2460$",
              face_col='#F5F3FF', edge_col='#7C3AED', text_col='#4C1D95', font_size=7.4, font_weight='bold')

    # Deep Baseline Comparison Pill
    draw_pill(ax, 15.70, 1.05, 2.30, 1.55,
              text="Contrast: FT-ProtoNet (Fig 4)\n" +
                   r"$\mathrm{Val:\ 0.8052 \to Test:\ 0.5429}$" + "\n" +
                   r"$\Delta = -0.2623\ (\mathrm{Overfit})$" + "\n" +
                   "Confirms regularized linear\nhead generalizability",
              face_col='#FFF7ED', edge_col='#EA580C', text_col='#9A3412', font_size=7.1, font_weight='normal')

    save_figure_to_all_locations(fig, "Fig3_Questionnaire_Pipeline")
    plt.close(fig)
    print("  [OK] Fig3_Questionnaire_Pipeline successfully generated.")


# =========================================================================
# FIGURE 4: DETAILED DEEP PROTONET ARCHITECTURE
# (Full Mathematical Equation Typography & True Logical Flow from PyTorch Model)
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

    # COLUMN 1: RAW INPUT VECTOR
    draw_3d_block(ax, 0.40, 2.00, 0.85, 6.20, depth=0.25, angle=35,
                  face_col='#E2E8F0', top_col='#F8FAFC', side_col='#CBD5E1', edge_col='#94A3B8', zorder=10)
    ax.text(0.82, 5.10, r"$\mathbf{x} = [x_1, \dots, x_D]^T \in \mathbb{R}^D$", ha='center', va='center',
            color='#0F172A', fontsize=9.5, fontweight='bold', rotation=90, zorder=30)
    ax.text(0.82, 1.55, r"$D=9\ (\mathrm{M2})$" + "\n" + r"$D=4\ (\mathrm{M3})$", ha='center', va='center',
            color='#475569', fontsize=8.0, zorder=30)

    draw_arrow(ax, 1.45, 5.10, 1.95, 5.10, color='#0F172A', lw=1.8, zorder=15)

    # COLUMN 2: FEATURE TOKENIZER
    panel_tok = patches.FancyBboxPatch((1.95, 0.80), 2.75, 8.20, boxstyle="round,pad=0.15,rounding_size=0.2",
                                       facecolor='#F0F9FF', edgecolor='#0284C7', lw=1.4, zorder=5)
    ax.add_patch(panel_tok)
    ax.text(3.32, 8.65, "Feature Tokenizer", ha='center', va='center',
            color='#0369A1', fontsize=10.5, fontweight='bold', zorder=25)
    ax.text(3.32, 8.28, r"$\mathbf{t}_i = x_i \cdot \mathbf{w}_i + \mathbf{b}_i + \mathbf{e}_i \in \mathbb{R}^{16}$", ha='center', va='center',
            color='#0284C7', fontsize=8.5, fontweight='bold', zorder=25)

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

    ax.plot([4.35, 4.55], [7.35, 7.35], color='#0284C7', lw=1.4, zorder=15)
    ax.plot([4.55, 4.55], [2.95, 7.35], color='#0284C7', lw=1.4, zorder=15)
    ax.plot([4.35, 4.55], [2.95, 2.95], color='#0284C7', lw=1.4, zorder=15)

    ax.text(3.32, 1.45, r"$\mathbf{T} = [\mathbf{t}_1, \dots, \mathbf{t}_D]^T$" + "\n" + r"$\mathbf{T} \in \mathbb{R}^{D \times 16}$ (Token Matrix)", ha='center', va='center',
            color='#0369A1', fontsize=7.8, fontweight='bold', zorder=25)

    ax.plot([4.55, 5.05, 5.05, 5.65], [5.15, 5.15, 7.65, 7.65], color='#0284C7', lw=1.8, zorder=15)
    draw_arrow(ax, 5.45, 7.65, 5.75, 7.65, color='#0284C7', lw=1.8, zorder=15)
    bbox_tok_exit = dict(boxstyle="round,pad=0.25,rounding_size=0.18", facecolor='#0284C7', edgecolor='none')
    ax.text(5.35, 7.95, r"$\mathbf{T} \in \mathbb{R}^{D \times 16}$",
            ha='center', va='center', color='white', fontsize=7.6, fontweight='bold', bbox=bbox_tok_exit, zorder=25)

    # COLUMN 3: INTRA-SAMPLE TRANSFORMER
    panel_sa = patches.FancyBboxPatch((5.40, 0.80), 4.90, 8.20, boxstyle="round,pad=0.15,rounding_size=0.2",
                                      facecolor='#F0FDF4', edgecolor='#16A34A', lw=1.4, zorder=5)
    ax.add_patch(panel_sa)
    ax.text(7.85, 8.65, "Pre-LN Transformer Block", ha='center', va='center',
            color='#15803D', fontsize=10.5, fontweight='bold', zorder=25)
    ax.text(7.85, 8.30, r"$\mathrm{Intra\text{-}Sample\ Multi\text{-}Head\ Attention\ across\ D\ Features}$", ha='center', va='center',
            color='#166534', fontsize=7.8, fontstyle='italic', zorder=25)
    
    draw_branch_dot(ax, 5.75, 7.65, r=0.065, col='#14532D', zorder=25)

    ax.plot([5.75, 7.85], [7.65, 7.65], color='#16A34A', lw=1.8, zorder=15)
    draw_arrow(ax, 7.85, 7.65, 7.85, 6.95, color='#16A34A', lw=1.5, zorder=15)

    draw_pill(ax, 6.70, 6.95, 2.30, 0.45, text=r"$\tilde{\mathbf{T}} = \mathrm{LayerNorm}(\mathbf{T}) \in \mathbb{R}^{D \times 16}$",
              face_col='#FFFFFF', edge_col='#16A34A', text_col='#166534', font_size=7.4, font_weight='normal')

    draw_pill(ax, 6.05, 5.25, 3.60, 1.35,
              text="Multi-Head Attention (H = 4, d_k = 4):\n" +
                   r"$\mathbf{Q}, \mathbf{K}, \mathbf{V} = \tilde{\mathbf{T}} \mathbf{W}_{Q,K,V} \in \mathbb{R}^{D \times 16}$" + "\n" +
                   r"$\mathbf{A} = \mathrm{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V} \mathbf{W}_{\mathrm{proj}}$",
              face_col='#FFFFFF', edge_col='#16A34A', text_col='#166534', font_size=7.3, font_weight='normal')
    draw_arrow(ax, 7.85, 6.95, 7.85, 6.60, color='#16A34A', lw=1.5, zorder=15)
    draw_arrow(ax, 7.85, 5.25, 7.85, 4.95, color='#16A34A', lw=1.5, zorder=15)

    ax.plot([5.75, 5.75], [7.65, 4.75], color='#059669', lw=1.8, zorder=20)
    ax.annotate('', xy=(7.62, 4.75), xytext=(5.75, 4.75),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color='#059669', mutation_scale=12), zorder=20)
    ax.text(5.62, 6.15, r"$\mathrm{Residual\ Skip\ 1}$", ha='center', va='center',
            color='#047857', fontsize=7.0, fontweight='bold', rotation=90, zorder=25)
    ax.text(6.65, 4.95, r"$+\mathbf{T}$", ha='center', va='bottom', color='#047857', fontsize=7.6, fontweight='bold', zorder=25)

    draw_addition_circle(ax, 7.85, 4.75, r=0.20, face_col='#DCFCE7', edge_col='#16A34A', text_col='#14532D', zorder=25)
    ax.text(8.20, 4.75, r"$\mathbf{H}' = \mathbf{T} + \mathbf{A} \in \mathbb{R}^{D \times 16}$", ha='left', va='center',
            color='#14532D', fontsize=7.8, fontweight='bold', zorder=25)

    ax.plot([7.85, 7.85], [4.55, 4.25], color='#16A34A', lw=1.6, zorder=15)
    draw_branch_dot(ax, 7.85, 4.25, r=0.065, col='#14532D', zorder=25)

    draw_pill(ax, 6.70, 3.65, 2.30, 0.45, text=r"$\tilde{\mathbf{H}}' = \mathrm{LayerNorm}(\mathbf{H}') \in \mathbb{R}^{D \times 16}$",
              face_col='#FFFFFF', edge_col='#16A34A', text_col='#166534', font_size=7.4, font_weight='normal')
    draw_arrow(ax, 7.85, 4.25, 7.85, 4.10, color='#16A34A', lw=1.4, zorder=15)

    draw_pill(ax, 6.05, 1.95, 3.60, 1.35,
              text="Position-wise Feed-Forward Network:\n" +
                   r"$\mathbf{F} = \mathrm{GELU}(\tilde{\mathbf{H}}' \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$" + "\n" +
                   r"$(d_{\mathrm{model}} = 16 \to d_{\mathrm{ff}} = 64 \to 16)$",
              face_col='#FFFFFF', edge_col='#16A34A', text_col='#166534', font_size=7.3, font_weight='normal')
    draw_arrow(ax, 7.85, 3.65, 7.85, 3.30, color='#16A34A', lw=1.4, zorder=15)
    draw_arrow(ax, 7.85, 1.95, 7.85, 1.65, color='#16A34A', lw=1.4, zorder=15)

    ax.plot([7.85, 5.75], [4.25, 4.25], color='#059669', lw=1.8, zorder=20)
    ax.plot([5.75, 5.75], [4.25, 1.45], color='#059669', lw=1.8, zorder=20)
    ax.annotate('', xy=(7.62, 1.45), xytext=(5.75, 1.45),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color='#059669', mutation_scale=12), zorder=20)
    ax.text(5.62, 2.85, r"$\mathrm{Residual\ Skip\ 2}$", ha='center', va='center',
            color='#047857', fontsize=7.0, fontweight='bold', rotation=90, zorder=25)
    ax.text(6.65, 1.65, r"$+\mathbf{H}'$", ha='center', va='bottom', color='#047857', fontsize=7.6, fontweight='bold', zorder=25)

    draw_addition_circle(ax, 7.85, 1.45, r=0.20, face_col='#DCFCE7', edge_col='#16A34A', text_col='#14532D', zorder=25)
    ax.text(8.20, 1.45, r"$\mathbf{H} = \mathbf{H}' + \mathbf{F} \in \mathbb{R}^{D \times 16}$", ha='left', va='center',
            color='#14532D', fontsize=7.8, fontweight='bold', zorder=25)

    draw_arrow(ax, 10.30, 5.10, 10.70, 5.10, color='#16A34A', lw=1.8, zorder=15)

    # COLUMN 4: QUERY ATTENTION POOLING
    panel_pool = patches.FancyBboxPatch((10.70, 0.80), 3.15, 8.20, boxstyle="round,pad=0.15,rounding_size=0.2",
                                        facecolor='#FFFBEB', edgecolor='#D97706', lw=1.4, zorder=5)
    ax.add_patch(panel_pool)
    ax.text(12.27, 8.65, "Query Attention Pooling", ha='center', va='center',
            color='#B45309', fontsize=10.5, fontweight='bold', zorder=25)
    ax.text(12.27, 8.30, r"$\mathrm{Permutation\text{-}Invariant\ Aggregation}$", ha='center', va='center',
            color='#92400E', fontsize=7.8, fontstyle='italic', zorder=25)

    draw_pill(ax, 10.95, 6.25, 2.65, 1.70,
              text="Learnable Query Vector:\n" +
                   r"$\mathbf{q} \in \mathbb{R}^{16}\ (\mathrm{Trained\ Task\ Query})$" + "\n\n" +
                   r"$a_i = \frac{\exp(\mathbf{q}^T \mathbf{h}_i / \sqrt{d})}{\sum_{j=1}^D \exp(\mathbf{q}^T \mathbf{h}_j / \sqrt{d})}$",
              face_col='#FFFFFF', edge_col='#D97706', text_col='#92400E', font_size=7.5, font_weight='normal')

    draw_arrow(ax, 12.27, 6.25, 12.27, 5.65, color='#D97706', lw=1.5, zorder=15)

    draw_pill(ax, 10.95, 4.35, 2.65, 1.30,
              text="Context Vector (Weighted Sum):\n\n" +
                   r"$\mathbf{c}_{\mathrm{pool}} = \sum_{i=1}^D a_i \mathbf{h}_i \in \mathbb{R}^{16}$",
              face_col='#FFFFFF', edge_col='#D97706', text_col='#92400E', font_size=7.6, font_weight='normal')

    draw_arrow(ax, 12.27, 4.35, 12.27, 3.85, color='#D97706', lw=1.5, zorder=15)

    draw_pill(ax, 10.95, 1.65, 2.65, 2.20,
              text=r"L2-Hypersphere Projection:" + "\n\n" +
                   r"$\mathbf{z} = \frac{\mathbf{c}_{\mathrm{pool}}}{\|\mathbf{c}_{\mathrm{pool}}\|_2} \in \mathbb{S}^{15}$" + "\n\n" +
                   r"$\|\mathbf{z}\|_2 = 1.0\ (\mathrm{Unit\ Norm})$" + "\n" +
                   r"$\bullet\ \mathrm{Eliminates\ token\ scale\ shift}$" + "\n" +
                   r"$\bullet\ \mathrm{Enforces\ spherical\ metric}$",
              face_col='#FEF3C7', edge_col='#D97706', text_col='#78350F', font_size=7.5, font_weight='normal')

    draw_arrow(ax, 13.85, 2.65, 14.35, 2.65, color='#D97706', lw=1.8, badge=r'$\mathbf{z} \in \mathbb{S}^{15}$', badge_col='#D97706', zorder=20)

    # COLUMN 5: EPISODIC HEAD
    panel_head = patches.FancyBboxPatch((14.35, 0.80), 3.00, 8.20, boxstyle="round,pad=0.15,rounding_size=0.2",
                                        facecolor='#FAF5FF', edgecolor='#9333EA', lw=1.6, zorder=5)
    ax.add_patch(panel_head)
    ax.text(15.85, 8.65, "Episodic Few-Shot Metric Head", ha='center', va='center',
            color='#7E22CE', fontsize=10.2, fontweight='bold', zorder=25)

    draw_sphere(ax, 14.85, 7.65, r=0.26, base_col='#3B82F6', highlight_col='#93C5FD', edge_col='#1D4ED8', zorder=25)
    ax.text(15.25, 7.65, r"$\mathbf{c}_0\ (\mathrm{Non\text{-}Transitioned})$", ha='left', va='center',
            color='#1E40AF', fontsize=7.6, fontweight='bold', zorder=25)

    draw_sphere(ax, 14.85, 6.85, r=0.26, base_col='#EF4444', highlight_col='#FCA5A5', edge_col='#B91C1C', zorder=25)
    ax.text(15.25, 6.85, r"$\mathbf{c}_1\ (\mathrm{Incident\ Anxiety})$", ha='left', va='center',
            color='#991B1B', fontsize=7.6, fontweight='bold', zorder=25)

    draw_pill(ax, 14.50, 4.85, 2.70, 1.75,
              text="12-Shot Support Prototypes:\n\n" +
                   r"$\mathbf{c}_k^* = \frac{\sum_{i \in S_k^*} \mathbf{z}_i}{\|\sum_{i \in S_k^*} \mathbf{z}_i\|_2}\ (|S_0^*|=12, |S_1^*|=12)$" + "\n\n" +
                   r"Identical 12-shot prototype rule across" + "\n" +
                   r"training, val checkpointing & test inference",
              face_col='#FFFFFF', edge_col='#9333EA', text_col='#6B21A8', font_size=7.1, font_weight='normal')

    draw_arrow(ax, 15.85, 4.85, 15.85, 4.25, color='#9333EA', lw=1.4, zorder=20)

    draw_pill(ax, 14.50, 2.45, 2.70, 1.80,
              text=r"Cosine Scoring on Test Query $\mathbf{z}^*$:" + "\n\n" +
                   r"$s_k(\mathbf{z}^*) = \frac{\mathbf{z}^{*T} \mathbf{c}_k^*}{\tau} = \frac{\cos(\mathbf{z}^*, \mathbf{c}_k^*)}{\tau}$" + "\n\n" +
                   r"$P(y = 1 \mid \mathbf{x}^*) = \frac{\exp(s_1)}{\exp(s_0) + \exp(s_1)}$" + "\n" +
                   r"$(\tau \in [0.05, 1.0]\ \mathrm{learnable})$",
              face_col='#F3E8FF', edge_col='#9333EA', text_col='#581C87', font_size=7.2, font_weight='normal')

    draw_pill(ax, 14.50, 0.90, 2.70, 1.35,
              text="Episodic Training Loss:\n" +
                   r"$\mathcal{L}_{\mathrm{meta}} = -\frac{1}{|Q|} \sum_{q \in Q} \log P(y_q \mid \mathbf{x}_q; \mathbf{c}_k)$" + "\n" +
                   r"Query $Q$: $12\ \mathrm{cases} : 144\ \mathrm{non\text{-}trans}\ (1:12)$" + "\n" +
                   r"Directly trains prototypes $\mathbf{c}_k^*$ used at test",
              face_col='#7E22CE', edge_col='#581C87', text_col='#FFFFFF', font_size=6.9, font_weight='bold')

    draw_arrow(ax, 17.15, 3.35, 17.55, 3.35, color='#0F172A', lw=1.8, zorder=20)

    # COLUMN 6: OUTPUT PREDICTION
    draw_3d_block(ax, 17.55, 2.00, 0.75, 6.20, depth=0.25, angle=35,
                  face_col='#E2E8F0', top_col='#F8FAFC', side_col='#CBD5E1', edge_col='#94A3B8', zorder=10)
    ax.text(17.92, 5.10, r"$\mathbf{P}_{\mathrm{Quest}} = P(y = 1 \mid \mathbf{x}) \in [0, 1]$", ha='center', va='center',
            color='#0F172A', fontsize=9.2, fontweight='bold', rotation=90, zorder=30)

    save_figure_to_all_locations(fig, "Fig4_Prototypical_Network")
    plt.close(fig)
    print("  [OK] Fig4_Prototypical_Network successfully generated.")


# =========================================================================
# FIGURE 5: MULTIMODAL LATE FUSION ARCHITECTURE & DUAL THRESHOLD TRIAGE
# =========================================================================
def build_fig5():
    print("Generating Figure 5: Multimodal Late Fusion Architecture & Prospective Triage...")
    fig, ax = plt.subplots(figsize=(18.8, 9.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 18.8)
    ax.set_ylim(0, 9.8)
    ax.axis('off')

    # Main Title & Subtitle
    ax.text(9.40, 9.55, "Figure 5: Multimodal Late Fusion Architecture & Dual Operational Threshold Calibration",
            ha='center', va='center', color='#0F172A', fontsize=13.8, fontweight='bold')
    ax.text(9.40, 9.24, r"Heterogeneous Unimodal Inputs $\to$ Non-Parametric Rank Transformation $\to$ Parameter-Free Ensembling $\to$ Dual Operational Triage",
            ha='center', va='center', color='#475569', fontsize=9.2, fontstyle='italic')

    # =========================================================================
    # COLUMN 1: UNIMODAL PREDICTIVE STREAMS (x in [0.45, 3.75], w = 3.30)
    # =========================================================================
    c1_box = patches.FancyBboxPatch((0.45, 0.80), 3.30, 8.20, boxstyle="round,pad=0.10,rounding_size=0.18",
                                    facecolor='#F8FAFC', edgecolor='#1F618D', lw=1.5, zorder=5)
    ax.add_patch(c1_box)

    draw_3d_block(ax, 0.55, 8.15, 3.10, 0.70, depth=0.18, face_col='#1F618D', top_col='#5DADE2', side_col='#154360')
    ax.text(2.10, 8.50, "1. Unimodal Model Streams", ha='center', va='center',
            color='white', fontsize=10.0, fontweight='bold', zorder=25)

    # Stream Cards with Icons
    streams = [
        ("M1: Structural MRI (sMRI)", 6.55, "#1E3A8A", "#DBEAFE", 'brain',
         r"$\mathbf{P}_{\mathrm{MRI}} \in [0, 1]$" + "\n" +
         "Med3D ResNet-18 + PCA-5 + Balanced LR\n" +
         r"$\mathrm{Test\ AUC} = 0.7221 \mid \mathrm{Brier} = 0.2266$"),
        ("M2: Anxiety Questionnaires", 5.05, "#065F46", "#D1FAE5", 'clip',
         r"$\mathbf{P}_{\mathrm{Quest}} \in [0, 1]$" + "\n" +
         "9 Transition Markers + PCA-3 + 5-Fold Bagged\n" +
         r"$\mathrm{Test\ AUC} = 0.6260 \mid \mathrm{Brier} = 0.2460$"),
        ("M3: Static Phenotypic Traits", 3.55, "#78350F", "#FEF3C7", 'user',
         r"$\mathbf{P}_{\mathrm{Pheno}} \in [0, 1]$" + "\n" +
         "4 Environmental/Pubertal Traits + Balanced LR\n" +
         r"$\mathrm{Test\ AUC} = 0.6870 \mid \mathrm{Brier} = 0.2238$")
    ]

    for s_title, sy, col_dark, col_bg, icon_t, s_desc in streams:
        card = patches.FancyBboxPatch((0.60, sy), 3.00, 1.35, boxstyle="round,pad=0.06,rounding_size=0.10",
                                      facecolor='#FFFFFF', edgecolor=col_dark, lw=1.3, zorder=10)
        ax.add_patch(card)
        banner = patches.FancyBboxPatch((0.60, sy + 1.02), 3.00, 0.33, boxstyle="round,pad=0.02,rounding_size=0.06",
                                        facecolor=col_dark, edgecolor='none', zorder=12)
        ax.add_patch(banner)
        if icon_t == 'brain':
            draw_brain_vector(ax, 0.85, sy + 1.18, s=0.17, col='white', zorder=25)
        elif icon_t == 'clip':
            draw_clipboard_vector(ax, 0.85, sy + 1.18, s=0.17, col='white', zorder=25)
        elif icon_t == 'user':
            draw_user_vector(ax, 0.85, sy + 1.18, s=0.16, col='white', zorder=25)
        ax.text(2.18, sy + 1.18, s_title, ha='center', va='center', color='white', fontsize=7.2, fontweight='bold', zorder=20)
        ax.text(2.10, sy + 0.50, s_desc, ha='center', va='center', color='#1E293B', fontsize=6.8, zorder=20)

    # Unimodal Calibration Challenge Card
    draw_pill(ax, 0.60, 1.05, 3.00, 2.30,
              text="Calibration Scale Heterogeneity:\n\n" +
                   r"$\bullet$ Disparate Margins: Sigmoid probabilities" + "\n" +
                   r"  exhibit differing variances & slopes" + "\n" +
                   r"$\bullet$ Direct Probability Averaging ($1/3$):" + "\n" +
                   r"  Subject to scale compression & drift" + "\n" +
                   r"$\bullet$ Rationale: Motivates scale-invariant" + "\n" +
                   r"  non-parametric rank transformation",
              face_col='#FFFFFF', edge_col='#CBD5E1', text_col='#475569', font_size=6.9, font_weight='normal')

    draw_arrow(ax, 3.75, 5.20, 4.10, 5.20, color='#1F618D', lw=2.0, badge=r'$\mathbf{p} \in [0,1]^3$', badge_col='#1F618D')

    # =========================================================================
    # COLUMN 2: TRANSFORMATION & NORMALIZATION ENGINE (x in [4.10, 7.50], w = 3.40)
    # =========================================================================
    c2_box = patches.FancyBboxPatch((4.10, 0.80), 3.40, 8.20, boxstyle="round,pad=0.10,rounding_size=0.18",
                                    facecolor='#F0FDF4', edgecolor='#059669', lw=1.5, zorder=5)
    ax.add_patch(c2_box)

    draw_3d_block(ax, 4.20, 8.15, 3.20, 0.70, depth=0.18, face_col='#059669', top_col='#34D399', side_col='#065F46')
    ax.text(5.80, 8.50, "2. Rank Normalization Engine", ha='center', va='center',
            color='white', fontsize=10.0, fontweight='bold', zorder=25)

    # Visual: Percentile Rank Mapping Transformation
    rank_box = patches.FancyBboxPatch((4.25, 4.95), 3.10, 2.95, boxstyle="round,pad=0.08,rounding_size=0.12",
                                      facecolor='#FFFFFF', edgecolor='#059669', lw=1.3, zorder=10)
    ax.add_patch(rank_box)
    ax.text(5.80, 7.65, "Non-Parametric Rank Transformation", ha='center', va='center',
            color='#065F46', fontsize=8.2, fontweight='bold', zorder=20)
    ax.text(5.80, 7.28, r"$r_m(i) = \frac{\mathrm{rank}(p_m(i))}{N} \in [0, 1]$", ha='center', va='center',
            color='#0F172A', fontsize=8.4, fontweight='bold', zorder=20)

    # Miniature graphic of 3 rank ladders mapping to uniform [0, 1]
    ax.plot([4.60, 4.60], [5.30, 6.90], color='#1E3A8A', lw=2.0, zorder=15)
    ax.plot([5.80, 5.80], [5.30, 6.90], color='#065F46', lw=2.0, zorder=15)
    ax.plot([7.00, 7.00], [5.30, 6.90], color='#D97706', lw=2.0, zorder=15)
    ax.text(4.60, 5.15, r"$r_1\ (\mathrm{sMRI})$", ha='center', va='top', color='#1E3A8A', fontsize=6.8, fontweight='bold', zorder=20)
    ax.text(5.80, 5.15, r"$r_2\ (\mathrm{Quest})$", ha='center', va='top', color='#065F46', fontsize=6.8, fontweight='bold', zorder=20)
    ax.text(7.00, 5.15, r"$r_3\ (\mathrm{Pheno})$", ha='center', va='top', color='#B45309', fontsize=6.8, fontweight='bold', zorder=20)

    for lx, col in [(4.60, '#1E3A8A'), (5.80, '#065F46'), (7.00, '#D97706')]:
        for tick in np.linspace(5.45, 6.75, 6):
            ax.plot([lx - 0.08, lx + 0.08], [tick, tick], color=col, lw=1.0, zorder=16)

    # Transformation Mathematical Rationale Card
    draw_pill(ax, 4.25, 1.05, 3.10, 3.65,
              text="Mathematical Rationale for Rank Fusion:\n\n" +
                   r"$\bullet$ Uniform Marginal Alignment:" + "\n" +
                   r"  Maps arbitrary calibration geometries" + "\n" +
                   r"  into commensurate uniform quantiles $\mathcal{U}[0, 1]$" + "\n\n" +
                   r"$\bullet$ Monotonic Invariance:" + "\n" +
                   r"  Strictly preserves ROC curve orderings;" + "\n" +
                   r"  immune to monotonic logit distortion" + "\n\n" +
                   r"$\bullet$ Outlier & Extremum Dampening:" + "\n" +
                   r"  Prevents over-confident single-modality" + "\n" +
                   r"  probabilities from dominating fusion" + "\n\n" +
                   r"$\bullet$ Zero Parameter Estimation Error:" + "\n" +
                   r"  Eliminates fragile weight tuning on $n=7$ cases",
              face_col='#FFFFFF', edge_col='#A7F3D0', text_col='#065F46', font_size=7.0, font_weight='normal')

    draw_arrow(ax, 7.50, 5.20, 7.85, 5.20, color='#059669', lw=2.0, badge=r'$\mathbf{r} \in [0, 1]^3$', badge_col='#059669')

    # =========================================================================
    # COLUMN 3: MULTI-PARADIGM ENSEMBLE ENGINE (x in [7.85, 11.40], w = 3.55)
    # =========================================================================
    c3_box = patches.FancyBboxPatch((7.85, 0.80), 3.55, 8.20, boxstyle="round,pad=0.10,rounding_size=0.18",
                                    facecolor='#F0F9FF', edgecolor='#0284C7', lw=1.5, zorder=5)
    ax.add_patch(c3_box)

    draw_3d_block(ax, 7.95, 8.15, 3.35, 0.70, depth=0.18, face_col='#0284C7', top_col='#38BDF8', side_col='#0369A1')
    ax.text(9.625, 8.50, "3. Multimodal Ensembling", ha='center', va='center',
            color='white', fontsize=10.0, fontweight='bold', zorder=25)

    # Primary Winner: Multimodal Rank Averaging (Blue Border Box)
    p_rank = patches.FancyBboxPatch((7.98, 5.45), 3.28, 2.45, boxstyle="round,pad=0.08,rounding_size=0.12",
                                    facecolor='#EFF6FF', edgecolor='#2563EB', lw=1.6, zorder=10)
    ax.add_patch(p_rank)
    ax.text(9.625, 7.62, "PRIMARY: Multimodal Rank Averaging", ha='center', va='center',
            color='#1E40AF', fontsize=8.2, fontweight='bold', zorder=20)
    ax.text(9.625, 7.25, r"$P_{\mathrm{Rank}} = \frac{1}{3} r_1 + \frac{1}{3} r_2 + \frac{1}{3} r_3$", ha='center', va='center',
            color='#0F172A', fontsize=8.4, fontweight='bold', zorder=20)

    draw_pill(ax, 8.12, 5.60, 3.00, 1.35,
              text="Held-Out Prospective Test Performance:\n" +
                   r"$\mathbf{Test\ ROC\text{-}AUC} = \mathbf{0.8091}$" + "\n" +
                   r"$95\%\ \mathrm{Bootstrap\ CI}: [0.6597,\ 0.9306]$" + "\n" +
                   r"$\mathrm{PR\text{-}AUC} = 0.3242 \mid \mathrm{Brier} = 0.0820$",
              face_col='#FFFFFF', edge_col='#93C5FD', text_col='#1E3A8A', font_size=7.2, font_weight='bold')

    # Secondary Paradigms Container
    sec_box = patches.FancyBboxPatch((7.98, 3.40), 3.28, 1.85, boxstyle="round,pad=0.06,rounding_size=0.10",
                                     facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=1.1, zorder=10)
    ax.add_patch(sec_box)
    ax.text(9.625, 5.02, "Alternative Zero-Leakage Paradigms", ha='center', va='center',
            color='#334155', fontsize=7.6, fontweight='bold', zorder=20)
    ax.text(9.625, 4.25,
            r"$\bullet$ Dawes' Equal Weighting: $P = \frac{1}{3}(p_1+p_2+p_3)$" + "\n" +
            r"  $\mathrm{Test\ AUC} = 0.7792\ [0.5578,\ 0.9404]$" + "\n" +
            r"$\bullet$ Train-OOF Stacking Meta-Learner (Ridge):" + "\n" +
            r"  $P = \sigma(\beta_0 + \mathbf{\beta}^T \mathbf{p}) \to \mathrm{Test\ AUC} = 0.8156$",
            ha='center', va='center', color='#1E293B', fontsize=6.8, zorder=20)

    # Parameterized Overfitting Contrast Pill
    draw_pill(ax, 7.98, 1.05, 3.28, 2.15,
              text="Contrast: Parameterized Overfitting Baselines:\n\n" +
                   r"$\bullet$ 186-Point Grid Search (Validation-Tuned):" + "\n" +
                   r"  $\mathrm{Val:\ 0.8524 \to Test:\ 0.6390}\ (\Delta = -0.2134)$" + "\n" +
                   r"$\bullet$ Cross-Modal Attention: $\mathrm{Test\ AUC} = 0.5896$" + "\n" +
                   "Confirms parameter-free fusion superiority\nunder small validation events ($n=7$)",
              face_col='#FFF7ED', edge_col='#EA580C', text_col='#9A3412', font_size=6.8, font_weight='normal')

    draw_arrow(ax, 11.40, 5.20, 11.75, 5.20, color='#0284C7', lw=2.0, badge=r'$P_{\mathrm{Rank}}$', badge_col='#0284C7')

    # =========================================================================
    # COLUMN 4: DUAL OPERATING THRESHOLDS (x in [11.75, 14.95], w = 3.20)
    # =========================================================================
    c4_box = patches.FancyBboxPatch((11.75, 0.80), 3.20, 8.20, boxstyle="round,pad=0.10,rounding_size=0.18",
                                    facecolor='#FAF5FF', edgecolor='#7C3AED', lw=1.5, zorder=5)
    ax.add_patch(c4_box)

    draw_3d_block(ax, 11.85, 8.15, 3.00, 0.70, depth=0.18, face_col='#7C3AED', top_col='#A78BFA', side_col='#5B21B6')
    ax.text(13.35, 8.50, "4. Dual Threshold Calibration", ha='center', va='center',
            color='white', fontsize=10.0, fontweight='bold', zorder=25)

    # Policy A: Screening Threshold tau = 0.50
    p_a = patches.FancyBboxPatch((11.90, 5.05), 2.90, 2.85, boxstyle="round,pad=0.08,rounding_size=0.12",
                                 facecolor='#FFFFFF', edgecolor='#059669', lw=1.4, zorder=10)
    ax.add_patch(p_a)
    ax.text(13.35, 7.62, r"A. Screening Cutoff ($\tau = 0.50$)", ha='center', va='center',
            color='#065F46', fontsize=8.0, fontweight='bold', zorder=20)
    ax.text(13.35, 7.30, "Public Health High-Sensitivity Rule-Out", ha='center', va='center',
            color='#047857', fontsize=6.8, fontstyle='italic', zorder=20)

    draw_pill(ax, 12.00, 5.20, 2.70, 1.85,
              text=r"$\mathbf{Sensitivity} = \mathbf{100.0\%\ (7/7\ Detected)}$" + "\n" +
                   r"$\mathbf{False\ Negatives} = \mathbf{0}\ (0.0\%)$" + "\n" +
                   r"$\mathrm{Specificity} = 52.7\%\ (29/55\ \mathrm{cleared})$" + "\n" +
                   r"$\mathbf{NPV} = \mathbf{100.0\%} \mid \mathrm{Brier} = 0.0820$" + "\n" +
                   r"$\mathrm{False\ Alerts} = 26\ (47.3\%)$",
              face_col='#ECFDF5', edge_col='#10B981', text_col='#065F46', font_size=7.0, font_weight='bold')

    # Policy B: Diagnostic Threshold tau* = 0.564
    p_b = patches.FancyBboxPatch((11.90, 1.85), 2.90, 2.95, boxstyle="round,pad=0.08,rounding_size=0.12",
                                 facecolor='#FFFFFF', edgecolor='#D97706', lw=1.4, zorder=10)
    ax.add_patch(p_b)
    ax.text(13.35, 4.52, r"B. Diagnostic Cutoff ($\tau^* = 0.56$)", ha='center', va='center',
            color='#B45309', fontsize=8.0, fontweight='bold', zorder=20)
    ax.text(13.35, 4.20, "Validation-Locked Balanced Triage", ha='center', va='center',
            color='#92400E', fontsize=6.8, fontstyle='italic', zorder=20)

    draw_pill(ax, 12.00, 2.00, 2.70, 1.95,
              text=r"$\mathbf{Sensitivity} = \mathbf{85.71\%\ (6/7\ Detected)}$" + "\n" +
                   r"$\mathbf{Specificity} = \mathbf{63.64\%\ (35/55\ Cleared)}$" + "\n" +
                   r"$\mathbf{NPV} = \mathbf{97.22\%} \mid \mathrm{PPV} = 23.08\%$" + "\n" +
                   r"$\mathrm{Prevalence\text{-}Adj.\ PPV} = 25.19\%$" + "\n" +
                   r"$\mathrm{Balanced}\ F_1 = 0.3636$",
              face_col='#FFFBEB', edge_col='#F59E0B', text_col='#92400E', font_size=7.0, font_weight='bold')

    # Policy Banner at Bottom
    draw_pill(ax, 11.90, 1.05, 2.90, 0.65,
              text="Zero Threshold Search on Test Cohort",
              face_col='#FAF5FF', edge_col='#C084FC', text_col='#6B21A8', font_size=7.0, font_weight='bold')

    draw_arrow(ax, 14.95, 5.20, 15.35, 5.20, color='#7C3AED', lw=2.0, badge=r'$\tau^*=0.56$', badge_col='#7C3AED')

    # =========================================================================
    # COLUMN 5: PROSPECTIVE CLINICAL TRIAGE MATRIX (x in [15.35, 18.35], w = 3.00)
    # =========================================================================
    c5_box = patches.FancyBboxPatch((15.35, 0.80), 3.00, 8.20, boxstyle="round,pad=0.10,rounding_size=0.18",
                                    facecolor='#F8FAFC', edgecolor='#0F172A', lw=1.5, zorder=5)
    ax.add_patch(c5_box)

    draw_3d_block(ax, 15.45, 8.15, 2.80, 0.70, depth=0.18, face_col='#0F172A', top_col='#334155', side_col='#020617')
    ax.text(16.85, 8.50, "5. Prospective Triage ($n=62$)", ha='center', va='center',
            color='white', fontsize=9.4, fontweight='bold', zorder=25)

    # Visual: Authentic Streamlined Confusion Matrix
    cm_x0, cm_y0, cm_w, cm_h = 15.55, 5.35, 2.60, 2.45
    cm_card = patches.FancyBboxPatch((cm_x0, cm_y0), cm_w, cm_h, boxstyle="round,pad=0.06,rounding_size=0.10",
                                     facecolor='#FFFFFF', edgecolor='#0F172A', lw=1.2, zorder=10)
    ax.add_patch(cm_card)

    ax.text(cm_x0 + cm_w/2, cm_y0 + cm_h - 0.22, r"Test Confusion Matrix ($\tau^* = 0.56$)",
            ha='center', va='center', color='#0F172A', fontsize=7.6, fontweight='bold', zorder=20)

    # Matrix Cells (35, 20, 1, 6)
    cells = [
        (cm_x0 + 0.25, cm_y0 + 1.05, 1.05, 0.85, '#DBEAFE', '35\n(63.6%)', 'TN', '#1E40AF'),
        (cm_x0 + 1.35, cm_y0 + 1.05, 1.05, 0.85, '#F1F5F9', '20\n(36.4%)', 'FP', '#475569'),
        (cm_x0 + 0.25, cm_y0 + 0.15, 1.05, 0.80, '#FEF2F2', '1\n(14.3%)', 'FN', '#DC2626'),
        (cm_x0 + 1.35, cm_y0 + 0.15, 1.05, 0.80, '#DCFCE7', '6\n(85.7%)', 'TP', '#15803D')
    ]
    for cx, cy, cw, ch, bg, txt, tag, txt_c in cells:
        ax.add_patch(patches.Rectangle((cx, cy), cw, ch, facecolor=bg, edgecolor='#CBD5E1', lw=0.8, zorder=12))
        ax.text(cx + cw/2, cy + ch/2 + 0.08, txt, ha='center', va='center', color=txt_c, fontsize=7.6, fontweight='bold', zorder=20)
        ax.text(cx + cw - 0.08, cy + 0.08, tag, ha='right', va='bottom', color='#94A3B8', fontsize=6.2, fontweight='bold', zorder=20)

    # Axis labels for matrix
    ax.text(cm_x0 + 0.77, cm_y0 + 1.98, "Non-Trans", ha='center', va='center', color='#64748B', fontsize=6.6, fontweight='bold', zorder=20)
    ax.text(cm_x0 + 1.87, cm_y0 + 1.98, "Incident", ha='center', va='center', color='#64748B', fontsize=6.6, fontweight='bold', zorder=20)
    ax.text(cm_x0 + 0.12, cm_y0 + 1.48, "0", ha='center', va='center', color='#64748B', fontsize=6.6, fontweight='bold', zorder=20)
    ax.text(cm_x0 + 0.12, cm_y0 + 0.55, "1", ha='center', va='center', color='#64748B', fontsize=6.6, fontweight='bold', zorder=20)

    # Clinical Impact & NNS Card
    draw_pill(ax, 15.55, 3.05, 2.60, 2.10,
              text="Public Health Utility:\n\n" +
                   r"$\bullet$ Number Needed to Screen:" + "\n" +
                   r"  $\mathrm{NNS} \approx 9.3 \approx 9\ \mathrm{youth}$" + "\n" +
                   r"$\bullet$ Pre-Filtering Efficiency:" + "\n" +
                   r"  Clears 63.6% of healthy youth" + "\n" +
                   r"$\bullet$ Doubled Risk Concentration:" + "\n" +
                   r"  $\mathrm{Prevalence:\ 12.5\% \to 25.2\%}$",
              face_col='#FFFFFF', edge_col='#94A3B8', text_col='#0F172A', font_size=6.9, font_weight='normal')

    # Statistical Multiplicity Verification Badge
    draw_pill(ax, 15.55, 1.05, 2.60, 1.85,
              text="Statistical Multiplicity Audit:\n(5,000 Family-Cluster Bootstrap)\n\n" +
                   r"$\bullet$ vs.\ Chance ($\mathcal{H}_0: \text{AUC} \leq 0.50$):" + "\n" +
                   r"  $p = 0.0020\ (**\ \mathrm{FDR}\ q=0.0160)$" + "\n" +
                   r"$\bullet$ vs.\ sMRI Alone:" + "\n" +
                   r"  $p = 0.4544\ (\mathrm{Non\text{-}Significant})$",
              face_col='#F1F5F9', edge_col='#64748B', text_col='#334155', font_size=6.8, font_weight='bold')

    save_figure_to_all_locations(fig, "Fig5_Multimodal_Fusion")
    plt.close(fig)
    print("  [OK] Fig5_Multimodal_Fusion successfully generated.")


if __name__ == "__main__":
    print("=" * 70)
    print("REGENERATING ALL 5 PUBLICATION ARCHITECTURE FIGURES (300 DPI)")
    print("=" * 70)
    build_fig1()
    build_fig2()
    build_fig3()
    build_fig4()
    build_fig5()
    print("=" * 70)
    print("ALL 5 FIGURES SUCCESSFULLY REGENERATED IN ALL LOCATIONS")
    print("=" * 70)
