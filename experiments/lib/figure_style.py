"""Figure layout constants.

Edit THIS file first when tweaking fonts, spacing, markers, or colors.
Nature Communications double-column width is handled by figsize_double_col()
in experiments.lib.figures.nc_style (183 mm @ 300 dpi).
All main figures share Times New Roman and the same base font sizes.
"""

from __future__ import annotations

# =============================================================================
# Global typography (all main / ExtData boards)
# =============================================================================
FONT_FAMILY = "Times New Roman"
FS = 7  # base body font (axes labels)
FS_TITLE = 8  # panel letter + short title (e.g. "a  Winter-…")
FS_LEGEND = FS  # legends match axis-label size
FS_TICK_COMPOSED = 6  # tick labels on composed (non-map) boards

# Map boards use the same family/size family as composed boards.
MAP_FS_LAB = FS
MAP_FS_TICK = FS_TICK_COMPOSED
MAP_FS_TAG = FS
MAP_FS_CLABEL = 5.0
MAP_FS_ROW_TITLE = FS_TITLE

# =============================================================================
# Contour board layout (Fig1 / Fig5 shared outer gridspec)
# =============================================================================
MAP_LEFT = 0.07
MAP_RIGHT = 0.90
MAP_TOP = 0.97
MAP_HSPACE = 0.12
MAP_TITLE_HEIGHT = 0.10
MAP_TITLE_HSPACE = 0.03
MAP_WSPACE = 0.05
MAP_CBAR_X = 0.915  # fallback; builders place colour bars from rightmost map
MAP_CBAR_W = 0.012
MAP_CBAR_PAD = 0.008
MAP_FIG_EXTRA = 0.52
MAP_FIG5_BOTTOM = 0.19
MAP_FIG5_EXTRA = 0.72
MAP_LEGEND_Y0 = 0.020
MAP_LEGEND_H = 0.050
MAP_FIG1_BOTTOM = 0.18

# =============================================================================
# Contour field / isolines
# =============================================================================
PHASE_VSHOW = 8.0
PHASE_LEVEL_STEP = 1.0
PHASE_CONTOUR_COLOR = "0.25"
PHASE_CONTOUR_LW = 0.35
MAE_N_LEVELS = 9
MAE_CONTOUR_COLOR = "0.35"
MAE_CONTOUR_LW = 0.3
FILL_ALPHA = 0.92

# =============================================================================
# Markers
# =============================================================================
FIG1_STATE_DOT_S = 4.0 * (2.0 / 3.0)  # all hubs including reference
FIG1_REF_BORDER_LW = 1.20  # 4× default CONUS border (0.30); thicker for NY visibility
FIG1_REF_BORDER_COLOR = "0.05"
FIG1_REF_SQUARE_S = 12.0  # legacy bullseye helper only (not used on Fig1)
FIG2_DOT_S = 18
FIG2_LEGEND_MS = 5.5
FIG5_STAR_S = 9.0  # 50% of previous 18
FIG5_STAR_EDGE = "0.15"
FIG5_STAR_EDGE_LW = 0.30
FIG5_META_COLOR = "#111111"
FIG5_LEGEND_MARKERSIZE = 5.5
FIG5_LEGEND_EDGE_LW = 0.35
FIG5_LEGEND_FONTSCALE = 1.0
FIG5_REF_BORDER_LW = 1.20  # Texas emphasis (4× CONUS)
FIG5_REF_BORDER_COLOR = "0.05"

# =============================================================================
# Portfolio / scenario display names (short NC style)
# =============================================================================
FCFS = "Price-independent"
FULL = "Price-responsive"
NODC = "Without data centers"
COLOR_FCFS = "#3C5488"
COLOR_FULL = "#E64B35"
COLOR_NODC = "#8491B4"

# =============================================================================
# Fig2 / Fig3 / Fig4 layout knobs (kept for other builders)
# =============================================================================
FIG2_LEFT = 0.10
FIG2_RIGHT = 0.98
FIG2_TOP = 0.92
FIG2_BOTTOM = 0.10
FIG2_WSPACE = 0.28
FIG2_HSPACE = 0.35

FIG3_LEFT = 0.05
FIG3_RIGHT = 0.90  # room for vertical loading colour bar
FIG3_TOP = 0.985
FIG3_BOTTOM = 0.07
FIG3_OUTER_HSPACE = 0.02
FIG3_MAP_WSPACE = 0.04
FIG3_MAP_HSPACE = 0.05
FIG3_LINE_WSPACE = 0.12
# Two map rows + line row; g/h ≈ 2/3 of prior 0.72 weight.
FIG3_MAP_LINE_HRATIO = (1.0, 1.0, 0.36)
FIG3_HEIGHT_IN = None  # computed from Texas map aspect in build_fig3
FIG3_PANEL_FS = FS_TITLE
FIG3_OP_LEGEND_FS = FS_LEGEND
FIG3_BOT_LEGEND_FS = FS_LEGEND
FIG3_CBAR_LABEL_FS = FS
FIG3_SAVE_PAD = 0.02

FIG4_LEFT = 0.10
FIG4_RIGHT = 0.98
FIG4_TOP = 0.92
FIG4_BOTTOM = 0.10
FIG4_HSPACE = 0.35
FIG4_PCT_FS = FS

EXT_LOOP_LEFT = 0.18
EXT_LOOP_RIGHT = 0.98
EXT_LOOP_TOP = 0.90
EXT_LOOP_BOTTOM = 0.20
EXT_LOOP_WSPACE = 0.58
EXT_LOOP_MARKER = 3.2
EXT_LOOP_YLABEL_PAD = 12.0
