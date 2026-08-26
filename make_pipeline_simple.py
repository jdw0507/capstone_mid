import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import svgwrite

W, H = 1100, 720
dwg = svgwrite.Drawing("pipeline_figure.svg", size=(f"{W}px", f"{H}px"),
                        viewBox=f"0 0 {W} {H}")

# Fonts & colors
FONT = "Consolas, 'Courier New', monospace"
TEAL = "#00B4D8"
RED = "#E63946"
DARK = "#1D3557"
GRAY = "#6C757D"
LIGHT = "#F8F9FA"
BORDER = "#ADB5BD"

# Background
dwg.add(dwg.rect((0,0), (W,H), fill="white"))

# Title
dwg.add(dwg.text("RL-based Dynamic Model Selection for Black-Litterman Portfolio Optimization",
                  insert=(W/2, 36), text_anchor="middle",
                  font_family=FONT, font_size=16, font_weight="bold", fill=DARK))
dwg.add(dwg.text("End-to-End Pipeline",
                  insert=(W/2, 56), text_anchor="middle",
                  font_family=FONT, font_size=11, fill=GRAY))

# Helper: rounded box with number badge
def stage_box(x, y, w, h, num, label, sublabel=None, color=TEAL):
    g = dwg.g()
    # Box
    g.add(dwg.rect((x, y), (w, h), rx=6, ry=6,
                    fill="white", stroke=color, stroke_width=1.5))
    # Number badge
    cx, cy = x + 16, y + 16
    g.add(dwg.circle((cx, cy), 10, fill=color))
    g.add(dwg.text(str(num), insert=(cx, cy + 4), text_anchor="middle",
                   font_family=FONT, font_size=10, font_weight="bold", fill="white"))
    # Label
    g.add(dwg.text(label, insert=(x + 32, y + 20), font_family=FONT,
                   font_size=13, font_weight="bold", fill=DARK))
    # Sublabel
    if sublabel:
        g.add(dwg.text(sublabel, insert=(x + 32, y + 36), font_family=FONT,
                       font_size=10, fill=GRAY))
    dwg.add(g)

# Helper: arrow (vertical)
def arrow_v(x, y1, y2):
    mid = x
    dwg.add(dwg.line((mid, y1), (mid, y2), stroke=BORDER, stroke_width=1.2))
    # arrowhead
    dwg.add(dwg.polygon([(mid, y2), (mid-4, y2-7), (mid+4, y2-7)],
                         fill=BORDER))

# Helper: arrow (horizontal)
def arrow_h(x1, x2, y):
    dwg.add(dwg.line((x1, y), (x2, y), stroke=BORDER, stroke_width=1.2))
    dwg.add(dwg.polygon([(x2, y), (x2-7, y-4), (x2-7, y+4)],
                         fill=BORDER))

# Helper: dashed arrow (horizontal)
def arrow_h_dash(x1, x2, y, color=BORDER):
    dwg.add(dwg.line((x1, y), (x2, y), stroke=color, stroke_width=1.2,
                     stroke_dasharray="5,3"))
    dwg.add(dwg.polygon([(x2, y), (x2-7, y-4), (x2-7, y+4)], fill=color))

# ── Layout ──
# 6 stages, top-to-bottom, centered
# Stage 1: Data
# Stage 2: Walk-Forward CV
# Stage 3 & 4 side by side: Forecasting | RL Selection
# Stage 5: Black-Litterman
# Stage 6: Evaluation

LEFT = 100
BOX_W = 900
SMALL_W = 420
ROW_H = 48
GAP = 28

y = 80

# ━━ Stage 1: Data ━━
stage_box(LEFT, y, BOX_W, ROW_H, 1, "Data Preparation",
          "Raw dataset  →  Asset panel (12 assets × 2,268 days)  →  5-day excess return targets",
          color="#457B9D")
y += ROW_H

# Arrow
arrow_v(W/2, y, y + GAP)
y += GAP

# ━━ Stage 2: Walk-Forward CV ━━
stage_box(LEFT, y, BOX_W, ROW_H + 12, 2, "Nested Walk-Forward Cross-Validation",
          "Train (1,008d)  |  View-Build (252d)  |  Test (756d)   —   Sliding window: 63d/fold",
          color="#7F8C8D")
y += ROW_H + 12

# Arrow splits into two
arrow_v(W/2, y, y + GAP)
y += GAP

# ━━ Stage 3 & 4 side by side ━━
s3x = LEFT
s4x = LEFT + SMALL_W + 60
side_h = 110

# Stage 3: Forecasting
g3 = dwg.g()
g3.add(dwg.rect((s3x, y), (SMALL_W, side_h), rx=6, ry=6,
                fill="white", stroke="#F5A623", stroke_width=1.5))
# badge
cx3, cy3 = s3x + 16, y + 16
g3.add(dwg.circle((cx3, cy3), 10, fill="#F5A623"))
g3.add(dwg.text("3", insert=(cx3, cy3+4), text_anchor="middle",
                font_family=FONT, font_size=10, font_weight="bold", fill="white"))
g3.add(dwg.text("Forecasting Models", insert=(s3x+32, y+20),
                font_family=FONT, font_size=13, font_weight="bold", fill=DARK))
# ML row
g3.add(dwg.text("ML (8):  OLS, Ridge, Lasso, DT, RF, XGB, LGBM, CatBoost",
                insert=(s3x+20, y+44), font_family=FONT, font_size=9, fill=GRAY))
# DL row
g3.add(dwg.text("DL (6):  PatchTST, CNN1D, LSTM, MLP, Transformer, Hybrid",
                insert=(s3x+20, y+62), font_family=FONT, font_size=9, fill=GRAY))
# Output
g3.add(dwg.text("Output: Per-asset 5d excess return predictions",
                insert=(s3x+20, y+86), font_family=FONT, font_size=9,
                font_weight="bold", fill="#F5A623"))
dwg.add(g3)

# Stage 4: RL Selection
g4 = dwg.g()
g4.add(dwg.rect((s4x, y), (SMALL_W, side_h), rx=6, ry=6,
                fill="white", stroke=RED, stroke_width=1.5))
cx4, cy4 = s4x + 16, y + 16
g4.add(dwg.circle((cx4, cy4), 10, fill=RED))
g4.add(dwg.text("4", insert=(cx4, cy4+4), text_anchor="middle",
                font_family=FONT, font_size=10, font_weight="bold", fill="white"))
g4.add(dwg.text("RL Model Selection (4 Agents)", insert=(s4x+32, y+20),
                font_family=FONT, font_size=13, font_weight="bold", fill=DARK))
# 2x2 compact
g4.add(dwg.text("DQN_Pred  /  PPO_Pred   (per-asset, MAE reward)",
                insert=(s4x+20, y+44), font_family=FONT, font_size=9, fill=GRAY))
g4.add(dwg.text("DQN_Port  /  PPO_Port   (cross-sect., Sharpe reward)",
                insert=(s4x+20, y+62), font_family=FONT, font_size=9, fill=GRAY))
# Output
g4.add(dwg.text("Output: Best model per (asset, date)",
                insert=(s4x+20, y+86), font_family=FONT, font_size=9,
                font_weight="bold", fill=RED))
dwg.add(g4)

# Arrow from stage 3 to stage 4
arrow_h(s3x + SMALL_W, s4x, y + side_h/2)

# Arrows down from both
mid3 = s3x + SMALL_W/2
mid4 = s4x + SMALL_W/2
y_bot_side = y + side_h

# Merge arrow down to stage 5
y += side_h
arrow_v(W/2, y, y + GAP)
y += GAP

# ━━ Stage 5: Black-Litterman ━━
stage_box(LEFT, y, BOX_W, ROW_H + 12, 5, "Black-Litterman Portfolio Construction",
          "View (P, Q, \u03a9)  →  Market Eq. (\u03c0 = \u03b4\u03a3w)  →  BL Posterior (\u03bc_BL)  →  MVO  →  Weights w*",
          color="#7B68EE")
y += ROW_H + 12

# Arrow
arrow_v(W/2, y, y + GAP)
y += GAP

# ━━ Stage 6: Evaluation ━━
stage_box(LEFT, y, BOX_W, ROW_H + 12, 6, "Walk-Forward Backtest & Evaluation",
          "5-day rebalancing  |  19 strategies compared  |  Sharpe, Return, MDD, Turnover, Win Rate",
          color="#2ECC71")
y += ROW_H + 12

# Arrow
arrow_v(W/2, y, y + GAP - 4)
y += GAP

# ━━ Result box ━━
res_w = 700
res_h = 36
rx = (W - res_w) / 2
dwg.add(dwg.rect((rx, y), (res_w, res_h), rx=4, ry=4,
                  fill=DARK, stroke="none"))
dwg.add(dwg.text("RL agents dynamically select optimal models  \u2192  Superior risk-adjusted returns",
                  insert=(W/2, y + 22), text_anchor="middle",
                  font_family=FONT, font_size=12, font_weight="bold", fill="white"))

# Save
dwg.saveas("C:/Users/탁준혁/Desktop/jay/capstone/pipeline_figure.svg", pretty=True)
print("Done: pipeline_figure.svg saved")
