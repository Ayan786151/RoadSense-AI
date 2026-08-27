# RoadSense AI — Crossroad Simulation Design System

Design tokens, visual hierarchy, and component rules for the urban traffic simulation frontend. This document locks the visual world and prevents regression into generic "AI slop" aesthetics.

---

## 1. Design Philosophy

- **Domain-First Realism**: Modeled after real-world municipal Intelligent Transportation Systems (ITS) and high-craft simulation games (Mini Motorways, Cities: Skylines).
- **Anti-Slop Directives**:
  - **No decorative neon glow**: No diffuse radial background glows, no colored 1px glowing card borders, no laser halos.
  - **Functional Color Only**: Red, Amber, and Green are reserved strictly for traffic signals and critical status states.
  - **Semantically Justified Typography**: Monospace is used **only** for tabular numeric data and timestamped logs; all UI headers, labels, and copy use proportional typography (`Inter`).
  - **Tangible Physical Assets**: Real top-down vehicle silhouettes with windshields, roofs, and mirrors instead of colored rectangles with floating text tags.

---

## 2. Color Palette & Tokens

### Canvas & Environment Tokens
| Token | Value | Purpose |
|---|---|---|
| `--asphalt-base` | `#2b2f36` | Pavement surface |
| `--asphalt-dark` | `#22252a` | Shoulder & deceleration zones |
| `--marking-white` | `#f8fafc` | Stop lines, zebra crossings, arrows |
| `--marking-dash` | `#94a3b8` | Centerlines & lane dividers |
| `--curb-stone` | `#475569` | Concrete curb perimeter |
| `--sidewalk-tile` | `#1e2227` | Concrete pedestrian zone |
| `--grass-verge` | `#1c2826` | Subtle corner verge / median |

### UI & Dashboard Tokens
| Token | Value | Purpose |
|---|---|---|
| `--bg-base` | `#0f1115` | Viewport & outer frame |
| `--bg-panel` | `#16191e` | Sidebar & header surface |
| `--bg-card` | `#1c2026` | KPI data tiles & story cards |
| `--border-subtle` | `#2d323b` | Structural container dividers |
| `--border-strong` | `#3e4450` | Interactive element borders |
| `--text-primary` | `#f1f5f9` | Metric values, titles |
| `--text-secondary` | `#94a3b8` | Subtitles, descriptions |
| `--text-muted` | `#64748b` | Captions, units, timestamps |

### Functional Signal & Status Colors (No Glow)
| Token | Value | Purpose |
|---|---|---|
| `--signal-red` | `#dc2626` | Signal RED, severe collision alert |
| `--signal-amber` | `#d97706` | Signal AMBER, clearance phase |
| `--signal-green` | `#16a34a` | Signal GREEN, safe equilibrium |
| `--accent-civic` | `#2563eb` | Primary UI action, slider thumb |
| `--accent-cctv` | `#0284c7` | Camera perception indicator |

---

## 3. Typography Scale

- **Primary Font**: `Inter`, `-apple-system`, `BlinkMacSystemFont`, `sans-serif`
  - Headers: `600` weight, `-0.02em` tracking
  - Labels & Body: `400` / `500` weight
- **Data & Monospace Font**: `JetBrains Mono`, `monospace`
  - Strict Rule: Used **only** for tabular numbers (`font-variant-numeric: tabular-nums`) and timestamped dispatch logs.

---

## 4. Vehicle Silhouette Standards (IISc UVH26)

All vehicles are drawn with proper top-down proportions and physical features:
- **Motorcycle / Scooter**: Width $10\text{px}$, Length $22\text{px}$. Front wheel fork, chassis, handlebar grips, and rider helmet.
- **Auto-Rickshaw**: Width $16\text{px}$, Length $26\text{px}$. Tapered front cab, black canvas canopy, windshield glass, yellow front fender.
- **Sedan / SUV**: Width $20\text{px}$, Length $36\text{px}$. Curved hood, dark windshield glass, side window pillars, rear glass, side mirrors.
- **Transit Bus**: Width $24\text{px}$, Length $62\text{px}$. Panoramic front windshield, white roof with rooftop ventilation pods, side glass bands.
- **Truck**: Width $24\text{px}$, Length $54\text{px}$. Heavy cab, cargo bed with tailgate.
