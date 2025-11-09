# Gemini Design Studio → 3D room layouts in seconds

<p align="center">
  <img src="./public/readme-hero.png" alt="Gemini Design Studio preview" width="900" />
</p>


## Hackathon Context
Built at **DeltaHacks Lite** in ~**1.5 hours**. The focus was a working end‑to‑end pipeline for “Best use of Gemini”

## What It Does
- You describe a room (style, furniture, vibe).
- The backend prompts Google Gemini (via Vercel AI SDK) to return a **strict JSON** layout (objects, sizes, positions, rotations, rationale).
- The frontend renders low‑poly GLB assets into a room with a light **sanitizer** to snap to grid, clamp to bounds, and nudge overlaps.

## Architecture (High‑Level)
- 3D asset pipeline: GLB models → metadata script → `public/models.json`.
- Retriever: select top‑K candidate models (from `models.json`) to keep the prompt compact.
- Gemini via Vercel AI SDK: system prompt enforces a strict `LayoutResponse` schema.
- API route: Next.js App Router endpoint (`/api/chat` — see note below).
- Sanitizer: grid snap (0.1m), clamp to room, simple AABB separation, set floor Y = h/2 when missing.
- UI: Left = chat (useChat), Right = live 3D (react‑three‑fiber + drei).

```
Text prompt → Retriever (top‑K from models.json) → Gemini (strict JSON)
           → /api/chat → Sanitize → Viewer (GLB load) → 3D scene
```

## models.json (How It’s Produced & Used)
A small Node script (not included here) scans low‑poly GLBs and precomputes a manifest at `public/models.json` with fields:

| field | purpose |
|---|---|
| `path` | Public URL under `/models/...` (served from `/public/models/...`). |
| `w`,`d`,`h` | Dimensions in meters (width, depth, height). |
| `bbox_min`,`bbox_max` | AABB corners in meters. |
| `pivot` | Local pivot in meters. |
| `anchors` | Optional named anchor points (meters). |
| `tags` | Search keywords (e.g., desk, wood, modern). |
| `margin` | Recommended clearance in meters. |

- Assets live under: `/public/models/lowpoly-room/*`.
- Names + tags enable retrieval (e.g., `desk_basic`, `chair_gaming`).

## Prompting & Gemini (Large Context + Strict Schema)
- The top‑K candidate slice (keys + w/d/h + optional anchors) is injected into the **system prompt** so Gemini picks valid assets.
- The full `models.json` is also included (compactly), improving key/path awareness.
- Gemini must return a single JSON object with shape:
  - `room` { `width_m`, `depth_m`, `height_m`, `floor_material?`, `wall_color?` }
  - `objects[]` { `id`, `type?`, `model: "gltf:<key>"`, `position_m`, `rotation_deg`, ... }
  - `rationale` short string
- Model is configurable via env. Suggested: **Gemini 2.0** Flash for large context + creative placement. Temperature controls creativity.

## Backend (Next.js App Router)
- Primary route in this repo: `src/app/api/chat/route.ts` (UI calls this).
- Responsibilities:
  - Validate request, load `public/models.json`.
  - Retrieve top‑K candidates; build a strict, placement‑aware system prompt.
  - Call Gemini via Vercel AI SDK; extract JSON (regex + fenced fallback).
  - Normalize common variants (e.g., `model_id` → `model`, arrays → `position_m`).
  - Sanitize: snap to 0.1m, clamp to room, set floor `y = h/2` when missing, simple overlap nudge.
  - Return `{ ok, data | error }`.
- Env:
  - `GOOGLE_GENERATIVE_AI_API_KEY` (required).
  - `GEMINI_MODEL` (optional; defaults to `gemini-2.0-flash`).

## Frontend (UI + 3D)
- Left panel (≈30%): `useChat` UI (Vercel AI SDK UI). Custom submit posts JSON to `/api/chat`, shows pretty JSON, and dispatches `layout:update` on success.
- Right panel (≈70%): `react-three-fiber` viewer:
  - Canvas + ambient/directional lights (no black scene) + light gray floor sized to room.
  - GLB loading using `models.json.path`. If `position_m.y` missing, viewer sets `y = h/2`.

## Repo Layout (Brief)
- `src/app/api/chat/route.ts` — current API used by the UI.
- `src/components/RoomUI/*` — Chat/UI and viewer panel.
- `src/components/Three/*` — `RoomViewer`, `SceneObjects`, `GLTFModel`, `PrimitiveBuilder`.
- `src/lib/*` — `models` (manifest I/O), `retriever`, `geminiClient`, `modelLoader`.
- `public/models/lowpoly-room/*` — low‑poly GLBs.
- `public/models.json` — precomputed manifest.

## Getting Started (Local)
**Prereqs**: Node 18+, pnpm.

Install:
```bash
pnpm install
```

Env (create `.env.local`):
```bash
# Google provider key for Gemini (used by @ai-sdk/google)
GOOGLE_GENERATIVE_AI_API_KEY=your_key_here
# Optional: override the default model
GEMINI_MODEL=gemini-2.0-flash
```

Run:
```bash
pnpm run dev
```
Open: http://localhost:3000

## Constraints & Future Work
- Time‑boxed hackathon prototype (1.5h): small catalog, minimal sanitizer, no walls/HDRI.
- Next:
  - Scripted manifest generator (CLI) — TODO: add `scripts/build-models-json.ts`.
  - Larger catalogs + richer anchors/walls.
  - Export screenshots/GLTF; budgeting/constraints; better lighting/materials.

## Credits
- Built at **DeltaHacks Lite**. Thanks to the organizers and mentors.
- Libraries: **Next.js**, **Vercel AI SDK**, **@ai-sdk/google**, **react‑three‑fiber**, **drei**, **Tailwind**/**shadcn/ui**.

— Thanks for reading! Devpost: https://devpost.com/software/gemini-layout-studio?ref_content=my-projects-tab&ref_feature=my_projects
