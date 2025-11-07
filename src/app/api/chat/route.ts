import { NextRequest } from "next/server";
import type {  LayoutObject, LayoutResponse, ModelsJson } from "@/types/models";
import { getModels } from "@/lib/models";
import { getTopKModels } from "@/lib/retriever";
import { buildSystemPrompt, callGemini } from "@/lib/geminiClient";

export const runtime = "nodejs";

const RAW_SNIPPET_LIMIT = 200;
const DEFAULT_TEMPERATURE = 0.2;
const TOP_K = 10;

type GenerateBody = {
  description?: unknown;
  style?: unknown;
  temperature?: unknown;
  previousLayout?: unknown;
};

/**
 * Lightweight readiness endpoint.
 */
export function GET() {
  return Response.json({ ok: true });
}

/**
 * POST handler that generates and sanitizes room layouts via Gemini.
 */
export async function POST(req: NextRequest): Promise<Response> {
  try {
    const body = (await req.json()) as GenerateBody;
    const description = extractDescription(body.description);

    if (!description) {
      return Response.json(
        { ok: false, error: "Invalid description" },
        { status: 400 }
      );
    }

    const style = toOptionalString(body.style);
    const temperature =
      typeof body.temperature === "number" ? body.temperature : DEFAULT_TEMPERATURE;
    const previousLayout = castLayoutResponse(body.previousLayout);

    const descriptionWithStyle = style
      ? `${description.trim()} style:${style.trim()}`
      : description.trim();

    console.info("[api/chat] request", {
      desc_len: descriptionWithStyle.length,
      style: style ?? null,
      temperature,
      has_previous_layout: Boolean(previousLayout),
    });

    // Fast-fail when Gemini credentials are not configured to avoid dev overlay HTML.
    if (!process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
      return Response.json(
        {
          ok: false,
          error:
            "Missing GOOGLE_GENERATIVE_AI_API_KEY. Set it in your environment to enable layout generation.",
        },
        { status: 503 }
      );
    }

    const models: ModelsJson = await getModels();
    const modelCount = Object.keys(models).length;
    console.info("[api/chat] models_loaded", { count: modelCount });
    const candidates = getTopKModels(models, descriptionWithStyle, TOP_K);
    console.info("[api/chat] candidate_count=%d", candidates.length);
    console.debug(
      "[api/chat] candidates_sample",
      candidates.slice(0, 5).map((m) => ({ path: m.path, tags: (m.tags || []).slice(0, 3) }))
    );

    // Build a rich system prompt tailored to this app and include a compact
    // candidate catalog and full models manifest for reference.
    const candidateCatalog = buildCandidateCatalog(models, candidates);
    const fullModelsJson = serializeModelsJson(models);
    const systemPrompt = buildRoomDesignerPrompt({
      candidateCatalog,
      userDescription: description,
      style,
      previousLayout: previousLayout ?? null,
      modelsJson: fullModelsJson,
    });
    console.debug("[api/chat] system_prompt_chars=%d", systemPrompt.length);

    const raw = await callGemini(
      descriptionWithStyle,
      candidates,
      temperature,
      systemPrompt
    );
    console.debug("[api/chat] raw_chars=%d", raw.length);
    console.debug("[api/chat] raw_snippet", raw.slice(0, RAW_SNIPPET_LIMIT));
    let parsed = safeParseLayout(raw);
    // Normalize common variations (e.g., model_id, position arrays, rotation number)
    parsed = normalizeLayout(parsed, models);

    // Clamp positions, snap to grid, and resolve overlaps before returning.
    const clean = sanitizeLayout(parsed, {
      snap: 0.1,
      roomFallback: { width_m: 4, depth_m: 3.5, height_m: 2.7 },
    });

    return Response.json({
      ok: true,
      data: clean,
    });
  } catch (error) {
    console.error("[api/chat] Unexpected error", error);
    const message =
      error instanceof Error ? error.message : "Failed to generate layout.";
    return Response.json(
      { ok: false, error: message },
      { status: 500 }
    );
  }
}

function extractDescription(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function toOptionalString(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function castLayoutResponse(value: unknown): LayoutResponse | null {
  if (!value || typeof value !== "object") return null;
  return value as LayoutResponse;
}

/**
 * Attempts to parse Gemini output safely, falling back to regex extraction when
 * the model wraps JSON in prose.
 */
function safeParseLayout(raw: string): LayoutResponse {
  try {
    return JSON.parse(raw) as LayoutResponse;
  } catch {
    console.warn("[api/chat] direct JSON.parse failed; trying fenced block");
    // Try extracting fenced JSON first, e.g., ```json { ... } ```
    const fence = raw.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
    if (fence && fence[1]) {
      try {
        return JSON.parse(fence[1]) as LayoutResponse;
      } catch {
        console.warn("[api/chat] fenced JSON parse failed; trying first brace block");
      }
    }

    // Regex fallback to capture the first JSON block (Gemini sometimes adds chatter).
    const match = raw.match(/\{[\s\S]*\}/);
    if (match) {
      try {
        return JSON.parse(match[0]) as LayoutResponse;
      } catch {
        console.warn("[api/chat] brace block parse failed; giving up");
      }
    }
    const snippet = raw.slice(0, RAW_SNIPPET_LIMIT);
    throw new Error(`Unable to parse layout JSON. Snippet: ${snippet}`);
  }
}

/**
 * Attempts to coerce common LLM variants into our LayoutResponse schema.
 */
function normalizeLayout(input: LayoutResponse, models: ModelsJson): LayoutResponse {
  const out: LayoutResponse = {
    room: input.room ?? { width_m: 4, depth_m: 3.5, height_m: 2.7 },
    objects: Array.isArray(input.objects) ? [...input.objects] : [],
    rationale: input.rationale,
  };

  const findKeyByPath = (p: string): string | undefined => {
    for (const [k, v] of Object.entries(models)) {
      if (v.path === p) return k;
    }
    return undefined;
  };

  out.objects = out.objects.map((obj, idx) => {
    const copy: any = { ...obj };

    // Model normalization: prefer obj.model (gltf:<key>)
    if (!copy.model) {
      const modelId: unknown = (obj as any).model_id ?? (obj as any).modelId;
      const path: unknown = (obj as any).path;
      if (typeof modelId === "string") {
        // Extract the last segment after ':' e.g., 'models:gaming-room:chair' -> 'chair'
        const segs = modelId.split(":");
        const last = segs[segs.length - 1];
        if (last) copy.model = `gltf:${last}`;
      } else if (typeof path === "string" && path.startsWith("/models/")) {
        const key = findKeyByPath(path);
        if (key) copy.model = `gltf:${key}`;
      }
    }

    // Position normalization
    if (!copy.position_m) {
      const p: unknown = (obj as any).position ?? (obj as any).pos;
      if (Array.isArray(p) && p.length >= 3) {
        copy.position_m = { x: Number(p[0]) || 0, y: Number(p[1]) || 0, z: Number(p[2]) || 0 };
      } else if (p && typeof p === "object") {
        const o = p as any;
        if (typeof o.x === "number" || typeof o.y === "number" || typeof o.z === "number") {
          copy.position_m = { x: o.x ?? 0, y: o.y ?? 0, z: o.z ?? 0 };
        }
      }
    }

    // Rotation normalization
    if (!copy.rotation_deg) {
      const r: unknown = (obj as any).rotation;
      if (typeof r === "number") {
        copy.rotation_deg = { y: r };
      } else if (Array.isArray(r) && r.length >= 3) {
        copy.rotation_deg = { x: Number(r[0]) || 0, y: Number(r[1]) || 0, z: Number(r[2]) || 0 };
      } else if (r && typeof r === "object") {
        const o = r as any;
        const x = typeof o.x === "number" ? o.x : undefined;
        const y = typeof o.y === "number" ? o.y : undefined;
        const z = typeof o.z === "number" ? o.z : undefined;
        if (x != null || y != null || z != null) copy.rotation_deg = { x, y, z };
      }
    }

    // Ensure an id exists for stability
    if (!copy.id) copy.id = obj.id ?? `obj_${idx + 1}`;

    return copy;
  });

  return out;
}

type SanitizeOptions = {
  snap: number;
  roomFallback: LayoutResponse["room"];
};

/**
 * Minimal sanitizer that snaps positions, keeps objects within room bounds, and
 * ensures unparented items rest on the floor plane.
 */
function sanitizeLayout(layout: LayoutResponse, options: SanitizeOptions): LayoutResponse {
  const room = layout.room ?? options.roomFallback;

  const snapValue = (value: number): number =>
    Math.round(value / options.snap) * options.snap;

  const clampWithinRoom = (value: number, halfSize = 0, axis: "width" | "depth"): number => {
    const limit = axis === "width" ? room.width_m / 2 : room.depth_m / 2;
    return Math.max(-limit + halfSize, Math.min(limit - halfSize, value));
  };

  const sanitizedObjects: LayoutObject[] = (layout.objects ?? []).map((object) => {
    const size = object.size_m ?? {};
    const position = object.position_m ?? { x: 0, y: 0, z: 0 };
    const snappedPosition = {
      x: snapValue(position.x ?? 0),
      y: snapValue(position.y ?? 0),
      z: snapValue(position.z ?? 0),
    };

    const halfWidth = (size.w ?? 0) / 2;
    const halfDepth = (size.d ?? 0) / 2;

    const clampedPosition = {
      x: clampWithinRoom(snappedPosition.x, halfWidth, "width"),
      y:
        object.parent || size.h == null
          ? snappedPosition.y
          : Math.max(size.h / 2, snappedPosition.y),
      z: clampWithinRoom(snappedPosition.z, halfDepth, "depth"),
    };

    return {
      ...object,
      position_m: clampedPosition,
    };
  });

  return {
    ...layout,
    room,
    objects: sanitizedObjects,
  };
}

/**
 * Formats top-K candidate lines for the catalog block.
 */
function buildCandidateCatalog(models: ModelsJson, candidates: Array<ReturnType<typeof getTopKModels>[number]>) {
  const findKeyForMeta = (meta: unknown): string | undefined => {
    for (const [key, m] of Object.entries(models)) {
      if (m === meta || m.path === (meta as any)?.path) return key;
    }
    return undefined;
  };

  const lines = candidates.map((m) => {
    const key = findKeyForMeta(m) ?? "unknown";
    const anchors = Object.entries(m.anchors ?? {})
      .slice(0, 4)
      .map(([name, v]) => `${name}(${v.x},${v.y},${v.z})`)
      .join(" ") || "none";
    const tags = (m.tags ?? []).slice(0, 6).join(", ");
    return `${key} | path:${m.path} | w:${m.w.toFixed(3)} d:${m.d.toFixed(3)} h:${m.h.toFixed(3)} | anchors: ${anchors} | tags: ${tags}`;
  });
  return lines.join("\n");
}

/**
 * Serializes the full models manifest for inclusion in the system prompt.
 * If serialization fails, returns an empty JSON object string.
 */
function serializeModelsJson(models: ModelsJson): string {
  try {
    return JSON.stringify(models, null, 2);
  } catch (err) {
    console.warn("[api/chat] Failed to serialize models.json for prompt:", err);
    return "{}";
  }
}

/**
 * Creates the system prompt using the provided template and app context.
 */
function buildRoomDesignerPrompt(args: {
  candidateCatalog: string;
  userDescription: string;
  style: string | null;
  previousLayout: LayoutResponse | null;
  modelsJson: string;
}) {
  const { candidateCatalog, userDescription, style, previousLayout, modelsJson } = args;
  const prev = previousLayout ? JSON.stringify(previousLayout, null, 2) : "(none)";
  return [
    "You are RoomLayoutDesigner, an LLM that designs simple 3D room layouts using a small catalog of low-poly GLB assets.",
    "Output must be a single, valid JSON object that matches the schema below—no prose, no markdown, no code fences.",
    "Units are meters for distances and degrees for rotations.",
    "Only use models from the candidate catalog provided below.",
    "",
    "Your role",
    "- Read the user’s description and (optionally) a previous layout.",
    "- Select appropriate models only from the provided catalog.",
    "- Place them inside the room without overlaps, respecting model dimensions.",
    "- Return one JSON object that conforms to the LayoutResponse schema.",
    "",
    "Catalog (you MUST use only these)",
    "Below is a compact catalog extracted from models.json.",
    "Each entry provides the key you must reference (via model: \"gltf:<key>\"), plus dimensions and path.",
    "If you need an item that isn’t in this list, choose the closest match from this list instead of inventing a new one.",
    "",
    candidateCatalog,
    "",
    "Full models.json manifest (for reference only — do not echo it back; use it to select correct keys and dimensions):",
    modelsJson,
    "",
    "Catalog line format (example):",
    "desk_basic | path:/models/lowpoly-room/desk_basic.glb | w:1.600 d:0.700 h:0.750 | anchors: top_center(0,0.75,0) | tags: desk,wood,basic",
    "",
    "Inputs",
    `User description: ${userDescription}`,
    `Style hint (optional): ${style ?? "(none)"}`,
    "Previous layout (optional, same schema as output):",
    prev,
    "",
    "Rules (follow strictly)",
    "- Schema only. Return exactly one JSON object that matches LayoutResponse (see “Schema” below). No extra keys, no comments, no text.",
    "- Models from catalog only. Every placed object that uses a GLB must set model: \"gltf:<key>\". Do not write file paths in model; the renderer resolves paths using the catalog.",
    "- Coordinate system: origin is the room center; X is left/right, Z is forward/back, Y is up. Room bounds along X are ±(width_m/2), along Z are ±(depth_m/2).",
    "- Dimensions & placement: Use catalog w,d,h (meters). Objects must be fully inside the room. Keep at least 0.2m walkway margin between objects (edge-to-edge) and 0.05m from walls.",
    "- Grid snap: Round X and Z to the nearest 0.1m.",
    "- Floor contact: For floor-standing furniture do NOT set position_m.y (omit it). The renderer will set y = h/2 so it rests on the floor. Only set y for stacked/anchored items.",
    "- Rotation: rotation_deg is in degrees. Prefer Y-axis rotations; keep X/Z at 0 unless specifically requested.",
    "- Anchors (optional): you may use parent/relative_position_m/anchor if available; otherwise place directly.",
    "- Creativity within constraints: match style using tags; if unsure, pick fewer objects and keep room clean.",
    "- Edits mode: if previous layout provided, modify minimally; keep IDs stable where possible.",
    "- Rationale: include a short rationale string explaining key placement choices (1–3 sentences).",
    "",
    "Placement heuristics (guidance, not mandatory):",
    "- Beds typically go with the headboard against a wall; leave ~0.6–0.9m clearance on at least one long side.",
    "- Desks align to a wall; keep ~0.6m chair clearance behind.",
    "- Wardrobes/closets go against walls; avoid blocking desk/bed access.",
    "- Side tables/nightstands sit near bed edges with ~0.05–0.1m gap.",
    "",
    "Schema (return exactly this shape)",
    "{",
    "  \"room\": { \"width_m\": 4.0, \"depth_m\": 3.5, \"height_m\": 2.7, \"floor_material\": \"light_wood\", \"wall_color\": \"soft_white\" },",
    "  \"objects\": [",
    "    { \"id\": \"desk1\", \"type\": \"desk\", \"label\": \"work desk\", \"model\": \"gltf:desk_basic\", \"position_m\": { \"x\": -0.8, \"y\": 0.375, \"z\": 0.0 }, \"rotation_deg\": { \"y\": 0 } }",
    "  ],",
    "  \"rationale\": \"Why these models and placements.\"",
    "}",
    "",
    "Field notes",
    "- room.*: use defaults width 4.0, depth 3.5, height 2.7 if unspecified.",
    "- objects[].model: must be \"gltf:<key>\" from the catalog (no paths in output).",
    "- objects[].position_m: meters; if omitted and on floor, renderer defaults y = h/2.",
    "- objects[].rotation_deg: degrees; typically { \"y\": 0..360 }.",
    "- objects[].construction: optional primitive recipe if no GLB matches (prefer GLB when available).",
  ].join("\n");
}

/**
 * Setup checklist:
 * - Configure the Gemini/AI SDK API key environment variable (see src/lib/geminiClient.ts).
 * - Ensure public/models.json exists and matches the ModelsJson schema.
 * - Optionally replace the inline sanitizeLayout helper with a shared src/lib/layoutSanitizer module.
 *
 * How to test now:
 * - Run pnpm dev
 * - curl -X POST http://localhost:3000/api/generate \
 *     -H "Content-Type: application/json" \
 *     -d '{"description":"Design a cozy gaming room","style":"cozy","temperature":0.3}'
 */
