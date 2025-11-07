import { google } from "@ai-sdk/google";
import { generateText } from "ai";
import type { LayoutResponse, ModelMeta, ModelsJson } from "../types/models";

type ModelId = keyof ModelsJson;

const DEFAULT_MODEL = "gemini-2.0-flash";
const DEFAULT_TEMPERATURE = 0.7;

/**
 * Builds the constrained system prompt describing allowed models and output schema.
 *
 * @param candidates - Ranked list of models Gemini may select from.
 * @param exampleLayout - Optional LayoutResponse sample included for schema fidelity.
 * @returns A string prompt emphasizing JSON-only output with strict schema rules.
 */
export function buildSystemPrompt(
	candidates: ModelMeta[],
	exampleLayout?: LayoutResponse
): string {
	const candidateLines =
		candidates.length > 0
			? candidates.map((meta, index) => formatCandidate(meta, index + 1))
			: [];

	const exampleBlock = exampleLayout
		? `\nStrictly follow this schema example:\n${JSON.stringify(
				exampleLayout,
				null,
				2
		  )}\n`
		: "";

	return [
		"You are a spatial layout planner AI.",
		"Respond ONLY with valid JSON that conforms to the LayoutResponse schema.",
		"Do not include markdown, prose, comments, or additional keys.",
		"",
		"Use ONLY the following models (refer to their IDs exactly):",
		candidateLines.join("\n") ||
			"- none provided; rely on procedural primitives.",
		"",
		"Output rules:",
		"- Positions and sizes are in meters.",
		"- Rotations are in degrees.",
		"- Snap positions to 0.1m increments when possible.",
		"- Keep objects within the room bounds and avoid overlaps.",
		"- Always include a rationale string summarizing key placement decisions.",
		exampleBlock.trim(),
		"",
		"Return STRICT JSON. Do not wrap it in markdown fences or add text before/after.",
	]
		.filter(Boolean)
		.join("\n");
}

/**
 * Calls Gemini via the Vercel AI SDK using the Google provider and returns the raw text.
 *
 * @param userPrompt - Free-form user instruction describing the desired room layout.
 * @param candidates - Top-K candidate models from getTopKModels.
 * @param temperature - Optional sampling temperature (default 0.7).
 * @returns Raw string response from Gemini (expected JSON per LayoutResponse schema).
 */
export async function callGemini(
	userPrompt: string,
	candidates: ModelMeta[],
	temperature: number = DEFAULT_TEMPERATURE
): Promise<string> {
	const systemPrompt = buildSystemPrompt(candidates);
	const modelName = process.env.GEMINI_MODEL ?? DEFAULT_MODEL;

	try {
		const { text } = await generateText({
			model: google(modelName),
			system: systemPrompt,
			prompt: userPrompt,
			temperature,
		});

		if (!text) {
			throw new Error("Gemini returned an empty response payload.");
		}

		return text;
	} catch (error) {
		console.error("[geminiClient] Gemini invocation failed.", {
			message: (error as Error).message,
			model: modelName,
		});
		throw new Error("Failed to generate layout. Verify Gemini configuration.");
	}
}

/**
 * Formats a single candidate line for the system prompt, including ID, size, and anchors.
 *
 * @param meta - The model metadata entry from models.json.
 * @param fallbackIndex - Index used when the manifest does not provide a recognizable ID.
 */
function formatCandidate(meta: ModelMeta, fallbackIndex: number): string {
	const anchors =
		Object.keys(meta.anchors ?? {})
			.slice(0, 4)
			.join(", ") || "none";

	const identifier =
		((meta as ModelMeta & { id?: ModelId }).id ??
			meta.path
				.replace(/^\/+/, "")
				.replace(/\.[^/.]+$/, "")
				.replace(/[\\/]/g, ":")) ||
		`model_${fallbackIndex}`;

	return `- ${identifier} | size: ${meta.w.toFixed(2)}x${meta.d.toFixed(
		2
	)}x${meta.h.toFixed(2)}m | anchors: ${anchors}`;
}

/**
 * Usage:
 *
 * const layoutJSON = await callGemini(
 *   "Design a cozy gaming room",
 *   topModels,
 *   0.7
 * );
 * console.log(layoutJSON);
 */

/**
 * Setup & TODOs:
 * - Requires GOOGLE_GENERATIVE_AI_API_KEY (or Vercel-managed key) in the runtime environment.
 * - Optionally set GEMINI_MODEL (default: models/gemini-1.5-large). `models/gemini-1.5-turbo` is a great alternative.
 * - Ensure system prompt customization matches your product voice or constraints before deployment.
 * - Always pass curated top-K candidates from getTopKModels; Gemini must not see the full manifest.
 * - Recommended models: "models/gemini-1.5-large" (maximum context) or "models/gemini-1.5-turbo" (faster).
 * - TODO: add streaming support (generateTextStream) and structured logging of token usage for cost tracking.
 */
