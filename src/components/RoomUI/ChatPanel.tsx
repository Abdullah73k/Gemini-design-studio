"use client";

import { useMemo, useState } from "react";
import { useChat } from "@ai-sdk/react";
import type { APIResponse, LayoutResponse } from "@/types/models";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import {
	Card,
	CardContent,
	CardHeader,
	CardTitle,
	CardDescription,
} from "@/components/ui/card";

const styleOptions = [
	{ label: "No preference", value: "" },
	{ label: "Cozy", value: "cozy" },
	{ label: "Minimal", value: "minimal" },
	{ label: "Modern", value: "modern" },
];

const randomId = () =>
	typeof crypto !== "undefined" && "randomUUID" in crypto
		? crypto.randomUUID()
		: Math.random().toString(36).slice(2, 9);

/**
 * Chat-driven control panel for authoring Gemini layout prompts.
 */
export default function ChatPanel() {
    const { messages, setMessages } = useChat();
	const [description, setDescription] = useState<string>("");
	const [style, setStyle] = useState<string>("cozy");
	const [temperature, setTemperature] = useState<number>(0.2);
	const [isPending, setIsPending] = useState(false);
	const [errorMessage, setErrorMessage] = useState<string | null>(null);
	const [responseOpen, setResponseOpen] = useState(true);
	const [lastLayout, setLastLayout] = useState<LayoutResponse | null>(null);

	// Derive latest assistant text for display when no structured layout is present yet.
	const latestAssistantText = useMemo(() => {
		const reversed = [...messages].reverse();
		const candidate = reversed.find(
			(m) => (m as { role?: string }).role === "assistant"
		) as
			| { parts?: Array<{ type: string; text?: string }>; content?: unknown }
			| undefined;
		if (!candidate) return undefined;
		if (Array.isArray(candidate.parts)) {
			const text = candidate.parts
				.filter((p) => p?.type === "text" && typeof p.text === "string")
				.map((p) => p.text as string)
				.join("\n");
			return text || undefined;
		}
		return typeof candidate.content === "string"
			? (candidate.content as string)
			: undefined;
	}, [messages]);

	async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
		event.preventDefault();
		const trimmed = description.trim();
		if (!trimmed) {
			setErrorMessage("Please describe the room you want to generate.");
			return;
		}

		setIsPending(true);
		setErrorMessage(null);
		setResponseOpen(true);
		setLastLayout(null);

		// Append the user message using the UI message parts format.
		setMessages((prev) => [
			...prev,
			{
				id: randomId(),
				role: "user",
				parts: [{ type: "text", text: trimmed }],
			} as unknown as (typeof prev)[number],
		]);

		const payload: Record<string, unknown> = {
			description: trimmed,
			temperature,
		};
		if (style) payload.style = style;

		try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const contentType = response.headers.get("content-type") || "";
      if (!contentType.includes("application/json")) {
        const text = await response.text();
        const snippet = text.slice(0, 200);
        throw new Error(
          `Server returned non-JSON (${response.status}). Snippet: ${snippet}`
        );
      }
      const json = (await response.json()) as APIResponse<LayoutResponse>;

			if (!response.ok || !json.ok || !json.data) {
				throw new Error(
					json.error ?? `Request failed with status ${response.status}`
				);
			}

			const pretty = JSON.stringify(json.data, null, 2);
			setMessages((prev) => [
				...prev,
				{
					id: randomId(),
					role: "assistant",
					parts: [{ type: "text", text: pretty }],
				} as unknown as (typeof prev)[number],
			]);
			setLastLayout(json.data);
		} catch (err) {
			const message =
				err instanceof Error
					? err.message
					: "Something went wrong while generating.";
			setMessages((prev) => [
				...prev,
				{
					id: randomId(),
					role: "assistant",
					parts: [{ type: "text", text: `Error: ${message}` }],
				} as unknown as (typeof prev)[number],
			]);
			setErrorMessage(message);
		} finally {
			setIsPending(false);
			setDescription("");
		}
	}

	return (
		<Card className="border-white/10 bg-white/5 text-slate-100 backdrop-blur-xl">
			<CardHeader>
				<CardTitle className="bg-gradient-to-r from-indigo-200 to-cyan-200 bg-clip-text text-transparent">
					Layout Assistant
				</CardTitle>
				<CardDescription className="text-slate-300">
					Describe your vision and let Gemini assemble a room layout. Adjust
					styling and sampling controls for creative freedom.
				</CardDescription>
			</CardHeader>
			<CardContent className="space-y-6">
				<form className="space-y-4" onSubmit={handleSubmit}>
					<div className="space-y-2">
						<Label htmlFor="description" className="text-slate-200">
							Room description
						</Label>
						<Textarea
							id="description"
							placeholder="Cozy gaming room with LED strips and dual monitors..."
							value={description}
							onChange={(e) => setDescription(e.target.value)}
							disabled={isPending}
							className="min-h-[120px] resize-none border-white/10 bg-white/10 text-slate-100 placeholder:text-slate-400"
						/>
					</div>

					<div className="grid gap-4 sm:grid-cols-2">
						<div className="space-y-2">
							<Label className="text-slate-200">Style</Label>
							<Select
								onValueChange={(value) => setStyle(value)}
								value={style}
								disabled={isPending}
							>
								<SelectTrigger className="border-white/10 bg-white/10 text-slate-100">
									<SelectValue placeholder="Choose style" />
								</SelectTrigger>
                <SelectContent className="bg-slate-900 text-slate-100">
                  {styleOptions
                    .filter((opt) => opt.value !== "")
                    .map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                </SelectContent>
							</Select>
						</div>

						<div className="space-y-2">
							<Label htmlFor="temperature" className="text-slate-200">
								Temperature
							</Label>
							<Input
								id="temperature"
								type="number"
								min={0}
								max={1}
								step={0.1}
								value={temperature}
								onChange={(event) => {
									const next = Number(event.target.value);
									if (Number.isNaN(next)) {
										setTemperature(0);
										return;
									}
									setTemperature(Math.min(1, Math.max(0, next)));
								}}
								disabled={isPending}
								className="border-white/10 bg-white/10 text-slate-100"
							/>
						</div>
					</div>

					{errorMessage && (
						<p className="text-sm text-rose-300">Error: {errorMessage}</p>
					)}

					<Button
						type="submit"
						disabled={isPending}
						className="w-full bg-white/20 text-slate-900 hover:bg-white/40"
					>
						{/* TODO: Add lucide-react Send/Loader icons once the icon package is added. */}
						{isPending ? "Generating..." : "Generate Layout"}
					</Button>
				</form>

				<div className="space-y-3">
					<div className="space-y-2">
						<Label className="text-slate-200">Conversation</Label>
						<div className="max-h-48 space-y-3 overflow-y-auto rounded-xl border border-white/10 bg-black/20 p-3">
							{messages.length === 0 && (
								<p className="text-sm text-slate-400">
									No messages yet. Try a prompt!
								</p>
							)}

							{messages.map((message) => (
								<div
									key={message.id}
									className="rounded-lg bg-white/5 p-3 text-sm text-slate-100 shadow-inner"
								>
									<p className="mb-1 text-xs uppercase tracking-wide text-slate-400">
										{message.role}
									</p>
									<p className="whitespace-pre-wrap text-slate-100">
										{Array.isArray(
											(
												message as {
													parts?: Array<{ type: string; text?: string }>;
												}
											).parts
										)
											? (
													message as {
														parts?: Array<{ type: string; text?: string }>;
													}
											  )
													.parts!.filter(
														(p) =>
															p?.type === "text" && typeof p.text === "string"
													)
													.map((p) => p.text as string)
													.join("\n")
											: (message as { content?: string }).content}
									</p>
								</div>
							))}
						</div>
					</div>

					<Card className="border-white/10 bg-black/30 text-slate-100">
						<CardHeader className="flex flex-row items-center justify-between space-y-0">
							<CardTitle className="text-base">Response</CardTitle>
							<Button
								type="button"
								variant="ghost"
								size="sm"
								onClick={() => setResponseOpen((value) => !value)}
								className="text-slate-200 hover:bg-white/10"
							>
								{responseOpen ? "Hide" : "Show"}
							</Button>
						</CardHeader>
						{responseOpen && (
							<CardContent>
								{/* TODO: Replace <pre> with react-syntax-highlighter once the dependency is available. */}
								<pre className="max-h-56 overflow-y-auto rounded-lg bg-black/50 p-4 text-xs text-emerald-200 whitespace-pre-wrap font-mono">
									{lastLayout
										? JSON.stringify(lastLayout, null, 2)
										: latestAssistantText ?? "// Gemini JSON will appear here."}
								</pre>
							</CardContent>
						)}
					</Card>
				</div>
			</CardContent>
		</Card>
	);
}
