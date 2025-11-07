"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/**
 * Placeholder visual panel until the Three.js RoomViewer is wired in.
 */
export default function ViewerPanel() {
	return (
		<Card className="h-full border-white/10 bg-white/5 text-slate-100 backdrop-blur-xl">
			<CardHeader>
				<CardTitle className="bg-gradient-to-r from-cyan-200 to-blue-200 bg-clip-text text-transparent">
					3D Viewer
				</CardTitle>
				<p className="text-sm text-slate-300">
					Layout previews will render here once the Three.js scene is connected.
				</p>
			</CardHeader>
			<CardContent>
				{/* TODO: Layer in framer-motion fade/scale once the dependency is available. */}
				<div className="relative h-[420px] w-full overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6 shadow-2xl">
					<div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(99,102,241,0.35),_transparent_45%)]" />
					<div className="absolute inset-0 opacity-40">
						<div className="h-full w-full bg-[linear-gradient(120deg,_rgba(255,255,255,0.08)_1px,_transparent_1px),_linear-gradient(0deg,_rgba(255,255,255,0.05)_1px,_transparent_1px)] bg-[size:40px_40px]" />
					</div>
					<div className="relative flex h-full flex-col items-center justify-center text-center text-slate-200">
						<p className="text-lg font-semibold">
							Interactive canvas coming soon
						</p>
						<p className="mt-2 max-w-sm text-sm text-slate-300">
							This panel will host the react-three-fiber RoomViewer with GLTF
							assets, lighting controls, and camera helpers.
						</p>
						<p className="mt-4 text-xs uppercase tracking-[0.3em] text-slate-400"></p>
					</div>
				</div>
			</CardContent>
		</Card>
	);
}
