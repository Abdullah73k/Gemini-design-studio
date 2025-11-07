import ChatPanel from "@/components/RoomUI/ChatPanel";
import ViewerPanel from "@/components/RoomUI/ViewerPanel";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 px-4 py-12 text-slate-100 md:px-8">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-10">
        <header className="space-y-4 text-center md:text-left">
          <p className="text-xs uppercase tracking-[0.6em] text-slate-400">
            Gemini design studio
          </p>
          <h1 className="text-4xl font-semibold tracking-tight text-white sm:text-5xl">
            Dream, prompt, and stage interiors in minutes
          </h1>
          <p className="text-base text-slate-300 md:max-w-3xl">
            Describe the vibe, furniture, and constraints. Gemini and our 3D toolkit will
            take it from concept to render-ready layout. Left panel handles prompting
            while the right panel will preview the 3D scene.
          </p>
        </header>

        <section className="flex flex-col gap-6 md:flex-row">
          <div className="md:w-[32%]">
            {/* Future: plug in preset prompt chips for common workflows */}
            <ChatPanel />
          </div>
          <div className="md:w-[68%]">
            {/* Future: RoomViewer + HUD controls land here */}
            <ViewerPanel />
          </div>
        </section>
      </div>
    </main>
  );
}

/**
 * Testing Guide:
 * 1. Ensure src/app/api/generate/route.ts returns APIResponse<LayoutResponse>.
 * 2. Run `pnpm dev`.
 * 3. Visit http://localhost:3000.
 * 4. Prompt example: "Cozy gaming room with desk, dual monitors, LED strip".
 * 5. Click "Generate Layout" and expect a pretty JSON payload in the Response card.
 * 6. If the backend is incomplete, temporarily mock the fetch logic in ChatPanel.
 */
