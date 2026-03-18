import logging
import os
from typing import Dict, List

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import anam

from rag_search import RAGSearch

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# ── Load FAISS index once (shared across all sessions, read-only) ─────────────
_RAG_INDEX_DIR  = os.environ.get("RAG_INDEX_DIR", "./rag_index")
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_FACE_ID = os.environ.get("FACE_ID", "")
_FACE_NAME = os.environ.get("FACE_NAME", "")


logger.info("Loading FAISS index from %s …", _RAG_INDEX_DIR)
_db = RAGSearch(index_dir=_RAG_INDEX_DIR, gemini_api_key=_GEMINI_API_KEY)
s = _db.stats()
logger.info("FAISS index ready — %d chunks, %d documents", s["total_chunks"], s["total_documents"])

# ── Global TTS cache (text → audio frames) ────────────────────────────────────
tts_cache: Dict[str, List[rtc.AudioFrame]] = {}

HOLD_TEXT = "Just a sec please."


async def say_cached(session: AgentSession, tts, text: str) -> None:
    """Synthesize on first use, replay from cache on subsequent calls."""
    if text not in tts_cache:
        stream = tts.synthesize(text)
        frames: List[rtc.AudioFrame] = []
        async for event in stream:
            frames.append(event.frame)
        tts_cache[text] = frames
        logger.debug("TTS cache MISS — %d frames stored for %r", len(frames), text[:60])
    else:
        logger.debug("TTS cache HIT for %r", text[:60])

    async def audio_gen():
        for frame in tts_cache[text]:
            yield frame

    await session.say(text, audio=audio_gen())


# ── Agent ─────────────────────────────────────────────────────────────────────

class Assistant(Agent):
    def __init__(self, tts) -> None:
        super().__init__(
            instructions="""

Speak as Bhagwan Chowdhry, a finance professor with genuine intellectual enthusiasm and deep concern for human welfare. Begin with personal anecdotes or credentials that establish your connection to the topic, creating intimacy through conversational authority. Structure arguments by moving from specific experience to broader principle to future implications. Use medium-length sentences (15-25 words) mixed with short, punchy declaratives for emphasis. Employ em-dashes generously for clarifying asides and rhetorical questions to engage readers. Ground every abstract claim in concrete examples—specific numbers, named people, places, and personal observations. Favor active voice and confident declaratives like 'nothing short of revolutionary' or 'completely serious.' Explain technical terms naturally without jargon. Weave personal narrative throughout rather than front-loading it. Connect abstract topics to human welfare, especially the poor and marginalized. Show measured optimism about solutions while remaining realistic about challenges. Propose specific, actionable ideas rather than vague principles. Acknowledge limitations humbly while asserting expertise. End with future-focused projections that inspire action. Vary paragraph length (3-7 sentences) with clear topic sentences and explicit transitions. Let substance create emphasis rather than formatting tricks. Sprinkle a little humor, now and then.
You represent his portfolio and answer questions on his behalf to visitors, students, and collaborators.
About Bhagwan Chowdhry:
Bhagwan Chowdhry is a Professor of Finance at the Indian School of Business and Research Professor at UCLA Anderson where he has held an appointment since 1988. He is the Executive Director of the Digital Identity Research Initiative (DIRI) and Faculty Director of I-Venture at ISB.
He has taught at the University of Chicago, University of Illinois at Chicago, and HKUST. He received his PhD from the University of Chicago Booth School of Business, an MBA in Finance from the University of Iowa, and a BTech in Mechanical Engineering from IIT Kanpur.
His research covers International Finance, Corporate Finance, Impact Investing, and FinTech. He has proposed the Financial Access at Birth (FAB) initiative, where every child born is given an initial deposit of $100 in an online bank account. He co-authored the book FinTech for Billions: Simple, Human, Ubiquitous.

Rules:
- When the user asks a factual question, call search_portfolio.
- Only respond without the tool for pure greetings like "hello" or "how are you".
- Do NOT answer from memory — always call the tool for any factual detail about Bhagwan.
- Keep answers short and spoken-friendly: no bullet points, no markdown, no asterisks, no emojis.
- Speak in plain, warm sentences as if talking face to face.
- If the tool finds nothing relevant, say so honestly and offer to help with something else.
- Before using the tool, say "Just a sec please.", then only call the tool.
""",
        )
        self._tts_instance = tts

    async def on_enter(self) -> None:
        """Greet the visitor when the agent joins.
        Pre-synthesize the hold phrase here — TTS is connected to the
        session at this point, so synthesize() works correctly.
        """
        if HOLD_TEXT not in tts_cache:
            frames: List[rtc.AudioFrame] = []
            async for event in self._tts_instance.synthesize(HOLD_TEXT):
                frames.append(event.frame)
            tts_cache[HOLD_TEXT] = frames
            logger.info("Hold phrase pre-synthesized (%d frames).", len(frames))

        await say_cached(
            self.session,
            self._tts_instance,
            "Hello! I'm here to tell you about Bhagwan Chowdhry. What would you like to know?",
        )

    @function_tool
    async def search_portfolio(
        self,
        context: RunContext,
        query: str,
        section: str = "",
    ):
        """Search Bhagwan Chowdhry's portfolio knowledge base for factual information.

        Use this tool for any question about his research papers, publications,
        career history, employment, education, advisory roles, executive teaching,
        media appearances, YouTube videos, opinions, blog posts, or contact details.

        Args:
            query: Natural language question or search phrase, e.g.
                   "PhD from University of Chicago" or "research on IPO underpricing".
            section: Optional. Restrict to one section:
                     biography, research, working_papers, employment, education,
                     video, opinion, executive_teaching, advisor, cases,
                     associate_editor, contact, fame, general.
                     Leave blank to search everything.
        """
        async def _hold_audio():
            for frame in tts_cache.get(HOLD_TEXT, []):
                yield frame

        hold_handle = context.session.say(
            HOLD_TEXT,
            audio=_hold_audio(),
            add_to_chat_ctx=False,
        )

        logger.info('[RAG] query="%s" section="%s"', query, section or "all")
        results = _db.search(
            query=query,
            top_k=6,
            section_filter=section if section else None,
        )

        if not hold_handle.interrupted and not hold_handle.done():
            hold_handle.interrupt()

        if not results:
            logger.info("[RAG] no results")
            return "No relevant information found in the portfolio for that query."

        lines = []
        for r in results:
            lines.append(f"[{r.rank}] {r.doc_title} (score {r.score:.3f})")
            snippet = r.raw_content[:400].strip()
            if len(r.raw_content) > 400:
                snippet += "…"
            lines.append(snippet)
            lines.append("")

        logger.info("[RAG] %d chunks, top score=%.3f", len(results), results[0].score)
        return "\n".join(lines)


# ── Server setup ──────────────────────────────────────────────────────────────

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    tts = inference.TTS(model="inworld/inworld-tts-1.5-max", voice="Edward")

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=tts,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    avatar = anam.AvatarSession(
        persona_config=anam.PersonaConfig(
            name=_FACE_NAME,
            avatarId=_FACE_ID,
        ),
    )

    # ── Start Anam avatar — fall back to audio-only if it fails ──────────────
    avatar_ok = True
    try:
        await avatar.start(session, room=ctx.room)
        logger.info("Anam avatar started successfully.")
    except Exception as exc:
        avatar_ok = False
        logger.warning(
            "Anam avatar failed to start (%s: %s) — running in audio-only mode.",
            type(exc).__name__, exc,
        )

    await session.start(
        agent=Assistant(tts=tts),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    await ctx.connect()

    # Notify user if avatar failed — runs after connect so they can hear it
    if not avatar_ok:
        await session.generate_reply(
            instructions=(
                "Apologise briefly that the video avatar is unavailable right now "
                "due to a service limit, but reassure the user that the voice "
                "assistant is fully working and you are happy to help."
            )
        )


if __name__ == "__main__":
    cli.run_app(server)