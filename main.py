"""
main.py
=======
This is the chatbot's backend server. It does three jobs for every
incoming message:

  1. RETRIEVE  - search the company knowledge base (built by ingest.py)
                 for the paragraphs most relevant to the user's question
  2. ASSEMBLE  - combine those paragraphs + a role-specific system prompt
                 + the conversation history into one message list
  3. GENERATE  - send that to the LLM (Groq's free API) and return the
                 model's reply to the frontend

Run this file with:
    uvicorn main:app --reload --port 8000
"""

import os
import json
import requests
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from groq import Groq
import chromadb
from chromadb.utils import embedding_functions


# =========================================================================
# SETUP — runs once when the server starts
# =========================================================================

# Reads the .env file and loads GROQ_API_KEY into os.environ
load_dotenv()

# The FastAPI application object — this is what uvicorn runs
app = FastAPI(title="Enterprise Support Chatbot")

# CORS = Cross-Origin Resource Sharing. Without this, a browser will block
# your frontend from calling this backend, since they're on different
# domains. In production, restrict this to your actual frontend's URL
# instead of "*" -- otherwise any website could call your API and burn
# through your Groq quota.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",              # local development
        "https://simpleenterprisechatbot.netlify.app/",  # replace with your real Netlify URL
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The Groq client — this is what actually talks to the LLM over the internet
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Same embedding function used in ingest.py -- MUST match, otherwise
# vectors computed here won't be comparable to what's stored.
#
# We use ChromaDB's DefaultEmbeddingFunction (onnxruntime-based) instead
# of loading sentence-transformers/torch directly. sentence-transformers
# pulls in the full PyTorch + CUDA stack, which alone is enough to exceed
# the 512MB RAM limit on Render's free tier and get the process killed
# (exit code 137). onnxruntime does the same embedding job with a much
# smaller memory footprint.
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Connect to the same ChromaDB folder that ingest.py wrote to. We pass
# the same embedding_function so queries later get embedded the same way
# documents were embedded during ingestion.
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(
    name="company_kb",
    embedding_function=embedding_function,
)

# Change this to your actual company name — it gets inserted into prompts
COMPANY_NAME = "Google"

# -------------------------------------------------------------------------
# ZOHO DESK CONFIG
# -------------------------------------------------------------------------
# Zoho uses OAuth2 rather than a simple API key. ZOHO_REFRESH_TOKEN,
# ZOHO_CLIENT_ID, and ZOHO_CLIENT_SECRET come from running
# get_zoho_refresh_token.py once. ZOHO_ORG_ID and ZOHO_DEPARTMENT_ID come
# from running get_zoho_org_and_department.py once (after the refresh
# token exists). ZOHO_DC is your data center suffix ("com", "in", "eu",
# etc.) -- same one used in both setup scripts.
# -------------------------------------------------------------------------
ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")
ZOHO_ORG_ID = os.getenv("ZOHO_ORG_ID")
ZOHO_DEPARTMENT_ID = os.getenv("ZOHO_DEPARTMENT_ID")
ZOHO_DC = os.getenv("ZOHO_DC", "com")
ZOHO_REQUESTER_EMAIL = os.getenv("ZOHO_REQUESTER_EMAIL", "chatbot@google.com")

# Zoho access tokens expire after about an hour. We cache the current one
# here in memory and refresh it automatically when it's missing/expired,
# rather than requesting a brand new one on every single ticket (which
# would work, but is wasteful and slower).
_zoho_access_token: str | None = None

# In-memory conversation storage: {session_id: [list of messages]}
# NOTE: this resets every time the server restarts, and doesn't work if
# you run multiple server instances (e.g. behind a load balancer).
# For real production use, replace this dict with a database table
# (e.g. a Supabase "conversations" table keyed by session_id).
sessions: dict[str, list[dict]] = {}

# -------------------------------------------------------------------------
# ESCALATION STORAGE
# -------------------------------------------------------------------------
# When a user asks to be connected to a human, we record it here. This is
# what makes the bot's "connecting you to a human agent" message TRUE
# instead of just something the LLM says: a real record gets created that
# a human (or another system) can act on.
#
# escalation_queue: in-memory list, cleared on restart -- good for the
#   demo and for the /escalations endpoint below to show "what a human
#   agent would see".
# ESCALATIONS_FILE: also appended to a local file, so escalations survive
#   a server restart (this is the "poor man's database" for a prototype;
#   swap for a real ticket in Zendesk/Slack/etc. in production).
# -------------------------------------------------------------------------
escalation_queue: list[dict] = []
ESCALATIONS_FILE = Path("escalations.jsonl")


# =========================================================================
# SYSTEM PROMPTS — one "personality + set of rules" per use case
# =========================================================================
# Each prompt tells the model exactly what job it's doing and what it's
# NOT allowed to do. The {company} placeholder gets filled in at request
# time with COMPANY_NAME.

BOT_PROMPTS = {
    "support": """You are {company}'s customer support assistant.
For factual questions about policies, pricing, or products, answer ONLY
using the retrieved company information provided in the conversation. If
that answer isn't in it, say you don't have that information and offer to
connect the user with a human agent -- do not guess or make up policy
details.
For conversational messages that are NOT factual questions -- greetings,
acknowledgements like "yeah"/"ok"/"thanks", or replies to a question YOU
just asked -- use the conversation history to understand what's being
responded to, and reply naturally. Never claim the user "didn't ask a
question" if they're simply replying to something you said. Keep answers
short, friendly, and to the point.""",

    "lead_qualification": """You are {company}'s sales assistant talking to a
potential customer. Your goal is to naturally learn four things over the
course of the conversation: their company size, their budget range, their
timeline for buying, and their main pain point. Ask ONE question at a time
-- never a list of questions in a single message. Pay close attention to
the conversation history: if the user's message is answering a question
you just asked, treat it as that answer, even if it's short (e.g. "yeah",
"around 20 people"). Once you have learned enough, summarize what you've
learned and tell them a sales representative will follow up. Never invent
pricing or promises that aren't in the retrieved company information.""",

    "hr_support": """You are {company}'s internal HR assistant, used only by
employees. For factual questions about HR policy, answer ONLY using the
retrieved company information provided in the conversation. For anything
involving a personal leave dispute, medical leave specifics, payroll
disputes, or a workplace grievance, do NOT attempt to answer -- instead
tell the employee to contact HR directly. Never ask the employee for
sensitive personal data (medical details, SSN, etc.) in this chat.
For conversational replies -- acknowledgements, answers to a question you
just asked, follow-ups -- use the conversation history to understand
context and respond naturally.""",
}


# =========================================================================
# REQUEST SCHEMA — defines exactly what JSON the frontend must send
# =========================================================================

class ChatRequest(BaseModel):
    session_id: str       # a unique ID per browser tab/user, generated by the frontend
    message: str           # the user's typed message
    bot_type: str = "support"   # which specialized bot: "support" | "lead_qualification" | "hr_support"


# =========================================================================
# CORE FUNCTION 1: retrieve_context
# =========================================================================

def retrieve_context(query: str, k: int = 4) -> str:
    """
    Given the user's question, finds the `k` most relevant chunks from the
    knowledge base and returns them joined into one string.

    How it works:
      1. Pass the raw question text to ChromaDB via query_texts -- the
         collection was created with embedding_function, so ChromaDB
         embeds the query internally using that same function before
         comparing it against stored vectors.
      2. Ask ChromaDB: "which stored chunks have embeddings closest to this
         one?" (closeness = similarity in meaning, not exact word match)
      3. Return those chunks' text, joined with blank lines between them
    """
    results = collection.query(
        query_texts=[query],
        n_results=k,
    )

    # results["documents"] is a list-of-lists (one inner list per query we
    # sent). We only sent one query, so we take index [0].
    matched_chunks = results["documents"][0]

    return "\n\n".join(matched_chunks)


# =========================================================================
# CORE FUNCTION 2: escalation detection + logging
# =========================================================================
# Phrases that, when sent right after the bot has offered a human agent,
# count as the user confirming they want one. Keep this list short and
# specific to avoid false positives (e.g. "yeah" answering an unrelated
# question shouldn't trigger this).
ESCALATION_CONFIRM_PHRASES = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay",
    "connect me", "please", "please do", "human agent", "talk to a human",
    "talk to someone", "speak to a human", "i want a human",
}

# Phrases that directly ask for a human, regardless of what the bot just said
ESCALATION_DIRECT_PHRASES = {
    "human agent", "real person", "talk to a human", "speak to a human",
    "talk to someone", "connect me to support", "customer support agent",
}


def wants_escalation(user_message: str, last_assistant_message: str | None) -> bool:
    """
    Decides whether THIS turn is a genuine request/confirmation for human
    escalation. This is what the earlier version of the bot was missing --
    it would just say "connecting you to a human agent" based on the LLM's
    own judgement, with nothing behind it actually checking or recording
    that intent.

    Two ways this fires:
      1. The user directly asks for a human, in any message
         (e.g. "can I talk to a real person?")
      2. The bot's PREVIOUS reply offered to connect them to a human, and
         the user's reply is a short confirmation (e.g. "yeah", "ok")
    """
    normalized_message = user_message.strip().lower()

    # Case 1: direct request, regardless of conversation state
    if any(phrase in normalized_message for phrase in ESCALATION_DIRECT_PHRASES):
        return True

    # Case 2: bot just offered escalation, user is confirming
    if last_assistant_message:
        bot_offered_human = "human agent" in last_assistant_message.lower()
        user_confirmed = normalized_message in ESCALATION_CONFIRM_PHRASES
        if bot_offered_human and user_confirmed:
            return True

    return False


def create_freshdesk_ticket(session_id: str, bot_type: str, transcript: list[dict]) -> dict | None:
    """
    Creates a real support ticket in Freshdesk so an actual human agent
    sees it in their queue. This is the piece that turns "connecting you
    to a human agent" from a claim into a fact.

    Returns a dict with the ticket's id and a direct link on success, or
    None if ticket creation failed (missing credentials, network issue,
    bad request, etc). Callers must handle None gracefully -- a Freshdesk
    outage should never crash the chatbot itself.

    NOTE on the requester email: Freshdesk requires every ticket to have
    a requester email. This demo doesn't collect the real user's email,
    so it falls back to FRESHDESK_REQUESTER_EMAIL for every ticket. For a
    real deployment, add an email field to the frontend (e.g. ask for it
    once escalation is confirmed) and pass the real address through
    ChatRequest instead.
    """
    if not FRESHDESK_DOMAIN or not FRESHDESK_API_KEY:
        print("Freshdesk is not configured (missing FRESHDESK_DOMAIN or "
              "FRESHDESK_API_KEY in .env) — skipping real ticket creation.")
        return None

    # Turn the transcript into readable plain text for the ticket body
    conversation_text = "\n\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in transcript
    )

    url = f"https://{FRESHDESK_DOMAIN}.freshdesk.com/api/v2/tickets"

    payload = {
        "subject": f"Chatbot escalation ({bot_type}) — session {session_id}",
        "description": conversation_text,
        "email": FRESHDESK_REQUESTER_EMAIL,
        "priority": 2,     # Freshdesk priority: 1=Low, 2=Medium, 3=High, 4=Urgent
        "status": 2,       # Freshdesk status: 2=Open
        "tags": ["chatbot-escalation", bot_type],
    }

    try:
        # Freshdesk's API uses HTTP Basic Auth with your API key as the
        # username and the literal string "X" as the password.
        response = requests.post(
            url,
            json=payload,
            auth=(FRESHDESK_API_KEY, "X"),
            timeout=10,
        )
        response.raise_for_status()   # raises an exception on 4xx/5xx responses

        ticket = response.json()
        ticket_id = ticket.get("id")
        return {
            "id": ticket_id,
            "url": f"https://{FRESHDESK_DOMAIN}.freshdesk.com/a/tickets/{ticket_id}",
        }

    except requests.exceptions.RequestException as e:
        # Covers network errors, timeouts, and non-2xx responses. We log
        # it and return None rather than letting this take down the
        # /chat endpoint -- the user should still get a reply even if
        # Freshdesk is temporarily unreachable.
        print(f"Freshdesk ticket creation failed: {e}")
        return None


def log_escalation(session_id: str, bot_type: str, history: list[dict]) -> dict:
    """
    Records a real escalation event: who (session_id), what kind of bot,
    when, the full conversation transcript, AND now a real Freshdesk
    ticket so a human agent actually gets notified.

    Writes to:
      - escalation_queue (in memory, for the /escalations endpoint)
      - escalations.jsonl (on disk, survives a server restart)
      - Freshdesk (a real ticket a support agent will see in their queue)
    """
    transcript = [m for m in history if m["role"] != "system"]

    freshdesk_ticket = create_freshdesk_ticket(session_id, bot_type, transcript)

    entry = {
        "session_id": session_id,
        "bot_type": bot_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transcript": transcript,
        "freshdesk_ticket": freshdesk_ticket,   # None if Freshdesk call failed
    }

    escalation_queue.append(entry)

    with open(ESCALATIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


# =========================================================================
# CORE FUNCTION 3: get_clean_history
# =========================================================================

def get_clean_history(session_id: str, bot_type: str) -> list[dict]:
    """
    Returns this session's conversation history, containing ONLY real
    dialogue: the system prompt (once) plus the actual raw text of every
    user and assistant turn — no retrieval clutter mixed in.

    This is what makes the model correctly remember "what it itself just
    said". Earlier, we were saving "Context:\n...\n\nUser question: yeah"
    into history instead of just "yeah" — which buried the real
    conversation under retrieval noise on every single turn, so by turn 3
    the model could no longer tell what it had actually asked or said.
    """
    history = sessions.setdefault(session_id, [])

    if not history:
        system_prompt = BOT_PROMPTS[bot_type].format(company=COMPANY_NAME)
        history.append({"role": "system", "content": system_prompt})

    return history


# =========================================================================
# CORE FUNCTION 3: the /chat endpoint — ties everything together
# =========================================================================

@app.post("/chat")
def chat(req: ChatRequest):
    """
    Main endpoint the frontend calls. Steps:
      1. Retrieve relevant knowledge-base chunks for the user's message
      2. Get clean history, and check whether THIS message is a genuine
         escalation request (using the last thing the bot said, before
         we append the new user message)
      3. If it is, actually log the escalation for real (this is what
         makes the bot's claim true instead of a hallucination)
      4. Append the user's raw message to clean history
      5. Build a temporary call list: system prompt + retrieved context +
         a note telling the model the REAL escalation status + history
      6. Call the Groq LLM
      7. Save the assistant's reply into clean session history
      8. Trim history so it doesn't grow forever
      9. Return the reply as JSON
    """

    # STEP 1: retrieval — find relevant chunks for THIS message only
    context = retrieve_context(req.message, k=4)

    # STEP 2: get history, and capture what the bot said last BEFORE we
    # append this new user turn — escalation detection needs to know what
    # question the user might be responding to
    history = get_clean_history(req.session_id, req.bot_type)
    last_assistant_message = None
    for m in reversed(history):
        if m["role"] == "assistant":
            last_assistant_message = m["content"]
            break

    escalated_this_turn = wants_escalation(req.message, last_assistant_message)

    # STEP 3: append the user's raw message to clean history
    history.append({"role": "user", "content": req.message})

    # STEP 4: if this turn is a genuine escalation, actually record it --
    # this now includes creating a real Freshdesk ticket. escalation_entry
    # stays None if nothing was escalated this turn.
    escalation_entry = None
    if escalated_this_turn:
        escalation_entry = log_escalation(req.session_id, req.bot_type, history)

    ticket = escalation_entry["freshdesk_ticket"] if escalation_entry else None

    # STEP 5: build a temporary call list. Alongside the retrieved
    # context, we tell the model the TRUE escalation status for this
    # turn -- so it can only say "connecting you to a human" when that
    # actually just happened, and must NOT say it otherwise.
    context_note = {
        "role": "system",
        "content": (
            f"Relevant company information for the CURRENT user message "
            f"only (may be irrelevant if the user is just chatting, "
            f"acknowledging, or asking a follow-up — in that case, rely on "
            f"the conversation history instead):\n\n{context}"
        ),
    }
    if escalated_this_turn and ticket:
        escalation_status_text = (
            f"A human agent has just been notified about this conversation "
            f"-- support ticket #{ticket['id']} was created. Confirm this "
            f"to the user and mention a support agent will follow up soon. "
            f"You may mention the ticket number."
        )
    elif escalated_this_turn:
        # Escalation was requested but the real Freshdesk call failed
        # (e.g. misconfigured credentials, Freshdesk down). Be honest
        # about this rather than claiming a ticket exists.
        escalation_status_text = (
            "The user asked for a human agent, but creating a support "
            "ticket failed on our end. Apologize, and tell them to email "
            "support directly instead as a fallback."
        )
    else:
        escalation_status_text = (
            "No human escalation has occurred yet. Do NOT tell the user "
            "you are connecting them to a human agent or that a ticket has "
            "been created -- only OFFER to connect them if that's "
            "genuinely appropriate, and wait for their confirmation."
        )
    escalation_note = {"role": "system", "content": escalation_status_text}
    call_messages = [history[0], context_note, escalation_note] + history[1:]

    # STEP 6: call the LLM
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=call_messages,
        temperature=0.4,   # lower temperature = more factual/consistent, less "creative"
        max_tokens=512,
    )
    reply = response.choices[0].message.content

    # STEP 7: remember the assistant's reply, in clean form, for next turn
    history.append({"role": "assistant", "content": reply})

    # STEP 8: keep only the last 20 messages so the session doesn't grow
    # forever and blow past the model's context window over a long chat
    sessions[req.session_id] = history[-20:]

    # STEP 9: send the reply back to the frontend, including whether a
    # real escalation happened this turn and the resulting ticket info
    # (useful for the UI to show a "connected to support" badge/link)
    return {
        "reply": reply,
        "bot_type": req.bot_type,
        "escalated": escalated_this_turn,
        "ticket_id": ticket["id"] if ticket else None,
        "ticket_url": ticket["url"] if ticket else None,
    }


# =========================================================================
# BONUS ENDPOINT: extract structured lead info (used by lead_qualification bot)
# =========================================================================

class ExtractRequest(BaseModel):
    session_id: str

@app.post("/extract-lead")
def extract_lead(req: ExtractRequest):
    """
    Looks at the conversation so far for a given session and asks the LLM
    to pull out structured fields (company size, budget, etc.) as JSON.
    Useful for pushing qualified leads into a CRM or a Supabase table.
    """
    history = sessions.get(req.session_id, [])

    # Flatten the conversation into plain text for the extraction prompt
    conversation_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in history if m["role"] != "system"
    )

    extraction_prompt = f"""From this conversation, extract lead info as JSON
with exactly these keys: company_size, budget_range, timeline, pain_point,
qualified (true or false). Use null for anything not mentioned. Return
ONLY valid JSON and nothing else -- no explanation, no markdown formatting.

Conversation:
{conversation_text}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": extraction_prompt}],
        temperature=0,   # 0 = as deterministic as possible, good for structured output
    )

    raw_output = response.choices[0].message.content

    try:
        lead_data = json.loads(raw_output)
    except json.JSONDecodeError:
        # If the model didn't return clean JSON, fail gracefully rather
        # than crashing the whole request
        lead_data = {"error": "Could not parse model output", "raw": raw_output}

    return lead_data


# =========================================================================
# ESCALATIONS ENDPOINT — simulates what a human agent would see
# =========================================================================

@app.get("/escalations")
def get_escalations():
    """
    Returns every escalation logged since the server started, most recent
    first. In a real deployment this is where you'd instead check your
    Zendesk/Slack/email inbox -- this endpoint exists so you can verify,
    right now, that "connecting you to a human agent" actually created a
    real record somewhere.
    """
    return {"count": len(escalation_queue), "escalations": list(reversed(escalation_queue))}


# =========================================================================
# HEALTH CHECK — lets you (or a hosting platform) verify the server is up
# =========================================================================

@app.get("/health")
def health():
    return {"status": "ok"}