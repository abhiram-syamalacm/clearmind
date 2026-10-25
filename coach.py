"""Pre-exam coach, written by a general-purpose language model (Gemini, or a free
OpenAI-compatible provider such as Groq). The model never computes the risk score --
it only receives the score and SHAP factors from model_core and explains them."""
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

CRISIS_PATTERNS = re.compile(
    r"\b(kill myself|suicid\w*|end (it|my life)|want to die|self[- ]?harm|"
    r"hurt myself|cutting myself|no reason to live|better off dead)\b", re.I)

CRISIS_MESSAGE = (
    "It sounds like you're going through something really heavy, and I'm glad you said it. "
    "I'm not the right kind of help for this, but people are available right now:\n\n"
    "- **US:** call or text **988** (Suicide & Crisis Lifeline), or text **HOME** to **741741**\n"
    "- **Outside the US:** find a local line at **findahelpline.com**\n"
    "- If you're in immediate danger, call your local emergency number.\n\n"
    "Please also tell a trusted adult -- a parent, teacher, or school counselor -- today."
)

SYSTEM_PROMPT = """You are ClearMind Coach, a calm, warm study and stress-relief coach for high
school and college students preparing for exams.

What you do:
- Help students lower pre-exam stress: breathing exercises (e.g. 4-7-8, box breathing),
  grounding, reframing anxious thoughts, sleep and routine tips, study planning
  (spaced repetition, active recall, Pomodoro), and night-before / morning-of plans.
- Use the student's ClearMind check-in (below) to prioritise advice: focus first on the
  factors that pushed their risk score UP the most, and suggest small, concrete steps.
- Keep replies short (under ~150 words), friendly, and practical. Use plain language.

Hard rules:
- You are not a therapist or doctor. Never diagnose. Don't claim the risk score is a
  medical or psychological assessment -- it is an estimate from a machine-learning model
  trained on survey data.
- Never invent or recompute the risk score; only use the numbers given to you.
- If the student mentions self-harm, suicide, abuse, or being in danger, stop coaching,
  respond with empathy, and direct them to 988 (US call/text), text HOME to 741741, or
  findahelpline.com, and encourage them to tell a trusted adult.
- If stress seems persistent or severe, gently suggest talking to a school counselor.
"""


def is_crisis(text):
    return bool(CRISIS_PATTERNS.search(text))


def checkin_context(result, labels):
    if not result:
        return "The student hasn't done a check-in yet. You can suggest they do one."
    lines = [f"Latest check-in: risk {result['risk']:.0%} ({result['band']} band)."]
    lines.append("Factor contributions (positive = pushed risk up, in log-odds):")
    for f, v in sorted(result["shap"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {labels(f)}: {v:+.2f}")
    return "\n".join(lines)


@dataclass
class Provider:
    """Which AI service writes the coach's replies. The coach is a general-purpose
    language model: it is not trained on ClearMind data, it only receives the
    student's score and factors in each request."""
    kind: str                 # "gemini" or "openai" (any OpenAI-compatible API, e.g. Groq)
    key: str
    base_url: str = ""
    models: tuple = ()


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Tried in order; Groq retires model names over time, so later entries are backups.
GROQ_MODELS = ("openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant")


def _setting(secrets, name):
    try:
        val = secrets.get(name)
    except Exception:  # no secrets.toml present
        val = None
    return val or os.environ.get(name)


def get_api_key(secrets):
    """The configured coach provider, or None. Gemini wins if both keys are set.
    COACH_API_BASE / COACH_API_KEY / COACH_MODEL point it at any other
    OpenAI-compatible service."""
    if key := _setting(secrets, "GEMINI_API_KEY"):
        return Provider("gemini", key)
    if key := _setting(secrets, "GROQ_API_KEY"):
        return Provider("openai", key, GROQ_BASE_URL, GROQ_MODELS)
    if key := _setting(secrets, "COACH_API_KEY"):
        model = _setting(secrets, "COACH_MODEL")
        return Provider("openai", key, _setting(secrets, "COACH_API_BASE") or GROQ_BASE_URL,
                        (model,) if model else GROQ_MODELS)
    return None


OPENER_PROMPT = (
    "I just finished my ClearMind check-in. Start the conversation: in 2-3 short sentences, "
    "explain what my score means in plain language, then give me ONE concrete thing to do today "
    "for my biggest risk factor, and end by asking what exam I'm preparing for.")


def opening_message(provider, model_name, context):
    """Coach speaks first, right after a check-in. The prompt itself isn't shown."""
    return reply(provider, model_name, [{"role": "user", "content": OPENER_PROMPT}], context)


def offline_opener(result, top_factors, labels, tips):
    """Rule-based opener used when no AI provider is configured."""
    lines = [f"Your stress risk is **{result['risk']:.0%}** ({result['band']})."]
    if top_factors:
        lines.append(f"The biggest thing pushing it up is **{labels(top_factors[0]).lower()}**, "
                     "so here's where I'd start:")
        lines += [f"- **{labels(f)}:** {tips[f]}" for f in top_factors]
    else:
        lines.append("Nothing is pushing your risk up much right now. Keep your sleep and routine "
                     "steady through exam week.")
    lines.append("Check in again in a few days to see how things change.")
    return "\n\n".join(lines)


# Tried in order when no GEMINI_MODEL is configured, so a retired model name
# falls through to the next one instead of breaking the coach.
DEFAULT_MODELS = ["gemini-flash-latest", "gemini-2.5-flash"]


def _model_missing(err):
    msg = str(err).lower()
    return "404" in msg or "not found" in msg or "not supported" in msg


def reply(provider, model_name, history, context):
    """history: list of {"role": "user"|"assistant", "content": str}.
    model_name (Gemini only) overrides the default model list."""
    if provider.kind == "openai":
        return _openai_reply(provider, history, context)
    return _gemini_reply(provider.key, model_name, history, context)


def _openai_reply(provider, history, context):
    """Chat completion from any OpenAI-compatible API (Groq by default), stdlib only."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context}] + [
        {"role": m["role"], "content": m["content"]} for m in history]
    for i, model in enumerate(provider.models):
        req = urllib.request.Request(
            provider.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps({"model": model, "messages": messages,
                             "temperature": 0.7, "max_tokens": 600}).encode(),
            headers={"Authorization": f"Bearer {provider.key}",
                     "Content-Type": "application/json", "User-Agent": "ClearMind"})
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code == 429:
                raise RuntimeError("The coach is busy right now (free-tier limit). "
                                   "Try again in a minute.") from None
            retired = e.code in (400, 404) and any(
                s in body.lower() for s in ("model_not_found", "does not exist", "decommissioned",
                                            "not found"))
            if not retired or i == len(provider.models) - 1:
                raise RuntimeError(f"HTTP {e.code}: {body[:300]}") from None


def _gemini_reply(api_key, model_name, history, context):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    contents = [
        types.Content(role="user" if m["role"] == "user" else "model",
                      parts=[types.Part(text=m["content"])])
        for m in history
    ]
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT + "\n\n" + context,
                                         temperature=0.7)
    candidates = [model_name] if model_name else DEFAULT_MODELS
    for i, name in enumerate(candidates):
        try:
            return client.models.generate_content(model=name, contents=contents, config=config).text
        except Exception as e:
            if i == len(candidates) - 1 or not _model_missing(e):
                raise
