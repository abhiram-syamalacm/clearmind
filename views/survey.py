"""Student survey: check-in, risk score, AI coach, personal dashboard.

Students can log in, create an account, or continue as a guest. Guest check-ins are
stored anonymously (random ID, no name or login) so they count as survey responses.
"""
import math
import os
import re
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

import coach
import mailer
import storage
from ui import brand_header
from model_core import HISTORY_FEATURE, QUESTIONS, RiskModel, feature_label, scale_answer

def risk_card(result):
    st.markdown(f"""
<div class="cm-card">
  <div style="font-size:.85rem;opacity:.75;margin-bottom:.4rem">Stress risk</div>
  <span class="cm-risk">{result['risk']:.0%}</span>
  <span class="cm-pill {result['band']}">{result['band'].upper()}</span>
  <div style="margin-top:.75rem">{BAND_TEXT[result['band']]}</div>
</div>""", unsafe_allow_html=True)


BAND_TEXT = {
    "low": "Your stress-risk signals look manageable right now. Keep doing what's working.",
    "medium": "Some stress signals are building. A few small changes could help a lot before your exam.",
    "high": "Several stress signals are high. It's worth easing the pressure now -- your coach below can help you plan.",
}
FALLBACK_TIPS = {
    "sleep_quality": "Protect your sleep: aim for 8+ hours, and stop screens 30 minutes before bed. Sleep is when memory consolidates -- an all-nighter usually costs more than it gains.",
    "anxiety_level": "Try box breathing (in 4, hold 4, out 4, hold 4) for two minutes, and write your worries down before studying -- 'expressive writing' has been shown to reduce test anxiety.",
    "study_load": "Break the workload into 25-minute focused blocks (Pomodoro) and use active recall -- quiz yourself instead of rereading.",
    "extracurricular_activities": "See if anything can pause until after the exam. Even freeing up one evening gives you breathing room.",
    "academic_performance": "Pick the 2-3 topics most likely to be tested and practise past questions on those first. Ask your teacher where to focus.",
    HISTORY_FEATURE: "If stress feels familiar or heavy, a school counselor is a great person to check in with -- you don't have to handle it alone.",
}


@st.cache_resource
def load_model():
    return RiskModel()


model = load_model()


# ---------------------------------------------------------------- auth
def auth_screen():
    brand_header()
    st.write("Check in on your exam stress, see what's driving it, and get a plan to feel better before test day.")
    with st.container(border=True):
        st.markdown("**No account? Take it as a guest.**")
        guest_ok = st.checkbox(
            "I understand my answers are saved anonymously (no name, no login) to count toward "
            "ClearMind's survey results, that this isn't medical advice, and -- if I'm under 18 -- "
            "a parent or guardian knows I'm using it.", key="guest_consent")
        if st.button("Continue as guest", type="primary", disabled=not guest_ok):
            st.session_state.user = {"id": storage.create_guest(), "name": "Guest", "guest": True}
            st.rerun()
    login_tab, signup_tab = st.tabs(["Log in", "Create account"])
    with login_tab:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Log in", type="primary"):
                uid = storage.check_login(u.strip(), p)
                if uid:
                    st.session_state.user = {"id": uid, "name": u.strip()}
                    st.rerun()
                st.error("Wrong username or password.")
    with signup_tab:
        with st.form("signup"):
            u = st.text_input("Choose a username (don't use your real name)")
            p = st.text_input("Choose a password (8+ characters)", type="password")
            consent = st.checkbox(
                "I understand ClearMind stores my check-in answers so I can see my dashboard, "
                "that it isn't medical advice, and -- if I'm under 18 -- a parent or guardian "
                "knows I'm using it.")
            if st.form_submit_button("Create account", type="primary"):
                if len(u.strip()) < 3 or len(p) < 8:
                    st.error("Username needs 3+ characters and password 8+.")
                elif storage.is_guest_name(u.strip()):
                    st.error("Usernames can't start with \"guest_\".")
                elif not consent:
                    st.error("Please tick the box to continue.")
                else:
                    uid = storage.create_user(u.strip(), p)
                    if uid is None:
                        st.error("That username is taken.")
                    else:
                        st.session_state.user = {"id": uid, "name": u.strip()}
                        st.rerun()
    st.caption("Not a crisis service. If you're struggling, call or text 988 (US) or visit findahelpline.com.")


STUDENT_KEYS = ["user", "last_result", "chat"]

with st.sidebar:
    st.page_link("views/home.py", label="Home", icon=":material/home:")
    st.page_link("views/science.py", label="The science behind", icon=":material/science:")

if "user" not in st.session_state:
    auth_screen()
    st.stop()

user = st.session_state.user
guest = user.get("guest", False)
history = storage.get_checkins(user["id"])
latest = history[-1] if history else None

with st.sidebar:
    if guest:
        st.write("Using ClearMind as a **guest**. Your answers are saved anonymously.")
    else:
        st.write(f"Signed in as **{user['name']}**")
    if st.button("Exit guest mode" if guest else "Log out"):
        for k in STUDENT_KEYS:
            st.session_state.pop(k, None)
        st.switch_page("views/home.py")
    st.divider()
    st.caption("Need to talk to someone now? Call or text **988** (US), text **HOME** to **741741**, "
               "or visit **findahelpline.com**.")

brand_header()
tab_check, tab_dash = st.tabs(["Check-in & coach", "Dashboard"])


# ---------------------------------------------------------------- reminders
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def reminder_box():
    st.divider()
    st.subheader("Check back in")
    current = storage.get_reminder(user["id"])
    if current:
        label = storage.REMINDER_MODES[current["mode"]].lower()
        when = current["next_send_at"][:10] if current["next_send_at"] else "after your exam date is set"
        st.success(f"Reminders are on: {label}, sent to {current['email']}. Next one: {when}.")
        if st.button("Stop reminders"):
            storage.cancel_reminder(user["id"])
            st.rerun()
        return

    st.write("Checking in regularly shows whether your stress is going up or down. "
             "Get an email when it's time to check in again."
             + (" As a guest, the email also has a private link back to your history." if guest else ""))
    with st.form("reminder"):
        email = st.text_input("Email")
        mode = st.radio("How often?", list(storage.REMINDER_MODES),
                        format_func=storage.REMINDER_MODES.get, horizontal=True)
        ok = st.checkbox("Email me check-in reminders. I can unsubscribe from any email, and my "
                         "address is only used for these reminders.")
        if st.form_submit_button("Remind me"):
            if not EMAIL_RE.match(email.strip()):
                st.error("That doesn't look like an email address.")
            elif not ok:
                st.error("Please tick the box to turn on reminders.")
            elif mode == "exam" and not storage.get_exam_date(user["id"]):
                st.error("Set your exam date on the Dashboard tab first.")
            else:
                storage.set_reminder(user["id"], email, mode)
                if mailer.configured():
                    r = storage.get_reminder(user["id"])
                    token = storage.issue_resume_token(user["id"]) if guest else None
                    try:
                        mailer.send(r["email"], *mailer.welcome_email(
                            guest, r["unsub_token"], token, storage.REMINDER_MODES[mode]))
                    except Exception:
                        st.warning("Reminders are on, but the confirmation email couldn't be sent.")
                st.rerun()


# ---------------------------------------------------------------- check-in
def factor_chart(shap):
    """Horizontal SHAP bars, biggest effect on top: yellow pushes risk up, blue pulls it down."""
    df = pd.DataFrame({"Factor": [feature_label(f) for f in shap], "Effect": list(shap.values())})
    df["Direction"] = df["Effect"].map(lambda v: "Raises risk" if v > 0 else "Lowers risk")
    order = df.reindex(df["Effect"].abs().sort_values(ascending=False).index)["Factor"].tolist()
    chart = alt.Chart(df).mark_bar(cornerRadius=4).encode(
        x=alt.X("Effect:Q", title="Effect on risk (log-odds)"),
        y=alt.Y("Factor:N", sort=order, title=None,
              axis=alt.Axis(labelLimit=220, labelOverlap=False)),
        color=alt.Color("Direction:N", scale=alt.Scale(
            domain=["Raises risk", "Lowers risk"], range=["#FFC718", "#1557F4"]),
            legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Factor", alt.Tooltip("Effect:Q", format="+.2f")],
    ).properties(height=alt.Step(38))
    st.altair_chart(chart, width="stretch")


with tab_check:
    st.subheader("How are you doing?")
    st.caption("Rate each from 1 to 10. Takes about 30 seconds.")
    with st.form("checkin"):
        answers = {}
        for f, q in QUESTIONS.items():
            answers[f] = st.slider(f"{q['prompt']}  \n*1 = {q['low']}, 10 = {q['high']}*", 1, 10, 5)
        hist = st.radio("Have you dealt with mental-health challenges (like anxiety or depression) before?",
                        ["No", "Yes", "Prefer not to say"], horizontal=True)
        submitted = st.form_submit_button("Get my risk score", type="primary")

    if submitted:
        raw = {f: scale_answer(f, v) for f, v in answers.items()}
        raw[HISTORY_FEATURE] = {"No": 0.0, "Yes": 1.0}.get(hist, math.nan)
        result = model.predict(raw)
        storage.add_checkin(user["id"], {**answers, HISTORY_FEATURE: hist}, result)
        st.session_state.last_result = result
        st.session_state.chat = []          # new check-in -> coach starts a fresh conversation
        st.session_state.coach_started = False
        st.rerun()

    result = st.session_state.get("last_result") or (latest and {
        "risk": latest["risk"], "band": latest["band"], "shap": latest["shap"]})
    if result:
        st.divider()
        risk_card(result)
        st.write("")
        st.markdown("**What's driving your score** (yellow pushes your risk up, blue pulls it down)")
        factor_chart(result["shap"])
        st.page_link("views/science.py", label="See the science behind your score",
                     icon=":material/science:")
        top = [f for f, v in sorted(result["shap"].items(), key=lambda kv: -kv[1]) if v > 0][:2]

        # ------------------------------------------------ coach (starts automatically)
        st.divider()
        st.subheader("Your ClearMind coach")
        api_key = coach.get_api_key(st.secrets)
        model_name = os.environ.get("GEMINI_MODEL")  # None -> coach.DEFAULT_MODELS
        try:
            model_name = st.secrets.get("GEMINI_MODEL") or model_name
        except Exception:
            pass
        exam = storage.get_exam_date(user["id"])
        context = coach.checkin_context(result, feature_label)
        if exam:
            context += f"\nExam date: {exam} (today is {date.today().isoformat()})."

        msgs = st.session_state.setdefault("chat", [])
        if not msgs and not st.session_state.get("coach_started"):
            st.session_state.coach_started = True
            opener = None
            if api_key:
                with st.spinner("Your coach is reading your results..."):
                    try:
                        opener = coach.opening_message(api_key, model_name, context)
                    except Exception as e:
                        st.caption(f"(The coach couldn't connect, so here are quick tips instead: {e})")
            msgs.append({"role": "assistant",
                         "content": opener or coach.offline_opener(result, top, feature_label, FALLBACK_TIPS)})

        st.caption("Chats aren't saved. ClearMind Coach is an AI, not a counselor -- it can make mistakes.")
        with st.container(height=440):
            for m in msgs:
                st.chat_message(m["role"]).write(m["content"])
        if not api_key:
            st.caption("Live chat turns on once `GROQ_API_KEY` (free) or `GEMINI_API_KEY` is added to the app's secrets.")
        elif prompt := st.chat_input("Reply to your coach"):
            msgs.append({"role": "user", "content": prompt})
            if coach.is_crisis(prompt):
                answer = coach.CRISIS_MESSAGE
            else:
                try:
                    answer = coach.reply(api_key, model_name, msgs, context)
                except Exception as e:
                    msgs.pop()
                    st.error(f"The coach couldn't reply: {e}")
                    st.stop()
            msgs.append({"role": "assistant", "content": answer})
            st.rerun()

        reminder_box()


# ---------------------------------------------------------------- dashboard
with tab_dash:
    st.subheader("Your dashboard")
    exam = storage.get_exam_date(user["id"])
    exam_day = date.fromisoformat(exam) if exam else None
    if exam_day and exam_day < date.today():
        exam_day = None  # past exams fall off so the picker asks for the next one
    new_exam = st.date_input("Next big exam", value=exam_day, min_value=date.today())
    if new_exam and new_exam.isoformat() != exam:
        storage.set_exam_date(user["id"], new_exam)
        rem = storage.get_reminder(user["id"])
        if rem and rem["mode"] == "exam":   # move the reminder to the new exam date
            storage.set_reminder(user["id"], rem["email"], "exam")
        st.rerun()

    if not history:
        st.info("Do your first check-in to start your dashboard.")
    else:
        df = pd.DataFrame([{"time": pd.to_datetime(h["created_at"]).tz_convert(None),
                            "risk": h["risk"] * 100, **{feature_label(f): h["answers"][f] for f in QUESTIONS}}
                           for h in history]).set_index("time")
        c1, c2, c3 = st.columns(3)
        c1.metric("Check-ins", len(history))
        delta = None if len(history) < 2 else f"{(history[-1]['risk'] - history[-2]['risk']) * 100:+.0f} pts"
        c2.metric("Latest risk", f"{history[-1]['risk']:.0%}", delta, delta_color="inverse")
        if new_exam:
            c3.metric("Days to exam", (new_exam - date.today()).days)

        st.markdown("**Stress risk over time (%)**")
        st.line_chart(df[["risk"]], y_label="Risk %")

        st.markdown("**Your answers over time (1-10)**")
        st.line_chart(df.drop(columns="risk"))

        st.markdown("**Average effect of each factor across your check-ins**")
        avg = pd.DataFrame([h["shap"] for h in history]).mean().to_dict()
        factor_chart(avg)

        bands = pd.Series([h["band"] for h in history]).value_counts()
        st.markdown("**Check-ins by band:** " + ", ".join(f"{b}: {bands.get(b, 0)}" for b in ["low", "medium", "high"]))

        st.download_button("Download my data (CSV)", df.to_csv().encode(), "clearmind_my_data.csv", "text/csv")

    if guest:
        st.info("You're a guest, so you can't log back in to this dashboard later. "
                "Create an account to track yourself over time.")
    else:
        with st.expander("Delete my account"):
            st.write("This permanently deletes your account and every check-in.")
            if st.checkbox("Yes, I'm sure") and st.button("Delete everything", type="primary"):
                storage.delete_user(user["id"])
                for k in STUDENT_KEYS:
                    st.session_state.pop(k, None)
                st.switch_page("views/home.py")
