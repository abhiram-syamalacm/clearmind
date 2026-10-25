"""Check-in reminder emails over plain SMTP.

Works with any SMTP provider. Settings come from the environment or Streamlit
secrets: SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASSWORD, MAIL_FROM,
and APP_URL (the public site address used in links).
"""
import html
import os
import smtplib
from email.message import EmailMessage

SETTINGS = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "MAIL_FROM", "APP_URL"]


def setting(name, default=None):
    val = os.environ.get(name)
    if not val:
        try:
            import streamlit as st
            val = st.secrets.get(name)
        except Exception:  # no secrets.toml, or not running under Streamlit
            val = None
    return val or default


def configured():
    return all(setting(n) for n in ["SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "MAIL_FROM", "APP_URL"])


def send(to, subject, text, html_body):
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = setting("MAIL_FROM"), to, subject
    msg.set_content(text)
    msg.add_alternative(html_body, subtype="html")
    with smtplib.SMTP(setting("SMTP_HOST"), int(setting("SMTP_PORT", 587)), timeout=30) as s:
        s.starttls()
        s.login(setting("SMTP_USER"), setting("SMTP_PASSWORD"))
        s.send_message(msg)


def _links(unsub_token, resume_token):
    base = setting("APP_URL", "").rstrip("/")
    checkin = f"{base}/?resume={resume_token}" if resume_token else f"{base}/survey"
    return checkin, f"{base}/?unsubscribe={unsub_token}"


def _wrap(heading, paragraphs, button_text, button_url, unsub_url):
    """Plain text and simple branded HTML versions of one email."""
    text = "\n\n".join([heading, *paragraphs, f"{button_text}: {button_url}",
                        "Need to talk to someone now? Call or text 988 (US), or visit findahelpline.com.",
                        f"Stop these emails: {unsub_url}"])
    paras = "".join(f'<p style="margin:0 0 14px">{html.escape(p)}</p>' for p in paragraphs)
    body = f"""<div style="font-family:Arial,sans-serif;max-width:520px;margin:auto;color:#14213D">
<p style="font-size:26px;font-weight:bold;margin:0 0 18px"><span style="color:#1557F4">clear</span><span style="color:#FFC718">mind</span></p>
<h2 style="margin:0 0 14px">{html.escape(heading)}</h2>{paras}
<p style="margin:22px 0"><a href="{html.escape(button_url)}" style="background:#1557F4;color:#fff;padding:12px 22px;border-radius:999px;text-decoration:none;font-weight:bold">{html.escape(button_text)}</a></p>
<p style="font-size:12px;color:#667">Need to talk to someone now? Call or text 988 (US), or visit findahelpline.com.<br>
ClearMind is an educational tool, not medical advice.<br>
<a href="{html.escape(unsub_url)}" style="color:#667">Stop these emails</a></p></div>"""
    return text, body


def reminder_email(is_guest, unsub_token, resume_token=None, last_band=None, days_to_exam=None):
    checkin_url, unsub_url = _links(unsub_token, resume_token)
    paras = ["It's time for your ClearMind check-in. It takes about 30 seconds, and you'll see how "
             "your stress risk has changed since last time."]
    if last_band:
        paras.append(f"Last time your stress risk was {last_band}.")
    if days_to_exam is not None and days_to_exam >= 0:
        paras.append("Your exam is tomorrow. You've got this: get some sleep tonight."
                     if days_to_exam <= 1 else f"Your exam is in {days_to_exam} days.")
    if is_guest:
        paras.append("This link opens your own guest history, so please don't forward this email.")
    else:
        paras.append("Log in with your ClearMind username to check in.")
    return ("Time for your ClearMind check-in",
            *_wrap("Time for your check-in", paras, "Check in now", checkin_url, unsub_url))


def welcome_email(is_guest, unsub_token, resume_token, mode_label):
    checkin_url, unsub_url = _links(unsub_token, resume_token)
    paras = [f"You're signed up for ClearMind check-in reminders ({mode_label.lower()}).",
             "Checking in regularly is how your dashboard shows whether your stress is going up or down."]
    if is_guest:
        paras.append("Use the button below any time to come back to your guest history. "
                     "Please don't forward this email.")
    return ("You're signed up for ClearMind reminders",
            *_wrap("Reminders are on", paras, "Open ClearMind", checkin_url, unsub_url))
