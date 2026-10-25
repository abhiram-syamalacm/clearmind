"""Storage for student accounts and check-ins.

Uses DATABASE_URL (from Streamlit secrets or the environment) when set -- e.g. a
Supabase Postgres connection string -- and falls back to a local SQLite file.
Streamlit Community Cloud wipes local files on redeploy, so a hosted version
must set DATABASE_URL.

Passwords are salted and hashed with PBKDF2 (Python stdlib). Chat messages
are never stored. Reminder emails are optional; the address is used only to send
check-in reminders and is deleted on unsubscribe or account deletion.
"""
import hashlib
import hmac
import json
import os
import secrets
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from sqlalchemy import (Column, Float, ForeignKey, Integer, LargeBinary, MetaData,
                        String, Table, Text, create_engine, delete, insert, select, update)
from sqlalchemy.exc import IntegrityError

LOCAL_DB = Path(__file__).parent / "data" / "clearmind.db"

metadata = MetaData()
users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String(64), unique=True, nullable=False),
    Column("pw_hash", LargeBinary, nullable=False),
    Column("salt", LargeBinary, nullable=False),
    Column("exam_date", String(10)),
    Column("created_at", String(32), nullable=False),
)
checkins = Table(
    "checkins", metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("created_at", String(32), nullable=False),
    Column("answers", Text, nullable=False),
    Column("risk", Float, nullable=False),
    Column("band", String(10), nullable=False),
    Column("shap", Text, nullable=False),
)
reminders = Table(
    "reminders", metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("email", String(254), nullable=False),
    Column("mode", String(10), nullable=False),          # "3days", "weekly" or "exam"
    Column("next_send_at", String(32)),                  # None once an exam reminder is sent
    Column("resume_hash", String(64)),                   # sha256 of the latest resume link token
    Column("resume_expires", String(32)),
    Column("unsub_token", String(64), unique=True, nullable=False),
    Column("created_at", String(32), nullable=False),
    Column("last_sent_at", String(32)),
)

REMINDER_MODES = {"weekly": "Once a week", "3days": "Every 3 days", "exam": "The day before my exam"}
RESUME_TTL_DAYS = 30


def _database_url():
    url = os.environ.get("DATABASE_URL")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("DATABASE_URL")
        except Exception:  # no secrets.toml present
            url = None
    return url


@lru_cache(maxsize=1)
def _engine():
    url = _database_url()
    if url:
        # Supabase gives "postgresql://..."; point SQLAlchemy at the psycopg 3 driver.
        url = url.replace("postgres://", "postgresql://", 1).replace(
            "postgresql://", "postgresql+psycopg://", 1)
        # prepare_threshold=None keeps it working through Supabase's transaction pooler.
        engine = create_engine(url, pool_pre_ping=True, connect_args={"prepare_threshold": None})
    else:
        LOCAL_DB.parent.mkdir(exist_ok=True)
        engine = create_engine(f"sqlite:///{LOCAL_DB}")

        from sqlalchemy import event

        @event.listens_for(engine, "connect")
        def _fk_on(conn, _):
            conn.execute("PRAGMA foreign_keys = ON")
    metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        # Supabase exposes the public schema over its REST API. Row-level security
        # with no policies blocks that path; this app connects as the table owner,
        # which RLS doesn't apply to.
        with engine.begin() as c:
            c.exec_driver_sql("ALTER TABLE users ENABLE ROW LEVEL SECURITY")
            c.exec_driver_sql("ALTER TABLE checkins ENABLE ROW LEVEL SECURITY")
            c.exec_driver_sql("ALTER TABLE reminders ENABLE ROW LEVEL SECURITY")
    return engine


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)


def create_user(username, password):
    salt = os.urandom(16)
    try:
        with _engine().begin() as c:
            res = c.execute(insert(users).values(
                username=username, pw_hash=_hash(password, salt), salt=salt, created_at=_now()))
            return res.inserted_primary_key[0]
    except IntegrityError:
        return None


GUEST_PREFIX = "guest_"


def create_guest():
    """Anonymous student: a hidden record with a random name and an unknowable
    password, so guest surveys are stored and counted like everyone else's."""
    while True:
        uid = create_user(GUEST_PREFIX + secrets.token_hex(4), secrets.token_urlsafe(24))
        if uid:
            return uid


def is_guest_name(username):
    return username.lower().startswith(GUEST_PREFIX)


def check_login(username, password):
    with _engine().connect() as c:
        row = c.execute(select(users.c.id, users.c.pw_hash, users.c.salt)
                        .where(users.c.username == username)).first()
    if row and hmac.compare_digest(bytes(row.pw_hash), _hash(password, bytes(row.salt))):
        return row.id
    return None


def get_exam_date(user_id):
    with _engine().connect() as c:
        return c.execute(select(users.c.exam_date).where(users.c.id == user_id)).scalar()


def set_exam_date(user_id, exam_date):
    with _engine().begin() as c:
        c.execute(update(users).where(users.c.id == user_id)
                  .values(exam_date=exam_date.isoformat() if exam_date else None))


def add_checkin(user_id, answers, result):
    with _engine().begin() as c:
        c.execute(insert(checkins).values(
            user_id=user_id, created_at=_now(), answers=json.dumps(answers),
            risk=result["risk"], band=result["band"], shap=json.dumps(result["shap"])))


def get_checkins(user_id):
    with _engine().connect() as c:
        rows = c.execute(select(checkins).where(checkins.c.user_id == user_id)
                         .order_by(checkins.c.created_at)).all()
    return [{"created_at": r.created_at, "answers": json.loads(r.answers), "risk": r.risk,
             "band": r.band, "shap": json.loads(r.shap)} for r in rows]


def list_users():
    """Admin: every account with its check-in count, latest risk and last activity."""
    with _engine().connect() as c:
        us = c.execute(select(users.c.id, users.c.username, users.c.created_at,
                              users.c.exam_date).order_by(users.c.username)).all()
        cks = c.execute(select(checkins.c.user_id, checkins.c.created_at, checkins.c.risk)
                        .order_by(checkins.c.created_at)).all()
    stats = {}
    for r in cks:
        s = stats.setdefault(r.user_id, {"n": 0, "last": None, "risk": None})
        s["n"] += 1
        s["last"], s["risk"] = r.created_at, r.risk
    return [{"id": u.id, "username": u.username,
             "type": "guest" if is_guest_name(u.username) else "account",
             "created_at": u.created_at,
             "exam_date": u.exam_date, "checkins": stats.get(u.id, {}).get("n", 0),
             "latest_risk": stats.get(u.id, {}).get("risk"),
             "last_checkin": stats.get(u.id, {}).get("last")} for u in us]


def all_checkins():
    """Admin: every check-in, keyed by user id (no usernames) for de-identified export."""
    with _engine().connect() as c:
        rows = c.execute(select(checkins).order_by(checkins.c.created_at)).all()
    return [{"user_id": r.user_id, "created_at": r.created_at, "risk": r.risk, "band": r.band,
             "answers": json.loads(r.answers), "shap": json.loads(r.shap)} for r in rows]


# ---------------------------------------------------------------- reminders
def _next_send(mode, exam_date, after):
    if mode == "3days":
        return after + timedelta(days=3)
    if mode == "weekly":
        return after + timedelta(days=7)
    if mode == "exam" and exam_date:
        send = datetime.combine(date.fromisoformat(exam_date) - timedelta(days=1),
                                datetime.min.time(), timezone.utc)
        return send if send > after else None
    return None


def set_reminder(user_id, email, mode):
    """Create or replace a student's reminder. Returns the next send time (or None)."""
    now = datetime.now(timezone.utc)
    nxt = _next_send(mode, get_exam_date(user_id), now)
    values = {"email": email.strip(), "mode": mode,
              "next_send_at": nxt.isoformat(timespec="seconds") if nxt else None}
    with _engine().begin() as c:
        exists = c.execute(select(reminders.c.user_id).where(reminders.c.user_id == user_id)).first()
        if exists:
            c.execute(update(reminders).where(reminders.c.user_id == user_id).values(**values))
        else:
            c.execute(insert(reminders).values(user_id=user_id, unsub_token=secrets.token_urlsafe(24),
                                               created_at=_now(), **values))
    return nxt


def get_reminder(user_id):
    with _engine().connect() as c:
        r = c.execute(select(reminders).where(reminders.c.user_id == user_id)).first()
    return dict(r._mapping) if r else None


def cancel_reminder(user_id):
    with _engine().begin() as c:
        c.execute(delete(reminders).where(reminders.c.user_id == user_id))


def unsubscribe(token):
    """One-click unsubscribe from an email link. Deletes the stored address."""
    with _engine().begin() as c:
        return c.execute(delete(reminders).where(reminders.c.unsub_token == token)).rowcount > 0


def _token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def issue_resume_token(user_id):
    """New single-purpose link token for a guest; replaces any earlier one."""
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(days=RESUME_TTL_DAYS)
    with _engine().begin() as c:
        c.execute(update(reminders).where(reminders.c.user_id == user_id).values(
            resume_hash=_token_hash(token), resume_expires=expires.isoformat(timespec="seconds")))
    return token


def resume_user(token):
    """Guest id for a valid, unexpired resume link, else None. Account holders log in instead."""
    with _engine().connect() as c:
        row = c.execute(select(reminders.c.user_id, reminders.c.resume_expires, users.c.username)
                        .join(users, users.c.id == reminders.c.user_id)
                        .where(reminders.c.resume_hash == _token_hash(token))).first()
    if not row or not is_guest_name(row.username):
        return None
    if datetime.fromisoformat(row.resume_expires) < datetime.now(timezone.utc):
        return None
    return row.user_id


def due_reminders(now=None):
    now = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    with _engine().connect() as c:
        rows = c.execute(select(reminders, users.c.username, users.c.exam_date)
                         .join(users, users.c.id == reminders.c.user_id)
                         .where(reminders.c.next_send_at.is_not(None),
                                reminders.c.next_send_at <= now)).all()
    return [dict(r._mapping) for r in rows]


def mark_reminder_sent(user_id, mode, exam_date):
    now = datetime.now(timezone.utc)
    nxt = None if mode == "exam" else _next_send(mode, exam_date, now)
    with _engine().begin() as c:
        c.execute(update(reminders).where(reminders.c.user_id == user_id).values(
            last_sent_at=now.isoformat(timespec="seconds"),
            next_send_at=nxt.isoformat(timespec="seconds") if nxt else None))


def reminder_count():
    with _engine().connect() as c:
        return len(c.execute(select(reminders.c.user_id)).all())


def delete_user(user_id):
    # Delete dependent rows explicitly too, in case the database doesn't enforce cascades.
    with _engine().begin() as c:
        c.execute(delete(reminders).where(reminders.c.user_id == user_id))
        c.execute(delete(checkins).where(checkins.c.user_id == user_id))
        c.execute(delete(users).where(users.c.id == user_id))


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
