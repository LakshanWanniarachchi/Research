"""
User accounts and sign-in
=========================

Who is allowed to look at the dashboard, and how the system knows it is them.

Two different kinds of caller reach this system, and they are authenticated in
two different ways. Conflating them would be a mistake.

  A PERSON  signs in with a username and password and gets a session cookie.
            People forget passwords, use the same one everywhere, and share
            browsers, so the password is never stored - only a salted hash of
            it - and the session expires.

  A DEVICE  has no browser, no cookie store and nobody to type a password. An
            ESP32 on someone's chest identifies itself with a device id and a
            long random token held in its firmware. That is checked in app.py
            against the patient the device is bound to, not here.

Password storage
----------------
`generate_password_hash` applies a salted PBKDF2 hash. The stored value cannot
be turned back into the password, so somebody who copies ecg.db still cannot
sign in as anyone or learn a password that person may have reused elsewhere.
"""

import os
import secrets
from functools import wraps

from flask import (Blueprint, flash, redirect, render_template, request,
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import models

auth_bp = Blueprint("auth", __name__)

# How long a session stays valid without activity.
SESSION_MINUTES = 60


# ---------------------------------------------------------------------------
# 1. Requiring a signed-in user
# ---------------------------------------------------------------------------
def login_required(view):
    """Refuse a page to anyone not signed in, and send them to the login form.

    The requested path is carried through as `next`, so signing in returns the
    user to the page they actually asked for rather than dumping them on the
    dashboard. That matters when a link to one patient's record is shared
    between colleagues.
    """
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user():
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    """Restrict a view to administrators."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if not user:
            return redirect(url_for("auth.login", next=request.path))
        if user["role"] != "admin":
            flash("That page needs an administrator account.", "error")
            return redirect(url_for("dashboard"))
        return view(*args, **kwargs)
    return wrapped


def current_user():
    """The signed-in user as a database row, or None.

    Read fresh from the database rather than from the session, so that a
    deleted or changed account takes effect on the next request instead of
    lasting until the cookie expires.
    """
    uid = session.get("user_id")
    if uid is None:
        return None
    conn = models.connect()
    try:
        return models.get_user(conn, uid)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 2. Sign in and out
# ---------------------------------------------------------------------------
@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", next=request.args.get("next", ""))

    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""

    conn = models.connect()
    try:
        user = models.get_user_by_name(conn, username)

        # One message for both "no such user" and "wrong password". Telling the
        # caller which of the two it was would let them discover valid
        # usernames by trying names until the message changed.
        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Incorrect username or password.", "error")
            return render_template("login.html",
                                   next=request.form.get("next", "")), 401

        session.clear()
        session["user_id"] = user["id"]
        session.permanent = True
        models.touch_last_login(conn, user["id"])
    finally:
        conn.close()

    # Only accept a relative path, so a crafted link cannot use the login form
    # to bounce somebody to another site after they sign in.
    nxt = request.form.get("next", "")
    if not nxt.startswith("/"):
        nxt = url_for("dashboard")
    return redirect(nxt)


@auth_bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("Signed out.", "ok")
    return redirect(url_for("auth.login"))


# ---------------------------------------------------------------------------
# 3. First-run account
# ---------------------------------------------------------------------------
def ensure_first_user(conn):
    """Create an administrator if the system has no accounts at all.

    A fresh install with no accounts cannot be signed into, and a hard-coded
    default password would ship a known credential in a public repository. So
    the first account is created with a random password which is printed once,
    to the console, at start-up. If the operator misses it they can delete the
    row and restart.

    The password can be set explicitly through ECG_ADMIN_PASSWORD, which is how
    the automated tests create a known account.
    """
    if models.count_users(conn) > 0:
        return None

    password = os.environ.get("ECG_ADMIN_PASSWORD") or secrets.token_urlsafe(12)
    models.create_user(conn, "admin", generate_password_hash(password), "admin")
    return password


def new_device_token():
    """A long random secret for one device to identify itself with."""
    return secrets.token_urlsafe(24)


def hash_password(password):
    """Exposed so that scripts and tests create hashes the same way."""
    return generate_password_hash(password)
