"""
Accounts: registration, profile, devices
========================================

A blueprint rather than more routes in app.py, because these three pages have
nothing to do with the model and everything to do with who is asking. Keeping
them separate also keeps the rule they all share in one file: a query about
devices or reports always carries the signed-in user's id.

The existing routes in app.py are left where they are, so `url_for("dashboard")`
and the templates that call it keep working.
"""

import datetime
import re
import sqlite3

from flask import (Blueprint, flash, redirect, render_template, request,
                   session, url_for)

import auth
import models

account_bp = Blueprint("account", __name__)

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")

# Ranges wide enough for any adult or child a research prototype might record,
# narrow enough to catch a typed-in centimetre value that is really metres, or
# a birth date in the future.
HEIGHT_RANGE_CM = (50.0, 250.0)
WEIGHT_RANGE_KG = (20.0, 400.0)
MAX_AGE_YEARS = 120


def validate_profile(form):
    """Check the profile fields. Returns (values, errors).

    Every field is optional: someone may want to record a height today and a
    weight next week. What is not allowed is a value that is present and
    impossible, because a height of 17 cm silently poisons anything later
    computed from it.
    """
    values = {"full_name": (form.get("full_name") or "").strip() or None,
              "dob": (form.get("dob") or "").strip() or None,
              "height_cm": None, "weight_kg": None}
    errors = {}

    if values["full_name"] and len(values["full_name"]) > 80:
        errors["full_name"] = "Name must be 80 characters or fewer."

    if values["dob"]:
        try:
            dob = datetime.date.fromisoformat(values["dob"])
        except ValueError:
            errors["dob"] = "Use the date picker, or type the date as YYYY-MM-DD."
        else:
            today = datetime.date.today()
            if dob > today:
                errors["dob"] = "Date of birth cannot be in the future."
            elif (today.year - dob.year) > MAX_AGE_YEARS:
                errors["dob"] = f"Date of birth must be within the last {MAX_AGE_YEARS} years."

    for field, (lo, hi), unit in (("height_cm", HEIGHT_RANGE_CM, "cm"),
                                  ("weight_kg", WEIGHT_RANGE_KG, "kg")):
        raw = (form.get(field) or "").strip()
        if not raw:
            continue
        try:
            number = float(raw)
        except ValueError:
            errors[field] = f"Enter a number in {unit}."
            continue
        if not lo <= number <= hi:
            errors[field] = f"Enter a value between {lo:g} and {hi:g} {unit}."
        else:
            values[field] = round(number, 1)

    return values, errors


def profile_age(dob):
    """Age in whole years, or None. Derived rather than stored, so it cannot
    go stale the way a stored age does."""
    if not dob:
        return None
    try:
        born = datetime.date.fromisoformat(dob)
    except ValueError:
        return None
    today = datetime.date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
@account_bp.route("/register", methods=["GET", "POST"])
def register():
    """Create an account.

    Self-registration exists because the system is meant to be used by the
    person wearing the device, and because user-level data isolation is only
    demonstrable with more than one account. The first account remains the
    seeded admin; these are ordinary users.
    """
    if auth.current_user():
        return redirect(url_for("dashboard"))

    if request.method == "GET":
        return render_template("register.html", values={}, errors={})

    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm") or ""
    values = {"username": username}
    errors = {}

    if not USERNAME_RE.match(username):
        errors["username"] = ("3 to 32 characters: letters, digits, dot, dash "
                              "or underscore.")
    if len(password) < 8:
        errors["password"] = "Use at least 8 characters."
    elif password != confirm:
        errors["confirm"] = "The two passwords do not match."

    conn = models.connect()
    try:
        if not errors and models.get_user_by_name(conn, username):
            errors["username"] = "That username is taken."
        if errors:
            return render_template("register.html", values=values, errors=errors), 400

        user_id = models.create_user(conn, username, auth.hash_password(password),
                                     role="user")
    finally:
        conn.close()

    session.clear()
    session["user_id"] = user_id
    session.permanent = True
    flash("Account created. Add your details so reports carry them.", "ok")
    return redirect(url_for("account.profile"))


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
@account_bp.route("/profile", methods=["GET", "POST"])
@auth.login_required
def profile():
    user = auth.current_user()
    if request.method == "GET":
        values = {"full_name": user["full_name"], "dob": user["dob"],
                  "height_cm": user["height_cm"], "weight_kg": user["weight_kg"]}
        return render_template("profile.html", user=user, values=values,
                               errors={}, age=profile_age(user["dob"]))

    values, errors = validate_profile(request.form)
    if errors:
        return render_template("profile.html", user=user, values=values,
                               errors=errors, age=profile_age(values["dob"])), 400

    conn = models.connect()
    try:
        updated = models.update_user_profile(
            conn, user["id"], full_name=values["full_name"], dob=values["dob"],
            height_cm=values["height_cm"], weight_kg=values["weight_kg"])
    finally:
        conn.close()

    flash("Profile saved.", "ok")
    return render_template("profile.html", user=updated, values=values,
                           errors={}, age=profile_age(values["dob"]))


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------
@account_bp.route("/devices", methods=["GET", "POST"])
@auth.login_required
def devices():
    """List this account's devices, and claim or release one.

    Claiming is how an ECG published under a device id becomes this user's
    recording. The device itself never learns who owns it: it publishes under
    its own id, and this row is the only link.
    """
    user = auth.current_user()
    conn = models.connect()
    try:
        if request.method == "POST":
            action = request.form.get("action", "claim")
            device_id = (request.form.get("device_id") or "").strip()

            if action == "release":
                if models.release_device(conn, user["id"], device_id):
                    flash(f"{device_id} released.", "ok")
                else:
                    flash("That device is not registered to your account.", "error")
            elif not device_id:
                flash("Enter the device id printed by the firmware, e.g. ESP32-001.",
                      "error")
            else:
                try:
                    models.claim_device(conn, user["id"], device_id,
                                        label=(request.form.get("label") or "").strip() or None,
                                        token=auth.new_device_token())
                    flash(f"{device_id} is now linked to your account.", "ok")
                except sqlite3.IntegrityError:
                    # UNIQUE(device_id). Say that it is taken without saying by
                    # whom: the owner of a device is not this caller's business.
                    flash(f"{device_id} is already registered to an account.", "error")
            return redirect(url_for("account.devices"))

        rows = models.list_devices(conn, user["id"])
    finally:
        conn.close()

    live = _live_state_for(rows)
    return render_template("devices.html", user=user, devices=rows, live=live)


def _live_state_for(rows):
    """What the MQTT ingester currently knows about each of these devices.

    Read from the running stream registry rather than from the database, so the
    page reports what is arriving now instead of what was last written down.
    """
    try:
        import mqtt_bridge
        registry = mqtt_bridge.STREAMS
    except Exception:                                  # pragma: no cover
        return {}
    state = {}
    for row in rows:
        stream = registry.get(row["device_id"], create=False)
        state[row["device_id"]] = {
            "online": bool(stream and stream.online),
            "samples": stream.n_samples if stream else 0,
            "rate_hz": stream.measured_rate_hz() if stream else None,
        }
    return state
