"""
ECG pages: live monitor, history, one report
============================================

The live page shows what is arriving over MQTT right now; the history and
report pages show what was stored. Both read through `models`, and every query
carries the signed-in user's id, so a report id guessed in the URL returns 404
rather than somebody else's recording.

Nothing here scores anything. Windows are scored where they arrive, in the MQTT
ingester (`app.score_mqtt_window`), so there is exactly one place in the system
where an ECG becomes a verdict.
"""

import json

from flask import (Blueprint, abort, jsonify, render_template, request)

import auth
import models

ecg_bp = Blueprint("ecg", __name__, url_prefix="/ecg")

HISTORY_PAGE_SIZE = 15


def _registry():
    """The live MQTT stream registry, or None if the transport is off."""
    try:
        import mqtt_bridge
        return mqtt_bridge.STREAMS
    except Exception:                                  # pragma: no cover
        return None


def _owned_device_ids(conn, user_id):
    return [row["device_id"] for row in models.list_devices(conn, user_id)]


# ---------------------------------------------------------------------------
# Live monitor
# ---------------------------------------------------------------------------
@ecg_bp.route("/live")
@auth.login_required
def live():
    user = auth.current_user()
    conn = models.connect()
    try:
        devices = models.list_devices(conn, user["id"])
    finally:
        conn.close()
    selected = request.args.get("device") or (devices[0]["device_id"] if devices else None)
    return render_template("ecg_live.html", user=user, devices=devices,
                           selected=selected)


@ecg_bp.route("/device")
@auth.login_required
def device_stream():
    """What the device is sending, whether or not it is a usable ECG.

    The live page answers "is there a valid ECG?"; this one answers the
    question that comes first - "is anything arriving at all?" - and reads the
    same ownership-checked endpoint, so it adds no new way to see a device.
    """
    user = auth.current_user()
    conn = models.connect()
    try:
        devices = models.list_devices(conn, user["id"])
    finally:
        conn.close()
    selected = request.args.get("device") or (devices[0]["device_id"] if devices else None)
    return render_template("ecg_device.html", user=user, devices=devices,
                           selected=selected)


@ecg_bp.route("/api/live")
@auth.login_required
def api_live():
    """Polled by the live page, several times a second.

    The device must belong to the caller. Without that check, anyone signed in
    could watch anyone else's heartbeat by typing a device id into the query
    string, which is the same leak as reading their reports.
    """
    user = auth.current_user()
    device_id = request.args.get("device")
    conn = models.connect()
    try:
        owned = _owned_device_ids(conn, user["id"])
    finally:
        conn.close()

    # Ownership is checked before anything else, including the "you have no
    # devices" shortcut. Answering that first would tell a caller with no
    # devices something different from a caller with one, which is a way of
    # confirming that a device id exists.
    if device_id is not None and device_id not in owned:
        abort(404)
    if not owned:
        return jsonify({"running": False, "no_device": True,
                        "diagnosis": "No device is linked to your account yet."})
    if device_id is None:
        device_id = owned[0]

    registry = _registry()
    if registry is None:
        return jsonify({"running": False, "device_id": device_id,
                        "diagnosis": "The MQTT transport is not enabled on the server."})

    # Changing the operating point moves the threshold for the next window;
    # it never re-scores a stored one, because a report records the point it
    # was judged at.
    mode = request.args.get("mode")
    stream = registry.get(device_id, create=False)
    if mode and stream is not None:
        import app
        if mode in app.THRESHOLDS:
            stream.mode = mode

    snapshot = registry.snapshot(device_id)
    snapshot["running"] = True
    snapshot["thresholds"] = _thresholds()
    snapshot["broker"] = _broker_state()
    return jsonify(snapshot)


def _thresholds():
    import app
    return app.THRESHOLDS


def _broker_state():
    import mqtt_bridge
    client = getattr(mqtt_bridge, "CLIENT", None)
    return {
        "enabled": mqtt_bridge.MQTT_ENABLED,
        "host": mqtt_bridge.MQTT_HOST,
        "port": mqtt_bridge.MQTT_PORT,
        "connected": bool(client and client.is_connected()),
        "messages": mqtt_bridge.STREAMS.n_messages,
        "rejected": mqtt_bridge.STREAMS.n_rejected,
        "last_rejection": mqtt_bridge.STREAMS.last_rejection,
    }


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
@ecg_bp.route("/history")
@auth.login_required
def history():
    user = auth.current_user()
    result = request.args.get("result") or None
    if result not in ("mi", "normal", None):
        result = None
    date_from = request.args.get("from") or None
    date_to = request.args.get("to") or None
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1

    conn = models.connect()
    try:
        total = models.count_ecg_reports(conn, user["id"], result=result,
                                         date_from=date_from, date_to=date_to)
        reports = models.list_ecg_reports(
            conn, user["id"], limit=HISTORY_PAGE_SIZE,
            offset=(page - 1) * HISTORY_PAGE_SIZE, result=result,
            date_from=date_from, date_to=date_to)
        stats = models.ecg_report_stats(conn, user["id"])
    finally:
        conn.close()

    pages = max(1, -(-total // HISTORY_PAGE_SIZE))     # ceiling division
    return render_template("ecg_history.html", user=user, reports=reports,
                           stats=stats, total=total, page=page, pages=pages,
                           filters={"result": result, "from": date_from,
                                    "to": date_to})


# ---------------------------------------------------------------------------
# One report
# ---------------------------------------------------------------------------
@ecg_bp.route("/<int:report_id>")
@auth.login_required
def report(report_id):
    user = auth.current_user()
    conn = models.connect()
    try:
        row = models.get_ecg_report(conn, user["id"], report_id)
    finally:
        conn.close()
    if row is None:
        # Not "forbidden": a user has no business learning that a report id
        # exists on another account.
        abort(404)

    signal = models.blob_to_signal(row["signal"])
    quality = {}
    if row["quality_json"]:
        try:
            quality = json.loads(row["quality_json"])
        except ValueError:
            quality = {}

    return render_template("ecg_report.html", user=user, report=row,
                           signal=[float(v) for v in signal], quality=quality)
