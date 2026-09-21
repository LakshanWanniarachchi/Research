"""
Configuration for the ECG screening system
==========================================

Settings come from the environment, and the environment can be filled from a
`.env` file sitting next to this module. `.env` is not in version control;
`.env.example` is, and lists every name with a safe placeholder.

Reading `.env` by hand rather than adding python-dotenv keeps the dependency
list honest: the file format used here is one `KEY=value` per line, which is
not worth a package. Real environment variables always win over the file, so a
deployment can set them the normal way without deleting anything.

Nothing here has a secret as its default. A missing secret is reported by
`missing_required()` rather than silently replaced with something guessable.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(HERE, ".env")


def load_env(path=ENV_PATH, override=False):
    """Load KEY=value lines from a .env file into os.environ.

    Blank lines and lines starting with '#' are skipped. Values may be quoted.
    Existing environment variables are left alone unless override=True, so an
    explicitly exported variable beats the file.
    """
    if not os.path.exists(path):
        return {}
    loaded = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
    return loaded


load_env()


def _flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# --- Web application -------------------------------------------------------
SECRET_KEY = os.environ.get("ECG_SECRET_KEY")          # None -> per-process key
ADMIN_PASSWORD = os.environ.get("ECG_ADMIN_PASSWORD")  # first-run seeding only
HOST = os.environ.get("ECG_HOST", "127.0.0.1")
PORT = int(os.environ.get("ECG_PORT", "5000"))

# --- Data and model --------------------------------------------------------
DB_PATH = os.environ.get("ECG_DB_PATH") or os.path.join(HERE, "ecg.db")
MODEL_PATH = os.environ.get("ECG_MODEL_PATH") or os.path.join(HERE, "model.keras")

# --- MQTT ------------------------------------------------------------------
MQTT_ENABLED = _flag("MQTT_ENABLED")
MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USERNAME = os.environ.get("MQTT_USERNAME") or None
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD") or None
MQTT_CLIENT_ID = os.environ.get("MQTT_CLIENT_ID", "ecg-server")
MQTT_TLS = _flag("MQTT_TLS")
MQTT_CA_CERT = os.environ.get("MQTT_CA_CERT") or None
REQUIRE_DEVICE_TOKEN = _flag("REQUIRE_DEVICE_TOKEN")

# Default device for the single-device prototype views.
LIVE_DEVICE_ID = os.environ.get("ECG_LIVE_DEVICE_ID", "ESP32-001")


def missing_required(for_mqtt=None):
    """Names that should be set but are not, so start-up can say so plainly."""
    missing = []
    if not SECRET_KEY:
        missing.append("ECG_SECRET_KEY")
    if (MQTT_ENABLED if for_mqtt is None else for_mqtt) and not MQTT_USERNAME:
        missing.append("MQTT_USERNAME")
    return missing


def describe():
    """A one-line summary for the start-up banner. Never prints a secret."""
    transport = (f"mqtt {MQTT_HOST}:{MQTT_PORT}"
                 f"{' TLS' if MQTT_TLS else ''}"
                 f" as {MQTT_USERNAME or 'anonymous'}") if MQTT_ENABLED else "mqtt disabled"
    return f"db={os.path.basename(DB_PATH)} | {transport}"
