"""The orchestrator service: one SuperLink container per training request. Runs on the orchestrator VM, not locally.

Ported from the flowertests prototype; `fedlab orchestrator up` installs it.

API on 127.0.0.1:8080 (reach it over SSH; it has no auth):
  POST   /requests            {"num_nodes": N}   -> bundle for the external hosts
                                                    (endpoint <id>.<domain>:443, via the nginx edge)
  GET    /requests/<id>                          -> SuperLink view: nodes and runs
  POST   /requests/<id>/run   {"app": "<dir>", "run_config": "k=v ..."}   -> submit APPS/<dir> as a run
  DELETE /requests/<id>                          -> tear the SuperLink down

Persistent: state lives under ROOT (a separate data volume); on start, reconcile() restores every recorded
request's SuperLink, so replacing the instance or rebooting loses nothing.
"""
import base64
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = "/opt/orchestrator"
APPS = f"{ROOT}/apps"
CERTS = "/opt/flower/certs"  # one cert for every SuperLink; shortcut: per-request certs
PORTS = range(9000, 9100)
IMAGE = f"flwr/superlink:{os.environ['FLWR_VERSION']}-py3.13-ubuntu24.04"
DOMAIN = os.environ["ENDPOINT_DOMAIN"]  # wildcard-cert domain; each request is <id>.<DOMAIN>:443
PORT_MAP = "/etc/nginx/sl-ports.map"  # nginx maps SNI hostname -> loopback port
FLWR = f"{ROOT}/venv/bin/flwr"
DB = f"{ROOT}/requests.db"
IMAGE_UID = 49999
WATCHDOG_INTERVAL = 10  # seconds between sweeps
OFFLINE_GRACE = int(os.environ.get("OFFLINE_GRACE", "30"))  # how long a node may stay offline during a run


def sh(*cmd, env=None):
    return subprocess.run(
        cmd, check=True, capture_output=True, text=True, env={**os.environ, **(env or {})}
    ).stdout


def flwr(rid, *args):
    return sh(FLWR, *args, env={"FLWR_HOME": f"{ROOT}/requests/{rid}/flwr"})


def q(sql, *args):
    with sqlite3.connect(DB) as c:
        return c.execute(sql, args).fetchall()


def create(n):
    rid = os.urandom(4).hex()
    for port in PORTS:
        try:
            q("INSERT INTO requests VALUES (?, ?, ?)", rid, port, time.time())
            break
        except sqlite3.IntegrityError:
            continue
    else:
        raise RuntimeError("no free SuperLink ports")
    try:
        return _start(rid, port, n)
    except Exception:
        destroy(rid)
        raise


def run_superlink(rid, port):
    """Start request `rid`'s SuperLink container from its persisted state directory."""
    # Fleet, Control and Runtime APIs share the one port in 1.41; loopback only, nginx fronts it
    sh(
        "docker", "run", "-d", "--restart", "unless-stopped", "--name", f"sl-{rid}", "-p", f"127.0.0.1:{port}:8000",
        "-v", f"{CERTS}:/app/certificates:ro", "-v", f"{ROOT}/requests/{rid}/state:/app/state",
        IMAGE, "--host", "0.0.0.0", "--port", "8000",
        "--ssl-ca-certfile", "certificates/ca.crt",
        "--ssl-certfile", "certificates/server.pem",
        "--ssl-keyfile", "certificates/server.key",
        "--enable-supernode-auth", "--isolation", "process",
        "--superexec-auth-secret-file", "state/secret",
        "--database", "state/state.db",
    )  # fmt: skip


def container_state(rid):
    p = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", f"sl-{rid}"], capture_output=True, text=True)
    return "missing" if p.returncode else ("running" if p.stdout.strip() == "true" else "stopped")


def reconcile():
    """After a reboot or instance replacement the database and state directories survive (they live on the data
    volume) but the containers may not: bring every recorded request's SuperLink back, then rebuild the proxy map."""
    for rid, port in q("SELECT id, port FROM requests"):
        try:
            if not os.path.isdir(f"{ROOT}/requests/{rid}/state"):
                print(f"reconcile {rid}: state directory is gone; leaving it for the operator", file=sys.stderr, flush=True)
                continue
            state = container_state(rid)
            if state == "missing":
                run_superlink(rid, port)
            elif state == "stopped":
                sh("docker", "start", f"sl-{rid}")
        except Exception as e:  # noqa: BLE001  one broken request must not block the others
            print(f"reconcile {rid}: {e}", file=sys.stderr, flush=True)
    write_port_map()


def _start(rid, port, n):
    d = f"{ROOT}/requests/{rid}"
    for sub in ("state", "keys", "flwr"):  # state is the only dir the container sees
        os.makedirs(f"{d}/{sub}")
    secret = os.urandom(32)
    with open(f"{d}/state/secret", "wb") as f:
        f.write(secret)
    shutil.chown(f"{d}/state", IMAGE_UID, IMAGE_UID)
    shutil.chown(f"{d}/state/secret", IMAGE_UID, IMAGE_UID)
    os.chmod(f"{d}/state/secret", 0o600)

    with open(f"{d}/flwr/config.toml", "w") as f:
        f.write(
            f'[superlink]\ndefault = "r"\n\n[superlink.r]\naddress = "127.0.0.1:{port}"\n'
            f'root-certificates = "{CERTS}/ca.crt"\n'
        )

    run_superlink(rid, port)
    for _ in range(60):
        try:
            socket.create_connection(("127.0.0.1", port), 1).close()
            break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError("SuperLink did not start listening")

    node_keys = []
    for i in range(n):
        key = f"{d}/keys/node{i}"
        sh("ssh-keygen", "-q", "-t", "ecdsa", "-b", "384", "-N", "", "-f", key)
        for attempt in range(10):  # listening is not the same as serving
            try:
                flwr(rid, "supernode", "register", f"{key}.pub", "r")
                break
            except subprocess.CalledProcessError:
                if attempt == 9:
                    raise
                time.sleep(1)
        node_keys.append(open(key).read())

    write_port_map()
    # Flush everything to disk before the credentials leave. After an unclean detach (instance crash, volume pulled
    # while mounted) XFS can bring back recently written files as zero-length, and a request whose config.toml,
    # secret or keys were lost would be unusable even though the caller was told it exists.
    os.sync()
    return {
        "id": rid,
        "endpoint": f"{rid}.{DOMAIN}:443",
        "ca_pem": open(f"{CERTS}/ca.crt").read(),
        "superexec_secret_b64": base64.b64encode(secret).decode(),
        "node_keys": node_keys,
    }


def write_port_map():
    """Rebuild nginx's hostname -> port map from the DB and reload. shortcut: concurrent creates race benignly."""
    lines = "".join(f'"{rid}.{DOMAIN}" {port};\n' for rid, port in q("SELECT id, port FROM requests"))
    with open(f"{PORT_MAP}.tmp", "w") as f:
        f.write(lines or "# no requests\n")
    os.replace(f"{PORT_MAP}.tmp", PORT_MAP)
    sh("systemctl", "reload", "nginx")


class SuperLinkDown(Exception):
    pass


def cli_json(rid, *args):
    """A flwr CLI call that prints JSON. The CLI exits 0 even when the SuperLink rejects the call."""
    try:
        out = json.loads(flwr(rid, *args))
    except (subprocess.CalledProcessError, ValueError) as e:
        raise SuperLinkDown(f"{rid}: {getattr(e, 'stderr', None) or e}"[:300]) from e
    if out.get("success") is not True:
        raise SuperLinkDown(f"{rid}: {out}"[:300])
    return out


offline_since = {}  # (request id, node id) -> when the watchdog first saw the node offline
events = {}  # request id -> what the watchdog did, shown by status()


def status(rid):
    nodes = cli_json(rid, "supernode", "list", "r", "--format", "json")["nodes"]
    runs = cli_json(rid, "ls", "r", "--format", "json")["runs"]
    return {"id": rid, "nodes": nodes, "runs": runs, "events": events.get(rid, [])}


def watch(rid):
    """Stop a run whose node stays offline. The SuperLink marks the node offline, but the run
    keeps waiting on it for up to an hour (measured: `flwr stop` ends it in seconds)."""
    active = [r for r in cli_json(rid, "ls", "r", "--format", "json")["runs"] if not r["status"].startswith("finished")]
    if not active:
        for k in [k for k in offline_since if k[0] == rid]:
            del offline_since[k]
        return
    now = time.time()
    for n in cli_json(rid, "supernode", "list", "r", "--format", "json")["nodes"]:
        if n["status"] == "offline":
            offline_since.setdefault((rid, n["node-id"]), now)
        else:
            offline_since.pop((rid, n["node-id"]), None)
    dead = [k[1] for k, t0 in offline_since.items() if k[0] == rid and now - t0 >= OFFLINE_GRACE]
    if dead:
        for r in active:
            flwr(rid, "stop", r["run-id"], "r")
            events.setdefault(rid, []).append(f"stopped run {r['run-id']}: node {dead[0]} offline over {OFFLINE_GRACE}s")


def watchdog():
    # shortcut: one flwr CLI call (~1.5s) per request per sweep; poll the SuperLink API directly past ~50 requests
    while True:
        time.sleep(WATCHDOG_INTERVAL)
        for (rid,) in q("SELECT id FROM requests"):
            try:
                watch(rid)
            except Exception as e:  # noqa: BLE001  one dead SuperLink must not end the sweep
                print(f"watchdog {rid}: {e}", file=sys.stderr, flush=True)


RUN_CONFIG = re.compile(r"[\w.= \"'-]{0,500}")  # k=v pairs only; passed as one argv element, never through a shell


def submit(rid, app, run_config=""):
    path = f"{APPS}/{app}"
    if not re.fullmatch(r"[\w.-]+", app) or not os.path.isdir(path):
        raise ValueError(f"no such app under {APPS}: {app!r}")
    if not isinstance(run_config, str) or not RUN_CONFIG.fullmatch(run_config):
        raise ValueError("run_config must be space-separated key=value pairs (letters, digits, . _ - = and quotes)")
    extra = ["--run-config", run_config] if run_config else []
    return json.loads(flwr(rid, "run", path, "r", "--format", "json", *extra))


def destroy(rid):
    subprocess.run(["docker", "rm", "-f", f"sl-{rid}"], capture_output=True)
    shutil.rmtree(f"{ROOT}/requests/{rid}", ignore_errors=True)
    q("DELETE FROM requests WHERE id = ?", rid)
    events.pop(rid, None)
    write_port_map()


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle(self, method):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            m = re.fullmatch(r"/requests(?:/([0-9a-f]{8})(/run)?)?", self.path)
            if not m:
                return self._reply(404, {"error": "not found"})
            rid, run = m.groups()
            if rid is None:
                if method != "POST":
                    return self._reply(404, {"error": "not found"})
                n = body.get("num_nodes")
                if not isinstance(n, int) or not 1 <= n <= 20:
                    return self._reply(400, {"error": "num_nodes must be an int in 1..20"})
                return self._reply(200, create(n))
            if not q("SELECT 1 FROM requests WHERE id = ?", rid):
                return self._reply(404, {"error": "unknown request"})
            if method == "GET" and not run:
                return self._reply(200, status(rid))
            if method == "POST" and run:
                return self._reply(200, submit(rid, body.get("app", ""), body.get("run_config", "")))
            if method == "DELETE" and not run:
                destroy(rid)
                return self._reply(200, {"deleted": rid})
            return self._reply(404, {"error": "not found"})
        except SuperLinkDown as e:
            self._reply(502, {"error": f"superlink unreachable: {e}"})
        except ValueError as e:
            self._reply(400, {"error": str(e)})
        except subprocess.CalledProcessError as e:
            self._reply(500, {"error": f"{e.cmd[0]} failed", "stderr": e.stderr})
        except Exception as e:  # noqa: BLE001
            self._reply(500, {"error": repr(e)})

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def do_DELETE(self):
        self._handle("DELETE")


if __name__ == "__main__":
    os.makedirs(f"{ROOT}/requests", exist_ok=True)
    q("CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, port INTEGER UNIQUE NOT NULL, created REAL NOT NULL)")
    # shortcut: requests from a crashed orchestrator are not reaped; add a TTL reaper
    reconcile()
    threading.Thread(target=watchdog, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
