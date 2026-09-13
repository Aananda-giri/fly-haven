import json
from functools import partial
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from flywirehead.server import Experiment, Handler, ThreadingHTTPServer
from flywirehead.engine import FRAME_HEIGHT, FRAME_WIDTH


@pytest.fixture
def app(tmp_path):
    # No synthetic engine is ever installed in the application: this fixture
    # exercises HTTP validation and queue behavior without starting a model.
    exp = Experiment(tmp_path)
    exp.state.update(phase="ready", message="Unit-test queue")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, experiment=exp))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield exp, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def request(url, path, body=None, **headers):
    req = Request(url + path, data=body, headers=headers)
    try:
        with urlopen(req, timeout=3) as response:
            return response.status, json.load(response)
    except HTTPError as response:
        return response.code, json.load(response)


def test_frames_require_local_session_and_exact_rgba_size(app):
    exp, url = app
    pixels = bytes(FRAME_WIDTH * FRAME_HEIGHT * 4)
    code, _ = request(url, "/api/frame", pixels, **{"Content-Type": "application/octet-stream", "X-Fly-Client": "test"})
    assert code == 403 and exp.pending is None
    headers = {"X-Fly-Token": exp.token, "X-Fly-Client": "test", "Content-Type": "application/octet-stream"}
    code, _ = request(url, "/api/frame", b"bad", **headers)
    assert code == 400 and exp.pending is None
    code, _ = request(url, "/api/frame", pixels, **headers)
    assert code == 202 and exp.pending[0].shape == (160, 90, 3)
    # A second window cannot overwrite the current sensory stream.
    headers["X-Fly-Client"] = "second"
    assert request(url, "/api/frame", pixels, **headers)[0] == 409


def test_pause_discards_queued_frames_and_requires_resume_before_stimulation(app):
    exp, url = app
    exp.submit(bytes(FRAME_WIDTH * FRAME_HEIGHT * 4), "test")
    exp.control("stimulate")
    headers = {"X-Fly-Token": exp.token, "Content-Type": "application/json"}
    code, data = request(url, "/api/control", b'{"action":"pause"}', **headers)
    assert code == 200 and data["paused"]
    assert exp.pending is None and not exp.injection_requested
    assert request(url, "/api/control", b'{"action":"stimulate"}', **headers)[0] == 409
    assert request(url, "/api/control", b'{"action":"resume"}', **headers)[0] == 200
    assert not exp.state["paused"]


def test_cross_origin_and_invalid_controls_cannot_mutate_the_brain(app):
    exp, url = app
    headers = {"X-Fly-Token": exp.token, "Content-Type": "application/json"}
    assert request(url, "/api/control", b'{"action":"pause"}', Origin="https://example.com", **headers)[0] == 403
    assert request(url, "/api/control", b'{"action":"invent_dopamine"}', **headers)[0] == 400
    assert request(url, "/api/control", b'{"action":"pause","extra":1}', **headers)[0] == 400
    assert not exp.state["paused"]
    code, data = request(url, "/api/status")
    assert code == 200 and data["telemetry"] is None

