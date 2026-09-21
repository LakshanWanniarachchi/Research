"""Authentication and access control."""

import pytest


class TestAccessControl:
    """Pages that show patient data must not be reachable without signing in."""

    @pytest.mark.parametrize("path", [
        "/", "/dashboard", "/api/history", "/api/patients",
        "/api/record", "/api/status", "/patient/1",
    ])
    def test_protected_pages_redirect_to_login(self, client, path):
        resp = client.get(path)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_login_page_is_public(self, client):
        assert client.get("/login").status_code == 200

    def test_health_is_public(self, client):
        """The simulator checks health before sending, so it stays open."""
        assert client.get("/api/health").status_code == 200

    def test_device_endpoint_does_not_need_a_session(self, client):
        """A device has no browser and cannot hold a session cookie."""
        resp = client.post("/api/predict",
                           json={"signal": [0.0] * 1000, "mode": "default"})
        assert resp.status_code == 200


class TestSignIn:
    def test_correct_credentials_sign_in(self, client):
        resp = client.post("/login", data={"username": "admin",
                                           "password": "test-password-123"})
        assert resp.status_code == 302
        assert "/dashboard" in resp.headers["Location"]

    def test_wrong_password_is_refused(self, client):
        resp = client.post("/login", data={"username": "admin",
                                           "password": "wrong"})
        assert resp.status_code == 401

    def test_unknown_user_is_refused(self, client):
        resp = client.post("/login", data={"username": "nobody",
                                           "password": "whatever"})
        assert resp.status_code == 401

    def test_the_two_failures_are_indistinguishable(self, client):
        """A different message for each would let an attacker enumerate users."""
        a = client.post("/login", data={"username": "admin", "password": "wrong"})
        b = client.post("/login", data={"username": "nobody", "password": "wrong"})
        assert a.status_code == b.status_code
        assert a.data == b.data

    def test_password_is_not_stored_in_the_clear(self, client, db):
        # `client` is required, not `db` alone: the admin account is seeded
        # when app.py is imported, which is what the client fixture does.
        import models
        row = models.get_user_by_name(db, "admin")
        assert row is not None
        assert "test-password-123" not in row["password_hash"]
        assert len(row["password_hash"]) > 40

    def test_signing_in_records_the_time(self, client, db):
        import models
        assert models.get_user_by_name(db, "admin")["last_login"] is None
        client.post("/login", data={"username": "admin",
                                    "password": "test-password-123"})
        assert models.get_user_by_name(db, "admin")["last_login"] is not None


class TestSession:
    def test_signed_in_user_reaches_the_dashboard(self, auth_client):
        resp = auth_client.get("/dashboard")
        assert resp.status_code == 200
        assert b"ECG Screening System" in resp.data

    def test_sign_out_ends_the_session(self, auth_client):
        auth_client.post("/logout")
        resp = auth_client.get("/dashboard")
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_login_returns_to_the_requested_page(self, client):
        resp = client.get("/api/history")
        assert "next=%2Fapi%2Fhistory" in resp.headers["Location"] or \
               "next=/api/history" in resp.headers["Location"]

    def test_redirect_target_must_be_relative(self, client):
        """A crafted link must not bounce a signed-in user to another site."""
        resp = client.post("/login", data={"username": "admin",
                                           "password": "test-password-123",
                                           "next": "http://evil.example.com/"})
        assert resp.status_code == 302
        assert "evil.example.com" not in resp.headers["Location"]
