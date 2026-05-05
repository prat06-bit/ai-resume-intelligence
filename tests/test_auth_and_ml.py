import os
import tempfile
import unittest

from backend.core import database

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-only")

from backend.api.auth_service import (
    LoginRequest,
    RefreshRequest,
    SignupRequest,
    decode_token,
    login,
    logout_session,
    refresh_access_token,
    signup,
)
from backend.core.database import get_cached_embedding
from backend.ml.semantic_matcher import SemanticMatcher


class AuthAndMLSmokeTests(unittest.TestCase):
    def setUp(self):
        db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db.close()
        self.db_path = db.name
        database.DB_PATH = self.db_path

    def test_signup_login_logout_revokes_token(self):
        signup_result = signup(SignupRequest(email="test@example.com", password="Testpass123"))
        self.assertEqual(signup_result["status"], "success")

        token = signup_result["data"]["access_token"]
        refresh_token = signup_result["data"]["refresh_token"]
        payload = decode_token(token)
        self.assertEqual(payload["user_id"], signup_result["data"]["user_id"])

        login_result = login(LoginRequest(email="test@example.com", password="Testpass123"))
        self.assertEqual(login_result["status"], "success")

        invalid_result = login(LoginRequest(email="test@example.com", password="wrong"))
        self.assertEqual(invalid_result, {"status": "error", "message": "Invalid credentials"})

        self.assertIn("refresh_token", login_result["data"])

        refreshed = refresh_access_token(refresh_token)
        self.assertEqual(refreshed["status"], "success")
        self.assertNotEqual(refreshed["data"]["refresh_token"], refresh_token)
        self.assertEqual(refresh_access_token(refresh_token)["status"], "error")

        self.assertEqual(
            logout_session(
                refreshed["data"]["access_token"],
                refresh_token=refreshed["data"]["refresh_token"],
            )["status"],
            "success",
        )
        with self.assertRaises(Exception):
            decode_token(refreshed["data"]["access_token"])

    def test_matcher_fallback_keeps_app_usable(self):
        matcher = SemanticMatcher()
        matcher.embed(["cache me"])
        cache_key, _ = matcher._cache_key("cache me")
        self.assertIsNotNone(get_cached_embedding(cache_key))

        scores = matcher.match_skills(["python", "docker"], ["python", "kubernetes"])
        self.assertGreaterEqual(scores["python"], 0.9)
        self.assertIn("kubernetes", scores)


if __name__ == "__main__":
    unittest.main()
