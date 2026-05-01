"""Test that QuestionCreatorAgent has a per-call API timeout configured.

Calibration regens (D7-PHY-195, D7-PHY-209) lost 4-9 questions to
indefinite Anthropic API hangs — python processes stayed alive with
near-zero CPU, waiting on an HTTP response that never came. The fix
wraps client.messages.create in asyncio.wait_for() with a per-call
timeout. This test locks in the timeout config so it can't drift.
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import conftest  # noqa: F401  — sets sys.path

from pipeline.agents import QuestionCreatorAgent


class TestApiTimeoutConstant(unittest.TestCase):
    def test_api_timeout_attribute_exists(self):
        # The constant must exist as a class attribute so future devs
        # don't accidentally remove the timeout wrap by removing the
        # config alone.
        self.assertTrue(
            hasattr(QuestionCreatorAgent, "_API_TIMEOUT_SEC"),
            "QuestionCreatorAgent must expose _API_TIMEOUT_SEC",
        )
        timeout = QuestionCreatorAgent._API_TIMEOUT_SEC
        self.assertIsInstance(timeout, (int, float))
        # Sanity bounds: not zero, not absurdly long. Empirically 120s is
        # the calibrated value (Opus 4.7 finishes in 30-60s normally).
        self.assertGreaterEqual(timeout, 30,
                                "timeout must be at least 30s for Opus")
        self.assertLessEqual(timeout, 300,
                             "timeout must be at most 300s — defeats the purpose")


class TestTimeoutTriggersRetry(unittest.IsolatedAsyncioTestCase):
    async def test_hanging_call_triggers_timeout_then_retry(self):
        # Simulate an API call that hangs forever. The asyncio.wait_for
        # wrap should fire its timeout, the retry loop should catch it
        # via the existing except Exception branch, retry once more,
        # then give up cleanly with an error rather than hanging.
        agent = QuestionCreatorAgent()

        # Override timeout to something fast for the test
        original_timeout = QuestionCreatorAgent._API_TIMEOUT_SEC
        QuestionCreatorAgent._API_TIMEOUT_SEC = 0.1
        try:
            # Mock client whose messages.create hangs indefinitely
            async def hang(*args, **kwargs):
                await asyncio.sleep(60)  # longer than our 0.1s timeout
                return MagicMock()
            client = MagicMock()
            client.messages.create = hang

            # Patch sleep so retries don't actually wait 10s/20s
            with patch("pipeline.agents.asyncio.sleep", new=AsyncMock()):
                result, meta = await agent.async_execute(
                    client, "system", "user", max_retries=1,
                )

            self.assertIsNone(result, "hanging call must return None result")
            self.assertIn("error", meta)
            # Timeout shows up as an api: error in the existing retry path
            self.assertIn("api:", meta["error"])
        finally:
            QuestionCreatorAgent._API_TIMEOUT_SEC = original_timeout


if __name__ == "__main__":
    unittest.main()
