import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

source_root = (Path(__file__).resolve().parents[1] / "src").resolve()
sys.path.insert(0, str(source_root))

from rula import local_services


class LocalTabbyApiTests(unittest.TestCase):
    def setUp(self):
        self.previous_process = local_services._tabby_process
        local_services._tabby_process = None

    def tearDown(self):
        local_services._tabby_process = self.previous_process

    @patch.object(local_services, "_is_reachable", return_value=True)
    @patch.object(local_services.subprocess, "Popen")
    def test_reuses_an_already_running_local_service(self, popen, _reachable):
        local_services.ensure_tabbyapi_started(
            "http://127.0.0.1:5000/v1", "test-key")

        popen.assert_not_called()

    @patch.object(local_services, "_is_reachable")
    @patch.object(local_services.subprocess, "Popen")
    def test_starts_tabbyapi_and_waits_until_its_api_is_ready(
            self, popen, reachable):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp) / "TabbyAPI"
            python = home / "venv" / "Scripts" / "python.exe"
            start_script = home / "start.py"
            python.parent.mkdir(parents=True)
            python.touch()
            start_script.touch()
            process = MagicMock()
            process.poll.return_value = None
            popen.return_value = process
            reachable.side_effect = [False, False, True]
            with patch.dict(os.environ, {
                    "TABBYAPI_HOME": str(home),
                    "APPDATA": temp,
            }), patch.object(local_services.time, "sleep"):
                local_services.ensure_tabbyapi_started(
                    "http://127.0.0.1:5000/v1", "test-key")

            args, kwargs = popen.call_args
            self.assertEqual(args[0], [str(python), str(start_script)])
            self.assertEqual(kwargs["cwd"], home)
            self.assertEqual(local_services._tabby_process, process)

    @patch.object(local_services, "_is_reachable", return_value=False)
    @patch.object(local_services.subprocess, "Popen")
    def test_does_not_start_a_remote_tabbyapi(self, popen, _reachable):
        local_services.ensure_tabbyapi_started(
            "https://inference.example/v1", "test-key")

        popen.assert_not_called()

    def test_shutdown_only_stops_a_process_owned_by_this_session(self):
        process = MagicMock()
        process.poll.return_value = None
        local_services._tabby_process = process

        local_services.stop_owned_tabbyapi()

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=15)
        self.assertIsNone(local_services._tabby_process)


if __name__ == "__main__":
    unittest.main()
