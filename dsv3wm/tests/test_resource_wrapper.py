import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('resource_wrapper', Path(__file__).parents[1] / 'scripts/run_measured.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResourceWrapperTests(unittest.TestCase):
    def test_cancellation_reaps_only_own_process(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], shell=False)
        try:
            module.stop_owned_processes(process, {})
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
