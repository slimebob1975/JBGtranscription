"""Static GUI checks and optional executable browser-lifecycle simulation."""
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProcessingOverlayTests(unittest.TestCase):
    def test_overlay_markup_and_layout(self):
        html = (ROOT / 'static/index.html').read_text(encoding='utf-8')
        css = (ROOT / 'static/styles/styles.css').read_text(encoding='utf-8')
        self.assertIn('id="spinner-container" hidden', html)
        self.assertIn('id="processingDetail" role="status"', html)
        self.assertIn('id="processingElapsed">00:00', html)
        self.assertRegex(css, r'#spinner-container\s*\{\s*position: absolute;\s*inset: 0;')
        self.assertIn('#spinner-container[hidden]', css)
        self.assertIn('prefers-reduced-motion: reduce', css)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is optional for GUI lifecycle tests')
    def test_javascript_lifecycle(self):
        subprocess.run(['node', '--test', str(ROOT / 'tests/test_processing_overlay.js')],
                       cwd=ROOT, check=True, timeout=30)
