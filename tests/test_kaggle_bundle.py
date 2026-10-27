import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.build_kaggle_bundle import (
    ROOT, MODULES, NOTEBOOK, build_bundle, bundled_literal, update_notebook,
)


class KaggleBundleTests(unittest.TestCase):
    def test_published_notebook_embeds_current_modules(self):
        notebook = json.loads((ROOT/NOTEBOOK).read_text())
        encoded = bundled_literal(notebook)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
            self.assertEqual(sorted(archive.namelist()), sorted(MODULES))
            for name in MODULES:
                self.assertEqual(archive.read(name), (ROOT/name).read_bytes(), name)
        self.assertTrue(update_notebook(check=True))

    def test_update_is_deterministic_and_preserves_other_notebook_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in MODULES:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# fixture '+name+'\n')
            encoded = build_bundle(root)
            self.assertEqual(encoded, build_bundle(root))
            notebook = {'metadata': {'keep': True}, 'cells': [
                {'cell_type': 'markdown', 'source': ['unchanged documentation']},
                {'cell_type': 'code', 'source': [
                    'before = 1\n', 'BUNDLE_BASE64 = "'+encoded+'"\n', 'after = 2\n'],
                 'outputs': [{'output_type': 'stream', 'text': 'unchanged output'}]},
            ]}
            path = root/NOTEBOOK
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(notebook, indent=1)+'\n')
            original = path.read_text()
            (root/MODULES[0]).write_text('# changed fixture\n')
            self.assertFalse(update_notebook(root, check=True))
            self.assertEqual(path.read_text(), original)
            self.assertFalse(update_notebook(root))
            self.assertEqual(path.read_text(), original.replace(encoded, build_bundle(root)))
            self.assertTrue(update_notebook(root))


if __name__ == '__main__':
    unittest.main()
