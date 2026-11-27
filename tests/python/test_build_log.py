"""The build keeps its noise in `build/build.log` and prints only the errors.

A hipcc build of this extension emits hundreds of warnings from torch and pybind
headers; printing them buries the two or three lines that matter, and on this
project the cost of missing the first error is another round trip to the DCU host.
"""

import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_setup_helpers():
    source = (ROOT / "setup.py").read_text()
    start = source.index("def _error_lines")
    end = source.index("class QuietBuildExt")
    namespace = {"re": re}
    exec(compile(source[start:end], "setup.py", "exec"), namespace)  # noqa: S102
    return namespace["_error_lines"]


class BuildLogTests(unittest.TestCase):
    def test_only_error_lines_are_extracted(self) -> None:
        extract = _load_setup_helpers()
        sample = "\n".join([
            "/src/a.cpp:12:5: error: no member named 'foo'",
            "/src/a.cpp:13:5: warning: unused variable 'x'",
            "/src/a.cpp:14:5: note: candidate function not viable",
            "ninja: build stopped: subcommand failed.",
            "/usr/bin/ld: undefined reference to `bar'",
            "FAILED: [code=1] /build/obj.o",
        ])
        lines = extract(sample)
        self.assertIn("/src/a.cpp:12:5: error: no member named 'foo'", lines)
        self.assertIn("/usr/bin/ld: undefined reference to `bar'", lines)
        self.assertIn("FAILED: [code=1] /build/obj.o", lines)
        self.assertFalse(any("warning" in line for line in lines))
        self.assertFalse(any("note:" in line for line in lines))

    def test_duplicates_collapse(self) -> None:
        # The same error repeats once per offload arch; one copy is enough.
        extract = _load_setup_helpers()
        line = "/src/a.cpp:12:5: error: boom"
        self.assertEqual(extract("\n".join([line, line, line])), [line])

    def test_a_build_without_an_error_line_is_reported_as_such(self) -> None:
        extract = _load_setup_helpers()
        self.assertEqual(extract("ninja: build stopped: subcommand failed."), [])

    def test_the_log_path_is_inside_build(self) -> None:
        source = (ROOT / "setup.py").read_text()
        self.assertIn('BUILD_LOG = pathlib.Path(__file__).resolve().parent / "build" / "build.log"',
                      source)
        self.assertIn('cmdclass = {"build_ext": QuietBuildExt}', source)


if __name__ == "__main__":
    unittest.main()
