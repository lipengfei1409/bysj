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
    start = source.index("def _fallback_lines")
    end = source.index("class QuietBuildExt")
    namespace = {"re": re}
    exec(compile(source[start:end], "setup.py", "exec"), namespace)  # noqa: S102
    return namespace["_error_lines"], namespace["_fallback_lines"]


class BuildLogTests(unittest.TestCase):
    def test_only_error_lines_are_extracted(self) -> None:
        extract, _ = _load_setup_helpers()
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
        extract, _ = _load_setup_helpers()
        line = "/src/a.cpp:12:5: error: boom"
        self.assertEqual(extract("\n".join([line, line, line])), [line])

    def test_a_build_without_an_error_line_is_reported_as_such(self) -> None:
        extract, _ = _load_setup_helpers()
        self.assertEqual(extract("ninja: build stopped: subcommand failed."), [])

    def test_the_errors_are_read_from_the_log_not_the_exception(self) -> None:
        # The compiler writes to the redirected descriptors, so the exception text
        # is only "Error compiling objects for extension" -- scanning it finds
        # nothing, which is exactly what happened on the first narrowed build.
        source = (ROOT / "setup.py").read_text()
        self.assertIn("text = BUILD_LOG.read_text", source)
        self.assertIn("errors = _error_lines(text)", source)

    def test_a_log_without_an_error_falls_back_to_its_tail(self) -> None:
        _, fallback = _load_setup_helpers()
        sample = "\n".join([
            "/src/a.cpp:1:1: warning: unused variable",
            "/src/a.cpp:2:1: note: in instantiation of",
            "In file included from /src/a.cpp:1:",
            "ld.lld: subcommand failed",
            "ninja: build stopped: subcommand failed.",
        ])
        lines = fallback(sample)
        self.assertEqual(lines[-2:], ["ld.lld: subcommand failed",
                                      "ninja: build stopped: subcommand failed."])
        self.assertFalse(any("warning:" in line or "note:" in line for line in lines))

    def test_the_arch_list_is_split_on_every_separator(self) -> None:
        # PYTORCH_ROCM_ARCH accepts "gfx906;gfx928" and people also paste commas or
        # spaces; a single-target list must not be mistaken for one weird target.
        source = (ROOT / "setup.py").read_text()
        self.assertIn(r're.split(r"[,;\s]+", _ARCH_ENV)', source)
        splitter = __import__("re").compile(r"[,;\s]+")
        for value in ("gfx906;gfx928", "gfx906,gfx928", "gfx906 gfx928"):
            self.assertEqual([a for a in splitter.split(value) if a],
                             ["gfx906", "gfx928"], value)

    def test_the_arch_defaults_to_the_device_and_widens_only_on_request(self) -> None:
        source = (ROOT / "setup.py").read_text()
        # detection runs only when nothing was exported, so an explicit multi-arch
        # list always wins
        self.assertIn("if not _ARCH_ENV:", source)
        self.assertIn("gcnArchName", source)

    def test_the_log_path_is_inside_build(self) -> None:
        source = (ROOT / "setup.py").read_text()
        self.assertIn('BUILD_LOG = pathlib.Path(__file__).resolve().parent / "build" / "build.log"',
                      source)
        self.assertIn('cmdclass = {"build_ext": QuietBuildExt}', source)


if __name__ == "__main__":
    unittest.main()
