"""The vendored upstream package must be byte-identical to the pinned upstream commit, and the installed package must not contain code
that reads environment variables, opens network connections or starts processes."""

import hashlib
import re
from pathlib import Path

import vendor_sync

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "s2tok_comfy" / "vendor"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _listed() -> dict:
    return dict(reversed(line.split("  ", 1)) for line in (VENDOR / "MANIFEST.sha256").read_text(encoding="ascii").splitlines())


def test_manifest_matches_every_vendored_file():
    actual = {p.relative_to(VENDOR).as_posix(): _sha(p) for p in (VENDOR / "s2tok").rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    actual.update({name: _sha(VENDOR / name) for name in vendor_sync.LICENSE_FILES})
    assert actual == _listed()


def test_vendor_notes_pin_the_commit_and_the_license():
    notes = (VENDOR / "UPSTREAM.md").read_text(encoding="utf-8")
    assert vendor_sync.UPSTREAM_COMMIT in notes and "CC BY-NC 4.0" in notes
    assert "Attribution-NonCommercial 4.0" in (VENDOR / "LICENSE").read_text(encoding="utf-8")


def test_identical_to_the_upstream_checkout(upstream_dir):
    """CI checks out the pinned commit (LF line endings on every platform); the patched file must equal upstream + the documented edit."""
    names = vendor_sync.upstream_files(upstream_dir)
    assert set(names) == set(_listed())
    for name in names:
        assert (VENDOR / name).read_bytes() == vendor_sync.expected_bytes(upstream_dir, name), name


def test_the_patch_changes_only_the_bf16_kernel_choice_in_step(upstream_dir):
    import ast

    def methods(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "S2Tok")
        rest = ast.dump(ast.Module(body=[n for n in tree.body if n is not cls], type_ignores=[]))
        return rest, {f.name: ast.dump(f) for f in cls.body if isinstance(f, ast.FunctionDef)}

    for rel in vendor_sync.PATCHED:
        up_rest, up = methods(upstream_dir / rel)
        our_rest, ours = methods(VENDOR / rel)
        assert up_rest == our_rest and up.keys() == ours.keys()
        assert [name for name in up if up[name] != ours[name]] == ["step"]
        assert "EFFICIENT_ATTENTION" in ours["step"] and "EFFICIENT_ATTENTION" not in up["step"]


PACKAGE_FILES = [p for p in [ROOT / "__init__.py", *(ROOT / "s2tok_comfy").rglob("*.py"), *(ROOT / "tools").rglob("*.py")]
                 if "__pycache__" not in p.parts]
FORBIDDEN = re.compile(r"os\.environ|getenv|putenv|urllib|requests\.|http\.client|socket\.|subprocess|os\.system|importlib|(?<![.\w])exec\(|(?<![.\w])eval\(|__import__")


def test_package_has_no_environment_network_or_process_code():
    """The categories of the Comfy Registry's automated scan (scripts/ and tests/ are not part of a Registry package)."""
    hits = [f"{p.relative_to(ROOT).as_posix()}:{n}: {line.strip()}" for p in PACKAGE_FILES
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if FORBIDDEN.search(line)]
    assert not hits, "\n".join(hits)
