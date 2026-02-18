# make_repo_dumps.py
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Tuple

# =========================
# CONFIG
# =========================
REPO_ROOT = Path(__file__).resolve().parent

# Outputs
TREE_FILE = REPO_ROOT / "repo_tree.txt"
PY_DUMP_FILE = REPO_ROOT / "all_python_files_dump.txt"
WEB_DUMP_FILE = REPO_ROOT / "all_web_files_dump.txt"
GRAPH_PNG = REPO_ROOT / "rifd_graph.png"

# Exclusions (dir names are matched case-insensitively)
EXCLUDE_DIRS = {
    ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".git", ".idea", ".vscode", "node_modules", "dist", "build",
    "lib", "scripts", "site-packages", ".ruff_cache", ".cache"
}

# File suffixes for web bundle
WEB_SUFFIXES = (".html", ".css", ".js")

# Optional: Skip obvious binary/noisy file suffixes in the tree
TREE_SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".mp3", ".mp4", ".wav", ".mov", ".avi",
    ".onnx", ".pt", ".ckpt"
}


# =========================
# HELPERS
# =========================
def _prune_dirs_inplace(dirs):
    """Prune excluded directories in-place (case-insensitive)."""
    dirs[:] = [d for d in dirs if d.lower() not in EXCLUDE_DIRS]


def _human_size(bytes_):
    try:
        b = float(bytes_)
    except Exception:
        return "?"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024.0:
            return f"{b:.0f}{unit}"
        b /= 1024.0
    return f"{b:.0f}PB"


def write_repo_tree(root: Path, out_file: Path):
    """
    Write a deep, pruned directory tree to out_file.
    Shows sizes; skips excluded dirs and noisy binary assets.
    """
    lines = []
    rel_root = root

    for curr_root, dirs, files in os.walk(root):
        _prune_dirs_inplace(dirs)

        curr_path = Path(curr_root)
        depth = len(curr_path.relative_to(rel_root).parts)
        indent = "    " * depth
        # Add the current directory header
        if curr_path == rel_root:
            lines.append(f"{curr_path.name}/")
        else:
            lines.append(f"{indent}{curr_path.name}/")

        # Sort files for stable output
        files_sorted = sorted(files, key=lambda n: n.lower())

        for fname in files_sorted:
            fpath = curr_path / fname
            # skip noisy/binary files in tree
            if fpath.suffix.lower() in TREE_SKIP_SUFFIXES:
                continue
            try:
                size = _human_size(fpath.stat().st_size)
            except Exception:
                size = "?"
            lines.append(f"{indent}    {fname}  ({size})")

    header = [
        f"# Repository Tree for: {root}",
        f"# Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"# Excluding dirs: {', '.join(sorted(EXCLUDE_DIRS))}",
        ""
    ]
    out_file.write_text("\n".join(header + lines), encoding="utf-8")


def dump_files(root: Path, suffixes, out_file: Path, label: str):
    """
    Dump all files ending with given suffixes into a single text file.
    Each file is separated by a clear header.
    """
    count = 0
    with out_file.open("w", encoding="utf-8") as out:
        out.write(f"# {label}\n")
        out.write(f"# Root: {root}\n")
        out.write(f"# Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")

        for curr_root, dirs, files in os.walk(root):
            _prune_dirs_inplace(dirs)
            for fname in files:
                if not fname.lower().endswith(suffixes):
                    continue
                fpath = Path(curr_root) / fname
                try:
                    code = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    code = f"Error reading file: {e}"
                out.write(f"=== FILE: {fpath} ===\n")
                out.write(code)
                out.write("\n\n" + "=" * 80 + "\n\n")
                count += 1
    return count


def _save_image_like(obj, png_path: Path) -> bool:
    """
    Try to persist various 'image-like' objects to disk as PNG.
    Returns True if the file was written.
    Supports: bytes/bytearray/memoryview, PIL.Image-like (has .save),
    and 'str path' returned from some APIs.
    """
    try:
        # Bytes-like → write directly
        if isinstance(obj, (bytes, bytearray, memoryview)):
            png_path.write_bytes(bytes(obj))
            return True

        # PIL.Image or similar → .save(path)
        if hasattr(obj, "save") and callable(getattr(obj, "save")):
            obj.save(str(png_path))
            return True

        # Some APIs return a filepath string
        if isinstance(obj, str):
            candidate = Path(obj)
            if candidate.exists():
                png_path.write_bytes(candidate.read_bytes())
                return True
    except Exception:
        pass

    return False


def generate_graph(png_path: Path) -> bool:
    """
    Try multiple export shapes:
      1) compiled graph object exposes .get_graph().{draw_png, draw_mermaid_png, draw_mermaid, draw_dot, to_*}
      2) graph object exposes {visualize, draw_mermaid, draw_dot, to_*}
    Save PNG if possible; otherwise write DOT (.dot) or Mermaid (.mmd).
    Returns True iff a PNG was written.
    """
    try:
        from backend.pipeline_bootstrap import graph
    except Exception as e:
        print(f"[graph] Could not import graph: {e}")
        return False

    wrote_png = False
    base = png_path.with_suffix("")  # <repo>/rifd_graph
    dot_path = base.with_suffix(".dot")
    mmd_path = base.with_suffix(".mmd")

    def _try_call(obj, name):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                return None
        return None

    # 0) Prefer compiled graph internals if available
    g_obj = _try_call(graph, "get_graph")  # many LangGraph versions expose this on the compiled graph
    exporters = []

    if g_obj is not None:
        exporters.extend([
            ("png", _try_call(g_obj, "draw_png")),
            ("png", _try_call(g_obj, "draw_mermaid_png")),
            ("mermaid", _try_call(g_obj, "draw_mermaid")),
            ("mermaid", _try_call(g_obj, "to_mermaid")),
            ("dot", _try_call(g_obj, "draw_dot")),
            ("dot", _try_call(g_obj, "to_dot")),
        ])

    # 1) Also try methods on the graph itself
    exporters.extend([
        ("dot", _try_call(graph, "visualize")),        # some builds return DOT here
        ("mermaid", _try_call(graph, "draw_mermaid")),
        ("mermaid", _try_call(graph, "to_mermaid")),
        ("dot", _try_call(graph, "draw_dot")),
        ("dot", _try_call(graph, "to_dot")),
    ])

    # 2) First: any PNG-like return (bytes/img/path) → write it
    for kind, payload in exporters:
        if kind != "png" or payload is None:
            continue
        if _save_image_like(payload, png_path):
            print(f"✅ Graph PNG exported to {png_path.name}")
            wrote_png = True
            break

    # 3) If no PNG, try DOT then Mermaid
    if not wrote_png:
        # DOT text
        dot_text = None
        for kind, payload in exporters:
            if kind == "dot" and isinstance(payload, str) and payload.strip():
                dot_text = payload
                break

        if dot_text:
            try:
                dot_path.write_text(dot_text, encoding="utf-8")
                print(f"✅ Wrote DOT graph to {dot_path.name}")
                try:
                    import graphviz
                    g = graphviz.Source(dot_text)
                    g.format = "png"
                    g.render(base, cleanup=True)
                    print(f"✅ Graph PNG exported to {png_path.name}")
                    wrote_png = True
                except Exception as e:
                    print(f"⚠️ graphviz not available or render failed: {e}")
                    print("⚠️ PNG skipped. DOT file is available.")
            except Exception as e:
                print(f"⚠️ Failed writing DOT: {e}")

    if not wrote_png:
        # Mermaid text as final fallback
        mmd_text = None
        for kind, payload in exporters:
            if kind == "mermaid" and isinstance(payload, str) and payload.strip():
                mmd_text = payload
                break
        if mmd_text:
            try:
                mmd_path.write_text(mmd_text, encoding="utf-8")
                print(f"✅ Wrote Mermaid graph to {mmd_path.name}")
                print("⚠️ To render PNG locally, use mermaid-cli (mmdc) or GitHub preview.")
            except Exception as e:
                print(f"⚠️ Failed writing Mermaid: {e}")

    return wrote_png


# =========================
# MAIN
# =========================
def main():
    print(f"[i] Repo root: {REPO_ROOT}")

    # 1) Deep, pruned tree
    write_repo_tree(REPO_ROOT, TREE_FILE)
    print(f"✅ Wrote tree to: {TREE_FILE}")

    # 2) Dump all Python files
    py_count = dump_files(REPO_ROOT, (".py",), PY_DUMP_FILE, "All Python Files Dump")
    print(f"✅ Dumped {py_count} Python files into: {PY_DUMP_FILE}")

    # 3) Dump all Web files (html/css/js)
    web_count = dump_files(REPO_ROOT, WEB_SUFFIXES, WEB_DUMP_FILE, "All Web Files Dump")
    print(f"✅ Collected {web_count} web files into: {WEB_DUMP_FILE}")

    # 4) Plan graph PNG
    wrote_graph = generate_graph(GRAPH_PNG)
    if wrote_graph:
        print(f"✅ Graph PNG: {GRAPH_PNG}")
    else:
        print("⚠️ Skipped graph export (import or render issue).")

    print("\nAll done.")


if __name__ == "__main__":
    # Ensure consistent CWD behavior
    os.chdir(REPO_ROOT)
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
