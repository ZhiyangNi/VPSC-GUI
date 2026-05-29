# -*- coding: utf-8 -*-
"""
VPSC Python GUI for Fortran-compatible pre-processing, execution and post-processing.
"""
from __future__ import annotations

import json
import logging
import math
import time
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib as mpl
from matplotlib import colors as mpl_colors
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, PathPatch
from matplotlib.text import Text as MplText

# -----------------------------------------------------------------------------
# Optional GUI toolkit
# -----------------------------------------------------------------------------
# Tkinter and the matplotlib Tk backend are required only to *launch* the
# graphical application.  They are imported defensively so that the scientific
# core (file parsers, crystallography, projections and the plotting helpers
# that operate on a bare ``matplotlib.figure.Figure``) can be imported and
# unit-tested in a headless / continuous-integration environment that has no
# display, and possibly no ``tkinter`` build at all.  When the toolkit is
# missing the GUI classes can still be *defined* (so ``import app`` succeeds)
# but cannot be *instantiated*; ``main()`` turns that into a clean message.
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, colorchooser, simpledialog
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
    GUI_AVAILABLE = True
    GUI_IMPORT_ERROR = None
except Exception as _gui_exc:  # pragma: no cover - headless environments
    import types as _types
    GUI_AVAILABLE = False
    GUI_IMPORT_ERROR = _gui_exc

    class _GuiUnavailable:
        """Placeholder base used when no GUI toolkit is available.

        Subclasses can be defined at import time but instantiating one raises a
        clear, actionable error instead of crashing the whole module import.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "The graphical interface requires Tkinter, which is not "
                f"available here ({GUI_IMPORT_ERROR!r}). Install a Python build "
                "with Tk support to launch the app. The scientific core can "
                "still be imported and scripted without a display."
            )

    tk = _types.SimpleNamespace(  # type: ignore[assignment]
        Tk=_GuiUnavailable, Toplevel=_GuiUnavailable, Frame=_GuiUnavailable,
        Canvas=_GuiUnavailable, Menu=_GuiUnavailable, StringVar=_GuiUnavailable,
        BooleanVar=_GuiUnavailable, IntVar=_GuiUnavailable,
        DoubleVar=_GuiUnavailable, TclError=RuntimeError,
    )
    ttk = _types.SimpleNamespace(  # type: ignore[assignment]
        Frame=_GuiUnavailable, Label=_GuiUnavailable, Button=_GuiUnavailable,
        Entry=_GuiUnavailable, Combobox=_GuiUnavailable, Notebook=_GuiUnavailable,
        Checkbutton=_GuiUnavailable, Treeview=_GuiUnavailable,
        Style=_GuiUnavailable, Scrollbar=_GuiUnavailable,
        PanedWindow=_GuiUnavailable, Separator=_GuiUnavailable,
    )
    filedialog = messagebox = colorchooser = simpledialog = None  # type: ignore[assignment]
    FigureCanvasTkAgg = NavigationToolbar2Tk = None  # type: ignore[assignment]

try:
    from scipy.ndimage import gaussian_filter  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    gaussian_filter = None
    SCIPY_AVAILABLE = False

__version__ = "1.0.0"
__license__ = "MIT"
APP_VERSION = f"vpsc-app-{__version__}"
LOG = logging.getLogger("vpsc_app")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

PREF_PATH = Path.home() / ".vpsc_app.json"

# =============================================================================
# Constants and small utilities
# =============================================================================

# Fortran D-exponent or scientific E.  No leading sign required because we are
# scanning for sub-tokens inside arbitrary text.
NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[EeDd][-+]?\d+)?")

COLOR_CHOICES: Dict[str, str] = {
    "Black": "#111827", "Gray": "#6b7280", "White": "#ffffff",
    "Blue": "#2563eb", "Sky": "#0284c7", "Teal": "#0f766e",
    "Green": "#16a34a", "Lime": "#65a30d", "Yellow": "#ca8a04",
    "Orange": "#ea580c", "Red": "#dc2626", "Rose": "#e11d48",
    "Purple": "#7c3aed", "Magenta": "#c026d3", "Brown": "#92400e",
    "Navy": "#1e3a8a", "Cyan": "#06b6d4", "Pink": "#ec4899",
    "Gold": "#f59e0b", "Olive": "#4d7c0f", "Slate": "#334155",
}

CMAP_CHOICES: List[str] = [
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
    "Blues", "Greens", "Reds", "Purples", "Oranges", "gray", "gray_r",
    "Greys", "YlOrRd", "YlGnBu", "Spectral", "coolwarm", "RdBu_r",
]
MARKER_CHOICES: List[str] = ["o", "s", "^", "v", "D", "x", "+", ".", "*"]
LINESTYLE_CHOICES: List[str] = ["-", "--", "-.", ":", "None"]


def safe_float(s: Any, default: float = 0.0) -> float:
    """Parse a Fortran-style number; accept D/d exponents."""
    try:
        return float(str(s).replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError):
        return default


def numeric_tokens(line: str) -> List[str]:
    return NUM_RE.findall(line)


def numeric_values(line: str) -> List[float]:
    return [safe_float(x) for x in numeric_tokens(line)]


def process_numeric_values(line: str) -> List[float]:
    """Extract numeric values from VPSC process data lines."""
    head = re.split(r"[|@]", line, maxsplit=1)[0].strip()
    if not re.match(r"^[+-]?(?:\d|\.)", head):
        return []
    return [safe_float(x) for x in numeric_tokens(head)]


def strict_numeric_values(line: str) -> List[float]:
    """Extract standalone numeric fields from FE/VPSC history tables.

    Digits embedded in labels such as L11 and L22 are ignored, so table
    headers are not mistaken for velocity-gradient data.
    """
    head = re.split(r"[#*!|@]", line, maxsplit=1)[0].replace(",", " ")
    vals: List[float] = []
    for token in head.split():
        if re.fullmatch(NUM_RE, token):
            vals.append(safe_float(token))
    return vals


def read_text(path: Path) -> str:
    for enc in ("utf-8", "gbk", "latin1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
        except OSError as e:
            LOG.warning("read_text failed on %s: %s", path, e)
            return ""
    return path.read_bytes().decode("utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def path_rel(base: Path, p: str | Path) -> Path:
    q = Path(str(p).strip())
    return q if q.is_absolute() else base / q


def noncomment_lines(text: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s or s[0] in "*#!":
            continue
        out.append((i, line))
    return out


def resolve_color(value: str, fallback: str = "#2563eb") -> str:
    s = str(value).strip()
    if not s:
        return fallback
    if s in COLOR_CHOICES:
        return COLOR_CHOICES[s]
    cap = s.capitalize()
    if cap in COLOR_CHOICES:
        return COLOR_CHOICES[cap]
    try:
        return mpl_colors.to_hex(mpl_colors.to_rgba(s))
    except (ValueError, TypeError):
        return fallback


def fmt_num(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(v - round(v)) < 1e-12 and abs(v) < 1e9:
        return str(int(round(v)))
    return f"{v:.10g}"


_FONT_FAMILY_CACHE: Optional[List[str]] = None
# Families that should appear first if they are actually installed.
_PREFERRED_FONT_FAMILIES = (
    "Arial", "Helvetica", "Times New Roman", "Times", "Calibri", "Cambria",
    "Georgia", "Verdana", "Tahoma", "Segoe UI", "DejaVu Sans", "DejaVu Serif",
    "Liberation Sans", "Liberation Serif", "CMU Serif", "Computer Modern",
)


def available_font_families(limit: int = 120) -> List[str]:
    """Return installed font families for the editor drop-downs.

    Discovered once from matplotlib's font manager (so the list reflects what
    will actually render), de-duplicated and sorted with common publication
    fonts first.  Falls back to a small built-in list if discovery fails, so the
    GUI is never left without choices.
    """
    global _FONT_FAMILY_CACHE
    if _FONT_FAMILY_CACHE is not None:
        return _FONT_FAMILY_CACHE
    fallback = ["Arial", "Helvetica", "Times New Roman", "DejaVu Sans",
                "DejaVu Serif", "Calibri"]
    try:
        from matplotlib import font_manager as _fm
        names = set()
        for f in _fm.fontManager.ttflist:
            try:
                nm = str(f.name).strip()
                if nm and not nm.startswith("."):
                    names.add(nm)
            except Exception:
                pass
        if not names:
            _FONT_FAMILY_CACHE = fallback
            return _FONT_FAMILY_CACHE
        installed = set(names)
        preferred = [n for n in _PREFERRED_FONT_FAMILIES if n in installed]
        rest = sorted(n for n in names if n not in set(preferred))
        ordered = preferred + rest
        _FONT_FAMILY_CACHE = ordered[:max(limit, len(preferred) + 1)]
    except Exception:
        _FONT_FAMILY_CACHE = fallback
    return _FONT_FAMILY_CACHE


def _file_key(path: Path) -> Tuple[str, float, int]:
    """Cache key: full path, mtime, size."""
    try:
        st = path.stat()
        return (str(path.resolve()), st.st_mtime, st.st_size)
    except OSError:
        return (str(path), 0.0, 0)


# =============================================================================
# VPSC8.IN parsing
# =============================================================================

@dataclass
class PhaseInfo:
    index: int
    texture_file: str = ""
    crystal_file: str = ""
    shape_file: str = ""
    diffraction_file: str = ""


@dataclass
class VPSCInInfo:
    regime: int = 1
    nph: int = 1
    wph: List[float] = field(default_factory=list)
    phases: List[PhaseInfo] = field(default_factory=list)
    errs: List[float] = field(default_factory=list)
    itmax: List[int] = field(default_factory=list)
    interaction: int = 0
    neff: float = 10.0
    iupdate: Tuple[int, int, int] = (1, 1, 1)
    nneigh: int = 0
    iflu: int = 0
    process_files: List[str] = field(default_factory=list)
    process_ivgvar: List[int] = field(default_factory=list)


def _next_nonempty_nonstar(lines: List[str], start: int) -> Tuple[int, str]:
    for j in range(start, len(lines)):
        st = lines[j].strip()
        if st and not st.startswith("*"):
            return j, st
    return len(lines), ""


def parse_vpsc8_in(path: Path) -> VPSCInInfo:
    """Parse VPSC8.IN by recognising its labelled comment blocks.

    The VPSC format is mostly free-format numbers separated by ``*`` comment
    lines.  Labels are more reliable than fixed line numbers because user
    edits change the line count.
    """
    text = read_text(path)
    lines = text.splitlines()
    data = noncomment_lines(text)
    info = VPSCInInfo()

    if data:
        v0 = numeric_values(data[0][1])
        info.regime = int(v0[0]) if v0 else 1
    if len(data) > 1:
        v1 = numeric_values(data[1][1])
        info.nph = int(v1[0]) if v1 else 1
    if len(data) > 2:
        info.wph = numeric_values(data[2][1])[:max(1, info.nph)]

    # Labelled phase files
    tex_files: List[str] = []
    sx_files: List[str] = []
    axes_files: List[str] = []
    dif_files: List[str] = []
    for i, line in enumerate(lines):
        low = line.lower()
        if "texture file" in low:
            _, val = _next_nonempty_nonstar(lines, i + 1)
            tex_files.append(val)
        elif "single crystal file" in low:
            _, val = _next_nonempty_nonstar(lines, i + 1)
            sx_files.append(val)
        elif "grain shape file" in low:
            _, val = _next_nonempty_nonstar(lines, i + 1)
            axes_files.append(val)
        elif "diffraction file" in low:
            j, val = _next_nonempty_nonstar(lines, i + 1)
            # idiff flag line may precede the file name
            tok = numeric_tokens(val)
            if tok and len(tok) <= 1:
                _, val = _next_nonempty_nonstar(lines, j + 1)
            dif_files.append(val)

    info.phases = [
        PhaseInfo(
            index=k + 1,
            texture_file=tex_files[k] if k < len(tex_files) else "",
            crystal_file=sx_files[k] if k < len(sx_files) else "",
            shape_file=axes_files[k] if k < len(axes_files) else "",
            diffraction_file=dif_files[k] if k < len(dif_files) else "",
        )
        for k in range(max(info.nph, len(tex_files), len(sx_files), 1))
    ]

    # Numeric control blocks with labels on the same line
    for _, line in data:
        vals = numeric_values(line)
        low = line.lower()
        if len(vals) >= 4 and "errs" in low:
            info.errs = vals[:4]
        if len(vals) >= 3 and "itmax" in low:
            info.itmax = [int(v) for v in vals[:3]]
        if len(vals) >= 2 and "interaction" in low:
            info.interaction = int(vals[0])
            info.neff = float(vals[1])
        if len(vals) >= 3 and "iupdate" in low:
            info.iupdate = (int(vals[0]), int(vals[1]), int(vals[2]))
        if len(vals) >= 1 and "nneigh" in low:
            info.nneigh = int(vals[0])
        if len(vals) >= 1 and "iflu" in low:
            info.iflu = int(vals[0])

    # Process files: scan *all* IVGVAR/process blocks, not just the first.
    info.process_files = []
    info.process_ivgvar = []
    for i, line in enumerate(lines):
        low = line.lower()
        if "ivgvar" in low and ("process" in low or "pcys" in low or "lankford" in low or "load" in low):
            j, val = _next_nonempty_nonstar(lines, i + 1)
            if numeric_tokens(val):
                info.process_ivgvar.append(int(numeric_values(val)[0]))
                _, fname = _next_nonempty_nonstar(lines, j + 1)
                if fname:
                    # Strip trailing inline comments like "filename.proc   ! label"
                    fname = re.split(r"[#!]", fname, maxsplit=1)[0].strip()
                    info.process_files.append(fname)
    return info



# =============================================================================
# Single crystal .sx parsing and editing
# =============================================================================

@dataclass
class SXParamLine:
    line_no: int
    values: List[str]
    comment: str
    raw: str
    category: str


@dataclass
class SXInfo:
    family: str = "cubic"
    crystal_class: str = ""
    unit_cell: List[float] = field(default_factory=list)
    elastic_start: int = -1
    elastic_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((6, 6))
    )
    params: List[SXParamLine] = field(default_factory=list)


def infer_family(text: str) -> str:
    """Infer lattice family from VPSC .sx content.

    VPSC single-crystal files often write only ``CUBIC crysym`` for both
    FCC and BCC.  Therefore we first inspect explicit material keywords,
    then characteristic slip families.  For plotting, FCC and BCC share the
    same cubic symmetry, but keeping the label improves default poles and
    user-facing summaries.

    Matching is done on *whole alphabetic tokens* (not raw substrings) so that
    short, ambiguous keys such as ``mg``, ``zr`` or ``cub`` cannot be triggered
    accidentally by unrelated identifiers, comments or file paths.  Slip-family
    patterns that contain braces/brackets are matched on a whitespace-stripped
    copy, where substring matching is unambiguous.
    """
    low = text.lower()
    compact = low.replace(" ", "")
    words = set(re.findall(r"[a-z]+", low))

    def has_word(*keys: str) -> bool:
        return any(k in words for k in keys)

    if (has_word("hexagonal", "hex", "hcp", "magnesium", "zirconium",
                 "zircaloy", "titanium", "mg", "zr")
            or "hexag" in compact or "titan" in compact
            or "{0001}" in compact or "<1120>" in compact
            or "<11-20>" in compact or "<-12-10>" in compact):
        return "hcp"
    if (has_word("bcc", "ferrite", "ferritic", "martensite")
            or "bodycenter" in compact or "body-cent" in low):
        return "bcc"
    # Characteristic BCC slip entries: {110}<111>, {112}<111>, {123}<111>.
    if (("{110}<111>" in compact or "{112}<111>" in compact
            or "{123}<111>" in compact) and "{111}<110>" not in compact):
        return "bcc"
    if (has_word("fcc", "austenite", "austenitic", "aluminium", "aluminum",
                 "copper", "nickel")
            or "facecenter" in compact or "face-cent" in low
            or "{111}<110>" in compact):
        return "fcc"
    if has_word("cubic", "cub"):
        return "cubic"
    return "cubic"


def line_category(line: str) -> str:
    """Label-driven category for VPSC .sx numeric rows."""
    low = line.lower()
    if any(k in low for k in ["rho", "burg", "kgener", "drag", "edot",
                              "deb", "chi", "dd", "dislocation"]):
        return "Dislocation-density / DD"
    if any(k in low for k in ["mts", "thermal", "activation", "threshold"]):
        return "MTS / thermal activation"
    if any(k in low for k in ["tau0", "tau1", "thet", "hpfac", "crss", "voce", "hlatex"]):
        return "Voce / phenomenological"
    if any(k in low for k in ["rate", "nrs", "gamd", "irate", "grsze", "grain size"]):
        return "Flow / rate sensitivity"
    if any(k in low for k in ["mode", "modex", "slip", "twin", "isectw",
                              "thres", "twsh", "nsmx", "isense", "itwtype"]):
        return "Modes / slip-twin"
    if any(k in low for k in ["elastic", "cij", "stiff", "compliance"]):
        return "Elastic"
    if any(k in low for k in ["lattice", "cell", "cdim", "cang", "c/a"]):
        return "Lattice"
    return "Other numeric"


def split_values_comment(raw: str) -> Tuple[List[str], str, str]:
    """Return numeric tokens, trailing label/comment, and numeric prefix.

    VPSC .sx rows often put the parameter label after the numbers without an
    explicit comment marker, e.g. ``255 700 650 0 0 tau0x,tau1x,...``.  This
    parser treats text after the last numeric token as the semantic label
    while still respecting explicit ``#`` / ``!`` markers.
    """
    matches = list(NUM_RE.finditer(raw))
    if not matches:
        return [], raw.strip(), ""
    vals = [m.group(0) for m in matches]
    last_end = matches[-1].end()
    tail = raw[last_end:].strip()
    explicit = re.search(r"[#!].*", raw)
    comment = explicit.group(0).strip() if explicit else tail
    head = raw[:last_end]
    return vals, comment, head


def parse_sx(path: Path) -> SXInfo:
    text = read_text(path)
    lines = text.splitlines()
    info = SXInfo(family=infer_family(text))

    # Prefer an explicit ``crysym`` label, then fall back to the first data line.
    for line in lines:
        if "crysym" in line.lower():
            vals = line.strip().split()
            if vals:
                info.crystal_class = vals[0]
                break
    if not info.crystal_class:
        for line in lines:
            s0 = line.strip()
            if s0 and not s0.startswith(("*", "#", "!")):
                info.crystal_class = s0.split()[0]
                break

    # Unit cell: first line with 3 lengths > 0 and 3 angles in [20, 180]
    for line in lines:
        vals = numeric_values(line)
        if len(vals) >= 6:
            if all(v > 0 for v in vals[:3]) and all(20 <= v <= 180 for v in vals[3:6]):
                info.unit_cell = vals[:6]
                break

    # Elastic matrix: six consecutive lines each holding >=6 numeric values.
    # Prefer a block that appears right after an explicit elastic/stiffness
    # label, and otherwise accept only a block that physically looks like a 6x6
    # stiffness (finite, positive diagonal, approximately symmetric) so that
    # unrelated numeric tables are not mistaken for Cij.
    def _looks_like_stiffness(arr: np.ndarray) -> bool:
        if not np.all(np.isfinite(arr)) or np.linalg.norm(arr) <= 0:
            return False
        if np.any(np.diag(arr) <= 0):
            return False
        scale = float(np.max(np.abs(arr))) or 1.0
        return bool(np.allclose(arr, arr.T, atol=1e-3 * scale))

    def _scan_elastic(start: int, end: int, require_symmetric: bool) -> bool:
        for i in range(max(0, start), min(end, len(lines) - 5)):
            block = [numeric_values(lines[i + j]) for j in range(6)]
            if not all(len(b) >= 6 for b in block):
                continue
            arr = np.array([b[:6] for b in block], dtype=float)
            ok = (_looks_like_stiffness(arr) if require_symmetric
                  else (np.all(np.isfinite(arr)) and np.linalg.norm(arr) > 0))
            if ok:
                info.elastic_start = i + 1  # 1-indexed
                info.elastic_matrix = arr
                return True
        return False

    label_hint = -1
    for idx, line in enumerate(lines):
        if any(k in line.lower() for k in ("elast", "cij", "stiff", "compliance")):
            label_hint = idx
            break

    found = False
    if label_hint >= 0:
        found = _scan_elastic(label_hint, label_hint + 8, require_symmetric=False)
    if not found:
        found = _scan_elastic(0, len(lines), require_symmetric=True)
    if not found:
        _scan_elastic(0, len(lines), require_symmetric=False)

    params: List[SXParamLine] = []
    for i, raw in enumerate(lines, start=1):
        vals, comment, _ = split_values_comment(raw)
        if not vals:
            continue
        cat = line_category(raw)
        if info.elastic_start >= 0 and info.elastic_start <= i < info.elastic_start + 6:
            cat = "Elastic matrix row"
        params.append(SXParamLine(i, vals, comment, raw, cat))
    info.params = params
    return info


def replace_numeric_values_in_line(raw: str, new_values_text: str) -> str:
    """Replace only the numeric prefix of a .sx line, preserve labels/comments."""
    _old_vals, comment, _head = split_values_comment(raw)
    new_vals = numeric_tokens(new_values_text)
    if not new_vals:
        return raw
    m = re.match(r"^\s*", raw)
    prefix_space = m.group(0) if m else ""
    label = (comment or "").strip()
    if label and not label.startswith(("#", "!")):
        return prefix_space + "  ".join(new_vals) + "        " + label
    return prefix_space + "  ".join(new_vals) + ("  " + label if label else "")


def apply_sx_param_changes(path: Path, changed: Dict[int, str]) -> None:
    lines = read_text(path).splitlines()
    for line_no, values_text in changed.items():
        if 1 <= line_no <= len(lines):
            lines[line_no - 1] = replace_numeric_values_in_line(lines[line_no - 1], values_text)
    write_text(path, "\n".join(lines) + "\n")


def apply_sx_elastic_changes(path: Path, start_line: int, matrix: np.ndarray) -> None:
    lines = read_text(path).splitlines()
    for j in range(6):
        idx = start_line - 1 + j
        if 0 <= idx < len(lines):
            _vals, comment, _ = split_values_comment(lines[idx])
            lines[idx] = (
                "  "
                + "  ".join(f"{matrix[j, k]:.8g}" for k in range(6))
                + ("  " + comment if comment else "")
            )
    write_text(path, "\n".join(lines) + "\n")


def build_sx_theory_notes(info: "SXInfo") -> str:
    """Produce a concise theory/parameter guide for the loaded .sx file."""
    counts: Dict[str, int] = {}
    examples: Dict[str, List[SXParamLine]] = {}
    for pl in info.params:
        counts[pl.category] = counts.get(pl.category, 0) + 1
        examples.setdefault(pl.category, [])
        if len(examples[pl.category]) < 5:
            examples[pl.category].append(pl)

    lines: List[str] = []
    lines.append(f"Crystal class : {info.crystal_class}")
    lines.append(f"Inferred family: {info.family.upper()}")
    lines.append(f"Elastic block  : line {info.elastic_start if info.elastic_start >= 0 else 'n/a'}")
    if info.unit_cell:
        lines.append("Unit cell      : " + "  ".join(fmt_num(v) for v in info.unit_cell))
    lines.append("")
    lines.append("Detected numeric parameter groups")
    lines.append("---------------------------------")
    for cat in ["Modes / slip-twin", "Flow / rate sensitivity", "Voce / phenomenological",
                "MTS / thermal activation", "Dislocation-density / DD", "Elastic",
                "Lattice", "Other numeric", "Elastic matrix row"]:
        if cat in counts:
            lines.append(f"{cat}: {counts[cat]} lines")
            for pl in examples.get(cat, []):
                label = pl.comment or pl.raw.strip()[:70]
                vals = " ".join(pl.values[:8])
                lines.append(f"  line {pl.line_no:>4}: {vals:<32}  {label}")
    lines.append("")
    lines.append("Formula notes")
    lines.append("-------------")
    lines.append("Rate-sensitive slip/twin flow:")
    lines.append("  gamma_dot^s = gamma_dot0 |tau^s/tau_c^s|^n sign(tau^s)")
    lines.append("  tau^s = m^s : sigma'   with m^s = sym(b^s ⊗ n^s)")
    lines.append("Twinning is polar: negative resolved shear does not activate the twin.")
    lines.append("")
    lines.append("Extended Voce hardening used by VPSC:")
    lines.append("  tau_c(Gamma)=tau0 + (tau1 + theta1 Gamma)[1-exp(-Gamma theta0/tau1)]")
    lines.append("  hlatex controls latent hardening coupling between active modes.")
    lines.append("")
    lines.append("Dislocation-density/DD files update CRSS from forest/reversible/debris")
    lines.append("density evolution after each deformation increment.  Keep units consistent")
    lines.append("with the .sx comments: MPa/GPa, Burgers vector, grain size and density.")
    lines.append("")
    lines.append("Orientation convention used by this App")
    lines.append("---------------------------------------")
    lines.append("g maps sample -> crystal:  v_xtal = g · v_sample.")
    lines.append("PF uses v_sample = g.T · p_xtal; IPF uses v_xtal = g · s_sample.")
    return "\n".join(lines)


# =============================================================================
# Texture parsing and orientation math
#
# CONVENTIONS (Bunge)
# -------------------
# The matrix ``g = euler_bunge_g(phi1, Phi, phi2)`` is the *standard* Bunge
# active rotation matrix such that
#
#     x_crystal = g · x_sample            (i.e. g maps SAMPLE → CRYSTAL)
#
# Equivalently g[i, j] = e^c_i · e^s_j.  Therefore:
#
#     v_sample  = g^T · v_crystal         (used by pole figures)
#     v_crystal = g    · v_sample         (used by inverse pole figures)
#
# The same convention is used consistently in PF and IPF calculations.
# =============================================================================

@dataclass
class TextureData:
    eulers: np.ndarray            # (n, 3) Bunge angles in degrees
    weights: np.ndarray           # (n,)   normalised to sum=1
    raw_rows: np.ndarray          # original parsed rows (n, ≥3)
    _matrices: Optional[np.ndarray] = None  # cache of (n, 3, 3) g matrices

    @property
    def n(self) -> int:
        return int(self.eulers.shape[0])

    # Convenience aliases.
    @property
    def n_grains(self) -> int:
        return self.n

    @property
    def convention(self) -> str:
        # Bunge is the only convention this app reads/writes.
        return "Bunge"

    @property
    def matrices(self) -> np.ndarray:
        """Lazily build and cache the (n, 3, 3) stack of Bunge g matrices."""
        if self._matrices is None or len(self._matrices) != self.n:
            self._matrices = euler_bunge_g_batch(self.eulers)
        return self._matrices


def parse_texture(path: Path, max_rows: Optional[int] = None) -> TextureData:
    """Parse a VPSC texture file or TEX_PH*.OUT robustly.

    TEX_PH*.OUT can contain many texture dumps concatenated.  We pick the
    *largest complete* block (preferring the latest in case of ties) rather
    than blindly the last block, which can be truncated for an aborted run.
    """
    lines = read_text(path).splitlines()

    def find_blocks() -> List[Tuple[int, np.ndarray]]:
        # returns list of (start_line_index, arr)
        blocks: List[Tuple[int, np.ndarray]] = []
        i = 0
        while i < len(lines):
            s = lines[i].strip()
            m = re.match(r"^[A-Za-z]\s+(\d+)\b", s)
            if not m:
                i += 1
                continue
            n_expected = int(m.group(1))
            rows: List[List[float]] = []
            j = i + 1
            while j < len(lines) and len(rows) < n_expected:
                vals = numeric_values(lines[j])
                if len(vals) >= 3:
                    rows.append(vals[:4] if len(vals) >= 4 else vals[:3] + [1.0])
                j += 1
            if len(rows) >= max(3, min(n_expected, 10)):
                arr = np.asarray(rows[:n_expected], dtype=float)
                blocks.append((i, arr))
            i = max(j, i + 1)
        return blocks

    blocks = find_blocks()
    if blocks:
        # Pick the largest block; on tie, prefer the latest.
        blocks.sort(key=lambda kv: (kv[1].shape[0], kv[0]))
        arr = blocks[-1][1]
        if max_rows:
            arr = arr[:max_rows]
    else:
        rows: List[List[float]] = []
        for line in lines:
            s = line.strip()
            if not s or s[0] in "#*!":
                continue
            vals = numeric_values(s)
            if len(vals) >= 3:
                rows.append(vals[:4] if len(vals) >= 4 else vals[:3] + [1.0])
                if max_rows and len(rows) >= max_rows:
                    break
        if not rows:
            return TextureData(np.zeros((0, 3)), np.zeros(0), np.zeros((0, 0)))
        arr = np.asarray(rows, dtype=float)

    eulers = arr[:, :3].astype(float).copy()
    # Bunge angles: phi1, phi2 free modulo 360; Phi ∈ [0, 180].
    # When Phi falls outside [0, 180], the Euler equivalent is
    #   (phi1 + 180, 360 - Phi, phi2 + 180)
    eulers[:, 0] = np.mod(eulers[:, 0], 360.0)
    eulers[:, 2] = np.mod(eulers[:, 2], 360.0)
    Phi = np.mod(eulers[:, 1], 360.0)
    flip = Phi > 180.0
    Phi = np.where(flip, 360.0 - Phi, Phi)
    eulers[:, 1] = Phi
    eulers[flip, 0] = np.mod(eulers[flip, 0] + 180.0, 360.0)
    eulers[flip, 2] = np.mod(eulers[flip, 2] + 180.0, 360.0)

    weights = arr[:, 3].astype(float) if arr.shape[1] >= 4 else np.ones(len(arr))
    weights = np.where(np.isfinite(weights) & (weights >= 0), weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0:
        weights = np.ones_like(weights)
        total = float(np.sum(weights))
    return TextureData(eulers, weights / total, arr)



def euler_bunge_g_batch(eulers_deg: np.ndarray) -> np.ndarray:
    """Vectorised batch of Bunge g matrices, shape (n, 3, 3)."""
    if eulers_deg.size == 0:
        return np.zeros((0, 3, 3))
    e = np.deg2rad(eulers_deg.astype(float))
    ph, th, tm = e[:, 0], e[:, 1], e[:, 2]
    c1, s1 = np.cos(ph), np.sin(ph)
    c, s = np.cos(th), np.sin(th)
    c2, s2 = np.cos(tm), np.sin(tm)
    g = np.empty((len(e), 3, 3))
    g[:, 0, 0] =  c1*c2 - s1*s2*c
    g[:, 0, 1] =  s1*c2 + c1*s2*c
    g[:, 0, 2] =  s2*s
    g[:, 1, 0] = -c1*s2 - s1*c2*c
    g[:, 1, 1] = -s1*s2 + c1*c2*c
    g[:, 1, 2] =  c2*s
    g[:, 2, 0] =  s1*s
    g[:, 2, 1] = -c1*s
    g[:, 2, 2] =  c
    return g


# --- Miller index handling ---------------------------------------------------

def parse_miller_token(token: str) -> List[int]:
    """Parse a Miller token like ``111``, ``11-20`` or ``[10-10]``."""
    t = token.strip().replace("(", "").replace(")", "")
    t = t.replace("[", "").replace("]", "")
    t = t.replace(" ", "").replace(",", "")
    if not t:
        return [0, 0, 1]
    nums = re.findall(r"-?\d", t)
    if len(nums) >= 4:
        return [int(x) for x in nums[:4]]
    if len(nums) >= 3:
        return [int(x) for x in nums[:3]]
    vals = [int(x) for x in numeric_tokens(t)]
    return vals if vals else [0, 0, 1]


def parse_poles(text: str) -> List[List[int]]:
    """Parse PF pole lists in several texture3/VPSC-friendly forms.

    Accepted examples::

        100 110 111
        100; 110; 111
        [1 0 0]; [1 1 0]; [1 1 1]
        0001; 10-10; 11-20

    If the user enters separated single integers, e.g. ``1 0 0 1 1 0``,
    they are grouped into triples.  Compact HCP 4-index tokens are kept as
    single poles.
    """
    txt = str(text).strip()
    if not txt:
        return [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
    if any(sep in txt for sep in ";,\n"):
        parts = [q.strip() for q in re.split(r"[;,\n]+", txt) if q.strip()]
    else:
        toks = [q.strip() for q in txt.split() if q.strip()]
        # group single-number tokens into triples: 1 0 0  1 1 0
        if toks and all(re.fullmatch(r"[-+]?\d+", q) for q in toks) and len(toks) % 3 == 0:
            parts = [" ".join(toks[i:i+3]) for i in range(0, len(toks), 3)]
        else:
            parts = toks
    poles = [parse_miller_token(q) for q in parts]
    return poles or [[1, 0, 0], [1, 1, 0], [1, 1, 1]]


def miller_cartesian(index: Sequence[int], family: str = "cubic",
                     c_over_a: float = 1.633) -> np.ndarray:
    """Return a unit vector representing the plane normal direction.

    For cubic family the plane normal of (hkl) is parallel to [hkl] in the
    same Cartesian basis as the unit cell, so the result is just
    ``[h, k, l] / |[h, k, l]|``.

    For HCP the function uses the proper *reciprocal* lattice vector of a
    (hkil) or (hkl) plane.  With Cartesian lattice vectors
        a1 = a(1, 0, 0),  a2 = a(-1/2, √3/2, 0),  a3 = c(0, 0, 1)
    the reciprocal vectors are
        a1* = (1/a)(1, 1/√3, 0)
        a2* = (1/a)(0, 2/√3, 0)
        a3* = (1/c)(0, 0, 1)
    so the plane normal (hkl) in Cartesian (with a=1) is
        n = (h, (h + 2k)/√3, l / (c/a))
    The 4-index (hkil) form is reduced via i = -h - k.

    
    """
    fam = family.lower()
    vals = list(index)
    if fam in {"hcp", "hex", "hexagonal"}:
        if len(vals) == 4:
            h, k, _i, l = vals
        else:
            h, k, l = (vals + [0, 0, 0])[:3]
        x = float(h)
        y = (h + 2.0 * k) / math.sqrt(3.0)
        z = float(l) / max(c_over_a, 1e-12)
        v = np.array([x, y, z], dtype=float)
    else:
        h, k, l = (vals + [0, 0, 0])[:3]
        v = np.array([h, k, l], dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return v / n


# --- Symmetry equivalents ----------------------------------------------------

def cubic_equivalents(v: np.ndarray) -> np.ndarray:
    """48-fold cubic m-3m symmetry equivalents (signed permutations)."""
    import itertools
    outs: List[np.ndarray] = []
    for p in itertools.permutations([0, 1, 2]):
        base = v[list(p)]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    outs.append(base * np.array([sx, sy, sz], dtype=float))
    arr = np.unique(np.round(np.asarray(outs), 12), axis=0)
    return arr


def hcp_equivalents(v: np.ndarray) -> np.ndarray:
    """24-fold 6/mmm point group equivalents (6 rotations × σh × σv × inv).

    
    """
    outs: List[np.ndarray] = []
    for deg in range(0, 360, 60):
        a = math.radians(deg)
        R = np.array([
            [math.cos(a), -math.sin(a), 0.0],
            [math.sin(a),  math.cos(a), 0.0],
            [0.0,          0.0,         1.0],
        ], dtype=float)
        for sigma_v in (False, True):  # vertical mirror y -> -y
            for sigma_h in (False, True):  # basal mirror z -> -z
                w = R @ v
                if sigma_v:
                    w = w * np.array([1.0, -1.0, 1.0])
                if sigma_h:
                    w = w * np.array([1.0, 1.0, -1.0])
                outs.append(w)
                outs.append(-w)
    arr = np.unique(np.round(np.asarray(outs), 12), axis=0)
    return arr


def equivalents_for_family(v: np.ndarray, family: str) -> np.ndarray:
    return hcp_equivalents(v) if family.lower() in {"hcp", "hex", "hexagonal"} \
        else cubic_equivalents(v)


# --- PF and IPF data generation (fully vectorised) ---------------------------


def reduce_cubic_ipf(v: np.ndarray) -> np.ndarray:
    """Fold a direction into the cubic standard triangle 001-101-111.

    Vectorised: accepts (3,) or (..., 3).
    """
    a = np.sort(np.abs(np.atleast_2d(v).astype(float)), axis=-1)
    # small, middle, large -> place as (x=middle, y=small, z=large)
    w = np.stack([a[..., 1], a[..., 0], a[..., 2]], axis=-1)
    n = np.linalg.norm(w, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return w / n


def reduce_hcp_ipf(v: np.ndarray) -> np.ndarray:
    """Fold a direction into the HCP standard triangle 0001-10-10-11-20.

    Vectorised.  Uses 6-fold rotation + mirror to bring the azimuth into the
    range [0, π/6].
    """
    w = np.atleast_2d(v).astype(float).copy()
    # Move to upper hemisphere
    w[w[:, 2] < 0] *= -1.0
    r = np.hypot(w[:, 0], w[:, 1])
    safe = r >= 1e-12
    phi = np.zeros_like(r)
    phi[safe] = np.arctan2(w[safe, 1], w[safe, 0])
    phi = np.mod(phi, math.pi / 3.0)
    mask = phi > math.pi / 6.0
    phi = np.where(mask, math.pi / 3.0 - phi, phi)
    out = np.stack([r * np.cos(phi), r * np.sin(phi), np.abs(w[:, 2])], axis=-1)
    n = np.linalg.norm(out, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return out / n




def smooth_hist(H: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return H
    if SCIPY_AVAILABLE and gaussian_filter is not None:
        return gaussian_filter(H, sigma=sigma, mode="nearest")
    out = H.copy()
    for _ in range(int(max(1, round(sigma)))):
        pad = np.pad(out, 1, mode="edge")
        out = (pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:]
               + pad[1:-1, :-2] + 4 * pad[1:-1, 1:-1] + pad[1:-1, 2:]
               + pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]) / 12.0
    return out


@dataclass
class PlotStyle:
    line_color: str = "Blue"
    line_style: str = "-"
    line_width: float = 2.0
    marker: str = "o"
    marker_color: str = "Blue"
    marker_size: float = 4.0
    point_color: str = "Blue"
    point_size: float = 10.0
    alpha: float = 0.75
    contour_color: str = "Black"
    contour_style: str = "-"
    contour_width: float = 0.8
    cmap: str = "magma"
    levels: int = 8
    bins: int = 81
    smooth: float = 1.2
    colorbar: bool = True
    log_scale: bool = False
    grid: bool = True
    manual_levels: str = ""
    texture_mode: str = "contourf"
    projection: str = "equal-area"



def draw_density(ax: Any, x: np.ndarray, y: np.ndarray, w: np.ndarray,
                 style: PlotStyle, region: str,
                 verts: Optional[np.ndarray] = None,
                 n_equivalents: int = 1) -> Any:
    """Smoothed MRD-like density map for PF / IPF.

    The histogram is normalised by the *mean density over the valid projected
    domain*, including a factor ``1 / n_equivalents`` so that PF densities are
    independent of crystal symmetry (each grain contributes ``n_equivalents``
    raw points, which would otherwise inflate the apparent MRD).
    """
    if x.size == 0:
        return None
    bins = max(25, min(301, int(style.bins)))
    if region == "circle":
        xlim = (-1.0, 1.0)
        ylim = (-1.0, 1.0)
        valid_area = math.pi
    elif verts is not None:
        xlim = (float(np.min(verts[:, 0])), float(np.max(verts[:, 0])))
        ylim = (float(np.min(verts[:, 1])), float(np.max(verts[:, 1])))
        vx, vy = verts[:, 0], verts[:, 1]
        valid_area = 0.5 * abs(np.dot(vx, np.roll(vy, -1)) - np.dot(vy, np.roll(vx, -1)))
    else:
        xlim = (float(np.min(x)), float(np.max(x)))
        ylim = (float(np.min(y)), float(np.max(y)))
        valid_area = max((xlim[1] - xlim[0]) * (ylim[1] - ylim[0]), 1e-12)

    H, xe, ye = np.histogram2d(x, y, bins=bins, range=[xlim, ylim], weights=w)
    H = smooth_hist(H.T, float(style.smooth))
    X = 0.5 * (xe[:-1] + xe[1:])
    Y = 0.5 * (ye[:-1] + ye[1:])
    XX, YY = np.meshgrid(X, Y)
    if region == "circle":
        physical_mask = (XX ** 2 + YY ** 2) <= 1.0
        # Plot slightly beyond the physical circle and clip to an analytic circle.
        # This removes the white saw-tooth holes that appear when contourf sees
        # NaN cells right at the projection rim (equal-area by default).
        plot_mask = np.ones_like(XX, dtype=bool)
    elif verts is not None:
        physical_mask = MplPath(verts).contains_points(
            np.column_stack([XX.ravel(), YY.ravel()]), radius=1e-10
        ).reshape(XX.shape)
        # As for pole figures, draw the surrounding rectangular grid and clip to
        # the curved fundamental sector to avoid jagged blank cells on the rim.
        plot_mask = np.ones_like(XX, dtype=bool)
    else:
        physical_mask = np.ones_like(XX, dtype=bool)
        plot_mask = physical_mask
    H[~physical_mask] = 0.0

    cell_area = max((xlim[1] - xlim[0]) * (ylim[1] - ylim[0]) / (bins * bins), 1e-30)
    total_weight = max(float(np.nansum(w)), 1e-30)
    # total_weight is sum of replicated grain weights.  Dividing by n_equivalents
    # recovers the underlying grain-weight total, so MRD becomes a per-crystal
    # quantity.  For IPF, n_equivalents == 1 by construction.
    grain_total = total_weight / max(n_equivalents, 1)
    mean_density = grain_total / max(valid_area, 1e-30)
    Z = (H / cell_area / max(n_equivalents, 1)) / max(mean_density, 1e-30)

    if style.log_scale:
        Z = np.log10(np.maximum(Z, 1.0e-6))
    finite = Z[np.isfinite(Z) & physical_mask]
    if finite.size == 0:
        return None

    nlev = max(4, int(style.levels))
    manual = (style.manual_levels or "").strip()
    if manual:
        try:
            levels = [float(v) for v in re.split(r"[,;\s]+", manual) if v.strip()]
        except ValueError:
            levels = []
        if len(levels) < 2:
            levels = list(np.linspace(float(np.nanmin(finite)),
                                       float(np.nanmax(finite)), nlev))
    else:
        # For linear MRD maps start at zero so clipped cells at the PF/IPF
        # boundary are filled with the lowest colour instead of becoming white
        # when their value is below the first level.
        zmin = (0.0 if not style.log_scale else float(np.nanmin(finite)))
        zmax = float(np.nanmax(finite))
        if zmax <= zmin + 1e-12:
            zmax = zmin + 1.0
        levels = list(np.linspace(zmin, zmax, nlev))

    mode = style.texture_mode
    im = None
    Zp = np.ma.array(Z, mask=~plot_mask)
    clip_patch = None
    if region == "circle":
        clip_patch = Circle((0.0, 0.0), 1.0005, transform=ax.transData, facecolor="none", edgecolor="none")
        ax.add_patch(clip_patch)
    elif verts is not None:
        clip_patch = PathPatch(MplPath(verts), transform=ax.transData, facecolor="none", edgecolor="none")
        ax.add_patch(clip_patch)
    if mode in {"density", "contourf", "both"}:
        im = ax.contourf(XX, YY, Zp, levels=levels, cmap=style.cmap,
                         antialiased=False, extend="neither", corner_mask=False)
        if clip_patch is not None:
            try:
                im.set_clip_path(clip_patch)
            except Exception:
                pass
        if hasattr(im, "collections"):
            for coll in im.collections:
                if clip_patch is not None:
                    try:
                        coll.set_clip_path(clip_patch)
                    except Exception:
                        pass
                try:
                    coll.set_edgecolor("face")
                    coll.set_linewidth(0.0)
                except Exception:
                    pass
    if mode in {"contour", "both"}:
        cs = ax.contour(XX, YY, Zp, levels=levels,
                        colors=resolve_color(style.contour_color),
                        linestyles=style.contour_style,
                        linewidths=style.contour_width, corner_mask=False)
        if clip_patch is not None:
            try:
                cs.set_clip_path(clip_patch)
            except Exception:
                pass
        if hasattr(cs, "collections"):
            for coll in cs.collections:
                if clip_patch is not None:
                    try:
                        coll.set_clip_path(clip_patch)
                    except Exception:
                        pass
    return im



# =============================================================================
# Process / boundary condition parsing
# =============================================================================

@dataclass
class ProcessInfo:
    nsteps: int = 0
    ictrl: int = 0
    eqincr: float = 0.0
    temp_i: float = 298.0
    temp_f: float = 298.0
    iudot: np.ndarray = field(default_factory=lambda: np.ones((3, 3), dtype=int))
    udot: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=float))
    iscau: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=int))
    scauchy: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))
    raw: str = ""
    variable_history: bool = False
    fe_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 11), dtype=float))

    # ``temperature`` is a convenience alias for the GUI which exposes a
    # single thermostat StringVar.  Reading returns the initial temperature;
    # writing applies the same value to both initial and final temperatures
    # so a one-control isothermal step is the default behaviour.
    @property
    def temperature(self) -> float:
        return self.temp_i

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.temp_i = float(value)
        self.temp_f = float(value)


@dataclass
class FEHistoryInfo:
    """Parsed FE/VPSC velocity-gradient history metadata."""
    history: np.ndarray
    source_format: str
    declared_nsteps: Optional[int] = None
    ictrl: Optional[int] = None
    eqincr: Optional[float] = None
    temperature: Optional[float] = None
    skipped_lines: int = 0

    def summary(self) -> str:
        parts = [f"format={self.source_format}", f"rows={self.history.shape[0]}"]
        if self.declared_nsteps is not None:
            parts.append(f"declared_nsteps={self.declared_nsteps}")
        if self.ictrl is not None:
            parts.append(f"legacy_ictrl={self.ictrl}")
        if self.eqincr is not None:
            parts.append(f"legacy_eqincr={self.eqincr:g}")
        if self.temperature is not None:
            parts.append(f"legacy_temp={self.temperature:g}")
        return ", ".join(parts)


def _history_candidate_entries(text: str) -> List[Tuple[int, str, List[float]]]:
    """Collect standalone numeric rows for velocity-gradient histories."""
    entries: List[Tuple[int, str, List[float]]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        vals = strict_numeric_values(line)
        if vals:
            entries.append((line_no, line, vals))
    return entries


def _looks_like_step_sequence(rows: List[List[float]], ncheck: int = 8) -> bool:
    """Check whether first column is 1,2,3,... for ten-column histories."""
    if not rows:
        return False
    for i, vals in enumerate(rows[:min(ncheck, len(rows))], start=1):
        if len(vals) != 10:
            return False
        if abs(vals[0] - round(vals[0])) > 1e-8 or int(round(vals[0])) != i:
            return False
    return True


def read_fe_velocity_history_info(path: Path, default_dt: float = 1.0) -> FEHistoryInfo:
    """Read VPSC7, VPSC8, or raw FE velocity-gradient history.

    Accepted row layouts are:

    * VPSC8: first row is only ``nsteps``; data rows are
      ``step L11 L12 ... L33 dt``.
    * Legacy VPSC7-style header: first row is
      ``nsteps ictrl eqincr temp``; the remaining data rows are converted to
      the VPSC8 layout while the legacy fields are kept in metadata.
    * Raw FE table: rows contain either ``L11..L33``, ``L11..L33 dt``,
      ``step L11..L33`` or ``step L11..L33 dt``.
    """
    text = read_text(path)
    entries = _history_candidate_entries(text)
    if not entries:
        raise ValueError("No numeric rows were found in the FE/VPSC history file.")

    source_format = "raw_fe_table"
    declared_nsteps: Optional[int] = None
    ictrl: Optional[int] = None
    eqincr: Optional[float] = None
    temp: Optional[float] = None
    start_idx = 0

    first_line_no, first_line, first_vals = entries[0]
    first_low = first_line.lower()

    if len(first_vals) == 1 and len(entries) > 1:
        source_format = "vpsc8_variable_history"
        declared_nsteps = int(round(first_vals[0]))
        start_idx = 1
    elif (len(first_vals) >= 4 and "nstep" in first_low
          and any(len(v) >= 9 for _, _, v in entries[1:])):
        source_format = "vpsc7_legacy_history"
        declared_nsteps = int(round(first_vals[0]))
        ictrl = int(round(first_vals[1]))
        eqincr = float(first_vals[2])
        temp = float(first_vals[3])
        start_idx = 1

    data_vals = [vals for _, _, vals in entries[start_idx:] if len(vals) >= 9]
    if not data_vals:
        raise ValueError(
            "No valid velocity-gradient data rows were found. Expected L11..L33 "
            "with optional step and time increment."
        )

    ten_is_step_l = _looks_like_step_sequence([v for v in data_vals if len(v) == 10])
    hist: List[List[float]] = []
    for vals in data_vals:
        if len(vals) >= 11:
            row = [vals[0], *vals[1:10], vals[10]]
        elif len(vals) == 10:
            if ten_is_step_l:
                row = [vals[0], *vals[1:10], default_dt]
            else:
                row = [len(hist) + 1, *vals[:9], vals[9]]
        elif len(vals) == 9:
            row = [len(hist) + 1, *vals[:9], default_dt]
        else:
            continue
        hist.append([float(v) for v in row])

    if not hist:
        raise ValueError("Velocity-gradient rows were detected but could not be normalised.")

    arr = np.asarray(hist, dtype=float)
    # Force monotonically increasing integer step IDs for raw files without reliable step labels.
    if source_format == "raw_fe_table" and not np.allclose(arr[:, 0], np.arange(1, len(arr)+1)):
        arr[:, 0] = np.arange(1, len(arr)+1, dtype=float)

    return FEHistoryInfo(
        history=arr,
        source_format=source_format,
        declared_nsteps=declared_nsteps,
        ictrl=ictrl,
        eqincr=eqincr,
        temperature=temp,
        skipped_lines=max(0, len(entries) - start_idx - len(data_vals)),
    )


def vpsc8_deviatoric_velocity_gradient(L: np.ndarray) -> np.ndarray:
    """Return the trace-free velocity gradient used by VPSC8 IVGVAR=1."""
    A = np.asarray(L, dtype=float).reshape(3, 3).copy()
    A -= np.eye(3) * (np.trace(A) / 3.0)
    return A


def equivalent_rate_from_L(L: np.ndarray) -> float:
    """Von-Mises equivalent rate of the symmetric deviatoric part of L."""
    A = vpsc8_deviatoric_velocity_gradient(L)
    D = 0.5 * (A + A.T)
    D -= np.eye(3) * (np.trace(D) / 3.0)
    return float(math.sqrt(max(0.0, 2.0 / 3.0 * np.sum(D * D))))


def stabilise_history_for_vpsc8(
    history: np.ndarray,
    *,
    force: bool = False,
    enable_auto_rescale: bool = False,
    small_rate_threshold: float = 1.0e-3,
    target_deq: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Optionally rescale tiny L histories to avoid VPSC8 zero-stress starts.

    VPSC8 solves the rate problem from the magnitude of L and uses ``tincr``
    for the increment.  Very small legacy VPSC7/FE rates can trigger the
    ``TAUMAX<1e-10`` guard in the first Newton iteration.  When enabled, this
    routine writes an equivalent path with ``Deq≈target_deq`` and ``tincr``
    scaled inversely, so the *incremental* deformation ``L * tincr`` is
    preserved to first order.

    WARNING
    -------
    The rescaling changes the *strain rate* and projects each step onto its
    deviatoric part.  Because the VPSC viscoplastic flow rule is rate sensitive
    (rate-sensitivity exponent ``n``), the predicted stresses under the rescaled
    history will in general differ from those of the original input.  For this
    reason the transformation is **never applied silently**:

    * ``force=True``            -- apply unconditionally (explicit user opt-in);
    * ``enable_auto_rescale``   -- apply only if a small rate is auto-detected;
    * default (both False)      -- only *detect*; the returned ``info`` reports
      ``detected_small_rate`` so a caller can prompt the user, but the history
      is returned unchanged.
    """
    hist = np.asarray(history, dtype=float).copy()
    if hist.size == 0:
        return hist, {"applied": False, "detected_small_rate": False,
                      "reason": "empty"}
    deq = np.array([equivalent_rate_from_L(row[1:10].reshape(3, 3)) for row in hist], dtype=float)
    finite = deq[np.isfinite(deq) & (deq > 0)]
    median_deq = float(np.median(finite)) if finite.size else 0.0
    detected = bool(0.0 < median_deq < small_rate_threshold)
    apply = bool(force or (enable_auto_rescale and detected))
    info: Dict[str, Any] = {
        "applied": apply,
        "detected_small_rate": detected,
        "median_deq": median_deq,
        "min_deq": float(np.min(finite)) if finite.size else 0.0,
        "max_deq": float(np.max(finite)) if finite.size else 0.0,
        "target_deq": float(target_deq),
    }
    if not apply:
        return hist, info

    for i in range(hist.shape[0]):
        L = hist[i, 1:10].reshape(3, 3)
        Ldev = vpsc8_deviatoric_velocity_gradient(L)
        r = equivalent_rate_from_L(Ldev)
        if not np.isfinite(r) or r <= 1.0e-30:
            continue
        dt = float(hist[i, 10]) if hist.shape[1] >= 11 else 1.0
        hist[i, 1:10] = (Ldev * (target_deq / r)).reshape(9)
        hist[i, 10] = dt * (r / target_deq)
    return hist, info


def _parse_process_labelled(text: str):
    """Parse a process file by locating labelled boundary-condition blocks.

    Returns ``(nsteps, ictrl, eqincr, temp_i, temp_f, iudot, udot, iscau,
    scauchy)`` when an unambiguous, complete and flag-consistent block is found,
    otherwise ``None`` so the caller can fall back to the positional parser.
    This is robust to stray inline numbers and reordered fields that would
    misalign a purely positional read, and it round-trips files written by
    :func:`write_process`.
    """
    lines = text.splitlines()
    low = [ln.lower() for ln in lines]

    def find_label(*keys: str, exclude: Tuple[str, ...] = ()) -> int:
        for i, l in enumerate(low):
            if any(k in l for k in keys) and not any(x in l for x in exclude):
                return i
        return -1

    def block_after(idx: int, need: int) -> Optional[List[float]]:
        if idx < 0:
            return None
        vals: List[float] = []
        j = idx + 1
        while j < len(lines) and len(vals) < need:
            s = lines[j].strip()
            if s and s[0] not in "*#!":
                vals.extend(strict_numeric_values(lines[j]))
            j += 1
        return vals[:need] if len(vals) >= need else None

    i_iudot = find_label("iudot")
    i_udot = find_label("udot", exclude=("iudot",))
    i_iscau = find_label("iscau")
    i_scauchy = find_label("scauchy")
    labels = (i_iudot, i_udot, i_iscau, i_scauchy)
    if min(labels) < 0:
        return None

    iudot = block_after(i_iudot, 9)
    udot = block_after(i_udot, 9)
    iscau = block_after(i_iscau, 6)
    scauchy = block_after(i_scauchy, 6)
    if any(b is None for b in (iudot, udot, iscau, scauchy)):
        return None

    # Header (nsteps ictrl eqincr temp_i temp_f) must appear above the b.c.
    # block; restrict the search so a 9-value iudot row is never mistaken for it.
    first_label = min(labels)
    header: Optional[List[float]] = None
    for ln in lines[:first_label]:
        s = ln.strip()
        if not s or s[0] in "*#!":
            continue
        v = strict_numeric_values(ln)
        if len(v) >= 5:
            header = v[:5]
            break
    if header is None:
        return None

    iudot_a = np.array(iudot, dtype=int).reshape(3, 3)
    iscau_a = np.array(iscau, dtype=int)
    if not (set(np.unique(iudot_a)).issubset({0, 1})
            and set(np.unique(iscau_a)).issubset({0, 1})):
        return None
    return (int(header[0]), int(header[1]), float(header[2]),
            float(header[3]), float(header[4]),
            iudot_a, np.array(udot, dtype=float).reshape(3, 3),
            iscau_a, np.array(scauchy, dtype=float))


def parse_process(path: Path) -> ProcessInfo:
    """Read a constant-L process file or a variable-L history file."""
    text = read_text(path)
    pi = ProcessInfo(raw=text)
    entries = _history_candidate_entries(text)

    # VPSC8 variable history or VPSC7 legacy history can be loaded directly.
    is_variable_candidate = False
    if entries:
        first_vals = entries[0][2]
        first_low = entries[0][1].lower()
        has_history_rows = any(len(vals) >= 9 for _, _, vals in entries[1:])
        is_variable_candidate = (
            (len(first_vals) == 1 and has_history_rows)
            or (len(first_vals) >= 4 and "nstep" in first_low and has_history_rows)
        )

    if is_variable_candidate:
        try:
            info = read_fe_velocity_history_info(path, default_dt=1.0)
            hist = info.history
            pi.variable_history = True
            pi.fe_history = hist
            pi.nsteps = int(info.declared_nsteps or hist.shape[0])
            pi.ictrl = int(info.ictrl or 0)
            pi.eqincr = float(hist[0, 10])
            if info.temperature is not None:
                pi.temp_i = float(info.temperature)
                pi.temp_f = float(info.temperature)
            pi.iudot = np.ones((3, 3), dtype=int)
            pi.udot = hist[0, 1:10].reshape(3, 3)
            pi.iscau = np.zeros(6, dtype=int)
            pi.scauchy = np.zeros(6, dtype=float)
            return pi
        except ValueError:
            pass

    # Prefer the label-driven parser; fall back to the positional read below
    # so files that already parsed continue to parse identically.
    labelled = _parse_process_labelled(text)
    if labelled is not None:
        (pi.nsteps, pi.ictrl, pi.eqincr, pi.temp_i, pi.temp_f,
         pi.iudot, pi.udot, pi.iscau, pi.scauchy) = labelled
        return pi

    rows = [(ln, process_numeric_values(line)) for ln, line in noncomment_lines(text)]
    numeric_rows = [(ln, vals) for ln, vals in rows if vals]
    if not numeric_rows:
        return pi

    flat: List[float] = [v for _, vals in numeric_rows for v in vals]
    if len(flat) >= 5:
        pi.nsteps = int(flat[0])
        pi.ictrl = int(flat[1])
        pi.eqincr = float(flat[2])
        pi.temp_i = float(flat[3])
        pi.temp_f = float(flat[4])
        rest = flat[5:]
        if len(rest) >= 9:
            pi.iudot = np.array(rest[:9], dtype=int).reshape(3, 3)
        if len(rest) >= 18:
            pi.udot = np.array(rest[9:18], dtype=float).reshape(3, 3)
        if len(rest) >= 24:
            pi.iscau = np.array(rest[18:24], dtype=int)
        if len(rest) >= 30:
            pi.scauchy = np.array(rest[24:30], dtype=float)
    return pi


def write_process(path: Path, p: ProcessInfo) -> None:
    lines: List[str] = []
    lines.append(
        f"{int(p.nsteps):8d} {int(p.ictrl):8d} "
        f"{p.eqincr:12.6g} {p.temp_i:12.6g} {p.temp_f:12.6g}"
        f"    nsteps ictrl eqincr temp_i temp_f"
    )
    lines.append("* boundary conditions generated by VPSC Python App")
    lines.append("* iudot: velocity-gradient flags (1=known, 0=unknown)")
    for r in p.iudot:
        lines.append("  " + "  ".join(str(int(v)) for v in r))
    lines.append("* udot: velocity gradient")
    for r in p.udot:
        lines.append("  " + "  ".join(f"{float(v):.10g}" for v in r))
    lines.append("* iscau: Cauchy stress flags in order 11 22 33 23 13 12")
    lines.append("  " + "  ".join(str(int(v)) for v in p.iscau.reshape(-1)))
    lines.append("* scauchy: Cauchy stress values in order 11 22 33 23 13 12")
    lines.append("  " + "  ".join(f"{float(v):.10g}" for v in p.scauchy.reshape(-1)))
    write_text(path, "\n".join(lines) + "\n")



def velocity_gradient_presets() -> Dict[str, Dict[str, np.ndarray]]:
    """Common constant-L boundary conditions used in VPSC examples."""
    one = np.ones((3, 3), dtype=float)
    z6 = np.zeros(6, dtype=float)

    def item(L: Sequence[Sequence[float]], mask: Optional[np.ndarray] = None,
             iscau: Optional[Sequence[float]] = None,
             scauchy: Optional[Sequence[float]] = None) -> Dict[str, np.ndarray]:
        return {
            "iudot": np.asarray(mask if mask is not None else one, dtype=float),
            "udot": np.asarray(L, dtype=float),
            "iscau": np.asarray(iscau if iscau is not None else z6, dtype=float),
            "scauchy": np.asarray(scauchy if scauchy is not None else z6, dtype=float),
        }

    uniaxial_x_mask = np.ones((3, 3), dtype=float)
    uniaxial_x_mask[1, 1] = 0.0
    uniaxial_x_mask[2, 2] = 0.0
    uniaxial_y_mask = np.ones((3, 3), dtype=float)
    uniaxial_y_mask[0, 0] = 0.0
    uniaxial_y_mask[2, 2] = 0.0
    uniaxial_z_mask = np.ones((3, 3), dtype=float)
    uniaxial_z_mask[0, 0] = 0.0
    uniaxial_z_mask[1, 1] = 0.0
    lateral_free_x = [0, 1, 1, 1, 1, 1]
    lateral_free_y = [1, 0, 1, 1, 1, 1]
    lateral_free_z = [1, 1, 0, 1, 1, 1]

    return {
        "Rolling / plane strain compression": item([[1, 0, 0], [0, 0, 0], [0, 0, -1]]),
        "Plane strain tension X": item([[1, 0, 0], [0, 0, 0], [0, 0, -1]]),
        "Plane strain tension Y": item([[0, 0, 0], [0, 1, 0], [0, 0, -1]]),
        "Uniaxial strain tension X": item([[1, 0, 0], [0, -0.5, 0], [0, 0, -0.5]]),
        "Uniaxial strain tension Y": item([[-0.5, 0, 0], [0, 1, 0], [0, 0, -0.5]]),
        "Uniaxial strain tension Z": item([[-0.5, 0, 0], [0, -0.5, 0], [0, 0, 1]]),
        "Uniaxial stress tension X": item([[1, 0, 0], [0, 0, 0], [0, 0, 0]], uniaxial_x_mask, lateral_free_x),
        "Uniaxial stress tension Y": item([[0, 0, 0], [0, 1, 0], [0, 0, 0]], uniaxial_y_mask, lateral_free_y),
        "Uniaxial stress tension Z": item([[0, 0, 0], [0, 0, 0], [0, 0, 1]], uniaxial_z_mask, lateral_free_z),
        "Compression X": item([[-1, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]),
        "Compression Y": item([[0.5, 0, 0], [0, -1, 0], [0, 0, 0.5]]),
        "Compression Z": item([[0.5, 0, 0], [0, 0.5, 0], [0, 0, -1]]),
        "Equi-biaxial tension XY": item([[0.5, 0, 0], [0, 0.5, 0], [0, 0, -1]]),
        "Pure shear XY": item([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
        "Simple shear 12 / L12": item([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
        "Simple shear 21 / L21": item([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
        "Simple shear 13 / L13": item([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
        "Simple shear 31 / L31": item([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
        "Simple shear 23 / L23": item([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
        "Simple shear 32 / L32": item([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
        "Rigid rotation 12": item([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]),
        "Rigid rotation 13": item([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]),
        "Rigid rotation 23": item([[0, 0, 0], [0, 0, 1], [0, -1, 0]]),
        "Hydrostatic tension": item([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        "Hydrostatic compression": item([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        "Plane stress tension X": item([[1, 0, 0], [0, 0, 0], [0, 0, 0]], uniaxial_x_mask, [0, 1, 1, 0, 0, 0]),
        "Plane stress tension Y": item([[0, 0, 0], [0, 1, 0], [0, 0, 0]], uniaxial_y_mask, [1, 0, 1, 0, 0, 0]),
        "Plane stress biaxial XY": item([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0]], np.ones((3, 3), dtype=float), [0, 0, 1, 0, 0, 0]),
        "Zero / user-defined": item([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
    }


def write_variable_velocity_history(path: Path, history: np.ndarray) -> None:
    """Write VPSC8 IVGVAR=1 variable velocity-gradient history."""
    hist = np.asarray(history, dtype=float)
    lines = [
        f"{hist.shape[0]:8d}    nsteps",
        " step      L11        L12        L13        L21        L22        L23        L31        L32        L33        tincr",
    ]
    for i, row in enumerate(hist, start=1):
        step = int(round(row[0])) if row.size >= 11 else i
        vals = row[1:10] if row.size >= 11 else row[:9]
        dt = float(row[10]) if row.size >= 11 else 1.0
        lines.append(
            f"{step:6d} "
            + " ".join(f"{float(v):13.6e}" for v in vals)
            + f" {dt:13.6e}"
        )
    write_text(path, "\n".join(lines) + "\n")


def patch_first_vpsc8_process(path: Path, process_file: str, ivgvar: int = 1) -> None:
    """Set the first VPSC8 process block to a new IVGVAR and file name."""
    if not path.exists():
        raise FileNotFoundError(path)
    lines = read_text(path).splitlines()
    for i, line in enumerate(lines):
        low = line.lower()
        if "ivgvar" not in low:
            continue
        # Replace the next non-empty, non-comment line with the IVGVAR value.
        j = i + 1
        while j < len(lines) and (not lines[j].strip() or lines[j].lstrip().startswith("*")):
            j += 1
        if j >= len(lines):
            break
        indent_j = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
        lines[j] = f"{indent_j}{int(ivgvar)}"

        # Replace the following non-empty, non-comment line with the process file.
        k = j + 1
        while k < len(lines) and (not lines[k].strip() or lines[k].lstrip().startswith("*")):
            k += 1
        if k >= len(lines):
            raise ValueError("Found IVGVAR line but no process-file line after it.")
        indent_k = lines[k][:len(lines[k]) - len(lines[k].lstrip())]
        lines[k] = f"{indent_k}{process_file}"
        write_text(path, "\n".join(lines) + "\n")
        return
    raise ValueError("No IVGVAR/process block was found in VPSC8.IN.")


# =============================================================================
# Output parsing
# =============================================================================

def primary_history_length(x: np.ndarray) -> int:
    """Number of leading rows belonging to the primary deformation history.

    VPSC appends post-processing probe blocks to STR_STR.OUT / ACT_PHn.OUT:
    the PCYS yield-surface scan and the Lankford angular scan each restart the
    abscissa as integer probe indices (1, 2, 3, ...).  When the whole column is
    plotted these blocks appear as a spurious decline (stress-strain) or a
    zig-zag to large abscissa values such as 72 (activity).  This returns the
    length of the real deformation segment, i.e. up to the first abrupt break
    in the abscissa (a large upward jump to probe indices, or a large reset).

    The test is relative to the typical deformation increment so it adapts to
    any strain step size and does not trigger on the small bounces of a merely
    noisy/under-converged run.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return n
    d = np.diff(x)
    pos = d[d > 0]
    if pos.size == 0:
        return n
    head = pos[: max(3, pos.size // 3)]
    typ = float(np.median(head))
    if not np.isfinite(typ) or typ <= 0:
        typ = float(np.median(pos))
    thresh = max(0.1, 8.0 * typ)
    for i in range(1, n):
        step = x[i] - x[i - 1]
        if abs(step) > thresh:          # probe block start (jump) or reset
            return i
    return n


def read_numeric_table(path: Path) -> np.ndarray:
    """Read a whitespace/free-format numeric VPSC output table."""
    rows: List[List[float]] = []
    for line in read_text(path).splitlines():
        vals = numeric_values(line)
        if len(vals) >= 2:
            rows.append(vals)
    if not rows:
        return np.zeros((0, 0))
    n = max(len(r) for r in rows)
    arr = np.full((len(rows), n), np.nan)
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r
    return arr


def _looks_numeric(tok: str) -> bool:
    """True if ``tok`` parses as a float (handles Fortran 'D' exponents)."""
    try:
        float(tok.replace("D", "E").replace("d", "e"))
        return True
    except ValueError:
        return False


def read_table_header(path: Path) -> Optional[List[str]]:
    """Return the column-name tokens of a VPSC table, or None if headerless.

    The header is the first non-empty line whose tokens are not all numeric.
    """
    for line in read_text(path).splitlines():
        s = line.strip()
        if not s:
            continue
        toks = s.split()
        return toks if not all(_looks_numeric(t) for t in toks) else None
    return None


def read_numeric_table_no_header(path: Path) -> np.ndarray:
    """Read a numeric VPSC table, skipping every text header line.

    VPSC column labels carry embedded digits (E11, E22, MODE1, WRATE2, SDEV11),
    which a naive numeric reader would parse as data rows. A header line is one
    whose tokens are not all pure numbers; such lines occur once per appended
    block (deformation, PCYS scan, Lankford scan) and are all skipped.
    """
    rows: List[List[float]] = []
    for ln in read_text(path).splitlines():
        s = ln.strip()
        if not s:
            continue
        toks = s.split()
        if not all(_looks_numeric(t) for t in toks):
            continue  # header / label line anywhere in the file
        vals = [safe_float(t) for t in toks]
        if len(vals) >= 2:
            rows.append(vals)
    if not rows:
        return np.zeros((0, 0))
    n = max(len(r) for r in rows)
    arr = np.full((len(rows), n), np.nan)
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r
    return arr


def read_activity_table(path: Path) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
    """Header-aware reader for ACT_PHn.OUT.

    Returns ``(strain, series)`` where ``series`` is a list of
    ``(label, values)`` for the *deformation-mode* activity columns only --
    those whose header name starts with ``MODE``.

    When twinning is active VPSC also writes bookkeeping columns, which are
    excluded:
        AVACS  -> average number of active systems, not an activity
        PRITW / SECTW -> primary / secondary twin volume fractions
        TWFRm / EFFRm -> twin reorientation fractions for twin mode m

    Falls back to positional parsing (col0 = strain, the rest = modes) for
    headerless / legacy files so FCC outputs keep working unchanged.
    """
    header = read_table_header(path)
    arr = read_numeric_table_no_header(path)
    if arr.size == 0 or arr.shape[1] < 2:
        return np.empty(0), []
    strain = arr[:, 0].astype(float)

    series: List[Tuple[str, np.ndarray]] = []
    if header:
        up = [h.upper() for h in header]
        for j, name in enumerate(up):
            if j >= arr.shape[1] or name == "STRAIN":
                continue
            if name.startswith("MODE"):
                tail = name[4:]
                label = f"Mode {tail}" if tail.isdigit() else header[j]
                series.append((label, arr[:, j].astype(float)))
    if not series:
        # Headerless or non-standard layout: treat each column after the first
        # as an activity (legacy behaviour).
        for j in range(1, arr.shape[1]):
            series.append((f"sys {j}", arr[:, j].astype(float)))
    return strain, series


def read_evm_svm(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Evm, Svm) from STR_STR.OUT, header-aware.

    Standard VPSC layout: column 1 = Evm (von Mises strain),
    column 2 = Svm (von Mises stress).  The header is used when present so the
    correct columns are chosen even though STR_STR.OUT has ~20 columns.
    """
    arr = read_numeric_table_no_header(path)
    if arr.size == 0 or arr.shape[1] < 2:
        return np.empty(0), np.empty(0)
    ix, iy = 0, 1
    header = read_table_header(path)
    if header:
        up = [h.upper() for h in header]
        if "EVM" in up and up.index("EVM") < arr.shape[1]:
            ix = up.index("EVM")
        if "SVM" in up and up.index("SVM") < arr.shape[1]:
            iy = up.index("SVM")
    return arr[:, ix].astype(float), arr[:, iy].astype(float)


def compact_numeric_rows(arr: np.ndarray, min_cols: int = 2) -> np.ndarray:
    """Keep rows with enough finite leading columns."""
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < min_cols:
        return np.zeros((0, 0))
    mask = np.all(np.isfinite(arr[:, :min_cols]), axis=1)
    return arr[mask]


def _robust_inlier_mask(values: np.ndarray, n_mad: float = 8.0) -> np.ndarray:
    """Boolean mask of points within ``n_mad`` median-absolute-deviations.

    Discards stray header/tensor values that survive section detection. The
    threshold is statistical (MAD-based) rather than an absolute magnitude, so
    it works for any unit system (MPa or GPa).
    """
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    if int(finite.sum()) < 4:
        return finite
    med = float(np.median(v[finite]))
    mad = float(np.median(np.abs(v[finite] - med)))
    if mad <= 0:
        return finite
    keep = finite.copy()
    keep[finite] = np.abs(v[finite] - med) <= n_mad * 1.4826 * mad
    return keep


def extract_lankford_curve(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract angle, R-value and optional Young's modulus from LANKFORD.OUT.

    Column layout: ANG, YOUNG, LANKF, D11, D22, D33, D12, S11, S12. The LANKF
    column is used for the R-value; the parser falls back for nonstandard files.
    """
    a = compact_numeric_rows(arr, min_cols=2)
    if a.size == 0:
        return np.empty(0), np.empty(0), None
    # Standard VPSC layout: col0=angle, col1=Young, col2=R.
    angle = a[:, 0].astype(float)
    if a.shape[1] >= 3:
        r = a[:, 2].astype(float)
        young = a[:, 1].astype(float)
    else:
        r = a[:, 1].astype(float)
        young = None
    mask = np.isfinite(angle) & np.isfinite(r) & (angle >= -1e-8) & (angle <= 180.0 + 1e-8)
    # R-values are normally of order 0--5.  Keep a generous upper bound to
    # discard accidental Young-modulus/header rows but allow highly anisotropic cases.
    mask &= (r > -20.0) & (r < 100.0)
    angle, r = angle[mask], r[mask]
    young = young[mask] if young is not None else None
    if angle.size == 0:
        return np.empty(0), np.empty(0), None
    order = np.argsort(angle)
    angle, r = angle[order], r[order]
    young = young[order] if young is not None else None
    # Average duplicated probe angles if present.
    uniq = np.unique(np.round(angle, 10))
    if uniq.size != angle.size:
        aa, rr, yy = [], [], []
        for u in uniq:
            m = np.isclose(angle, u, atol=1e-9)
            aa.append(float(np.median(angle[m])))
            rr.append(float(np.median(r[m])))
            if young is not None:
                yy.append(float(np.median(young[m])))
        angle = np.asarray(aa)
        r = np.asarray(rr)
        young = np.asarray(yy) if young is not None else None
    return angle, r, young


def read_lankford_output(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Read a standard VPSC ``LANKFORD.OUT`` file safely.

    Only the first ``ANG / YOUNG / LANKF`` block is used.  Later tensor blocks
    containing strain-rate or stress components are deliberately ignored, because
    treating every numeric row as curve data creates spurious points such as a
    sudden drop of Young's modulus to zero.

    Accepted layouts:
      1. ``angle  young  lankford`` on one line;
      2. ``angle  young`` followed by one line containing ``lankford``.
    """
    text = read_text(path)
    rows: List[Tuple[float, float, float]] = []
    pending: Optional[Tuple[float, float]] = None
    in_block = False
    saw_header = False

    def _is_tensor_or_next_section(line: str) -> bool:
        low = line.lower()
        return any(tok in low for tok in (
            "d(1", "d11", "dbar", "s(1", "s11", "sbar",
            "strain rate", "cauchy", "stress component", "tensor",
        ))

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if "ang" in low and "young" in low and ("lank" in low or "r-value" in low or "r value" in low):
            in_block = True
            saw_header = True
            pending = None
            continue
        if _is_tensor_or_next_section(line):
            if in_block or rows:
                break
        if not in_block and not saw_header:
            vals0 = numeric_values(line)
            if vals0 and len(vals0) >= 2 and -1.0e-8 <= vals0[0] <= 180.0 + 1.0e-8:
                in_block = True
            else:
                continue
        elif not in_block:
            continue

        vals = numeric_values(line)
        if not vals:
            continue

        if pending is not None:
            if len(vals) == 1:
                ang, young = pending
                r_value = vals[0]
                if rows and ang < rows[-1][0] - 1.0e-8:
                    break
                rows.append((ang, young, r_value))
                pending = None
                continue
            pending = None

        if len(vals) >= 3:
            ang, young, r_value = vals[0], vals[1], vals[2]
            if rows and ang < rows[-1][0] - 1.0e-8:
                break
            rows.append((ang, young, r_value))
        elif len(vals) == 2:
            ang, young = vals[0], vals[1]
            if rows and ang < rows[-1][0] - 1.0e-8:
                break
            pending = (ang, young)

    if not rows:
        return extract_lankford_curve(read_numeric_table(path))

    arr = np.asarray(rows, dtype=float)
    angle, young, r_value = arr[:, 0], arr[:, 1], arr[:, 2]
    # Unit-agnostic sanity filtering: keep finite rows in the angular range with
    # positive Young's modulus, then drop statistical outliers (stray header or
    # tensor values).  No absolute GPa/MPa magnitude limit is assumed.
    # NOTE: negative Lankford R-values are physically valid (common for textured
    # HCP at low angles, e.g. Mg compression), so they must NOT be filtered out.
    base = (
        np.isfinite(angle) & np.isfinite(young) & np.isfinite(r_value)
        & (angle >= -1.0e-8) & (angle <= 180.0 + 1.0e-8)
        & (young > 0.0)
        & (r_value > -50.0)
    )
    angle, young, r_value = angle[base], young[base], r_value[base]
    if angle.size:
        inl = _robust_inlier_mask(young) & _robust_inlier_mask(r_value)
        angle, young, r_value = angle[inl], young[inl], r_value[inl]
    if angle.size == 0:
        LOG.warning("read_lankford_output: no rows survived filtering in %s", path)
        return np.empty(0), np.empty(0), None

    order = np.argsort(angle)
    angle, young, r_value = angle[order], young[order], r_value[order]
    uniq = np.unique(np.round(angle, 10))
    if uniq.size != angle.size:
        aa: List[float] = []
        yy: List[float] = []
        rr: List[float] = []
        for u in uniq:
            m = np.isclose(angle, u, atol=1.0e-9)
            aa.append(float(np.median(angle[m])))
            yy.append(float(np.median(young[m])))
            rr.append(float(np.median(r_value[m])))
        angle = np.asarray(aa)
        young = np.asarray(yy)
        r_value = np.asarray(rr)
    return angle, r_value, young


def extract_pcys_curve(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract a clean closed PCYS stress locus and optional rate vectors.

    VPSC PCYS output can contain probe/origin rows or radial search points.
    The plotting curve should use the outer stress envelope only.  This routine
    removes near-origin rows, keeps the largest radius in each angular sector,
    sorts by polar angle, and closes the surface with the first boundary point.
    """
    a = compact_numeric_rows(arr, min_cols=2)
    if a.size == 0:
        return np.empty(0), np.empty(0), None, None

    x = a[:, 0].astype(float)
    y = a[:, 1].astype(float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    a = a[finite]
    if x.size == 0:
        return np.empty(0), np.empty(0), None, None

    norm = np.hypot(x, y)
    max_norm = float(np.nanmax(norm)) if norm.size else 0.0
    if max_norm <= 0:
        return np.empty(0), np.empty(0), None, None

    # Remove initial zero/probe rows and small radial points that create a
    # spurious chord from the origin to the surface.
    keep = norm > max(1e-9, 0.02 * max_norm)
    x, y, norm, a = x[keep], y[keep], norm[keep], a[keep]
    if x.size == 0:
        return np.empty(0), np.empty(0), None, None

    rates = a[:, 2:4].astype(float) if a.shape[1] >= 4 else None
    angle = np.mod(np.arctan2(y, x), 2.0 * math.pi)

    # If multiple rows exist along the same loading direction, keep the outer
    # point.  This is robust for both direct boundary files and radial probe
    # files.
    nbins = max(90, min(720, int(len(angle) * 2)))
    bins = np.floor(angle / (2.0 * math.pi) * nbins).astype(int)
    chosen: List[int] = []
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        if idx.size == 0:
            continue
        chosen.append(int(idx[np.argmax(norm[idx])]))
    chosen_arr = np.asarray(chosen, dtype=int)
    x, y, angle = x[chosen_arr], y[chosen_arr], angle[chosen_arr]
    rates = rates[chosen_arr] if rates is not None else None

    order = np.argsort(angle)
    pts = np.column_stack([x[order], y[order]])
    rates = rates[order] if rates is not None else None

    # Remove immediate duplicates after angular sorting.
    if len(pts) > 1:
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep2 = np.r_[True, d > max_norm * 1e-8]
        pts = pts[keep2]
        rates = rates[keep2] if rates is not None else None

    if len(pts) >= 3 and np.linalg.norm(pts[0] - pts[-1]) > max_norm * 1e-6:
        pts = np.vstack([pts, pts[0]])
        rates = np.vstack([rates, rates[0]]) if rates is not None else None

    rx = rates[:, 0] if rates is not None else None
    ry = rates[:, 1] if rates is not None else None
    return pts[:, 0], pts[:, 1], rx, ry



def split_monotonic_segments(x: np.ndarray, y: np.ndarray, *others: np.ndarray) -> List[Tuple[np.ndarray, ...]]:
    """Split a curve whenever the x-axis restarts or contains non-finite values.

    Several VPSC outputs append multiple probe blocks into the same file.
    Plotting them as a single polyline creates spurious long chords. This
    helper splits the arrays into monotonic segments and returns the non-empty
    pieces.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    extras = [np.asarray(v) for v in others]
    if x.size == 0 or y.size == 0:
        return []
    finite = np.isfinite(x) & np.isfinite(y)
    for arr in extras:
        if arr.ndim == 1 and arr.size == x.size:
            finite &= np.isfinite(arr)
    xf = x[finite]
    yf = y[finite]
    extras_f = []
    for arr in extras:
        if arr.ndim >= 1 and arr.shape[0] == x.shape[0]:
            extras_f.append(arr[finite])
        else:
            extras_f.append(arr)
    if xf.size == 0:
        return []
    cuts = [0]
    for i in range(1, xf.size):
        if xf[i] < xf[i - 1] - 1.0e-12:
            cuts.append(i)
    cuts.append(xf.size)
    segs: List[Tuple[np.ndarray, ...]] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 2:
            continue
        item: List[np.ndarray] = [xf[a:b], yf[a:b]]
        for arr in extras_f:
            if arr.ndim >= 1 and arr.shape[0] == xf.shape[0]:
                item.append(arr[a:b])
            else:
                item.append(arr)
        segs.append(tuple(item))
    return segs or [(xf, yf, *extras_f)]


def preferred_segment(segments: List[Tuple[np.ndarray, ...]]) -> Optional[Tuple[np.ndarray, ...]]:
    """Choose the most representative monotonic segment."""
    if not segments:
        return None
    def _key(seg: Tuple[np.ndarray, ...]) -> Tuple[float, int]:
        xs = np.asarray(seg[0], dtype=float)
        span = float(np.nanmax(xs) - np.nanmin(xs)) if xs.size else -1.0
        return span, int(xs.size)
    return max(segments, key=_key)


def prepare_result_figure(fig: Figure) -> None:
    """Reset the results figure with stable margins for GUI display and export."""
    fig.clear()
    fig.patch.set_facecolor("white")
    try:
        fig.set_layout_engine(None)
    except Exception:
        pass
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.92)


def find_output_files(run_dir: Path) -> Dict[str, Path]:
    """Find VPSC outputs case-insensitively and choose the most useful file."""
    out: Dict[str, Path] = {}
    if not run_dir or not run_dir.exists():
        return out
    files = [p for p in run_dir.iterdir() if p.is_file()]
    by_upper = [(p.name.upper(), p) for p in files]

    def first_exact(name: str) -> Optional[Path]:
        nameu = name.upper()
        matches = [p for u, p in by_upper if u == nameu]
        if matches:
            return sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        return None

    def first_prefix(prefix: str, suffix: str = ".OUT") -> Optional[Path]:
        pre = prefix.upper()
        suf = suffix.upper()
        matches = [p for u, p in by_upper if u.startswith(pre) and u.endswith(suf)]
        if matches:
            return sorted(matches, key=lambda x: (x.stat().st_mtime, x.name),
                          reverse=True)[0]
        return None

    mapping = {
        "STR_STR": first_exact("STR_STR.OUT"),
        "ACT": first_prefix("ACT_PH"),
        "R": first_exact("LANKFORD.OUT"),
        "PCYS": first_exact("PCYS.OUT"),
        "TEX": first_prefix("TEX_PH"),
        "RUN_LOG": first_exact("RUN_LOG.OUT"),
    }
    return {k: v for k, v in mapping.items() if v is not None}


# =============================================================================
# Project state and run directory preparation
# =============================================================================

@dataclass
class ProjectState:
    base_dir: Path = field(default_factory=Path.cwd)
    vpsc_in: Path = field(default_factory=lambda: Path("vpsc8.in"))
    executable: Path = field(default_factory=lambda: Path(""))
    run_root: Path = field(default_factory=lambda: Path("vpsc_runs"))
    last_run_dir: Path = field(default_factory=lambda: Path(""))
    backend: str = "Fortran-compatible"


def prepare_run_dir(state: ProjectState, info: VPSCInInfo) -> Path:
    root = path_rel(state.base_dir, state.run_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / time.strftime("vpsc_case_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    src_in = path_rel(state.base_dir, state.vpsc_in)
    if src_in.exists():
        txt = read_text(src_in)
        # Write both common capitalisations to handle case-sensitive Fortran builds
        for name in ["vpsc8.in", "VPSC8.IN", "VPSC8.in"]:
            write_text(run_dir / name, txt)

    deps: List[str] = []
    for ph in info.phases:
        deps.extend([ph.texture_file, ph.crystal_file, ph.shape_file, ph.diffraction_file])
    deps.extend(info.process_files)
    for d in deps:
        if not d or d.lower() == "dummy":
            continue
        p = path_rel(state.base_dir, d)
        if p.exists() and p.is_file():
            shutil.copy2(p, run_dir / p.name)

    (run_dir / "app_project.json").write_text(json.dumps({
        "base_dir": str(state.base_dir),
        "vpsc_in": str(state.vpsc_in),
        "executable": str(state.executable),
        "backend": state.backend,
    }, indent=2), encoding="utf-8")
    return run_dir


# =============================================================================
# Preferences
# =============================================================================

def load_prefs() -> Dict[str, Any]:
    try:
        return json.loads(PREF_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_prefs(prefs: Dict[str, Any]) -> None:
    try:
        PREF_PATH.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    except OSError as e:
        LOG.warning("Could not save prefs: %s", e)

# =============================================================================
# Theory notes shown in the GUI
# =============================================================================

VPSC_THEORY_NOTES = r"""
VPSC theory notes - core formulas from Tome and Lebensohn VPSC book
===================================================================

This page is intentionally written with ASCII symbols only, so it can be
shown safely in the Tk text widget on Windows without missing-glyph boxes.
Notation: repeated indices are summed.  A:B = A_ij B_ij.
For fourth-order tensors, (C:A)_ij = C_ijkl A_kl and
(A::B)_ijmn = A_ijkl B_klmn.  <q> denotes the volume average over grains.

0. Polycrystal averaging and basic conventions
----------------------------------------------
The aggregate is represented by weighted grains/orientations g:

    <q> = sum_g w_g q^g,                       sum_g w_g = 1
    eps_dot_bar = <eps_dot^g>,                 sigma_bar = <sigma^g>

The Cauchy stress is decomposed into hydrostatic and deviatoric parts:

    sigma'_ij = sigma_ij - (sigma_kk/3) delta_ij
    tr(sigma') = 0

In the VPSC viscoplastic formulation the plastic strain rate is incompressible,
therefore only the 5 independent deviatoric stress/strain-rate components enter
the nonlinear crystal-plasticity equations.

1. Governing equations: strain compatibility and stress equilibrium
-------------------------------------------------------------------
Small-strain kinematics:

    eps_ij(x) = 1/2 [ u_i,j(x) + u_j,i(x) ]

Compatibility condition:

    eps_ij,kl + eps_kl,ij - eps_ik,jl - eps_jl,ik = 0

Quasi-static stress equilibrium without body force:

    sigma_ij,j(x) = 0

Cauchy stress symmetry from angular momentum balance:

    sigma_ij = sigma_ji

Linear elasticity and linear viscosity:

    sigma_ij = C_ijkl eps_kl
    eps_ij   = S_ijkl sigma_kl,        S = C^{-1}
    eps_dot_ij = M_ijkl sigma_kl

Elastic energy density:

    U(eps) = 1/2 C_ijkl eps_ij eps_kl
    sigma_ij = dU/deps_ij

2. Tensor rotation, Euler angles, PF and IPF conventions
--------------------------------------------------------
For a rotation matrix A mapping crystal components to sample components:

    v_sample_i = A_ij v_crystal_j
    T_sample_ij = A_ip A_jq T_crystal_pq
    C_sample_ijkl = A_ip A_jq A_kr A_ls C_crystal_pqrs

This App stores the standard Bunge matrix g such that

    x_crystal = g x_sample

Therefore:

    v_sample  = g^T v_crystal      (pole figure, crystal pole to sample)
    v_crystal = g   v_sample       (inverse pole figure, sample axis to crystal)

Bunge Euler angles (phi1, Phi, phi2).  Let

    c1=cos(phi1), s1=sin(phi1), c=cos(Phi), s=sin(Phi),
    c2=cos(phi2), s2=sin(phi2).

The crystal-to-sample matrix A is

    A11 =  c1*c2 - s1*c*s2      A12 = -c1*s2 - s1*c*c2      A13 =  s1*s
    A21 =  s1*c2 + c1*c*s2      A22 = -s1*s2 + c1*c*c2      A23 = -c1*s
    A31 =  s*s2                 A32 =  s*c2                 A33 =  c

and the App convention is

    g = A^T.

Pole-figure projection for a unit vector with polar angle theta:

    equal-area Lambert:     r = sqrt(2) sin(theta/2)
    stereographic:          r = tan(theta/2)       (up to a display scale)

3. Voigt notation and VPSC b-basis
----------------------------------
The GUI uses the VPSC order

    11, 22, 33, 23, 13, 12

For engineering-strain Voigt vectors:

    eps_V = [eps11, eps22, eps33, 2eps23, 2eps13, 2eps12]
    sig_V = [sig11, sig22, sig33, sig23, sig13, sig12]

The orthonormal symmetric b-basis maps a symmetric tensor a_ij to six
components:

    a1 = (a22 - a11)/sqrt(2)
    a2 = (2a33 - a22 - a11)/sqrt(6)
    a3 = sqrt(2) a23
    a4 = sqrt(2) a13
    a5 = sqrt(2) a12
    a6 = (a11 + a22 + a33)/sqrt(3)

For plastic incompressibility and deviatoric stress, VPSC uses the first five
components.  The sixth component is the hydrostatic part.

4. Slip and twinning kinematics
-------------------------------
For slip/twin system s with unit shear direction b^s and plane normal n^s:

    m^s_ij = 1/2 ( b^s_i n^s_j + b^s_j n^s_i )       Schmid tensor
    q^s_ij = 1/2 ( b^s_i n^s_j - b^s_j n^s_i )       spin tensor

Because b^s . n^s = 0,

    tr(m^s) = 0,              tau^s = m^s : sigma = m^s : sigma'

The plastic velocity gradient, strain rate and spin are

    Lp^g_ij        = sum_s gamma_dot^{s,g} b^s_i n^s_j
    eps_dot_p^g_ij = sum_s gamma_dot^{s,g} m^s_ij
    omega_dot_p^g_ij = sum_s gamma_dot^{s,g} q^s_ij

The accumulated shear in grain g is

    Gamma^g = sum_s |gamma^{s,g}|,        Delta Gamma^g = sum_s |Delta gamma^{s,g}|

For twinning, the characteristic twin shear is S_t, and a shear gamma^t gives a
local twin volume fraction

    v^{t,g} = gamma^{t,g} / S_t.

5. Rate-sensitive slip/twin flow law
------------------------------------
The standard VPSC power law for slip is

    gamma_dot^s = gamma_dot0^s |tau^s/tau_c^s|^{n_s} sign(tau^s)

where tau_c^s is the CRSS and n_s is the stress exponent.  If the usual
rate-sensitivity m_rate is used, then n_s = 1/m_rate.

For twinning, the law is polar.  In the usual VPSC implementation:

    gamma_dot^t = gamma_dot0^t (tau^t/tau_c^t)^{n_t},      tau^t > 0
    gamma_dot^t = 0,                                      tau^t <= 0

The grain strain rate is therefore

    eps_dot^g = gamma_dot0 sum_s m^s
                ( |m^s:sigma'^g| / tau_c^s )^{n_s}
                sign(m^s:sigma'^g) + eps_dot_tr^g

where eps_dot_tr^g is an optional transformation strain rate, for example
irradiation growth.

6. Schmid factor, Taylor factor and plastic work
------------------------------------------------
For an arbitrary stress state, the generalized Schmid factor is

    SF^s = tau_res^s / ||sigma|| = (m^s_ij sigma_ij) / sqrt(sigma_ij sigma_ij)

For uniaxial loading this reduces to the usual cos(lambda) cos(phi) form.

Hill's lemma for a representative volume is

    sigma_bar : eps_dot_bar = < sigma^g : eps_dot^g >

For single-CRSS Taylor analysis, define

    Gamma_dot = < sum_s |gamma_dot^{s,g}| >
    Mhat ||eps_dot_bar|| = Gamma_dot
    Mhat = (sigma_bar : eps_dot_bar) / (tau_CRSS ||eps_dot_bar||)

For uniaxial tension under the Von-Mises strain-rate norm, the conventional
Taylor factor satisfies approximately

    M = sigma_33 / tau_CRSS.

7. Thermoelastic self-consistent model
--------------------------------------
Single-grain thermoelastic law:

    sigma^g = C^g : (eps^g - alpha^g DeltaT)
    eps^g   = (C^g)^{-1} : sigma^g + alpha^g DeltaT

Effective medium law:

    sigma_bar = C_bar : (eps_bar - alpha_bar DeltaT)
    eps_bar   = C_bar^{-1} : sigma_bar + alpha_bar DeltaT

Elastic inclusion interaction:

    eps_tilde^g = S^g : eps_star^g
    sigma_tilde^g = C_bar : (eps_tilde^g - eps_star^g)
    sigma_tilde^g = - Ctilde^g : eps_tilde^g
    Ctilde^g = C_bar : (I - S^g) : (S^g)^{-1}

Strain localization:

    eps^g = A^g : eps_bar + D^g DeltaT
    A^g = (C^g + Ctilde^g)^{-1} : (C_bar + Ctilde^g)
    D^g = (C^g + Ctilde^g)^{-1} : (C^g:alpha^g - C_bar:alpha_bar)

Stress localization:

    sigma^g = B^g : sigma_bar + E^g DeltaT
    B^g = C^g : A^g : C_bar^{-1}

Self-consistent stiffness and thermal expansion:

    C_bar = < C^g : A^g > : < A^g >^{-1}
    C_bar : alpha_bar = < Ay^g >^{-1} : < Ay^g : C^g : alpha^g >
    Ay^g = (C_bar + Ctilde^g) : (C^g + Ctilde^g)^{-1}

If all ellipsoids have the same shape and orientation, the simplified forms are

    C_bar = < Ay^g : C^g >
    C_bar : alpha_bar = < Ay^g : C^g : alpha^g >

8. Local linearized viscoplastic law
------------------------------------
VPSC replaces the nonlinear grain response by a pseudolinear law during each
self-consistent iteration:

    eps_dot^g = M^g : sigma'^g + eps_dot_o^g + eps_dot_tr^g
    sigma'^g  = L^g : (eps_dot^g - eps_dot_o^g - eps_dot_tr^g)
    L^g = (M^g)^{-1}

At the aggregate level:

    eps_dot_bar = M_bar : sigma'_bar + eps_dot_o_bar
    sigma'_bar  = L_bar : (eps_dot_bar - eps_dot_o_bar)
    L_bar = M_bar^{-1}

The back-extrapolated term eps_dot_o depends on the chosen linearization.

9. Viscoplastic inclusion, interaction and localization
-------------------------------------------------------
The equivalent inclusion eigenstrain-rate is

    eps_dot_star^g = (M^g - M_bar):sigma'^g
                     + (eps_dot_o^g - eps_dot_o_bar)
                     + eps_dot_tr^g

The viscoplastic Eshelby tensor gives

    eps_dot_tilde^g = S^g : eps_dot_star^g
    omega_dot_tilde^g = Pi^g : eps_dot_star^g
                      = Pi^g : (S^g)^{-1} : eps_dot_tilde^g

The interaction equation is

    eps_dot_tilde^g = - Mtilde^g : sigma_tilde'^g
    eps_dot^g - eps_dot_bar = - Mtilde^g : (sigma'^g - sigma'_bar)

with

    Mtilde^g = (I - S^g)^{-1} : S^g : M_bar

The stress localization equation is

    sigma'^g = B^g : sigma'_bar + E^g

where

    B^g = (M^g + Mtilde^g)^{-1} : (M_bar + Mtilde^g)
    E^g = (M^g + Mtilde^g)^{-1}
          : (eps_dot_o_bar - eps_dot_o^g - eps_dot_tr^g)

10. VPSC self-consistent equations
----------------------------------
The effective compliance and back-extrapolated strain rate are obtained by fixed
point iteration:

    M_bar = < M^g : B^g > : < B^g >^{-1}

    eps_dot_o_bar = < By^g >^{-1}
                    : < By^g : (eps_dot_o^g + eps_dot_tr^g) >

where

    By^g = (M_bar + Mtilde^g) : (M^g + Mtilde^g)^{-1}

If all ellipsoids have the same shape and orientation:

    M_bar = < M^g : B^g >
    eps_dot_o_bar = < By^g : (eps_dot_o^g + eps_dot_tr^g) >

After convergence of M_bar and eps_dot_o_bar, the grain stresses are corrected by
solving the nonlinear five-component deviatoric system

    gamma_dot0 sum_s m^s ( |m^s:sigma'^g| / tau_c^s )^n
                sign(m^s:sigma'^g)
    - eps_dot_bar
    = - Mtilde^g : (sigma'^g - sigma'_bar)

The Taylor model is the limiting case Mtilde^g = 0.  The Sachs model is the
uniform-stress lower-bound limit.

11. Linearization choices
-------------------------
Secant linearization passes through the origin of the stress/strain-rate relation:

    eps_dot^g ~= M_sec^g : sigma'^g,          eps_dot_o^g = 0

Tangent linearization uses the local derivative at the current grain stress:

    M_tan^g = d eps_dot^g / d sigma'^g
    eps_dot_o^g = eps_dot^g - M_tan^g : sigma'^g

Affine linearization has the same algebraic form as the tangent one but is used
inside an outer iteration that corrects grain stresses and then rebuilds
M_aff^g and eps_dot_o,aff^g.

Second-order linearization uses stress second moments inside each grain.  The
linearized effective stress potential is written as

    U = 1/2 M_bar :: (sigma'_bar tensor sigma'_bar)
        + eps_dot_o_bar : sigma'_bar + 1/2 G

and derivatives of U with respect to grain moduli provide intragranular stress
fluctuations.

12. Microstructure evolution: orientation and grain shape
--------------------------------------------------------
The deformation gradient is updated incrementally by

    F(t + Dt) = (I + L Dt) : F(t)

For each grain:

    L^g = eps_dot^g + omega_dot^g
    omega_dot^g = omega_dot_bar + omega_dot_tilde^g

The crystal orientation matrix R evolves as

    R_dot = [ omega_dot_bar + omega_dot_tilde^g
              - R : omega_dot_p^g : R^T ] : R
          = Omega_dot^g : R

and the exponential update is

    R(t + Dt) = exp(Omega_dot^g Dt) : R(t)

If Omega = Omega_dot Dt is skew-symmetric,

    omega = sqrt(Omega12^2 + Omega13^2 + Omega23^2)
    n_hat = ( -Omega23, Omega13, -Omega12 ) / omega

and exp(Omega) is evaluated by Rodrigues' formula.

Grain shape is updated from the grain deformation gradient.  If the initial shape
is spherical, the deformed ellipsoid is

    [ (F^g : F^{gT})^{-1} ]_jk x_j x_k = 1

The eigenvectors of F^g F^{gT} give the ellipsoid axes directions, and the square
roots of its eigenvalues give the axis lengths.

13. Extended Voce hardening
---------------------------
For system s the extended Voce threshold is

    tau_hat^s(Gamma) = tau0^s
        + (tau1^s + theta1^s Gamma) [1 - exp(-a^s Gamma)]

where

    Gamma = sum_s |gamma^s|,       a^s = |theta0^s / tau1^s|

The hardening rate is

    d tau_hat^s/dGamma = theta1^s [1 - exp(-a^s Gamma)]
                         + a^s (tau1^s + theta1^s Gamma) exp(-a^s Gamma)

Self and latent hardening are introduced through h_ss':

    Delta tau^s = (d tau_hat^s/dGamma) sum_s' h_ss' |Delta gamma^s'|

If h_ss' = 1 for all systems, then

    Delta tau^s = (d tau_hat^s/dGamma) Delta Gamma

VPSC input rows for Voce modes normally contain

    nrsx
    tau0x  tau1x  thet0  thet1  hpfac
    hlatex(1,im), im=1,nmodes

14. Mechanical Threshold Stress (MTS) hardening
-----------------------------------------------
The MTS slip resistance is written as the sum of athermal, intrinsic thermal and
evolving thermal components:

    tau^s = tau_a^s + tau_i^s + tau_e^s
          = tau_a^s
            + (mu/mu0) S_i(eps_dot,T) tau_i_hat^s
            + (mu/mu0) S_e(eps_dot,T) tau_e_hat^s

A common Kocks-type thermal activation scaling is

    S_x(eps_dot,T) = [ 1 - { kB T/(mu b^3 g0_x)
                         ln(eps0_x/eps_dot) }^{1/q_x} ]^{1/p_x}

with x = i or e, provided the quantity inside the outer brackets is in the
physical interval [0,1].  The evolving structure term follows

    d tau_e_hat/dGamma = theta0 [ 1 - tau_e_hat/tau_eS_hat ]^kappa

and the saturation threshold is

    tau_eS_hat = tau_eS0_hat
                 ( eps_dot / eps0_eS )^{ kB T/(mu b^3 g0_eS) }

Adiabatic heating from plastic work is commonly estimated as

    DeltaT = xi [ sum_g sum_s tau^{s,g} Delta gamma^{s,g} w_g ] / (rho Cp)

where xi is the Taylor-Quinney heat fraction, rho is density and Cp is specific
heat.

15. Dislocation-density hardening (DD)
--------------------------------------
The DD model writes the CRSS as

    tau^s = tau0^s + tau_for^s + tau_sub^s + tau_HP^s

with initial/friction resistance

    tau0^s(T) = A0^s + A1^s exp(-T/A2^s)

and Hall-Petch contribution

    tau_HP^s = mu^s H^s sqrt(b^s/d_g)

The forest density evolves by storage minus dynamic recovery:

    d rho_for^s/d gamma^s
        = k1^s sqrt(rho_for^s) - k2^s(eps_dot,T) rho_for^s

The corresponding saturation value is

    rho_sat^s = (k1^s/k2^s)^2

Forest resistance is commonly written as

    tau_for^s = chi_for mu^s b^s sqrt( sum_s' rho_for^{s'} )

The substructure resistance is

    tau_sub^s = chi_sub mu^s b^s sqrt(rho_sub)
                log[ 1/(b^s sqrt(rho_sub)) ]

The substructure density increment is coupled to the recovered forest density:

    d rho_sub = sum_s q^s k2^s rho_for^s d gamma^s
    q^s = F^s(T,eps_dot) b^s sqrt(rho_sub)

All length, density and stress units must be consistent with the single-crystal
.sx file comments.

16. DD_REV strain-path-change hardening
---------------------------------------
DD_REV separates forest, reversible and debris dislocation populations.  The
forest/debris part gives a deformation resistance tau_d^s.  Reversible
dislocations carry a back-stress effect during shear reversal.  A compact form is

    tau_eff^s = tau_d^s + Delta tau_B^s
    Delta tau_B^s = - tau_d^s f_B^s (rho_rev,opposite^s / rho^s)^{q_B}

when the current shear direction is opposite to the direction that stored the
reversible population.  As reverse shear remobilizes and annihilates those
reversible dislocations, Delta tau_B^s gradually vanishes, representing the
Bauschinger effect.

17. Twinning reorientation: PTR, MC and VFT
-------------------------------------------
Twinning is treated as polar shear plus possible reorientation.  For a twin mode,
the accumulated twin fraction is

    V_acc,mode = sum_g sum_t ( gamma^{t,g} / S_t ) w_g

The effective reoriented fraction in the PTR/MC schemes is

    V_eff,mode = sum_{reoriented grains g} w_g

The PTR threshold is

    V_th,mode = A_th1 + A_th2 V_eff,mode / V_acc,mode

A grain is reoriented if its predominant twin system exceeds this threshold and
the global effective twin fraction has not exceeded V_acc,mode.

MC twinning selects a twin variant with probability proportional to its accumulated
local twin volume fraction.

In VFT, parent and child orientations coexist and volume is transferred as

    Delta w^{t,g} = (Delta gamma^{t,g} / S_t) w_parent^g

from the parent grain to the child twin orientation.

18. Boundary conditions and mixed control
-----------------------------------------
At the aggregate level the VPSC state equation is

    d = eps_dot_bar - eps_dot_o_bar = M_bar : sigma'_bar

In mixed loading, each Voigt component is either prescribed as a stress component
or as a strain-rate component.  The STATE_6x6 / STATE_NxN routines rearrange
this linear system so the unknown stress and strain-rate components are solved
consistently.

For a process file with constant velocity gradient, the imposed macroscopic
velocity-gradient matrix L is decomposed as

    eps_dot_bar = 1/2 (L + L^T)
    omega_dot_bar = 1/2 (L - L^T)

19. PCYS, Lankford R-value and output interpretation
----------------------------------------------------
Polycrystal yield-surface probes use the same VPSC equations but prescribe a
stress direction or stress point and solve the compatible plastic strain-rate
direction.  Plastic power is

    P = sigma_bar : eps_dot_bar

Lankford coefficient for a tensile direction beta in the sheet plane is

    R_beta = eps_dot_width^p / eps_dot_thickness^p

For uniaxial sheet tension with tensile axis 1, width 2 and thickness 3:

    R = eps_dot_22^p / eps_dot_33^p

because both lateral plastic strain rates are usually negative.

Main output files:

    STR_STR.OUT                 macroscopic stress-strain history
    STR_STR_STATS.OUT           phase/grain stress-strain statistics
    ACT_PHn.OUT                 relative slip/twin activity by phase
    TEX_PHn.OUT                 texture after selected increments
    EL-TH-MOD.OUT               elastic/thermal effective moduli
    LANKFORD.OUT                angle, R-value and optional Young modulus
    PCYS.OUT                    polycrystal yield-surface probe data
    TWIN_*_STATS_PHn_MODEm.OUT  twin Schmid-factor and variant statistics

20. Practical reading of VPSC input rows
----------------------------------------
VPSC8.IN defines the global regime, phases, texture files, single-crystal files,
precision, interaction option and process list.

Typical interaction options are

    0  Full constraint / Taylor-like upper bound
    1  Affine / affine-like interaction
    2  Secant interaction
    3  n_eff interaction
    4  Tangent interaction
    5  Second-order interaction

The line

    iupdate: update orient, grain shape, hardening

controls whether crystallographic orientation, ellipsoid shape and CRSS/internal
variables are updated after each increment.

End of theory notes.
"""

# =============================================================================
# Tk helper widgets
# =============================================================================


class TextEditor(ttk.Frame):
    def __init__(self, master: Any, wrap: str = "none", height: int = 12) -> None:
        super().__init__(master)
        self.text = tk.Text(self, wrap=wrap, undo=True, height=height,
                            font=("Consolas", 10),
                            bg="#ffffff", fg="#0f172a", insertbackground="#0f172a")
        y = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        x = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=y.set, xscrollcommand=x.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        y.grid(row=0, column=1, sticky="ns")
        x.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def set(self, value: str) -> None:
        self.text.delete("1.0", "end")
        self.text.insert("1.0", value)

    def get(self) -> str:
        return self.text.get("1.0", "end-1c")



class ScrollableFrame(ttk.Frame):
    """A lightweight vertical scroll container used for long option panels.

    Tk/ttk widgets do not scroll automatically.  This wrapper keeps long option panels scrollable while leaving the drawing area visible.
    """
    def __init__(self, master: Any, width: int = 320) -> None:
        super().__init__(master)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                background="#f6f8fb", width=width)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        def _on_configure(_event: Any = None) -> None:
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            self.canvas.itemconfigure(self.window_id, width=self.canvas.winfo_width())

        self.inner.bind("<Configure>", _on_configure)
        self.canvas.bind("<Configure>", _on_configure)

        def _wheel(event: Any) -> None:
            delta = -1 * int(event.delta / 120) if getattr(event, "delta", 0) else 0
            if delta:
                self.canvas.yview_scroll(delta, "units")

        self.canvas.bind_all("<MouseWheel>", _wheel)

class MatrixEditor(ttk.Frame):
    def __init__(self, master: Any, rows: int, cols: int,
                 labels: Optional[List[str]] = None, width: int = 11) -> None:
        super().__init__(master)
        self.rows, self.cols = rows, cols
        self.vars: List[List[tk.StringVar]] = [
            [tk.StringVar(value="0") for _ in range(cols)] for _ in range(rows)
        ]
        if labels and len(labels) == cols:
            for j, lab in enumerate(labels):
                ttk.Label(self, text=lab, anchor="center").grid(
                    row=0, column=j + 1, sticky="ew", padx=1, pady=1
                )
        start = 1 if labels else 0
        for i in range(rows):
            if labels:
                ttk.Label(self, text=str(i + 1), width=3, anchor="center").grid(
                    row=i + start, column=0, padx=1, pady=1
                )
            for j in range(cols):
                ttk.Entry(self, textvariable=self.vars[i][j],
                          width=width, justify="center").grid(
                    row=i + start, column=j + 1 if labels else j,
                    padx=1, pady=1, sticky="ew"
                )
                self.grid_columnconfigure(j + 1 if labels else j, weight=1)

    def set_array(self, arr: np.ndarray) -> None:
        for i in range(self.rows):
            for j in range(self.cols):
                val = arr[i, j] if i < arr.shape[0] and j < arr.shape[1] else 0
                self.vars[i][j].set(fmt_num(val))

    def array(self, dtype=float) -> np.ndarray:
        a = np.zeros((self.rows, self.cols), dtype=dtype)
        for i in range(self.rows):
            for j in range(self.cols):
                a[i, j] = safe_float(self.vars[i][j].get())
        return a.astype(dtype)

    # Aliases matching TextEditor's set/get conventions (used by callers)
    def set(self, arr: np.ndarray) -> None:
        self.set_array(np.asarray(arr, dtype=float))

    def get(self, dtype=float) -> np.ndarray:
        return self.array(dtype=dtype)


class PlotCanvas(ttk.Frame):
    def __init__(self, master: Any, figsize: Tuple[float, float] = (8.6, 5.2)) -> None:
        super().__init__(master)
        self.fig = Figure(figsize=figsize, dpi=100, facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self._drag_artist: Optional[Any] = None
        self._drag_offset: Tuple[float, float] = (0.0, 0.0)
        self._drag_cids: List[int] = []
        self._last_click: Optional[Tuple[float, float, float, Any]] = None
        self._tk_double_binding_set: bool = False
        self._text_style_callback: Optional[Callable[[Any], None]] = None
        self._advanced_text_callback: Optional[Callable[[Any], None]] = None
        # In-place (on-canvas) text editor state.
        self._inplace_widgets: List[Any] = []
        self._inplace_artist: Optional[Any] = None
        self._inplace_redraw: Optional[Callable[[], None]] = None
        widget = self.canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def set_text_style_editor_callback(self, callback: Optional[Callable[[Any], None]]) -> None:
        """Register the dialog opener used when a text artist is double-clicked."""
        self._text_style_callback = callback

    def _text_artist_at_event(self, event: Any) -> Optional[Any]:
        """Return the top-most text artist under a mouse event.

        Matplotlib pick events can miss small legend labels or colorbar tick labels,
        especially after toolbar operations.  This fallback directly checks the
        rendered text bounding boxes, so double-click editing works for every
        visible text item in the figure.
        """
        if event is None or getattr(event, "x", None) is None or getattr(event, "y", None) is None:
            return None
        try:
            renderer = self.canvas.get_renderer()
        except Exception:
            try:
                self.canvas.draw()
                renderer = self.canvas.get_renderer()
            except Exception:
                renderer = None
        texts: List[Any] = []
        try:
            texts = list(self.fig.findobj(match=MplText))
        except Exception:
            texts = []
        # Search in reverse drawing order; legend / colorbar texts usually occur later.
        for artist in reversed(texts):
            try:
                if not artist.get_visible():
                    continue
                ok, _info = artist.contains(event)
                if ok:
                    return artist
            except Exception:
                pass
            if renderer is not None:
                try:
                    bb = artist.get_window_extent(renderer=renderer).expanded(1.25, 1.55)
                    if bb.contains(event.x, event.y):
                        return artist
                except Exception:
                    pass
        return None

    def _legend_at_event(self, event: Any) -> Optional[Any]:
        """Return the legend under a mouse event, if any."""
        if event is None or getattr(event, "x", None) is None or getattr(event, "y", None) is None:
            return None
        try:
            renderer = self.canvas.get_renderer()
        except Exception:
            try:
                self.canvas.draw()
                renderer = self.canvas.get_renderer()
            except Exception:
                renderer = None
        for ax in reversed(self.fig.axes):
            try:
                leg = ax.get_legend()
            except Exception:
                leg = None
            if leg is None or not leg.get_visible():
                continue
            try:
                frame = leg.get_frame()
                ok, _info = frame.contains(event)
                if ok:
                    return leg
            except Exception:
                pass
            if renderer is not None:
                try:
                    if leg.get_window_extent(renderer=renderer).expanded(1.08, 1.20).contains(event.x, event.y):
                        return leg
                except Exception:
                    pass
        return None

    def _is_user_double_click(self, event: Any) -> bool:
        """Backend-independent double-click detector.

        Some Tk/Matplotlib combinations used on Windows do not set
        ``event.dblclick`` reliably for embedded canvases.  This fallback treats
        two left-button clicks close in time and position as a double click so
        the text editor always opens.
        """
        try:
            if bool(getattr(event, "dblclick", False)):
                self._last_click = None
                return True
        except Exception:
            pass
        try:
            x = float(getattr(event, "x", -1.0))
            y = float(getattr(event, "y", -1.0))
            button = getattr(event, "button", 1)
        except Exception:
            return False
        now = time.monotonic()
        last = self._last_click
        self._last_click = (now, x, y, button)
        if last is None:
            return False
        t0, x0, y0, b0 = last
        try:
            same_button = (button == b0) or (str(button) == str(b0))
        except Exception:
            same_button = True
        return bool(same_button and (now - t0) <= 0.55 and abs(x - x0) <= 8.0 and abs(y - y0) <= 8.0)

    def set_advanced_text_editor_callback(self, callback: Optional[Callable[[Any], None]]) -> None:
        """Register the full property dialog opened from the in-place editor's 'More…'."""
        self._advanced_text_callback = callback

    def _open_text_or_legend_at_event(self, event: Any) -> bool:
        """Open an editor for the text/legend under a mouse event.

        Plain text artists open the lightweight *in-place* editor (an entry box
        placed directly on top of the text, plus a small quick-style strip), so
        the common case — retype, resize, recolour — needs no dialog at all.
        Legends keep the structured dialog because they manage multiple entries.
        """
        try:
            if self.toolbar.mode:
                return False
        except Exception:
            pass
        text_target = self._text_artist_at_event(event)
        if text_target is not None and hasattr(text_target, "get_text") \
                and hasattr(text_target, "set_text"):
            return self._begin_inplace_edit(text_target, event)
        target = text_target or self._legend_at_event(event)
        if target is None:
            # Empty area: a double-click here inserts a new annotation and opens
            # the in-place editor on it, so adding labels is as direct as editing.
            new_artist = self._add_annotation_at_event(event)
            if new_artist is not None:
                return self._begin_inplace_edit(new_artist, event, is_new=True)
            return False
        cb = getattr(self, "_text_style_callback", None)
        if cb is None:
            return False
        try:
            cb(target, self)
        except TypeError:
            try:
                cb(target)
            except Exception:
                return False
        except Exception:
            return False
        self._drag_artist = None
        return True

    # ------------------------------------------------------------------ #
    # In-place on-canvas text editor
    # ------------------------------------------------------------------ #
    def _axes_at_display(self, x: float, y: float) -> Optional[Any]:
        """Return the axes whose rendered area contains display point (x, y)."""
        try:
            renderer = self.canvas.get_renderer()
        except Exception:
            renderer = None
        for ax in reversed(self.fig.axes):
            try:
                if not ax.get_visible():
                    continue
                bb = ax.get_window_extent(renderer=renderer)
                if bb.contains(x, y):
                    return ax
            except Exception:
                pass
        return None

    def _add_annotation_at_event(self, event: Any) -> Optional[Any]:
        """Create a new, editable text annotation under a mouse event.

        Placed in the axes under the cursor (axes-fraction coordinates, so it
        stays put when data limits change) or in figure coordinates when the
        click is outside every axes.  The new artist is draggable and editable
        exactly like the figure's own labels.
        """
        x = getattr(event, "x", None)
        y = getattr(event, "y", None)
        if x is None or y is None:
            return None
        try:
            ax = self._axes_at_display(float(x), float(y))
        except Exception:
            ax = None
        try:
            if ax is not None:
                bb = ax.get_window_extent()
                fx = (float(x) - bb.x0) / max(bb.width, 1e-9)
                fy = (float(y) - bb.y0) / max(bb.height, 1e-9)
                fx = min(max(fx, 0.0), 1.0)
                fy = min(max(fy, 0.0), 1.0)
                artist = ax.text(fx, fy, "New text", transform=ax.transAxes,
                                 ha="center", va="center")
            else:
                w = float(self.fig.bbox.width) or 1.0
                h = float(self.fig.bbox.height) or 1.0
                artist = self.fig.text(min(max(float(x) / w, 0.0), 1.0),
                                       min(max(float(y) / h, 0.0), 1.0),
                                       "New text", ha="center", va="center")
            try:
                artist.set_picker(6)
            except Exception:
                pass
            self.canvas.draw()
            return artist
        except Exception:
            return None

    def _cancel_inplace_edit(self) -> None:
        for w in getattr(self, "_inplace_widgets", []):
            try:
                w.destroy()
            except Exception:
                pass
        self._inplace_widgets = []
        self._inplace_artist = None
        self._inplace_redraw = None

    def _artist_pixel_box(self, artist: Any) -> Optional[Tuple[int, int, int, int]]:
        """Window-extent of a text artist as integer (x, y, w, h) in Tk pixels."""
        try:
            renderer = self.canvas.get_renderer()
        except Exception:
            try:
                self.canvas.draw()
                renderer = self.canvas.get_renderer()
            except Exception:
                return None
        try:
            bb = artist.get_window_extent(renderer=renderer)
        except Exception:
            return None
        try:
            widget = self.canvas.get_tk_widget()
            height = float(widget.winfo_height())
        except Exception:
            height = float(self.fig.bbox.height)
        # Matplotlib y is bottom-up; Tk is top-down.
        x = int(round(bb.x0))
        y = int(round(height - bb.y1))
        w = max(40, int(round(bb.width)))
        h = max(16, int(round(bb.height)))
        return x, y, w, h

    def _begin_inplace_edit(self, artist: Any, event: Any, *, is_new: bool = False) -> bool:
        """Place a Tk entry directly over the text artist for immediate editing.

        When ``is_new`` is True the artist was just created by a double-click on
        empty canvas; cancelling (Esc) or committing an empty string removes it,
        so an accidental insert leaves no stray placeholder behind.
        """
        self._cancel_inplace_edit()
        widget = self.canvas.get_tk_widget()

        def _redraw() -> None:
            try:
                self.canvas.draw_idle()
            except Exception:
                pass

        self._inplace_artist = artist
        self._inplace_redraw = _redraw

        box = self._artist_pixel_box(artist)
        if box is None:
            return False
        x, y, w, h = box

        # Current properties.
        try:
            cur_text = str(artist.get_text() or "")
        except Exception:
            cur_text = ""
        try:
            cur_size = float(artist.get_fontsize())
        except Exception:
            cur_size = 10.0
        try:
            fam = artist.get_fontfamily()
            cur_family = str(fam[0]) if isinstance(fam, (list, tuple)) and fam else str(fam)
        except Exception:
            cur_family = "Arial"
        try:
            cur_color = mpl_colors.to_hex(mpl_colors.to_rgba(artist.get_color()))
        except Exception:
            cur_color = "#111827"
        try:
            cur_weight = str(artist.get_fontweight())
        except Exception:
            cur_weight = "normal"

        state = {"size": cur_size, "weight": cur_weight, "color": cur_color}

        var = tk.StringVar(value=cur_text)
        entry = tk.Entry(widget, textvariable=var, font=("TkDefaultFont", max(8, int(round(cur_size)))))
        entry.place(x=max(0, x - 2), y=max(0, y - 2), width=w + 16, height=h + 6)
        entry.icursor("end")
        entry.select_range(0, "end")
        entry.focus_set()

        def apply_text(*_a: Any) -> None:
            try:
                artist.set_text(var.get())
            except Exception:
                pass
            _redraw()

        def _remove_artist() -> None:
            try:
                artist.remove()
            except Exception:
                try:
                    artist.set_visible(False)
                except Exception:
                    pass

        def commit(*_a: Any) -> None:
            apply_text()
            # A new annotation left empty is discarded rather than leaving an
            # invisible zero-length artist on the figure.
            if is_new and not str(var.get()).strip():
                _remove_artist()
            self._cancel_inplace_edit()
            _redraw()

        def on_focus_out(_e: Any = None) -> None:
            # Only commit if focus left the editor entirely.  Clicking a
            # quick-style button (size/bold/color) moves focus within our own
            # widgets and must not close the editor.
            try:
                focused = widget.focus_get()
            except Exception:
                focused = None
            for w in self._inplace_widgets:
                try:
                    if focused is w or (focused is not None and str(focused).startswith(str(w))):
                        return
                except Exception:
                    pass
            commit()

        def cancel(*_a: Any) -> None:
            if is_new:
                # Discard a freshly-inserted annotation entirely.
                _remove_artist()
            else:
                try:
                    artist.set_text(cur_text)
                    artist.set_fontsize(cur_size)
                    artist.set_fontweight(cur_weight)
                    artist.set_color(cur_color)
                except Exception:
                    pass
            _redraw()
            self._cancel_inplace_edit()

        def bump_size(delta: float) -> None:
            state["size"] = max(1.0, state["size"] + delta)
            try:
                artist.set_fontsize(state["size"])
            except Exception:
                pass
            try:
                entry.configure(font=("TkDefaultFont", max(8, int(round(state["size"])))))
            except Exception:
                pass
            _redraw()

        def toggle_bold() -> None:
            state["weight"] = "normal" if state["weight"] in ("bold", "semibold") else "bold"
            try:
                artist.set_fontweight(state["weight"])
            except Exception:
                pass
            _redraw()

        def pick_color() -> None:
            try:
                picked = colorchooser.askcolor(color=state["color"], parent=widget)
                if picked and picked[1]:
                    state["color"] = picked[1]
                    artist.set_color(picked[1])
                    _redraw()
            except Exception:
                pass

        def open_more() -> None:
            cb = getattr(self, "_advanced_text_callback", None) or getattr(self, "_text_style_callback", None)
            self._cancel_inplace_edit()
            if cb is None:
                return
            try:
                cb(artist, self)
            except TypeError:
                try:
                    cb(artist)
                except Exception:
                    pass
            except Exception:
                pass

        entry.bind("<Return>", commit)
        entry.bind("<KP_Enter>", commit)
        entry.bind("<Escape>", cancel)
        entry.bind("<FocusOut>", on_focus_out)
        entry.bind("<KeyRelease>", apply_text)
        entry.bind("<Control-plus>", lambda e: bump_size(1))
        entry.bind("<Control-equal>", lambda e: bump_size(1))
        entry.bind("<Control-minus>", lambda e: bump_size(-1))
        entry.bind("<Control-b>", lambda e: toggle_bold())

        # Mouse-wheel font sizing while editing (cross-platform).
        def on_wheel(ev: Any) -> str:
            try:
                delta = getattr(ev, "delta", 0)
                if delta:
                    bump_size(1 if delta > 0 else -1)
                else:
                    # X11 reports wheel as Button-4 (up) / Button-5 (down).
                    bump_size(1 if int(getattr(ev, "num", 5)) == 4 else -1)
            except Exception:
                pass
            return "break"

        entry.bind("<MouseWheel>", on_wheel)
        entry.bind("<Button-4>", on_wheel)
        entry.bind("<Button-5>", on_wheel)

        # Small quick-style strip just below the entry.
        bar = tk.Frame(widget, bd=1, relief="solid", bg="#f8fafc")
        bar.place(x=max(0, x - 2), y=max(0, y - 2) + h + 6)
        tk.Button(bar, text="A-", width=2, command=lambda: bump_size(-1)).pack(side="left")
        tk.Button(bar, text="A+", width=2, command=lambda: bump_size(1)).pack(side="left")
        tk.Button(bar, text="B", width=2, command=toggle_bold).pack(side="left")
        tk.Button(bar, text="Color", command=pick_color).pack(side="left")
        tk.Button(bar, text="More…", command=open_more).pack(side="left")
        tk.Button(bar, text="✓", width=2, command=commit).pack(side="left")
        tk.Button(bar, text="✕", width=2, command=cancel).pack(side="left")

        self._inplace_widgets = [entry, bar]
        return True

    def _bind_tk_double_click_fallback(self) -> None:
        """Bind a Tk-level <Double-Button-1> fallback to the canvas widget.

        This covers cases where Matplotlib's button_press_event is emitted but
        the ``dblclick`` attribute is never set, or where the picked artist is
        too small for Matplotlib's picker.
        """
        if self._tk_double_binding_set:
            return
        widget = self.canvas.get_tk_widget()

        def on_tk_double(ev: Any) -> str:
            try:
                height = widget.winfo_height()
                pseudo = SimpleNamespace(x=float(ev.x), y=float(height - ev.y), button=1, dblclick=True)
                self._open_text_or_legend_at_event(pseudo)
            except Exception:
                pass
            return "break"

        try:
            widget.bind("<Double-Button-1>", on_tk_double, add="+")
            self._tk_double_binding_set = True
        except Exception:
            pass

    def enable_artist_dragging(self, enabled: bool = True) -> None:
        """Allow publication annotations, axis labels, titles and legends to be moved.

        Drag with the left mouse button to reposition. Double-click any title,
        label, tick, annotation, colour-bar text or pole-figure label to edit it
        *in place*: a text box opens directly on top of the item with a small
        quick-style strip (A-/A+ size, B bold, Color, More… for the full
        property dialog). Press Enter to commit, Esc to cancel. Legends open the
        structured legend dialog because they manage several entries at once.
        Text artists are moved in their own transform system, so manual
        placement is preserved on export.
        """
        for cid in getattr(self, "_drag_cids", []):
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._drag_cids = []
        self._drag_artist = None
        if not enabled:
            return

        def _make_pickable() -> None:
            # Make every Matplotlib Text artist interactive: axes titles, axis
            # labels, tick labels, annotations, colour-bar labels/ticks and
            # legend labels.  This is deliberately figure-wide so every plot
            # module has the same publication-editing behaviour.
            try:
                for text in self.fig.findobj(match=MplText):
                    try:
                        text.set_picker(6)
                    except Exception:
                        pass
            except Exception:
                pass
            for ax in self.fig.axes:
                leg = ax.get_legend()
                if leg is not None:
                    try:
                        leg.set_draggable(True)
                    except Exception:
                        pass
                    try:
                        leg.set_picker(True)
                    except Exception:
                        pass
                    try:
                        leg.get_frame().set_picker(True)
                    except Exception:
                        pass
                    try:
                        for text in leg.get_texts():
                            text.set_picker(6)
                        if leg.get_title() is not None:
                            leg.get_title().set_picker(6)
                    except Exception:
                        pass

        def on_pick(event: Any) -> None:
            artist = getattr(event, "artist", None)
            mouse = getattr(event, "mouseevent", None)
            if artist is None or mouse is None or mouse.x is None or mouse.y is None:
                return
            # Avoid taking over toolbar pan/zoom interactions.
            try:
                if self.toolbar.mode:
                    return
            except Exception:
                pass
            if not hasattr(artist, "get_position") or not hasattr(artist, "set_position"):
                if self._is_user_double_click(mouse):
                    self._open_text_or_legend_at_event(mouse)
                return
            if self._is_user_double_click(mouse):
                # Route through the shared opener so text gets the in-place
                # editor and legends get the structured dialog.  Prefer the
                # actual text under the cursor, because a legend frame may be
                # picked before the legend label.
                self._open_text_or_legend_at_event(mouse)
                self._drag_artist = None
                return
            inv = artist.get_transform().inverted()
            try:
                mx, my = inv.transform((mouse.x, mouse.y))
                x0, y0 = artist.get_position()
            except Exception:
                return
            self._drag_artist = artist
            self._drag_offset = (float(x0) - float(mx), float(y0) - float(my))

        def on_motion(event: Any) -> None:
            if self._drag_artist is None or event.x is None or event.y is None:
                return
            try:
                inv = self._drag_artist.get_transform().inverted()
                mx, my = inv.transform((event.x, event.y))
                self._drag_artist.set_position((mx + self._drag_offset[0], my + self._drag_offset[1]))
                self.canvas.draw_idle()
            except Exception:
                pass

        def on_button_press(event: Any) -> None:
            if not self._is_user_double_click(event):
                return
            self._open_text_or_legend_at_event(event)

        def on_release(event: Any) -> None:
            self._drag_artist = None

        def on_draw(event: Any) -> None:
            _make_pickable()

        _make_pickable()
        self._bind_tk_double_click_fallback()
        self._drag_cids = [
            self.canvas.mpl_connect("pick_event", on_pick),
            self.canvas.mpl_connect("button_press_event", on_button_press),
            self.canvas.mpl_connect("motion_notify_event", on_motion),
            self.canvas.mpl_connect("button_release_event", on_release),
            self.canvas.mpl_connect("draw_event", on_draw),
        ]

    @property
    def figure(self) -> Figure:
        """Alias for ``self.fig`` to match the matplotlib-style accessor that
        GUI callers expect (``canvas.figure.savefig(...)``).
        """
        return self.fig

    def draw(self) -> None:
        # A full (re)draw replaces the figure content, so close any in-place
        # editor that was hovering over the previous text.
        self._cancel_inplace_edit()
        self.canvas.draw_idle()


# =============================================================================
# Cached parsing wrappers (keyed on file path + mtime + size)
# =============================================================================


class ParseCache:
    """Tiny LRU keyed by (resolved_path, mtime, size)."""

    def __init__(self, capacity: int = 16) -> None:
        self.capacity = capacity
        self._store: Dict[Tuple[str, float, int], Any] = {}
        self._order: List[Tuple[str, float, int]] = []

    def get_or_compute(self, path: Path, fn: Callable[[Path], Any]) -> Any:
        key = _file_key(path)
        if key in self._store:
            return self._store[key]
        value = fn(path)
        self._store[key] = value
        self._order.append(key)
        while len(self._order) > self.capacity:
            old = self._order.pop(0)
            self._store.pop(old, None)
        return value

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()


PARSE_CACHE = ParseCache(capacity=16)


# =============================================================================
# Main Application
# =============================================================================


class VPSCApp(tk.Tk):
    """Top-level Tk window orchestrating all VPSC workflows."""

    NAV_NAMES = [
        "Dashboard", "Project", "VPSC8.IN", "Single Crystal", "Texture",
        "Process / BC", "Solver", "Run", "Results", "Files", "Theory",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.title(f"VPSC Python App · {APP_VERSION}")
        self.geometry("1450x900")
        self.minsize(1100, 720)

        self.state_data = ProjectState(base_dir=Path.cwd())
        self.vpsc_info = VPSCInInfo()
        self.sx_info: Optional[SXInfo] = None
        self.texture_data: Optional[TextureData] = None
        self.process_info: Optional[ProcessInfo] = None

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.process: Optional[subprocess.Popen[str]] = None
        self._process_lock = threading.Lock()
        self._run_log_error_detected = False

        self.prefs = load_prefs()

        self._setup_style()
        self._build_layout()
        self._load_defaults_from_data_dir()
        self.after(200, self._drain_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ styling
    def _setup_style(self) -> None:
        self.colors = {
            "bg": "#f6f8fb", "panel": "#ffffff", "ink": "#0f172a",
            "muted": "#64748b", "teal": "#0f766e", "teal2": "#115e59",
            "line": "#dbe4ef",
        }
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.configure(bg=self.colors["bg"])
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("Panel.TFrame", background=self.colors["panel"],
                        relief="solid", borderwidth=1)
        style.configure("Header.TFrame", background=self.colors["teal"])
        style.configure("Header.TLabel", background=self.colors["teal"],
                        foreground="white", font=("Segoe UI", 18, "bold"))
        style.configure("SubHeader.TLabel", background=self.colors["teal"],
                        foreground="#d1fae5", font=("Segoe UI", 9, "bold"))
        style.configure("Title.TLabel", background=self.colors["bg"],
                        foreground=self.colors["ink"],
                        font=("Segoe UI", 16, "bold"))
        style.configure("CardTitle.TLabel", background=self.colors["panel"],
                        foreground=self.colors["ink"],
                        font=("Segoe UI", 11, "bold"))
        style.configure("Muted.TLabel", background=self.colors["panel"],
                        foreground=self.colors["muted"],
                        font=("Segoe UI", 9))
        style.configure("TLabel", background=self.colors["bg"],
                        foreground=self.colors["ink"], font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9), padding=(10, 6))
        style.configure("Accent.TButton", background=self.colors["teal"],
                        foreground="white", font=("Segoe UI", 9, "bold"))
        style.map("Accent.TButton",
                  background=[("active", self.colors["teal2"])])
        style.configure("Nav.TButton", anchor="w", padding=(14, 9),
                        font=("Segoe UI", 10, "bold"))
        style.configure("Active.Nav.TButton",
                        background=self.colors["teal"], foreground="white",
                        anchor="w", padding=(14, 9),
                        font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", background="#ffffff",
                        fieldbackground="#ffffff",
                        foreground=self.colors["ink"], rowheight=24,
                        font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    # ------------------------------------------------------------------- layout
    def _build_layout(self) -> None:
        header = ttk.Frame(self, style="Header.TFrame")
        header.pack(side="top", fill="x")
        ttk.Label(header, text="VPSC Python App", style="Header.TLabel").pack(
            side="left", padx=18, pady=14
        )
        ttk.Label(
            header,
            text="Fortran-compatible VPSC platform · texture3-style PF/IPF studio · boundary visualisation",
            style="SubHeader.TLabel",
        ).pack(side="left", padx=10)
        ttk.Label(header, text=APP_VERSION, style="SubHeader.TLabel").pack(
            side="right", padx=20
        )

        body = ttk.Frame(self)
        body.pack(side="top", fill="both", expand=True)
        self.nav = ttk.Frame(body, style="Panel.TFrame", width=170)
        self.nav.pack(side="left", fill="y", padx=(8, 4), pady=8)
        self.nav.pack_propagate(False)
        ttk.Label(self.nav, text="Workflow", style="CardTitle.TLabel").pack(
            anchor="w", padx=16, pady=(18, 8)
        )
        self.nav_buttons: Dict[str, ttk.Button] = {}
        for name in self.NAV_NAMES:
            b = ttk.Button(self.nav, text=name, style="Nav.TButton",
                           command=lambda n=name: self.show_page(n))
            b.pack(fill="x", padx=10, pady=2)
            self.nav_buttons[name] = b
        self.status_label = ttk.Label(self.nav, text="Status\nReady",
                                       style="Muted.TLabel", justify="left")
        self.status_label.pack(anchor="w", padx=18, pady=(26, 8))

        self.content = ttk.Frame(body)
        self.content.pack(side="left", fill="both", expand=True,
                           padx=(4, 8), pady=8)
        self.pages: Dict[str, ttk.Frame] = {}
        for name in self.NAV_NAMES:
            frame = ttk.Frame(self.content)
            self.pages[name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self._build_dashboard_page()
        self._build_project_page()
        self._build_vpsc_in_page()
        self._build_single_crystal_page()
        self._build_texture_page()
        self._build_process_page()
        self._build_solver_page()
        self._build_run_page()
        self._build_results_page()
        self._build_files_page()
        self._build_theory_page()
        self.show_page("Project")

    def card(self, master: Any, title: str, subtitle: str = "") -> ttk.Frame:
        f = ttk.Frame(master, style="Panel.TFrame", padding=12)
        ttk.Label(f, text=title, style="CardTitle.TLabel").pack(anchor="w")
        if subtitle:
            ttk.Label(f, text=subtitle, style="Muted.TLabel").pack(
                anchor="w", pady=(2, 8)
            )
        return f

    def show_page(self, name: str) -> None:
        for n, b in self.nav_buttons.items():
            b.configure(style="Active.Nav.TButton" if n == name else "Nav.TButton")
        self.pages[name].tkraise()
        if name == "Results":
            self.refresh_results_files()

    # ----------------------------------------------------------- compact rows
    def _path_row(self, master: Any, label: str, var: tk.StringVar,
                  cmd: Callable[[], None]) -> None:
        row = ttk.Frame(master, style="Panel.TFrame")
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, style="Muted.TLabel", width=18).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x",
                                                expand=True, padx=6)
        ttk.Button(row, text="Browse", command=cmd).pack(side="left")

    def _entry_row(self, master: Any, label: str, var: tk.StringVar,
                   width: int = 12) -> ttk.Entry:
        row = ttk.Frame(master)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, style="Muted.TLabel", width=14).pack(side="left")
        ent = ttk.Entry(row, textvariable=var, width=width)
        ent.pack(side="left", fill="x", expand=True, padx=(4, 0))
        return ent

    def _combo_row(self, master: Any, label: str, var: tk.StringVar,
                   values: Sequence[str], width: int = 12) -> ttk.Combobox:
        row = ttk.Frame(master)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, style="Muted.TLabel", width=14).pack(side="left")
        cb = ttk.Combobox(row, textvariable=var, values=list(values),
                          state="readonly", width=width)
        cb.pack(side="left", fill="x", expand=True, padx=(4, 0))
        return cb

    def _color_row(self, master: Any, label: str, var: tk.StringVar,
                   width: int = 12) -> ttk.Combobox:
        row = ttk.Frame(master)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, style="Muted.TLabel", width=14).pack(side="left")
        cb = ttk.Combobox(row, textvariable=var,
                          values=list(COLOR_CHOICES.keys()),
                          state="readonly", width=width)
        cb.pack(side="left", fill="x", expand=True, padx=(4, 4))
        swatch = tk.Label(row, text="  ", width=2, relief="solid",
                          bg=resolve_color(var.get()))
        swatch.pack(side="left", padx=(0, 4))

        def refresh_swatch(*_: Any) -> None:
            try:
                swatch.configure(bg=resolve_color(var.get()))
            except tk.TclError:
                swatch.configure(bg="#2563eb")

        def pick_color() -> None:
            picked = colorchooser.askcolor(
                color=resolve_color(var.get()), parent=self
            )[1]
            if picked:
                var.set(picked)
                refresh_swatch()

        var.trace_add("write", refresh_swatch)
        ttk.Button(row, text="Pick", command=pick_color, width=7).pack(side="left")
        return cb

    # ------------------------------------------------------- publication styling
    def _ensure_publication_style_vars(self) -> None:
        """Global controls shared by all output modules.

        They are intentionally simple: the same text family/colour/size/weight
        is applied to titles, labels, tick labels, annotations and colorbar text.
        Users can then drag any visible text object to fine tune its position.
        """
        if hasattr(self, "plot_font_family"):
            return
        self.plot_font_family = tk.StringVar(value="Arial")
        self.plot_font_size = tk.StringVar(value="10")
        self.plot_title_size = tk.StringVar(value="12")
        self.plot_font_weight = tk.StringVar(value="normal")
        self.plot_text_color = tk.StringVar(value="Black")
        self.plot_title_x = tk.StringVar(value="0.5")
        self.plot_title_y = tk.StringVar(value="1.04")
        self.plot_xlabel_x = tk.StringVar(value="0.5")
        self.plot_xlabel_y = tk.StringVar(value="-0.10")
        self.plot_ylabel_x = tk.StringVar(value="-0.10")
        self.plot_ylabel_y = tk.StringVar(value="0.5")
        self.plot_left = tk.StringVar(value="auto")
        self.plot_right = tk.StringVar(value="auto")
        self.plot_top = tk.StringVar(value="auto")
        self.plot_bottom = tk.StringVar(value="auto")
        self.plot_enable_drag = tk.BooleanVar(value=True)
        self.bc_view = tk.StringVar(value="RD-TD")

    def _build_publication_style_controls(self, master: Any, title: str = "Publication text / layout") -> None:
        self._ensure_publication_style_vars()
        c = self.card(master, title,
                      "Font / layout settings are hidden by default to keep the workspace clean. Open them only when needed.")
        c.pack(fill="x", padx=4, pady=4)
        if not hasattr(self, "_pub_style_summary_var"):
            self._pub_style_summary_var = tk.StringVar()
            self._pub_style_dialog = None
            for v in [self.plot_font_family, self.plot_font_size, self.plot_title_size,
                      self.plot_font_weight, self.plot_text_color, self.plot_enable_drag]:
                try:
                    v.trace_add("write", lambda *_: self._update_publication_style_summary())
                except Exception:
                    pass
        self._update_publication_style_summary()
        ttk.Label(c, textvariable=self._pub_style_summary_var, style="Muted.TLabel",
                  wraplength=320, justify="left").pack(anchor="w", pady=(0, 4))
        row = ttk.Frame(c)
        row.pack(fill="x")
        ttk.Button(row, text="Text / layout settings…",
                   command=lambda t=title: self._open_publication_style_dialog(t)).pack(side="left")
        ttk.Button(row, text="Edit figure text…",
                   command=lambda t=title: self._open_figure_text_manager(t)).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(row, text="drag text/legend", variable=self.plot_enable_drag).pack(side="left", padx=(10, 0))

    def _update_publication_style_summary(self) -> None:
        if not hasattr(self, "_pub_style_summary_var"):
            return
        self._pub_style_summary_var.set(
            f"Current: {self.plot_font_family.get()} · size {self.plot_font_size.get()} · "
            f"title {self.plot_title_size.get()} · {self.plot_font_weight.get()} · "
            f"text {self.plot_text_color.get()} · drag={'on' if self.plot_enable_drag.get() else 'off'}"
        )

    def _populate_publication_style_dialog(self, master: Any) -> None:
        self._combo_row(master, "Font family", self.plot_font_family,
                        tuple(available_font_families()))
        self._combo_row(master, "Font weight", self.plot_font_weight,
                        ("normal", "bold", "semibold", "light"))
        self._entry_row(master, "Font size", self.plot_font_size)
        self._entry_row(master, "Title size", self.plot_title_size)
        self._color_row(master, "Text color", self.plot_text_color)
        self._entry_row(master, "Title x", self.plot_title_x)
        self._entry_row(master, "Title y", self.plot_title_y)
        self._entry_row(master, "X label x", self.plot_xlabel_x)
        self._entry_row(master, "X label y", self.plot_xlabel_y)
        self._entry_row(master, "Y label x", self.plot_ylabel_x)
        self._entry_row(master, "Y label y", self.plot_ylabel_y)
        self._entry_row(master, "left margin", self.plot_left)
        self._entry_row(master, "right margin", self.plot_right)
        self._entry_row(master, "top margin", self.plot_top)
        self._entry_row(master, "bottom margin", self.plot_bottom)
        ttk.Checkbutton(master, text="drag text/legend", variable=self.plot_enable_drag).pack(anchor="w", pady=(4, 0))

    def _open_publication_style_dialog_for_artist(self, artist: Optional[Any] = None, plot_canvas: Optional[PlotCanvas] = None) -> None:
        """Open the selected text editor when a Matplotlib text artist is double-clicked.

        If the double-click target is not a text object, fall back to the global
        publication style dialog.
        """
        if artist is not None and hasattr(artist, "get_text") and hasattr(artist, "set_text"):
            self._open_text_artist_dialog(artist, plot_canvas)
            return
        if artist is not None and hasattr(artist, "get_texts") and hasattr(artist, "set_visible"):
            self._open_legend_artist_dialog(artist, plot_canvas)
            return
        self._open_publication_style_dialog("Text / layout settings")

    def _redraw_plot_canvas_for_artist(self, plot_canvas: Optional[PlotCanvas]) -> None:
        try:
            if plot_canvas is not None:
                plot_canvas.canvas.draw_idle()
                return
        except Exception:
            pass
        for name in ("texture_canvas", "process_canvas", "results_canvas"):
            pc = getattr(self, name, None)
            try:
                if pc is not None:
                    pc.canvas.draw_idle()
            except Exception:
                pass

    def _available_plot_canvases(self) -> List[Tuple[str, PlotCanvas]]:
        """Return currently created plotting canvases for explicit text editing.

        This manager is intentionally independent of mouse double-click events.
        Users can always open it from the right-side panel and edit the text
        objects listed for the active figure.
        """
        out: List[Tuple[str, PlotCanvas]] = []
        for attr, label in (("results_canvas", "Results figure"),
                            ("texture_canvas", "Texture figure"),
                            ("process_canvas", "Process / BC figure")):
            try:
                pc = getattr(self, attr, None)
                if pc is not None and getattr(pc, "figure", None) is not None:
                    out.append((label, pc))
            except Exception:
                pass
        return out

    def _preferred_canvas_label(self, title: str = "") -> str:
        low = (title or "").lower()
        if "result" in low:
            return "Results figure"
        if "bc" in low or "boundary" in low or "process" in low:
            return "Process / BC figure"
        if "texture" in low or "pf" in low or "ipf" in low:
            return "Texture figure"
        canvases = self._available_plot_canvases()
        return canvases[0][0] if canvases else ""

    def _classify_text_artist(self, fig: Figure, artist: Any) -> str:
        """Human-readable role for a Matplotlib Text artist."""
        try:
            for i, ax in enumerate(fig.axes, start=1):
                if artist is getattr(ax, "title", None):
                    return f"Axes {i}: title"
                if artist is ax.xaxis.label:
                    return f"Axes {i}: x label"
                if artist is ax.yaxis.label:
                    return f"Axes {i}: y label"
                try:
                    if artist in ax.get_xticklabels() or artist in ax.get_xticklabels(minor=True):
                        return f"Axes {i}: x tick"
                    if artist in ax.get_yticklabels() or artist in ax.get_yticklabels(minor=True):
                        return f"Axes {i}: y tick"
                except Exception:
                    pass
                try:
                    if artist in ax.texts:
                        return f"Axes {i}: annotation"
                except Exception:
                    pass
                try:
                    leg = ax.get_legend()
                except Exception:
                    leg = None
                if leg is not None:
                    try:
                        if artist is leg.get_title():
                            return f"Axes {i}: legend title"
                        if artist in leg.get_texts():
                            return f"Axes {i}: legend label"
                    except Exception:
                        pass
            try:
                if artist in fig.texts:
                    return "Figure text"
            except Exception:
                pass
        except Exception:
            pass
        return "Text"

    def _collect_text_items_for_canvas(self, plot_canvas: PlotCanvas, *, include_ticks: bool = True,
                                       include_empty: bool = True) -> List[Tuple[str, Any]]:
        """Collect editable Text artists from a figure without relying on picking."""
        fig = plot_canvas.figure
        items: List[Tuple[str, Any]] = []
        seen: set[int] = set()
        try:
            texts = list(fig.findobj(match=MplText))
        except Exception:
            texts = []
        for t in texts:
            if id(t) in seen:
                continue
            seen.add(id(t))
            try:
                role = self._classify_text_artist(fig, t)
                if (not include_ticks) and (" tick" in role):
                    continue
                s = str(t.get_text() or "").replace("\n", " ").strip()
                if (not include_empty) and not s:
                    continue
                show = "" if bool(t.get_visible()) else " [hidden]"
                label = f"{role}: {s if s else '(empty)'}{show}"
                items.append((label, t))
            except Exception:
                pass
        # Stable order: important labels first, tick labels after.
        def _rank(pair: Tuple[str, Any]) -> Tuple[int, str]:
            lab = pair[0]
            if "title" in lab: return (0, lab)
            if " label" in lab and "legend" not in lab: return (1, lab)
            if "legend" in lab: return (2, lab)
            if "annotation" in lab: return (3, lab)
            if "tick" in lab: return (4, lab)
            return (5, lab)
        items.sort(key=_rank)
        return items

    def _open_figure_text_manager(self, title: str = "Figure text manager") -> None:
        """Explicit text manager for the current figure.

        This is the robust, primary editing path.  It does not require double
        clicking on the embedded Matplotlib canvas, so it works consistently on
        Windows/Tk backends.  It edits titles, axis labels, tick labels,
        annotations, PF/IPF labels, colorbar text, and legend labels.
        """
        self._ensure_publication_style_vars()
        canvases = self._available_plot_canvases()
        if not canvases:
            messagebox.showinfo("No figure", "No plotting canvas has been created yet.")
            return

        win = tk.Toplevel(self)
        win.title(title if title else "Figure text manager")
        win.geometry("980x640")
        win.minsize(820, 520)
        win.transient(self)
        win.resizable(True, True)

        canvas_by_label: Dict[str, PlotCanvas] = {lab: pc for lab, pc in canvases}
        canvas_var = tk.StringVar(value=self._preferred_canvas_label(title))
        if canvas_var.get() not in canvas_by_label:
            canvas_var.set(canvases[0][0])
        include_ticks_var = tk.BooleanVar(value=True)
        include_empty_var = tk.BooleanVar(value=True)
        selected_artist: Dict[str, Any] = {"artist": None, "canvas": None}
        loading = {"value": False}
        item_cache: List[Tuple[str, Any]] = []

        top = ttk.Frame(win, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text="Figure").pack(side="left")
        canvas_combo = ttk.Combobox(top, textvariable=canvas_var, values=list(canvas_by_label.keys()),
                                    state="readonly", width=24)
        canvas_combo.pack(side="left", padx=(6, 12))
        ttk.Checkbutton(top, text="include tick labels", variable=include_ticks_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(top, text="include empty text", variable=include_empty_var).pack(side="left", padx=(0, 10))

        main = ttk.Frame(win, padding=(10, 0, 10, 8))
        main.pack(fill="both", expand=True)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_rowconfigure(1, weight=1)
        ttk.Label(left, text="Editable text objects").grid(row=0, column=0, sticky="w")
        list_frame = ttk.Frame(left)
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(0, weight=1)
        lb = tk.Listbox(list_frame, exportselection=False, activestyle="dotbox")
        lb.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=lb.yview)
        sb.grid(row=0, column=1, sticky="ns")
        lb.configure(yscrollcommand=sb.set)

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(1, weight=1)
        r = 0
        text_var = tk.StringVar(value="")
        visible_var = tk.BooleanVar(value=True)
        family_var = tk.StringVar(value=self.plot_font_family.get() or "Arial")
        size_var = tk.StringVar(value=self.plot_font_size.get() or "10")
        weight_var = tk.StringVar(value="normal")
        style_var = tk.StringVar(value="normal")
        color_var = tk.StringVar(value="#111827")
        rotation_var = tk.StringVar(value="0")
        ha_var = tk.StringVar(value="center")
        va_var = tk.StringVar(value="center")
        x_var = tk.StringVar(value="0")
        y_var = tk.StringVar(value="0")
        status_var = tk.StringVar(value="Select a text object, edit fields, then click Save / apply.")

        def add_row(label: str, var: tk.StringVar, values: Optional[Sequence[str]] = None) -> None:
            nonlocal r
            ttk.Label(right, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=4)
            if values is None:
                ttk.Entry(right, textvariable=var).grid(row=r, column=1, sticky="ew", pady=4)
            else:
                ttk.Combobox(right, textvariable=var, values=tuple(values), state="readonly").grid(row=r, column=1, sticky="ew", pady=4)
            r += 1

        ttk.Label(right, text="Content / displayed name").grid(row=r, column=0, sticky="nw", padx=(0, 8), pady=4)
        ttk.Entry(right, textvariable=text_var).grid(row=r, column=1, sticky="ew", pady=4)
        r += 1
        ttk.Checkbutton(right, text="Show / keep this text", variable=visible_var).grid(row=r, column=0, columnspan=2, sticky="w", pady=4)
        r += 1
        add_row("Font family", family_var, tuple(available_font_families()))
        add_row("Font size", size_var)
        add_row("Font weight", weight_var, ("normal", "bold", "semibold", "light"))
        add_row("Font style", style_var, ("normal", "italic", "oblique"))
        add_row("Text color", color_var)
        add_row("Rotation", rotation_var)
        add_row("Horizontal align", ha_var, ("left", "center", "right"))
        add_row("Vertical align", va_var, ("top", "center", "bottom", "baseline"))
        add_row("Position x", x_var)
        add_row("Position y", y_var)

        def _current_canvas() -> PlotCanvas:
            return canvas_by_label.get(canvas_var.get(), canvases[0][1])

        def refresh_list(*_args: Any) -> None:
            nonlocal item_cache
            pc = _current_canvas()
            item_cache = self._collect_text_items_for_canvas(pc,
                                                             include_ticks=bool(include_ticks_var.get()),
                                                             include_empty=bool(include_empty_var.get()))
            lb.delete(0, "end")
            for label, _artist in item_cache:
                lb.insert("end", label)
            selected_artist["artist"] = None
            selected_artist["canvas"] = pc
            status_var.set(f"Found {len(item_cache)} editable text objects.")

        def load_selected(*_args: Any) -> None:
            sel = lb.curselection()
            if not sel:
                return
            idx = int(sel[0])
            if idx < 0 or idx >= len(item_cache):
                return
            artist = item_cache[idx][1]
            selected_artist["artist"] = artist
            selected_artist["canvas"] = _current_canvas()
            loading["value"] = True
            try:
                text_var.set(str(artist.get_text() or ""))
            except Exception:
                text_var.set("")
            try:
                visible_var.set(bool(artist.get_visible()))
            except Exception:
                visible_var.set(True)
            try:
                fam = artist.get_fontfamily()
                family_var.set(str(fam[0] if isinstance(fam, (list, tuple)) and fam else fam))
            except Exception:
                family_var.set(self.plot_font_family.get() or "Arial")
            try:
                size_var.set(fmt_num(float(artist.get_fontsize()), 4))
            except Exception:
                size_var.set(self.plot_font_size.get() or "10")
            try:
                weight_var.set(str(artist.get_fontweight() or "normal"))
            except Exception:
                weight_var.set("normal")
            try:
                style_var.set(str(artist.get_fontstyle() or "normal"))
            except Exception:
                style_var.set("normal")
            try:
                color_var.set(mpl_colors.to_hex(mpl_colors.to_rgba(artist.get_color())))
            except Exception:
                color_var.set("#111827")
            try:
                rot = artist.get_rotation()
                if isinstance(rot, str):
                    rot = 0 if rot.lower() in {"horizontal", "none"} else 90
                rotation_var.set(fmt_num(float(rot), 4))
            except Exception:
                rotation_var.set("0")
            try:
                ha_var.set(str(artist.get_ha() or "center"))
                va_var.set(str(artist.get_va() or "center"))
            except Exception:
                pass
            try:
                x0, y0 = artist.get_position()
                x_var.set(fmt_num(float(x0), 6))
                y_var.set(fmt_num(float(y0), 6))
            except Exception:
                x_var.set("0"); y_var.set("0")
            loading["value"] = False
            status_var.set("Loaded selected text. Modify fields and click Save / apply.")

        def redraw() -> None:
            pc = selected_artist.get("canvas") or _current_canvas()
            try:
                pc.canvas.draw_idle()
            except Exception:
                self._redraw_plot_canvas_for_artist(pc)

        def apply_selected(*_args: Any) -> None:
            if loading["value"]:
                return
            artist = selected_artist.get("artist")
            if artist is None:
                status_var.set("No text object selected.")
                return
            pc = selected_artist.get("canvas") or _current_canvas()
            try: artist.set_text(text_var.get())
            except Exception: pass
            try: artist.set_visible(bool(visible_var.get()))
            except Exception: pass
            try: artist.set_fontfamily(family_var.get() or "Arial")
            except Exception: pass
            try: artist.set_fontsize(max(1.0, safe_float(size_var.get(), 10.0)))
            except Exception: pass
            try: artist.set_fontweight(weight_var.get() or "normal")
            except Exception: pass
            try: artist.set_fontstyle(style_var.get() or "normal")
            except Exception: pass
            try: artist.set_color(resolve_color(color_var.get(), color_var.get() or "#111827"))
            except Exception: pass
            try: artist.set_rotation(safe_float(rotation_var.get(), 0.0))
            except Exception: pass
            try: artist.set_ha(ha_var.get() or "center")
            except Exception: pass
            try: artist.set_va(va_var.get() or "center")
            except Exception: pass
            try: artist.set_position((safe_float(x_var.get(), 0.0), safe_float(y_var.get(), 0.0)))
            except Exception: pass
            try:
                leg, idx = self._find_parent_legend_for_text(artist, pc)
                if leg is not None and idx >= 0:
                    handles = self._legend_handles_for_dialog(leg)
                    if idx < len(handles):
                        handles[idx].set_visible(bool(visible_var.get()))
            except Exception:
                pass
            redraw()
            status_var.set("Saved to the current figure. Export PNG now to keep this layout.")
            # Keep the row name synchronized after a rename, without changing the selection.
            try:
                cur = lb.curselection()
                old_idx = int(cur[0]) if cur else -1
                refresh_list()
                if 0 <= old_idx < lb.size():
                    lb.selection_clear(0, "end")
                    lb.selection_set(old_idx)
                    lb.see(old_idx)
                    load_selected()
            except Exception:
                pass

        def save_and_close() -> None:
            apply_selected()
            win.destroy()

        def hide_selected() -> None:
            visible_var.set(False)
            apply_selected()

        def pick_color() -> None:
            picked = colorchooser.askcolor(color=color_var.get() or "#111827", parent=win)
            if picked and picked[1]:
                color_var.set(str(picked[1]))

        btns = ttk.Frame(right)
        btns.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        ttk.Button(btns, text="Save / apply", command=apply_selected).pack(side="left")
        ttk.Button(btns, text="Save and close", command=save_and_close).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Delete / hide", command=hide_selected).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Pick color", command=pick_color).pack(side="left", padx=(8, 0))
        r += 1
        ttk.Label(right, textvariable=status_var, style="Muted.TLabel", wraplength=430, justify="left").grid(row=r, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        lb.bind("<<ListboxSelect>>", load_selected)
        canvas_combo.bind("<<ComboboxSelected>>", refresh_list)
        for v in (include_ticks_var, include_empty_var):
            try:
                v.trace_add("write", refresh_list)
            except Exception:
                pass
        ttk.Button(top, text="Refresh list", command=refresh_list).pack(side="right")
        ttk.Button(top, text="Close", command=win.destroy).pack(side="right", padx=(0, 8))

        refresh_list()
        if item_cache:
            lb.selection_set(0)
            load_selected()
        try:
            win.lift(); win.focus_force()
        except Exception:
            pass

    def _find_parent_legend_for_text(self, artist: Any, plot_canvas: Optional[PlotCanvas] = None) -> Tuple[Optional[Any], int]:
        """Return (legend, entry_index) when a text artist belongs to a legend."""
        figs: List[Any] = []
        try:
            if plot_canvas is not None:
                figs.append(plot_canvas.figure)
        except Exception:
            pass
        for name in ("texture_canvas", "process_canvas", "results_canvas"):
            try:
                pc = getattr(self, name, None)
                if pc is not None and pc.figure not in figs:
                    figs.append(pc.figure)
            except Exception:
                pass
        for fig in figs:
            try:
                axes = list(fig.axes)
            except Exception:
                axes = []
            for ax in axes:
                try:
                    leg = ax.get_legend()
                except Exception:
                    leg = None
                if leg is None:
                    continue
                try:
                    texts = list(leg.get_texts())
                    for i, t in enumerate(texts):
                        if t is artist:
                            return leg, i
                    if leg.get_title() is artist:
                        return leg, -1
                except Exception:
                    pass
        return None, -999

    def _legend_handles_for_dialog(self, legend: Any) -> List[Any]:
        try:
            handles = list(getattr(legend, "legend_handles", []))
            if handles:
                return handles
        except Exception:
            pass
        try:
            handles = list(getattr(legend, "legendHandles", []))
            if handles:
                return handles
        except Exception:
            pass
        return []

    def _delete_legend_entry(self, legend: Any, index: int, plot_canvas: Optional[PlotCanvas] = None) -> None:
        """Hide one legend entry in-place without redrawing the data curve."""
        try:
            texts = list(legend.get_texts())
            if 0 <= index < len(texts):
                texts[index].set_visible(False)
                texts[index].set_text("")
        except Exception:
            pass
        try:
            handles = self._legend_handles_for_dialog(legend)
            if 0 <= index < len(handles):
                handles[index].set_visible(False)
        except Exception:
            pass
        self._redraw_plot_canvas_for_artist(plot_canvas)

    def _open_legend_artist_dialog(self, legend: Any, plot_canvas: Optional[PlotCanvas] = None) -> None:
        """Edit or hide a complete Matplotlib legend.

        This editor is deliberately self-contained and robust: even if a legend
        handle is missing in a Matplotlib version, the text entries remain
        editable and the window never opens as a blank panel.
        """
        try:
            old = getattr(legend, "_vpsc_legend_editor", None)
            if old is not None and bool(old.winfo_exists()):
                old.deiconify(); old.lift(); old.focus_force()
                return
        except Exception:
            pass
        win = tk.Toplevel(self)
        try:
            legend._vpsc_legend_editor = win
        except Exception:
            pass
        win.title("Edit legend")
        win.geometry("520x520")
        win.transient(self)
        win.resizable(True, True)

        outer = ttk.Frame(win, padding=8)
        outer.pack(fill="both", expand=True)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(1, weight=1)

        visible_var = tk.BooleanVar(value=bool(getattr(legend, "get_visible", lambda: True)()))
        ttk.Checkbutton(outer, text="Show legend", variable=visible_var).grid(row=0, column=0, sticky="w", pady=(0, 6))

        scroll = ScrollableFrame(outer, width=480)
        scroll.grid(row=1, column=0, sticky="nsew")
        body = scroll.inner
        body.grid_columnconfigure(1, weight=1)

        title_obj = None
        try:
            title_obj = legend.get_title()
        except Exception:
            title_obj = None
        title_var = tk.StringVar(value=str(title_obj.get_text()) if title_obj is not None else "")
        ttk.Label(body, text="Legend title").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(body, textvariable=title_var).grid(row=0, column=1, sticky="ew", pady=4)
        r = 1

        ttk.Label(body, text="Legend entries", font=("Arial", 9, "bold")).grid(row=r, column=0, columnspan=3, sticky="w", pady=(10, 4))
        r += 1
        entry_vars: List[tk.StringVar] = []
        entry_show_vars: List[tk.BooleanVar] = []
        try:
            texts = list(legend.get_texts())
        except Exception:
            texts = []
        handles = self._legend_handles_for_dialog(legend)
        for i, txt in enumerate(texts):
            show_var = tk.BooleanVar(value=bool(getattr(txt, "get_visible", lambda: True)()))
            label_var = tk.StringVar(value=str(getattr(txt, "get_text", lambda: "")()))
            entry_show_vars.append(show_var)
            entry_vars.append(label_var)
            ttk.Checkbutton(body, variable=show_var).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Entry(body, textvariable=label_var).grid(row=r, column=1, sticky="ew", pady=3)

            def hide_one(index: int = i) -> None:
                if index < len(entry_show_vars):
                    entry_show_vars[index].set(False)
                apply_changes()

            ttk.Button(body, text="Delete", command=hide_one).grid(row=r, column=2, sticky="e", padx=(6, 0), pady=3)
            r += 1
        if not texts:
            ttk.Label(body, text="No legend entries detected.", style="Muted.TLabel").grid(row=r, column=0, columnspan=3, sticky="w", pady=6)
            r += 1

        def apply_changes(*_args: Any) -> None:
            try:
                legend.set_visible(bool(visible_var.get()))
            except Exception:
                pass
            try:
                if title_obj is not None:
                    title_obj.set_text(title_var.get())
            except Exception:
                pass
            for i, (label_var, show_var) in enumerate(zip(entry_vars, entry_show_vars)):
                try:
                    if i < len(texts):
                        texts[i].set_text(label_var.get())
                        texts[i].set_visible(bool(show_var.get()))
                except Exception:
                    pass
                try:
                    if i < len(handles):
                        handles[i].set_visible(bool(show_var.get()))
                except Exception:
                    pass
            self._redraw_plot_canvas_for_artist(plot_canvas)

        for v in [visible_var, title_var] + entry_vars + entry_show_vars:
            try:
                v.trace_add("write", apply_changes)
            except Exception:
                pass

        btns = ttk.Frame(win, padding=(8, 6))
        btns.pack(fill="x")
        ttk.Button(btns, text="Save / apply", command=apply_changes).pack(side="left")
        ttk.Button(btns, text="Hide legend", command=lambda: (visible_var.set(False), apply_changes())).pack(side="left", padx=(8, 0))

        def _closed() -> None:
            try:
                legend._vpsc_legend_editor = None
            except Exception:
                pass
            win.destroy()

        ttk.Button(btns, text="Close", command=_closed).pack(side="right")
        win.protocol("WM_DELETE_WINDOW", _closed)
        try:
            win.lift(); win.focus_force()
        except Exception:
            pass

    def _open_text_artist_dialog(self, artist: Any, plot_canvas: Optional[PlotCanvas] = None) -> None:
        """Robust editor for any Matplotlib text artist or legend label.

        The dialog is intentionally built without a scroll-only body so the
        essential controls are always visible: content, font settings, color,
        save/apply, save-and-close, and delete/hide.  It is safe for ordinary
        text, axis labels, tick labels, colorbar labels, pole-figure labels,
        legend labels, and legend titles.
        """
        self._ensure_publication_style_vars()
        try:
            old = getattr(artist, "_vpsc_text_editor", None)
            if old is not None and bool(old.winfo_exists()):
                old.deiconify(); old.lift(); old.focus_force()
                return
        except Exception:
            pass

        def _safe_call(name: str, default: Any = "") -> Any:
            try:
                fn = getattr(artist, name, None)
                if callable(fn):
                    return fn()
            except Exception:
                pass
            return default

        def _safe_text() -> str:
            try:
                value = _safe_call("get_text", "")
                return "" if value is None else str(value)
            except Exception:
                return ""

        def _safe_float_value(value: Any, default: float = 0.0) -> float:
            try:
                if isinstance(value, str):
                    low = value.strip().lower()
                    if low in {"horizontal", "none"}:
                        return 0.0
                    if low == "vertical":
                        return 90.0
                return float(value)
            except Exception:
                return float(default)

        def _safe_font_family() -> str:
            fam = _safe_call("get_fontfamily", self.plot_font_family.get() or "Arial")
            try:
                if isinstance(fam, (list, tuple)) and fam:
                    return str(fam[0])
                return str(fam)
            except Exception:
                return self.plot_font_family.get() or "Arial"

        def _safe_color() -> str:
            col = _safe_call("get_color", self.plot_text_color.get() or "Black")
            try:
                return mpl_colors.to_hex(mpl_colors.to_rgba(col))
            except Exception:
                return str(col) if col is not None else (self.plot_text_color.get() or "Black")

        title_text = _safe_text().replace("\n", " ").strip() or "Text artist"
        legend_parent, legend_index = self._find_parent_legend_for_text(artist, plot_canvas)

        # Values are prepared before any Tk controls are created.  Fallbacks are
        # deliberately conservative so the dialog cannot become half-built.
        initial_text = _safe_text()
        if not str(initial_text).strip() and title_text != "Text artist":
            initial_text = title_text
        try:
            x0, y0 = artist.get_position()
        except Exception:
            x0, y0 = 0.0, 0.0

        text_var = tk.StringVar(value=str(initial_text))
        visible_var = tk.BooleanVar(value=bool(_safe_call("get_visible", True)))
        family_var = tk.StringVar(value=_safe_font_family())
        weight_var = tk.StringVar(value=str(_safe_call("get_fontweight", self.plot_font_weight.get() or "normal")))
        style_var = tk.StringVar(value=str(_safe_call("get_fontstyle", "normal")))
        size_var = tk.StringVar(value=fmt_num(_safe_float_value(_safe_call("get_fontsize", safe_float(self.plot_font_size.get(), 10.0)), 10.0)))
        color_var = tk.StringVar(value=_safe_color())
        rotation_var = tk.StringVar(value=fmt_num(_safe_float_value(_safe_call("get_rotation", 0.0), 0.0)))
        ha_var = tk.StringVar(value=str(_safe_call("get_ha", "center")))
        va_var = tk.StringVar(value=str(_safe_call("get_va", "center")))
        x_var = tk.StringVar(value=fmt_num(_safe_float_value(x0, 0.0)))
        y_var = tk.StringVar(value=fmt_num(_safe_float_value(y0, 0.0)))
        live_var = tk.BooleanVar(value=True)
        status_var = tk.StringVar(value="Live editing on — changes apply as you type. Toggle 'Live apply' to batch them.")

        win = tk.Toplevel(self)
        try:
            artist._vpsc_text_editor = win
        except Exception:
            pass
        win.title(f"Edit text / legend label — {title_text[:56]}")
        win.geometry("620x560")
        win.minsize(560, 480)
        win.transient(self)
        win.resizable(True, True)

        outer = ttk.Frame(win, padding=10)
        outer.pack(fill="both", expand=True)
        outer.grid_columnconfigure(1, weight=1)

        def _redraw() -> None:
            self._redraw_plot_canvas_for_artist(plot_canvas)

        def apply_changes(*_args: Any) -> None:
            try:
                artist.set_text(text_var.get())
            except Exception:
                pass
            try:
                artist.set_fontfamily(family_var.get() or "Arial")
            except Exception:
                pass
            try:
                artist.set_fontweight(weight_var.get() or "normal")
            except Exception:
                pass
            try:
                artist.set_fontstyle(style_var.get() or "normal")
            except Exception:
                pass
            try:
                artist.set_fontsize(max(1.0, safe_float(size_var.get(), 10.0)))
            except Exception:
                pass
            try:
                artist.set_color(resolve_color(color_var.get(), color_var.get() or "#111827"))
            except Exception:
                pass
            try:
                artist.set_rotation(safe_float(rotation_var.get(), 0.0))
            except Exception:
                pass
            try:
                artist.set_ha(ha_var.get() or "center")
            except Exception:
                pass
            try:
                artist.set_va(va_var.get() or "center")
            except Exception:
                pass
            try:
                artist.set_position((safe_float(x_var.get(), _safe_float_value(x0, 0.0)),
                                     safe_float(y_var.get(), _safe_float_value(y0, 0.0))))
            except Exception:
                pass
            try:
                artist.set_visible(bool(visible_var.get()))
            except Exception:
                pass
            if legend_parent is not None and legend_index >= 0:
                try:
                    handles = self._legend_handles_for_dialog(legend_parent)
                    if legend_index < len(handles):
                        handles[legend_index].set_visible(bool(visible_var.get()))
                except Exception:
                    pass
            _redraw()
            status_var.set("Saved to current figure canvas.")

        def delete_this_text() -> None:
            visible_var.set(False)
            try:
                artist.set_visible(False)
            except Exception:
                pass
            if legend_parent is not None and legend_index >= 0:
                self._delete_legend_entry(legend_parent, legend_index, plot_canvas)
            else:
                _redraw()
            status_var.set("Hidden/deleted from current figure canvas.")

        def _closed() -> None:
            try:
                artist._vpsc_text_editor = None
            except Exception:
                pass
            try:
                win.destroy()
            except Exception:
                pass

        def save_and_close() -> None:
            apply_changes()
            _closed()

        def pick_color() -> None:
            try:
                picked = colorchooser.askcolor(color=resolve_color(color_var.get(), "#111827"), parent=win)
                if picked and picked[1]:
                    color_var.set(picked[1])
                    if live_var.get():
                        apply_changes()
            except Exception:
                pass

        def maybe_live(*_args: Any) -> None:
            if bool(live_var.get()):
                apply_changes()

        # --- Always-visible content and action area ---
        ttk.Label(outer, text="Content / displayed name").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        content = ttk.Entry(outer, textvariable=text_var)
        content.grid(row=0, column=1, columnspan=3, sticky="ew", pady=(0, 6))
        ttk.Checkbutton(outer, text="Show / keep this text", variable=visible_var).grid(row=1, column=0, sticky="w", pady=(0, 8))
        ttk.Checkbutton(outer, text="Live apply", variable=live_var).grid(row=1, column=1, sticky="w", pady=(0, 8))

        action = ttk.Frame(outer)
        action.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        ttk.Button(action, text="Save / apply", command=apply_changes).pack(side="left")
        ttk.Button(action, text="Save and close", command=save_and_close).pack(side="left", padx=(8, 0))
        ttk.Button(action, text="Delete / hide", command=delete_this_text).pack(side="left", padx=(8, 0))
        if legend_parent is not None:
            ttk.Button(action, text="Edit full legend", command=lambda: self._open_legend_artist_dialog(legend_parent, plot_canvas)).pack(side="left", padx=(8, 0))

        row = 3
        if legend_parent is not None and legend_index >= 0:
            ttk.Label(outer, text="Legend entry detected: the Content field renames this legend item; Delete hides its symbol and label.",
                      style="Muted.TLabel", wraplength=540, justify="left").grid(row=row, column=0, columnspan=4, sticky="ew", pady=(0, 8))
            row += 1

        def add_combo(label: str, var: tk.StringVar, values: Sequence[str]) -> None:
            nonlocal row
            ttk.Label(outer, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            cb = ttk.Combobox(outer, textvariable=var, values=tuple(values), state="readonly")
            cb.grid(row=row, column=1, columnspan=3, sticky="ew", pady=4)
            row += 1

        def add_entry(label: str, var: tk.StringVar, button: Optional[Tuple[str, Callable[[], None]]] = None) -> None:
            nonlocal row
            ttk.Label(outer, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            ent = ttk.Entry(outer, textvariable=var)
            if button is None:
                ent.grid(row=row, column=1, columnspan=3, sticky="ew", pady=4)
            else:
                ent.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
                ttk.Button(outer, text=button[0], command=button[1]).grid(row=row, column=3, sticky="e", padx=(6, 0), pady=4)
            row += 1

        add_combo("Font family", family_var, tuple(available_font_families()))
        add_combo("Font weight", weight_var, ("normal", "bold", "semibold", "light"))
        add_combo("Font style", style_var, ("normal", "italic", "oblique"))
        add_entry("Font size", size_var)
        add_entry("Text color", color_var, ("Pick", pick_color))
        add_entry("Rotation", rotation_var)
        add_combo("Horizontal align", ha_var, ("left", "center", "right"))
        add_combo("Vertical align", va_var, ("top", "center", "bottom", "baseline"))
        add_entry("Position x", x_var)
        add_entry("Position y", y_var)

        ttk.Label(outer, textvariable=status_var, style="Muted.TLabel", wraplength=540, justify="left").grid(row=row, column=0, columnspan=4, sticky="ew", pady=(10, 6))
        row += 1

        bottom = ttk.Frame(outer)
        bottom.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        ttk.Button(bottom, text="Save / apply", command=apply_changes).pack(side="left")
        ttk.Button(bottom, text="Save and close", command=save_and_close).pack(side="left", padx=(8, 0))
        ttk.Button(bottom, text="Delete / hide", command=delete_this_text).pack(side="left", padx=(8, 0))
        ttk.Button(bottom, text="Global text / layout…", command=lambda: self._open_publication_style_dialog("Publication text / layout")).pack(side="left", padx=(8, 0))
        ttk.Button(bottom, text="Close", command=_closed).pack(side="right")

        for var in [text_var, visible_var, family_var, weight_var, style_var, size_var, color_var, rotation_var, ha_var, va_var, x_var, y_var]:
            try:
                var.trace_add("write", maybe_live)
            except Exception:
                pass

        win.protocol("WM_DELETE_WINDOW", _closed)
        try:
            content.focus_set(); win.lift(); win.focus_force()
        except Exception:
            pass

    def _open_publication_style_dialog(self, title: str = "Publication text / layout") -> None:
        self._ensure_publication_style_vars()
        win = getattr(self, "_pub_style_dialog", None)
        if win is not None and bool(win.winfo_exists()):
            win.deiconify()
            win.lift()
            win.focus_force()
            return
        win = tk.Toplevel(self)
        self._pub_style_dialog = win
        win.title(title)
        win.geometry("460x640")
        win.transient(self)
        body = ScrollableFrame(win)
        body.pack(fill="both", expand=True)
        self._populate_publication_style_dialog(body.inner)
        btns = ttk.Frame(win, padding=(8, 6))
        btns.pack(fill="x")

        def _closed() -> None:
            self._update_publication_style_summary()
            self._pub_style_dialog = None
            win.destroy()

        ttk.Button(btns, text="Close", command=_closed).pack(side="right")
        win.protocol("WM_DELETE_WINDOW", _closed)

    def _auto_margin(self, value: tk.StringVar, default: float) -> float:
        txt = value.get().strip().lower()
        if not txt or txt == "auto":
            return default
        return max(0.02, min(0.98, safe_float(txt, default)))

    def _apply_publication_style(self, fig: Figure) -> None:
        self._ensure_publication_style_vars()
        family = self.plot_font_family.get() or "Arial"
        text_color = resolve_color(self.plot_text_color.get(), "#111827")
        base_size = max(5.0, safe_float(self.plot_font_size.get(), 10.0))
        title_size = max(5.0, safe_float(self.plot_title_size.get(), base_size + 2.0))
        weight = self.plot_font_weight.get() or "normal"
        tx = safe_float(self.plot_title_x.get(), 0.5)
        ty = safe_float(self.plot_title_y.get(), 1.04)
        xx = safe_float(self.plot_xlabel_x.get(), 0.5)
        xy = safe_float(self.plot_xlabel_y.get(), -0.10)
        yx = safe_float(self.plot_ylabel_x.get(), -0.10)
        yy = safe_float(self.plot_ylabel_y.get(), 0.5)

        for ax in fig.axes:
            try:
                ax.title.set_position((tx, ty))
                ax.xaxis.set_label_coords(xx, xy)
                ax.yaxis.set_label_coords(yx, yy)
            except Exception:
                pass
        try:
            all_text = fig.findobj(match=MplText)
        except Exception:
            all_text = []
        for t in all_text:
            try:
                t.set_fontfamily(family)
                # Keep titles slightly larger; all other text uses base size.
                if any(t is ax.title for ax in fig.axes):
                    t.set_fontsize(title_size)
                elif t.axes is not None and any(t in ax.get_legend().get_texts() for ax in fig.axes if ax.get_legend() is not None):
                    t.set_fontsize(max(6.0, base_size * 0.85))
                else:
                    t.set_fontsize(base_size)
                t.set_fontweight(weight)
                col = str(t.get_color()).lower()
                if col in {"black", "#000000", "#111827", "#0f172a"}:
                    t.set_color(text_color)
                t.set_picker(6)
            except Exception:
                pass
        for ax in fig.axes:
            try:
                leg = ax.get_legend()
            except Exception:
                leg = None
            if leg is not None:
                try:
                    leg.set_picker(True)
                    leg.get_frame().set_picker(True)
                    for lt in leg.get_texts():
                        lt.set_picker(6)
                    if leg.get_title() is not None:
                        leg.get_title().set_picker(6)
                except Exception:
                    pass

    def _finalize_plot(self, plot_canvas: PlotCanvas, *, tight: bool = True,
                       left: float = 0.10, right: float = 0.97,
                       bottom: float = 0.12, top: float = 0.92) -> None:
        fig = plot_canvas.figure
        self._apply_publication_style(fig)
        l = self._auto_margin(self.plot_left, left)
        r = self._auto_margin(self.plot_right, right)
        b = self._auto_margin(self.plot_bottom, bottom)
        t = self._auto_margin(self.plot_top, top)
        if l < r and b < t:
            try:
                fig.subplots_adjust(left=l, right=r, bottom=b, top=t)
            except Exception:
                pass
        if tight:
            try:
                fig.tight_layout(pad=1.0)
            except Exception:
                pass
        try:
            plot_canvas.set_text_style_editor_callback(self._open_publication_style_dialog_for_artist)
            plot_canvas.set_advanced_text_editor_callback(self._open_publication_style_dialog_for_artist)
        except Exception:
            pass
        # Always enable per-artist editing.  The checkbox is kept as a user-facing
        # preference, but the final publication workflow requires every title,
        # axis label, tick label, colour-bar label/tick and legend entry to be
        # double-click editable and draggable on every plot.
        plot_canvas.enable_artist_dragging(True)
        plot_canvas.draw()

    def _on_texture_family_change(self, *_: Any) -> None:
        """Switch PF/IPF default lists when the user changes lattice family."""
        if not hasattr(self, "style_family"):
            return
        fam = self.style_family.get().strip().lower()
        if fam in {"hcp", "hex", "hexagonal"}:
            self.style_poles.set("0002; 10-10; 11-20")
            self.style_directions.set("ND")
        else:
            self.style_poles.set("100; 110; 111")
            self.style_directions.set("ND")

    # ---------------------------------------------------------------- Defaults
    def _load_defaults_from_data_dir(self) -> None:
        # Honour prefs if pointing to an existing case
        candidates: List[Path] = []
        if "base_dir" in self.prefs:
            try:
                candidates.append(Path(self.prefs["base_dir"]).expanduser())
            except (TypeError, ValueError):
                pass
        # Optional override for batch/CI use; never assume a sandbox path.
        env_dir = os.environ.get("VPSC_DATA_DIR")
        if env_dir:
            candidates.append(Path(env_dir).expanduser())
        candidates.append(Path.cwd())
        chosen = None
        for base in candidates:
            try:
                if (base / "vpsc8.in").exists() or (base / "VPSC8.IN").exists():
                    chosen = base
                    break
            except OSError:
                continue
        if chosen is not None:
            self.state_data.base_dir = chosen
            self.state_data.vpsc_in = Path(self.prefs.get("vpsc_in") or "vpsc8.in")
            if "executable" in self.prefs:
                self.state_data.executable = Path(self.prefs["executable"])
            if "run_root" in self.prefs:
                self.state_data.run_root = Path(self.prefs["run_root"])

        self.var_base.set(str(self.state_data.base_dir))
        self.var_vpsc_in.set(str(self.state_data.vpsc_in))
        self.var_run_root.set(str(self.state_data.run_root))
        if self.state_data.executable and str(self.state_data.executable):
            self.var_exe.set(str(self.state_data.executable))
        self.load_vpsc_in()

    def _on_close(self) -> None:
        # Persist a minimal set of paths
        try:
            self.prefs.update({
                "base_dir": str(self.state_data.base_dir),
                "vpsc_in": str(self.state_data.vpsc_in),
                "executable": str(self.state_data.executable),
                "run_root": str(self.state_data.run_root),
            })
            save_prefs(self.prefs)
        except Exception as e:
            LOG.warning("save_prefs on exit failed: %s", e)
        self.destroy()

    # ---------------------------------------------------------------- Dashboard
    def _build_dashboard_page(self) -> None:
        p = self.pages["Dashboard"]
        ttk.Label(p, text="Dashboard", style="Title.TLabel").pack(
            anchor="w", pady=(4, 12)
        )
        grid = ttk.Frame(p)
        grid.pack(fill="x")
        self.dashboard_cards: Dict[str, ttk.Label] = {}
        for i, title in enumerate(["Regime", "Phases", "Interaction", "Current run"]):
            c = self.card(grid, title)
            c.grid(row=0, column=i, sticky="nsew", padx=5, pady=5)
            val = ttk.Label(c, text="—", style="CardTitle.TLabel",
                             font=("Segoe UI", 18, "bold"))
            val.pack(anchor="w", pady=6)
            self.dashboard_cards[title] = val
            grid.grid_columnconfigure(i, weight=1)
        quick = self.card(
            p, "Quick actions",
            "Workflow: load inputs → edit → run Fortran VPSC → select outputs",
        )
        quick.pack(fill="x", pady=10)
        ttk.Button(quick, text="Load VPSC8.IN", style="Accent.TButton",
                   command=self.load_vpsc_in).pack(side="left", padx=4)
        ttk.Button(quick, text="Go to Single Crystal",
                   command=lambda: self.show_page("Single Crystal")).pack(
            side="left", padx=4
        )
        ttk.Button(quick, text="Go to Results",
                   command=lambda: self.show_page("Results")).pack(
            side="left", padx=4
        )
        self.dashboard_text = TextEditor(p, height=16)
        self.dashboard_text.pack(fill="both", expand=True, pady=8)

    def update_dashboard(self) -> None:
        self.dashboard_cards["Regime"].configure(
            text="VP" if self.vpsc_info.regime == 1 else "EL"
        )
        self.dashboard_cards["Phases"].configure(text=str(self.vpsc_info.nph))
        self.dashboard_cards["Interaction"].configure(
            text=f"{self.vpsc_info.interaction} / neff={self.vpsc_info.neff:g}"
        )
        self.dashboard_cards["Current run"].configure(
            text=self.state_data.last_run_dir.name or "—"
        )
        lines = [
            "VPSC input summary",
            "==================",
            f"Base dir: {self.state_data.base_dir}",
            f"VPSC8.IN: {self.state_data.vpsc_in}",
            f"Regime: {self.vpsc_info.regime}",
            f"Phases: {self.vpsc_info.nph}",
            f"Interaction: {self.vpsc_info.interaction}, neff={self.vpsc_info.neff}",
            f"Errors: {self.vpsc_info.errs}",
            f"itmax: {self.vpsc_info.itmax}",
            "",
            "Phase files:",
        ]
        for ph in self.vpsc_info.phases:
            lines.append(
                f"  Phase {ph.index}: texture={ph.texture_file}, "
                f"crystal={ph.crystal_file}"
            )
        lines.append("")
        proc_summary = ", ".join(
            f"{f}(ivg={iv})"
            for f, iv in zip(self.vpsc_info.process_files,
                             self.vpsc_info.process_ivgvar or [0] * len(self.vpsc_info.process_files))
        ) or "—"
        lines.append(f"Process files: {proc_summary}")
        self.dashboard_text.set("\n".join(lines))
        self.status_label.configure(text=f"Status\nLoaded: {self.state_data.vpsc_in}")

    # ---------------------------------------------------------------- Project
    def _build_project_page(self) -> None:
        p = self.pages["Project"]
        ttk.Label(p, text="Project", style="Title.TLabel").pack(
            anchor="w", pady=(4, 12)
        )
        c = self.card(
            p, "Project paths",
            "Choose the VPSC case folder and executable. "
            "Inputs are copied to an independent run directory."
        )
        c.pack(fill="x")
        self.var_base = tk.StringVar()
        self.var_vpsc_in = tk.StringVar(value="vpsc8.in")
        self.var_exe = tk.StringVar()
        self.var_run_root = tk.StringVar(value="vpsc_runs")
        self._path_row(c, "Base directory", self.var_base, self.browse_base_dir)
        self._path_row(c, "VPSC8.IN", self.var_vpsc_in, self.browse_vpsc_in)
        self._path_row(c, "VPSC executable", self.var_exe, self.browse_exe)
        self._path_row(c, "Run output root", self.var_run_root, self.browse_run_root)
        actions = ttk.Frame(c, style="Panel.TFrame")
        actions.pack(fill="x", pady=(8, 0))
        ttk.Button(actions, text="Apply paths", style="Accent.TButton",
                   command=self.apply_project_paths).pack(side="left", padx=4)
        ttk.Button(actions, text="Load VPSC8.IN",
                   command=self.load_vpsc_in).pack(side="left", padx=4)
        ttk.Button(actions, text="Open base folder",
                   command=lambda: self.open_folder(self.state_data.base_dir)).pack(
            side="left", padx=4
        )
        self.project_preview = TextEditor(p, height=24)
        self.project_preview.pack(fill="both", expand=True, pady=10)

    def browse_base_dir(self) -> None:
        d = filedialog.askdirectory(initialdir=str(self.state_data.base_dir))
        if d:
            self.var_base.set(d)

    def browse_vpsc_in(self) -> None:
        f = filedialog.askopenfilename(
            initialdir=str(self.state_data.base_dir),
            filetypes=[("VPSC input", "*.in *.IN"), ("All", "*")],
        )
        if f:
            p = Path(f)
            self.var_base.set(str(p.parent))
            self.var_vpsc_in.set(p.name)

    def browse_exe(self) -> None:
        f = filedialog.askopenfilename(
            initialdir=str(self.state_data.base_dir),
            filetypes=[("Executable", "*.exe *"), ("All", "*")],
        )
        if f:
            self.var_exe.set(f)

    def browse_run_root(self) -> None:
        d = filedialog.askdirectory(initialdir=str(self.state_data.base_dir))
        if d:
            self.var_run_root.set(d)

    def apply_project_paths(self) -> None:
        self.apply_project_paths_no_reload()
        self.load_vpsc_in()

    def apply_project_paths_no_reload(self) -> None:
        try:
            self.state_data.base_dir = Path(self.var_base.get()).expanduser().resolve()
            self.state_data.vpsc_in = Path(self.var_vpsc_in.get())
            self.state_data.executable = (
                Path(self.var_exe.get()) if self.var_exe.get().strip() else Path("")
            )
            self.state_data.run_root = Path(self.var_run_root.get())
        except (OSError, RuntimeError) as e:
            LOG.warning("apply_project_paths_no_reload: %s", e)

    # ---------------------------------------------------------------- VPSC8.IN
    def _build_vpsc_in_page(self) -> None:
        p = self.pages["VPSC8.IN"]
        ttk.Label(p, text="VPSC8.IN", style="Title.TLabel").pack(
            anchor="w", pady=(4, 8)
        )
        bar = ttk.Frame(p)
        bar.pack(fill="x", pady=(0, 8))
        ttk.Button(bar, text="Reload", command=self.load_vpsc_in).pack(
            side="left", padx=3
        )
        ttk.Button(bar, text="Save and overwrite", style="Accent.TButton",
                   command=self.save_vpsc_in).pack(side="left", padx=3)
        self.in_editor = TextEditor(p, height=20)
        self.in_editor.pack(fill="both", expand=True)

    def load_vpsc_in(self) -> None:
        self.apply_project_paths_no_reload()
        path = path_rel(self.state_data.base_dir, self.state_data.vpsc_in)
        if not path.exists():
            self.project_preview.set(f"Could not find {path}")
            return
        self.in_editor.set(read_text(path))
        self.vpsc_info = PARSE_CACHE.get_or_compute(path, parse_vpsc8_in)
        self.project_preview.set(read_text(path))
        self.update_dashboard()
        self.update_solver_text()
        self.refresh_phase_related_paths()

    def save_vpsc_in(self) -> None:
        path = path_rel(self.state_data.base_dir, self.state_data.vpsc_in)
        if not messagebox.askyesno(
            "Overwrite VPSC8.IN", f"Save changes to\n{path}?"
        ):
            return
        write_text(path, self.in_editor.get())
        PARSE_CACHE.clear()  # invalidate
        self.vpsc_info = parse_vpsc8_in(path)
        self.update_dashboard()

    def refresh_phase_related_paths(self) -> None:
        if self.vpsc_info.phases:
            ph = self.vpsc_info.phases[0]
            self.var_sx_path.set(ph.crystal_file)
            self.var_texture_path.set(ph.texture_file)
        if self.vpsc_info.process_files:
            self.var_process_path.set(self.vpsc_info.process_files[0])
        self.load_single_crystal()
        self.load_texture()
        self.load_process()

    # ----------------------------------------------------------- Single Crystal
    def _build_single_crystal_page(self) -> None:
        p = self.pages["Single Crystal"]
        ttk.Label(p, text="Single Crystal", style="Title.TLabel").pack(
            anchor="w", pady=(4, 8)
        )
        bar = ttk.Frame(p)
        bar.pack(fill="x", pady=(0, 6))
        self.var_sx_path = tk.StringVar()
        ttk.Label(bar, text="SX file", style="Muted.TLabel").pack(side="left", padx=(0, 4))
        ttk.Entry(bar, textvariable=self.var_sx_path, width=42).pack(side="left", padx=4)
        ttk.Button(bar, text="Browse", command=self.browse_sx).pack(side="left", padx=2)
        ttk.Button(bar, text="Reload", command=self.load_single_crystal).pack(
            side="left", padx=2
        )

        self.sx_summary = tk.StringVar(value="No SX loaded.")
        ttk.Label(p, textvariable=self.sx_summary, style="Muted.TLabel").pack(
            anchor="w", pady=(0, 6)
        )

        nb = ttk.Notebook(p)
        nb.pack(fill="both", expand=True)

        # ---- Tab: Elastic constants 6x6
        tab_el = ttk.Frame(nb, padding=8)
        nb.add(tab_el, text="Elastic constants")
        ttk.Label(tab_el, text="C_ij (6×6, in the same units as the .sx file — e.g. MPa)",
                  style="Muted.TLabel").pack(anchor="w")
        self.elastic_editor = MatrixEditor(tab_el, rows=6, cols=6, width=10)
        self.elastic_editor.pack(fill="x", pady=4)
        btns_el = ttk.Frame(tab_el)
        btns_el.pack(fill="x", pady=4)
        ttk.Button(btns_el, text="Save elastic matrix", style="Accent.TButton",
                   command=self.save_sx_elastic).pack(side="left", padx=2)

        # ---- Tab: Slip parameters
        tab_p = ttk.Frame(nb, padding=8)
        nb.add(tab_p, text="Parameters")
        ttk.Label(tab_p, text="Per-line numeric parameters parsed from the SX file. "
                  "Double-click the Values column to edit.",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 4))
        cols = ("line", "category", "comment", "values")
        self.sx_param_tree = ttk.Treeview(tab_p, columns=cols, show="headings",
                                          height=14)
        for c, w in zip(cols, (60, 110, 360, 360)):
            self.sx_param_tree.heading(c, text=c.title())
            self.sx_param_tree.column(c, width=w, anchor="w")
        self.sx_param_tree.pack(fill="both", expand=True)
        self.sx_param_tree.bind("<Double-1>", self.edit_sx_param_cell)
        btns_p = ttk.Frame(tab_p)
        btns_p.pack(fill="x", pady=4)
        ttk.Button(btns_p, text="Save parameter changes", style="Accent.TButton",
                   command=self.save_sx_params).pack(side="left", padx=2)
        self._sx_param_changes: Dict[int, str] = {}

        # ---- Tab: Theory notes
        tab_th = ttk.Frame(nb, padding=8)
        nb.add(tab_th, text="Theory")
        self.sx_notes = TextEditor(tab_th, height=18)
        self.sx_notes.pack(fill="both", expand=True)
        self.sx_notes.text.configure(state="disabled")

        # ---- Tab: Raw editor
        tab_raw = ttk.Frame(nb, padding=8)
        nb.add(tab_raw, text="Raw")
        self.sx_raw = TextEditor(tab_raw, height=18)
        self.sx_raw.pack(fill="both", expand=True)
        ttk.Button(tab_raw, text="Save raw text", style="Accent.TButton",
                   command=self.save_sx_raw).pack(anchor="e", pady=4)

    def browse_sx(self) -> None:
        f = filedialog.askopenfilename(
            initialdir=str(self.state_data.base_dir),
            filetypes=[("SX files", "*.sx *.SX"), ("All", "*")],
        )
        if f:
            self.var_sx_path.set(
                str(Path(f).relative_to(self.state_data.base_dir))
                if Path(f).is_absolute() and str(Path(f)).startswith(str(self.state_data.base_dir))
                else f
            )
            self.load_single_crystal()

    def sx_path(self) -> Path:
        return path_rel(self.state_data.base_dir, self.var_sx_path.get())

    def load_single_crystal(self) -> None:
        if not self.var_sx_path.get().strip():
            return
        path = self.sx_path()
        if not path.is_file():
            self.sx_summary.set(f"SX file not found: {path}")
            return
        try:
            self.sx_info = PARSE_CACHE.get_or_compute(path, parse_sx)
        except Exception as e:
            LOG.warning("parse_sx failed: %s", e)
            self.sx_info = None
            self.sx_summary.set(f"Parse failed: {e}")
            return
        info = self.sx_info
        self.sx_summary.set(
            f"{path.name}  ·  crystal={info.crystal_class}  ·  "
            f"family={info.family}  ·  {len(info.params)} numeric lines  ·  "
            f"elastic block @ line {info.elastic_start if info.elastic_start >= 0 else '?'}"
        )
        # Use SX symmetry to drive texture plotting defaults.
        if hasattr(self, "style_family"):
            is_hcp = info.family.lower().startswith("h")
            self.style_family.set("hcp" if is_hcp else "cubic")
            if is_hcp:
                self.style_poles.set("0002; 10-10")
                self.style_directions.set("ND")
            else:
                self.style_poles.set("100; 110; 111")
            if getattr(self, "texture_data", None) is not None and hasattr(self, "texture_canvas"):
                self.after(80, self.draw_texture_studio_pf)
        self.sx_raw.set(read_text(path))
        # Populate elastic
        self.elastic_editor.set(info.elastic_matrix)
        # Populate parameter tree
        self.populate_sx_params()
        # Theory notes
        self.sx_notes.text.configure(state="normal")
        self.sx_notes.set(build_sx_theory_notes(info))
        self.sx_notes.text.configure(state="disabled")

    def populate_sx_params(self) -> None:
        self.sx_param_tree.delete(*self.sx_param_tree.get_children())
        self._sx_param_changes.clear()
        if not self.sx_info:
            return
        for pl in self.sx_info.params:
            self.sx_param_tree.insert(
                "", "end", iid=str(pl.line_no),
                values=(pl.line_no, pl.category,
                        pl.comment[:80], " ".join(pl.values))
            )

    def edit_sx_param_cell(self, event: Any) -> None:
        item = self.sx_param_tree.identify_row(event.y)
        col = self.sx_param_tree.identify_column(event.x)
        if not item or col != "#4":
            return
        cur = self.sx_param_tree.item(item, "values")[3]
        new = simpledialog.askstring(
            "Edit values", "Space-separated numeric values:",
            initialvalue=cur, parent=self
        )
        if new is None:
            return
        line_idx = int(item)
        self._sx_param_changes[line_idx] = new
        vals = self.sx_param_tree.item(item, "values")
        self.sx_param_tree.item(item, values=(vals[0], vals[1], vals[2], new))

    def save_sx_params(self) -> None:
        if not self.sx_info or not self._sx_param_changes:
            messagebox.showinfo("No changes", "No parameter edits to save.")
            return
        path = self.sx_path()
        if not messagebox.askyesno(
            "Overwrite SX",
            f"Apply {len(self._sx_param_changes)} parameter changes to\n{path}?"
        ):
            return
        try:
            apply_sx_param_changes(path, self._sx_param_changes)
            PARSE_CACHE.clear()
            self._sx_param_changes.clear()
            self.load_single_crystal()
        except OSError as e:
            messagebox.showerror("Save failed", str(e))

    def save_sx_elastic(self) -> None:
        if not self.sx_info or self.sx_info.elastic_start < 0:
            messagebox.showwarning("No elastic block",
                                   "Cannot find elastic matrix in SX file.")
            return
        path = self.sx_path()
        if not messagebox.askyesno(
            "Overwrite SX",
            f"Apply elastic matrix changes to\n{path}?"
        ):
            return
        try:
            apply_sx_elastic_changes(path, self.sx_info.elastic_start,
                                     self.elastic_editor.get())
            PARSE_CACHE.clear()
            self.load_single_crystal()
        except OSError as e:
            messagebox.showerror("Save failed", str(e))

    def save_sx_raw(self) -> None:
        path = self.sx_path()
        if not messagebox.askyesno(
            "Overwrite SX", f"Save raw SX text to\n{path}?"
        ):
            return
        write_text(path, self.sx_raw.get())
        PARSE_CACHE.clear()
        self.load_single_crystal()

    # --------------------------------------------------------------- Texture
    def _build_texture_page(self) -> None:
        p = self.pages["Texture"]
        ttk.Label(p, text="Texture / Pole-Figure Studio", style="Title.TLabel").pack(
            anchor="w", pady=(4, 8)
        )
        bar = ttk.Frame(p)
        bar.pack(fill="x", pady=(0, 6))
        self.var_texture_path = tk.StringVar()
        ttk.Label(bar, text="Texture file", style="Muted.TLabel").pack(
            side="left", padx=(0, 4)
        )
        ttk.Entry(bar, textvariable=self.var_texture_path, width=52).pack(
            side="left", fill="x", expand=True, padx=4
        )
        ttk.Button(bar, text="Browse", command=self.browse_texture).pack(side="left", padx=2)
        ttk.Button(bar, text="Reload", command=self.load_texture).pack(side="left", padx=2)
        ttk.Button(bar, text="Draw PF", style="Accent.TButton",
                   command=self.draw_texture_studio_pf).pack(side="left", padx=2)
        ttk.Button(bar, text="Draw IPF",
                   command=self.draw_texture_studio_ipf).pack(side="left", padx=2)
        ttk.Button(bar, text="Export PNG",
                   command=self.export_texture_figure).pack(side="left", padx=2)

        self.texture_summary = tk.StringVar(value="No texture loaded.")
        ttk.Label(p, textvariable=self.texture_summary, style="Muted.TLabel").pack(
            anchor="w", pady=(0, 6)
        )

        # Layout: data preview | large drawing canvas | scrollable style panel.
        paned = ttk.PanedWindow(p, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, padding=4)
        paned.add(left, weight=1)
        left_nb = ttk.Notebook(left)
        left_nb.pack(fill="both", expand=True)
        tab_table = ttk.Frame(left_nb, padding=4)
        tab_raw = ttk.Frame(left_nb, padding=4)
        left_nb.add(tab_table, text="Euler table")
        left_nb.add(tab_raw, text="Raw editor")

        cols = ("idx", "phi1", "Phi", "phi2", "w")
        self.texture_table = ttk.Treeview(tab_table, columns=cols, show="headings", height=18)
        for c, w in zip(cols, (60, 86, 86, 86, 80)):
            self.texture_table.heading(c, text=c)
            self.texture_table.column(c, width=w, anchor="e")
        ytab = ttk.Scrollbar(tab_table, orient="vertical", command=self.texture_table.yview)
        self.texture_table.configure(yscrollcommand=ytab.set)
        self.texture_table.grid(row=0, column=0, sticky="nsew")
        ytab.grid(row=0, column=1, sticky="ns")
        tab_table.grid_rowconfigure(0, weight=1)
        tab_table.grid_columnconfigure(0, weight=1)

        ttk.Label(tab_raw, text="Raw texture file. You may edit and overwrite it.",
                  style="Muted.TLabel").pack(anchor="w")
        self.texture_raw = TextEditor(tab_raw, height=16)
        self.texture_raw.pack(fill="both", expand=True, pady=(4, 4))
        ttk.Button(tab_raw, text="Save raw text", style="Accent.TButton",
                   command=self.save_texture_raw).pack(anchor="e", pady=4)

        center = ttk.Frame(paned, padding=4)
        paned.add(center, weight=4)
        canvas_header = ttk.Frame(center)
        canvas_header.pack(fill="x", pady=(0, 4))
        ttk.Label(canvas_header, text="PF / IPF preview", style="CardTitle.TLabel").pack(side="left")
        ttk.Label(canvas_header,
                  text="Use the right panel to change colour, mode, levels, grid and labels.",
                  style="Muted.TLabel").pack(side="left", padx=10)
        self.texture_canvas = PlotCanvas(center, figsize=(8.8, 6.6))
        self.texture_canvas.pack(fill="both", expand=True)

        right_scroll = ScrollableFrame(paned, width=350)
        paned.add(right_scroll, weight=0)
        self._build_texture_style_controls(right_scroll.inner)

    def _ensure_texture_style_vars(self) -> None:
        """Create shared PF/IPF style variables once.

        Both the Texture page and Results page bind to these same variables so
        post-processing texture plots can be styled without leaving Results.
        """
        if hasattr(self, "style_family"):
            return
        self.style_family = tk.StringVar(value="cubic")
        self.style_proj = tk.StringVar(value="equal-area")
        self.style_poles = tk.StringVar(value="100; 110; 111")
        self.style_directions = tk.StringVar(value="ND")
        self.style_texture_mode = tk.StringVar(value="fill")
        self.style_cmap = tk.StringVar(value="turbo")
        self.style_marker = tk.StringVar(value="o")
        self.style_marker_color = tk.StringVar(value="Navy")
        self.style_point_color = tk.StringVar(value="Blue")
        self.style_marker_size = tk.StringVar(value="6")
        self.style_alpha = tk.StringVar(value="0.78")
        self.style_grid_n = tk.StringVar(value="100")
        self.style_sigma = tk.StringVar(value="2.5")
        self.style_levels = tk.StringVar(value="9")
        self.style_manual_levels = tk.StringVar(value="")
        self.style_contour_color = tk.StringVar(value="Black")
        self.style_contour_style = tk.StringVar(value="-")
        self.style_contour_width = tk.StringVar(value="0.9")
        self.style_colorbar = tk.BooleanVar(value=True)
        self.style_grid_show = tk.BooleanVar(value=True)
        # texture3-compatible controls
        self.style_dph = tk.StringVar(value="7.5")
        self.style_dth = tk.StringVar(value="7.5")
        self.style_n_rim = tk.StringVar(value="2")
        self.style_mn = tk.StringVar(value="")
        self.style_mx = tk.StringVar(value="")
        self.style_log_levels = tk.BooleanVar(value=False)
        self.style_ix = tk.StringVar(value="RD")
        self.style_iy = tk.StringVar(value="TD")
        self.style_rot = tk.StringVar(value="0")
        self._ensure_publication_style_vars()
        try:
            self.style_family.trace_add("write", self._on_texture_family_change)
        except Exception:
            pass

    def _build_texture_style_controls(self, master: Any, title: str = "Texture3-style PF/IPF controls") -> None:
        self._ensure_texture_style_vars()
        c = self.card(master, title,
                      "Shared by pre-processing Texture Studio and Results. Colours are selected by name or Pick.")
        c.pack(fill="x", padx=4, pady=4)
        self._combo_row(c, "Family", self.style_family, ("cubic", "hcp"))
        self._combo_row(c, "Projection", self.style_proj,
                        ("equal-area", "stereographic"))
        self._entry_row(c, "PF poles", self.style_poles)
        self._entry_row(c, "IPF directions", self.style_directions)
        self._combo_row(c, "Mode", self.style_texture_mode,
                        ("line", "fill", "dot", "dotc", "fill+dot",
                         "contour", "contourf", "scatter", "density", "both"))
        self._combo_row(c, "Colormap", self.style_cmap, tuple(CMAP_CHOICES))
        self._combo_row(c, "Marker", self.style_marker, tuple(MARKER_CHOICES))
        self._color_row(c, "Point color", self.style_point_color)
        self._color_row(c, "Edge color", self.style_marker_color)
        self._entry_row(c, "Marker size", self.style_marker_size)
        self._entry_row(c, "Alpha", self.style_alpha)
        self._color_row(c, "Contour color", self.style_contour_color)
        self._combo_row(c, "Contour style", self.style_contour_style, tuple(LINESTYLE_CHOICES))
        self._entry_row(c, "Contour width", self.style_contour_width)
        self._entry_row(c, "Grid N", self.style_grid_n)
        self._entry_row(c, "Smoothing σ", self.style_sigma)
        self._entry_row(c, "Levels", self.style_levels)
        self._entry_row(c, "Manual levels", self.style_manual_levels)
        self._entry_row(c, "dphi", self.style_dph)
        self._entry_row(c, "dtheta", self.style_dth)
        self._entry_row(c, "n_rim", self.style_n_rim)
        self._entry_row(c, "min level", self.style_mn)
        self._entry_row(c, "max level", self.style_mx)
        self._combo_row(c, "axis ix", self.style_ix, ("RD", "TD", "ND", "-RD", "-TD", "-ND"))
        self._combo_row(c, "axis iy", self.style_iy, ("RD", "TD", "ND", "-RD", "-TD", "-ND"))
        self._entry_row(c, "rot deg", self.style_rot)
        ttk.Checkbutton(c, text="show colorbar", variable=self.style_colorbar).pack(anchor="w", pady=(4, 0))
        ttk.Checkbutton(c, text="show grid / RD-TD rim", variable=self.style_grid_show).pack(anchor="w", pady=(2, 0))
        ttk.Checkbutton(c, text="log levels", variable=self.style_log_levels).pack(anchor="w", pady=(2, 0))
        self._build_publication_style_controls(master)

    def current_texture_style(self) -> PlotStyle:
        self._ensure_texture_style_vars()
        style = PlotStyle(
            cmap=self.style_cmap.get(),
            marker=self.style_marker.get(),
            marker_color=resolve_color(self.style_marker_color.get()),
            point_color=resolve_color(self.style_point_color.get()),
            marker_size=safe_float(self.style_marker_size.get(), 6.0),
            point_size=safe_float(self.style_marker_size.get(), 6.0) * 2.0,
            alpha=max(0.0, min(1.0, safe_float(self.style_alpha.get(), 0.78))),
            contour_color=resolve_color(self.style_contour_color.get()),
            contour_style=self.style_contour_style.get() or "-",
            contour_width=safe_float(self.style_contour_width.get(), 0.9),
            bins=int(safe_float(self.style_grid_n.get(), 100)),
            smooth=safe_float(self.style_sigma.get(), 2.5),
            levels=int(safe_float(self.style_levels.get(), 9)),
            manual_levels=self.style_manual_levels.get(),
            texture_mode=self.style_texture_mode.get(),
            projection=(self.style_proj.get() or "equal-area"),
            colorbar=bool(self.style_colorbar.get()),
            grid=bool(self.style_grid_show.get()),
        )
        # Dynamic texture3 options.  Stored on the style object to keep the GUI
        # and renderer decoupled.
        style.texture3_dph = safe_float(self.style_dph.get(), 7.5)
        style.texture3_dth = safe_float(self.style_dth.get(), 7.5)
        style.texture3_n_rim = int(safe_float(self.style_n_rim.get(), 2))
        style.texture3_mn = None if not self.style_mn.get().strip() else safe_float(self.style_mn.get())
        style.texture3_mx = None if not self.style_mx.get().strip() else safe_float(self.style_mx.get())
        style.texture3_log = bool(self.style_log_levels.get())
        style.texture3_ix = self.style_ix.get() or "RD"
        style.texture3_iy = self.style_iy.get() or "TD"
        style.texture3_rot = safe_float(self.style_rot.get(), 0.0)
        return style

    def browse_texture(self) -> None:
        f = filedialog.askopenfilename(
            initialdir=str(self.state_data.base_dir),
            filetypes=[("Texture files", "*.tex *.TEX *.txt"), ("All", "*")],
        )
        if f:
            self.var_texture_path.set(f)
            self.load_texture()

    def texture_path(self) -> Path:
        return path_rel(self.state_data.base_dir, self.var_texture_path.get())

    def load_texture(self) -> None:
        if not self.var_texture_path.get().strip():
            return
        path = self.texture_path()
        if not path.is_file():
            self.texture_summary.set(f"Texture not found: {path}")
            return
        try:
            self.texture_data = PARSE_CACHE.get_or_compute(path, parse_texture)
        except Exception as e:
            LOG.warning("parse_texture failed: %s", e)
            self.texture_summary.set(f"Parse failed: {e}")
            return
        t = self.texture_data
        self.texture_summary.set(
            f"{path.name}  ·  {t.n_grains} grains  ·  convention={t.convention}"
        )
        # Bulk update treeview: clear, then insert in a single batch
        self.texture_table.delete(*self.texture_table.get_children())
        eulers = t.eulers
        weights = t.weights
        max_rows = min(2000, len(eulers))
        for i in range(max_rows):
            self.texture_table.insert(
                "", "end",
                values=(i + 1,
                        f"{eulers[i, 0]:.3f}",
                        f"{eulers[i, 1]:.3f}",
                        f"{eulers[i, 2]:.3f}",
                        f"{weights[i]:.5f}")
            )
        self.texture_raw.set(read_text(path))
        # Show a real PF preview immediately when the texture is loaded.
        if hasattr(self, "texture_canvas"):
            self.after(80, self.draw_texture_studio_pf)

    def save_texture_raw(self) -> None:
        path = self.texture_path()
        if not messagebox.askyesno(
            "Overwrite texture", f"Save raw texture to\n{path}?"
        ):
            return
        write_text(path, self.texture_raw.get())
        PARSE_CACHE.clear()
        self.load_texture()

    def _texture_for_plot(self) -> Optional[TextureData]:
        if self.texture_data is None:
            messagebox.showinfo("No texture", "Load a texture file first.")
            return None
        return self.texture_data

    def draw_texture_studio_pf(self) -> None:
        t = self._texture_for_plot()
        if t is None:
            return
        style = self.current_texture_style()
        poles = parse_poles(self.style_poles.get())
        if not poles:
            messagebox.showinfo("PF poles", "Enter at least one Miller indices triplet, e.g. 100 110 111.")
            return
        fig = self.texture_canvas.figure
        draw_pf_ipf_figure(fig, t, style, kind="pf",
                           family=self.style_family.get(),
                           projection=self.style_proj.get(),
                           items=poles)
        self._finalize_plot(self.texture_canvas, tight=True, left=0.06, right=0.98, bottom=0.07, top=0.93)

    def draw_texture_studio_ipf(self) -> None:
        t = self._texture_for_plot()
        if t is None:
            return
        style = self.current_texture_style()
        dirs = [d.strip().upper() for d in re.split(r"[\s,]+",
                self.style_directions.get()) if d.strip()]
        if not dirs:
            dirs = ["ND"]
        fig = self.texture_canvas.figure
        draw_pf_ipf_figure(fig, t, style, kind="ipf",
                           family=self.style_family.get(),
                           projection=self.style_proj.get(),
                           items=dirs)
        self._finalize_plot(self.texture_canvas, tight=True, left=0.06, right=0.98, bottom=0.07, top=0.93)

    def _save_figure_with_dialog(self, fig: Figure, default_name: str = "vpsc_figure.png") -> None:
        f = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("TIFF image", "*.tif *.tiff"),
                ("JPEG image", "*.jpg *.jpeg"),
                ("PDF vector", "*.pdf"),
                ("SVG vector", "*.svg"),
                ("All files", "*.*"),
            ],
        )
        if not f:
            return
        dpi0 = int(self.prefs.get("export_dpi", 300)) if isinstance(self.prefs, dict) else 300
        dpi = simpledialog.askinteger(
            "Export resolution",
            "DPI for bitmap export (PNG/TIFF/JPEG):",
            initialvalue=dpi0,
            minvalue=72,
            maxvalue=1200,
            parent=self,
        )
        if dpi is None:
            return
        self.prefs["export_dpi"] = int(dpi)
        suffix = Path(f).suffix.lower()
        save_kwargs: Dict[str, Any] = {"dpi": int(dpi), "bbox_inches": "tight", "facecolor": "white"}
        if suffix in {".jpg", ".jpeg"}:
            save_kwargs["pil_kwargs"] = {"quality": 95}
        fig.savefig(f, **save_kwargs)
        messagebox.showinfo("Saved", f"Figure saved to\n{f}\nResolution: {dpi} DPI")

    def export_texture_figure(self) -> None:
        self._save_figure_with_dialog(self.texture_canvas.figure, "vpsc_texture.png")

    # ------------------------------------------------------------ Process / BC
    def _build_process_page(self) -> None:
        p = self.pages["Process / BC"]
        ttk.Label(p, text="Process / Boundary conditions", style="Title.TLabel").pack(
            anchor="w", pady=(4, 8)
        )
        bar = ttk.Frame(p)
        bar.pack(fill="x", pady=(0, 6))
        self.var_process_path = tk.StringVar()
        ttk.Label(bar, text="Process file", style="Muted.TLabel").pack(
            side="left", padx=(0, 4)
        )
        ttk.Entry(bar, textvariable=self.var_process_path, width=42).pack(
            side="left", padx=4
        )
        ttk.Button(bar, text="Browse", command=self.browse_process).pack(
            side="left", padx=2
        )
        ttk.Button(bar, text="Reload", command=self.load_process).pack(
            side="left", padx=2
        )

        paned = ttk.PanedWindow(p, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, padding=8)
        paned.add(left, weight=1)

        nb = ttk.Notebook(left)
        nb.pack(fill="both", expand=True)

        # Tab: Step / counters
        tab_s = ttk.Frame(nb, padding=8)
        nb.add(tab_s, text="Step")
        self.var_nsteps = tk.StringVar(value="50")
        self.var_eqincr = tk.StringVar(value="0.01")
        self.var_ictrl = tk.StringVar(value="7")
        self.var_temp = tk.StringVar(value="298.0")
        self._entry_row(tab_s, "Nsteps", self.var_nsteps)
        self._entry_row(tab_s, "EQ increment", self.var_eqincr)
        self._entry_row(tab_s, "ICTRL", self.var_ictrl)
        self._entry_row(tab_s, "Temperature", self.var_temp)

        # Tab: Velocity gradient
        tab_v = ttk.Frame(nb, padding=8)
        nb.add(tab_v, text="Velocity gradient")
        ttk.Label(tab_v, text="iudot (3x3 mask, 1=prescribed)",
                  style="Muted.TLabel").pack(anchor="w")
        self.iudot_editor = MatrixEditor(tab_v, rows=3, cols=3, width=6)
        self.iudot_editor.pack(pady=4)
        ttk.Label(tab_v, text="udot (3x3 strain-rate)", style="Muted.TLabel").pack(
            anchor="w", pady=(8, 0)
        )
        self.udot_editor = MatrixEditor(tab_v, rows=3, cols=3, width=10)
        self.udot_editor.pack(pady=4)
        presets = ttk.Frame(tab_v)
        presets.pack(fill="x", pady=4)
        ttk.Label(presets, text="Preset", style="Muted.TLabel").pack(side="left")
        self.var_velocity_preset = tk.StringVar(value="Rolling / plane strain compression")
        preset_names = list(velocity_gradient_presets().keys())
        ttk.Combobox(presets, textvariable=self.var_velocity_preset,
                     values=preset_names, state="readonly", width=32).pack(side="left", padx=4)
        ttk.Button(presets, text="Apply",
                   command=self.apply_selected_velocity_preset).pack(side="left", padx=2)
        ttk.Button(presets, text="Import FE L history",
                   command=self.import_fe_velocity_history).pack(side="left", padx=2)

        # Tab: Stress
        tab_st = ttk.Frame(nb, padding=8)
        nb.add(tab_st, text="Stress")
        ttk.Label(tab_st, text="iscau (2x3 mask)", style="Muted.TLabel").pack(anchor="w")
        self.iscau_editor = MatrixEditor(tab_st, rows=2, cols=3, width=6)
        self.iscau_editor.pack(pady=4)
        ttk.Label(tab_st, text="scauchy (2x3 stress)", style="Muted.TLabel").pack(
            anchor="w", pady=(8, 0)
        )
        self.scauchy_editor = MatrixEditor(tab_st, rows=2, cols=3, width=10)
        self.scauchy_editor.pack(pady=4)

        actions = ttk.Frame(left)
        actions.pack(fill="x", pady=6)
        ttk.Button(actions, text="Save (structured)", style="Accent.TButton",
                   command=self.save_process_structured).pack(side="left", padx=2)
        ttk.Button(actions, text="Save raw",
                   command=self.save_process_raw).pack(side="left", padx=2)

        # Raw editor on the same left side, smaller
        ttk.Label(left, text="Raw process file (advanced editing)",
                  style="Muted.TLabel").pack(anchor="w", pady=(8, 0))
        self.process_raw = TextEditor(left, height=10)
        self.process_raw.pack(fill="both", expand=True, pady=4)

        # Right: visualisation
        right = ttk.Frame(paned, padding=4)
        paned.add(right, weight=2)
        ttk.Label(right, text="Boundary-condition visualisation",
                  style="CardTitle.TLabel").pack(anchor="w")
        self._ensure_publication_style_vars()
        opt = ttk.Frame(right)
        opt.pack(fill="x", pady=(2, 4))
        ttk.Label(opt, text="Sketch view", style="Muted.TLabel").pack(side="left")
        ttk.Combobox(opt, textvariable=self.bc_view,
                     values=["RD-TD", "RD-ND", "TD-ND"], state="readonly", width=10).pack(side="left", padx=4)
        ttk.Button(opt, text="Apply view", command=self.draw_boundary_visual).pack(side="left", padx=2)
        self._build_publication_style_controls(right, title="BC figure text / layout")
        self.process_canvas = PlotCanvas(right, figsize=(7.8, 6.4))
        self.process_canvas.pack(fill="both", expand=True, pady=4)
        ttk.Button(right, text="Redraw visualisation",
                   command=self.draw_boundary_visual).pack(anchor="e", padx=2)

    def browse_process(self) -> None:
        f = filedialog.askopenfilename(
            initialdir=str(self.state_data.base_dir),
            filetypes=[("Process files", "*.in *.IN *.txt"), ("All", "*")],
        )
        if f:
            self.var_process_path.set(f)
            self.load_process()

    def process_path(self) -> Path:
        return path_rel(self.state_data.base_dir, self.var_process_path.get())

    def load_process(self) -> None:
        if not self.var_process_path.get().strip():
            return
        path = self.process_path()
        if not path.is_file():
            return
        try:
            self.process_info = parse_process(path)
        except Exception as e:
            LOG.warning("parse_process failed: %s", e)
            return
        pi = self.process_info
        self.var_nsteps.set(str(pi.nsteps))
        self.var_eqincr.set(fmt_num(pi.eqincr))
        self.var_ictrl.set(str(pi.ictrl))
        self.var_temp.set(fmt_num(pi.temperature))
        self.iudot_editor.set(pi.iudot.astype(float))
        self.udot_editor.set(pi.udot)
        self.iscau_editor.set(self._stress6_to_2x3(pi.iscau.astype(float)))
        self.scauchy_editor.set(self._stress6_to_2x3(pi.scauchy))
        self.process_raw.set(read_text(path))
        self.draw_boundary_visual()

    @staticmethod
    def _stress6_to_2x3(vec6: np.ndarray) -> np.ndarray:
        v = np.asarray(vec6, dtype=float).ravel()
        # Voigt order 11,22,33,23,13,12 -> two-row 2x3 grid
        out = np.zeros((2, 3))
        out[0, :] = v[0:3]
        out[1, :] = v[3:6]
        return out

    @staticmethod
    def _matrix_2x3_to_6(m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=float)
        return np.concatenate([m[0, :], m[1, :]])

    def apply_selected_velocity_preset(self) -> None:
        """Apply a preset L, mask and stress-control state to the editors."""
        name = self.var_velocity_preset.get()
        preset = velocity_gradient_presets().get(name)
        if not preset:
            return
        self.iudot_editor.set(preset["iudot"])
        self.udot_editor.set(preset["udot"])
        self.iscau_editor.set(self._stress6_to_2x3(preset["iscau"]))
        self.scauchy_editor.set(self._stress6_to_2x3(preset["scauchy"]))
        if self.process_info is not None:
            self.process_info.variable_history = False
            self.process_info.fe_history = np.zeros((0, 11), dtype=float)
        self.draw_boundary_visual()

    def import_fe_velocity_history(self) -> None:
        """Convert FE/VPSC7/VPSC8 velocity-gradient histories to VPSC8 format."""
        src = filedialog.askopenfilename(
            initialdir=str(self.state_data.base_dir),
            filetypes=[
                ("Velocity-gradient history", "*.txt *.dat *.DAT *.csv *.out"),
                ("All", "*"),
            ],
        )
        if not src:
            return
        try:
            info = read_fe_velocity_history_info(
                Path(src), default_dt=safe_float(self.var_eqincr.get(), 1.0)
            )
            hist = info.history
        except Exception as e:
            messagebox.showerror("Import failed", str(e))
            return

        default_name = Path(src).stem + "_vpsc8_ivgvar1.dat"
        dst = filedialog.asksaveasfilename(
            initialdir=str(self.state_data.base_dir),
            initialfile=default_name,
            defaultextension=".dat",
            filetypes=[("VPSC8 IVGVAR=1 history", "*.dat *.DAT *.txt"), ("All", "*")],
        )
        if not dst:
            return
        # Legacy/FE histories can contain very small L values, which can make
        # VPSC8 stop at the first step (TAUMAX<1e-10).  Rescaling L to Deq~=1
        # and scaling tincr inversely preserves the incremental path L*tincr,
        # but it CHANGES THE STRAIN RATE; for rate-sensitive flow this changes
        # the predicted stresses.  It is therefore offered explicitly rather
        # than applied silently.
        _, stab = stabilise_history_for_vpsc8(hist, enable_auto_rescale=False)
        do_rescale = False
        if stab.get("detected_small_rate"):
            do_rescale = bool(messagebox.askyesno(
                "Very small strain rate detected",
                "The imported velocity-gradient history has a very small "
                f"equivalent rate (median Deq={stab.get('median_deq', 0.0):.3e}).\n\n"
                "VPSC8 may stop at the first step with TAUMAX<1e-10. I can "
                "rescale L to Deq\u22481 and scale tincr inversely so the "
                "incremental path L*tincr is preserved.\n\n"
                "WARNING: this changes the strain RATE. Because the flow rule "
                "is rate sensitive, the predicted stresses will differ from the "
                "unscaled input. Apply the rescaling?",
            ))
        hist_to_write, stab = stabilise_history_for_vpsc8(hist, force=do_rescale)
        try:
            write_variable_velocity_history(Path(dst), hist_to_write)
        except OSError as e:
            messagebox.showerror("Save failed", str(e))
            return

        try:
            rel = str(Path(dst).relative_to(self.state_data.base_dir))
        except ValueError:
            rel = str(dst)
        self.var_process_path.set(rel)
        self.load_process()

        stab_msg = ""
        if stab.get("applied"):
            stab_msg = (
                "\nNumerical stabilisation was applied for VPSC8: L was normalised to "
                f"Deq≈{stab.get('target_deq', 1.0):g} and tincr was scaled so that "
                "the incremental path L*tincr is preserved. "
                f"Original median Deq={stab.get('median_deq', 0.0):.3e}.\n"
            )
        msg = (
            "The selected velocity-gradient history was converted to the VPSC8 "
            "IVGVAR=1 format.\n\n"
            f"{info.summary()}\n"
            f"{stab_msg}\n"
            "Legacy VPSC7 header fields such as ictrl, eqincr and temperature "
            "are kept only as metadata; VPSC8 IVGVAR=1 reads nsteps, Lij and "
            "tincr from the history file.\n"
        )
        if messagebox.askyesno(
            "FE history imported",
            msg + "\nUpdate the first process entry in VPSC8.IN to IVGVAR=1 and this new file?",
        ):
            try:
                vpsc_path = path_rel(self.state_data.base_dir, self.state_data.vpsc_in)
                patch_first_vpsc8_process(vpsc_path, rel, ivgvar=1)
                PARSE_CACHE.clear()
                self.load_vpsc_in()
                messagebox.showinfo("VPSC8.IN updated", f"First process set to IVGVAR=1 and file:\n{rel}")
            except Exception as e:
                messagebox.showwarning("VPSC8.IN not updated", str(e))

    def process_from_widgets(self) -> ProcessInfo:
        pi = self.process_info or ProcessInfo()
        pi.nsteps = int(safe_float(self.var_nsteps.get(), 50))
        pi.eqincr = safe_float(self.var_eqincr.get(), 0.01)
        pi.ictrl = int(safe_float(self.var_ictrl.get(), 7))
        pi.temperature = safe_float(self.var_temp.get(), 298.0)
        pi.iudot = self.iudot_editor.get().astype(int)
        pi.udot = self.udot_editor.get()
        pi.iscau = self._matrix_2x3_to_6(self.iscau_editor.get()).astype(int)
        pi.scauchy = self._matrix_2x3_to_6(self.scauchy_editor.get())
        return pi

    def save_process_structured(self) -> None:
        path = self.process_path()
        if not messagebox.askyesno(
            "Overwrite process",
            f"Save structured edits to\n{path}?"
        ):
            return
        try:
            pi = self.process_from_widgets()
            pi.variable_history = False
            pi.fe_history = np.zeros((0, 11), dtype=float)
            write_process(path, pi)
            self.process_info = pi
            self.process_raw.set(read_text(path))
            self.draw_boundary_visual()
        except OSError as e:
            messagebox.showerror("Save failed", str(e))

    def save_process_raw(self) -> None:
        path = self.process_path()
        if not messagebox.askyesno(
            "Overwrite process",
            f"Save raw process text to\n{path}?"
        ):
            return
        write_text(path, self.process_raw.get())
        self.load_process()

    def draw_boundary_visual(self) -> None:
        """Visualise the current process boundary condition in a balanced 2×2 layout."""
        pi = self.process_from_widgets()
        self.process_info = pi

        L = np.asarray(pi.udot, dtype=float)
        D = 0.5 * (L + L.T)
        W = 0.5 * (L - L.T)
        Ddev = D - np.eye(3) * np.trace(D) / 3.0
        deq = math.sqrt(max(0.0, 2.0 / 3.0 * float(np.sum(Ddev * Ddev))))
        wnorm = float(np.linalg.norm(W))
        trL = float(np.trace(L))
        scale = 0.35 / max(1.0, float(np.max(np.abs(L))))

        fig = self.process_canvas.figure
        fig.clear()
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.08,
                            wspace=0.34, hspace=0.48)
        gs = fig.add_gridspec(2, 2)

        fig.suptitle(
            f"Boundary summary: nsteps={pi.nsteps}, eqincr={pi.eqincr:g}, "
            f"tr(L)={trL:.3g}, Deq={deq:.3g}, |W|={wnorm:.3g}",
            fontsize=11, fontweight="bold", y=0.975,
        )

        def draw_matrix(ax: Any, mat: np.ndarray, title: str,
                        mask: Optional[np.ndarray] = None) -> None:
            vmax = max(1.0e-12, float(np.max(np.abs(mat))))
            ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=10, pad=8)
            ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(["1", "2", "3"], fontsize=8)
            ax.set_yticklabels(["1", "2", "3"], fontsize=8)
            ax.set_xlim(-0.5, 2.5); ax.set_ylim(2.5, -0.5)
            ax.set_aspect("equal")
            for i in range(3):
                for j in range(3):
                    value = mat[i, j]
                    known = True if mask is None else bool(mask[i, j])
                    txt = f"{value:.2g}" if known else f"{value:.2g}\nfree"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                            color="white" if abs(value) > 0.55 * vmax else "black")
            for edge in np.arange(-0.5, 3.5, 1.0):
                ax.axhline(edge, color="white", lw=0.6, alpha=0.7)
                ax.axvline(edge, color="white", lw=0.6, alpha=0.7)
            ax.text(0.02, -0.12, "blue: negative   red: positive",
                    transform=ax.transAxes, fontsize=7, color="#64748b")

        ax1 = fig.add_subplot(gs[0, 0])
        draw_matrix(ax1, L, "Velocity gradient L", pi.iudot)

        ax2 = fig.add_subplot(gs[0, 1])
        draw_matrix(ax2, D, "Symmetric rate D")
        ax2.text(1.02, 0.98,
                 f"D11={D[0,0]:.3g}\nD22={D[1,1]:.3g}\nD33={D[2,2]:.3g}\nDeq={deq:.3g}",
                 transform=ax2.transAxes, va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", alpha=0.95))

        ax3 = fig.add_subplot(gs[1, 0])
        labels = ["σ11", "σ22", "σ33", "σ23", "σ13", "σ12"]
        values = np.asarray(pi.scauchy, dtype=float)
        flags = np.asarray(pi.iscau, dtype=int)
        x = np.arange(len(labels))
        colors = np.where(flags.astype(bool), "#0f766e", "#cbd5e1")
        bars = ax3.bar(x, values, color=colors, edgecolor="#334155", linewidth=0.6)
        ymax = max(1.0, float(np.max(np.abs(values))) * 1.35)
        ax3.set_ylim(-ymax, ymax)
        ax3.axhline(0.0, color="#334155", lw=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=8)
        ax3.set_title("Cauchy stress constraints", fontsize=10, pad=8)
        ax3.grid(axis="y", alpha=0.25)
        for xi, b, f, val in zip(x, bars, flags, values):
            state = "fixed" if f else "free"
            ypos = val + 0.04 * ymax if val >= 0 else val - 0.04 * ymax
            ax3.text(xi, ypos, state, ha="center",
                     va="bottom" if val >= 0 else "top", fontsize=7, rotation=90)
        ax3.text(0.02, 0.96, "teal=fixed  gray=free",
                 transform=ax3.transAxes, va="top", fontsize=8, color="#475569")

        ax4 = fig.add_subplot(gs[1, 1])
        view = getattr(self, "bc_view", tk.StringVar(value="RD-TD")).get().upper()
        if view == "RD-ND":
            idx = [0, 2]; view_label = "RD-ND"
        elif view == "TD-ND":
            idx = [1, 2]; view_label = "TD-ND"
        else:
            idx = [0, 1]; view_label = "RD-TD"
        grid = np.linspace(-1.0, 1.0, 9)
        F2 = np.eye(2) + scale * L[np.ix_(idx, idx)]
        D2 = D[np.ix_(idx, idx)]
        for g in grid:
            line_h = np.vstack([np.linspace(-1, 1, 90), np.full(90, g)])
            line_v = np.vstack([np.full(90, g), np.linspace(-1, 1, 90)])
            qh = F2 @ line_h
            qv = F2 @ line_v
            ax4.plot(line_h[0], line_h[1], color="#dbeafe", lw=0.45, ls="--")
            ax4.plot(line_v[0], line_v[1], color="#dbeafe", lw=0.45, ls="--")
            ax4.plot(qh[0], qh[1], color="#0f766e", lw=0.8)
            ax4.plot(qv[0], qv[1], color="#0f766e", lw=0.8)
        vals, vecs = np.linalg.eigh(D2)
        for val, vec in zip(vals, vecs.T):
            length = 0.55 * (1.0 + min(1.0, abs(float(val))))
            col = "#dc2626" if val >= 0 else "#2563eb"
            ax4.arrow(0, 0, length * vec[0], length * vec[1],
                      head_width=0.045, head_length=0.07, fc=col, ec=col,
                      lw=1.2, length_includes_head=True)
            ax4.arrow(0, 0, -length * vec[0], -length * vec[1],
                      head_width=0.045, head_length=0.07, fc=col, ec=col,
                      lw=1.2, alpha=0.65, length_includes_head=True)
        ax4.axhline(0, color="#94a3b8", lw=0.6)
        ax4.axvline(0, color="#94a3b8", lw=0.6)
        ax4.set_aspect("equal")
        ax4.set_xlim(-1.22, 1.22); ax4.set_ylim(-1.22, 1.22)
        ax4.set_xticks([]); ax4.set_yticks([])
        ax4.set_title(f"2D deformation sketch ({view_label})  scale={scale:.2g}", fontsize=10, pad=8)
        ax4.text(0.02, 0.02,
                 "dashed: reference grid\nsolid: deformed grid\nred/blue: principal rates",
                 transform=ax4.transAxes, fontsize=7, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", alpha=0.92))
        if getattr(pi, "variable_history", False) and getattr(pi, "fe_history", np.zeros((0, 0))).size:
            ax4.text(0.02, 0.98, f"FE history: {pi.fe_history.shape[0]} increments",
                     transform=ax4.transAxes, fontsize=8, va="top",
                     bbox=dict(boxstyle="round,pad=0.25", fc="#ecfeff", ec="#0891b2", alpha=0.95))

        self._finalize_plot(self.process_canvas, tight=False, left=0.07, right=0.97, bottom=0.08, top=0.90)

    # ------------------------------------------------------------------ Solver
    def _build_solver_page(self) -> None:
        p = self.pages["Solver"]
        ttk.Label(p, text="Solver", style="Title.TLabel").pack(
            anchor="w", pady=(4, 8)
        )
        ttk.Label(p, text="Read-only summary of VPSC8.IN solver options. "
                  "Edit on the VPSC8.IN page.", style="Muted.TLabel").pack(
            anchor="w", pady=(0, 6)
        )
        self.solver_text = TextEditor(p, height=24)
        self.solver_text.pack(fill="both", expand=True)
        self.solver_text.text.configure(state="disabled")

    def update_solver_text(self) -> None:
        info = self.vpsc_info
        lines: List[str] = []
        lines.append(f"VPSC8.IN: {self.state_data.vpsc_in}")
        lines.append(f"Regime / IVGVAR: {info.regime}")
        lines.append(f"Number of phases: {len(info.phases)}")
        for i, ph in enumerate(info.phases, 1):
            lines.append(f"  Phase {i}:")
            lines.append(f"    crystal_file = {ph.crystal_file}")
            lines.append(f"    texture_file = {ph.texture_file}")
            lines.append(f"    grain_shape  = {ph.shape_file}")
        lines.append("")
        lines.append(f"Process files ({len(info.process_files)} found):")
        ivs = info.process_ivgvar or [0] * len(info.process_files)
        for pf, iv in zip(info.process_files, ivs):
            lines.append(f"  ivgvar={iv:<3d}  {pf}")
        lines.append("")
        lines.append("Solver options")
        lines.append("--------------")
        lines.append(f"interaction = {info.interaction}   neff = {info.neff}")
        lines.append(f"iupdate     = {info.iupdate}")
        lines.append(f"nneigh      = {info.nneigh}   iflu = {info.iflu}")
        if info.errs:
            lines.append(f"errs        = {info.errs}")
        if info.itmax:
            lines.append(f"itmax       = {info.itmax}")
        self.solver_text.text.configure(state="normal")
        self.solver_text.set("\n".join(lines))
        self.solver_text.text.configure(state="disabled")

    # --------------------------------------------------------------------- Run
    def _build_run_page(self) -> None:
        p = self.pages["Run"]
        ttk.Label(p, text="Run VPSC", style="Title.TLabel").pack(
            anchor="w", pady=(4, 8)
        )
        bar = ttk.Frame(p)
        bar.pack(fill="x", pady=(0, 6))
        ttk.Button(bar, text="Prepare run dir", command=self.prepare_only).pack(
            side="left", padx=2
        )
        ttk.Button(bar, text="Run", style="Accent.TButton",
                   command=self.run_fortran).pack(side="left", padx=2)
        ttk.Button(bar, text="Stop", command=self.stop_run).pack(side="left", padx=2)

        self.run_status = tk.StringVar(value="Idle")
        ttk.Label(p, textvariable=self.run_status, style="Muted.TLabel").pack(
            anchor="w", pady=(2, 4)
        )
        self.run_log = TextEditor(p, height=24)
        self.run_log.pack(fill="both", expand=True)

    def prepare_only(self) -> None:
        try:
            run_dir = prepare_run_dir(self.state_data, self.vpsc_info)
            self.state_data.last_run_dir = run_dir
            self._log(f"[prepare] Run dir staged at {run_dir}")
            self.run_status.set(f"Prepared {run_dir}")
        except (OSError, RuntimeError) as e:
            self._log(f"[prepare] failed: {e}")
            messagebox.showerror("Prepare failed", str(e))

    def run_fortran(self) -> None:
        with self._process_lock:
            if self.process is not None and self.process.poll() is None:
                messagebox.showinfo("Already running",
                                     "A VPSC process is already running.")
                return
        exe = self.state_data.executable
        if not str(exe).strip() or not Path(exe).is_file():
            messagebox.showerror("Executable missing",
                                  f"VPSC executable not found:\n{exe}")
            return
        try:
            run_dir = prepare_run_dir(self.state_data, self.vpsc_info)
            self.state_data.last_run_dir = run_dir
        except (OSError, RuntimeError) as e:
            messagebox.showerror("Prepare failed", str(e))
            return
        self._run_log_error_detected = False
        self._log(f"[run] cwd = {run_dir}")
        self._log(f"[run] exe = {exe}")
        self.run_status.set(f"Running in {run_dir}")

        def worker() -> None:
            try:
                with self._process_lock:
                    self.process = subprocess.Popen(
                        [str(exe)], cwd=str(run_dir),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                    )
                    proc = self.process
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.log_queue.put(line.rstrip())
                proc.wait()
                self.log_queue.put(f"[run] exit code = {proc.returncode}")
            except Exception as e:
                self.log_queue.put(f"[run] worker exception: {e}")
            finally:
                with self._process_lock:
                    self.process = None
                self.log_queue.put("__DONE__")

        threading.Thread(target=worker, daemon=True).start()

    def stop_run(self) -> None:
        with self._process_lock:
            if self.process and self.process.poll() is None:
                try:
                    self.process.terminate()
                    self._log("[run] terminate signal sent")
                except OSError as e:
                    self._log(f"[run] terminate failed: {e}")
            else:
                self._log("[run] no active process")

    def _log(self, msg: str) -> None:
        self.run_log.text.configure(state="normal")
        self.run_log.text.insert("end", msg + "\n")
        self.run_log.text.see("end")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if msg == "__DONE__":
                    self.run_status.set("Finished with warnings" if self._run_log_error_detected else "Done")
                else:
                    low = msg.lower()
                    if ("taumax<" in low or "ieee_invalid" in low or
                            "floating-point exceptions" in low or "stop" == low.strip()):
                        self._run_log_error_detected = True
                    self._log(msg)
        except queue.Empty:
            pass
        self.after(150, self._drain_log_queue)

    # ----------------------------------------------------------------- Results
    def _build_results_page(self) -> None:
        p = self.pages["Results"]
        ttk.Label(p, text="Results", style="Title.TLabel").pack(
            anchor="w", pady=(4, 8)
        )

        bar = ttk.Frame(p)
        bar.pack(fill="x", pady=(0, 6))
        ttk.Label(bar, text="Run directory:", style="Muted.TLabel").pack(side="left")
        self.var_run_dir = tk.StringVar()
        ttk.Entry(bar, textvariable=self.var_run_dir, width=60).pack(
            side="left", padx=4
        )
        ttk.Button(bar, text="Browse", command=self.choose_run_dir).pack(
            side="left", padx=2
        )
        ttk.Button(bar, text="Refresh", command=self.refresh_results_files).pack(
            side="left", padx=2
        )

        paned = ttk.PanedWindow(p, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, padding=4)
        paned.add(left, weight=1)
        ttk.Label(left, text="Kind", style="Muted.TLabel").pack(anchor="w")
        self.var_result_kind = tk.StringVar(value="Stress-strain")
        kinds = ["Stress-strain", "Slip activity", "Relative activity",
                 "R-value", "Young's modulus", "R-value + Young's modulus",
                 "Yield surface (PCYS)",
                 "Texture PF", "Texture IPF"]
        ttk.Combobox(left, textvariable=self.var_result_kind, values=kinds,
                     state="readonly").pack(fill="x", pady=2)
        # Stress unit selector: VPSC writes stress quantities in the unit of the
        # .sx elastic constants (MPa here).  This lets the user view stress,
        # Young's modulus and PCYS axes in MPa or GPa without re-running.
        ttk.Label(left, text="Stress unit", style="Muted.TLabel").pack(anchor="w", pady=(6, 0))
        self.var_stress_unit = tk.StringVar(value="MPa")
        ttk.Combobox(left, textvariable=self.var_stress_unit, values=["MPa", "GPa"],
                     state="readonly", width=8).pack(fill="x", pady=2)
        ttk.Label(left, text="Available files", style="Muted.TLabel").pack(
            anchor="w", pady=(8, 0)
        )
        self.results_files = ttk.Treeview(left, columns=("name", "size"),
                                          show="headings", height=14)
        self.results_files.heading("name", text="Name")
        self.results_files.heading("size", text="Size")
        self.results_files.column("name", width=180, anchor="w")
        self.results_files.column("size", width=70, anchor="e")
        self.results_files.pack(fill="both", expand=True)
        self.results_files.bind("<<TreeviewSelect>>", self.preview_selected_result_file)
        ttk.Button(left, text="Draw selected", style="Accent.TButton",
                   command=self.draw_selected_outputs).pack(fill="x", pady=4)
        ttk.Button(left, text="Export PNG",
                   command=self.export_current_figure).pack(fill="x", pady=2)

        mid = ttk.Frame(paned, padding=4)
        paned.add(mid, weight=2)
        self.results_canvas = PlotCanvas(mid, figsize=(6.6, 6.0))
        self.results_canvas.pack(fill="both", expand=True)
        self.results_preview = TextEditor(mid, height=8)
        self.results_preview.pack(fill="x", pady=4)

        right = ttk.Frame(paned, padding=4)
        paned.add(right, weight=1)
        self._init_result_style_vars(right)

    def _init_result_style_vars(self, master: Any) -> None:
        nb = ttk.Notebook(master)
        nb.pack(fill="both", expand=True)
        curve_tab = ttk.Frame(nb, padding=4)
        tex_tab = ScrollableFrame(nb, width=320)
        nb.add(curve_tab, text="Curve")
        nb.add(tex_tab, text="Texture PF/IPF")

        c = self.card(curve_tab, "Curve style", "Stress-strain, activity, R-value, Young's modulus and PCYS.")
        c.pack(fill="x")
        self.curve_color = tk.StringVar(value="Navy")
        self.curve_lw = tk.StringVar(value="2.0")
        self.curve_marker = tk.StringVar(value=".")
        self.curve_msize = tk.StringVar(value="4.0")
        self.curve_ls = tk.StringVar(value="-")
        self._color_row(c, "Line color", self.curve_color)
        self._combo_row(c, "Line style", self.curve_ls, tuple(LINESTYLE_CHOICES))
        self._combo_row(c, "Marker", self.curve_marker, tuple(MARKER_CHOICES))
        self._entry_row(c, "Marker size", self.curve_msize)
        self._entry_row(c, "Line width", self.curve_lw)
        self._build_publication_style_controls(curve_tab, title="Result figure text / layout")

        # Same PF/IPF controls as the Texture page, bound to the same variables.
        self._build_texture_style_controls(tex_tab.inner, title="Texture post-processing style")

    def current_style(self) -> PlotStyle:
        # Reuse the texture-page style for texture results.
        style = self.current_texture_style()
        style.line_color = resolve_color(self.curve_color.get())
        style.line_width = safe_float(self.curve_lw.get(), 2.0)
        style.marker = self.curve_marker.get() or "."
        return style

    def _line_style(self) -> str:
        return self.curve_ls.get() or "-"

    def _stress_scale(self) -> float:
        """Multiplier from the native VPSC stress unit (MPa) to the chosen unit."""
        unit = getattr(self, "var_stress_unit", None)
        return 1.0e-3 if (unit is not None and unit.get() == "GPa") else 1.0

    def _stress_unit_label(self) -> str:
        unit = getattr(self, "var_stress_unit", None)
        return unit.get() if unit is not None else "MPa"

    def choose_run_dir(self) -> None:
        d = filedialog.askdirectory(initialdir=str(self.state_data.base_dir))
        if d:
            self.var_run_dir.set(d)
            self.state_data.last_run_dir = Path(d)
            self.refresh_results_files()

    def refresh_results_files(self) -> None:
        # ``Path('')`` is truthy in Python, so test the string contents.
        text = self.var_run_dir.get().strip()
        if text:
            run_dir: Optional[Path] = Path(text)
        else:
            rd = getattr(self.state_data, "last_run_dir", Path(""))
            run_dir = rd if str(rd).strip() else None
            if run_dir is not None:
                self.var_run_dir.set(str(run_dir))
        self.results_files.delete(*self.results_files.get_children())
        if run_dir is None or not run_dir.exists():
            return
        for path in sorted(run_dir.glob("*")):
            if path.is_file():
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                self.results_files.insert(
                    "", "end", iid=str(path),
                    values=(path.name, f"{size//1024} KB" if size > 1024 else f"{size} B")
                )

    def preview_selected_result_file(self, _event: Any = None) -> None:
        sel = self.results_files.selection()
        if not sel:
            return
        path = Path(sel[0])
        if not path.exists():
            return
        try:
            content = read_text(path)
        except OSError as e:
            content = f"<read failed: {e}>"
        self.results_preview.set(content[:5000])

    def _auto_result_paths_for_kind(self, kind: str) -> List[Path]:
        """Return appropriate output file(s) for a result kind.

        Users frequently press Draw before selecting a file.  The Results page
        should therefore behave like a post-processing dashboard: infer the
        correct output file from the run directory and only fall back to the
        Treeview selection when inference fails.
        """
        text = self.var_run_dir.get().strip()
        run_dir = Path(text) if text else self.state_data.last_run_dir
        outs = find_output_files(run_dir) if run_dir and run_dir.exists() else {}
        key_map = {
            "Stress-strain": "STR_STR",
            "Slip activity": "ACT",
            "Relative activity": "ACT",
            "R-value": "R",
            "Young's modulus": "R",
            "R-value + Young's modulus": "R",
            "Yield surface (PCYS)": "PCYS",
            "Texture PF": "TEX",
            "Texture IPF": "TEX",
        }
        key = key_map.get(kind)
        if key and key in outs:
            return [outs[key]]
        sel = self.results_files.selection()
        return [Path(s) for s in sel if Path(s).exists()]

    def draw_selected_outputs(self) -> None:
        kind = self.var_result_kind.get()
        paths = self._auto_result_paths_for_kind(kind)
        if not paths:
            messagebox.showinfo(
                "No output file",
                "No matching output file was found. Choose a run directory or select a file first.",
            )
            return
        fig = self.results_canvas.figure
        prepare_result_figure(fig)
        try:
            if kind == "Stress-strain":
                self.plot_stress(fig, paths)
            elif kind == "Slip activity":
                self.plot_activity(fig, paths, relative=False)
            elif kind == "Relative activity":
                self.plot_activity(fig, paths, relative=True)
            elif kind == "R-value":
                self.plot_lankford(fig, paths)
            elif kind == "Young's modulus":
                self.plot_young_modulus(fig, paths)
            elif kind == "R-value + Young's modulus":
                self.plot_lankford_and_young(fig, paths)
            elif kind == "Yield surface (PCYS)":
                self.plot_pcys(fig, paths)
            elif kind == "Texture PF":
                self.plot_texture(fig, paths[0], kind="pf")
            elif kind == "Texture IPF":
                self.plot_texture(fig, paths[0], kind="ipf")
            else:
                self.plot_table_xy(fig, paths)
        except Exception as e:
            LOG.warning("draw_selected_outputs (%s): %s", kind, e)
            fig.clear()
            ax = fig.add_subplot(1, 1, 1)
            ax.axis("off")
            ax.text(0.5, 0.55, "Plot failed", ha="center", va="center",
                    fontsize=13, fontweight="bold", color="#b91c1c",
                    transform=ax.transAxes)
            ax.text(0.5, 0.45, str(e), ha="center", va="center",
                    fontsize=10, color="#334155", wrap=True,
                    transform=ax.transAxes)
            self._finalize_plot(self.results_canvas, tight=True, left=0.10, right=0.97, bottom=0.12, top=0.92)
            messagebox.showerror("Plot failed", str(e))
            return
        # Keep a larger right margin for the dual-axis R/E panel.
        if kind == "R-value + Young's modulus":
            self._finalize_plot(self.results_canvas, tight=False, left=0.10, right=0.86, bottom=0.13, top=0.90)
        elif kind in {"Texture PF", "Texture IPF"}:
            self._finalize_plot(self.results_canvas, tight=True, left=0.06, right=0.98, bottom=0.07, top=0.93)
        else:
            self._finalize_plot(self.results_canvas, tight=True, left=0.10, right=0.97, bottom=0.12, top=0.92)

    def plot_stress(self, fig: Figure, paths: List[Path]) -> None:
        """Plot the macroscopic von Mises stress-strain response.

        Uses the Evm/Svm columns (header-aware) rather than the last finite
        column, and scales the stress to the selected unit.  A robust mask
        drops the occasional non-converged VPSC step whose stress spikes by
        orders of magnitude, so one bad point cannot flatten the curve.
        """
        ax = fig.add_subplot(1, 1, 1)
        scale = self._stress_scale()
        for p in paths:
            evm, svm = read_evm_svm(p)
            if evm.size == 0:
                continue
            # Keep only the primary deformation history; drop appended PCYS /
            # Lankford probe blocks that otherwise create a spurious decline.
            k = primary_history_length(evm)
            evm, svm = evm[:k], svm[:k]
            svm = svm * scale
            keep = _robust_inlier_mask(svm, n_mad=12.0)
            evm, svm = evm[keep], svm[keep]
            seg = preferred_segment(split_monotonic_segments(evm, svm))
            if seg is None:
                continue
            xs, ys = np.asarray(seg[0], dtype=float), np.asarray(seg[1], dtype=float)
            ax.plot(xs, ys, label=p.name,
                    color=resolve_color(self.curve_color.get()),
                    lw=safe_float(self.curve_lw.get(), 2.0),
                    marker=self.curve_marker.get() or "",
                    linestyle=self._line_style())
        ax.set_xlabel("Von Mises strain")
        ax.set_ylabel(f"Von Mises stress ({self._stress_unit_label()})")
        ax.set_title("Stress–strain response", pad=10)
        ax.grid(alpha=0.30)
        if ax.lines:
            ax.legend(fontsize=8, loc="best")

    def plot_activity(self, fig: Figure, paths: List[Path],
                       *, relative: bool) -> None:
        """Plot deformation-mode activity (mode columns only).

        Only MODE1..MODEn columns are drawn; the AVACS / PRITW / SECTW / TWFRm /
        EFFRm bookkeeping columns are excluded. In the relative view each step is
        normalised by the sum of the mode activities, so the curves sum to one.
        """
        ax = fig.add_subplot(1, 1, 1)
        color_cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
        for p in paths:
            strain, series = read_activity_table(p)
            if strain.size == 0 or not series:
                continue
            # Restrict to the primary deformation history; the appended PCYS /
            # Lankford probe blocks (where the abscissa jumps to integer probe
            # indices, e.g. up to 72) are not part of the activity evolution.
            k = primary_history_length(strain)
            strain = strain[:k]
            series = [(lbl, vals[:k]) for lbl, vals in series]
            if relative:
                stack = np.vstack([vals for _lbl, vals in series])
                tot = np.nansum(stack, axis=0)
                tot[np.abs(tot) < 1.0e-14] = 1.0
                series = [(lbl, vals / tot) for lbl, vals in series]
            for kk, (lbl, vals) in enumerate(series):
                xs = np.asarray(strain, dtype=float)
                ys = np.asarray(vals, dtype=float)
                finite = np.isfinite(xs) & np.isfinite(ys)
                xs, ys = xs[finite], ys[finite]
                if xs.size < 2:
                    continue
                ax.plot(xs, ys, label=f"{p.name} · {lbl}",
                        lw=safe_float(self.curve_lw.get(), 1.6),
                        linestyle=self._line_style(),
                        color=color_cycle[kk % len(color_cycle)])
        ax.set_xlabel("Equivalent strain / step-like abscissa")
        ax.set_ylabel("Relative activity" if relative else "Activity")
        ax.set_title("Slip system activity" + (" (relative)" if relative else ""), pad=10)
        ax.grid(alpha=0.30)
        if ax.lines:
            ncol = 1 if len(ax.lines) <= 8 else 2
            ax.legend(fontsize=7, ncol=ncol, loc="upper right", framealpha=0.9)

    def plot_lankford(self, fig: Figure, paths: List[Path]) -> None:
        """Plot Lankford coefficients in Cartesian coordinates."""
        ax = fig.add_subplot(1, 1, 1)
        for p in paths:
            angle, r_value, _young = read_lankford_output(p)
            if angle.size == 0:
                continue
            seg = preferred_segment(split_monotonic_segments(angle, r_value))
            if seg is None:
                continue
            aa, rr = np.asarray(seg[0], dtype=float), np.asarray(seg[1], dtype=float)
            ax.plot(aa, rr, label=p.name,
                    color=resolve_color(self.curve_color.get()),
                    lw=safe_float(self.curve_lw.get(), 1.8),
                    marker=self.curve_marker.get() or "o",
                    markersize=max(3.0, safe_float(self.curve_msize.get(), 4.0)),
                    linestyle=self._line_style())
        ax.set_xlabel("Angle from RD to TD, α (deg)")
        ax.set_ylabel("Lankford coefficient R")
        ax.set_title("Lankford coefficient", pad=10)
        xmax = 90.0 if (not ax.lines or max(float(np.nanmax(line.get_xdata())) for line in ax.lines) <= 90.0 + 1.0e-8) else 180.0
        ax.set_xlim(0.0, xmax)
        ax.set_xticks(np.linspace(0.0, xmax, int(xmax / 15) + 1))
        ax.grid(alpha=0.30)
        if ax.lines:
            ymin = min(float(np.nanmin(line.get_ydata())) for line in ax.lines)
            ymax = max(float(np.nanmax(line.get_ydata())) for line in ax.lines)
            pad = 0.08 * max(1.0, ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.legend(fontsize=8, loc="best")

    def plot_young_modulus(self, fig: Figure, paths: List[Path]) -> None:
        """Plot directional Young's modulus from VPSC LANKFORD.OUT.

        Standard VPSC LANKFORD.OUT columns are interpreted as
        angle, Young modulus, Lankford coefficient, followed by tensor
        components.  This plot uses the second column as E(α), which is
        the directional Young's modulus between RD and TD.
        """
        ax = fig.add_subplot(1, 1, 1)
        scale = self._stress_scale()
        for p in paths:
            angle, _r_value, young = read_lankford_output(p)
            if angle.size == 0 or young is None or young.size == 0:
                continue
            young = young * scale
            seg = preferred_segment(split_monotonic_segments(angle, young))
            if seg is None:
                continue
            aa, yy = np.asarray(seg[0], dtype=float), np.asarray(seg[1], dtype=float)
            ax.plot(aa, yy, label=p.name,
                    color=resolve_color(self.curve_color.get()),
                    lw=safe_float(self.curve_lw.get(), 1.8),
                    marker=self.curve_marker.get() or "o",
                    markersize=max(3.0, safe_float(self.curve_msize.get(), 4.0)),
                    linestyle=self._line_style())
        ax.set_xlabel("Angle from RD to TD, α (deg)")
        ax.set_ylabel(f"Young's modulus E ({self._stress_unit_label()})")
        ax.set_title("Directional Young's modulus", pad=10)
        xmax = 90.0 if (not ax.lines or max(float(np.nanmax(line.get_xdata())) for line in ax.lines) <= 90.0 + 1.0e-8) else 180.0
        ax.set_xlim(0.0, xmax)
        ax.set_xticks(np.linspace(0.0, xmax, int(xmax / 15) + 1))
        ax.grid(alpha=0.30)
        if ax.lines:
            ymin = min(float(np.nanmin(line.get_ydata())) for line in ax.lines)
            ymax = max(float(np.nanmax(line.get_ydata())) for line in ax.lines)
            pad = 0.08 * max(1.0, ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.legend(fontsize=8, loc="best")

    def plot_lankford_and_young(self, fig: Figure, paths: List[Path]) -> None:
        """Plot Lankford coefficient and directional Young's modulus together.

        The right y-axis is kept in the same physical unit as the standalone
        Young's-modulus panel.  This avoids the misleading 18--24 style axis
        that may appear when the Young-modulus column is read in 10*GPa units
        or when Matplotlib applies a compact tick representation.
        """
        from matplotlib.ticker import MaxNLocator, ScalarFormatter

        # The second y-axis needs extra right margin in the embedded Tk canvas;
        # otherwise the right-side tick labels and axis label can be clipped by
        # the neighbouring control panel.
        fig.subplots_adjust(left=0.10, right=0.86, bottom=0.13, top=0.90)

        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = ax1.twinx()
        any_line = False
        e_ydata: List[np.ndarray] = []

        for p in paths:
            angle, r_value, young = read_lankford_output(p)
            if angle.size == 0:
                continue
            if young is not None:
                young = young * self._stress_scale()
            if young is not None and young.size == angle.size:
                seg = preferred_segment(split_monotonic_segments(angle, r_value, young))
                if seg is None:
                    continue
                aa = np.asarray(seg[0], dtype=float)
                rr = np.asarray(seg[1], dtype=float)
                yy = np.asarray(seg[2], dtype=float)
            else:
                seg = preferred_segment(split_monotonic_segments(angle, r_value))
                if seg is None:
                    continue
                aa = np.asarray(seg[0], dtype=float)
                rr = np.asarray(seg[1], dtype=float)
                yy = np.empty(0)

            ax1.plot(aa, rr, label=f"R · {p.name}",
                     color=resolve_color(self.curve_color.get(), "#111827"),
                     lw=safe_float(self.curve_lw.get(), 1.8),
                     marker=self.curve_marker.get() or "o",
                     markersize=max(3.0, safe_float(self.curve_msize.get(), 4.0)),
                     linestyle=self._line_style())
            if yy.size:
                ax2.plot(aa, yy, label=f"E · {p.name}",
                         color="#dc2626", lw=safe_float(self.curve_lw.get(), 1.8),
                         marker="", linestyle="-")
                e_ydata.append(yy[np.isfinite(yy)])
            any_line = True

        ax1.set_xlabel("Angle from RD to TD, α (deg)")
        ax1.set_ylabel("Lankford coefficient R")
        ax2.set_ylabel(f"Young's modulus E ({self._stress_unit_label()})", labelpad=8)
        ax1.set_title("Lankford coefficient and Young's modulus", pad=10)

        xmax = 90.0
        all_x = [line.get_xdata() for line in ax1.lines + ax2.lines if len(line.get_xdata())]
        if all_x and max(float(np.nanmax(x)) for x in all_x) > 90.0 + 1.0e-8:
            xmax = 180.0
        ax1.set_xlim(0.0, xmax)
        ax1.set_xticks(np.linspace(0.0, xmax, int(xmax / 15) + 1))
        ax1.grid(alpha=0.30)

        # Force full numeric labels on the right axis.  Without this, the GUI can
        # display compact values such as 18--24 instead of 180--240, which is
        # ambiguous for a publication figure.
        if e_ydata:
            ey = np.concatenate([v for v in e_ydata if v.size])
            if ey.size:
                emin, emax = float(np.nanmin(ey)), float(np.nanmax(ey))
                epad = 0.08 * max(1.0, emax - emin)
                ax2.set_ylim(emin - epad, emax + epad)
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        ax2.yaxis.set_major_formatter(formatter)
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax2.tick_params(axis="y", which="major", pad=3)

        if any_line:
            lines = ax1.lines + ax2.lines
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, fontsize=8, loc="best")

    def plot_pcys(self, fig: Figure, paths: List[Path]) -> None:
        """Plot a clean closed polycrystal yield surface with optional rate ticks."""
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal", adjustable="box")
        scale = self._stress_scale()
        for p in paths:
            x, y, rx, ry = extract_pcys_curve(read_numeric_table(p))
            if x.size == 0:
                continue
            x = x * scale
            y = y * scale
            ax.plot(x, y, label=p.name,
                    color=resolve_color(self.curve_color.get()),
                    lw=safe_float(self.curve_lw.get(), 1.8),
                    linestyle=self._line_style())
            if rx is not None and ry is not None and len(x) > 8:
                step = max(1, len(x) // 28)
                span = max(np.ptp(x), np.ptp(y), 1.0)
                scale = 0.035 * span
                mag = np.nanmax(np.hypot(rx, ry))
                if np.isfinite(mag) and mag > 0:
                    ax.quiver(x[::step], y[::step], rx[::step] / mag * scale, ry[::step] / mag * scale,
                              angles="xy", scale_units="xy", scale=1,
                              color="#64748b", width=0.0022, alpha=0.70,
                              headwidth=3.5, headlength=5.0, headaxislength=4.5)
        ax.axhline(0, color="#94a3b8", lw=0.7, zorder=0)
        ax.axvline(0, color="#94a3b8", lw=0.7, zorder=0)
        ax.set_xlabel(f"Stress component 1 ({self._stress_unit_label()})")
        ax.set_ylabel(f"Stress component 2 ({self._stress_unit_label()})")
        ax.set_title("Polycrystal yield surface", pad=10)
        ax.grid(alpha=0.30)
        if ax.lines:
            allx = np.concatenate([line.get_xdata() for line in ax.lines if len(line.get_xdata())])
            ally = np.concatenate([line.get_ydata() for line in ax.lines if len(line.get_ydata())])
            if allx.size and ally.size:
                span = max(float(np.nanmax(allx) - np.nanmin(allx)), float(np.nanmax(ally) - np.nanmin(ally)), 1.0)
                xc = 0.5 * (float(np.nanmax(allx)) + float(np.nanmin(allx)))
                yc = 0.5 * (float(np.nanmax(ally)) + float(np.nanmin(ally)))
                half = 0.56 * span
                ax.set_xlim(xc - half, xc + half)
                ax.set_ylim(yc - half, yc + half)
            ax.legend(fontsize=8, loc="best")

    def plot_table_xy(self, fig: Figure, paths: List[Path]) -> None:
        ax = fig.add_subplot(1, 1, 1)
        for p in paths:
            arr = read_numeric_table(p)
            if arr.size == 0:
                continue
            if arr.shape[1] >= 2:
                ax.plot(arr[:, 0], arr[:, 1], label=p.name,
                        lw=safe_float(self.curve_lw.get(), 1.5))
            else:
                ax.plot(arr[:, 0], label=p.name,
                        lw=safe_float(self.curve_lw.get(), 1.5))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    def plot_texture(self, fig: Figure, path: Path, *, kind: str) -> None:
        try:
            t = parse_texture(path)
        except Exception as e:
            raise RuntimeError(f"Cannot parse texture: {e}") from e
        style = self.current_style()
        family = self.style_family.get()
        projection = self.style_proj.get()
        if kind == "pf":
            poles = parse_poles(self.style_poles.get()) or [[1, 0, 0]]
            draw_pf_ipf_figure(fig, t, style, kind="pf", family=family,
                               projection=projection, items=poles)
        else:
            dirs = [d.strip().upper() for d in re.split(r"[\s,]+",
                    self.style_directions.get()) if d.strip()] or ["ND"]
            draw_pf_ipf_figure(fig, t, style, kind="ipf", family=family,
                               projection=projection, items=dirs)

    def export_current_figure(self) -> None:
        self._save_figure_with_dialog(self.results_canvas.figure, "vpsc_result.png")

    # ------------------------------------------------------------------- Files
    def _build_files_page(self) -> None:
        p = self.pages["Files"]
        ttk.Label(p, text="Files", style="Title.TLabel").pack(
            anchor="w", pady=(4, 8)
        )
        bar = ttk.Frame(p)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Button(bar, text="Refresh", command=self.refresh_file_browser).pack(
            side="left", padx=2
        )
        ttk.Button(bar, text="Open base folder",
                   command=lambda: self.open_folder(self.state_data.base_dir)).pack(
            side="left", padx=2
        )

        paned = ttk.PanedWindow(p, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned, padding=4)
        paned.add(left, weight=1)
        self.file_tree = ttk.Treeview(left, columns=("name", "size"),
                                      show="headings", height=24)
        self.file_tree.heading("name", text="Name")
        self.file_tree.heading("size", text="Size")
        self.file_tree.column("name", width=240, anchor="w")
        self.file_tree.column("size", width=80, anchor="e")
        self.file_tree.pack(fill="both", expand=True)
        self.file_tree.bind("<<TreeviewSelect>>", self.load_selected_file_text)

        right = ttk.Frame(paned, padding=4)
        paned.add(right, weight=2)
        self.file_editor = TextEditor(right, height=24)
        self.file_editor.pack(fill="both", expand=True)
        ttk.Button(right, text="Save edits", style="Accent.TButton",
                   command=self.save_selected_file_text).pack(anchor="e", pady=4)

    def refresh_file_browser(self) -> None:
        self.file_tree.delete(*self.file_tree.get_children())
        base = self.state_data.base_dir
        if not base.exists():
            return
        for path in sorted(base.iterdir()):
            try:
                if path.is_file():
                    size = path.stat().st_size
                    self.file_tree.insert(
                        "", "end", iid=str(path),
                        values=(path.name,
                                f"{size//1024} KB" if size > 1024 else f"{size} B")
                    )
            except OSError:
                continue

    def load_selected_file_text(self, _event: Any = None) -> None:
        sel = self.file_tree.selection()
        if not sel:
            return
        path = Path(sel[0])
        try:
            self.file_editor.set(read_text(path))
        except OSError as e:
            self.file_editor.set(f"<read failed: {e}>")

    def save_selected_file_text(self) -> None:
        sel = self.file_tree.selection()
        if not sel:
            return
        path = Path(sel[0])
        if not messagebox.askyesno("Overwrite", f"Save to\n{path}?"):
            return
        try:
            write_text(path, self.file_editor.get())
            PARSE_CACHE.clear()
        except OSError as e:
            messagebox.showerror("Save failed", str(e))

    # ------------------------------------------------------------------- Theory
    def _build_theory_page(self) -> None:
        p = self.pages["Theory"]
        ttk.Label(p, text="Theory notes", style="Title.TLabel").pack(
            anchor="w", pady=(4, 8)
        )
        ed = TextEditor(p, height=30)
        ed.pack(fill="both", expand=True)
        ed.set(VPSC_THEORY_NOTES)
        ed.text.configure(state="disabled")

    # ----------------------------------------------------------- folder opener
    def open_folder(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            messagebox.showinfo("Open folder", f"Path does not exist:\n{path}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as e:
            LOG.warning("open_folder failed: %s", e)



# =============================================================================
# Embedded texture plotting core
# =============================================================================
# This section provides a texture3-style PF/IPF workflow inside the VPSC GUI.
# The public options follow the usual polefigure(...).pf_new(...) convention:
# proj="pf"/"ipf", poles, dph, dth, n_rim, mode, mn/mx, levels, cmap
# and smoothing.

TEXTURE3_MODE_ALIASES = {
    "scatter": "dot",
    "density": "fill",
    "contourf": "fill",
    "both": "fill+dot",
    "contour": "line",
    "line": "line",
    "fill": "fill",
    "dot": "dot",
    "dotc": "dotc",
    "dotm": "dotm",
}


DEFAULT_TEXTURE_PROJECTION = "equal-area"
# Renderer-wide active projection.  PF/IPF density, scatter, boundary and
# decorations all read this single value, so a figure can never mix two
# projections (which would misalign the contour field and the rim).  It is set
# by EmbeddedTexture3PoleFigure.pf_new from the requested PlotStyle for the
# duration of one render.  Equal-area (Schmidt) is the default because it maps a
# random texture to a uniform areal density, which is what makes the flat-mean
# MRD normalisation in draw_density quantitatively correct; stereographic is
# offered for users who prefer the conformal (angle-true) Wulff layout.
_ACTIVE_PROJECTION = DEFAULT_TEXTURE_PROJECTION


def _is_equal_area(kind: Optional[str]) -> bool:
    k = (kind or _ACTIVE_PROJECTION or DEFAULT_TEXTURE_PROJECTION).strip().lower()
    return not k.startswith("stereo")


def texture3_projection(v: np.ndarray, kind: Optional[str] = None) -> np.ndarray:
    """Project crystal/sample directions onto the PF/IPF plane.

    Two area conventions are supported, selected by ``kind`` (or, when ``None``,
    by the renderer-wide ``_ACTIVE_PROJECTION``):

    * ``equal-area`` (default) -- Lambert/Schmidt projection.  A random texture
      maps to a uniform areal density, so the flat-mean MRD normalisation in
      :func:`draw_density` is quantitatively correct.
    * ``stereographic`` -- conformal Wulff projection (angle-true but area
      biased); density maps drawn on it are radially distorted and should be
      read qualitatively.

    Both fold directions onto the lower hemisphere with identical azimuth and
    handedness, and both place the equator at radius 1, so the unit-circle rim,
    axis labels and clip paths are shared between the two conventions.
    """
    arr = np.atleast_2d(np.asarray(v, dtype=float)).copy()
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    n[n < 1e-30] = 1.0
    arr = arr / n
    # Use the lower hemisphere.  Equatorial directions are folded with the
    # upper hemisphere to keep HCP IPF rim vertices in the same sector as the
    # reduced IPF points.
    arr[arr[:, 2] >= 0] *= -1.0
    a, b, c = arr[:, 0], arr[:, 1], arr[:, 2]
    rho = np.hypot(a, b)
    safe = rho >= 1e-30
    if _is_equal_area(kind):
        # R = sqrt(1 + c): 0 at the south pole (centre) and 1 at the equator
        # (rim).  The leading minus reproduces the historical azimuth/handedness
        # so switching projections never mirrors the figure.
        radial = np.sqrt(np.maximum(0.0, 1.0 + c))
        scale = np.zeros_like(c)
        scale[safe] = -radial[safe] / rho[safe]
        xy = np.column_stack([a * scale, b * scale])
    else:
        den = c - 1.0
        den[np.abs(den) < 1e-30] = -1e-30
        xy = np.column_stack([a / den, b / den])
    xy[~safe] = 0.0
    return xy if np.asarray(v).ndim > 1 else xy[0]


def texture3_slerp(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a / max(np.linalg.norm(a), 1e-30)
    b = b / max(np.linalg.norm(b), 1e-30)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    om = math.acos(dot)
    if abs(om) < 1e-12:
        return np.repeat(a[None, :], n, axis=0)
    t = np.linspace(0.0, 1.0, n)
    so = math.sin(om)
    return (np.sin((1-t)*om)[:, None] * a + np.sin(t*om)[:, None] * b) / so


def texture3_axis2vect(label: str | int) -> np.ndarray:
    """texture3 axis label helper: 1/2/3 or RD/TD/ND -> unit vector."""
    if isinstance(label, str):
        lab = label.strip().upper()
        if lab in {"RD", "X", "1"}: return np.array([1.0, 0.0, 0.0])
        if lab in {"TD", "Y", "2"}: return np.array([0.0, 1.0, 0.0])
        if lab in {"ND", "Z", "3"}: return np.array([0.0, 0.0, 1.0])
        if lab in {"-RD", "-X", "-1"}: return np.array([-1.0, 0.0, 0.0])
        if lab in {"-TD", "-Y", "-2"}: return np.array([0.0, -1.0, 0.0])
        if lab in {"-ND", "-Z", "-3"}: return np.array([0.0, 0.0, -1.0])
    i = int(label)
    return np.array({1:(1,0,0), 2:(0,1,0), 3:(0,0,1),
                     -1:(-1,0,0), -2:(0,-1,0), -3:(0,0,-1)}.get(i, (0,0,1)), dtype=float)

def _normalise_vec(v: np.ndarray, default: Sequence[float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    w = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(w))
    if n < 1e-12:
        return np.asarray(default, dtype=float)
    return w / n


def texture3_display_transform(ix: str = "RD", iy: str = "TD") -> np.ndarray:
    """Return an orthonormal sample-frame transform for the requested display axes."""
    ex = _normalise_vec(texture3_axis2vect(ix), (1.0, 0.0, 0.0))
    ey0 = _normalise_vec(texture3_axis2vect(iy), (0.0, 1.0, 0.0))
    ey = ey0 - np.dot(ey0, ex) * ex
    if np.linalg.norm(ey) < 1e-8:
        for cand in (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])):
            ey = cand - np.dot(cand, ex) * ex
            if np.linalg.norm(ey) >= 1e-8:
                break
    ey = _normalise_vec(ey, (0.0, 1.0, 0.0))
    ez = np.cross(ex, ey)
    if np.linalg.norm(ez) < 1e-8:
        ez = np.array([0.0, 0.0, 1.0])
    ez = _normalise_vec(ez, (0.0, 0.0, 1.0))
    ey = _normalise_vec(np.cross(ez, ex), (0.0, 1.0, 0.0))
    return np.vstack([ex, ey, ez])


def texture3_rotate_xy(xy: np.ndarray, rot_deg: float = 0.0) -> np.ndarray:
    ang = math.radians(float(rot_deg))
    if abs(ang) < 1e-12 or np.asarray(xy).size == 0:
        return np.asarray(xy, dtype=float)
    c = math.cos(ang)
    s = math.sin(ang)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return np.asarray(xy, dtype=float) @ R.T



def texture3_get_ipf_boundary(family: str = "cubic", nres: int = 90,
                              projection_name: str = "stereographic") -> Tuple[np.ndarray, List[int], List[int], List[int]]:
    """Curved IPF fundamental boundary, equivalent to texture3 get_ipf_boundary."""
    fam = family.lower()
    if fam in {"hcp", "hex", "hexag", "hexagonal"}:
        labels = ([0,0,0,1], [1,0,-1,0], [1,1,-2,0])
        # HCP IPF uses crystallographic directions, not reciprocal plane normals.
        # The reduced sector is [0001] -> [10-10] -> [11-20].
        dirs = [
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([math.cos(math.pi / 6.0), math.sin(math.pi / 6.0), 0.0]),
        ]
    else:
        labels = ([0,0,1], [1,0,1], [1,1,1])
        dirs = [miller_cartesian(q, "cubic") for q in labels]
    arcs = []
    for a,b in [(dirs[0], dirs[1]), (dirs[1], dirs[2]), (dirs[2], dirs[0])]:
        pts = texture3_slerp(a, b, max(3, int(nres)))
        arcs.append(pts[:-1])
    sph = np.vstack(arcs + [dirs[0][None, :]])
    # Project the boundary with the same (renderer-wide) projection as the
    # plotted points so the curved sector outline always matches the data,
    # whether the active projection is equal-area or stereographic.
    xy = texture3_projection(sph).T
    return xy, list(labels[0]), list(labels[1]), list(labels[2])


def texture3_get_within_boundary(boundary: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Boolean mask selecting points inside a texture3 boundary polygon."""
    if xy.size == 0:
        return np.zeros(0, dtype=bool)
    verts = np.asarray(boundary).T
    return MplPath(verts).contains_points(np.asarray(xy), radius=1e-9)


def texture3_equiv_poles(pole: Sequence[int], family: str, antipodal: bool = True) -> np.ndarray:
    p0 = miller_cartesian(pole, family)
    eq = equivalents_for_family(p0, family)
    if antipodal:
        eq = np.unique(np.round(np.vstack([eq, -eq]), 12), axis=0)
    return eq


def texture3_agr2pol_batch(texture: TextureData, pole: Sequence[int] | str,
                           proj: str, family: str,
                           antipodal: bool = True,
                           transform: Optional[np.ndarray] = None,
                           rot_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised texture3 agr2pol/core equivalent.

    PF: crystal pole equivalents are mapped to sample axes via g.T and can be
    rotated into a user-selected display frame so changing ix/iy rotates the
    actual figure, not just the labels.
    IPF: sample direction RD/TD/ND is mapped into crystal axes via g and folded.
    Returns xy and point weights.
    """
    G = texture.matrices
    if proj == "pf":
        eq = texture3_equiv_poles(pole, family, antipodal=antipodal)
        v = np.einsum("nji,kj->nki", G, eq).reshape(-1, 3)
        if transform is not None:
            v = np.asarray(v, dtype=float) @ np.asarray(transform, dtype=float).T
        w = np.repeat(texture.weights, eq.shape[0])
        xy = texture3_rotate_xy(texture3_projection(v), rot_deg)
        r = np.hypot(xy[:, 0], xy[:, 1])
        keep = r <= 1.0 + 2.5e-3
        return xy[keep], w[keep]
    else:
        svec = texture3_axis2vect(str(pole))
        vc = G @ svec
        if family.lower() in {"hcp", "hex", "hexag", "hexagonal"}:
            red = reduce_hcp_ipf(vc)
        else:
            red = reduce_cubic_ipf(vc)
        xy = texture3_rotate_xy(texture3_projection(red), rot_deg)
        return xy, texture.weights.copy()


def texture3_deco_pf(ax: Any, *, proj: str, boundary: Optional[np.ndarray], miller: Any,
                     ix: str = "RD", iy: str = "TD", mode: str = "line",
                     ires: bool = True) -> None:
    """texture3 deco_pf equivalent: circle/triangle, labels, Miller title."""
    ax.set_aspect("equal")
    ax.set_axis_off()
    if proj == "pf":
        t = np.linspace(0.0, 2.0*math.pi, 721)
        ax.plot(np.cos(t), np.sin(t), "k-", lw=1.05, solid_joinstyle="round", solid_capstyle="round")
        ax.plot([0, 0], [0.975, 1.035], "k-", lw=0.8)
        ax.plot([0.975, 1.035], [0, 0], "k-", lw=0.8)
        if ires:
            ax.axhline(0, color="#cbd5e1", lw=0.45, zorder=0)
            ax.axvline(0, color="#cbd5e1", lw=0.45, zorder=0)
        ax.text(1.065, 0.0, ix, va="center", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.80))
        ax.text(0.0, 1.065, iy, va="bottom", ha="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.80))
        ax.set_xlim(-1.10, 1.30)
        ax.set_ylim(-1.20, 1.20)
    else:
        if boundary is not None:
            ax.plot(boundary[0], boundary[1], "k-", lw=1.2, zorder=100, solid_joinstyle="round", solid_capstyle="round")
            bx = np.asarray(boundary[0], dtype=float)
            by = np.asarray(boundary[1], dtype=float)
            dx = max(float(np.nanmax(bx) - np.nanmin(bx)), 1e-6)
            dy = max(float(np.nanmax(by) - np.nanmin(by)), 1e-6)
            pad = 0.08 * max(dx, dy)
            ax.set_xlim(float(np.nanmin(bx)) - pad, float(np.nanmax(bx)) + pad)
            ax.set_ylim(float(np.nanmin(by)) - pad, float(np.nanmax(by)) + pad)
        else:
            ax.set_xlim(-0.05, 0.85)
            ax.set_ylim(-0.05, 0.85)
    # Use compact Miller text such as (100), (0002), (10-10).
    if isinstance(miller, (list, tuple, np.ndarray)):
        lab = "(" + "".join(str(int(v)) for v in miller) + ")"
    else:
        lab = str(miller)
    ax.set_title(lab, fontsize=10, pad=2)


class EmbeddedTexture3PoleFigure:
    """Small in-app counterpart of texture3.upf.polefigure.

    It exposes the familiar ``pf_new`` API and delegates parsing/orientation to the
    VPSC app's TextureData.  It supports VPSC/TEX_PH*.OUT orientations, FCC/BCC
    cubic and HCP/hexagonal symmetry, equal-area (default) or stereographic
    projection, contour PF, dot PF, dot-coloured PF, and curved IPF sectors.
    """
    def __init__(self, grains: Optional[np.ndarray] = None, filename: Optional[str] = None,
                 fnsx: Optional[str] = None, csym: Optional[str] = None,
                 cdim: Optional[Sequence[float]] = None, cang: Optional[Sequence[float]] = None):
        if filename is not None:
            self.texture = parse_texture(Path(filename))
        elif grains is not None:
            arr = np.asarray(grains, dtype=float)
            if arr.shape[1] < 3:
                raise ValueError("grains must contain at least phi1,Phi,phi2")
            w = arr[:, 3] if arr.shape[1] >= 4 else np.ones(arr.shape[0])
            w = np.asarray(w, dtype=float); w = w / max(float(w.sum()), 1e-30)
            self.texture = TextureData(arr[:, :3].copy(), w, arr.copy())
        else:
            raise ValueError("grains or filename must be supplied")
        self.fnsx = fnsx
        self.csym = (csym or "cubic").lower()
        if self.csym in {"hexag", "hexagonal", "hcp"}:
            self.family = "hcp"
        else:
            self.family = "cubic"
        if fnsx:
            try:
                sx = parse_sx(Path(fnsx))
                self.family = "hcp" if sx.family.lower().startswith("h") else "cubic"
                if sx.unit_cell and len(sx.unit_cell) >= 3:
                    self.cdim = sx.unit_cell[:3]
                else:
                    self.cdim = list(cdim or [1,1,1])
                self.cang = sx.unit_cell[3:6] if sx.unit_cell and len(sx.unit_cell) >= 6 else list(cang or [90,90,90])
            except Exception:
                self.cdim = list(cdim or [1,1,1])
                self.cang = list(cang or [90,90,90])
        else:
            self.cdim = list(cdim or [1,1,1])
            self.cang = list(cang or [90,90,90])

    def pf_new(self, ifig: Optional[int] = None, axs: Optional[Sequence[Any]] = None,
               proj: str = "pf", poles: Sequence[Any] = ((1,0,0), (1,1,0)),
               ix: str = "RD", iy: str = "TD", mode: str = "line",
               dth: float = 10.0, dph: float = 10.0, n_rim: int = 2,
               cdim: Optional[Sequence[float]] = None, ires: bool = True,
               mn: Optional[float] = None, mx: Optional[float] = None,
               lev_norm_log: bool = True, nlev: int = 7, ilev: int = 1,
               levels: Optional[Sequence[float]] = None, cmap: str = "magma",
               rot: float = 0.0, iline_khi80: bool = False,
               transform: Optional[np.ndarray] = None, ideco_lev: bool = True,
               ismooth: float = 1.0, fig: Optional[Figure] = None,
               style: Optional[PlotStyle] = None, **kwargs: Any) -> Figure:
        if style is None:
            style = PlotStyle(cmap=cmap, levels=nlev, bins=int(round(360/max(dph,1e-6))),
                              smooth=max(float(ismooth), 0.0), texture_mode=mode)
        else:
            style.cmap = cmap or style.cmap
            style.levels = int(nlev or style.levels)
            style.smooth = float(ismooth if ismooth is not None else style.smooth)
            style.texture_mode = mode or style.texture_mode
        setattr(style, "texture3_log", bool(lev_norm_log))
        # Set the renderer-wide projection for this figure.  Equal-area keeps the
        # MRD normalisation in draw_density quantitatively correct; the single
        # global guarantees density, scatter, boundary and rim all agree.
        global _ACTIVE_PROJECTION
        _ACTIVE_PROJECTION = str(getattr(style, "projection", DEFAULT_TEXTURE_PROJECTION)
                                 or DEFAULT_TEXTURE_PROJECTION)
        setattr(style, "texture3_mn", mn)
        setattr(style, "texture3_mx", mx)
        if fig is None:
            fig = Figure(figsize=(3.3 * max(1, len(poles)), 3.0), dpi=100, facecolor="white")
        fig.clear()
        n = len(poles)
        ncols = min(3, max(1, n)); nrows = int(math.ceil(n / ncols))
        boundary = None
        if proj == "ipf":
            boundary, a, b, c = texture3_get_ipf_boundary(self.family, nres=120, projection_name="stereographic")
            boundary = np.asarray(boundary, dtype=float)
            boundary[:] = texture3_rotate_xy(boundary.T, rot).T
        pf_transform = texture3_display_transform(ix, iy) if proj == "pf" else None
        mode2 = TEXTURE3_MODE_ALIASES.get(str(mode).lower(), str(mode).lower())
        for i, pole in enumerate(poles, 1):
            ax = fig.add_subplot(nrows, ncols, i)
            if proj == "pf":
                if mode2 in {"dot", "dotc", "dotm"}:
                    xy, w = texture3_agr2pol_batch(self.texture, pole, "pf", self.family, transform=pf_transform, rot_deg=rot)
                    if mode2 == "dotm":
                        ax.text(0.02, 0.98, f"{len(xy)} points", transform=ax.transAxes, va="top")
                    else:
                        kw = dict(s=float(getattr(style, "point_size", 10)), marker=getattr(style, "marker", "o"),
                                  alpha=float(getattr(style, "alpha", 0.7)), rasterized=True)
                        if mode2 == "dotc":
                            sc = ax.scatter(xy[:,0], xy[:,1], c=w, cmap=style.cmap, **kw)
                            if style.colorbar:
                                fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.08, fraction=0.046)
                        else:
                            ax.scatter(xy[:,0], xy[:,1], c=resolve_color(style.point_color), edgecolors="none", **kw)
                else:
                    # Fill/contour maps are generated from the same projected
                    # PF points that are used by scatter mode.  This avoids the
                    # radial reversal that can appear when angular-cell data and
                    # projected scatter points use different radial orderings.
                    xy, w = texture3_agr2pol_batch(self.texture, pole, "pf", self.family, transform=pf_transform, rot_deg=rot)
                    old_mode = style.texture_mode
                    style.texture_mode = "contour" if mode2 == "line" else "contourf"
                    cnt = draw_density(ax, xy[:, 0], xy[:, 1], w, style,
                                       "circle", verts=None, n_equivalents=1)
                    style.texture_mode = old_mode
                    if mode2 == "fill+dot":
                        ax.scatter(xy[:, 0], xy[:, 1],
                                   s=max(1.0, float(style.point_size) * 0.25),
                                   c=resolve_color(style.point_color),
                                   alpha=min(0.28, float(style.alpha)),
                                   edgecolors="none", rasterized=True)
                    if style.colorbar and cnt is not None and mode2 != "line":
                        fig.colorbar(cnt, ax=ax, shrink=0.78, pad=0.08, fraction=0.046, label="MRD")
                texture3_deco_pf(ax, proj="pf", boundary=None, miller=pole, ix=ix, iy=iy, mode=mode2, ires=ires)
            else:
                # texture3 originally offered IPF dot only; the app also provides contour/fill
                # by binning dots in the curved IPF triangle.
                xy, w = texture3_agr2pol_batch(self.texture, str(pole).upper(), "ipf", self.family, rot_deg=rot)
                mask = texture3_get_within_boundary(boundary, xy) if boundary is not None else np.ones(len(xy), dtype=bool)
                xy = xy[mask]; w = w[mask]
                if mode2 in {"dot", "dotc", "dotm"}:
                    if mode2 == "dotc":
                        sc = ax.scatter(xy[:,0], xy[:,1], c=w, cmap=style.cmap,
                                        s=float(style.point_size), marker=style.marker,
                                        alpha=float(style.alpha), rasterized=True)
                        if style.colorbar:
                            fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.08, fraction=0.046)
                    else:
                        ax.scatter(xy[:,0], xy[:,1], c=resolve_color(style.point_color),
                                   s=float(style.point_size), marker=style.marker,
                                   alpha=float(style.alpha), edgecolors="none", rasterized=True)
                else:
                    verts = boundary.T if boundary is not None else None
                    # Reuse app density engine in triangle, with texture3 mode aliases.
                    mode_save = style.texture_mode
                    style.texture_mode = "contour" if mode2 == "line" else "contourf"
                    im = draw_density(ax, xy[:,0], xy[:,1], w, style, "triangle", verts=verts, n_equivalents=1)
                    style.texture_mode = mode_save
                    if style.colorbar and im is not None and mode2 != "line":
                        fig.colorbar(im, ax=ax, shrink=0.82, pad=0.08, fraction=0.046, label="MRD")
                    if mode2 == "fill+dot":
                        ax.scatter(xy[:,0], xy[:,1], c=resolve_color(style.point_color),
                                   s=max(1.0, float(style.point_size)*0.35), alpha=min(0.35, float(style.alpha)),
                                   edgecolors="none", rasterized=True)
                texture3_deco_pf(ax, proj="ipf", boundary=boundary, miller=pole, ix=ix, iy=iy, mode=mode2, ires=ires)
                # corner labels: use the exact crystallographic vertices and offset them away
                # from the rim so they never overlap the boundary line.
                if boundary is not None:
                    fam = self.family.lower()
                    labs = ["0001", "10-10", "11-20"] if fam == "hcp" else ["001", "101", "111"]
                    if fam == "hcp":
                        vtx_dirs = np.array([[0.0, 0.0, 1.0],
                                             [1.0, 0.0, 0.0],
                                             [math.cos(math.pi / 6.0), math.sin(math.pi / 6.0), 0.0]], dtype=float)
                    else:
                        vtx_dirs = np.vstack([miller_cartesian([0, 0, 1], "cubic"),
                                              miller_cartesian([1, 0, 1], "cubic"),
                                              miller_cartesian([1, 1, 1], "cubic")])
                    pts = texture3_rotate_xy(texture3_projection(vtx_dirs), rot)
                    centroid = np.mean(boundary.T, axis=0)
                    for lab, (xx, yy) in zip(labs, pts):
                        vec = np.array([xx, yy], dtype=float) - centroid
                        nrm = float(np.linalg.norm(vec)) or 1.0
                        xlab = float(xx + 0.060 * vec[0] / nrm)
                        ylab = float(yy + 0.060 * vec[1] / nrm)
                        ax.text(xlab, ylab, lab, fontsize=8, color="#0f172a",
                                ha="center", va="center", clip_on=False,
                                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88),
                                zorder=150)
                    bx = np.asarray(boundary[0], dtype=float)
                    by = np.asarray(boundary[1], dtype=float)
                    pad = 0.14 * max(float(np.ptp(bx)), float(np.ptp(by)), 1e-6)
                    ax.set_xlim(float(np.min(bx) - pad), float(np.max(bx) + pad))
                    ax.set_ylim(float(np.min(by) - pad), float(np.max(by) + pad))
        fig.tight_layout(pad=1.35, w_pad=2.6, h_pad=1.4)
        return fig


def draw_pf_ipf_figure(fig: Figure, texture: TextureData, style: PlotStyle, *,
                       kind: str, family: str,
                       items: Sequence[Any],
                       projection: str = "equal-area",
                       mode: Optional[str] = None,
                       title_prefix: str = "",
                       c_over_a: float = 1.633) -> None:
    """Unified texture renderer using the embedded texture3 core.

    PF/IPF are drawn through ``EmbeddedTexture3PoleFigure.pf_new`` so Results and
    Texture Studio share one renderer: curved IPF sector, equal-area (default) or
    stereographic projection, line/fill/dot/dotc modes, angular grid, rim
    smoothing and shared level controls.
    """
    proj = "pf" if kind.lower() == "pf" else "ipf"
    poles_or_dirs = list(items) if items else ([[1,0,0],[1,1,0],[1,1,1]] if proj == "pf" else ["ND"])
    # Use SX-derived family convention.  FCC/BCC are both cubic symmetry for PF/IPF.
    csym = "hexag" if family.lower() in {"hcp", "hex", "hexag", "hexagonal"} else "cubic"
    t3 = EmbeddedTexture3PoleFigure(grains=np.column_stack([texture.eulers, texture.weights]), csym=csym)
    tex_mode = mode or getattr(style, "texture_mode", "fill") or "fill"
    dph = float(getattr(style, "texture3_dph", max(360.0 / max(style.bins, 1), 5.0)))
    dth = float(getattr(style, "texture3_dth", max(90.0 / max(style.bins//2, 1), 5.0)))
    n_rim = int(getattr(style, "texture3_n_rim", max(1, int(round(style.smooth)))))
    mn = getattr(style, "texture3_mn", None)
    mx = getattr(style, "texture3_mx", None)
    ix = str(getattr(style, "texture3_ix", "RD"))
    iy = str(getattr(style, "texture3_iy", "TD"))
    rot = float(getattr(style, "texture3_rot", 0.0))
    log_levels = bool(getattr(style, "texture3_log", False))
    t3.pf_new(fig=fig, proj=proj, poles=poles_or_dirs, ix=ix, iy=iy,
              mode=tex_mode, dth=dth, dph=dph, n_rim=n_rim,
              mn=mn, mx=mx, lev_norm_log=log_levels, nlev=int(style.levels),
              levels=None, cmap=style.cmap, rot=rot, ires=style.grid,
              ismooth=style.smooth, style=style)

# =============================================================================
# Main entry
# =============================================================================
def run_self_test(base: Path = Path.cwd()) -> None:
    """Small non-GUI smoke test for parsers and PF/IPF rendering."""
    vi = parse_vpsc8_in(base / "vpsc8.in")
    assert vi.phases, "VPSC8.IN phase parsing failed"
    for sx_name in ["FCC.sx", "Fe3.sx", "Mg.sx", "Mg_voce.sx", "Zr_DD.SX", "AL_MTS.sx"]:
        sx_path = base / sx_name
        if sx_path.exists():
            sx = parse_sx(sx_path)
            assert sx.elastic_matrix.shape == (6, 6), f"bad elastic matrix: {sx_name}"
    tex = parse_texture(base / "Rand500.tex")
    assert tex.n > 0 and abs(float(tex.weights.sum()) - 1.0) < 1e-8
    proc = parse_process(base / "rolling.3")
    assert proc.udot.shape == (3, 3)
    fig = Figure(figsize=(7, 3), dpi=100)
    style = PlotStyle(texture_mode="both", cmap="turbo", bins=65, levels=9,
                      smooth=1.2, colorbar=False, point_size=2.0)
    draw_pf_ipf_figure(fig, tex, style, kind="pf", family="cubic",
                       projection="equal-area", items=[[1,0,0],[1,1,0],[1,1,1]])
    fig.savefig(base / "vpsc_app_selftest_pf.png", dpi=120)
    fig2 = Figure(figsize=(4, 4), dpi=100)
    draw_pf_ipf_figure(fig2, tex, style, kind="ipf", family="cubic",
                       projection="equal-area", items=["ND"])
    fig2.savefig(base / "vpsc_app_selftest_ipf.png", dpi=120)
    fig3 = Figure(figsize=(6, 3), dpi=100)
    draw_pf_ipf_figure(fig3, tex, style, kind="pf", family="hcp",
                       projection="equal-area", items=[[0, 0, 0, 2], [1, 0, -1, 0]])
    fig3.savefig(base / "vpsc_app_selftest_hcp_pf.png", dpi=120)
    fig4 = Figure(figsize=(4, 4), dpi=100)
    draw_pf_ipf_figure(fig4, tex, style, kind="ipf", family="hcp",
                       projection="equal-area", items=["ND"])
    fig4.savefig(base / "vpsc_app_selftest_hcp_ipf.png", dpi=120)
    no_hist = base / "FE-Lij_hist_no.dat"
    if no_hist.exists():
        info = read_fe_velocity_history_info(no_hist)
        # Explicit opt-in stabilisation (never silent).
        stable, meta = stabilise_history_for_vpsc8(
            info.history, force=True
        )
        assert stable.shape[1] == 11 and meta.get("applied", False)
        assert equivalent_rate_from_L(stable[0, 1:10].reshape(3, 3)) > 0.1
    print("self-test passed")


def main() -> None:
    if "--self-test" in sys.argv:
        idx = sys.argv.index("--self-test")
        base = Path(sys.argv[idx + 1]) if len(sys.argv) > idx + 1 else Path.cwd()
        run_self_test(base)
        return
    if not GUI_AVAILABLE:
        sys.stderr.write(
            "The graphical interface requires Tkinter, which is not available "
            f"in this Python environment ({GUI_IMPORT_ERROR!r}).\n"
            "Install a Python build with Tk support (e.g. 'sudo apt install "
            "python3-tk' on Debian/Ubuntu) to launch the app.\n"
            "The scientific core can still be scripted, and 'python app.py "
            "--self-test [DIR]' runs without a display.\n"
        )
        sys.exit(1)
    try:
        app = VPSCApp()
    except tk.TclError as e:
        sys.stderr.write(f"Cannot start Tk: {e}\n")
        sys.exit(1)
    app.mainloop()


if __name__ == "__main__":
    main()