"""
TLM analysis script for Origin Pro.

Usage inside Origin Pro's Python console:
    import tlm_analysis
    tlm_analysis.run_tlm(r"C:\\path\\to\\folder", base_name="300CS_13_A4")

The folder must contain raw U-I .txt files named like
    <base_name>_1_2.txt, <base_name>_2_3.txt, ..., <base_name>_8_9.txt
where the trailing pair encodes the channel length:
    1_2 -> 320 um, 2_3 -> 280 um, 3_4 -> 240 um, 4_5 -> 200 um,
    5_6 -> 160 um, 6_7 -> 120 um, 7_8 -> 80 um,  8_9 -> 40 um.

If base_name is omitted and the folder holds only one series the script
auto-detects it; otherwise pass base_name explicitly.
"""

import os
import re
import numpy as np
import originpro as op

CHANNEL_LENGTH_MAP = {
    (1, 2): 320,
    (2, 3): 280,
    (3, 4): 240,
    (4, 5): 200,
    (5, 6): 160,
    (6, 7): 120,
    (7, 8): 80,
    (8, 9): 40,
}

FILE_RE = re.compile(r"^(?P<base>.+)_(?P<a>\d+)_(?P<b>\d+)\.txt$", re.IGNORECASE)

# Distinct, colour-blind-friendly cycle for I-V overlay (longest L first).
IV_COLORS = [
    "#000000",  # black
    "#0072BD",  # blue
    "#D95319",  # orange
    "#EDB120",  # yellow
    "#7E2F8E",  # purple
    "#77AC30",  # green
    "#4DBEEE",  # light blue
    "#A2142F",  # dark red
]


def _set(obj, attr, value):
    """setattr that silently ignores attribute names not supported by this
    Origin/originpro version. Used for all aesthetic styling so the analysis
    never crashes because of a property name change."""
    try:
        setattr(obj, attr, value)
    except Exception:
        pass


def _try(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception:
        pass


def _collect_files(folder, base_name=None):
    found = {}
    detected_bases = set()
    for fname in os.listdir(folder):
        m = FILE_RE.match(fname)
        if not m:
            continue
        key = (int(m.group("a")), int(m.group("b")))
        if key not in CHANNEL_LENGTH_MAP:
            continue
        base = m.group("base")
        detected_bases.add(base)
        if base_name is not None and base != base_name:
            continue
        found.setdefault(base, []).append((key, os.path.join(folder, fname)))

    if base_name is None:
        if len(detected_bases) == 1:
            base_name = detected_bases.pop()
        else:
            raise ValueError(
                "Multiple base names in folder, pass base_name=...: "
                + ", ".join(sorted(detected_bases))
            )

    entries = found.get(base_name, [])
    if not entries:
        raise FileNotFoundError(f"No TLM files for base '{base_name}' in {folder}")

    entries.sort(key=lambda x: CHANNEL_LENGTH_MAP[x[0]])
    return base_name, entries


def _load_iv(path):
    """Load U (V) and I from a .txt file. Auto-skips non-numeric header lines."""
    skip = 0
    with open(path, "r") as f:
        for line in f:
            try:
                parts = line.replace(",", " ").split()
                float(parts[0]); float(parts[1])
                break
            except (ValueError, IndexError):
                skip += 1
    data = np.loadtxt(path, skiprows=skip)
    return data[:, 0], data[:, 1]


def run_tlm(folder, base_name=None, make_iv_plot=True):
    """Run TLM analysis on a folder of U-I .txt files."""
    folder = os.path.abspath(folder)
    base_name, entries = _collect_files(folder, base_name)

    # Warn about missing channel lengths.
    present = {CHANNEL_LENGTH_MAP[key] for key, _ in entries}
    missing = sorted(set(CHANNEL_LENGTH_MAP.values()) - present, reverse=True)
    if missing:
        print(f"  WARNING: missing channel length(s) for '{base_name}': "
              + ", ".join(f"{L} um" for L in missing))

    # 1) Workbook for raw I-V curves -------------------------------------------------
    wb_raw = op.new_book("w", lname=f"{base_name}_IV")
    raw_sheet = wb_raw[0]
    raw_sheet.name = "IV"

    channel_lengths, resistances, slopes, intercepts = [], [], [], []

    col = 0
    for key, path in entries:
        L = CHANNEL_LENGTH_MAP[key]
        U, I = _load_iv(path)

        slope, intercept = np.polyfit(U, I, 1)
        if slope == 0:
            raise ValueError(f"Zero slope for L={L} um ({path})")

        channel_lengths.append(L)
        resistances.append(1.0 / slope)
        slopes.append(slope)
        intercepts.append(intercept)

        raw_sheet.from_list(col,     U.tolist(),
                            lname=f"U L={L} µm", units="V", axis="X")
        raw_sheet.from_list(col + 1, I.tolist(),
                            lname=f"I L={L} µm", units="A", axis="Y")
        col += 2

    channel_lengths = np.array(channel_lengths, dtype=float)
    resistances = np.array(resistances, dtype=float)

    # 2) Workbook for TLM summary ----------------------------------------------------
    wb_tlm = op.new_book("w", lname=f"{base_name}_TLM")
    tlm_sheet = wb_tlm[0]
    tlm_sheet.name = "TLM"

    tlm_sheet.from_list(0, channel_lengths.tolist(),
                        lname="Channel length L", units="µm", axis="X")
    tlm_sheet.from_list(1, resistances.tolist(),
                        lname="Total resistance R", units="Ω", axis="Y")
    tlm_sheet.from_list(2, slopes,
                        lname="dI/dU slope", units="A/V")

    # Linear fit R = m*L + b
    m_tlm, b_tlm = np.polyfit(channel_lengths, resistances, 1)
    Rc = b_tlm / 2.0
    Lt = -b_tlm / (2.0 * m_tlm) if m_tlm != 0 else float("nan")

    # Extend the fit line so both intercepts (y at L=0 and x at R=0) are
    # visible -- this is the canonical TLM plot.
    L_max = float(channel_lengths.max())
    x_neg = -2.0 * Lt if (not np.isnan(Lt) and Lt > 0) else -0.1 * L_max
    x_fit_min = min(0.0, x_neg) - 0.15 * L_max
    x_fit_max = L_max * 1.10
    L_fit = np.linspace(x_fit_min, x_fit_max, 200)
    R_fit = m_tlm * L_fit + b_tlm

    tlm_sheet.from_list(3, L_fit.tolist(),
                        lname="L fit", units="µm", axis="X")
    tlm_sheet.from_list(4, R_fit.tolist(),
                        lname="R fit", units="Ω", axis="Y")

    # 3) I-V overview plot -----------------------------------------------------------
    if make_iv_plot:
        gp_iv = op.new_graph(lname=f"{base_name}_IV_plot")
        gl_iv = gp_iv[0]
        for i, L in enumerate(channel_lengths):
            p = gl_iv.add_plot(raw_sheet, coly=2 * i + 1, colx=2 * i, type="l")
            _set(p, "color", IV_COLORS[i % len(IV_COLORS)])
            _set(p, "line_width", 2)
        gl_iv.rescale()
        try:
            gl_iv.axis("x").title = "Voltage U (V)"
            gl_iv.axis("y").title = "Current I (A)"
        except Exception:
            pass
        _try(op.lt_exec, f'win -o {gp_iv.lt_range()} {{legend -r 1;}}')

    # 4) Final TLM plot: scatter (L, R) + linear fit line ----------------------------
    gp_tlm = op.new_graph(lname=f"{base_name}_TLM_plot")
    gl_tlm = gp_tlm[0]

    plot_data = gl_tlm.add_plot(tlm_sheet, coly=1, colx=0, type="s")
    _set(plot_data, "symbol_kind", 2)        # 2 = circle
    _set(plot_data, "symbol_size", 10)
    _set(plot_data, "symbol_interior", 1)    # 1 = hollow
    _set(plot_data, "color", "#000000")

    plot_fit = gl_tlm.add_plot(tlm_sheet, coly=4, colx=3, type="l")
    _set(plot_fit, "color", "#C00000")
    _set(plot_fit, "line_width", 2.5)

    gl_tlm.rescale()

    # Force the axis to show both intercepts (the visual signature of TLM).
    R_max = float(resistances.max())
    R_min_plot = min(0.0, b_tlm * 1.1)
    R_max_plot = R_max * 1.15
    _try(gl_tlm.set_xlim, begin=x_fit_min, end=x_fit_max)
    _try(gl_tlm.set_ylim, begin=R_min_plot, end=R_max_plot)

    # Axis titles
    try:
        gl_tlm.axis("x").title = "Channel length L (µm)"
        gl_tlm.axis("y").title = "Total resistance R (Ω)"
    except Exception:
        pass

    # Bigger, bold axis titles and tick labels via LabTalk.
    _try(op.lt_exec,
         f'win -o {gp_tlm.lt_range()} {{'
         f'layer.x.label.pt = 20; layer.y.label.pt = 20;'
         f'layer.x.label.bold = 1; layer.y.label.bold = 1;'
         f'layer.xt.label.pt = 16; layer.yt.label.pt = 16;'
         f'}}')

    # Fit-info annotation on the plot. Position: upper-left of plot area.
    x_lbl = x_fit_min + 0.05 * (x_fit_max - x_fit_min)
    y_lbl = R_min_plot + 0.93 * (R_max_plot - R_min_plot)
    info = (f"R(L) = {m_tlm:.3g} L + {b_tlm:.3g}\r\n"
            f"R\\-(C) = {Rc:.3g} \\g(W)\r\n"
            f"L\\-(T) = {Lt:.3g} \\g(m)m")
    _try(op.lt_exec,
         f'win -o {gp_tlm.lt_range()} {{'
         f'label -s -a {x_lbl} {y_lbl} -j 0 -n fitinfo "{info}";'
         f'fitinfo.fsize = 18;'
         f'}}')

    # 5) Console summary -------------------------------------------------------------
    print(f"--- TLM results for '{base_name}' ---")
    for L, R in zip(channel_lengths, resistances):
        print(f"  L = {L:6.1f} um   R = {R:.6g} Ohm")
    print(f"  slope  m  = {m_tlm:.6g} Ohm/um")
    print(f"  inter. b  = {b_tlm:.6g} Ohm   (= 2 * Rc)")
    print(f"  Rc        = {Rc:.6g} Ohm")
    print(f"  Lt        = {Lt:.6g} um")

    return {
        "base_name": base_name,
        "channel_lengths_um": channel_lengths,
        "resistances_ohm": resistances,
        "slope_ohm_per_um": m_tlm,
        "intercept_ohm": b_tlm,
        "Rc_ohm": Rc,
        "Lt_um": Lt,
        "missing_lengths_um": missing,
    }
