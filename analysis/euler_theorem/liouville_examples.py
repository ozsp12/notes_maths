"""Generate SVG figures for the Liouville-formula lecture note."""

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent


def _polyline(xs, ys, xlim, ylim, box):
    x0, y0, w, h = box
    X = x0 + (np.asarray(xs) - xlim[0]) / (xlim[1] - xlim[0]) * w
    Y = y0 + h - (np.asarray(ys) - ylim[0]) / (ylim[1] - ylim[0]) * h
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(X, Y))


def _svg_document(title, xlabel, ylabel, xlim, ylim, curves, legends, xticks, yticks):
    width, height = 760, 520
    left, top, plot_w, plot_h = 90, 70, 620, 380
    box = (left, top, plot_w, plot_h)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-size="22" font-family="sans-serif">{escape(title)}</text>',
    ]

    for tick in xticks:
        x = left + (tick - xlim[0]) / (xlim[1] - xlim[0]) * plot_w
        lines += [
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}" stroke="#dddddd" stroke-width="1"/>',
            f'<text x="{x:.2f}" y="{top+plot_h+24}" text-anchor="middle" font-size="13" font-family="sans-serif">{tick:g}</text>',
        ]
    for tick in yticks:
        y = top + plot_h - (tick - ylim[0]) / (ylim[1] - ylim[0]) * plot_h
        lines += [
            f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>',
            f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-size="13" font-family="sans-serif">{tick:g}</text>',
        ]

    lines += [
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="black" stroke-width="1.3"/>',
        f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle" font-size="16" font-family="sans-serif">{escape(xlabel)}</text>',
        f'<text x="22" y="{top+plot_h/2}" text-anchor="middle" font-size="16" font-family="sans-serif" transform="rotate(-90 22 {top+plot_h/2})">{escape(ylabel)}</text>',
    ]

    for i, (xs, ys) in enumerate(curves):
        pts = _polyline(xs, ys, xlim, ylim, box)
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{colors[i]}" stroke-width="2.4"/>')

    legend_x, legend_y = left + 18, top + 18
    for i, label in enumerate(legends):
        yy = legend_y + 24 * i
        lines += [
            f'<line x1="{legend_x}" y1="{yy}" x2="{legend_x+34}" y2="{yy}" stroke="{colors[i]}" stroke-width="2.4"/>',
            f'<text x="{legend_x+44}" y="{yy+5}" font-size="13" font-family="sans-serif">{escape(label)}</text>',
        ]

    lines.append('</svg>')
    return "\n".join(lines) + "\n"


def plot_volume_factors() -> Path:
    t = np.linspace(0.0, 3.0, 140)
    curves = [(t, np.exp(t)), (t, np.ones_like(t)), (t, np.exp(-t))]
    svg = _svg_document(
        "Volume factor V(t)/V(0) = exp(c t)",
        "t",
        "V(t)/V(0)",
        (0.0, 3.0),
        (0.0, 21.0),
        curves,
        ["tr A = +1", "tr A = 0", "tr A = -1"],
        np.arange(0.0, 3.01, 0.5),
        np.arange(0.0, 20.01, 2.5),
    )
    output = HERE / "figures_volume_factors.svg"
    output.write_text(svg, encoding="utf-8")
    return output


def oscillator_rhs(t, state, gamma=0.12, omega0=1.0):
    x, v = state
    return np.array([v, -omega0**2 * x - 2.0 * gamma * v])


def plot_oscillator_phase_space() -> Path:
    t = np.linspace(0.0, 30.0, 420)
    undamped = solve_ivp(lambda s, y: oscillator_rhs(s, y, 0.0, 1.0), (0.0, 30.0), [1.0, 0.0], t_eval=t)
    damped = solve_ivp(lambda s, y: oscillator_rhs(s, y, 0.12, 1.0), (0.0, 30.0), [1.0, 0.0], t_eval=t)
    curves = [(undamped.y[0], undamped.y[1]), (damped.y[0], damped.y[1])]
    svg = _svg_document(
        "Damped harmonic oscillator: phase-space trajectories",
        "x",
        "v = dx/dt",
        (-1.1, 1.1),
        (-1.1, 1.1),
        curves,
        ["gamma = 0", "gamma = 0.12"],
        np.arange(-1.0, 1.01, 0.25),
        np.arange(-1.0, 1.01, 0.25),
    )
    output = HERE / "figures_oscillator_phase_space.svg"
    output.write_text(svg, encoding="utf-8")
    return output


def main() -> None:
    plot_volume_factors()
    plot_oscillator_phase_space()


if __name__ == "__main__":
    main()
