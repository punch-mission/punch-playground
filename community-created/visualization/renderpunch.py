#!/usr/bin/env python3
"""
renderpunch.py — Render PUNCH FITS mosaics to annotated JPEGs.

A small rendering toolkit for turning PUNCH mosaic data products into display
JPEGs (and, optionally, downsampled FITS). Everything is collected behind a
single ``RenderPUNCH`` driver that walks a directory of frames, plus the
lower-level helpers it is built from (load_frame, render_frame,
render_sequence, save_fits, …) for callers that need finer control. Every
rendering choice — the photometric stretch (VMIN/VMAX/GAMMA), the missing-data
thresholds, the output size, the persistence-of-vision filter, the Sun disk,
and the annotation slug/stamp — is a keyword argument with a module-level
default.

Works for any PUNCH mosaic product — CTM, PTM, CAM, PAM. The 2-D products (CTM,
CAM) are rendered as-is; the polarized products (PTM, PAM) arrive as a 3-D cube
whose leading axis runs over (B, pB, pB'), and the `plane` argument selects which
2-D plane to render (default 'B') or the derived 'opB' magnitude
sqrt(pB**2 + pB'**2).

The per-frame pipeline is: read the FITS mosaic and demote missing/out-of-range
pixels to NaN; block-average from the native grid down to ``out_size`` in linear
B-sun units (so the downsample doesn't distort the photometry); optionally
combine each frame with its neighbours via a persistence-of-vision filter
(exponential, centered rank, or gap-filling persistent rank); optionally rescale
each pixel by a power of its solar elongation; stretch with a gamma-corrected
power law under the PUNCH colormap; overlay a to-scale solar disk and text
annotations; and write the result as a JPEG.

Jupyter usage
-------------
    from pathlib import Path
    from renderpunch import RenderPUNCH

    # Reproduce the notebook's first "render all frames" loop (2-D CTM):
    RenderPUNCH(Path('data/CTM'), Path('data/CTM-jpeg'))

    # Polarized 3-D product (PTM), rendering the pB plane:
    RenderPUNCH(Path('data/PTM'), Path('data/PTM-pB-jpeg'), plane='pB')

    # Persistent (non-causal centered-rank + gap-filling) render, also dumping
    # the filtered stream as downsampled FITS for a later photometric mix:
    RenderPUNCH(Path('data/CTM'), Path('data/CTM-jpeg-persistent'),
                persist_mode='persistent-rank', n_frames=5, m_rank=1, kappa=0.5,
                name=None, fits_dir=Path('data/CTM-persistent-fits'))

    # Display a single frame inline (a punchbowl NDCube or a FITS path) instead
    # of writing a JPEG -- run %matplotlib inline first (import forces Agg):
    show_frame(cube_or_path, vmin=2e-15, vmax=6e-13, constellations=True)

The lower-level helpers (load_frame, render_frame, render_sequence, save_fits,
show_frame, …) remain importable for callers that need finer control than
RenderPUNCH offers.
"""

import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body
from scipy.ndimage import map_coordinates, zoom
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from punchbowl.data.visualize import cmap_punch

_FONT_PATH = findfont(FontProperties(family='monospace'))


# --------------------------------------------------------------------------- #
# Default parameters (the notebook's "Parameters" globals).  These are the
# defaults for RenderPUNCH and the helpers below; override per call as needed.
# --------------------------------------------------------------------------- #

VMIN  = 3e-15    # B-sun, lower clip of the color stretch
VMAX  = 6e-13    # B-sun, upper clip of the color stretch
GAMMA = 1 / 2.4  # gamma correction applied for JPEG display (encode with 1/2.2)

# Anything below MIN_VALID (or above MAX_VALID) is treated as missing data rather
# than as a real measurement: uncovered mosaic pixels come through as zeros
# rather than NaN, and this catches them along with noise excursions.  Applied at
# load time, before the block-average, so sub-threshold pixels don't drag down
# the downsampled value -- and so every persistence mode inherits it, since they
# all treat non-finite as missing.
MIN_VALID = VMIN / 10   # B-sun
MAX_VALID = VMAX * 10   # B-sun

OUT_SIZE = 4096         # target square output size [px]
JPEG_QUALITY = 99

# Radial (elongation) scaling: multiply each pixel by (eps/eps0)**EPS_POWER,
# where eps is the pixel's elongation -- its angular distance from the Sun in the
# image plane [deg] -- and EPS0 is the datum elongation at which the weight is 1.
# EPS_POWER = 0 (the default) disables it (every pixel x1); a positive power
# boosts the faint outer corona relative to the bright inner field.
EPS_POWER = 0.0         # exponent on (eps/eps0); 0 disables radial scaling
EPS0      = 10.0        # datum elongation [deg] where the weight is unity

# Which 2-D plane to pull from a 3-D polarized product (PTM/PAM): the leading
# axis runs over (B, pB, pB'), aka pBp.  Besides those three stored planes, the
# derived selector 'opB' renders the polarized-brightness magnitude
# sqrt(pB**2 + pB'**2).  Ignored for 2-D products (CTM/CAM).
PLANE = 'B'
_PLANES = {'tB': 0, 'B': 0, 'pB': 1, 'pBp': 2, "pB'": 2}

# A to-scale solar disk drawn at the center of the frame, for size reference.
DRAW_SUN     = True
SUN_DIAM_DEG = 0.6            # angular diameter of the drawn disk [deg]
SUN_COLOR    = (255, 255, 0)  # RGB

# Persistence of vision: how successive frames are combined before display.
#   'none'            -- no persistence, each frame rendered on its own
#   'exponential'     -- running IIR blend, weight EPSILON on the new frame
#                        (causal)
#   'rank'            -- pixelwise M_RANK'th-smallest value over a window of
#                        N_FRAMES frames CENTERED on the frame being rendered
#                        (non-causal; features aren't displaced in time)
#   'persistent-rank' -- the centered rank filter above, then a z-filter with
#                        weight KAPPA over the rank-filtered stream, holding the
#                        last good value across coverage gaps
PERSIST_MODE = 'none'
EPSILON  = 0.2   # 'exponential' only: weight of the incoming frame
N_FRAMES = 5     # rank modes: centered window length in frames (forced odd)
M_RANK   = 1     # rank modes: order statistic, 0=smallest .. N_FRAMES-1=largest
KAPPA    = 0.5   # 'persistent-rank' only: weight of the incoming rank value

SLUG = 'PUNCH MOSAIC'   # optional dataset label shown in the upper-left

# Annotation text: base font size [px] and the corner the data-type/timestamp
# pair sits in. FONT_SIZE is 2/3 of the historical 56 px; pass fontscale=1.5 to
# render_frame / RenderPUNCH to recover that old size (fontscale multiplies it).
FONT_SIZE = 56 * 2 / 3         # ~37 px
STAMP_POS = 'upper-right'      # 'upper-right' or 'lower-right'

# Sky overlays (both off by default). The PUNCH mosaics carry a celestial WCS
# (alternate key 'A', RA/Dec ARC projection) alongside the primary helioprojective
# one, so stars and solar-system bodies can be projected into the frame.
#   constellations : draw IAU constellation stick figures
#   planet_labels  : mark solar-system bodies (see PLANET_BODIES) with a text label
CONSTELLATION_COLOR = (90, 130, 210)     # RGB for the constellation lines
CONSTELLATION_MAX_ELONG = 47.0           # deg; clip constellation lines to within
                                         # this elongation (angular dist. from Sun)
PLANET_COLOR        = (255, 235, 130)    # RGB for planet markers + labels
# Bodies for the planet-label overlay: naked-eye planets + Moon + outer giants.
# Only those falling within the frame are drawn.
PLANET_BODIES = ('mercury', 'venus', 'mars', 'jupiter', 'saturn',
                 'uranus', 'neptune', 'moon')

# Public-domain constellation stick-figure lines (RA/Dec polylines), fetched from
# the BSD-licensed d3-celestial project and stashed next to this module:
#   https://github.com/ofrohn/d3-celestial  (data/constellations.lines.json)
_CONSTELLATION_FILE = Path(__file__).with_name('constellation_lines.json')
_constellation_cache = None   # (ra_flat, dec_flat, [(start, stop), ...]) once loaded

# Inner-corona composite: fill the central hole with the nearest-in-time CCOR
# (GOES-19 coronagraph) frame, reprojected onto the PUNCH grid. Off unless a
# directory is given. CCOR is in mean-solar-brightness (MSB), the same scale as
# PUNCH's B-sun, so ccor_scale=1.0 is photometric; the inner corona is far
# brighter than the outer field, so lower it if it saturates the shared stretch.
CCOR_GLOB   = '*.fits'
CCOR_SCALE  = 1.0
_ccor_index_cache = {}        # (dir, glob) -> sorted [(datetime, Path), ...]

# Minimum-image subtraction: subtract a static per-pixel floor (e.g. the
# per-pixel 5th-percentile background built elsewhere) from each frame before
# the stretch, to remove the F-corona / stray-light / pedestal. Off unless a
# path is given. The floor is read raw (not through load_frame) so its
# invalid-marking large values survive, and cached by (path, mtime).
_min_image_cache = {}


def _plane_index(plane):
    """Resolve a polarization-plane selector to a 0-based index into the leading
    axis of a 3-D PTM/PAM cube: 0='B', 1='pB', 2='pBp' (aka pB'). Accepts either
    one of those names or an integer index. (The derived selector 'opB' is not a
    single stored plane; it is handled by _extract_plane, not here.)"""
    if isinstance(plane, str):
        try:
            return _PLANES[plane]
        except KeyError:
            raise ValueError(f"unknown plane {plane!r}; expected one of "
                             f"{sorted(_PLANES)}, 'opB', or an integer index")
    return int(plane)


def _extract_plane(cube, plane):
    """Pull the requested 2-D plane out of a 3-D PTM/PAM cube whose leading axis
    runs over (B, pB, pB'). For the stored planes, `plane` is a name or integer
    index resolved by _plane_index. The derived selector 'opB' (case-insensitive)
    instead returns the polarized-brightness magnitude sqrt(pB**2 + pB'**2),
    combining the pB and pB' planes rather than selecting one."""
    if isinstance(plane, str) and plane.lower() == 'opb':
        pB, pBp = cube[_PLANES['pB']], cube[_PLANES['pBp']]
        return np.sqrt(pB ** 2 + pBp ** 2)
    return cube[_plane_index(plane)]


def _plane_label(plane):
    """Corner-annotation label for a polarization plane: 'tB' (total brightness,
    the 'B'/index-0 plane; the longer 'tB' form is preferred over 'B'), 'pB',
    "pB'", or '°pB' (the derived 'opB' magnitude). Accepts the same selectors as
    _extract_plane -- a name, an integer index into (B, pB, pB'), or 'opB'."""
    if isinstance(plane, str):
        return {'b': 'tB', 'tb': 'tB', 'pb': 'pB', 'pbp': "pB'",
                "pb'": "pB'", 'opb': '°pB'}[plane.lower()]
    return {0: 'tB', 1: 'pB', 2: "pB'"}[int(plane)]


# --------------------------------------------------------------------------- #
# Sky overlays (constellation lines and planet labels)
# --------------------------------------------------------------------------- #

def _load_constellation_lines():
    """Load the bundled constellation stick figures once, flattened for a single
    batched projection per frame. Returns (ra, dec, segments): 1-D RA/Dec arrays
    [deg] of every polyline vertex concatenated, and a list of (start, stop)
    index ranges, one per polyline, into those arrays."""
    global _constellation_cache
    if _constellation_cache is None:
        with open(_CONSTELLATION_FILE) as fh:
            gj = json.load(fh)
        ra, dec, segments = [], [], []
        for feat in gj['features']:
            for poly in feat['geometry']['coordinates']:
                arr = np.asarray(poly, dtype=float)      # (n, 2): [RA(-180..180), Dec]
                start = len(ra)
                ra.extend(arr[:, 0]); dec.extend(arr[:, 1])
                segments.append((start, len(ra)))
        _constellation_cache = (np.asarray(ra), np.asarray(dec), segments)
    return _constellation_cache


def _celestial_render_wcs(header, render_h):
    """Celestial (RA/Dec) WCS for the *rendered* grid. The mosaics store a
    celestial WCS as alternate key 'A' (falling back to deriving one from the
    helioprojective WCS), calibrated to the native NAXIS; it is rescaled here to
    the possibly-downsampled render grid the same way save_fits/_draw_sun handle
    the block-average (CRPIX toward the new center, CDELT up by the factor)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            w = WCS(header, key='A')
            if not w.has_celestial:
                raise ValueError('no celestial alternate WCS')
        except Exception:
            from punchbowl.data.wcs import calculate_celestial_wcs_from_helio
            w = calculate_celestial_wcs_from_helio(
                WCS(header).copy(), Time(header['DATE-OBS']),
                (header['NAXIS2'], header['NAXIS1']))
    factor = header['NAXIS2'] / render_h
    if factor != 1:
        w = w.deepcopy()
        w.wcs.crpix = (w.wcs.crpix - 0.5) / factor + 0.5
        w.wcs.cdelt = np.asarray(w.wcs.cdelt) * factor
    return w


def _sky_to_image(wcel, sky, render_h):
    """Project a SkyCoord to rendered-image pixel coords (x right, y down). The
    render flips the data vertically (np.flipud), so the WCS row is mirrored."""
    x, y = wcel.world_to_pixel(sky)
    return np.atleast_1d(np.asarray(x, float)), (render_h - 1) - np.atleast_1d(np.asarray(y, float))


def _draw_constellations(img, wcel, header, color, width, max_elong):
    """Draw the constellation stick figures onto `img` (a PIL image on the render
    grid) through the celestial WCS `wcel`. Vertices are projected in one batch.

    Everything is clipped to within `max_elong` degrees of the Sun: vertices
    beyond that elongation are dropped, and a segment straddling the limit is
    truncated at it. Because the helioprojective WCS is an ARC (zenithal
    equidistant) projection centered on the Sun, elongation is exactly a radial
    pixel distance from the frame center, so the limit is a circle there. A
    segment is also skipped if either endpoint is non-finite or the projection
    is implausibly long (a wrap artifact near the ARC far side)."""
    ra, dec, segments = _load_constellation_lines()
    xs, ys = _sky_to_image(wcel, SkyCoord(ra * u.deg, dec * u.deg, frame='icrs'),
                           img.height)
    draw = ImageDraw.Draw(img)
    W, H = img.width, img.height
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    deg_per_px = abs(header['CDELT2']) * (header['NAXIS2'] / H)
    limit = max_elong / deg_per_px               # elongation limit as a px radius
    rad = np.hypot(xs - cx, ys - cy)             # per-vertex elongation [px]
    max_seg = 0.5 * max(W, H)                     # px; longer => projection wrap

    def _clip(xi, yi, xo, yo):
        # Point where the segment from inside (xi,yi) to outside (xo,yo) crosses
        # the elongation-limit circle centered at (cx,cy).
        dx, dy = xo - xi, yo - yi
        fx, fy = xi - cx, yi - cy
        a = dx * dx + dy * dy
        if a == 0:
            return xi, yi
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - limit * limit
        disc = b * b - 4 * a * c
        if disc < 0:
            return xi, yi
        t = min(1.0, max(0.0, (-b + np.sqrt(disc)) / (2 * a)))
        return xi + t * dx, yi + t * dy

    for start, stop in segments:
        for i in range(start, stop - 1):
            x0, y0, x1, y1 = xs[i], ys[i], xs[i + 1], ys[i + 1]
            if not (np.isfinite(x0) and np.isfinite(y0)
                    and np.isfinite(x1) and np.isfinite(y1)):
                continue
            if abs(x1 - x0) > max_seg or abs(y1 - y0) > max_seg:
                continue
            in0, in1 = rad[i] <= limit, rad[i + 1] <= limit
            if not in0 and not in1:
                continue                          # both outside -> omit
            if in0 and not in1:
                x1, y1 = _clip(x0, y0, x1, y1)     # truncate at the limit
            elif in1 and not in0:
                x0, y0 = _clip(x1, y1, x0, y0)
            draw.line([(x0, y0), (x1, y1)], fill=color, width=width)


def _draw_planet_labels(img, wcel, header, bodies, color, font, marker_r):
    """Mark solar-system bodies within the frame with a small ring and a text
    label. Positions are geocentric-apparent (astropy get_body), matching the
    mosaics' Earth-centered observer metadata; only bodies landing inside the
    frame are drawn."""
    draw = ImageDraw.Draw(img)
    W, H = img.width, img.height
    t = Time(header['DATE-OBS'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name in bodies:
            try:
                body = get_body(name, t)
            except Exception:
                continue
            sky = SkyCoord(body.ra, body.dec, frame='icrs')
            xs, ys = _sky_to_image(wcel, sky, H)
            x, y = float(xs[0]), float(ys[0])
            if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < W and 0 <= y < H):
                continue
            draw.ellipse([x - marker_r, y - marker_r, x + marker_r, y + marker_r],
                         outline=color, width=max(1, marker_r // 3))
            draw.text((x + marker_r + 4, y - marker_r), name.capitalize(),
                      fill=color, font=font)


# --------------------------------------------------------------------------- #
# Inner-corona composite (CCOR)
# --------------------------------------------------------------------------- #

def _ccor_index(ccor_dir, glob_pat):
    """Timestamp index for a CCOR directory, built once and cached: a sorted list
    of (datetime, Path) parsed from the YYYYMMDDThhmmss stamp in each filename."""
    key = (str(ccor_dir), glob_pat)
    if key not in _ccor_index_cache:
        idx = []
        for f in Path(ccor_dir).glob(glob_pat):
            m = re.search(r'(\d{8})T(\d{6})', f.name)
            if m:
                idx.append((datetime.strptime(m.group(1) + m.group(2),
                                               '%Y%m%d%H%M%S'), f))
        _ccor_index_cache[key] = sorted(idx)
    return _ccor_index_cache[key]


def _closest_ccor(ccor_dir, when, glob_pat, max_dt):
    """Path of the CCOR frame nearest `when` (a datetime), or None if the index
    is empty or the nearest frame is more than `max_dt` minutes away."""
    idx = _ccor_index(ccor_dir, glob_pat)
    if not idx:
        return None
    stamp, path = min(idx, key=lambda it: abs((it[0] - when).total_seconds()))
    if max_dt is not None and abs((stamp - when).total_seconds()) > max_dt * 60:
        return None
    return path


def _composite_ccor(header, shape, ccor_dir, glob_pat, max_dt):
    """Reproject the nearest-in-time CCOR frame onto the PUNCH render grid.

    Returns (ccor, valid) both of `shape` in the same (un-flipped) orientation as
    the freshly-loaded PUNCH data: `ccor` the resampled MSB values and `valid` a
    boolean mask of the pixels actually covered by CCOR. Returns None when no
    suitable CCOR frame is found.

    The resample maps each PUNCH render pixel through its helioprojective WCS to a
    sky angle and back through CCOR's WCS to a CCOR pixel, sampling there. Both are
    treated as plain angular (HPLN/HPLT) coordinates rather than observer-aware
    solar frames -- an angular alignment that is well-conditioned and sub-pixel
    accurate here since both observers sit ~1 AU from the Sun, and which avoids
    the degenerate full-frame Helioprojective transform reproject would attempt
    once sunpy has registered those frame types."""
    when = Time(header['DATE-OBS']).to_datetime()
    cfile = _closest_ccor(ccor_dir, when, glob_pat, max_dt)
    if cfile is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with fits.open(cfile) as hdul:
            chdu = hdul[1]                          # CompImageHDU with the image
            cdata = chdu.data.astype(np.float32)
            cwcs = WCS(chdu.header)
        wt = WCS(header).deepcopy()
        factor = header['NAXIS2'] / shape[0]
        if factor != 1:
            wt.wcs.crpix = (wt.wcs.crpix - 0.5) / factor + 0.5
            wt.wcs.cdelt = np.asarray(wt.wcs.cdelt) * factor

        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w]
        lon, lat = wt.all_pix2world(xx, yy, 0)      # PUNCH pixel -> HPLN/HPLT [deg]
        cx, cy = cwcs.all_world2pix(lon, lat, 0)    # -> CCOR pixel
    ch, cw = cdata.shape
    valid = (np.isfinite(cx) & np.isfinite(cy)
             & (cx >= 0) & (cx <= cw - 1) & (cy >= 0) & (cy <= ch - 1))
    ccor = map_coordinates(cdata, [cy, cx], order=1, mode='constant', cval=0.0)
    return ccor, valid


def _min_image_for(path, shape):
    """Load a minimum/floor image (read raw so large invalid-markers survive) and
    return it resampled to `shape`. The file is cached by (path, mtime); the
    resample is a plain zoom, correct because the floor and the frame both tile
    the same full square field, just at different sampling."""
    path = Path(path)
    key = (str(path), path.stat().st_mtime)
    if key not in _min_image_cache:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with fits.open(path) as hdul:
                _min_image_cache[key] = hdul[1].data.astype(np.float32)
    floor = _min_image_cache[key]
    if floor.shape != tuple(shape):
        floor = zoom(floor, (shape[0] / floor.shape[0], shape[1] / floor.shape[1]),
                     order=1)
    return floor


# --------------------------------------------------------------------------- #
# Annotation
# --------------------------------------------------------------------------- #

def _annotate(img: Image.Image, type_label: str, date_label: str, slug: str,
              font_size: float = FONT_SIZE, margin: int = 20,
              stamp_pos: str = STAMP_POS, logo=None,
              logoheight: float = 4) -> Image.Image:
    """Draw the slug in the upper-left, and the data-type / date-stamp pair in a
    right-hand corner chosen by `stamp_pos`, in the black margins outside the
    circular PUNCH FOV.

    stamp_pos='upper-right' (default) keeps the historical layout: the date on
    top with the data type just below it, growing downward from the top margin.
    stamp_pos='lower-right' anchors the pair to the bottom-right corner instead,
    with the data type ABOVE the timestamp so the two sit neatly in the
    otherwise-empty lower corner.

    If `logo` is given (a path to a transparency-capable image, e.g. a PNG), it
    is alpha-composited into the lower-left corner, scaled to `logoheight`
    annotation line-heights tall and inset by the same `margin` the corner text
    uses. `logo` may instead be a tuple/list of paths, which are laid out
    side-by-side sharing a common bottom edge; pass a matching tuple/list for
    `logoheight` to size each one independently (a scalar sizes them all).

    Sized for both analyst review and talk slides; text may slightly overrun the
    FOV margin at large sizes, which is fine."""
    pos = stamp_pos.lower().replace(' ', '-')
    if pos not in ('upper-right', 'lower-right'):
        raise ValueError(f"stamp_pos must be 'upper-right' or 'lower-right'; "
                         f"got {stamp_pos!r}")

    draw = ImageDraw.Draw(img)
    px = int(round(font_size))
    font = ImageFont.truetype(_FONT_PATH, px)
    step = px + 6                        # line-to-line advance

    if slug:
        draw.text((margin, margin), slug, fill='white', font=font)

    if logo:
        # One or several logos along the lower-left, alpha-composited so
        # transparency shows through. A tuple/list of paths lays them out
        # side-by-side (left to right); a tuple/list `logoheight` sizes each
        # independently, while a scalar applies to all. Each is scaled (aspect
        # preserved) to its height in line-heights, and all share a common bottom
        # edge inset by `margin`.
        logos = list(logo) if isinstance(logo, (tuple, list)) else [logo]
        heights = (list(logoheight) if isinstance(logoheight, (tuple, list))
                   else [logoheight] * len(logos))
        if len(heights) != len(logos):
            raise ValueError(f"logoheight has {len(heights)} entries but there "
                             f"are {len(logos)} logo(s)")
        x = margin
        for path, lh in zip(logos, heights):
            badge = Image.open(path).convert('RGBA')
            target_h = max(1, int(round(lh * step)))
            target_w = max(1, round(badge.width * target_h / badge.height))
            badge = badge.resize((target_w, target_h), Image.LANCZOS)
            img.paste(badge, (x, img.height - margin - target_h), badge)
            x += target_w + margin       # gap before the next logo

    def _right(line, y):
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.text((img.width - margin - (bbox[2] - bbox[0]), y),
                  line, fill='white', font=font)

    if pos == 'upper-right':
        # Date on top, data type below it, growing down from the top margin.
        y = margin
        for line in (date_label, type_label):
            _right(line, y)
            y += step
    else:
        # Bottom-up from the bottom margin: timestamp on the last line, data
        # type on the line above it.
        y = img.height - margin - px
        for line in (date_label, type_label):
            _right(line, y)
            y -= step

    return img


def _draw_sun(rgb, header, diam_deg: float = SUN_DIAM_DEG,
              color=SUN_COLOR):
    """Composite a to-scale solar disk into the center of an RGB array.

    The plate scale comes from the header's CDELT, rescaled for the downsample
    (the rendered array is smaller than NAXIS, so each output pixel covers
    proportionally more sky). The disk is centered on the array rather than on
    CRPIX; for these mosaics the two agree to within half a pixel at 4k.

    Anti-aliased by computing each pixel's fractional coverage analytically --
    alpha ramps linearly from 1 to 0 across the one-pixel band straddling the
    limb -- and alpha-compositing. That is smoother than a hard mask and
    cheaper than supersampling, which matters because at 0.6 deg the disk is
    only a few pixels across and a jagged limb would be obvious."""
    h, w = rgb.shape[:2]
    deg_per_px = abs(header['CDELT2']) * (header['NAXIS2'] / h)
    radius = 0.5 * diam_deg / deg_per_px          # [output px]

    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    # Work in a small box around the disk rather than over the whole frame.
    pad = int(np.ceil(radius)) + 2
    y0, y1 = max(0, int(cy) - pad), min(h, int(cy) + pad + 1)
    x0, x1 = max(0, int(cx) - pad), min(w, int(cx) + pad + 1)

    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = np.hypot(yy - cy, xx - cx)
    alpha = np.clip(radius + 0.5 - dist, 0.0, 1.0)[..., np.newaxis]

    patch = rgb[y0:y1, x0:x1].astype(np.float32)
    blended = patch * (1 - alpha) + np.asarray(color, np.float32) * alpha
    rgb[y0:y1, x0:x1] = np.rint(blended).astype(np.uint8)
    return rgb


_ELON_CACHE = {}


def _elongation_scale(shape, header, eps_power: float = EPS_POWER,
                      eps0: float = EPS0):
    """Per-pixel radial weight (eps/eps0)**eps_power for the (downsampled) render
    grid, where eps is a pixel's elongation -- its angular distance from the Sun
    in the image plane [deg] -- and eps0 the datum elongation at which the weight
    is unity. Returns a float32 array shaped like `shape`.

    The PUNCH mosaics use a Sun-centred zenithal-equidistant (HPLN/HPLT-ARC)
    projection, so a pixel's elongation is simply its radial distance from the
    reference point in the projection plane:

        wx = CRVAL1 + CDELT1 * (x - CRPIX1),   wy likewise for y   (FITS 1-indexed)
        eps = hypot(wx, wy)                                          [deg]

    CDELT/CRPIX are rescaled for the downsample exactly as save_fits and
    _draw_sun do (bigger output pixels cover proportionally more sky). The map is
    cached on the WCS and parameters, which are constant across a data set, so
    the grid is built only once and every subsequent frame reuses it."""
    h, w = shape
    factor = header['NAXIS2'] / h                 # native -> downsampled
    cd1, cd2 = header['CDELT1'] * factor, header['CDELT2'] * factor
    cp1 = (header['CRPIX1'] - 0.5) / factor + 0.5
    cp2 = (header['CRPIX2'] - 0.5) / factor + 0.5
    cv1, cv2 = header.get('CRVAL1', 0.0), header.get('CRVAL2', 0.0)

    key = (h, w, cd1, cd2, cp1, cp2, cv1, cv2, eps_power, eps0)
    scale = _ELON_CACHE.get(key)
    if scale is None:
        x = np.arange(1, w + 1)                   # FITS 1-indexed pixel centres
        y = np.arange(1, h + 1)
        wx = cv1 + cd1 * (x - cp1)
        wy = cv2 + cd2 * (y - cp2)
        eps = np.hypot(wy[:, np.newaxis], wx[np.newaxis, :])   # [deg], (h, w)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = ((eps / eps0) ** eps_power).astype(np.float32)
        _ELON_CACHE[key] = scale
    return scale


# --------------------------------------------------------------------------- #
# Load / render / persistence
# --------------------------------------------------------------------------- #

def load_frame(fits_path: Path, out_size: int = OUT_SIZE,
               min_valid: float = MIN_VALID, max_valid: float = MAX_VALID,
               plane=PLANE):
    """Read one PUNCH FITS mosaic and return (data, header), block-averaged down
    to out_size x out_size. Averaging happens in linear (B-sun) units, before any
    stretching, so the downsample doesn't distort the photometric scaling; doing
    it here rather than at render time also keeps the persistence window small.

    Polarized products (PTM/PAM) store a 3-D cube whose leading axis runs over
    (B, pB, pB'); `plane` selects which 2-D plane to render (a name from
    {'B','pB','pBp'/"pB'"} or an integer index, default 'B'). The derived
    selector 'opB' instead renders the polarized-brightness magnitude
    sqrt(pB**2 + pB'**2). 2-D products (CTM/CAM) are used as-is and `plane` is
    ignored.

    Values below `min_valid` (or above `max_valid`) are demoted to NaN before
    averaging: uncovered mosaic pixels arrive as zeros rather than NaN, and this
    catches those along with noise excursions, so they neither drag down the
    block average nor count as samples in the rank filter. Every persistence mode
    inherits the threshold, since they all treat non-finite as missing."""
    with fits.open(fits_path) as hdul:
        header = hdul[1].header
        data = hdul[1].data                      # 'PRIMARY DATA ARRAY' HDU
    return _process_frame(data, header, out_size, min_valid, max_valid, plane)


def _process_frame(data, header, out_size, min_valid, max_valid, plane):
    """Shared post-read prep for a raw (data, header): pick the plane of a 3-D
    cube, demote out-of-range pixels to NaN, and block-average to out_size. Used
    by both load_frame (from a file) and show_frame (from a file or NDCube)."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 3:                           # PTM/PAM cube: (B, pB, pB')
        data = _extract_plane(data, plane)

    # NaN-safe: the comparison is False for existing NaNs, which stay NaN.
    data = np.where(data < min_valid, np.nan, data)
    data = np.where(data > max_valid, np.nan, data)

    h, w = data.shape
    factor = h // out_size
    if factor > 1:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN blocks
            data = np.nanmean(
                data.reshape(out_size, factor, out_size, factor), axis=(1, 3))
    return data, header


def render_frame(data, header, out_path: Path = None,
                 vmin: float = VMIN, vmax: float = VMAX, gamma: float = GAMMA,
                 cmap=cmap_punch,
                 quality: int = JPEG_QUALITY, slug: str = SLUG,
                 draw_timestamp: bool = True, type_label=None,
                 month_abbr: bool = False,
                 draw_sun: bool = DRAW_SUN, sun_diam_deg: float = SUN_DIAM_DEG,
                 sun_color=SUN_COLOR,
                 stamp_pos: str = STAMP_POS, font_size: float = FONT_SIZE,
                 fontscale: float = 1.0, logo=None,
                 logoheight: float = 4, crop_inner: float = None,
                 crop_outer: float = None, plane=PLANE,
                 eps_power: float = EPS_POWER, eps0: float = EPS0,
                 constellations: bool = False, planet_labels: bool = False,
                 constellation_color=CONSTELLATION_COLOR,
                 constellation_max_elong: float = CONSTELLATION_MAX_ELONG,
                 planet_color=PLANET_COLOR, planet_bodies=PLANET_BODIES,
                 ccor_dir=None, ccor_scale: float = CCOR_SCALE,
                 ccor_glob: str = CCOR_GLOB, ccor_max_dt: float = None,
                 min_image=None, peg_median_to: float = None):
    """Stretch, colormap, annotate one already-loaded frame; return the finished
    PIL RGB image and, when `out_path` is given, also write it as a JPEG. Pass
    out_path=None to render without saving (e.g. for show_frame / inline display).

    `cmap` is the matplotlib colormap (a callable mapping normalized [0,1] to
    RGBA) applied after the vmin/vmax/gamma stretch; it defaults to the PUNCH
    colormap `cmap_punch`. Any CCOR fill is rendered through an R/B-swapped
    analog of this same colormap so it stays visually distinct.

    `draw_timestamp` toggles the data-type/timestamp pair. `type_label` overrides
    the auto data-product label (the header TYPECODE/OBSCODE, e.g. "QAM") with an
    arbitrary string such as "QuickLook data". `month_abbr` writes the timestamp
    month as a 3-letter abbreviation, e.g. "2026-Aug-04" rather than "2026-08-04".

    `stamp_pos` picks the corner for the data-type/timestamp pair ('upper-right'
    or 'lower-right'). `font_size` is the base annotation font in pixels
    (default FONT_SIZE, ~37 px) and `fontscale` multiplies it, so the rendered
    text is `font_size * fontscale` px; either can be used, e.g. font_size=56 or
    fontscale=1.5 both restore the historical 56 px. `logo` is an optional path
    to a transparency-capable image overlaid in the lower-left corner (or a
    tuple/list of paths, laid out side-by-side), and `logoheight` its size in
    annotation line-heights (a matching tuple/list sizes each logo).

    `plane` is the polarization plane this frame was loaded from; for polarized
    products (a 3-D PTM/PAM source, NAXIS==3) its label is appended to the
    data-type annotation, e.g. "PTM pB" or "PTM °pB". Ignored for 2-D products.

    `eps_power`/`eps0` apply a radial scaling in linear units before the stretch:
    each pixel is multiplied by (eps/eps0)**eps_power, where eps is its elongation
    (angular distance from the Sun in the image plane, deg) and eps0 the datum
    elongation (default 10 deg). eps_power=0 (the default) disables it.

    `crop_inner`, if given, is a radius in SOURCE pixels (native FITS resolution,
    before the downsample) around the frame center; everything inside it is
    blacked out. The Sun disk is drawn afterward, so it stays visible through the
    crop. None disables it.

    `crop_outer` is the complementary crop: a radius in the same SOURCE pixels
    around the frame center, outside of which everything is blacked out (trimming
    the outer field of view). It combines with `crop_inner` to leave an annulus.
    The Sun disk is likewise drawn afterward. None disables it.

    `constellations` overlays the IAU constellation stick figures, and
    `planet_labels` marks the solar-system bodies in `planet_bodies` with a text
    label; both are off by default and are projected through the mosaic's
    celestial WCS. `constellation_color`/`planet_color` set their RGB. The
    constellation lines are clipped to within `constellation_max_elong` degrees
    of the Sun (vertices beyond it dropped, crossing segments truncated at it).

    `ccor_dir`, if given, composites the inner corona from CCOR (GOES-19): the
    nearest-in-time frame in that directory is reprojected onto this grid, and
    each output pixel takes the PUNCH value where PUNCH is valid (>0) and the CCOR
    value where CCOR covers it and PUNCH does not -- filling the central hole. The
    composite is photometric (before the stretch), so `ccor_scale` multiplies the
    CCOR MSB values (=B-sun; 1.0 is physical) to tame the much brighter inner
    corona under the shared stretch. The CCOR fill is rendered through a blue
    analog of the PUNCH colormap so it reads as a distinct dataset. `ccor_glob`
    selects the files and `ccor_max_dt` (minutes) caps how far in time a match may
    be (None = no cap). A `crop_inner` circle still cleans ragged edges but spares
    CCOR-filled pixels.

    `min_image`, if given, is a path to a static per-pixel floor image (e.g. a
    per-pixel percentile background) subtracted from the frame in linear units
    before anything else, to remove the F-corona / stray-light / pedestal. It is
    resampled to the frame if their grids differ; NaN frame pixels stay invalid,
    and pixels where the floor is a large invalid-marker go negative and hence
    dark. Subtraction precedes the CCOR composite, so CCOR fill is unaffected.

    `peg_median_to`, if given, is a target median (in the same linear units,
    typically a reference frame's median after F-corona removal): the frame's own
    median is measured after `min_image` subtraction and a scalar offset is added
    so the frame's median lands on that target, holding the overall level steady
    from frame to frame. RenderPUNCH computes this target from `peg_median_frame`."""
    if min_image is not None:
        data = data - _min_image_for(min_image, data.shape)

    if peg_median_to is not None:
        # Shift the whole frame by a scalar so its median matches the reference,
        # removing frame-to-frame level flicker. Measured on valid pixels only.
        cur = np.nanmedian(data)
        if np.isfinite(cur):
            data = data + (peg_median_to - cur)

    ccor_filled = None
    if ccor_dir is not None:
        comp = _composite_ccor(header, data.shape, ccor_dir, ccor_glob, ccor_max_dt)
        if comp is not None:
            ccor, ccor_valid = comp
            fill = ccor_valid & ~(np.isfinite(data) & (data > 0))
            data = np.where(fill, ccor * ccor_scale, data)
            ccor_filled = fill

    if eps_power:
        # Radial (elongation) scaling in linear B-sun units, on the unflipped
        # grid so the WCS-derived elongation map lines up with the data.
        data = data * _elongation_scale(data.shape, header, eps_power, eps0)

    # FITS row 0 is the bottom of the image (origin='lower'), but
    # PIL.Image.fromarray treats row 0 as the top -- flip so the rendered
    # JPEG isn't upside down.
    data = np.flipud(data)
    if ccor_filled is not None:
        ccor_filled = np.flipud(ccor_filled)

    norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax, clip=True)
    normed = norm(np.nan_to_num(data, nan=vmin))
    rgb = (cmap(normed)[..., :3] * 255).astype(np.uint8)

    if ccor_filled is not None:
        # Render the CCOR fill through a blue analog of the render colormap
        # (same luminance ramp, R/B swapped) so it reads as a distinct dataset.
        blue = (cmap(normed)[..., [2, 1, 0]] * 255).astype(np.uint8)
        rgb[ccor_filled] = blue[ccor_filled]

    if crop_inner is not None:
        # Radius given in source pixels; rescale to the (downsampled) render
        # grid the same way _draw_sun does, then black out the central disk --
        # except pixels filled from CCOR, which the crop is meant to reveal.
        h, w = rgb.shape[:2]
        radius = crop_inner * (h / header['NAXIS2'])      # [render px]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        yy, xx = np.ogrid[:h, :w]
        crop_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        if ccor_filled is not None:
            crop_mask = crop_mask & ~ccor_filled
        rgb[crop_mask] = 0

    if crop_outer is not None:
        # Complementary crop: same source-pixel radius rescaled to the render
        # grid, but black out everything OUTSIDE the disk (trim the outer FOV).
        h, w = rgb.shape[:2]
        radius = crop_outer * (h / header['NAXIS2'])      # [render px]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        yy, xx = np.ogrid[:h, :w]
        crop_mask = (yy - cy) ** 2 + (xx - cx) ** 2 > radius ** 2
        rgb[crop_mask] = 0

    if draw_sun:
        # Drawn last -- after the flip (centered, so the flip is a no-op for it)
        # and after any crop, so the reference disk stays visible even when the
        # inner region is blacked out.
        rgb = _draw_sun(rgb, header, sun_diam_deg, sun_color)

    if draw_timestamp:
        if type_label is None:
            # Auto data-product label from the header (e.g. "QAM"), with the
            # polarization plane appended for 3-D products.
            type_str = f"{header.get('TYPECODE', '')}{header.get('OBSCODE', '')}"
            if header.get('NAXIS') == 3:
                type_str = f"{type_str} {_plane_label(plane)}"
        else:
            type_str = type_label            # caller-supplied product label
        date_fmt = ('%Y-%b-%d %H:%M UT' if month_abbr else '%Y-%m-%d %H:%M UT')
        date_str = datetime.strptime(header['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f') \
                       .strftime(date_fmt)
    else:
        type_str = ""
        date_str = ""
    img = Image.fromarray(rgb)
    if constellations or planet_labels:
        # Both overlays share one celestial WCS scaled to the render grid.
        wcel = _celestial_render_wcs(header, img.height)
        if constellations:
            _draw_constellations(img, wcel, header, constellation_color,
                                 width=max(1, round(img.height / 1024)),
                                 max_elong=constellation_max_elong)
        if planet_labels:
            eff = font_size * fontscale
            pfont = ImageFont.truetype(_FONT_PATH, max(8, int(round(eff * 0.7))))
            _draw_planet_labels(img, wcel, header, planet_bodies, planet_color,
                                pfont, marker_r=max(3, int(round(eff * 0.35))))

    img = _annotate(img, type_str, date_str, slug,
                    font_size=font_size * fontscale, stamp_pos=stamp_pos,
                    logo=logo, logoheight=logoheight)
    if out_path is not None:
        img.save(out_path, quality=quality)
    return img                               # the finished PIL RGB image


def render_to_jpeg(fits_path: Path, out_path: Path, **kw):
    """Render a single PUNCH FITS mosaic to a JPEG, with no persistence."""
    plane = kw.pop('plane', PLANE)
    data, header = load_frame(fits_path,
                              out_size=kw.pop('out_size', OUT_SIZE),
                              min_valid=kw.pop('min_valid', MIN_VALID),
                              max_valid=kw.pop('max_valid', MAX_VALID),
                              plane=plane)
    render_frame(data, header, out_path, plane=plane, **kw)


def _combine_rank(stack, m: int, n: int):
    """Pixelwise m'th-smallest value across the leading axis of `stack`, with
    non-finite values treated as missing. m is indexed against a full window of
    `n` frames: m=0 is the smallest, m=n-1 the largest, m=(n-1)//2 the median.

    Pixels differ in how many valid samples k they actually have -- frames drop
    out individually, sub-threshold values are demoted to NaN at load, and the
    window is short at the ends of the sequence -- so a literal index m would
    run off the end of the valid data whenever k <= m. Instead m is read as the
    quantile m/(n-1) and applied to the k values that exist. That keeps the
    meaning of m fixed no matter how much data is missing: 0 stays the min, n-1
    stays the max, the middle stays the median.

    A pixel with no valid samples at all has no rank value and comes back NaN;
    the caller decides whether to zero those gaps or hold across them."""
    k = np.isfinite(stack).sum(axis=0)

    # Sort with missing values pushed to the high end, so the first k entries
    # along the axis are exactly the valid samples in ascending order.
    ordered = np.sort(np.where(np.isfinite(stack), stack, np.inf), axis=0)

    q = m / (n - 1) if n > 1 else 0.0
    idx = np.rint(q * np.maximum(k - 1, 0)).astype(np.intp)
    out = np.take_along_axis(ordered, idx[np.newaxis], axis=0)[0]
    return np.where(k > 0, out, np.nan)


def save_fits(data, header, out_path: Path):
    """Write one persistent frame to a FITS file in the same 1=image-HDU layout
    that load_frame reads back, preserving orientation (no flip) and the linear
    B-sun units. `data` is the downsampled OUT_SIZE array, so NAXIS and the WCS
    scale/reference pixel are rewritten to match the smaller grid; the rest of
    the original header (DATE-OBS, TYPECODE, ...) is carried through unchanged so
    the file stays self-describing and re-renders identically."""
    hdr = header.copy()
    h, w = data.shape
    factor = header['NAXIS2'] / h                 # 4096 -> OUT_SIZE, usually 2
    hdr['NAXIS1'], hdr['NAXIS2'] = w, h
    for k in ('CDELT1', 'CDELT2'):                # bigger pixels cover more sky
        if k in hdr:
            hdr[k] = hdr[k] * factor
    for k in ('CRPIX1', 'CRPIX2'):                # 1-indexed FITS convention
        if k in hdr:
            hdr[k] = (hdr[k] - 0.5) / factor + 0.5
    out_path.parent.mkdir(exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data.astype(np.float32), hdr)]
                 ).writeto(out_path, overwrite=True)


def render_sequence(files, out_dir: Path, mode: str = PERSIST_MODE,
                    epsilon: float = EPSILON, n: int = N_FRAMES,
                    m: int = M_RANK, kappa: float = KAPPA,
                    fits_dir: Path = None, name: str = '{stem}.jpg', **kw):
    """Render a time-ordered sequence of PUNCH FITS mosaics to JPEGs in
    `out_dir`, applying the selected persistence-of-vision filter across frames.

      'none'            -- each frame on its own.
      'exponential'     -- causal IIR blend, weight `epsilon` on the incoming
                           frame. Smoothly-decaying tails; one bright frame
                           (cosmic ray, star) bleeds into every frame after it.
      'rank'            -- pixelwise m'th-smallest value over `n` frames
                           CENTERED on the frame being rendered:
                           [i-n//2, i+n//2]. Symmetric, so features are not
                           shifted in time the way a trailing window shifts
                           them late. m selects the behaviour: m=0 (min)
                           strips anything that isn't present in every frame,
                           the middle (median) rejects transients shorter than
                           half the window, m=n-1 (max) keeps the brightest
                           value any frame saw and so accumulates trails.
                           Pixels with no valid data in the window render as 0.
      'persistent-rank' -- that same centered rank filter, followed by a
                           z-filter of weight `kappa` over the resulting stream:

                               s[i] = (1-kappa)*s[i-1] + kappa*r[i]

                           evaluated only where r[i] is finite; where it is NaN
                           (the pixel is missing throughout its whole window)
                           the state is held, so coverage gaps keep showing
                           their last good value instead of blinking to zero.
                           kappa=1 is pure gap-filling with no temporal
                           smoothing; smaller kappa also smooths and lengthens
                           the hold.

    Pixels that are non-finite, or below MIN_VALID, are treated as missing
    throughout. Near the ends of the sequence the window is clipped to the
    frames that exist, so it is shorter and one-sided there.

    Any render_frame keyword (vmin/vmax/gamma, slug, font_size/fontscale,
    eps_power, …) passes through **kw to every frame; e.g. font_size sets the
    annotation font in pixels. For 3-D polarized products (PTM/PAM), pass
    `plane=` the same way to choose which plane is loaded for every frame; see
    render_frame and load_frame.

    Output JPEGs are named with the `name` template, formatted per frame with
    `i` (0-based index) and `stem` (input filename stem); the default
    '{stem}.jpg' names each output after its input.

    If `fits_dir` is given, each persistent frame is also written there as a
    FITS file (linear B-sun, downsampled, unflipped) via save_fits, one per
    input file, so the filtered stream can be recombined photometrically
    later. Default None leaves the original JPEG-only behaviour untouched.
    """
    if mode not in ('none', 'exponential', 'rank', 'persistent-rank'):
        raise ValueError(f"unknown persistence mode {mode!r}")

    files = list(files)
    out_paths = [out_dir / name.format(i=i, stem=f.stem)
                 for i, f in enumerate(files)]
    load_kw = {'out_size': kw.pop('out_size', OUT_SIZE),
               'min_valid': kw.pop('min_valid', MIN_VALID),
               'max_valid': kw.pop('max_valid', MAX_VALID),
               'plane': kw.pop('plane', PLANE)}
    if fits_dir is not None:
        fits_dir.mkdir(exist_ok=True)

    if mode in ('none', 'exponential'):
        last = None
        for f, out_path in zip(tqdm(files), out_paths):
            data, header = load_frame(f, **load_kw)
            if mode == 'exponential':
                data = (np.where(np.isfinite(data), data, 0.0) if last is None else
                        np.where(np.isfinite(data),
                                 last * (1 - epsilon) + data * epsilon, last))
                last = data
            if fits_dir is not None:
                save_fits(data, header, fits_dir / (f.stem + '.fits'))
            render_frame(data, header, out_path, plane=load_kw['plane'], **kw)
        return

    # Centered rank filter. The window for frame i spans [i-h, i+h], so this is
    # non-causal -- rendering frame i requires frames that come after it. Walk
    # forward keeping a cache of the frames currently in the window; each file
    # is read exactly once and dropped as soon as it falls off the trailing
    # edge, so peak memory is n downsampled frames, not the whole sequence.
    if n % 2 == 0:
        n += 1
        print(f"N_FRAMES must be odd for a centered window; using n={n}")
    if not 0 <= m < n:
        raise ValueError(f"M_RANK must satisfy 0 <= m < n; got m={m}, n={n}")
    h = n // 2
    cache = {}      # index -> (data, header)
    state = None    # 'persistent-rank' z-filter state

    for i, out_path in enumerate(tqdm(out_paths)):
        lo, hi = max(0, i - h), min(len(files) - 1, i + h)
        for j in range(lo, hi + 1):
            if j not in cache:
                cache[j] = load_frame(files[j], **load_kw)
        for j in [j for j in cache if j < lo]:
            del cache[j]

        ranked = _combine_rank(
            np.stack([cache[j][0] for j in range(lo, hi + 1)]), m, n)

        if mode == 'rank':
            data = np.nan_to_num(ranked, nan=0.0)
        else:
            # Seed from the first frame; pixels already in a gap at the very
            # start have nothing to hold, so they start at zero.
            if state is None:
                state = np.nan_to_num(ranked, nan=0.0)
            else:
                state = np.where(np.isfinite(ranked),
                                 state * (1 - kappa) + ranked * kappa, state)
            data = state

        if fits_dir is not None:
            save_fits(data, cache[i][1], fits_dir / (files[i].stem + '.fits'))
        render_frame(data, cache[i][1], out_path, plane=load_kw['plane'], **kw)


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #

def RenderPUNCH(in_dir, out_dir, *,
                glob_pat: str = '*.fits',
                name: str = 'frame-{i:04d}.jpg',
                plane=PLANE,
                vmin: float = VMIN, vmax: float = VMAX, gamma: float = GAMMA,
                cmap=cmap_punch,
                min_valid: float = MIN_VALID, max_valid: float = MAX_VALID,
                out_size: int = OUT_SIZE, quality: int = JPEG_QUALITY,
                slug: str = SLUG, draw_timestamp: bool = True,
                type_label=None, month_abbr: bool = False,
                draw_sun: bool = DRAW_SUN, sun_diam_deg: float = SUN_DIAM_DEG,
                sun_color=SUN_COLOR,
                stamp_pos: str = STAMP_POS, fontscale: float = 1.0, logo=None,
                logoheight: float = 4, crop_inner: float = None,
                crop_outer: float = None,
                eps_power: float = EPS_POWER, eps0: float = EPS0,
                constellations: bool = False, planet_labels: bool = False,
                constellation_color=CONSTELLATION_COLOR,
                constellation_max_elong: float = CONSTELLATION_MAX_ELONG,
                planet_color=PLANET_COLOR, planet_bodies=PLANET_BODIES,
                ccor_dir=None, ccor_scale: float = CCOR_SCALE,
                ccor_glob: str = CCOR_GLOB, ccor_max_dt: float = None,
                min_image=None, peg_median_frame=None,
                persist_mode: str = PERSIST_MODE, epsilon: float = EPSILON,
                n_frames: int = N_FRAMES, m_rank: int = M_RANK,
                kappa: float = KAPPA, fits_dir=None):
    """Render every FITS mosaic in `in_dir` to annotated JPEGs in `out_dir`.

    This is the notebook's "render all frames" loop, generalized to any PUNCH
    product (CTM, PTM, CAM, PAM): with the defaults it reproduces that loop
    exactly (no persistence, frames written as ``frame-0000.jpg``,
    ``frame-0001.jpg``, …). Every knob the notebook exposed as a global is a
    keyword argument here.

    Parameters
    ----------
    in_dir, out_dir : path-like
        Input directory of FITS mosaics and output directory for JPEGs
        (created if absent).
    glob_pat : str
        Which files in `in_dir` to render, time-ordered by sorted name.
    name : str or None
        Output filename template, honored on every path (with or without
        persistence), formatted with `i` (frame index) and `stem` (input
        filename stem), e.g. ``'frame-{i:04d}.jpg'`` or ``'{stem}.jpg'``. Set to
        None to name each output after its input stem.
    plane : str or int
        For 3-D polarized products (PTM/PAM), which plane of the (B, pB, pB')
        leading axis to render: a name from {'B','pB','pBp'/"pB'"} or an integer
        index. The derived selector 'opB' renders the polarized-brightness
        magnitude sqrt(pB**2 + pB'**2). Defaults to 'B'. Ignored for 2-D products
        (CTM/CAM).
    vmin, vmax, gamma, cmap, min_valid, max_valid, out_size, quality, slug,
    draw_timestamp, draw_sun, sun_diam_deg, sun_color :
        Photometric stretch, downsample, annotation, and Sun-disk controls;
        see `load_frame` and `render_frame`.
    type_label : str or None
        Override the auto data-product label (header TYPECODE/OBSCODE, e.g. "QAM")
        with an arbitrary string, e.g. "QuickLook data". None (default) keeps the
        header-derived label.
    month_abbr : bool
        Write the timestamp month as a 3-letter abbreviation ("2026-Aug-04")
        instead of the numeric month ("2026-08-04"). Default False.
    stamp_pos : {'upper-right', 'lower-right'}
        Corner for the data-type/timestamp pair. 'lower-right' puts the data
        type above the timestamp in the bottom corner.
    fontscale : float
        Multiplier on the base annotation font (FONT_SIZE, ~37 px); 1.5 restores
        the historical 56 px.
    logo : path-like, tuple/list of path-like, or None
        Optional transparency-capable image (e.g. a PNG) overlaid in the
        lower-left corner, scaled to `logoheight` annotation line-heights and
        inset by the same margin as the corner text. A tuple/list of paths is
        laid out side-by-side sharing a common bottom edge.
    logoheight : float or tuple/list of float
        Logo height in annotation line-heights (default 4). A tuple/list sizes
        each logo in `logo` independently and must match it in length.
    crop_inner : float or None
        Radius in SOURCE pixels (native FITS resolution) around the frame center
        to black out; None (default) leaves the frame untouched.
    crop_outer : float or None
        Complementary crop: radius in SOURCE pixels around the frame center
        outside of which everything is blacked out (trims the outer FOV; combine
        with crop_inner for an annulus). None (default) leaves the frame untouched.
    eps_power, eps0 : float
        Radial (elongation) scaling: each pixel is multiplied by
        (eps/eps0)**eps_power, where eps is its elongation (angular distance from
        the Sun in the image plane, deg) and eps0 the datum elongation. Applied
        at render time in linear units, before the stretch, so any FITS dump
        stays unscaled. eps_power=0 (default) disables it; eps0 defaults to 10 deg.
    constellations, planet_labels : bool
        Sky overlays, both off by default and projected through the mosaic's
        celestial WCS. `constellations` draws the IAU constellation stick figures;
        `planet_labels` marks the bodies in `planet_bodies` (naked-eye planets +
        Moon + Uranus/Neptune) that fall within the frame with a text label.
    constellation_max_elong : float
        Clip constellation lines to within this elongation (deg from the Sun);
        default 47. Vertices beyond it are dropped and crossing segments cut at it.
    constellation_color, planet_color : RGB tuple
        Colors for the two overlays. `planet_bodies` overrides which solar-system
        bodies the planet-label overlay considers.
    ccor_dir : path-like or None
        Directory of CCOR (GOES-19) FITS frames to composite into the central
        hole. The nearest-in-time frame is reprojected onto the render grid; each
        pixel is PUNCH where PUNCH is valid (>0), else CCOR where it covers. None
        (default) disables it. `ccor_scale` multiplies the CCOR MSB values (=B-sun;
        1.0 physical) for display, `ccor_glob` selects the files, and `ccor_max_dt`
        (minutes) caps how far in time a match may be (None = no cap). See
        render_frame.
    min_image : path-like or None
        FITS path to a static per-pixel floor image subtracted from every frame
        (in linear units, before the stretch) to remove the F-corona / stray-light
        / pedestal; resampled to the render grid if needed. None (default)
        disables it. See render_frame.
    peg_median_frame : int or None
        Frame number (index into the sorted input files) whose median -- measured
        after F-corona removal (`min_image` subtraction) -- becomes the target
        level. Every frame is then scalar-shifted so its own median matches it,
        holding the overall brightness steady across the movie. None (default)
        disables it; negative indices count from the end.
    persist_mode, epsilon, n_frames, m_rank, kappa, fits_dir :
        Persistence-of-vision controls; see `render_sequence`. `persist_mode`
        defaults to 'none'. Any mode other than 'none' (or passing `fits_dir`)
        routes through `render_sequence`, which honors the same `name` template
        and optionally dumps per-frame FITS to `fits_dir`.

    Returns
    -------
    list[Path]
        The input files that were rendered, in render order.
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob(glob_pat))

    # Median pegging: measure the reference frame's median (after F-corona
    # removal) once, so every frame can be scalar-shifted to match it below.
    peg_median_to = None
    if peg_median_frame is not None and peg_median_frame is not False:
        if not -len(files) <= peg_median_frame < len(files):
            raise ValueError(f"peg_median_frame {peg_median_frame} out of range "
                             f"for {len(files)} frame(s)")
        rdata, _ = load_frame(files[peg_median_frame], out_size=out_size,
                              min_valid=min_valid, max_valid=max_valid, plane=plane)
        if min_image is not None:
            rdata = rdata - _min_image_for(min_image, rdata.shape)
        peg_median_to = float(np.nanmedian(rdata))
        if not np.isfinite(peg_median_to):
            print(f"peg_median_frame {peg_median_frame} has no valid median; "
                  f"pegging disabled")
            peg_median_to = None

    # Shared render/annotation knobs handed to render_frame (directly, or via
    # render_to_jpeg / render_sequence).
    render_kw = dict(vmin=vmin, vmax=vmax, gamma=gamma, cmap=cmap, quality=quality,
                     slug=slug, draw_timestamp=draw_timestamp,
                     type_label=type_label, month_abbr=month_abbr,
                     draw_sun=draw_sun, sun_diam_deg=sun_diam_deg,
                     sun_color=sun_color, stamp_pos=stamp_pos, fontscale=fontscale,
                     logo=logo, logoheight=logoheight, crop_inner=crop_inner,
                     crop_outer=crop_outer,
                     eps_power=eps_power, eps0=eps0,
                     constellations=constellations, planet_labels=planet_labels,
                     constellation_color=constellation_color,
                     constellation_max_elong=constellation_max_elong,
                     planet_color=planet_color, planet_bodies=planet_bodies,
                     ccor_dir=ccor_dir, ccor_scale=ccor_scale,
                     ccor_glob=ccor_glob, ccor_max_dt=ccor_max_dt,
                     min_image=min_image, peg_median_to=peg_median_to,
                     out_size=out_size, min_valid=min_valid,
                     max_valid=max_valid, plane=plane)

    if persist_mode == 'none' and name is not None and fits_dir is None:
        # Faithful to the notebook's first "render all frames" loop.
        for i, f in enumerate(tqdm(files)):
            render_to_jpeg(f, out_dir / name.format(i=i, stem=f.stem),
                           **render_kw)
    else:
        render_sequence(files, out_dir, mode=persist_mode, epsilon=epsilon,
                        n=n_frames, m=m_rank, kappa=kappa,
                        fits_dir=None if fits_dir is None else Path(fits_dir),
                        name=('{stem}.jpg' if name is None else name),
                        **render_kw)

    print(f"Done. {len(files)} JPEG(s) written to {out_dir}")
    return files


# --------------------------------------------------------------------------- #
# Inline display
# --------------------------------------------------------------------------- #

def _as_frame(cube, out_size, min_valid, max_valid, plane):
    """Resolve a "PUNCH cube" -- a punchbowl NDCube or a path to a FITS mosaic --
    to a rendered (data, header) via the same prep load_frame uses. out_size=None
    keeps the native resolution (no downsample)."""
    if isinstance(cube, (str, Path)):
        with fits.open(cube) as hdul:
            header = hdul[1].header
            data = hdul[1].data
    elif hasattr(cube, "meta") and hasattr(cube, "wcs"):   # punchbowl NDCube
        data = np.asarray(cube.data)
        header = cube.meta.to_fits_header(wcs=cube.wcs, write_celestial_wcs=True)
    else:
        raise TypeError("cube must be an NDCube or a path to a FITS mosaic; "
                        f"got {type(cube).__name__}")
    if out_size is None:
        out_size = np.asarray(data).shape[-1]              # native, no downsample
    return _process_frame(data, header, out_size, min_valid, max_valid, plane)


def show_frame(cube, ax=None, out_size=None, plane=PLANE,
               min_valid: float = MIN_VALID, max_valid: float = MAX_VALID,
               cmap=cmap_punch, figsize=(8, 8), **render_kw):
    """Render a single PUNCH frame and show it as a static matplotlib pane.

    `cube` is a punchbowl NDCube or a path to a FITS mosaic. The frame is run
    through the full render_frame pipeline (stretch, Sun disk, overlays,
    annotations, ... -- pass any render_frame keyword such as vmin/vmax, cmap,
    slug, constellations, ccor_dir), then drawn into `ax` (a new figure if None) as a
    finished RGB image. Returns the Axes. out_size=None renders at native
    resolution; pass e.g. out_size=1024 to downsample first.

    The sequence-only controls (persist_mode, peg_median_frame) do not apply --
    this is a single frame. In a notebook, run %matplotlib inline first, since
    importing this module forces the non-interactive Agg backend."""
    import matplotlib.pyplot as plt
    data, header = _as_frame(cube, out_size, min_valid, max_valid, plane)
    img = render_frame(data, header, None, plane=plane, cmap=cmap, **render_kw)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(np.asarray(img))
    ax.set_axis_off()
    return ax
