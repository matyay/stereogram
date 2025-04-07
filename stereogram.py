#!/usr/bin/env python3
import argparse
import colorsys

from numba import jit
import numpy as np
from PIL import Image

# =============================================================================

# http://www.techmind.org/stereo/stech.html

@jit
def _make_links(depth, link_l, link_r, maxsep, oversample=1):
    """
    The core of pixel link builder
    """

    dy, dx = depth.shape

    # Active range
    x0 = oversample * (maxsep // 2 + 1)
    x1 = oversample * (dx - maxsep // 2)
    xm = oversample * dx

    # Make links
    for y in range(dy):
        for x in range(x0, x1):

            s = depth[y, x // oversample]
            l = x - s // 2
            r = l + s
 
            if l >= 0 and r < xm:
                link_l[y, r] = l
                link_r[y, l] = r


def make_links(depth, eye_sep, view_dist, minz, maxz, dpi, oversample):
    """
    Make pixel links
    """

    dy, dx = depth.shape
    link_l = np.tile(np.arange(0, dx * oversample, dtype=np.int16), (dy, 1))
    link_r = np.tile(np.arange(0, dx * oversample, dtype=np.int16), (dy, 1))

    # De-normalize depth
    depth = maxz + depth.astype(np.float32) * (minz - maxz) / 255.0

    # Beyond the screen
    if minz > 0.0 and maxz > 0.0:
        maxsep = maxz / (maxz + view_dist) * eye_sep
        k = 1.0

    # In front of the screen
    elif minz < 0.0 and maxz < 0.0:
        maxsep = -minz / (minz + view_dist) * eye_sep
        k = -1.0

    else:
        print(f"Invalid Z range [{minz}, {maxz}]")
        exit(-1)

    maxsep = int(maxsep * dpi / 2.54 + 0.5)

    depth = oversample * dpi / 2.54 * k * depth / (depth + view_dist) * eye_sep + 0.5
    depth = np.clip(depth, 0, None).astype(np.int32)

    dmin  = int(np.floor(np.min(depth.flatten()) / oversample))
    dmax  = int(np.ceil( np.max(depth.flatten()) / oversample))
    print(f" disparity : [{dmin}, {dmax}]")

    # Build the links
    _make_links(depth, link_l, link_r, maxsep, oversample)

    return (link_l, link_r), maxsep

# =============================================================================

@jit
def _render(link_l, link_r, pattern, oversample, direction="both"):

    base  = np.zeros((*link_l.shape, 3), dtype=np.uint8)
    index = np.zeros_like(link_l, dtype=np.int32)

    dy, dx, _ = base.shape
    cx = dx // 2

    py, px, _ = pattern.shape

    # Render
    for y in range(dy):

        # Left to right
        if direction == "right":

            for x in range(dx):
                l = link_l[y, x]
                if l == x or base[y, l].all() == 0:
                    base[y, x] = pattern[y, x % px]
                else:
                    base[y, x] = base[y, l]
                    index[y, x] = index[y, l] + 1

        # Right to left
        elif direction == "left":

            for x in range(dx - 1, -1, -1):
                r = link_r[y, x]
                if r == x or base[y, r].all() == 0:
                    base[y, x] = pattern[y, (dx - x) % px]
                else:
                    base[y, x] = base[y, r]
                    index[y, x] = index[y, r] + 1

        # Both ways 
        elif direction == "both":

            for xl in range(cx, dx):
                xr = dx - 1 - xl

                l = link_l[y, xl]
                if l == x or base[y, l].all() == 0:
                    base[y, xl] = pattern[y, xl % px]
                else:
                    base[y, xl] = base[y, l]
                    index[y, xl] = index[y, l] + 1

                r = link_r[y, xr]
                if r == x or base[y, r].all() == 0:
                    base[y, xr] = pattern[y, xr % px]
                else:
                    base[y, xr] = base[y, r]
                    index[y, xr] = index[y, r] + 1

    return base, index

#@jit # Unsupported colorsys
def _colorize(base, index):

    # Color by pixel repetitions
    base = np.sum(base, axis=2) // 3
    dy, dx = base.shape

    image = np.empty((dy, dx, 3), dtype=np.uint8)
    n = np.max(index.flatten())

    for y in range(dy):
        for x in range(dx):
            h = index[y, x] / (n + 1)
            l = base[y, x]  / 255.0
            s = 0.5

            c = colorsys.hls_to_rgb(h, l, s)
            image[y, x, :] = np.array(c) * 255.0

    return image


def render(link_l, link_r, pattern=None, oversample=1, direction="both", colorize_links=False):

    # Pattern not provided, randomize it to effectively get an RDS
    if pattern is None:

        dy, dx = link_l.shape

        pattern = np.random.rand(dy, (oversample * dx) // 2) # Assume maxsep < width
        pattern = (pattern * 255.0).astype(np.uint8)
        pattern = (pattern // oversample) * oversample # Quantize
        pattern = np.tile(pattern[:, :, None], (1, 1, 3)) # Make RGB

    # Ensure no zeros in the pattern
    pattern = np.clip(pattern, 1, None)

    # Render
    base, index = _render(link_l, link_r, pattern, oversample, direction)

    # Colorize
    if colorize_links:
        return _colorize(base, index)

    return base

# =============================================================================

def fit_image(image, width, height):
    """
    Crops & resized the input image to fit in width x height while preserving
    its aspect.
    """

    src_aspect = image.width / image.height
    dst_aspect = width / height

    if src_aspect > dst_aspect:

        # Scale
        h = int(width / src_aspect + 0.5)
        image = image.resize((width, h), Image.Resampling.LANCZOS)

        # Pad
        pad = height - image.height
        if pad > 0:
            p1 = pad // 2
            p2 = pad - p1

            image = np.array(image)
            image = np.pad(image,
                [(p1, p2), (0, 0)] + [(0, 0)] * (image.ndim - 2), mode="edge")
            image = Image.fromarray(image)

    else:

        # Scale
        w = int(height * src_aspect + 0.5)
        image = image.resize((w, height), Image.Resampling.LANCZOS)

        # Pad
        pad = width - image.width
        if pad > 0:
            p1 = pad // 2
            p2 = pad - p1

            image = np.array(image)
            image = np.pad(image,
                [(0, 0), (p1, p2)] + [(0, 0)] * (image.ndim - 2), mode="edge")
            image = Image.fromarray(image)

    return image

# =============================================================================

def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "depth",
        type=str,
        help="Input depth map",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Tileable pattern image",
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=4,
        help="Oversampling factor",
    )
    parser.add_argument(
        "-s","--scale",
        dest="scale",
        type=float,
        default=None,
        help="Input depth map image scaling factor"
    )
    parser.add_argument(
        "--fit",
        type=int,
        nargs=2,
        default=None,
        help="Fit depth map image resolution to the given one",
    )
    parser.add_argument(
        "-o","--out",
        dest="out",
        type=str,
        default=None,
        help="Output file name"
    )
    parser.add_argument(
        "--eye-sep",
        type=float,
        default=6.0,
        help="Eye separation [cm]",
    )
    parser.add_argument(
        "--view-dist",
        type=float,
        default=50.0,
        help="Viewing distance [cm]",
    )
    parser.add_argument(
        "--z-range",
        type=float,
        nargs=2,
        default=[40.0, 60.0],
        help="Depth range [cm]",
    )
    parser.add_argument(
        "--dpi",
        type=float,
        default=96.0,
        help="Viewing device DPI",
    )
    parser.add_argument(
        "-p", "--pscale",
        type=int,
        default=1,
        help="Display point scale",
    )
    parser.add_argument(
        "-d", "--direction",
        type=str,
        choices=["left", "right", "both"],
        default="both",
        help="Rendering direction"
    )
    parser.add_argument(
        "--show-links",
        action="store_true",
        help="Colorize pixel repetitions",
    )

    args = parser.parse_args()

    if args.scale is not None and args.fit is not None:
        print("Use either --scale or --fit")
        exit(1)

    # Load depth map
    print(f"Loading depth map '{args.depth}'...")
    img = Image.open(args.depth)
    print(f" {img.width}x{img.height}")

    # Fit
    if args.fit is not None:
        print("Fitting...")
        scale = (args.fit[0] // args.pscale, args.fit[1] // args.pscale)
        img = fit_image(img, *scale)
        print(f" {img.width}x{img.height}")

    # Scale
    if args.scale is not None:
        scale = args.scale / args.pscale

        print("Scaling...")
        img = img.resize((
            int(img.width  * scale + 0.5),
            int(img.height * scale + 0.5)),
            Image.Resampling.LANCZOS
        )
        print(f" {img.width}x{img.height}")

    # Convert
    depth = np.array(img)
    depth = np.clip(0, 255, depth)
    depth = depth.astype(np.uint8)

    # Make grayscale if not already
    assert depth.ndim in [2, 3], depth.ndim
    if depth.ndim == 3:
        depth = np.average(depth, axis=2)

    # Load pattern
    if args.pattern:
        print(f"Loading pattern '{args.pattern}'")
        pattern = Image.open(args.pattern)
        print(f" {pattern.width}x{pattern.height}")

    else:
        pattern = None
        

    print("Processing...")

    params = {
        "eye_sep":    args.eye_sep,
        "view_dist":  args.view_dist,
        "minz":       args.z_range[0],
        "maxz":       args.z_range[1],
        "dpi":        args.dpi / args.pscale,
        "oversample": args.oversample,
    }

    for k, v in params.items():
        print(f" {k:<10s}: {v}")

    # Make links
    print("Making links...")
    (link_l, link_r), maxsep = make_links(depth, **params)

    # Scale pattern
    if pattern:

        print("Scaling pattern...")

        # Fit width
        dx = (maxsep * args.oversample)
        dy = int(maxsep * pattern.height / pattern.width)

        pat = pattern.resize((dx, dy), Image.Resampling.LANCZOS)

        # Tile vertically
        pattern = Image.new("RGB", (dx, depth.shape[0]))
        for i in range(depth.shape[0] // dy + 1):
            pattern.paste(pat, (0, i * dy))

        print(f" {pattern.width // args.oversample}x{pattern.height}")

        # Convert
        pattern = np.array(pattern)

    # Render
    print("Rendering...")
    image = render(link_l, link_r,
        pattern=pattern,
        direction=args.direction,
        oversample=args.oversample,
        colorize_links=args.show_links
    )

    # Downscale & save
    print("Scaling...")
    img = Image.fromarray(image)

    if args.oversample > 1:
        img = img.resize(
            (img.width // args.oversample, img.height),
            Image.Resampling.LANCZOS
        )

    if args.pscale != 1:
        img = img.resize(
            (args.pscale * img.width, args.pscale * img.height),
            Image.Resampling.NEAREST
        )

    print(f" {img.width}x{img.height}")

    print("Saving...")
    fname = "stereogram.png" if args.out is None else args.out
    img.save(fname)

if __name__ == "__main__":
    main()
