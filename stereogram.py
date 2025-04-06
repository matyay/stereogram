#!/usr/bin/env python3
import argparse
import random
import colorsys

import numpy as np
from PIL import Image

# =============================================================================

# http://www.techmind.org/stereo/stech.html

def make_links(depth, eye_sep, view_dist, minz, maxz, dpi, oversample):

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

    dmin  = np.min(depth.flatten()) // oversample
    dmax  = np.max(depth.flatten()) // oversample
    print(f" disparity : [{dmin}, {dmax}]")

    # Active range
    x0 = oversample * (maxsep // 2 + 1)
    x1 = oversample * (dx - maxsep // 2)
    xm = oversample * dx

    # Make links
    for y in range(dy):
        for x in range(x0, x1):

            s = depth[y, x // oversample]
            l = x - s // 2
            r = x + s
 
            if l >= 0 and r < xm:
                link_l[y, r] = l
                link_r[y, l] = r

    return link_l, link_r


# =============================================================================

def render(links, contrast=1):

    image = np.empty_like(links, dtype=np.uint8)
    dy, dx = image.shape

    # Render
    for y in range(dy):
        for x in range(dx):

            l = links[y, x]
            if l == x:
                image[y, x] = random.randint(0, 255)
            else:
                image[y, x] = image[y, l]

    # Increase contrast
    if contrast != 1:
        image = image.astype(np.int32)
        image = image - 128
        image = image * contrast
        image = np.clip(image, -128, +127)
        image = image + 128
        image = image.astype(np.uint8)

    return image


def render2(link_l, link_r, contrast=1, colorize_links=False):

    base  = np.zeros_like(link_l, dtype=np.uint8)
    index = np.zeros_like(base, dtype=np.int32)

    dy, dx = base.shape
    cx = dx // 2

    # Render
    for y in range(dy):

        for x in range(cx, dx):
            l = link_l[y, x]
            if l == x or base[y, l] == 0:
                base[y, x] = random.randint(1, 255)
            else:
                base[y, x] = base[y, l]
                index[y, x] = index[y, l] + 1

        for x in reversed(range(0, cx)):
            r = link_r[y, x]
            if r == x or base[y, r] == 0:
                base[y, x] = random.randint(1, 255)
            else:
                base[y, x] = base[y, r]
                index[y, x] = index[y, r] + 1

    # Increase contrast
    if contrast != 1:
        base = base.astype(np.int32)
        base = base - 128
        base = base * contrast
        base = np.clip(base, -128, +127)
        base = base + 128
        base = base.astype(np.uint8)

    if not colorize_links:
        return base

    # Color by pixel repetitions
    image = np.empty((dy, dx, 3), dtype=np.uint8)
    n = np.max(index.flatten())

    for y in range(dy):
        for x in range(dx):
            h = index[y, x]    / n
            l = base[y, x] / 255.0
            s = 0.5

            c = colorsys.hls_to_rgb(h, l, s)
            image[y, x, :] = np.array(c) * 255.0

    return image


def render_links(links, contrast=1):

    import colorsys

    dy, dx = links.shape

    image = np.empty((dy, dx, 3), dtype=np.uint8)
    index = np.zeros_like(links, dtype=np.int32)

    # Render, count repetitions
    for y in range(dy):
        for x in range(dx):

            l = links[y, x]
            if l == x:
                image[y, x, 0] = random.randint(0, 255)
                index[y, x]    = 0
            else:
                image[y, x] = image[y, l]
                index[y, x] = index[y, l] + 1

    # Increase contrast
    if contrast != 1:
        image = image.astype(np.int32)
        image = image - 128
        image = image * contrast
        image = np.clip(image, -128, +127)
        image = image + 128
        image = image.astype(np.uint8)

    # Color by pixel repetitions
    n = np.max(index.flatten())
    for y in range(dy):
        for x in range(dx):
            h = index[y, x]    / n
            l = image[y, x, 0] / 255.0
            s = 0.5

            c = colorsys.hls_to_rgb(h, l, s)
            image[y, x, :] = np.array(c) * 255.0

    return image

# =============================================================================

def fit_image(image, width, height):

    src_aspect = image.width / image.height
    dst_aspect = width / height

    if src_aspect > dst_aspect:

        # Scale
        h = int(width / src_aspect + 0.5)
        m = Image.Resampling.BILINEAR if h > height else Image.Resampling.BOX
        image = image.resize((width, h), m)

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
        m = Image.Resampling.BILINEAR if w > width else Image.Resampling.BOX
        image = image.resize((w, height), m)

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
        if scale > 1.0:
            mode = Image.Resampling.BILINEAR
        else:
            mode = Image.Resampling.NEAREST

        print("Scaling...")
        img = img.resize((
            int(img.width  * scale + 0.5),
            int(img.height * scale + 0.5)),
            mode
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
    link_l, link_r = make_links(depth, **params)
    # Render
    print("Rendering...")
#    if args.show_links:
#        image = render_links(link_l, args.oversample)
#    else:
#        image = render(link_l, args.oversample)
    image = render2(link_l, link_r, args.oversample, colorize_links=args.show_links)

    # Downscale & save
    print("Scaling...")
    img = Image.fromarray(image)
    img = img.resize(
        ((args.pscale * img.width) // args.oversample, args.pscale * img.height),
        Image.Resampling.BOX
    )
    print(f" {img.width}x{img.height}")

    print("Saving...")
    fname = "stereogram.png" if args.out is None else args.out
    img.save(fname)

if __name__ == "__main__":
    main()
