# Stereogram generator

A Python script for generating stereograms.

## Installation

1. Clone the repo
```
git clone
```

2. Create Python virtual env and install dependencies
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

1. Activate the Python virtual environment (if not already active)
```
source env/bin/activate
```

2. Run the main script, provide depth map and appropriate parameters. Usage:
```
usage: stereogram.py [-h] [--pattern PATTERN] [--oversample OVERSAMPLE] [-s SCALE] [--fit FIT FIT] [-o OUT] [--eye-sep EYE_SEP] [--view-dist VIEW_DIST]
                     [--z-range Z_RANGE Z_RANGE] [--dpi DPI] [-p PSCALE] [-d {left,right,both}] [--seed SEED] [--show-links] [--disparity] [-b BRIGHTNESS]
                     [-g GAMMA] [--focus-aids]
                     depth

positional arguments:
  depth                 Input depth map

options:
  -h, --help            show this help message and exit
  --pattern PATTERN     Tileable pattern image
  --oversample OVERSAMPLE
                        Oversampling factor
  -s SCALE, --scale SCALE
                        Input depth map image scaling factor
  --fit FIT FIT         Fit depth map image resolution to the given one
  -o OUT, --out OUT     Output file name
  --eye-sep EYE_SEP     Eye separation [cm]
  --view-dist VIEW_DIST
                        Viewing distance [cm]
  --z-range Z_RANGE Z_RANGE
                        Depth range [cm]
  --dpi DPI             Viewing device DPI
  -p PSCALE, --pscale PSCALE
                        Display point scale
  -d {left,right,both}, --direction {left,right,both}
                        Rendering direction
  --seed SEED           RNG Seed
  --show-links          Colorize pixel repetitions
  --disparity           The input is disparity instead of distance
  -b BRIGHTNESS, --brightness BRIGHTNESS
                        Brightness adjust (from -1.0 to +1.0)
  -g GAMMA, --gamma GAMMA
                        Gamma
  --focus-aids          Draw focus aids
```

## Examples

1. Generate a stereogram from the `depth.png` depth map with default settings:
```
./stereogram.py depth.png
```
If option `-o` is ommitted the output image will be saved as `stereogram.png`

