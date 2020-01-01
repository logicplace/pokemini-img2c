#/usr/bin/env python3
import os
import re
import sys
import glob
import argparse
import itertools
from typing import List, NamedTuple, Optional, Tuple
from collections import defaultdict

from PIL import Image, ImageColor

COMMENT = r"//.*|/\*[\s\S]*?\*/"

AT_PATTERN = re.compile(
	r"\[(\d*)\] +_at\(([^)]+)\)\s*"
	r"\{((?:%s|[\dxa-fA-F,\s]))\}" % (COMMENT,)
)

REMOVE_COMMENTS = re.compile(COMMENT)


class ProgramError(Exception): pass


class Colors(NamedTuple):
	class Color(NamedTuple):
		rgba: Tuple[int, int, int, int]
		hsv: Tuple[int, int, int]
		l: int

	colors: List[Color]
	transparent: Optional[Color]

	@property
	def white(self):
		return self.colors[0]

	@property
	def black(self):
		return self.colors[-1]

	def get(self, rgba: tuple):
		for i, c in enumerate(self.colors):
			if c.rgba == rgba:
				return i
		return "t"


def main():
	parser = argparse.ArgumentParser("img2h")
	parser.add_argument("--output", "-o",
		help="Output folder for .h files")
	parser.add_argument("--base", "-b",
		type=str2int, default=0x010000,
		help="Base address to use (Default: 0x010000)")
	parser.add_argument("--colors", "--colours", "-c",
		type=int, default=3,
		help="Number of colors to support (Default: 3)")
	parser.add_argument("--sprites", "-s", nargs="+", default=[],
		help="Sprites to convert")
	parser.add_argument("--tiles", "-t", nargs="+", default=[],
		help="Tiles to convert")
	args = parser.parse_args()

	if args.colors not in [2, 3]:
		print("Only supports 2 or 3 colors atm", file=sys.stderr)
		sys.exit(1)

	base = args.base

	# TODO: use read_h to find open blocks and etc

	for n, l in [("tiles", args.tiles), ("sprites", args.sprites)]:
		convert = {
			"tiles": convert_tiles,
			"sprites": convert_sprites,
		}[n]

		for x in l:
			# TODO: group tiles into blocks of 256, add base defines
			# TODO: alignment
			fname = os.path.splitext(x)[0]
			name = os.path.basename(fname)
			var_name = name if name.endswith(n) else f"{name}_{n}"
			im = Image.open(x)
			colors = identify_colors(im, args.colors, n == "sprites")
			px_bands = convert(im, colors)
			with open(f"{fname}.h", "w") as f:
				f.write(
					f"#ifndef __{name.upper()}_H__\n"
					f"#define __{name.upper()}_H__\n\n"
					'#include "pm.h"\n'
					'#include <stdint.h>\n\n'
				)

				for i, band in enumerate(px_bands):
					len_band = len(band)
					f.write(f"const uint8_t _rom {var_name}{i+1}[{len_band}] _at(0x{base:06x}) = {{\n")
					for i in range(0, len_band, 8):
						line = ", ".join(f"0x{px:02x}" for px in band[i:i+8])
						f.write(f"\t{line},\n")
					f.write("};\n\n")
					base += len_band
				
				f.write(f"#endif // __{name.upper()}_H__\n")
			print(f"wrote {fname}.h")


def str2int(s: str, c=False):
	if s.startswith("0x"):
		return int(s[2:], 16)
	elif s.startswith("$"):
		return int(s[1:], 16)
	elif s.startswith("0b"):
		return int(s[2:], 2)
	elif c and s.startswith("0"):
		return int(s, 8)
	return int(s)


def read_h(path: str):
	times = {}
	filled = []
	for hfn in glob.glob(os.path.join(path, "*.h")):
		name = os.path.basename(hfn)[:-2]
		times[name] = os.path.getmtime(hfn)
		with open(hfn) as f:
			ats = AT_PATTERN.findall(f.read())
		for count, address, contents in ats:
			if not count:
				values = REMOVE_COMMENTS.sub("", contents).split(",")
				count = len(values)
				if not values[-1].strip():
					count -= 1
			filled.append((address, count))
	filled.sort()
	prev = filled[0]
	ret = []
	for x in filled[1:]:
		if sum(prev) >= x[0]:
			# TODO: warn on >
			prev = (prev[0], prev[1] + x[1])
		else:
			ret.append(prev)
			prev = x
	return ret


def rgb(hex):
	return (hex >> 16, (hex >> 8) & 0xff, hex & 0xff)


def identify_colors(img: Image, max_colors: int, transparency: bool) -> Colors:
	if max_colors < 2 or max_colors > 6:
		raise ProgramError("max colors should be between 2 and 6 (inclusive)")

	expected = max_colors + int(transparency)
	hsv_im: Image = img.convert("HSV")
	colors = hsv_im.getcolors()
	if len(colors) > expected:
		print(colors)
		raise ProgramError(f"image has too many colors: expected {expected}, got {len(colors)}")

	l_im: Image = img.convert("L")
	rgba_im: Image = img.convert("RGBA")
	rem_colors = {c[1] for c in colors}
	cdata = []
	width = img.size[0]
	for i, p in enumerate(hsv_im.getdata()):
		if p in rem_colors:
			rem_colors.remove(p)
			xy = (i % width, i // width)
			cdata.append(Colors.Color(rgba_im.getpixel(xy), p, l_im.getpixel(xy)))
		if not rem_colors:
			break

	transparent = None

	if transparency:
		# One should either have alpha or the most distinct hue (with saturation)
		have_sat = []
		for c in cdata:
			if c.rgba[3] == 0:
				transparent = c
				break
			elif c.hsv[1]:
				have_sat.append(c)
		
		if transparent is None:
			if len(have_sat) == 1:
				# Everything was grayscale
				transparent = have_sat[0]
			elif len(have_sat) > 2:
				hues = {c.hsv[0]: c for c in have_sat}
				min_for: Dict[int, int] = defaultdict(lambda: 256)
				for a, b in itertools.combinations(list(hues.keys()), 2):
					d = abs(a - b)
					d = min(256 - d, d)
					if d < min_for[a]:
						min_for[a] = d
					if d < min_for[b]:
						min_for[b] = d

				most_distinct_hue = max(min_for.keys(), key=min_for.get)
				if most_distinct_hue > 5:
					transparent = hues[most_distinct_hue]
				# else assume no transparency
	
	# Sort remaining colors by lightness
	if transparent is not None:
		cdata.remove(transparent)
	cdata.sort(key=lambda x: x.l)

	return Colors(cdata, transparent)


	# scheme = {
	# 	2: (0xa9bda9, 0x203220),
	# 	3: (0xa9bda9, 0x556855, 0x203220),
	# 	4: (0xa9bda9, 0x708370, 0x3d4f3d, 0x203220),
	# 	5: (0xa9bda9, 0x7b8f7b, 0x556855, 0x304230, 0x203220),
	# 	#6: (0xa9bda9, 0x203220), # TODO
	# }


def convert_tiles(img: Image, colors: Colors):
	pixels = []
	width, height = img.size

	img = img.convert("RGBA")
	ncolors = len(colors.colors)
	
	for row in range(0, height, 8):
		for col in range(0, width, 8):
			for x in range(col, col + 8):
				pxs = [0] * (ncolors - 1)
				for y in range(row + 7, row - 1, -1):
					px = img.getpixel((x, y))
					idx = colors.get(px)
					if idx == "t":
						print(px, colors)
						raise ProgramError("tiles can't be transparent")
					grays = get_grays[ncolors](idx, x, y)
					for i, gray in enumerate(grays):
						pxs[i] |= gray << (y - row)
				pixels.append(pxs)
	
	# Return as bands
	return tuple(zip(*pixels))


def convert_sprites(img: Image, colors: Colors):
	# Format: UL mask, LL mask, UL, LL, UR mask, LR mask, UR, LR
	pixels = []
	width, height = img.size

	img = img.convert("RGBA")
	ncolors = len(colors.colors)
	
	for row in range(0, height, 16):
		for col in range(0, width, 16):
			for x_segment in range(col, col + 16, 8):
				segment = []
				for y_segment in range(row, row + 16, 8):
					for x in range(x_segment, x_segment + 8):
						mask = 0
						pxs = [0] * (ncolors - 1)
						for y in range(y_segment + 7, y_segment - 1, -1):
							px = img.getpixel((x, y))
							idx = colors.get(px)
							shift = y - y_segment
							if idx == "t":
								mask |= 1 << shift
							
							grays = get_grays[ncolors](idx, x, y)
							for i, gray in enumerate(grays):
								pxs[i] |= gray << shift
						pixels.append([mask] * ncolors)
						segment.append(pxs)
				pixels.extend(segment)

	
	# Return as bands
	return tuple(zip(*pixels))


def get_grays2(b: int, x: int, y: int) -> tuple:
	return (b,)

def get_grays3(b: int, x: int, y: int) -> tuple:
	if b == 1:
		return (0, 1) if (x + y) % 2 else (1, 0)
	b = 0 if b else 1
	return (b, b)

get_grays = {
	2: get_grays2,
	3: get_grays3,
}


if __name__ == "__main__":
	main()
