#/usr/bin/env python3
import re
import sys
import string
import pathlib
import argparse
import datetime
import itertools
from io import BytesIO
from typing import Dict, List, NamedTuple, Optional, Tuple
from collections import defaultdict

from PIL import Image

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

	def __len__(self) -> int:
		return len(self.colors) + int(bool(self.transparent))


def main():
	parser = argparse.ArgumentParser("img2h")
	parser.add_argument("--output", "-o", default="",
		help="Output folder (Default is same as file)")
	parser.add_argument("--output-headers", "-O", default="",
		help="Output folder for header stubs (Default is same as --output)")
	parser.add_argument("--base", default=".",
		help="Base directory for your project (Default is cwd)")
	parser.add_argument("--no-stdint", action="store_true",
		help="Don't use stdint when generating header stubs")
	outfmt = parser.add_mutually_exclusive_group()
	outfmt.add_argument("--obj", action="store_true",
		help="Output IEEE695 object format (default)")
	outfmt.add_argument("--asm", "-a", action="store_true",
		help="Output assembler file")
	parser.add_argument("--memory-model", "-M", default="d",
		help="Memory model to use for output")
	# parser.add_argument("--data", action="store_true",
	# 	help="Output as DATA section type instead of CODE (for instance, when using -Mc)")
	parser.add_argument("--colors", "--colours", "-c",
		type=int, default=3,
		help="Number of colors to support (Default: 3)")
	parser.add_argument("--invert", "-N", action="store_true",
		help="Invert the colors")
	parser.add_argument("--sprites", "-s", nargs="+", default=[],
		help="Sprites to convert")
	parser.add_argument("--tiles", "-t", nargs="+", default=[],
		help="Tiles to convert")
	parser.add_argument("--debug", "-D", action="store_true",
		help="Output a randomized color palette version of the image(s) in order to highlight issues")
	parser.add_argument("imgs", nargs="*", default=[],
		help="Images to convert. Filenames must end in 'sprites' or 'tiles' before the extension")
	args = parser.parse_args()

	if args.colors not in [2, 3]:
		print("Only supports 2 or 3 colors atm", file=sys.stderr)
		sys.exit(1)

	mem_model: str = args.memory_model.lower()
	if mem_model not in tuple("scdl"):
		print("Memory model must be S, C, D, or L", file=sys.stderr)
		sys.exit(1)

	output = pathlib.Path(args.output)
	output_h = pathlib.Path(args.output_headers) if args.output_headers else output

	tiles = args.tiles + [t for t in args.imgs if pathlib.Path(t).stem.endswith("tiles")]
	sprites = args.sprites + [s for s in args.imgs if pathlib.Path(s).stem.endswith("sprites")]

	for n, l in [("tiles", tiles), ("sprites", sprites)]:
		convert = {
			"tiles": convert_tiles,
			"sprites": convert_sprites,
		}[n]

		for x in l:
			im_file = pathlib.Path(x)
			name = im_file.stem
			out = output / name if args.output else im_file.with_suffix("")
			out_h = output_h / name if args.output_headers else out

			im = Image.open(x)
			colors = identify_colors(im, args.colors, n == "sprites", args.invert)
			if args.debug:
				out_im = random_redraw(im, colors)
				out = out.with_name(out.stem + "_randomized").with_suffix(".png")
				with out.open("wb") as f:
					out_im.save(f)
			else:
				px_bands = convert(im, colors)

				outargs = (n, out, args, px_bands, mem_model)
				if args.asm:
					write_asm(*outargs)
				else:
					write_ieee695(*outargs)
				write_h_stub(n, out_h, args, px_bands)


def chunk(b: bytes, size: int):
	reader = BytesIO(b)
	for _ in range(0, len(b), size):
		yield reader.read(size)


def get_var_name(mode, fn: pathlib.Path):
	name = fn.stem
	return name if name.endswith(mode) else f"{name}_{mode}"


def write_h_stub(mode, fn: pathlib.Path, args, px_bands):
	out = fn.with_suffix(".h")
	name = fn.stem.upper()
	var_name = get_var_name(mode, fn)
	with out.open("wt") as f:
		f.write(
			f"#ifndef {name}_H\n"
			f"#define {name}_H\n\n"
			+ (
				"extern const _far unsigned char "
				if args.no_stdint else
				"#include <stdint.h>\n\nextern const _far uint8_t "
			)
			+ ", ".join([
				f"{var_name}{i}[]"
				for i in range(1, len(px_bands) + 1)
			])
			+ ";\n\n"
			f"#endif // {name}_H\n"
		)
	print(f"wrote {out}")


def write_ieee695(mode, fn: pathlib.Path, args, px_bands, memory_model="l"):
	out = fn.with_suffix(".obj")
	name = fn.stem
	var_name = get_var_name(mode, fn)
	with out.open("wb") as f:
		# Module Begin, built for E0C88d, filename is {obj_fn}
		memory_model = memory_model.lower()
		assert memory_model in tuple("scdl")
		f.write(b"\xe0\x06E0C88" + memory_model.encode("ascii"))
		obj_fn = str(out.relative_to(args.base).as_posix())
		f.write(ieee695_str(obj_fn, "File path"))

		# Address Description, 8 bit MAU, 3 bytes max AU, big endian
		f.write(b"\xec\x08\x03\xcc")

		# Make parts in order to address them...
		now = datetime.datetime.now(datetime.timezone.UTC)
		mode_align = (64 if mode == "sprites" else 8).to_bytes(1, "big")
		band_data = [
			(
				(i + 1),
				(i + 1).to_bytes(1, "big"),
				(i + 0x20).to_bytes(1, "big"),
				len(band),
				ieee695_int(len(band)),
				bytes(band),
			)
			for i, band in enumerate(px_bands)
		]

		w = [
			# ASW0 - AD Extension Record
			(
				ieee695_co(0x25, "Object version: 1.1")
				+ ieee695_co(0x26, "Object format: Relocatable")
			),
			# ASW1 - Environmental record
			(
				# Date record
				b"\xeb"
				+ ieee695_int(now.year)
				+ ieee695_int(now.month)
				+ ieee695_int(now.day)
				+ ieee695_int(now.hour)
				+ ieee695_int(now.minute)
				+ ieee695_int(now.second)
				# TODO: Can we be honest in these?
				+ ieee695_co(0x35, "WINDOWS")
				+ ieee695_co(0x36, "E0C88 assembler (061)")
			),
			# ASW2 - Section part
			b"".join([
				(
					# Section Type, index, modes R Y4 C, name
					# Read-only, addressing mode 2, cumulative
					b"\xe6" + ib1 + b"\xD2\xD9\x02\xC3"
					+ ieee695_str(f".pm{mode}{i:05d}")
					# Section alignment, index, align by mode
					+ b"\xe7" + ib1 + mode_align
					# Section size, index, size in MAUs
					+ b"\xe2\xd3" + ib1
					+ len_band_int
				)
				for i, ib1, _, _, len_band_int, _ in band_data
			]),
			# ASW3 - External part
			b"".join([
				(
					# Internal(?) Name, index, name
					b"\xe8" + ib20
					+ ieee695_str(f"_{var_name}{i}")
					# Attribute, index, unspecified type, count (why did I write count?)
					+ b"\xf1\xc9" + ib20 + b"\x00"
					# lex_level ?
					+ b"\x81\x84"  # probably a 1 byte int of 84, also have seen 81 and 82, not sure why
					# Assign value record, index, R{i}
					+ b"\xe2\xc9" + ib20
					+ b"\xd2" + ib1
				)
				for i, ib1, ib20, _, _, _ in band_data
			]),
			# ASW4 - Debug information definition part
			(
				# Scope definitions
				# Note: official tools have 0 block size for all but 11
				# unique typedefs for module
				ieee695_sc(1, ieee695_str(name) + b"\xf9")
				# high level module scope beginning
				+ ieee695_sc(3, ieee695_str(name) + b"\xf9")
				# assembler module scope beginning, empty string, tool type 0
				+ ieee695_sc(10, ieee695_str(name) + b"\x00\x00")
				# module section
				+ b"".join([
					ieee695_sc(11, (
						# name, type = read-only, index, R{i}
						ieee695_str(f".pm{mode}{i:05d}")
						+ b"\x03" + ib1 + b"\xd2" + ib1
						# size
						+ b"\xf9" + len_band_int
					))
					for i, ib1, _, _, len_band_int, _ in band_data
				])
				+ b"\xf9"
			),
			# ASW5 - Data part
			b"".join([
				# Section begin, index
				b"\xe5" + ib1
				# Assign program pointer, index, value (R{i})
				+ b"\xe2\xd0" + ib1 + b"\xd2" + ib1
				# Load data, size in MAUs (1-127), data
				+ b"".join([
					b"\xed"
					+ len(data).to_bytes(1, "big")
					+ data[::-1]
					for data in chunk(band, 127)
				])
				for _, ib1, _, _, _, band in band_data
			]),
			# ASW6 - Trailing part (unused)
			b"",
			# ASW7 - Module end part
			b"\xe1"
		]

		# Finish header
		# TODO: test if can be compressed
		header_size = f.tell() + 8 * 8
		next_offset = header_size
		for i, x in enumerate(w):
			f.write(
				b"\xe2\xd7" + i.to_bytes(1, "big")
				+ ieee695_int(next_offset if x else 0, size=4)
			)
			next_offset += len(x)

		# Write parts
		for x in w:
			f.write(x)
	print(f"wrote {out}")


def ieee695_str(s: str, context="String"):
	assert len(s) < 0x80, f'{context} "{s}" is too long.'
	return len(s).to_bytes(1, "big") + s.encode("ascii")


def ieee695_int(i: int, context="Number", *, size=0):
	if not (0 <= size <= 4):
		raise ValueError("size must be between 0 and 4")

	if size == 0 and 0 <= i < 0x80:
		return i.to_bytes(1, "big")

	assert i < 0x8000_0000, f'{context} "{i}" is too large.'

	if not size:
		if i < 0 or i & 0xff000000:
			size = 4
		elif i & 0x00ff0000:
			size = 3
		elif i & 0x0000ff00:
			size = 2
		elif i & 0x000000ff:
			size = 1
	elif i < 0 and size != 4:
		raise ValueError("signed numbers must be size=4")

	return (
		(0x80 | size).to_bytes(1, "big")
		+ i.to_bytes(size, "big", signed=i < 0)
	)


def ieee695_co(level: int, comment: str):
	return (
		b"\xea"
		+ level.to_bytes(1, "little")
		+ ieee695_str(comment, "Comment")
	)


def ieee695_sc(block: int, content: str):
	# Final block size contains all but the F8
	len_content = len(content)
	len_len = 1
	as_int = ieee695_int(len_content + 1 + len_len)
	while len(as_int) > len_len:
		len_len = len(as_int)
		as_int = ieee695_int(len_content + 1 + len_len)

	return (
		b"\xf8"
		+ block.to_bytes(1, "little")
		+ as_int
		+ content
	)


def write_asm(mode, fn: pathlib.Path, args, px_bands, memory_model="", sect_type="DATA"):
	out = fn.with_suffix(".src")
	var_name = get_var_name(mode, fn)
	mode_align = 64 if mode == "sprites" else 8

	with out.open("wt", encoding="ascii") as f:
		f.write(
			"; Generated by img2c\n\n"
			"$CASE ON\n"
		)
		if memory_model:
			memory_model = memory_model.upper()
			assert memory_model in tuple("SCDL")
			f.write(f"$MODEL {memory_model}\n")
		for i, band in enumerate(px_bands):
			i += 1
			sect = f".pm{mode}{i:05d}"
			label = f"_{var_name}{i}"
			f.write(
				f"\n\tDEFSECT '{sect}', {sect_type}, ROMDATA, FIT 10000H\n"
				f"\tSECT    '{sect}'\n"
				f"\tALIGN   {mode_align}\n"
				f"{label}:\n"
			)

			# Stylized this a bit like the official graphics utility
			# But that also names tiles and doesn't write section info
			for data in chunk(bytes(band), mode_align):
				f.write(f"\tDB ")
				f.write(",".join(as88_hex_byte(x) for x in data))
				f.write("\n")
			
			f.write(f"\n\tGLOBAL {label}\n")
		f.write("\tEND\n")
	print(f"wrote {out}")


def as88_hex_byte(n: int):
	return f"0{n:02X}h"


def random_redraw(im: Image.Image, colors: Colors):
	# Select distinct colors
	import random
	distinct = [
		# https://stackoverflow.com/a/4382138
		0xFFB300, # Vivid Yellow
		0x803E75, # Strong Purple
		0xFF6800, # Vivid Orange
		0xA6BDD7, # Very Light Blue
		0xC10020, # Vivid Red
		0xCEA262, # Grayish Yellow
		0x817066, # Medium Gray
	]
	selected = [
		(x >> 16, (x >> 8) & 0xff, x & 0xff)
		for x in random.sample(distinct, len(colors))
	]

	out = im.convert("RGBA")
	out.putdata([
		selected[-1 if idx == "t" else idx]
		for px in out.getdata()
		for idx in [colors.get(px)]
	])
	return out


def rgb(hex):
	return (hex >> 16, (hex >> 8) & 0xff, hex & 0xff)


def identify_colors(img: Image.Image, max_colors: int, transparency: bool, invert: bool) -> Colors:
	if max_colors < 2 or max_colors > 6:
		raise ProgramError("max colors should be between 2 and 6 (inclusive)")

	expected = max_colors + int(transparency)
	rgba_im: Image = img.convert("RGBA")
	colors = rgba_im.getcolors()
	if len(colors) > expected:
		print(colors)
		raise ProgramError(f"image has too many colors: expected {expected}, got {len(colors)}")

	l_im: Image = img.convert("L")
	hsv_im: Image = img.convert("HSV")
	rem_colors = {c[1] for c in colors}
	cdata = []
	width = img.size[0]
	for i, p in enumerate(rgba_im.getdata()):
		if p in rem_colors:
			rem_colors.remove(p)
			xy = (i % width, i // width)
			cdata.append(Colors.Color(p, hsv_im.getpixel(xy), l_im.getpixel(xy)))
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

	if invert:
		cdata.sort(key=lambda x: x.l)
	else:
		cdata.sort(key=lambda x: -x.l)

	return Colors(cdata, transparent)


	# scheme = {
	# 	2: (0xa9bda9, 0x203220),
	# 	3: (0xa9bda9, 0x556855, 0x203220),
	# 	4: (0xa9bda9, 0x708370, 0x3d4f3d, 0x203220),
	# 	5: (0xa9bda9, 0x7b8f7b, 0x556855, 0x304230, 0x203220),
	# 	#6: (0xa9bda9, 0x203220), # TODO
	# }


def convert_tiles(img: Image.Image, colors: Colors):
	pixels = []
	width, height = img.size

	img = img.convert("RGBA")
	ncolors = max(len(colors.colors), 2)
	if ncolors not in get_grays:
		raise ProgramError(f"Found unacceptible number of colors: {ncolors}\n{colors.colors}")

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
	ncolors = max(len(colors.colors), 2)
	if ncolors not in get_grays:
		raise ProgramError(f"Found unacceptible number of colors: {ncolors}\n{colors.colors}")

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
	return (1 if b == "t" else b,)

def get_grays3(b: int, x: int, y: int) -> tuple:
	if b == 1:
		return (0, 1) if (x + y) % 2 else (1, 0)
	b = 1 if b else 0
	return (b, b)

get_grays = {
	2: get_grays2,
	3: get_grays3,
}


if __name__ == "__main__":
	main()
