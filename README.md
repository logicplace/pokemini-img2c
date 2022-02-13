# pokemini-img2c

Convert graphics to be usable by the Epson c88 compiler system.

## Graphics

The tool tries to be intelligent about the graphics you make and as such does not have any strict requirements on which colors to use.
 While you should use white and black for the solid colors, any gray will work, just use the same gray throughout.

From a technical standpoint, the tool will sort colors used by luminosity. For sprites, it will prefer actual transparency for transparent pixels, but if there are none, it will use the most chromographically distinct color as transparency, if any exist (for instance, magenta). Of course, tiles don't support transparency.

The tool currently supports up to three colors images. Eventually, it will be able to support up to 5 by use of the `--colors` flag.

If you only expect 2-color images, it would be good to restrict the call with `--colors 2` or `-c 2` so that any off-black or off-white pixels will cause an error.

Any file format which Pillow supports is allowed. So: PNG, BMP, etc.

## Usage

Run from root folder of your project. You must install Pillow first.

```sh
# Install Pillow
pip install Pillow --user

# If your images are in rsc and code in src
py img2c.py -o src -s rsc/*_sprites.png -t rsc/*_tiles.png

# If your images are in the root directory and code in src
py img2c.py -o src -s *_sprites.png -t *_tiles.png

# If your images are in the the src directory with the code
py img2c.py -s src/*_sprites.png -t src/*_tiles.png

# If your images are names like the above examples in rsc
py img2c.py rsc/*.png
```

## Makefile

You'll need to update your project's Makefile to include the new objects.
 You can also include rules to generate them.
 Here's one using the rsc directory.
 It can't compile them to the src directory, though.

```makefile
TARGET = my_project

C_SOURCES = src\isr.c src\main.c # others...
ASM_SOURCES = src\startup.asm # others...
IMAGES = rsc/my_sprites.png rsc/my_tiles.png # others...

OBJS = $(IMAGES:.png=.obj)

include ../../pm.mk

.SUFFIXES: .png
.png.obj:
    py img2c.py -O src $<
```

