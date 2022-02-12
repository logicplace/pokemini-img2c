# pokemini-img2c

Convert graphics to be usable by the Epson c88 compiler system.

## Usage

Run from root folder of your project.

```sh
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
 You can also include rules to generate them. Here's one using the rsc directory.
 It can't compile them to the src directory, though, unless you want to specify
 the new obj names manually...

```makefile
TOOLCHAIN_DIR := ../..
TARGET = my_project

C_SOURCES = src/isr.c src/main.c # others...
ASM_SOURCES = src/startup.asm # others...
IMAGES = rsc/my_sprites.png rsc/my_tiles.png # others...

OBJS = $(IMAGES:.png=.obj)

.SUFFIXES: .png
.png.obj:
    py img2c.py $<

include $(TOOLCHAIN_DIR)/pm.mk
```

