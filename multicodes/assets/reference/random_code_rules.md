# Four-Bit Random-Code Rules

Each cross-shaped DNA origami structure has horizontal and vertical arms.
Streptavidin appears as white dots at designed sites.

An orange marker identifies the upper-left side of the vertical arm and defines
the orientation of the structure.

The four bits are assigned in this order:

1. Vertical arm, left side: top = 1, bottom = 0.
2. Vertical arm, right side: top = 1, bottom = 0.
3. Horizontal arm, upper side: right = 1, left = 0.
4. Horizontal arm, lower side: right = 1, left = 0.

Always determine orientation from the marker before reading the bits.

Concatenate the four decisions in the stated order:

```text
bit1 bit2 bit3 bit4
```

The result is one of:

```text
0000, 0001, ..., 1111
```

For a cross chain, read each cross independently and concatenate the codes in
left-to-right or top-to-bottom biological order.

Example:

```text
left cross = 00
right cross = 11
chain code = 0011
```

The reference images in this directory show canonical geometry and example
placements. The orange marker is the orientation anchor. Colored points in
pre-annotated images should be treated as positional evidence unless they
clearly contradict the underlying white-dot structure.
