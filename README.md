# Corner-Detection

COMP3317 Computer Vision - Assignment 2 [Corner Detection Algorithm]

## Guide

Run the Commands:

```bash
cd Project/
python3 assign2.py -i [[filename]]
```

## Functions

- rgb2gray
  - converts RGB to greyscale by multiplying RGB values against weighted brightness of YIQ.

* smooth1D

  - Apply gaussain weighted smoothing to smooth image linearly in one direction (horizontally)
  - Using sigma we calcuate a discreet range and create a 1D kernel using this range.
  - Firstly, we convolve the image with kernel. Then we convolve a 1's matrix of same size with kernel. Then we divide the result of the First by the second. This gives us an image smoothes in one direction.

* smooth2D

  - We smooth image across one direction.
  - We transpose the result
  - We smooth it again, this allows smoothing across the horizontal axis as well.
  - we transpose the result back to original orientation

* test_local_maxima

  - searches if a local maxima exists for a pixel within a 9x9 group (m - adjacent).
  - we iterate through each cell and match it againsts its neighbors
  - border cases are exempted

* quadratic_approximation

  - obtains value of local maxima through quadratic approximation.
  - We use it indiscriminately and on each attempt as we are of the assumption that it is highly unlikely that a corner would lie on a discreet pixel.
  - Even if corner does lie on a discreet pixel then function simply returns the value of that pixel.

* harris
  - finds gradient images using numpy.gradient
  - finds Ix2, Iy2, IxIy using numpy.multiply and numpy.square
  - smooths images of Ix2, Iy2, IxIy using function smooth2D
  - creates image R by formula det(A) - k(trace(A) \*\* 2)
  - iterates through R testing for local maxima and using quadratic approximation where found.
  - corners are appended into a list and returned
