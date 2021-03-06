################################################################################
# COMP3317 Computer Vision
# Assignment 2 - Conrner detection
################################################################################
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

################################################################################
#  perform RGB to grayscale conversion
################################################################################


def rgb2gray(img_color):
    # img_gray = np.dot(img_color[..., :3], [0.2989, 0.5870, 0.1140])
    img_gray = img_color[..., :3] @ [0.2989, 0.5870, 0.1140]
    return img_gray

################################################################################
#  perform 1D smoothing using a 1D horizontal Gaussian filter
################################################################################


def smooth1D(img, sigma):

    filter_range = (sigma * ((2 * np.log(1000))**0.5))
    kernel = np.arange((-1 * filter_range), (filter_range + 1))
    g_filter = np.exp((kernel**2)/-2/(sigma**2))
    # g_filter /= g_filter.sum()

    dimx, dimy = img.shape
    one_matrix = np.zeros((dimx, dimy)) + 1

    filtered_img = convolve1d(img, g_filter, 1, np.float64, 'constant', 0, 0)
    weigth_filter = convolve1d(
        one_matrix, g_filter, 1, np.float64, 'constant', 0, 0)

    img_smoothed = np.divide(filtered_img, weigth_filter)

    return img_smoothed

################################################################################
#  perform 2D smoothing using 1D convolutions
################################################################################


def smooth2D(img, sigma):
    img_smoothed = smooth1D(smooth1D(img, sigma).T, sigma).T
    return img_smoothed

################################################################################
#   perform Harris corner detection
################################################################################


def test_local_maxima(x, y, matrix):  # test for presence of a local maxima
    for i in range(-1, 2):
        for j in range(-1, 2):
            if matrix[y+j, x+i] > matrix[y, x]:
                return False
    return True


def quadratic_approximation(x, y, matrix):
    top = matrix[y-1, x] if (y > 0) else 0
    left = matrix[y, x-1] if (x > 0) else 0
    right = matrix[y, x+1] if (x < (matrix.shape[1]-1)) else 0
    bottom = matrix[y+1, x] if (y < (matrix.shape[0]-1)) else 0
    a = (left + right - 2*matrix[y, x])/2
    b = (top + bottom - 2*matrix[y, x])/2
    c = (right - left)/2
    d = (bottom - top)/2
    e = matrix[y, x]
    dx = -c/2/(a+1e-8)
    dy = -d/2/(b+1e-8)
    R = (a*(dx**2)) + (b*(dy**2)) + (c*dx) + (d*dy) + e
    return (dx, dy, R)


def harris(img, sigma, threshold):
    # input:
    #    img - a h x w numpy ndarry holding the input image
    #    sigma - sigma value of the Gaussian function used in smoothing
    #    threshold - threshold value used for identifying corners
    # return:
    #    corners - a list of tuples (x, y, r) specifying the coordinates
    #              (up to sub-pixel accuracy) and cornerness value of each corner

    # TODO: compute Ix & Iy

    k_const = 0.04
    Ix, Iy = np.gradient(img)

    # TODO: compute Ix2, Iy2 and IxIy

    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    IxIy = np.multiply(Ix, Iy)

    # TODO: smooth the squared derivatives

    Ix2 = smooth2D(Ix2, sigma)
    Iy2 = smooth2D(Iy2, sigma)
    IxIy = smooth2D(IxIy, sigma)

    # TODO: compute cornesness function R

    R = np.zeros_like(img)
    det = np.multiply(Ix2, Iy2) - np.square(IxIy)
    trace = np.add(Ix2, Iy2)
    R = det - (k_const * np.square(trace))

    # TODO: mark local maxima as corner candidates;
    #       perform quadratic approximation to local corners upto sub-pixel accuracy

    cndidate = []  # aims to stores the sub-pixel accuracy pixel value of candidates
    corners = []
    for y in range(R.shape[0]-1):
        for x in range(R.shape[1]-1):
            if x == 0 or y == 0:
                continue
            if test_local_maxima(x, y, R):
                dx, dy, r = quadratic_approximation(x, y, R)
                cndidate.append((x+dx, y+dy, r))

    # TODO: perform thresholding and discard weak corners

    for val in cndidate:
        if val[2] >= threshold:
            corners.append(val)

    return sorted(corners, key=lambda corner: corner[2], reverse=True)

################################################################################
#   save corners to a file
################################################################################


def save(outputfile, corners):
    try:
        file = open(outputfile, 'w')
        file.write('%d\n' % len(corners))
        for corner in corners:
            file.write('%.4f %.4f %.4f\n' % corner)
        file.close()
    except:
        print('Error occurs in writting output to \'%s\'' % outputfile)
        sys.exit(1)

################################################################################
#   load corners from a file
################################################################################


def load(inputfile, outputfile=None):
    try:
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        print('loading %d corners' % nc)
        corners = list()
        for i in range(nc):
            line = file.readline()
            (x, y, r) = line.split()
            corners.append((float(x), float(y), float(r)))
        file.close()
        return corners
    except:
        print('Error occurs in writting output to \'%s\'' % outputfile)
        sys.exit(1)

################################################################################
# main
################################################################################


def main():
    parser = argparse.ArgumentParser(description='COMP3317 Assignment 2')
    parser.add_argument('-i', '--inputfile', type=str,
                        default='grid1.jpg', help='filename of input image')
    parser.add_argument('-s', '--sigma', type=float,
                        default=1.0, help='sigma value for Gaussain filter')
    parser.add_argument('-t', '--threshold', type=float,
                        default=1e6, help='threshold value for corner detection')
    parser.add_argument('-o', '--outputfile', type=str,
                        help='filename for outputting corner detection result')
    args = parser.parse_args()

    print('------------------------------')
    print('COMP3317 Assignment 2')
    print('input file : %s' % args.inputfile)
    print('sigma      : %.2f' % args.sigma)
    print('threshold  : %.2e' % args.threshold)
    print('output file: %s' % args.outputfile)
    print('------------------------------')

    # load the image
    try:
        # img_color = imageio.imread(args.inputfile)
        img_color = plt.imread(args.inputfile)
        print('%s loaded...' % args.inputfile)
    except:
        print('Cannot open \'%s\'.' % args.inputfile)
        sys.exit(1)
    # uncomment the following 2 lines to show the color image
    # plt.imshow(np.uint8(img_color))
    # plt.show()

    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)
    # uncomment the following 2 lines to show the grayscale image
    # plt.imshow(np.float32(img_gray), cmap='gray')
    # plt.show()

    # perform corner detection
    print('perform Harris corner detection...')
    corners = harris(img_gray, args.sigma, args.threshold)
    #corners = load("./corners.lst")

    # plot the corners
    print('%d corners detected...' % len(corners))
    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]
    fig = plt.figure()
    plt.imshow(np.float32(img_gray), cmap='gray')
    plt.plot(x, y, 'r+', markersize=5)
    plt.show()

    # save corners to a file
    if args.outputfile:
        save(args.outputfile, corners)
        print('corners saved to \'%s\'...' % args.outputfile)


if __name__ == '__main__':
    main()
