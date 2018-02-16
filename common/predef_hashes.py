"""
Image hashing library
======================

"""

# from PIL import Image
import numpy
import scipy.fftpack
# import pywt
# import os.path
import cv2
import fliphandling


def __resize_image_downscale(aInputImage, lImageWidth, lImageHeight):
    """resizing an image to a given size"""
    return cv2.resize(aInputImage, (lImageWidth, lImageHeight), interpolation=cv2.INTER_AREA)


def __convert_image_to_grayscale(aInputImage):
    """converting an image to grayscale"""
    return cv2.cvtColor(aInputImage, cv2.COLOR_BGR2GRAY)


def average_hash(image, hash_size=8, bFlipHandling=False):
    """
    Average Hash computation

    Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

    Step by step explanation: https://www.safaribooksonline.com/blog/2013/11/26/image-hashing-with-python/

    @image must be a PIL instance.
    """
    if hash_size < 0:
        raise ValueError("Hash size must be positive")

    # fliphandling
    if bFlipHandling:
        image = fliphandling.handle_flip(image)

    # reduce size and complexity, then covert to grayscale
    # image = image.convert("L").resize((hash_size, hash_size),
    # Image.ANTIALIAS)
    image = __convert_image_to_grayscale(image)
    pixels = __resize_image_downscale(image, hash_size, hash_size)

    # find average pixel value; 'pixels' is an array of the pixel values,
    # ranging from 0 (black) to 255 (white)
    # pixels = numpy.array(image.getdata()).reshape((hash_size, hash_size))
    avg = pixels.mean()

    # create string of bits
    diff = pixels > avg
    # make a hash
    return diff.flatten()


def phash(image, hash_size=8, highfreq_factor=4, bFlipHandling=False):
    """
    Perceptual Hash computation.

    Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

    @image must be a PIL instance.
    """
    if hash_size < 0:
        raise ValueError("Hash size must be positive")

    # fliphandling
    if bFlipHandling:
        image = fliphandling.handle_flip(image)

    import scipy.fftpack
    img_size = hash_size * highfreq_factor
    image = __convert_image_to_grayscale(image)
    pixels = __resize_image_downscale(image, img_size, img_size)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
    dctlowfreq = dct[:hash_size, :hash_size]
    avg = numpy.mean(dctlowfreq)
    diff = dctlowfreq > avg
    return diff.flatten()


def phash_simple(image, hash_size=8, highfreq_factor=4, bFlipHandling=False):
    """
    Perceptual Hash computation.

    Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

    @image must be a PIL instance.
    """

    # fliphandling
    if bFlipHandling:
        image = fliphandling.handle_flip(image)

    import scipy.fftpack
    img_size = hash_size * highfreq_factor
    image = __convert_image_to_grayscale(image)
    pixels = __resize_image_downscale(image, img_size, img_size)
    dct = scipy.fftpack.dct(pixels)
    dctlowfreq = dct[:hash_size, 1:hash_size + 1]
    avg = dctlowfreq.mean()
    diff = dctlowfreq > avg
    return diff.flatten()


def dhash(image, hash_size=8, bFlipHandling=False):
    """
    Difference Hash computation.

    following http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

    computes differences horizontally

    @image must be a PIL instance.
    """
    # resize(w, h), but numpy.array((h, w))
    if hash_size < 0:
        raise ValueError("Hash size must be positive")

    # fliphandling
    if bFlipHandling:
        image = fliphandling.handle_flip(image)

    image = __convert_image_to_grayscale(image)
    pixels = __resize_image_downscale(
        image, hash_size + 1, hash_size)
    # compute differences between columns
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()


def dhash_vertical(image, hash_size=8, bFlipHandling=False):
    """
    Difference Hash computation.

    following http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

    computes differences vertically

    @image must be a PIL instance.
    """
    # fliphandling
    if bFlipHandling:
        image = fliphandling.handle_flip(image)

    # resize(w, h), but numpy.array((h, w))
    image = __convert_image_to_grayscale(image)
    pixels = __resize_image_downscale(
        image, hash_size, hash_size + 1)  # TODO: right order?
    # compute differences between rows
    diff = pixels[1:, :] > pixels[:-1, :]
    return diff.flatten()


def whash(image, hash_size=8, image_scale=None, mode='haar', remove_max_haar_ll=True, bFlipHandling=False):
    """
    Wavelet Hash computation.

    based on https://www.kaggle.com/c/avito-duplicate-ads-detection/

    @image must be a PIL instance.
    @hash_size must be a power of 2 and less than @image_scale.
    @image_scale must be power of 2 and less than image size. By default is equal to max
            power of 2 for an input image.
    @mode (see modes in pywt library):
            'haar' - Haar wavelets, by default
            'db4' - Daubechies wavelets
    @remove_max_haar_ll - remove the lowest low level (LL) frequency using Haar wavelet.
    """
    # fliphandling
    if bFlipHandling:
        image = fliphandling.handle_flip(image)

    import pywt
    if image_scale is not None:
        assert image_scale & (
            image_scale - 1) == 0, "image_scale is not power of 2"
    else:
        # TODO: correct translation? vvvv
        image_natural_scale = 2**int(numpy.log2(min(image.shape[:2])))
        image_scale = max(image_natural_scale, hash_size)

    ll_max_level = int(numpy.log2(image_scale))

    level = int(numpy.log2(hash_size))
    assert hash_size & (hash_size - 1) == 0, "hash_size is not power of 2"
    assert level <= ll_max_level, "hash_size in a wrong range"
    dwt_level = ll_max_level - level

    image = __convert_image_to_grayscale(image)
    pixels = __resize_image_downscale(image, image_scale, image_scale)
    pixels = pixels.astype(float)
    pixels /= 255

    # Remove low level frequency LL(max_ll) if @remove_max_haar_ll using haar
    # filter
    if remove_max_haar_ll:
        coeffs = pywt.wavedec2(pixels, 'haar', level=ll_max_level)
        coeffs = list(coeffs)
        coeffs[0] *= 0
        pixels = pywt.waverec2(coeffs, 'haar')

    # Use LL(K) as freq, where K is log2(@hash_size)
    coeffs = pywt.wavedec2(pixels, mode, level=dwt_level)
    dwt_low = coeffs[0]

    # Substract median and compute hash
    med = numpy.median(dwt_low)
    diff = dwt_low > med
    return diff.flatten()
