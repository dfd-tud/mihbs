#!/usr/bin/env python3

import cv2
import os
import string


def escape_home_in_path(sPath):
    return os.path.expanduser(sPath)


def save_image(aImage, sPathToImage):
    """saves a given image to the given location"""
    cv2.imwrite(escape_home_in_path(sPathToImage), aImage)


def load_image(sPathToImage):
    """load an image from disk"""
    return cv2.imread(escape_home_in_path(sPathToImage))


def create_path(sPath):
    """ create a path if it is not existend yet"""
    sPathEscape = os.path.expanduser(sPath)
    try:
        if not os.path.exists(sPathEscape):
            os.makedirs(sPathEscape)
        return sPathEscape
    except:
        print("some error occurred while creating dir %s" % (sPathEscape))


def check_if_path_exists(sPath):
    """ returns Ture if a given path is existend """
    sPathEscape = os.path.expanduser(sPath)
    return os.path.exists(sPathEscape)


def check_if_file_exists(sPath):
    """ returns Ture if a given path is a file and is existend """
    sPathEscape = os.path.expanduser(sPath)
    return os.path.isfile(sPathEscape)


def list_all_images_in_directory(sDirectoryPath):
    """ returns as list of all images in a given subfolder """
    sDirectoryPathEscaped = escape_home_in_path(sDirectoryPath)
    if not check_if_path_exists(sDirectoryPathEscaped):
        raise Exception("path %s does not exist" % sDirectoryPathEscaped)
    if not sDirectoryPathEscaped.endswith("/"):
        sDirectoryPathEscaped += "/"
    aFiles = next(os.walk(sDirectoryPathEscaped))[2]
    # filter image files
    return [sDirectoryPathEscaped +
            m for m in aFiles if m.lower().endswith((".png", ".bmp", ".jpg", ".jpeg", ".tiff"))]


def format_filename(s):
    """
        Thanks to Sean Hammond (https://github.com/seanh)
        Take a string and return a valid filename constructed from the string.
        Uses a whitelist approach: any characters not present in valid_chars are
        removed. Also spaces are replaced with underscores.

        Note: this method may produce invalid filenames such as ``, `.` or `..`
        When I use this method I prepend a date string like '2009_01_15_19_46_32_'
        and append a file extension like '.txt', so I avoid the potential of using
        an invalid filename.

        """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ', '_')  # I don't like spaces in filenames.
    return filename
