#!/usr/bin/env python
#
# Copyright (C) 2014 Vangelis Tasoulas <vangelis@tasoulas.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


# This script can be used to stabilize or deshake the source pictures taken
# for a timelapse if your tripod was slighly moving during capturing (it could
# be very windy for example).
#
# The script is using both GraphicsMagick and ImageMagick libraries to perform its tasks.
# ImageMagick has a much better documentation and available examples since it is much
# more popular, but GraphicsMagic is much faster in many operations. So whenever I was
# able to make things work with GraphicsMagick and I tested it to be faster, I used
# GraphicsMagick instead of ImageMagick.
#
# In an Ubuntu 14.04 system install the following packages before you run the script:
#
# apt-get install python-pgmagick python-wand python-progressbar libmagickwand-dev
#
# 1. All of the pictures MUST have same dimensions
# 2. The pictures will be sorted by the exif:DateTimeDigitized info
#
# parallel -j $(cat /proc/cpuinfo | grep processor | wc -l) gm mogrify -quality 100 -output-directory resized -resize 1920x1080 ::: *.JPG # graphicsmagick
# align_image_stack -C -a aligned_ -l DSC_68* # hugin-tools

# TODO: Convert the script to use only graphicsmagick since it looks like it's much faster.

import os
import sys
import re
import argparse
import logging
import traceback
import progressbar
import imghdr
import math
import shutil
import ctypes
from collections import OrderedDict
# GraphicsMagick Documentation: http://pythonhosted.org//pgmagick/
#                               http://www.graphicsmagick.org/Magick++/Image.html
#                               http://www.youlikeprogramming.com/2014/01/graphicsmagicks-pgmagick-library-for-python-a-quick-reference/
import pgmagick

# ImageMagick Documentation: http://docs.wand-py.org/en/0.3.8/
#                            https://media.readthedocs.org/pdf/wand/0.3-maintenance/wand.pdf
import wand
from wand.image import Image
from wand.display import display

__all__ = [
    'print_', 'isValidImg', 'LOG'
]

PROGRAM_NAME = 'Vag Timelapse Photo Deshaker'
VERSION = '0.0.1'
AUTHOR = 'Vangelis Tasoulas'

LOG = logging.getLogger('default.' + __name__)

RESOLUTIONS = {"720p":(1280, 720),
               "1080p":(1920, 1080),
               "4k":(3840, 2160)}

########################################
###### Configure logging behavior ######
########################################
# No need to change anything here

def _configureLogging(loglevel):
    """
    Configures the default logger.

    If the log level is set to NOTSET (0), the
    logging is disabled

    # More info here: https://docs.python.org/2/howto/logging.html
    """
    numeric_log_level = getattr(logging, loglevel.upper(), None)
    try:
        if not isinstance(numeric_log_level, int):
            raise ValueError()
    except ValueError:
        error_and_exit('Invalid log level: %s\n'
        '\tLog level must be set to one of the following:\n'
        '\t   CRITICAL <- Least verbose\n'
        '\t   ERROR\n'
        '\t   WARNING\n'
        '\t   INFO\n'
        '\t   DEBUG    <- Most verbose'  % loglevel)

    defaultLogger = logging.getLogger('default')

    # If numeric_log_level == 0 (NOTSET), disable logging.
    if(not numeric_log_level):
        numeric_log_level = 1000
    defaultLogger.setLevel(numeric_log_level)

    logFormatter = logging.Formatter()

    defaultHandler = logging.StreamHandler()
    defaultHandler.setFormatter(logFormatter)

    defaultLogger.addHandler(defaultHandler)

#######################################################
###### Add command line options in this function ######
#######################################################
# Add the user defined command line arguments in this function

def _command_Line_Options():
    """
    Define the accepted command line arguments in this function

    Read the documentation of argparse for more advanced command line
    argument parsing examples
    http://docs.python.org/2/library/argparse.html
    """

    parser = argparse.ArgumentParser(description=PROGRAM_NAME + " version " + VERSION)

    parser.add_argument("-v", "--version",
                        action="version", default=argparse.SUPPRESS,
                        version=VERSION,
                        help="show program's version number and exit")
    parser.add_argument("-r", "--reference-image",
                        action="store",
                        dest="ref_img",
                        metavar="IMG",
                        help="Image to use as a reference. All the other images have to be located in the same folder as the reference image and will be shifted according to the reference image. The reference image will not be shifted.")
    parser.add_argument("-m", "--max-shake",
                        action="store",
                        type=float,
                        default=1.0,
                        dest="max_shake",
                        metavar="MAX_SHAKE",
                        help="Defines the maximum shake between the pictures given in a percentage of the widest side of the picture. For example, if you choose the max-shake to be 1.5 and your resolution is 1280x960, you tell to the program that the maximum shake between consecutive pictures is 1280*0.015 = 19.2 pixels rounded up to 20. Default Value = 1")
    parser.add_argument("-o", "--output-resolution",
                        action="store",
                        choices=["720p", "1080p", "4k", "original_size"],
                        dest="output_res",
                        default="1080p",
                        metavar="OUTPUT_RES",
                        help="Chooses the output resolution of the resulting pictures. Valid choices are: 720p, 1080p, 4k or original_size. Default value: 1080p")


    loggingGroupOpts = parser.add_argument_group('Logging Options', 'List of optional logging options')
    loggingGroupOpts.add_argument("-q", "--quiet",
                                  action="store_true",
                                  default=False,
                                  dest="isQuiet",
                                  help="Disable logging in the console. Nothing will be printed.")
    loggingGroupOpts.add_argument("-l", "--loglevel",
                                  action="store",
                                  default="INFO",
                                  dest="loglevel",
                                  metavar="LOG_LEVEL",
                                  help="LOG_LEVEL might be set to: CRITICAL, ERROR, WARNING, INFO, DEBUG. (Default: INFO)")

    opts = parser.parse_args()

    if opts.isQuiet:
        opts.loglevel = "NOTSET"

    if opts.ref_img:
        if not os.path.isfile(opts.ref_img):
            error_and_exit("The provided reference image is not a valid file")

    if opts.max_shake <= 0 or opts.max_shake > 100:
        error_and_exit("A valid value for the max shake is in this range: '0 < max_range <= 100'")

    return opts

################################################
############### HELPER FUNCTIONS ###############
################################################
# I have already added a bunch of helper functions
# that I use often. If you don't need them, feel
# free to remove them (except the error_and_exit() function)

#----------------------------------------------------------------------
def error_and_exit(message):
    """
    Prints the "message" and exits with status 1
    """
    print("\nFATAL ERROR:\n" + message + "\n")
    exit(1)

#----------------------------------------------------------------------
def print_(value_to_be_printed, print_indent=0, spaces_per_indent=4, endl="\n"):
    """
    This function, among anything else, it will print dictionaries (even nested ones) in a good looking way

    # value_to_be_printed: The only needed argument and it is the
                           text/number/dictionary to be printed
    # print_indent: indentation for the printed text (it is used for
                    nice looking dictionary prints) (default is 0)
    # spaces_per_indent: Defines the number of spaces per indent (default is 4)
    # endl: Defines the end of line character (default is \n)

    More info here:
    http://stackoverflow.com/questions/19473085/create-a-nested-dictionary-for-a-word-python?answertab=active#tab-top
    """

    if isinstance(value_to_be_printed, dict):
        for key, value in value_to_be_printed.iteritems():
            if isinstance(value, dict):
                print_('{0}{1!r}:'.format(print_indent * spaces_per_indent * ' ', key))
                print_(value, print_indent + 1)
            else:
                print_('{0}{1!r}: {2}'.format(print_indent * spaces_per_indent * ' ', key, value))
    else:
        string = ('{0}{1}{2}'.format(print_indent * spaces_per_indent * ' ', value_to_be_printed, endl))
        sys.stdout.write(string)

#----------------------------------------------------------------------
def isValidImg(filename):
    """
    Check the file to see if it is an image file.

    The function will return True or False

    More info on the imghdr library
    http://docs.python.org/library/imghdr.html
    """
    try:
        if imghdr.what(filename):
            return True
    except:
        pass

    return False

#----------------------------------------------------------------------
# Very useful: http://stackoverflow.com/questions/19339894/how-to-threshold-an-image-using-wand-in-python
# convert DSC_6848.JPG[+0+0] DSC_6849.JPG[+0+0] -compose difference -composite -verbose info:
MagickEvaluateImageChannel = wand.api.library.MagickEvaluateImageChannel
MagickEvaluateImageChannel.argtypes = [ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_double]

def evaluateImageChannel(self, channel, operation, argument):
    """
    ctypes wrapper to call MagickEvaluateImageChannel and return
    the result which is boolean.

    Usage: evaluateImageChannel(img2clone, 'all_channels', 'threshold', 0.60)
    """
    ret = MagickEvaluateImageChannel(self.wand,
                                     wand.image.CHANNELS[channel],
                                     wand.image.EVALUATE_OPS.index(operation),
                                     argument)

    if ret == 0:
        return False
    else:
        return True

#----------------------------------------------------------------------
# Useful links to build the GetImageChannelStatistics:
# https://github.com/Nullicopter/python-magickwand/blob/master/magickwand/api.py
# http://www.imagemagick.org/ImageMagick-7.0.0/api/MagickCore/struct__ChannelStatistics.html
#
# According to this http://www.imagemagick.org/api/statistic.php#GetImageChannelStatistics and this:
# http://web.njit.edu/all_topics/Prog_Lang_Docs/html/imagemagick/www/api/magick-image.html#MagickGetImageChannelStatistics
# I should use MagickRelinquishMemory() to free the statistics buffer.
class _ChannelStatistics(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ('depth', ctypes.c_ulong),
        ('minima', ctypes.c_double),
        ('maxima', ctypes.c_double),
        ('sum', ctypes.c_double),
        ('sum_squared', ctypes.c_double),
        ('sum_cubed', ctypes.c_double),
        ('sum_fourth_power', ctypes.c_double),
        ('mean', ctypes.c_double),
        ('variance', ctypes.c_double),
        ('standard_deviation', ctypes.c_double),
        ('kurtosis', ctypes.c_double),
        ('skewness', ctypes.c_double),
    ]
ChannelStatistics = _ChannelStatistics

MagickGetImageChannelStatistics = wand.api.library.MagickGetImageChannelStatistics
MagickGetImageChannelStatistics.restype = ctypes.POINTER(ChannelStatistics)
MagickGetImageChannelStatistics.argtypes = [ctypes.c_void_p]

# Accoding to the following info
# http://www.imagemagick.org/discourse-server/viewtopic.php?f=3&t=17583#p66194
#
# and some trial and error from the following info:
# http://www.imagemagick.org/Usage/compare/#sub-image
##################################################################################
# To get the average (mean) grey level as a percentage you can use this
#  command...
#
#     convert image1 image2 -compose Difference -composite \
#           -colorspace gray -format '%[fx:mean*100]' info:
#
#  For non-percentage you can use the even simplier..
#
#     convert image1 image2 -compose Difference -composite \
#           -colorspace gray -format '%[mean]' info:
##################################################################################
# I realize that the statistics return values between 0 and The quantum range.
# (get the quantumrange with img.quantum_range)
# If I want to convert to values between 0 and 1 (what the fx: is doing when I use
# the command line), just divide with the quantum range (and multiply with 100 to
# get percentages).
def getImageStatistics(img):
    """
    ctype wrapper to call MagickGetImageChannelStatistics and return the result
    which is of type ChannelStatistics.

    Return a dict with image statistics for all of the available channels
    """
    img_stats_dict = dict()
    img_stats = MagickGetImageChannelStatistics(img.wand)

    for ch in ["red", "green", "blue", "gray", "yellow", "cyan", "magenta", "composite_channels", "alpha", "opacity"]:
        img_stats_dict[ch] = dict()
        img_stats_dict[ch]['depth'] = img_stats[wand.image.CHANNELS[ch]].depth
        img_stats_dict[ch]['minima'] = img_stats[wand.image.CHANNELS[ch]].minima
        img_stats_dict[ch]['maxima'] = img_stats[wand.image.CHANNELS[ch]].maxima
        img_stats_dict[ch]['mean'] = img_stats[wand.image.CHANNELS[ch]].mean
        img_stats_dict[ch]['standard_deviation'] = img_stats[wand.image.CHANNELS[ch]].standard_deviation
        img_stats_dict[ch]['kurtosis'] = img_stats[wand.image.CHANNELS[ch]].kurtosis
        img_stats_dict[ch]['skewness'] = img_stats[wand.image.CHANNELS[ch]].skewness

    wand.api.library.MagickRelinquishMemory(img_stats)

    return img_stats_dict

#----------------------------------------------------------------------
def find_image_shift(img1_path, img2_path, start_x = 0, start_y = 0, max_shake = 1, resize = 0.25):
    """
    Load img1 from img1_path and img2 from img2_path.
    start_x: x coordinate where to look around for the best match.
    start_y: y coordinate where to look around for the best match.

             start_x and start_y can get negative values as well
             A negative value means that we have to crop the image
             from the right or bottom side, while a positive is the
             other way around

    max_shake: The percentage of the maximum shake (defaule value: 1%)
    resize: Percentage of the original size to resize the pictures
            temporarily to make the calculations faster (default 20%).

    Start shifting img2 and subtract from img1
    Find the mean value of the result.

    The lowest mean value, shows the best match.

    Returns a tuple with the shift_x and shift_y
    """

    mean = dict()

    # Steps to Crop and pad with black color img2:
    # 1. When xShift is positive we crop from the left side so we need
    #     to pad from the right side.
    # 2. When yShift is positive we crop from the top side so we need
    #     to pad the bottom side side.
    # 3. xShift is negative. crop from the right side and pad on the
    #     left to push the image to the right!
    # 4. yShift is negative. Crop from the bottom side and pad on the
    #     top to push the image down!
    with Image(filename = img1_path) as img1:
        img1.type = "grayscale"
        with Image(filename = img2_path) as img2:
            img2.type = "grayscale"
            isinstance(img1, Image)
            isinstance(img2, Image)

            wide_side = img1.width * resize if img1.width > img1.height else img1.height * resize
            max_shake_px = int(math.ceil((max_shake / 100.0) * wide_side))
            if (resize < 1):
                LOG.debug("Will resize pictures to {}% ({}x{}) for faster processing".format(resize * 100, int(img1.width * resize), int(img1.height * resize)))
                LOG.debug(u'Max shake is \u00B1{} pixels on the resized pictures'.format(max_shake_px))
            else:
                LOG.debug(u'Max shake is \u00B1{} pixels'.format(max_shake_px))


            with img1.clone() as img1resized:
                if resize != 1:
                    img1resized.transform(resize='{}%'.format(resize * 100))
                with img2.clone() as img2resized:
                    if resize != 1:
                        img2resized.transform(resize='{}%'.format(resize * 100))

                    # Iterate from (start_y - max_shake_px) to (start_y + max_shake_px)
                    # and (start_x - max_shake_px) to (start_x + max_shake_px) to find
                    # the best match. Crop and move the pictures and get the mean of the
                    # difference between them. The lowest mean shows the best match.
                    for yShift in range(start_y - max_shake_px, start_y + max_shake_px + 1):
                        if abs(yShift) >= img2resized.height:
                            break
                        for xShift in range(start_x - max_shake_px, start_x + max_shake_px + 1):
                            if abs(xShift) >= img2resized.width:
                                break
                            with img1resized.clone() as img1master:
                                with img2resized.clone() as img2victim:
                                    LOG.debug("Moving to the left. xShift: {0}, yShift: {1}".format(xShift, yShift))

                                    if(xShift < 0):
                                        victimLeft = 0
                                        victimWidth = img2victim.width + xShift
                                        masterLeft = abs(xShift)
                                        masterWidth = None
                                    else:
                                        victimLeft = xShift
                                        victimWidth = None
                                        masterLeft = 0
                                        masterWidth = img1master.width + xShift

                                    if(yShift < 0):
                                        victimTop = 0
                                        victimHeight = img2victim.height + yShift
                                        masterTop = abs(yShift)
                                        masterHeight = None
                                    else:
                                        victimTop = yShift
                                        victimHeight = None
                                        masterTop = 0
                                        masterHeight = img1master.height + yShift

                                    img2victim.crop(left = victimLeft, top = victimTop, width = victimWidth, height = victimHeight)
                                    img1master.crop(left = masterLeft, top = masterTop, width = masterWidth, height = masterHeight)
                                    #img1master.composite_channel('all_channels', img2victim, 'difference', 0, 0)
                                    #img1master.type = "grayscale"
                                    img1master.composite_channel('gray', img2victim, 'difference', 0, 0)

                                    image_statistics = getImageStatistics(img1master)
                                    mean[image_statistics['gray']['mean']] = dict()
                                    mean[image_statistics['gray']['mean']]['xShift'] = xShift
                                    mean[image_statistics['gray']['mean']]['yShift'] = yShift

                                    LOG.debug("Current mean at {0}x{1} = {2}".format(xShift, yShift, image_statistics['gray']['mean']))

    smallest_mean = min(mean.keys())
    LOG.debug("Smallest mean of difference is '{}' and located at {}x{}".format(smallest_mean, mean[smallest_mean]['xShift'], mean[smallest_mean]['yShift']))

    return (mean[smallest_mean]['xShift'], mean[smallest_mean]['yShift'])


##################################################
############### WRITE MAIN PROGRAM ###############
##################################################

if __name__ == '__main__':
    """
    Write the main program here
    """
    # Parse the command line options
    options = _command_Line_Options()
    # Configure logging
    _configureLogging(options.loglevel)

    LOG.info("Welcome to " + PROGRAM_NAME + " v" + str(VERSION))

    ######################################
    ### Starting adding your code here ###
    ######################################
    #LOG.critical("CRITICAL messages are printed")
    #LOG.error("ERROR messages are printed")
    #LOG.warning("WARNING messages are printed")
    #LOG.info("INFO message are printed")
    #LOG.debug("DEBUG messages are printed")

    # If options.ref_img exists at this point, it is a valid file.
    # We need to check if it is a valid image as well.
    if options.ref_img:
        if isValidImg(options.ref_img):
            # Get the path for the rest of the images
            working_dir = os.path.dirname(os.path.realpath(options.ref_img))
        else:
            error_and_exit("Not a valid reference image: '" + options.ref_img + "'\nPlease check the mimetype with a command such as:\n\"file -bi '" + options.ref_img + "'\"")
    else:
        working_dir = "./"

    # Save the full path in the variable working_dir with a trailing slash (/)
    working_dir = os.path.realpath(working_dir) + "/"
    LOG.info("The Working directory is: '{}'".format(working_dir))

    # Get a list of all the file/folder names in the working_dir
    img_list = os.listdir(working_dir)
    # Prepend the working dir in the list of images and save them in a dict
    img_dict = {s:dict({'fullpath_original': working_dir + s}) for s in img_list}

    LOG.info("\nSearching for valid images in the working directory")

    myProgressBarFd = sys.stderr
    # If log level is set to 1000, the -q option has been used so redirect the progress bar to /dev/null
    if LOG.getEffectiveLevel() == 1000:
        myProgressBarFd = open("/dev/null", "w")

    pbar = progressbar.ProgressBar(maxval=len(img_list), fd=myProgressBarFd).start()
    pbar_counter = 0
    for f in img_list:
        if not isValidImg(img_dict[f]['fullpath_original']):
            # If the file is not a valid image,
            # then remove it from the img_dict
            LOG.debug("File '" + img_dict[f]['fullpath_original'] + "' is not a valid image file")
            del img_dict[f]
        pbar.update(pbar_counter)
        pbar_counter += 1
    pbar.finish()
    # We don't need img_list anymore since we have img_dict
    del img_list

    LOG.info(str(len(img_dict)) + " images found.")

    source_dimension_x = 0
    source_dimension_y = 0
    resized_width = 0
    resized_height = 0
    for image in img_dict.keys():
        #img = Image(filename=img_dict[image]['fullpath'])
        #img_dict[image]['exif:DateTimeDigitized'] = img.metadata['exif:DateTimeDigitized']
        #x = img.width
        #y = img.height

        # Use GraphicsMagick here since it is much faster. Uncomment the above lines and comment the next ones to use ImageMagick instead.
        img = pgmagick.Image(img_dict[image]['fullpath_original'])
        img_dict[image]['exif:DateTimeDigitized'] = img.attribute("exif:DateTimeDigitized")
        x = img.columns()
        y = img.rows()

        if source_dimension_x == 0 and source_dimension_y == 0:
            source_dimension_x = x
            source_dimension_y = y

            LOG.debug("Resolution of images is {}x{}".format(source_dimension_x, source_dimension_y))
            wide_side = source_dimension_x if source_dimension_x > source_dimension_y else source_dimension_y
            max_shake_px = int(math.ceil((options.max_shake /100.0) * wide_side))
            LOG.debug(u'Max shake is \u00B1{} pixels on the original picture size'.format(max_shake_px))

            if options.output_res != "original_size":
                if RESOLUTIONS[options.output_res][0] > source_dimension_x and RESOLUTIONS[options.output_res][1] > source_dimension_y:
                    error_and_exit("The resolution of the provided photos is smaller than '{} ({}x{})'.\n"
                                   "Please use the '--output-resolution' option to choose either a smaller"
                                   " resolution or 'original_size'".format(options.output_res,
                                                                           RESOLUTIONS[options.output_res][0],
                                                                           RESOLUTIONS[options.output_res][1]))
                else:
                    resized_wide_side = RESOLUTIONS[options.output_res][0] if RESOLUTIONS[options.output_res][0] > RESOLUTIONS[options.output_res][1] else RESOLUTIONS[options.output_res][1]
                    resized_max_shake_px = int(math.ceil((options.max_shake /100.0) * resized_wide_side))

                    resized_width = RESOLUTIONS[options.output_res][0] + 2 * resized_max_shake_px
                    resized_height = RESOLUTIONS[options.output_res][1] + 2 * resized_max_shake_px

                    LOG.info("The final resolution of the pictures will be set to {}".format(options.output_res))
        else:
            if x != source_dimension_x or y != source_dimension_y:
                error_and_exit("Image '{}' dimensions are {}x{}\n"
                               "The dimensions should be {}x{} for all of the images".format(image,
                                                                                             x, y,
                                                                                             source_dimension_x,
                                                                                             source_dimension_y))

        # TODO VERY IMPORTANT: 'fullpath_resized' needs to exist even if the original_size images is chosen!!!!!
        #                       In the very end, we need to crop the images, so we do not want to alter the
        #                       Original images!
        # Resize the picture to perform the rest of the calculations mush faster
        if options.output_res != "original_size":
            if source_dimension_x > RESOLUTIONS[options.output_res][0] or source_dimension_y > RESOLUTIONS[options.output_res][1]:
                tmp_dir = "{}resized_tmp".format(working_dir)
                # TODO: If the path exists, check for a config file to resume operations.
                #       Make a command line option to tune this behavior
                if os.path.isdir(tmp_dir):
                    pass
                    #shutil.rmtree(tmp_dir)
                else:
                    os.makedirs(tmp_dir)

                geometry = pgmagick.Geometry(resized_width, resized_height) # Specifies the desired width, height.
                geometry.aspect(False) # False = maintain aspect ratio.
                img.quality(100)    # Use the maximum quality to avoid quality loss
                img.scale(geometry) # Perform the resize.
                img_dict[image]['fullpath_resized'] = "{}/{}".format(tmp_dir, image)
                img_dict[image]['fullpath'] = img_dict[image]['fullpath_resized']
                img.write(img_dict[image]['fullpath_resized'])
        else:
            img_dict[image]['fullpath'] = img_dict[image]['fullpath_original']



    # Sort the images by capture date and store them in an OrderedDict
    img_dict = OrderedDict(sorted(img_dict.items(), key=lambda x: x[1]['exif:DateTimeDigitized']))
    #print_(img_dict)

    # Up to this point, we have read all of the pictures and the reference image should exist somewhere in img_dict.
    # If no ref_img has been supplied, use the 1st image at index 0 (img_dict.items()[0][0])
    if options.ref_img:
        ref_img = os.path.basename(options.ref_img)
    else:
        ref_img = img_dict.items()[0][0]

    # Find index of the reference image
    for i in range(0, len(img_dict)):
        try:
            if img_dict.items()[i].index(ref_img) == 0:
                ref_img_index = i
                break
        except:
            pass

    LOG.info("Reference image is '" + ref_img + "' at index '" + str(ref_img_index) + "'")

    max_left_offset = 0
    max_right_offset = 0
    max_top_offset = 0
    max_bottom_offset = 0

    pbar_widgets = [progressbar.FormatLabel(''), ' ', progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ', progressbar.RotatingMarker()]
    pbar = progressbar.ProgressBar(widgets = pbar_widgets, maxval = len(img_dict) - 1).start()
    pbar_counter = 0
    # Start from the reference picture and process backwards
    resize = 0.25
    max_shake = 1 / resize / wide_side * 100
    # TODO: Move the contents of the for loops in a new function
    # TODO: Save the pictures if negative xShift and yShift values are chosen
    for i in range(ref_img_index, 0, -1):
        i = i - 1
        pbar_counter = pbar_counter + 1
        pbar_widgets[0] = progressbar.FormatLabel("Processing image '{0}' ({1}/{2})".format(img_dict.items()[i][0], pbar_counter, len(img_dict) - 1))
        pbar.update(pbar_counter)
        temp_xShift, temp_yShift = find_image_shift(img_dict.items()[i+1][1]['fullpath'], img_dict.items()[i][1]['fullpath'], max_shake = options.max_shake, resize=resize)
        # The first run is resizing the pictures to 25% for a rough quick parse so we get a temp_xShift and temp_yShift.
        # After we find the temp shifts, we know that the real shifts are (1 / resize / wide_side) * 100 percent pixels around the temp shifts * 1 / resize, so rerun the find image shift with resize = 1 this time.
        temp_xShift = int(temp_xShift * 1 / resize)
        temp_yShift = int(temp_yShift * 1 / resize)
        img_dict.items()[i][1]['xShift'], img_dict.items()[i][1]['yShift'] = find_image_shift(img_dict.items()[i+1][1]['fullpath'], img_dict.items()[i][1]['fullpath'], start_x = temp_xShift, start_y = temp_yShift, max_shake = max_shake, resize=1)
        with Image(filename = img_dict.items()[i][1]['fullpath']) as img:
            with Image().blank(width=img.width, height=img.height, background=wand.color.Color(string="transparent")) as shifted_img:
                img.crop(left=img_dict.items()[i][1]['xShift'], top=img_dict.items()[i][1]['yShift'], width=None, height=None)
                shifted_img.composite(img, 0, 0)
                shifted_img.save(filename = img_dict.items()[i][1]['fullpath_resized'])

    # After we processed all of the pictures backwards from the reference picture,
    # let's do the same operation but forward until the last picture.
    for i in range(ref_img_index + 1, len(img_dict)):
        pbar_counter = pbar_counter + 1
        pbar_widgets[0] = progressbar.FormatLabel("Processing image '{0}' ({1}/{2})".format(img_dict.items()[i][0], pbar_counter, len(img_dict) - 1))
        pbar.update(pbar_counter)
        temp_xShift, temp_yShift = find_image_shift(img_dict.items()[i-1][1]['fullpath'], img_dict.items()[i][1]['fullpath'], max_shake = options.max_shake, resize=resize)
        temp_xShift = int(temp_xShift * 1 / resize)
        temp_yShift = int(temp_yShift * 1 / resize)
        img_dict.items()[i][1]['xShift'], img_dict.items()[i][1]['yShift'] = find_image_shift(img_dict.items()[i-1][1]['fullpath'], img_dict.items()[i][1]['fullpath'], start_x = temp_xShift, start_y = temp_yShift, max_shake = max_shake, resize=1)
        with Image(filename = img_dict.items()[i][1]['fullpath']) as img:
            with Image().blank(width=img.width, height=img.height, background=wand.color.Color(string="transparent")) as shifted_img:
                img.crop(left=img_dict.items()[i][1]['xShift'], top=img_dict.items()[i][1]['yShift'], width=None, height=None)
                shifted_img.composite(img, 0, 0)
                shifted_img.save(filename = img_dict.items()[i][1]['fullpath_resized'])


    print_(img_dict)

    # TODO: Crop the pictures to the given resolution, or minimum crop (find the max xShift and yShift) if original_size is chosen.
    # Since we have found the shift, crop the original picture and pad with a transparent background.
    #for image in img_dict.keys():
        #if image != ref_img:
            #with Image(filename = img_dict[image]['fullpath']) as img:
                #with Image().blank(width=img.width, height=img.height, background=wand.color.Color(string="transparent")) as shifted_img:
                    #img.crop(left=img_dict[image]['xShift'], top=img_dict[image]['yShift'], width=None, height=None)
                    #shifted_img.composite(img, 0, 0)
                    #display(shifted_img)


    pbar.finish()
