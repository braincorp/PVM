"""
Exports a movie in pickled labeled movie format to a zip archive containing the images and text description
of the bounding boxes for each target.

This format is often used in academic circles for sharing data.

Usage:
::

    python export_to_zip.py -i my_file.pkl -o myfile.zip -c low_res

"""

# ==================================================================================
# Copyright (c) 2016, Brain Corporation
#
# This software is released under Creative Commons
# Attribution-NonCommercial-ShareAlike 3.0 (BY-NC-SA) license.
# Full text available here in LICENSE.TXT file as well as:
# https://creativecommons.org/licenses/by-nc-sa/3.0/us/legalcode
#
# In summary - you are free to:
#
#    Share - copy and redistribute the material in any medium or format
#    Adapt - remix, transform, and build upon the material
#
# The licensor cannot revoke these freedoms as long as you follow the license terms.
#
# Under the following terms:
#    * Attribution - You must give appropriate credit, provide a link to the
#                    license, and indicate if changes were made. You may do so
#                    in any reasonable manner, but not in any way that suggests
#                    the licensor endorses you or your use.
#    * NonCommercial - You may not use the material for commercial purposes.
#    * ShareAlike - If you remix, transform, or build upon the material, you
#                   must distribute your contributions under the same license
#                   as the original.
#    * No additional restrictions - You may not apply legal terms or technological
#                                   measures that legally restrict others from
#                                   doing anything the license permits.
# ==================================================================================
import zipfile
from PVM_tools import labeled_movie
import PVM_framework.PVM_Storage as PVM_Storage
import PVM_framework.PVM_datasets as PVM_datasets
import argparse
import cv2
import time
import os


def export_to_zip(infile, outfile, channel, compression_level=80):
    """
    Exports a movie in pickled labeled movie format to a zip archive containing the images and text description
    of the bounding boxes for each target.

    This format is often used in academic circles for sharing data.

    :param infile: name of the input pickle
    :param outfile: name of the output zip archive
    :param channel: channel to be processed
    :param compression_level: jpeg compression level
    :return:
    """
    fc = labeled_movie.FrameCollection()
    fc.load_from_file(infile)
    myzip = zipfile.ZipFile(outfile, "w")
    folder = os.path.basename(outfile)[:-4]
    targets = fc.Frame(0).get_targets(channel=channel)
    poster_frame = fc.Frame(0).get_image(channel=channel)
    label_str = {}
    for target in targets:
        label_str[target] = ""
    for i in xrange(len(fc)):
        zipi = zipfile.ZipInfo()
        zipi.filename = folder+"/img/%04d.jpg" % i
        zipi.date_time = time.localtime()[:6]
        zipi.compress_type = zipfile.ZIP_DEFLATED
        zipi.external_attr = 0777 << 16L
        img = fc.Frame(i).get_image(channel=channel)
        for target in targets:
            label = fc.Frame(i).get_label(target=target)
            if label.empty:
                label_str[target] += "-1, -1, -1, -1\n"
            else:
                box = label.get_box_pixels()
                label_str[target] += "%d, %d, %d, %d\n" % box

        (ret, buf) = cv2.imencode(".jpg", img, (cv2.IMWRITE_JPEG_QUALITY, compression_level))
        myzip.writestr(zipi, buf)
    for (i, target) in enumerate(targets):
        zipi = zipfile.ZipInfo()
        zipi.filename = folder+"/groundtruth_rect.%d.txt" % i
        zipi.date_time = time.localtime()[:6]
        zipi.compress_type = zipfile.ZIP_DEFLATED
        zipi.external_attr = 0777 << 16L
        myzip.writestr(zipi, label_str[target])
    myzip.close()
    poster_frame = cv2.resize(poster_frame, dsize=(120, 120), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(outfile.split('.')[0]+".jpg", poster_frame)


if __name__ == "__main__":
    desc = """
    Exports a movie in pickled labeled movie format to a zip archive containing the images and text description
    of the bounding boxes for each target.

    This format is often used in academic circles for sharing data.

    Usage:

    python export_to_zip.py -i my_file.pkl -o myfile.zip -c low_res
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-c", "--channel", type=str, default="default", help="Channel")
    parser.add_argument("-i", "--input", type=str, default="example.pkl", help="Input labeled movie")
    parser.add_argument("-o", "--output", type=str, default="example.zip", help="Output zip archive")
    parser.add_argument("-s", "--set", type=str, default="", help="Dataset")
    parser.add_argument("-O", "--out_folder", type=str, default="", help="Folder to store multiple files")
    parser.add_argument("-p", "--print_html", help="Print tr/td html elements to paste into a website", action="store_true")
    args = parser.parse_args()
    if args.set == "":
        export_to_zip(infile=args.input, outfile=args.output, channel=args.channel)
    else:
        storage = PVM_Storage.Storage()
        dataset = PVM_datasets.PVMDataset(name="all")
        for (i, [f, c]) in enumerate(dataset.all):
            if not args.print_html:
                print "Processing file %s" % f
            local_path = storage.get(f)
            out_filename = os.path.splitext(os.path.basename(local_path))[0]
            out_filename_zip = out_filename+".zip"
            out_complete = os.path.expanduser(os.path.join(args.out_folder, out_filename_zip))
            export_to_zip(infile=local_path, outfile=out_complete, channel=args.channel)
            if args.print_html:
                if i % 6 == 0:
                    print "</tr><tr>"
                out_filename_add = "PVM_set/" + out_filename
                print "<td><img src=\""+out_filename_add+".jpg\" height=\"120\" width=\"120\"><br><p><a href=\""+out_filename_add+".zip\">"+out_filename+"</a></p></td>"
