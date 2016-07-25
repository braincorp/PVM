"""
This small tool will go over all the pkl files stores in the ts amazon bucket,
download each one of them, load and upgrade the movie version to the latest,
save the file and upload back to amazon bucket.

Use this when something changes about the data format.

Usage:
::

    python upgrade_cloud_lib.py -u -f

Options:

    * -f force download to make sure all the files are actually those stored in S3
    * -u upload files, otherwise will run dry, that is upgrade files only locally

When running first time after some major changes in the format I suggest to skip -u option
for the first run, to make sure all the movies convert without issues.

"""

import PVM_tools.labeled_movie as lm
import PVM_framework.PVM_datasets as PVM_datasets
import PVM_framework.PVM_Storage as PVM_Storage
import argparse
import cv2
import os
import logging


if __name__ == "__main__":
    doc="""
    This small tool will go over all the pkl files stores in the ts amazon bucket,
    download each one of them, load and upgrade the movie version to the latest,
    save the file and upload back to amazon bucket.

    Use this when something changes about the data format.
    """
    logging.basicConfig(filename="upgrade.log", level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(thread)d PVM_run : %(message)s ')
    logging.getLogger().addHandler(logging.StreamHandler())
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("-u", "--upload", help="Actually upload the movies", action="store_true")
    parser.add_argument("-f", "--force", help="Force download", action="store_true")
    dsize = (480, 270)
    args = parser.parse_args()
    storage = PVM_Storage.Storage()
    dataset = PVM_datasets.PVMDataset(name="all")
    for [f, c] in dataset.all:
        print "Processing file %s" % f
        local_path = storage.get(f)
        # continue
        fc = lm.FrameCollection()
        fc_scaled = lm.FrameCollection()
        fc.load_from_file(local_path)
        for i in range(len(fc)):
            frame = fc.Frame(i)
            image = frame.get_image(channel=c)
            label = frame.get_label(channel=c)
            resized = cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_CUBIC)
            label.scale_to_new_image_shape(new_image_shape=resized.shape, old_image_shape=image.shape)
            new_frame = lm.LabeledMovieFrame(internal_storage_method="jpg", compression_level=75)
            new_frame.set_image(resized, channel=c)
            new_frame.set_label(label=label, channel=c)
            fc_scaled.append(Frame=new_frame)
        local_saved = os.path.join("/tmp", os.path.basename(local_path))
        remote_saved = os.path.join("PVM_data/", os.path.basename(local_path))
        fc_scaled.write_to_file(local_saved)
        storage.put(path=remote_saved, from_path=local_saved)
