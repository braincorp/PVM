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

from setuptools import find_packages
from distutils.core import Extension, setup
import os
import cv2
from Cython.Build import cythonize
import numpy


this_dir = os.path.dirname(os.path.realpath(__file__))
cv_folder = [l[l.find('/'):] for l in cv2.getBuildInformation().splitlines() if 'Install path' in l][0]

# struck tracker

struck_src_dir = this_dir + "/original_struck/src/"
struck_src = ["Tracker.cpp",
              "Config.cpp",
              "Features.cpp",
              "HaarFeature.cpp",
              "HaarFeatures.cpp",
              "HistogramFeatures.cpp",
              "ImageRep.cpp",
              "LaRank.cpp",
              "MultiFeatures.cpp",
              "RawFeatures.cpp",
              "Sampler.cpp",
              "GraphUtils/GraphUtils.cpp"]

struck_all_src = [struck_src_dir+src for src in struck_src]

struck_module = Extension("struck_bindings",
                          sources=["./struck.cpp",
                                   "./struck_bindings.pyx"] + struck_all_src,
                          language="c++",
                          include_dirs=['.', numpy.get_include(), '/usr/include/eigen3/', "/usr/include/"],
                          libraries=['opencv_core', 'opencv_highgui'],
                          extra_compile_args=["-O2", "-m64"])

struck_module = cythonize([struck_module])[0]

setup(
    name='Other tracker bindings',
    author='Brain Corporation',
    author_email='piekniewski@braincorporation.com',
    url='https://github.com/braincorp/PVM',
    long_description='',
    version='dev',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    ext_modules=[struck_module])
