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


this_dir = os.path.dirname(os.path.realpath(__file__))
cv_folder = [l[l.find('/'):] for l in cv2.getBuildInformation().splitlines() if 'Install path' in l][0]
opentld_python_root = this_dir
opentld_cpp_root = this_dir + "/original_opentld/src"
opentld_source_folders = [opentld_cpp_root + '/libopentld/tld', opentld_cpp_root + '/libopentld/mftracker', opentld_cpp_root + '/3rdparty/cvblobs']

opentld_sources = []
for sf in opentld_source_folders:
    opentld_sources += [os.path.join(sf, i) for i in os.listdir(sf) if i.endswith('cpp')]

opentld_module = Extension('tld',
                           include_dirs=["/usr/include",
                                         "/usr/local/include",
                                         "/usr/include/boost",
                                         "/usr/include/opencv",
                                         "/usr/local/include/opencv",
                                         cv_folder + "/include", cv_folder + "/include/opencv",
                                         cv_folder + "/include/opencv2"] + opentld_source_folders,
                           libraries=['boost_python', 'opencv_core',
                                      'opencv_imgproc', 'opencv_video'],
                           library_dirs=['/usr/local/lib', '/usr/lib', cv_folder + '/lib'],
                           runtime_library_dirs=[cv_folder + '/lib'],
                           sources=[opentld_python_root + '/opentld_python.cpp'] + opentld_sources)


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
    ext_modules=[opentld_module])
