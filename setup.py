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
from setuptools import setup, find_packages
from distutils.core import Extension
import os
from Cython.Build import cythonize
from Cython.Compiler.Main import default_options
import numpy
import platform


def ver_to_float(ver):
    nums = ver.split('.')
    result = 0.0
    base = 1.0
    for num in nums:
        result += float(num)*base
        base *= 0.001
    return result

default_options['emit_linenums'] = True
gcc_ver = os.popen("gcc -dumpversion").read()
SSE_FLAGS = []
if platform.machine() == 'x86_64':
    SSE_FLAGS.append("-m64")
    SSE_FLAGS.append("-msse")
    SSE_FLAGS.append("-msse2")
    SSE_FLAGS.append("-msse3")
    SSE_FLAGS.append("-mfpmath=sse")

Extensions = [Extension("PVM_framework/fast_routines",
                        sources=["PVM_framework/fast_routines.pyx", "PVM_framework/Accelerated.cpp"],
                        language="c++",
                        include_path=['.', numpy.get_include()],
                        extra_compile_args=["-O3"]+SSE_FLAGS)]

if (ver_to_float(gcc_ver) < ver_to_float("4.7")):
    print "gcc version > 4.7 is nescessary for some extensions"
else:
    Extensions.append(Extension("SyncUtils",
                                sources=["PVM_framework/SyncUtils.pyx", "PVM_framework/Sync.cpp"],
                                language="c++",
                                include_path=['.', numpy.get_include()],
                                extra_compile_args=["-O3"]+SSE_FLAGS))

if platform.machine() == 'x86_64' or platform.machine() == 'i386':
    Extensions.append(Extension("LowLevelCPUControlx86",
                                sources=["PVM_framework/LowLevelCPUControlx86.pyx"],
                                language="c++",
                                include_path=['.', numpy.get_include()],
                                extra_compile_args=["-O3"] + SSE_FLAGS))


setup(
    ext_package="PVM_framework",
    ext_modules=cythonize(Extensions),
    include_dirs=[numpy.get_include()],
    name='PVM_framework',
    author='Brain Corporation',
    author_email='piekniewski@braincorporation.com',
    url='https://github.com/braincorp/PVM',
    long_description='',
    version='1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[])
