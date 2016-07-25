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

import cPickle
import numpy as np


class DebugLogger(object):
    """
    Logs data from the simulation going through debug hookups in the dictionary.

    Each hookup is a structure consisting of:
        filename - string
        object - object being dumped
        interval - integer every how many steps log th data
        phase - integer (< than interval) depicting phase of when to dump the data
        path - path of the element being dumped
    """
    def __init__(self, dict, buffer_len=100):
        """

        :param dict: simulation dictionary
        :type dict: dict
        :param buffer_len: length of the memory buffer (default 100)
        :type buffer_len: int
        :return: No value
        """
        self.dict = dict
        self.files = {}
        self.buffers = {}
        self.buffer_len = buffer_len

    def process_hookups(self):
        """
        This is called on every step of the simulation when the data is consistent.
        :return: No value
        """
        current_hookups = []
        for hookhash in self.dict['debug_infrastructure']['enabled_hookups'].keys():
            hookup = self.dict['debug_infrastructure']['enabled_hookups'][hookhash]
            current_hookups.append(hookhash)
            if hookhash not in self.files:
                file = open(hookhash, "wb")
                self.files[hookhash] = file
            if hookhash not in self.buffers:
                self.buffers[hookhash] = []
            if self.dict['N'][0] % hookup['interval'] == hookup['phase']:
                self.buffers[hookhash].append(hookup['object'].view(np.ndarray).copy())
                if len(self.buffers[hookhash])>=self.buffer_len:
                    self.flush(hookhash)
        for name in self.files.keys():
            if name not in current_hookups:
                self.flush(name)
                self.files[name].close()
                del self.files[name]
                del self.buffers[name]

    def flush(self, hookhash):
        """
        Empty the buffer and save data o disk
        :param hookhash:
        :return: No value
        """
        for object in self.buffers[hookhash]:
            cPickle.dump(object.view(np.ndarray), self.files[hookhash], protocol=-1)
        self.buffers[hookhash] = []

    def flushall(self):
        """
        Flush all active buffers
        :return: No value
        """
        for hookhash in self.dict['debug_infrastructure']['enabled_hookups'].keys():
            self.flush(hookhash)

    def __del__(self):
        self.flushall()
        for file in self.files.items():
            file.close()
