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


import PVM_framework.PVM_Storage as PVM_Storage

import logging

sets = {
    "face_training": [["PVM_data/face01.pkl", "default"],
                      ["PVM_data/face03.pkl", "default"],
                      ["PVM_data/no_target.pkl", "default"],
                      ["PVM_data/face16.pkl", "default"],
                      ["PVM_data/face17.pkl", "default"],
                      ["PVM_data/face18.pkl", "default"],
                      ["PVM_data/face19.pkl", "default"],
                      ["PVM_data/no_target_01.pkl", "default"],
                      ["PVM_data/face20.pkl", "default"],
                      ["PVM_data/face21.pkl", "default"],
                      ],
    "face_testing": [["PVM_data/face02.pkl", "default"],
                     ["PVM_data/face04.pkl", "default"],
                     ["PVM_data/face05.pkl", "default"],
                     ["PVM_data/face06.pkl", "default"],
                     ["PVM_data/face07.pkl", "default"],
                     ["PVM_data/face08.pkl", "default"],
                     ["PVM_data/face09.pkl", "default"],
                     ["PVM_data/face10.pkl", "default"],
                     ["PVM_data/face11.pkl", "default"],
                     ["PVM_data/face12.pkl", "default"],
                     ["PVM_data/face13.pkl", "default"],
                     ["PVM_data/face14.pkl", "default"],
                     ["PVM_data/face15.pkl", "default"],
                     ["PVM_data/face22.pkl", "default"],
                     ["PVM_data/face23.pkl", "default"],
                     ["PVM_data/face24.pkl", "default"],
                     ],
    "face_ex_testing": [["PVM_data/face25.pkl", "default"],
                        ["PVM_data/face26.pkl", "default"],
                        ["PVM_data/face27.pkl", "default"],
                        ["PVM_data/face28.pkl", "default"],
                        ["PVM_data/face29.pkl", "default"],
                        ["PVM_data/face30.pkl", "default"],
                        ["PVM_data/face31.pkl", "default"],
                        ["PVM_data/face32.pkl", "default"],
                        ["PVM_data/face33.pkl", "default"],
                        ],
    "face_additional": [],
    "green_ball_training": [["PVM_data/green_ball_long.pkl", "default"],
                            ["PVM_data/green_ball_on_grass.pkl", "default"],
                            ["PVM_data/green_ball_test_14.pkl", "default"],
                            ["PVM_data/green_ball_test_15.pkl", "default"],
                            ["PVM_data/green_ball_test_16.pkl", "default"],
                            ],
    "green_ball_testing": [["PVM_data/green_ball_test.pkl", "default"],
                           ["PVM_data/green_ball_test_01.pkl", "default"],
                           ["PVM_data/green_ball_test_02.pkl", "default"],
                           ["PVM_data/green_ball_test_03.pkl", "default"],
                           ["PVM_data/green_ball_test_04.pkl", "default"],
                           ["PVM_data/green_ball_test_05.pkl", "default"],
                           ["PVM_data/green_ball_test_06.pkl", "default"],
                           ["PVM_data/green_ball_test_07.pkl", "default"],
                           ["PVM_data/green_ball_test_08.pkl", "default"],
                           ["PVM_data/green_ball_test_09.pkl", "default"],
                           ["PVM_data/green_ball_test_10.pkl", "default"],
                           ["PVM_data/green_ball_test_11.pkl", "default"],
                           ["PVM_data/green_ball_test_12.pkl", "default"],
                           ["PVM_data/green_ball_test_13.pkl", "default"],
                           ["PVM_data/green_ball_01_small.pkl", "default"],
                           ["PVM_data/green_ball_bc_office.pkl", "default"],
                           ],
    "green_ball_ex_testing": [["PVM_data/green_ball_test_17.pkl", "default"],
                              ["PVM_data/green_ball_test_18.pkl", "default"],
                              ["PVM_data/green_ball_test_19.pkl", "default"],
                              ["PVM_data/green_ball_test_20.pkl", "default"],
                              ["PVM_data/green_ball_test_21.pkl", "default"],
                              ["PVM_data/green_ball_test_22.pkl", "default"],
                              ["PVM_data/green_ball_test_23.pkl", "default"],
                              ["PVM_data/green_ball_test_24.pkl", "default"],
                              ["PVM_data/green_ball_test_25.pkl", "default"],
                              ["PVM_data/green_ball_test_26.pkl", "default"],
                              ["PVM_data/green_ball_test_27.pkl", "default"],
                              ["PVM_data/green_ball_test_28.pkl", "default"],
                              ["PVM_data/green_ball_test_29.pkl", "default"],
                              ],
    "green_ball_additional": [["PVM_data/blue_ball_on_grass_daytime.pkl", "default"],
                              ["PVM_data/blue_ball_at_home_02.pkl", "default"],
                              ["PVM_data/blue_ball_at_home_01.pkl", "default"],
                              ],
    "stop_sign_training": [["PVM_data/stop01.pkl", "default"],
                           ["PVM_data/stop03.pkl", "default"],
                           ["PVM_data/stop05.pkl", "default"],
                           ["PVM_data/stop07.pkl", "default"],
                           ["PVM_data/stop09.pkl", "default"],
                           ["PVM_data/stop11.pkl", "default"],
                           ["PVM_data/stop13.pkl", "default"],
                           ["PVM_data/stop15.pkl", "default"],
                           ["PVM_data/stop17.pkl", "default"],
                           ["PVM_data/stop19.pkl", "default"],
                           ["PVM_data/stop21.pkl", "default"],
                           ["PVM_data/stop23.pkl", "default"],
                           ["PVM_data/stop25.pkl", "default"],
                           ["PVM_data/stop27.pkl", "default"],
                           ["PVM_data/stop29.pkl", "default"],
                           ["PVM_data/stop32.pkl", "default"],
                           ["PVM_data/stop34.pkl", "default"],
                           ["PVM_data/stop36.pkl", "default"],
                           ["PVM_data/stop38.pkl", "default"],
                           ["PVM_data/stop40.pkl", "default"],
                           ],
    "stop_sign_testing": [["PVM_data/stop02.pkl", "default"],
                          ["PVM_data/stop04.pkl", "default"],
                          ["PVM_data/stop06.pkl", "default"],
                          ["PVM_data/stop08.pkl", "default"],
                          ["PVM_data/stop10.pkl", "default"],
                          ["PVM_data/stop12.pkl", "default"],
                          ["PVM_data/stop14.pkl", "default"],
                          ["PVM_data/stop16.pkl", "default"],
                          ["PVM_data/stop18.pkl", "default"],
                          ["PVM_data/stop20.pkl", "default"],
                          ["PVM_data/stop22.pkl", "default"],
                          ["PVM_data/stop24.pkl", "default"],
                          ["PVM_data/stop26.pkl", "default"],
                          ["PVM_data/stop28.pkl", "default"],
                          ["PVM_data/stop30.pkl", "default"],
                          ["PVM_data/stop33.pkl", "default"],
                          ["PVM_data/stop35.pkl", "default"],
                          ["PVM_data/stop37.pkl", "default"],
                          ["PVM_data/stop39.pkl", "default"],
                          ],
    "stop_sign_additional": [],
    "stop_sign_ex_testing": [["PVM_data/stop41.pkl", "default"],
                             ["PVM_data/stop42.pkl", "default"],
                             ["PVM_data/stop43.pkl", "default"],
                             ["PVM_data/stop44.pkl", "default"],
                             ["PVM_data/stop45.pkl", "default"],
                             ["PVM_data/stop46.pkl", "default"],
                             ["PVM_data/stop47.pkl", "default"],
                             ["PVM_data/stop48.pkl", "default"],
                             ["PVM_data/stop49.pkl", "default"],
                             ["PVM_data/stop50.pkl", "default"],
                             ["PVM_data/stop51.pkl", "default"],
                             ],
    "short_training": [["PVM_data/stop40.pkl", "default"]],
    "short_testing": [["PVM_data/stop41.pkl", "default"]],
    "short_additional": [],

    "non_spec_training": [["PVM_data/no_target.pkl", "default"],
                          ["PVM_data/no_target_01.pkl", "default"],
                          ["PVM_data/green_ball_long.pkl", "default"],
                          ["PVM_data/stop01.pkl", "default"],
                          ["PVM_data/green_ball_on_grass.pkl", "default"],
                          ["PVM_data/stop03.pkl", "default"],
                          ["PVM_data/face01.pkl", "default"],
                          ["PVM_data/stop05.pkl", "default"],
                          ["PVM_data/face03.pkl", "default"],
                          ["PVM_data/stop07.pkl", "default"],
                          ["PVM_data/face16.pkl", "default"],
                          ["PVM_data/stop09.pkl", "default"],
                          ["PVM_data/face17.pkl", "default"],
                          ["PVM_data/stop11.pkl", "default"],
                          ["PVM_data/face18.pkl", "default"],
                          ["PVM_data/stop13.pkl", "default"],
                          ["PVM_data/green_ball_test_14.pkl", "default"],
                          ["PVM_data/stop15.pkl", "default"],
                          ["PVM_data/green_ball_test_15.pkl", "default"],
                          ["PVM_data/stop17.pkl", "default"],
                          ["PVM_data/green_ball_test_16.pkl", "default"],
                          ["PVM_data/stop19.pkl", "default"],
                          ["PVM_data/stop21.pkl", "default"],
                          ["PVM_data/stop23.pkl", "default"],
                          ],
    "non_spec_testing": [["PVM_data/stop41.pkl", "default"]],
    "non_spec_additional": [],
}


sets["stop_sign_full_testing"] = sets["stop_sign_testing"] + sets["stop_sign_ex_testing"]
sets["green_ball_full_testing"] = sets["green_ball_testing"] + sets["green_ball_ex_testing"]
sets["face_full_testing"] = sets["face_testing"] + sets["face_ex_testing"]

sets["stop_sign_fast_testing"] = [sets["stop_sign_testing"][0]]
sets["green_ball_fast_testing"] = [sets["green_ball_testing"][0]]
sets["face_fast_testing"] = [sets["face_testing"][0]]


class PVMDataset(object):
    def __init__(self, name, storage=None):
        logging.info("Parsing dataset info on %s" % name)
        self.name = name
        self.all = []
        if name+"_training" in sets.keys():  # ok so we have a good dataset name
            self.training = sets[name+"_training"]
            self.testing = sets[name+"_testing"]
            self.additional = sets[name+"_additional"]
            logging.info("Dataset appears to be defined withing datasets")
        elif name == "all":
            self.training = []
            self.testing = []
            self.additional = []
            for ds in ("stop_sign", "green_ball", "face"):
                self.training.extend(sets["%s_training" % ds])
                self.testing.extend(sets["%s_testing" % ds])
                self.testing.extend(sets["%s_ex_testing" % ds])
                self.additional.extend(sets["%s_additional" % ds])
            logging.info("Dataset is a special keyword")
        elif name == "all_short":
            self.training = []
            self.testing = []
            self.additional = []
            for ds in ("stop_sign", "green_ball", "face"):
                self.training.append(sets["%s_training" % ds][0])
                self.testing.append(sets["%s_testing" % ds][0])
                self.additinal.append(sets["%s_testing" % ds][0])
            logging.info("Dataset is a special keyword")
        else:  # it does not match anything but maybe it is just a filename directly
            logging.info("Dataset is unknown, trying a file")
            try:
                logging.info("Trying to download the file %s" % name)
                filename = storage.get(name)
                logging.info("got file %s" % filename)
            except:
                logging.error("Invalid dataset")
                raise(Exception("Invalid dataset"))
            if filename is not None:
                # the file is valid
                logging.info("Got the file")
                self.training = [[name, "default"]]
                self.testing = [[name, "default"]]
                self.additional = [[name, "default"]]
            else:
                logging.error("Invalid dataset")
                raise(Exception("Invalid dataset"))
        self.all.extend(self.training)
        self.all.extend(self.testing)
        self.all.extend(self.additional)


if __name__ == "__main__":
    import PVM_tools.labeled_movie as lm
    TS = PVM_Storage.Storage()
    for set in sets.keys():
        print "Set :" + str(set)
        set_l = 0
        for element in sets[set]:
            local_path = TS.get(element[0])
            fc = lm.FrameCollection()
            fc.load_from_file(local_path)
            fc.write_to_file(local_path)
            print "File %s, %d frames %f minutes" % (local_path, len(fc), len(fc)/(25.0*60))
            set_l += len(fc)
        print "Set %s, %f frames, %f minutes" % (set, set_l, set_l/(25.0*60))
