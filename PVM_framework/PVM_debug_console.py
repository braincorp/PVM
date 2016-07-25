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

import numpy as np
import PVM_framework.SharedArray as SharedArray
import PVM_framework.PVM_Create as PVM_Create
import os
import sys
import argparse
import pprint
import cmd
import traceback
import time
import logging


class InteractiveDictionaryExplorer(cmd.Cmd):
    use_rawinput = False

    def __init__(self,
                 stdin=None,
                 stdout=None,
                 infilename=None,
                 dict=None,
                 pprint=False,
                 sprint=False,
                 nans=None,
                 gt=None,
                 lt=None,
                 abss=None,
                 filter_name=None,
                 upgrade=False):
        """
        Console object used to traverse the simulation dictionary. Can be used standalone on a saved distinary or
        live on a running simulation by logging into the debug port.

        :param stdin:
        :param stdout:
        :param infilename:
        :param dict:
        :param pprint:
        :param sprint:
        :param nans:
        :param gt:
        :param lt:
        :param abss:
        :param filter_name:
        :return:
        """

        if stdin is not None:
            self.stdin = stdin
        else:
            self.stdin=sys.stdin
            self.use_rawinput=True
        if stdout is not None:
            self.stdout = stdout
        else:
            self.stdout=sys.stdout
        cmd.Cmd.__init__(self, stdin=self.stdin, stdout=self.stdout)
        if dict is None and infilename is None:
            sys.stderr.write("dict and infilename cannot be empty at the same time")
        if dict is not None:
            self.dict = dict
            self.filename = str(infilename)
        elif os.path.exists(infilename) and os.path.isfile(infilename):
            # This import below needs to go here as otherwise it would
            # lead to a circular import and failure.
            import PVM_framework.CoreUtils as CoreUtils
            self.dict = CoreUtils.load_model(infilename)
            if upgrade:
                import PVM_framework.PVM_Create as PVM_Create
                PVM_Create.upgrade_dictionary_to_ver1_0(self.dict)
        else:
            sys.stderr.write("Input file not found\n")
        self.pprint = pprint
        self.sprint = sprint
        self.filename = str(infilename)
        self.gt = None
        self.lt = None
        self.nans = None
        self.filter_name = None
        self.abss=None
        if gt is not None:
            self.gt = float(gt)
        if lt is not None:
            self.lt = float(lt)
        if nans is not None:
            self.nans = True
        if filter_name is not None:
            self.filter_name = filter_name
        if abss is not None:
            self.abss = float(abss)
        self.text_bold = '\033[1m'
        self.text_end = '\033[0m'
        self.prompt = self.text_bold + str(self.dict['name'])+"/$ " + self.text_end
        self.current_element = self.dict
        self.pwd = "/"
        logging.info("Created an interactive dictionary explorer session")

    def getsizeof(self, element):
        size=0
        try:
            size = element.nbytes
        except:
            size = sys.getsizeof(element)
        return size

    def getshapeof(self, element):
        shape = "---"
        try:
            shape = str(element.shape)
        except:
            shape = "---"
        return shape

    def gettypename(self, element):
        name = ""
        try:
            name = element.__class__.__name__
        except:
            name = type(element).__name__
        return name

    def emptyline(self):
        pass

    def do_ls(self, line, quiet=False):
        """
        List the contents of the current element in the dictionary
        """
        if line.startswith("/"):
            listed_element = self.getsubelement(self.dict, line)
        else:
            listed_element = self.getsubelement(self.current_element, line)
        if not quiet:
            print >>self.stdout, "%-*s " % (30, "Element name"),
            print >>self.stdout, "%*s " % (20, "type"),
            print >>self.stdout, "%*s " % (12, "shape"),
            print >>self.stdout, "%*s " % (12, "size (bytes)"),
            print >>self.stdout, "%*s " % (15, "id")
            print >>self.stdout, "%-*s " % (30, "."),
            print >>self.stdout, "%*s " % (20, self.gettypename(listed_element)),
            print >>self.stdout, "%*s " % (12, self.getshapeof(listed_element)),
            print >>self.stdout, "%*s " % (12, self.getsizeof(listed_element)),
            print >>self.stdout, "%*s " % (15, id(listed_element))
        if isinstance(listed_element, dict):
            for key in listed_element.keys():
                element = listed_element[key]
                print >>self.stdout, "%-*s " % (30, key),
                print >>self.stdout, "%*s " % (20, self.gettypename(element)),
                print >>self.stdout, "%*s " % (12, self.getshapeof(element)),
                print >>self.stdout, "%*s " % (12, self.getsizeof(element)),
                print >>self.stdout, "%*s " % (15, id(element))
        elif isinstance(listed_element, list):
            for i, element in enumerate(listed_element):
                print >>self.stdout, "%-*s " % (30, str(i)),
                print >>self.stdout, "%*s " % (20, self.gettypename(element)),
                print >>self.stdout, "%*s " % (12, self.getsizeof(element)),
                print >>self.stdout, "%*s " % (12, self.getshapeof(element)),
                print >>self.stdout, "%*s " % (15, id(element))
        elif isinstance(listed_element, tuple):
            for i, element in enumerate(listed_element):
                print >>self.stdout, "%-*s " % (30, str(i)),
                print >>self.stdout, "%*s " % (20, self.gettypename(element)),
                print >>self.stdout, "%*s " % (12, self.getsizeof(element)),
                print >>self.stdout, "%*s " % (12, self.getshapeof(element)),
                print >>self.stdout, "%*s " % (15, id(element))
        else:
            pprint.pprint(listed_element, stream=self.stdout)

    def do_cd(self, path):
        """
        Change the current element in the dictionary
        """
        wd = ""
        current_dir = self.dict
        if path.endswith("/") and len(path)>1:
            path = path[:-1]
        if path.startswith("/"):
            wd = path
        elif path == ".." or path == "../":
            wd = "/"+"/".join(self.pwd.split("/")[:-1])
        elif path == "../.." or path == "../../":
            wd = "/"+"/".join(self.pwd.split("/")[:-2])
        else:
            wd = self.pwd + "/" + path
        subdirs = wd.split("/")
        try:
            for subdir in subdirs:
                if subdir.isdigit():
                    i = int(subdir)
                    current_dir = current_dir[i]
                elif subdir != '':
                    current_dir = current_dir[subdir]
            self.current_element = current_dir
            self.pwd = wd.replace("//", "/")
            self.prompt = self.text_bold + self.dict['name'] + self.pwd + "$ " + self.text_end
        except:
            print >>self.stdout, "Invalid path"

    def getsubelement(self, element, path):
        current_dir = element
        subdirs = path.split("/")
        try:
            for subdir in subdirs:
                if subdir.isdigit():
                    i = int(subdir)
                    current_dir = current_dir[i]
                elif subdir != '':
                    current_dir = current_dir[subdir]
            return current_dir
        except:
            print >>self.stdout, "Invalid path"

    def getsubelements(self, element):
        if isinstance(element, dict):
            return element.keys()
        if isinstance(element, list):
            return map(lambda x: str(x), range(len(element)))
        return []

    def generic_complete(self, text, line, beginidx, endidx):
        if "/" in line:
            (cmd, arg) = line.split(" ")
            if not arg.endswith("/"):
                arg = "/".join(arg.split("/")[:-1])
            element = self.getsubelement(self.current_element, arg)
            prefix = "/".join(text.split("/")[:-1])
        else:
            element = self.current_element
            prefix = ""
        all_completions = map(lambda x: prefix + x, self.getsubelements(element))
        if not text:
            return all_completions
        else:
            return [c for c in all_completions if c.startswith(text)]

    def complete_cd(self, text, line, beginidx, endidx):
        return self.generic_complete(text, line, beginidx, endidx)

    def complete_ls(self, text, line, beginidx, endidx):
        return self.generic_complete(text, line, beginidx, endidx)

    def do_cat(self, line):
        try:
            print >>self.stdout, self.current_element[line]
        except:
            print >>self.stdout, "Cound not access the element"

    def do_debug_create(self, line):
        """
        Creates a debug hookup. Called in a given context needs:
        Nescessary arguments:

                filename='...'
                object=name of the object in the given context
                interval=number of steps between successive dumps

        Optional:

                phase=phase shift of the dump (default 0)

        E.g.:
                debug_create filename='my_debug.p' object=weights0 interval=10 phase=5
        """
        try:
            params = line.split(" ")
            args = {}
            for param in params:
                (L, R) = param.split('=')
                args[L]=R
            if "filename" not in args.keys():
                raise
            if "object" not in args.keys():
                raise
            if "interval" not in args.keys():
                raise
            if 'phase' not in args.keys():
                args['phase'] = 0
        except:
            print >>self.stdout, "Error processing arguments"
            logging.debug("Exception in debug hookup creation, error processing arguments")
        try:
            hookup = {}
            hookup['filename'] = args['filename']
            # Will raise an exception unless the call below succedes
            self.current_element[args['object']].view(np.ndarray)
            hookup['object'] = self.current_element[args['object']]
            hookup['interval'] = int(args['interval'])
            hookup['phase'] = int(args['phase'])
            hookup['path'] = self.pwd + '/' + args['object']
            self.dict['debug_infrastructure']['disabled_hookups'][args['filename']] = hookup
        except:
            print >>self.stdout, "Error making a debug hookup"
            traceback.print_exc(file=self.stdout)
            logging.debug("Exception in debug hookup creation")
        logging.info("Created a debug hookup with parameters " + line)

    def do_debug_disable(self, line):
        """
        Disable a currently enabled debug hookup
        """
        try:
            hookup = self.dict['debug_infrastructure']['enabled_hookups'][line]
            self.dict['debug_infrastructure']['disabled_hookups'][line]=hookup
            del self.dict['debug_infrastructure']['enabled_hookups'][line]
            print >>self.stdout, "Disabled "+line
        except:
            print >>self.stdout, "Error disabling a debug hookup"
            traceback.print_exc(file=self.stdout)
            logging.debug("Exception while disabling a debug hookup " + line)
        logging.info("Disabled a debug hookup " + line)

    def do_debug_enable(self, line):
        """
        Enable a currently disabled debug hookup
        """
        try:
            hookup = self.dict['debug_infrastructure']['disabled_hookups'][line]
            self.dict['debug_infrastructure']['enabled_hookups'][line]=hookup
            del self.dict['debug_infrastructure']['disabled_hookups'][line]
            print >>self.stdout, "Enabled "+line
        except:
            print >>self.stdout, "Error enabling a debug hookup"
            traceback.print_exc(file=self.stdout)
            logging.debug("Exception while enabling a debug hookup " + line)
        logging.info("Enabled a debug hookup " + line)

    def do_debug_delete(self, line):
        """
        Delete an existing enabled or disabled debug hookup
        """
        try:
            del self.dict['debug_infrastructure']['disabled_hookups'][line]
            print >>self.stdout, "Deleted the debug hookup from disabled"
            logging.info("Deleted the debug hookup from disabled")
        except:
            pass
        try:
            del self.dict['debug_infrastructure']['enabled_hookups'][line]
            print >>self.stdout, "Deleted the debug hookup from enabled"
            logging.info("Deleted the debug hookup from enabled")
        except:
            pass

    def do_debug_list_enabled(self, line):
        """
        List existing enabled debug hookups
        """
        self.do_ls("/debug_infrastructure/enabled_hookups", quiet=True)

    def do_debug_list_disabled(self, line):
        """
        List existing disabled debug hookups
        """
        self.do_ls("/debug_infrastructure/disabled_hookups", quiet=True)

    def do_debug_list(self, line):
        """
        List existing debug hookups
        """
        print >>self.stdout, "--Enabled-------------------------------------------------------------------"
        self.do_debug_list_enabled(line)
        print >>self.stdout, "--Disabled------------------------------------------------------------------"
        self.do_debug_list_disabled(line)

    def do_quit(self, line):
        """
        Exit program
        """
        return True

    def list_to_dict(self, something):
        if isinstance(something, dict):
            return something.copy()
        if isinstance(something, list):
            d = {}
            for i in xrange(len(something)):
                d["EL_"+str(i)] = self.list_to_dict(something[i])
            return d

    def do_python(self, line):
        """
        Execute a python command in the current context. Objects appearing in the current context
        will be accesible by their name. One can put import statements etc.
        :param line:
        :return:
        """
        oldout = sys.stdout
        olderr = sys.stderr
        env = self.list_to_dict(self.current_element)
        sys.stdout=self.stdout
        sys.stderr=self.stdout
        try:
            exec(line, env, env)
        except:
            traceback.print_exc(file=self.stdout)
        sys.stdout = oldout
        sys.stderr = olderr

    def help_quit(self):
        print >>self.stdout, "Exit the interactive shell"

    do_EOF = do_quit
    help_EOF = help_quit

    def search_recursive(self, dict_object, path, name, leaf_method):
        if (isinstance(dict_object, np.ndarray) or
           isinstance(dict_object, SharedArray.SharedNumpyArray) or
           isinstance(dict_object, SharedArray.DoubleBufferedSharedNumpyArray)):
                if self.filter_name is not None:
                    if name == self.filter_name:
                        leaf_method(path, name, dict_object)
                else:
                    leaf_method(path, name, dict_object)
                return
        if isinstance(dict_object, dict):
            for key in dict_object.keys():
                self.search_recursive(dict_object[key], path + "/" + str(key), key, leaf_method)
        if isinstance(dict_object, list):
            for i, element in enumerate(dict_object):
                self.search_recursive(element, path + "[" +str(i) + "]", "", leaf_method)

    def print_path(self, path, name, element):
        print >>self.stdout, path

    def check_gt(self, path, name, element):
        if np.max(element) > self.gt:
            print >>self.stdout, path + " has elements greater than " + str(self.gt) + " (" + str(np.max(element)) +")"
            print >>self.stdout, element

    def check_abs(self, path, name, element):
        if np.max(np.fabs(element)) > self.abss:
            print >>self.stdout, path + " has elements with absolute value greater than " + str(self.abss) + " (" + str(np.max(np.fabs(element))) +")"
            print >>self.stdout, element

    def check_lt(self, path, name, element):
        if np.min(element) < self.lt:
            print >>self.stdout, path + " has elements less than " + str(self.lt) + " (" + str(np.min(element)) +")"
            print >>self.stdout, element

    def check_nans(self, path, name, element):
        if np.isnan(element).any():
            print >>self.stdout, path + " contains NANs "
            print >>self.stdout, element

    def check_id(self, path, name, element):
        if int(id(element)) == self.search_identity:
            print >>self.stdout, path

    def run_noninteractive(self):
        if self.pprint:
            pprint.pprint(self.dict, stream=self.stdout)
        if self.sprint:
            self.search_recursive(self.dict, self.filename, "", self.print_path)
        if self.gt is not None:
            self.search_recursive(self.dict, self.filename, "", self.check_gt)
        if self.lt is not None:
            self.search_recursive(self.dict, self.filename, "", self.check_lt)
        if self.nans is not None:
            self.search_recursive(self.dict, self.filename, "", self.check_nans)
        if self.abss is not None:
            self.search_recursive(self.dict, self.filename, "", self.check_abs)

    def do_nanscheck(self, line):
        """
        Find all elements in the current subtree that contain NANs
        """
        self.search_recursive(self.current_element, self.pwd, "", self.check_nans)

    def do_gtcheck(self, line):
        """
        Find all elements in the current subtree that contain values greater than a given value
        Usage:

        gtcheck value

        where value is a numeric value
        """
        try:
            self.gt = float(line)
            self.search_recursive(self.current_element, self.pwd, "", self.check_gt)
        except:
            print >>self.stdout, "Error executing a command"
            self.do_help("gtcheck")

    def do_ltcheck(self, line):
        """
        Find all elements in the current subtree that contain values less than a given value
        Usage:

          ltcheck value

        where value is a numeric value
        """
        try:
            self.lt = float(line)
            self.search_recursive(self.current_element, self.pwd, "", self.check_lt)
        except:
            print >>self.stdout, "Error executing a command"
            self.do_help("ltcheck")

    def do_abscheck(self, line):
        """
        Find all elements in the current subtree that contain values greater in absolute value than a given value.
        Usage:

          abscheck value

        where value is a numeric value
        """
        try:
            self.abss = float(line)
            self.search_recursive(self.current_element, self.pwd, "", self.check_abs)
        except:
            print >>self.stdout, "Error executing a command"
            self.do_help("abscheck")

    def do_findid(self, line):
        """
        Find all the occurrences of an object with a given id.
        Usage:

          findid value

        where value is the numeric id
        """
        try:
            self.search_identity = int(line)
            self.search_recursive(self.dict, self.filename, "", self.check_id)
        except:
            print >>self.stdout, "Error executing a command"
            self.do_help("findid")

    def do_findrefs(self, line):
        """
        Finds all the references to a given object inside of the whole structure.
        Usage:

          findrefs object

        where object is in the current scope.
        """
        try:
            if line.isdigit():
                element = self.current_element[int(line)]
            else:
                element = self.current_element[line]
            self.search_identity = int(id(element))
            self.search_recursive(self.dict, self.filename, "", self.check_id)
        except:
            print >>self.stdout, "Error executing a command"
            self.do_help("findrefs")

    def do_pause(self, line):
        """
        Set the paused flag of a simulation (works when connected to a running simulation)
        """
        self.dict['paused'][0] = PVM_Create.PVM_PAUSE

    def do_step(self, line):
        """
        Stepwise execution. Be careful, works when connected to a runing simulation, otherwise will hang.
        """
        N = self.dict['N'][0]
        self.dict['paused'][0] = PVM_Create.PVM_RESUME
        while self.dict['N'][0] <= (N+1):
            pass
        self.dict['paused'][0] = PVM_Create.PVM_PAUSE

    def do_finish(self, line):
        """
        Set the paused flag of a simulation (works when connected to a running simulation)
        """
        self.dict['finished'][0] = PVM_Create.PVM_FLAG_VAL_TRIGGER
        return True

    def do_toggle_display(self, line):
        """
        Changes the logic for displaying a window (works when connected to a running simulation)
        """
        self.dict['finished'][0] = PVM_Create.PVM_FLAG_TRIGGER_DISPLAY

    def do_record(self, line):
        """
        Start recording to a file given after space, e.g.:

          record myfilename.avi

        """
        if line == "":
            print >>self.stdout, "No file name given"
        else:
            name = np.fromstring(line, dtype=np.uint8)
            self.dict["record_filename"][:] = 0
            self.dict["record_filename"][:name.shape[0]] = name
            self.dict['finished'][0] = PVM_Create.PVM_FLAG_TRIGGER_RECORD
            print >>self.stdout, "Started recording"

    def do_stop_record(self, line):
        """
        Stop recording to a file
        """
        self.dict['finished'][0] = PVM_Create.PVM_FLAG_TRIGGER_RECORD
        print >>self.stdout, "Stopped recording"

    def do_toggle_dream(self, line):
        """
        Sets the flags to inform the manager to run the dream mode in which predicted input is fed back as new input
        """
        if self.dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_RESET:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_DREAM
        else:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_RESET

    def do_toggle_blindspot(self, line):
        """
        Sets the flags to inform the manager to show a gray spot in the center of the image
        """
        if self.dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_RESET:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_BLINDSPOT
        else:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_RESET

    def do_toggle_gray(self, line):
        """
        Set neutral gray as input
        """
        if self.dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_RESET:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_GRAY
        else:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_RESET

    def do_toggle_noise(self, line):
        """
        Set white noise as input
        """
        if self.dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_RESET:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_NOISE
        else:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_RESET

    def do_toggle_noise_spot(self, line):
        """
        Set white noise spot in the center of the image
        """
        if self.dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_RESET:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_NOISE_SPOT
        else:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_RESET

    def do_toggle_inverse_spot(self, line):
        """
        Set inverse blind spot (gray out periphery)
        """
        if self.dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_RESET:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_INV_BLINDSPOT
        else:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_RESET

    def do_toggle_inverse_spot_noise(self, line):
        """
        Set inverse blind spot noise (noise out periphery)
        """
        if self.dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_RESET:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_INV_NOISE_SPOT
        else:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_RESET

    def do_toggle_partial_dream(self, line):
        """
        Set dream mode in the periphery of the visual field
        """
        if self.dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_RESET:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_INV_DREAM_SPOT
        else:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_RESET

    def do_toggle_dream_spot(self, line):
        """
        Set dream mode in the periphery of the visual field
        """
        if self.dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_RESET:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_DREAM_SPOT
        else:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_RESET

    def do_toggle_deep_dream(self, line):
        """
        Make all the units in the hierachy dream
        """
        if self.dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_RESET:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_DEEP_DREAM
        else:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_RESET

    def do_toggle_blinks(self, line):
        """
        Set dream mode in the periphery of the visual field
        """
        if self.dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_RESET:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_BLINKS
        else:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_RESET

    def do_toggle_noisy_signal(self, line):
        """
        Set dream mode in the periphery of the visual field
        """
        if self.dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_RESET:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_NOISY_SIGNAL
        else:
            self.dict['flags'][0] = PVM_Create.PVM_FLAG_VAL_RESET

    def do_disable_lateral(self, line):
        """
        Remove the lateral communication between the units
        """
        self.dict['context_factor_lateral'][0] = 0

    def do_disable_feedback(self, line):
        """
        Remove the feedback communication between the units
        """
        self.dict['context_factor_feedback'][0] = 0

    def do_enable_lateral(self, line):
        """
        Remove the lateral communication between the units
        """
        self.dict['context_factor_lateral'][0] = 1.0

    def do_enable_feedback(self, line):
        """
        Remove the feedback communication between the units
        """
        self.dict['context_factor_feedback'][0] = 1.0

    def do_freeze_learning(self, line):
        """
        Sets the flags to inform the manager to run the dream mode in which predicted input is fed back as new input
        """
        self.dict['flags'][PVM_Create.PVM_LEARNING_FLAG] = PVM_Create.PVM_LEARNING_FREEZE

    def do_unfreeze_learning(self, line):
        """
        Sets the flags to inform the manager to run the dream mode in which predicted input is fed back as new input
        """
        self.dict['flags'][PVM_Create.PVM_LEARNING_FLAG] = PVM_Create.PVM_LEARNING_UNFREEZE

    def do_resume(self, line):
        """
        Set the paused flag of a simulation (works when connected to a running simulation)
        """
        self.dict['paused'][0] = PVM_Create.PVM_RESUME

    def do_dump(self, line):
        """
        Dump the state of the current simulation to a given file. Warning, this method will attempt to pause
        a simulation and wait 1s, but is not guaranteed to save a consistent state. Use only in an emergency.
        """
        self.dict['paused'][0] = PVM_Create.PVM_PAUSE
        time.sleep(1)
        import PVM_framework.CoreUtils as CoreUtils
        CoreUtils.save_model(self.dict, line)
        self.dict['paused'][0] = PVM_Create.PVM_RESUME


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        default='demo.p.gz',
                        nargs=1,
                        help='Input file dictionary')
    parser.add_argument("-p", "--pprint", help="Print the contents of the entire dictionary", action="store_true")
    parser.add_argument("-s", "--sprint", help="Print all the paths in the dictionary", action="store_true")
    parser.add_argument("-n", "--nans", help="Search for NANs", action="store_true")
    parser.add_argument("-u", "--upgrade", help="Upgrade to ver 1", action="store_true")
    parser.add_argument("-g", "--gt", type=str, help="Search for values greater than ")
    parser.add_argument("-l", "--lt", type=str, help="Search for values less than ")
    parser.add_argument("-a", "--abs", type=str, help="Search for values with absol;ute value greater than ")
    parser.add_argument("-f", "--filter", type=str, help="Display only filtered the elements")
    parser.add_argument("-i", "--interactive", help="Start a shell like command line interface to explore the dictionary", action="store_true")
    args = parser.parse_args()
    if not args.input_file:
        parser.print_help()
    else:
        app = InteractiveDictionaryExplorer(infilename=args.input_file[0],
                                            pprint=args.pprint,
                                            sprint=args.sprint,
                                            gt=args.gt,
                                            lt=args.lt,
                                            nans=args.nans,
                                            abss=args.abs,
                                            filter_name=args.filter,
                                            upgrade=args.upgrade)
        if args.interactive:
            app.cmdloop()
        else:
            app.run_noninteractive()
