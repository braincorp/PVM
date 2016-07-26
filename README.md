# Predictive Vision Model 

## Introduction
This repository contains necessary files and demos used for running multicore simulations for the Predictive Vision Model paper entitled *"Unsupervised Learning from Continuous Video in a Scalable Predictive Recurrent Network"* (Piekniewski et al., 2016 (https://arxiv.org/abs/1607.06854)).

The code includes a framework that can run a large number of python objects in parallel, synchronized by global barriers, utilizing state kept in shared memory.
## Starting up

For the quickest setup get a clean Ubuntu 16.04 machine with as many compute cores as you can get (should also work fine
on 15.10, 15.04 and 14.04 but 16.04 is tested). Next clone
the repo and run the following commands:

```
git clone git@github.com:braincorp/PVM
cd PVM
sudo ./install_ubuntu_dependecies.sh
```
This script will install all the necessary packages like opencv, numpy, gcc, cython etc. This may take a while
depending on your internet connection. Once this is done, you have to compile a few things and initialize github
modules. To do that, run:
```
source install_local.sh
```

This among other things will compile Cython/Boost bindings. Make sure there are no error messages before you proceed.

The script above will also update the PYTHONPATH variable so that you can now run the content from the current terminal
window. Note that if you open another window you will have to set the PYTHONPATH.

[Documentation](http://pvm.braincorporation.net/docs/index.html) for the project is available online.

### Structure of the repo

There are several packages available in the repo.  Since this code is meant for experimentation, they are not very
strictly separated:

 * PVM_framework - classes and functions necessary to instantiate and run multicore simulations
 * PVM_models - demos and PVM run scripts, where you will spend most of the time initially
 * PVM_tools - classes and functions related to labeling and storing labeled movies, tracker benchmark functions
 * other_trackers - several git submodules and python binding for state of the art trackers. Included are CMT, OpenTLD, STRUCK, a simple color tracker and stationary control trackers
 * tracker_tools - additional scripts for importing/exporting/labeling/playing the tracking datasets
 * docs - is where you can generate the documentation using sphinx (just run the ```generate.sh``` script)

## Download data

In order to run some of the demos and models you will need data. Please download the following files and unzip
the contents into the HOME directory:
```
cd
wget http://pvm.braincorporation.net/PVM_data_sequences.zip
unzip PVM_data_sequences.zip
rm PVM_data_sequences.zip
```
If you are interested in replicating the results from Piekniewski et al., 2016 (https://arxiv.org/abs/1607.06854), you may want to download a few pre-trained models:
```
cd
wget http://pvm.braincorporation.net/PVM_data_models01.zip
unzip PVM_data_models01.zip
rm PVM_data_models01.zip
```
You can also download the data and the models automatically by running the following script and answering 'yes' to each prompt:

```
./download_data.sh
```
Note that the files range from 1.5GB to 3.5GB totalling up to 7GB of drive space necessary. Make sure to have enough
space available before you proceed.

The above commands will create a directory subtree PVM_data in your home directory. PVM will load and save
a bunch of data in that structure. In addition by configuring the PVM_Storage.py and setting Amazon S3 credentials
in ~/.aws/credentials file, PVM can use an S3 bucket to mirror the data directory. This is useful when running
simulations on a cluster etc.

## Basic demos
In order to familiarize yourself with this framework you may want to run/read several small demos. Running them is not necessary for reproducing the results of the paper, but could be fun on its own. If you are more interested with the PVM itself, it is safe to skip to the next section. Otherwise go to PVM_models directory and run:
```
python demo00_run.py
```
A window full of flickering black and white pixels should appear. Depending on you machine speed the pixels will update
randomly. Check your CPU usage with top or activity monitor. It should be high. Next run:
```
python demo01_run.py
```
This is a bit more interesting as this demo will simulate the Ising model at critical temperature. You may decide to look
at it for a while, as interesting structures may emerge. [ESC] will quit.

Demos 2,3,4 require a camera or an input movie. These demos are more along the lines of what we will see with the full PVM simulation (however we provide the data files for PVM). Unless a movie is given through -f parameter the demos will try to open a camera. 
```
python demo02_run.py -f my_movie_file.avi
python demo03_run.py -f my_movie_file.avi
python demo04_run.py -f my_movie_file.avi
```
where ```my_movie_file.avi``` is any movie file which you have to supply. Demo 2 is a tile of small predictive
encoders trying to predict a movie frame based on the previous two frames. In Demo 3, an image is being predicted by a set of
predictive encoders that look at different aspects of the input. One set of these "future encoders" digests two 8x8
frames to predict the next one, another set taking 8 4x4 frames to predict 4 subsequent frames and another set taking
32 2x2 frames to predict 16 subsequent frames. Additionally there is one unit that takes the whole image as
8x8 block whose internal representations are shared as context with all the other units.
The system has feedback connections, those units with larger temporal buffers feed back to those with more spatial information. Also a cross-like
neighbourhood of lateral projections is instantiated. 

In Demo 4, a future/predictive encoder is instantiated
that predicts a camera image based on two previous frames. Here, the encoder predicts the signal and its own error for that
signal (i.e., an additional set of units tries to predict the magnitude of error between the prediction and the signal).
In addition the hidden layer activations from the previous step of execution are used as as context input block.
Second order error is calculated as the error of the error prediction. Also, the learning rate of the primary signal
is modulated by the magnitude of the second order error.

## The PVM model

The PVM is a larger project that to some extent resembles Demos 2-4, but is much more complex. It constructs
a hierarchy of predictive encoders as described in the paper *[Unsupervised Learning from Continuous Video in a Scalable Predictive Recurrent Network]* (https://arxiv.org/abs/1607.06854). The model may be evaluated on online visual tracking task. The descriptions of meta-parameters of the models that can be simulated are available in
```PVM_models/model_zoo```. Feel free to modify these settings, but be aware that simulating some of these models to convergence
may take days if not weeks (at least until a GPU version is implemented, which is not yet the case). For your convenience
several pre-trained models are available.

### PVM data storage
In general all training data and models for PVM are stored in the ```~/PVM_data``` directory. If you have amazon credentials
and an S3 bucket then ```~/PVM_data``` can be mirrored to that bucket. This is useful when running models on the Amazon Web Services EC2 (Elastic Compute Cloud) on a larger scale.

### Running PVM
Once the data and code is in place you may run one of the already pre-trained instances e.g.:

```
cd PVM/PVM_models
python PVM_run.py -D -c 10 -r PVM_models/2016_01_08_18_54_44_stop_simple____dd25d826/PVM_failsafe_0095000000.p.gz -s stop_sign
```
This will load and run the simulation stored in "PVM_models/2016_01_08_18_54_44_stop_simple____dd25d826/PVM_failsafe_0095000000.p.gz"
utilizing 10 compute cores (-c 10), on the stop sign data set training files (-s) and will display a live preview window
(-D). Live preview is good for tinkering but may slow down the execution. ESC pressed on the window will finish the execution.

### Debug interface
When PVM_run.py executes it will typically print out several messages, one of which may look like this:

```
Listening on port 9000 for debug connections
```

This indicates a port on which a debug server is listening. Debug shell is a powerful tool that allows for
various experiments on a running model. Invoke the shell by logging into the port, e.g. using netcat:

```
nc localhost 9000
```
Now you are in the debug shell. Type "help" for list of possible commands and "help command" for help on particular
command. Using debug shell you can pause and step simulation, freeze and unfreeze learning, modify parameters and
even execute full blown python commands. The simulation is presented in the shell as a file system that you can
traverse just like an ordinary Unix filesystem. Some elements are leafs of the dictionary (which would correspond to files).
You can display their contents via "cat". Others that are subdictionaries or sublists correspond to dictionaries
that you can traverse with "cd". 

Importantly, while connected to the debug console, you can enable/disable the window preview of the running simulation via:
```
toggle_display
```
This window allows you to see what is going on, but it will slow down the simulation a bit, therefore we recommend disabling it for long bouts of training. 


#### Leftover processes
This code is just for experiments and not for production so things may sometimes go wrong and the simulation either
crashes or you have to Ctrl+C it. In such situation some of the
compute threads may keep running on the CPU in a busy loop and substantially decrease the performance of your machine. It is
recommended that you verify what is running every once in a while with the top command. To kill processes selectively
you may use commands like this:
```
ps -ef | grep PVM_run | cut -f 2 -d " " | xargs kill
```
This will kill every process that contains ```PVM_run``` in its command line. By tweaking the grep string you may be more
or less selective about what you will kill. Be careful, since this command may accidentally kill the processes you did not intend to kill! Make sure the grep selection sieves out only those proceeses that you actually need removed!

### Tracker benchmark

The PVM package comes with infrastructure for evaluating tracking performance. The provided data files are divided into
essentially three groups (check PVM_datasets.py for details):

* Training data (selected by adding suffix ```"_training"``` to dataset name)
* Testing data (selected by adding suffix ```"_testing"``` to dataset name)
* Extended testing data (selected by adding suffix ```"_ex_testing"``` to dataset name)

The selection is arbitrary and you are free to modify those assignments. In addition both testing sets can ebe combined
by selecting suffix ```"_full_testing"```.

You can run benchmark by invoking the command in ```PVM_models```, e.g:
```
python run_tracking_benchmark.py -T0 -T1 -T2 -s stop_sign_full -r PVM_models/
2016_01_08_18_54_44_stop_simple____dd25d826/PVM_failsafe_0095000000.p.gz -c 5 -S 2 -e -p test
```
This will run the PVM tracker (-T0) along with null and center trackers (-T1, -T2) on ```stop_sign_full``` set, the PVM tracker
will run two steps on every frame to allow its dynamics to settle a bit (-S 2), the results will be saved in ```~/benchmark_results/test[...]```.
The -e flag is required to actually **execute** the benchmark, otherwise the script will try to recompute the results saved from
previous runs. The full set of options is available by running the script without any parameters, i.e.: 

```
python run_tracking_benchmark.py
```
### Reproducing paper results

The paper *"Unsupervised Learning from Continuous Video in a Scalable Predictive Recurrent Network"* (Piekniewski et al., 2016)(https://arxiv.org/abs/1607.06854) contains three experiments. The software included here allows for reproduction of two of them. The third one involves an integration with virtual world (Unreal 4.9), which we are currently not releasing, but may release in the future. 

#### Experiment 1 

Experiment 1 can be reproduced from scratch (i.e., training) or from a set of pre-trained models that we release along with the code. To reproduce from scratch run the following commands:
```
python PVM_run.py -B -S model_zoo/experiment1_green_ball.json
python PVM_run.py -B -S model_zoo/experiment1_stop_sign.json
python PVM_run.py -B -S model_zoo/experiment1_face.json
```
We recommend to run each of these commands on a separate, powerful machine. Now all you need to do is wait. Depending on the available CPU resources, it will take 10h-30h to simulate 1M steps. To reproduce the data presented in the paper you need to run for 95M steps, which might be a while (although you should get reasonable performance after 20M though). As the simulation proceeds it will write snapshots every 100k steps, so make sure there is enough drive space available as well. Each snapshot may take 200MB. After this is completed you may directly run the benchmark on these snapshots as here:
```
python run_tracking_benchmark.py -T0 -T1 -T2 -s stop_sign_full -r PVM_models/
PATH_TO_MY_SNAPSHOT/PVM_failsafe_0095000000.p.gz -c 5 -S 4 -e -p test
```
If you dont want to wait several months of training then you can use the pre trained models we ship with the code. Download and unzip the file ```PVM_data_models01.zip```. The file contains 9 files:
```
drwxrwxr-x  3.0 unx        0 bx stor 16-Jun-23 10:49 PVM_data/PVM_models/2016_01_08_18_56_56_face_simple____47aaabd4/
-rw-rw-r--  3.0 unx 187379490 bx defN 16-Jun-23 10:48 PVM_data/PVM_models/2016_01_08_18_56_56_face_simple____47aaabd4/PVM_failsafe_0040000000.p.gz
-rw-rw-r--  3.0 unx 189374856 bx defN 16-Jun-23 10:45 PVM_data/PVM_models/2016_01_08_18_56_56_face_simple____47aaabd4/PVM_failsafe_0095000000.p.gz
-rw-rw-r--  3.0 unx 186499761 bx defN 16-Jun-23 10:49 PVM_data/PVM_models/2016_01_08_18_56_56_face_simple____47aaabd4/PVM_failsafe_0020000000.p.gz
drwxrwxr-x  3.0 unx        0 bx stor 16-Jun-23 10:19 PVM_data/PVM_models/2016_01_08_18_54_44_stop_simple____dd25d826/
-rw-rw-r--  3.0 unx 186445343 bx defN 16-Jun-23 10:44 PVM_data/PVM_models/2016_01_08_18_54_44_stop_simple____dd25d826/PVM_failsafe_0020000000.p.gz
-rw-rw-r--  3.0 unx 187455605 bx defN 16-Jun-23 10:43 PVM_data/PVM_models/2016_01_08_18_54_44_stop_simple____dd25d826/PVM_failsafe_0040000000.p.gz
-rw-rw-r--  3.0 unx 189399162 bx defN 16-Jun-23 10:40 PVM_data/PVM_models/2016_01_08_18_54_44_stop_simple____dd25d826/PVM_failsafe_0095000000.p.gz
drwxrwxr-x  3.0 unx        0 bx stor 16-Jun-23 10:13 PVM_data/PVM_models/2016_01_08_19_01_50_green_b_simple_d547417c/
-rw-rw-r--  3.0 unx 186473825 bx defN 16-Jun-23 10:39 PVM_data/PVM_models/2016_01_08_19_01_50_green_b_simple_d547417c/PVM_failsafe_0020000000.p.gz
-rw-rw-r--  3.0 unx 187426899 bx defN 16-Jun-23 10:37 PVM_data/PVM_models/2016_01_08_19_01_50_green_b_simple_d547417c/PVM_failsafe_0040000000.p.gz
-rw-rw-r--  3.0 unx 189394630 bx defN 16-Jun-23 10:36 PVM_data/PVM_models/2016_01_08_19_01_50_green_b_simple_d547417c/PVM_failsafe_0095000000.p.gz
```
Pass these to the tracking benchmark to reproduce the majority of the results.

#### Experiment 2

In this case the model is trained in an unsupervised way for a long time. Much like in the Experiment 1 you can train a model from scratch or use one of our pre-trained instances. We constructed a dataset out of clips from all the three categories plus some movies without any target labeled. The dataset is called "non_spec" to indicate there is no specific target in it. To train the model run:
```
python PVM_run.py -B -S model_zoo/experiment2.json
```
And wait. Much like in Experiment 1 it will take weeks if not months to get to 100M steps. 

For the pre-trained files, download and unzip ```PVM_data_models03.zip```. The zip contains 5 files:
```
-rw-rw-r--  3.0 unx 187227041 bx defN 16-Jun-23 11:31 PVM_data/PVM_models/2016_03_21_17_03_18_primable_simple_polynomial_small_11e4cc9f/PVM_failsafe_0020000000.p.gz
-rw-rw-r--  3.0 unx 187648051 bx defN 16-Jun-23 11:30 PVM_data/PVM_models/2016_03_21_17_03_18_primable_simple_polynomial_small_11e4cc9f/PVM_failsafe_0040000000.p.gz
-rw-rw-r--  3.0 unx 190867620 bx defN 16-Jun-23 11:28 PVM_data/PVM_models/2016_03_21_17_03_18_primable_simple_polynomial_small_11e4cc9f/PVM_failsafe_0095000000.p.gz
drwxrwxr-x  3.0 unx        0 bx stor 16-Jun-23 11:44 PVM_data/PVM_models/2016_02_10_02_16_33_primable_simple_large_52ae3b91/
-rw-rw-r--  3.0 unx 679544447 bx defN 16-Jun-23 11:44 PVM_data/PVM_models/2016_02_10_02_16_33_primable_simple_large_52ae3b91/PVM_failsafe_0020000000.p.gz
-rw-rw-r--  3.0 unx 680852880 bx defN 16-Jun-23 11:35 PVM_data/PVM_models/2016_02_10_02_16_33_primable_simple_large_52ae3b91/PVM_failsafe_0040000000.p.gz
```
The "primable_simple_polynomial_small" snapshots are instances of the models presented in the paper. The "primable_simple_large" is a similar instance but much bigger (and slower). We have included the bigger instance for completeness, but it is not used for any of the results presented in the paper. 

Once the unsupervised model is pre-trained,  you can now start the second phase of training, in supervised mode. To do that, run:
```
python PVM_run.py -r PATH_TO/pretrained_model.p.gz -D -O '{"supervised": "1", "supervised_rate": "0.0002", "dataset": "stop_sign", "steps": "10000" }'
```
Modify the parameters to fit the needs of your experiment. For one of our pre-trained models this command looks like this (recall you may skip the display option ```-D``` for faster run time):

```
python PVM_run.py -r PVM_models/2016_03_21_17_03_18_primable_simple_polynomial_small_11e4cc9f/PVM_failsafe_0095000000.p.gz -D -O '{"supervised": "1", "supervised_rate": "0.0002", "dataset": "stop_sign", "steps": "100000" }'
```

## Summary

The software presented here allows for reproduction of the majority of tracking results presented in the paper *"Unsupervised Learning from Continuous Video in a Scalable Predictive Recurrent Network"* (Piekniewski et al., 2016) (https://arxiv.org/abs/1607.06854).
This code is meant for scientific experimental purposes and should not be considered production-ready software (see also
the license restrictions preventing you from using this code for commercial purposes). Brain Corporation provides no
warranty about functionality of this code, and you are using it at your own risk. In particular, these simulations are compute-intensive and we are not responsible if they cause your computer to run at high temperatures and/or cause excessive CPU heating. You can and should read all the source provided here to make sure you know what this code is doing (hopefully it is reasonably documented). We recommend also reading the paper.

Enjoy!

P.S.: Contact us if you wish to contribute.
