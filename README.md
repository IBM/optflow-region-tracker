Implementation of the region tracker described in the MICCAI 2022 submission "A Real-Time Region Tracking Algorithm Tailored to Endoscopic Video with Open-Source Implementation" (preprint in the `docs/` folder).
## Installation
Using `conda`, it should be sufficient to create a new environment (I'm calling it `tracker` here), activate it, and
install the requirements
```bash
conda create -n tracker python=3.7
conda activate tracker
conda install -c conda-forge --file requirements.txt
```
The `conda-forge` channel typically contains more current versions of more packages.

Installation of requirements using `pip` should work similarly
```bash
pip install -r requirements.txt
```
but is untested. 
The `conda` version should also install a working version of `ffmpeg`, which is used by OpenCV to read and write video files; with `pip`, it might install only python bindings for `ffmpeg`. However, `ffmpeg` is installed on most systems anyway.

## An Example

To see all available command-line options, change into the main folder and run
```bash
python -i ./optflow_region_tracker/region_tracking.py --help
```

To see the tool in action (and make sure the installation worked), try
```bash
python -i ./optflow_region_tracker/region_tracking.py -s test --merged_video "./data/synthetic_example.mp4" --store_folder ~/tmp --show-tracking --panel-locations 0 0 0 256 389 256
```
* The only required argument is `-s SHORT_NAME`, where `SHORT_NAME` is used to generate file- and variable names.
* `--merged_video` allows you to specify the location a video which contains panels of visible light and infrared spectrum images, as is the case e.g. in Stryker Pinpoint systems.
* `--store_folder` allows you to specify a folder in which the resulting files (CSV file for the collected data, PNG files with the initial frame and a plot of the collected intensities, and an MP4 file to check on the tracking) are stored
* if either aren't given, file dialogues are opened
* `--panel-locations` allows you to specify the locations of the panels in the video. The first two coordinates are the 
(x,y) of the left-top corner (y=0 is at the top of the frame!) of the visible light image, used for tracking,
 the next 2 are the (x,y) of the left-top corner of the infrared image (or more general, the image in which data is to be collected), and the last two are width and height (visible light and infrared panels have to have the same size). The default panel locations are `0 0 0 360 480 360` which corresponds to the layout stored by e.g. Stryker Pinpoint systems.
* `--show-tracking` indicates that you'd like to see the tracking and data collection as they happen
* invoking python with `-i` drops you into the interpreter after tracking is done. This ensures that the windows remain open. You can exit the interpreter with CTRL+D or by typing `exit()`.

If everything is installed properly, you should be presented with a small GUI on which you can draw a few regions of interest. Once they're drawn, press "S" or "ESC" to start tracking. Two new windows open, one presenting the tracking (in the upper panel) and data collection (in the lower panel), the other updating curves of the collected data, one line per ROI drawn; see below and also `data/ScreenCapture.mov`.
The folder specified after `--store_folder` should now contain the files `test_Intensities.csv`, `test_Tracking.mp4`, `test_InitialFrame.png` and `test_Intensities_plot.PNG`.
  ![Screenshot](./data/synthetic_example_screenshot.PNG "Screenshot")  

### Licensing Info
The frame `data/frames/Angiodysplasie_cropped.jpg` was used to generate the example video file `data/synthetic_example.mp4`, using the same process as described in the paper.  
The frame is taken from wikimedia, where it was uploaded by "Joachim Guntau (=J.Guntau) - Endoskopiebilder.de" and is used under the
[GNU Free Documentation License, Version 1.2](https://commons.wikimedia.org/wiki/Commons:GNU_Free_Documentation_License,_version_1.2). The frame has been modified by cropping it to remove the time stamp.

## LICENSE
SPDX-License-Identifier: Apache-2.0

Copyright 2019,2020,2021,2021 IBM Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.