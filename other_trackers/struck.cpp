/*
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
*/
#include "struck.h"
#include "original_struck/src/Config.h"
#include "opencv2/opencv.hpp"

Tracker* tracker;
Config conf;
int width;
int height;

double scaleW;
double scaleH;

void struck_init(unsigned char* frame_data, int nrows, int ncols, const BoundingBox& bbox){
    std::string configPath = "config.txt";
	conf = Config(configPath);    
	tracker = new Tracker(conf);
    height = nrows;
    width = ncols;
	scaleW = (double)conf.frameWidth/ncols;
	scaleH = (double)conf.frameHeight/nrows;
    cv::Mat original_frame;
    original_frame = cv::Mat(height, width, CV_8UC3, frame_data);
    cv::Mat frame;
	cv::resize(original_frame, frame, cv::Size(conf.frameWidth, conf.frameHeight));
	FloatRect init_bb(bbox.xmin*scaleW, bbox.ymin*scaleH, bbox.width*scaleW, bbox.height*scaleH);
    tracker->Initialise(frame, init_bb);
}

void struck_track(unsigned char* frame_data){
    if(!tracker->IsInitialised()){
        std::cout << "Tracker is not initialized!!!" << std::endl;
        return;
    }
    cv::Mat original_frame;
    original_frame = cv::Mat(height, width, CV_8UC3, frame_data);
    cv::Mat frame;
	cv::resize(original_frame, frame, cv::Size(conf.frameWidth, conf.frameHeight));
    tracker->Track(frame);
    // tracker->Debug();
}

BoundingBox struck_get_bbox(){
    FloatRect struck_bbox = tracker->GetBB(); 
    BoundingBox bbox;
    bbox.xmin = static_cast<int>(struck_bbox.XMin()/scaleW);
    bbox.ymin = static_cast<int>(struck_bbox.YMin()/scaleH);
    bbox.width = static_cast<int>(struck_bbox.Width()/scaleW);
    bbox.height = static_cast<int>(struck_bbox.Height()/scaleH);
    return bbox; 
}
