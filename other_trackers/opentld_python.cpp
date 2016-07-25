/**
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
 **/

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/class.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/pure_virtual.hpp>
#include <iostream>
#include "TLD.h"
#include "DetectorCascade.h"
#include "ASMSearcher.hpp"
#include <boost/python/tuple.hpp>
#include <boost/python.hpp>

using namespace boost::python;
using namespace cv;
using namespace std;
namespace tld
{

class TLD2
{
public:
	TLD2();
    // Boost python wrapper functions
    boost::python::tuple getCurrBB();
    void selectObject_numpy(boost::python::object img, const boost::python::tuple bb);
    void processImage_numpy(boost::python::object img);
	void set_width_and_height(boost::python::tuple wh);
    float currConf;
    TLD* Tracker;
    cv::Rect bounding_box_;
};


TLD2::TLD2()
{
	Tracker = new TLD();
	currConf = Tracker->currConf;
}

void TLD2::selectObject_numpy(boost::python::object img,
		const boost::python::tuple bb) {
	bounding_box_ = cv::Rect(extract<int>(bb[0]), extract<int>(bb[1]),
			extract<int>(bb[2]), extract<int>(bb[3]));
	cv::Mat image = convertObj2Mat(img);

	Tracker->selectObject(image, &bounding_box_);
}

void TLD2::processImage_numpy(boost::python::object img) {
	cv::Mat image = convertObj2Mat(img);
	Tracker->processImage(image);
	currConf = Tracker->currConf;
}

boost::python::tuple TLD2::getCurrBB() {
	if (Tracker->currBB != NULL)
		return make_tuple(Tracker->currBB->x, Tracker->currBB->y, Tracker->currBB->width, Tracker->currBB->height);
	else
		return boost::python::tuple();
}

void TLD2::set_width_and_height(boost::python::tuple wh){
	Tracker->detectorCascade->imgWidth = extract<int>(wh[0]);
	Tracker->detectorCascade->imgWidthStep = extract<int>(wh[0]);
	Tracker->detectorCascade->imgHeight = extract<int>(wh[1]);
}

}


BOOST_PYTHON_MODULE(tld)
{
	using namespace boost::python;

	class_<tld::DetectorCascade>("DetectorCascade", no_init)
	.def_readwrite("imgWidth", &tld::DetectorCascade::imgWidth)
	.def_readwrite("imgHeight", &tld::DetectorCascade::imgHeight)
	.def_readwrite("imgWidthStep", &tld::DetectorCascade::imgWidthStep)
	;

	class_<tld::TLD2>("TLD2")
	.def(init<>())
	.def_readwrite("currConf", &tld::TLD2::currConf)
	.def("selectObject", &tld::TLD2::selectObject_numpy)
	.def("processImage", &tld::TLD2::processImage_numpy)
	.def("getCurrBB", &tld::TLD2::getCurrBB)
	.def("set_width_and_height", &tld::TLD2::set_width_and_height)
	.add_property("detectorCascade",
			make_getter(&tld::TLD::detectorCascade, return_value_policy<reference_existing_object>()),
			make_setter(&tld::TLD::detectorCascade, return_value_policy<reference_existing_object>()))
	;

}
