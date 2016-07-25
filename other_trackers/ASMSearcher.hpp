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

#ifndef ASMSearcher_HPP
#define ASMSearcher_HPP

#include <string>
#include <boost/python.hpp>
#include <opencv2/core/core.hpp>

struct cvmat_t {
	PyObject_HEAD
	CvMat *a;
	PyObject *data;
	size_t offset;
};

struct iplimage_t {
	PyObject_HEAD
	IplImage *a;
	PyObject *data;
	size_t offset;
};

cv::Mat convert_from_cvmat(PyObject *o, const char* name) {
	cv::Mat dest;
	cvmat_t *m = (cvmat_t*) o;
	void *buffer;
	Py_ssize_t buffer_len;

	m->a->refcount = NULL;
	if (m->data && PyString_Check(m->data)) {
		assert(cvGetErrStatus() == 0);
		char *ptr = PyString_AsString(m->data) + m->offset;
		cvSetData(m->a, ptr, m->a->step);
		assert(cvGetErrStatus() == 0);
		dest = m->a;

	} else if (m->data
			&& PyObject_AsWriteBuffer(m->data, &buffer, &buffer_len) == 0) {
		cvSetData(m->a, (void*) ((char*) buffer + m->offset), m->a->step);
		assert(cvGetErrStatus() == 0);
		dest = m->a;
	} else {
		printf("CvMat argument '%s' has no data", name);
	}
	return dest;

}

cv::Mat convert_from_cviplimage(PyObject *o, const char *name) {
	cv::Mat dest;
	iplimage_t *ipl = (iplimage_t*) o;
	void *buffer;
	Py_ssize_t buffer_len;

	if (PyString_Check(ipl->data)) {
		cvSetData(ipl->a, PyString_AsString(ipl->data) + ipl->offset,
				ipl->a->widthStep);
		assert(cvGetErrStatus() == 0);
		dest = ipl->a;
	} else if (ipl->data
			&& PyObject_AsWriteBuffer(ipl->data, &buffer, &buffer_len) == 0) {
		cvSetData(ipl->a, (void*) ((char*) buffer + ipl->offset),
				ipl->a->widthStep);
		assert(cvGetErrStatus() == 0);
		dest = ipl->a;
	} else {
		printf("IplImage argument '%s' has no data", name);
	}
	return dest;
}

cv::Mat convertObj2Mat(boost::python::object image) {
	if (strcmp(image.ptr()->ob_type->tp_name, "cv2.cv.iplimage") == 0) {
		return convert_from_cviplimage(image.ptr(),
				image.ptr()->ob_type->tp_name);
	} else
		return convert_from_cvmat(image.ptr(), image.ptr()->ob_type->tp_name);
}

#endif
