#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as opt
import cv2
from helper_functions import cv2_helpers
import warnings
import time

def affine_flow(nghood, flows, model='scale', return_full=False, norm=2, **kwargs):
    """ Estimate affine flow in a neighborhood `nghood` from the given `flows`

    nghood : 2-D array
        nghood[i, :] = [coord in dimension 1, coord in dimension 2] of flow flows[i, :]
    flows : 2-D array
        flows[i, :] = [flow in direction of dim 1, flow in direction of dim 2] nghood[i, :]
        NOTE: typically, dim 1 will be the x-direction (and 2 y), but it is only important
            that the order is the same in `flows` and `nghood`
    model : string (optional)
        The type of affine function we estimate. Currently available:
            'scale': scaling + translation, i.e. A|b = s*eye(2) | b.
                    Preserves all angles
            'scale2': anisotropic scaling along the coord axes, i.e. A|b = diag(s1, s2) | b
                    Preserves rectangles parallel to the coord axes

    return_full : boolean (optional)
        if True, returns the full OptimizeResult object as the second output

    """

    # -- Assemble matrices for cost function
    # Coordinates
#    Z = np.kron( np.eye(2),
#                 np.hstack( [nghood.reshape((-1, 2)), np.ones((nghood.size//2, 1))]  )
#                )
    Z = np.kron( np.eye(2), np.hstack( [nghood, np.ones((nghood.shape[0], 1) )]) )
    # Flows
#    U = flows.reshape((-1, 2)).flatten(order='F')
    U = flows.flatten(order='F')

    # Define matrix that maps full A.flatten() to unique parameters
    if model == 'scale':
        # maps A.flatten to [scale, transl1, transl2]
        P = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0],
                      [0, 0, 0], [1, 0, 0], [0, 0, 1] ])
    elif model == 'scale2':
        P = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0],
                      [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1] ])

    ZP = np.dot(Z, P)

    if norm == 2:
        res = opt.lsq_linear(ZP, U, **kwargs)

    # Assemble result into affine transform matrix
    if model == 'scale':
        A = np.hstack([res.x[0]*np.eye(2), res.x[1:].reshape((2, 1))])
    elif model=='scale2':
        A = np.hstack([np.diag(res.x[:2]), res.x[2:].reshape((2, 1))])

    if return_full:
        return A, res
    else:
        return A


def flow_rectangle(rect, A=np.eye(2), t=np.zeros(2)):
    """Advects the rectangle rect=[left, top, width, height] according to the transform
        z' = A z + t, where A is diagonal (hence the transform preserves rectangles)
    """
    _, _, w, h = rect
    rectp = np.hstack([
                        rect[:2] + np.dot(A, rect[:2]) + t,
                        w + A[0, 0]*w, h + A[1, 1]*h
                        ])
    rectp_int = np.int0(np.round(rectp))

    return rectp, rectp_int


class ROI:
    """
    A class to represent a region of interest.
    The basic representation is ROI.rect=np.array([left, top, width, height])
    Setting lefttop (resp. rightbottom) directly does NOT preserve width & height of the rectangle but rather the position of the right-bottom (resp. left-top) corner.
    """
    def __init__(self, rect, success=True, quality=None, lost=False, label=''):

        if isinstance(rect, np.ndarray):
            self.rect = rect
        else:
            self.rect = np.array(rect)
        self.success = success
        self.quality = quality
        self.lost = lost

        self.label = label

        self.Ab = None

    @property
    def lefttop(self):
        return self.rect[:2]
    @lefttop.setter
    def lefttop(self, val):
        if np.any( val > self.rightbottom):
            raise ValueError("Left top corner needs to be to the left of and above the right bottom {}".
                             format(self.rightbottom))
        else:
            self.rect = np.hstack( (val, self.rightbottom-val) )

    @property
    def rightbottom(self):
        return self.rect[:2] + self.rect[2:]
    @rightbottom.setter
    def rightbottom(self, val):
        if np.any( val < self.lefttop):
            raise ValueError("Right bottom corner needs to be to the right of and below the left top {}".
                             format(self.lefttop))
        else:
            self.rect = np.hstack( (self.lefttop, val-self.lefttop) )

    def __str__(self):
        return 'ROI [left={}, top={}, width={}, height={}], lost={}'.format(*self.rect, self.lost)

    def __repr__(self):
        return "ROI({}, success={}, quality={}, lost={}, label={})".format(
            self.rect, self.success, self.quality, self.lost, self.label)

    def left(self):
        return self.rect[0]

    def right(self):
        return self.rect[0] + self.rect[2]

    def top(self):
        return self.rect[1]

    def bottom(self):
        return self.rect[1] + self.rect[3]

    def width(self):
        return self.rect[2]

    def height(self):
        return self.rect[3]

    def translate(self, t):
        self.rect = np.hstack( (self.rect + t, self.rect[2:]) )
        return self.rect

    def by_corners(self):
        """
        Returns ROI as an array [left, top, right, bottom]
        """
        return np.hstack((self.lefttop, self.rightbottom))

    def as_slice(self, order='yx'):
        left, top, right, bottom = self.by_corners_as_int()
        if order.lower() == 'yx':
            return	(slice(top, bottom+1), slice(left, right + 1))
        if order.lower() == 'xy':
            return	(slice(left, right + 1), slice(top, bottom+1))
        else:
            print("Order not understood, assuming 'yx'")  #TODO Upgrade to a warning
            return	(slice(top, bottom+1), slice(left, right + 1))

    def as_int(self):
        """ will convert to representation by corners, then round, then re-convert
        to representation by [left, top, width, height].
        Else we risk cases like left=1.49 and width=3.49 leads to the true right limit
        4.98 being rounded to 4 instead of 5.
        """
        by_corners_as_int = self.by_corners_as_int()
        return np.hstack( ( by_corners_as_int[:2], by_corners_as_int[2:] - by_corners_as_int[:2] ))

    def by_corners_as_int(self):
        return  np.uint32(np.round(self.by_corners()))


class optflow_tracker:
    """
    An optical-flow-based tracker for one or more regions of interest.

    NOTE: All measures of "quality" and "success" apart from marking regions as lost
    once they moved fully outside the frame, are currently not implemented.

    TODO: gray_tracker option is a bit misleading and should be renamed. If True, then
    frame is converted to grey scale before flow is computed, but if a grey scale frame is
    supplied, then option should be set to False.
    """

    def __init__(self, denseflow=None, initial_frame=None, rois=None,
                 fit_flow_fun='affine', flow_roi_fun=None, succ_eps=None, succ_required=np.all,
                 gray_tracker=False, prev_frame=None, label=None):

        # Available functions for local flow estimation
        self.__fit_flow_functions = {'median': [self.median_flow, np.inf],
                                      'affine': [self.affine_flow, np.inf]}

        if label is None:
            self.label = str(np.random.randint(65536))
        else:
            self.label = label

        try:
            # the functions to fit a flow field and apply it to a roi
            self._fit_flow, self.succ_eps  = self.__fit_flow_functions[fit_flow_fun]
            if succ_eps is not None:
                self.succ_eps = succ_eps
            self.succ_required = succ_required
        except KeyError:
                raise NotImplementedError('The local flow fitting method {} is not known. Known are {}'.
                                          format(fit_flow_fun, self.__fit_flow_functions.keys()))

        self._flow_roi = flow_roi_fun if flow_roi_fun is not None else \
                self.flow_rectangle

        # List of rois
        self.rois = [ roi if isinstance(roi, ROI) else ROI(roi, label=str(ii)) for ii,roi in enumerate(rois) ]
        # a cv2.DenseOpticalFlow object
        self._flow = denseflow
        self.gray_tracker = gray_tracker

        self.frame = initial_frame
        self.prev_frame = prev_frame
        self.flow = None
        self.t_flow_calc = None
        self.t_update = None

    def update(self, next_frame, flow=None, tflow=None):
        """
        If `flow` is given, the flow is not recomputed but the provided one is used.
        That is useful if we want to compare different methods of fitting a local flow model.
        `tflow` is the time it took to compute `flow` -- in case we want to report fair FPS.
        """


        if self.frame is None:
            warnings.warn('`self.frame` is `None`. Set initial frame first!',RuntimeWarning)
            return None

        # width and height of frame
        wh = np.array(next_frame.shape[:2][::-1])

        # Compute the flow (if it isn't given)
        if flow is None:
            t0 = time.perf_counter()
            if self.gray_tracker:
                self.flow = self._flow.calc(cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY), None)
            else:
                self.flow = self._flow.calc(self.frame, next_frame, None)
            self.t_flow_calc = time.perf_counter() - t0
        else:
            self.flow = flow
            if tflow is None:
                warnings.warn("Flow was given but tflow was not, so we don't know how long flow computation took", RuntimeWarning)
                self.t_flow_calc = np.nan
            else:
                self.t_flow_calc = tflow

        t00 = time.perf_counter()
        for roi in self.rois:
            if not roi.lost:
                # Fit the flow model
                roi.Ab, roi.succ, roi.quality = self._fit_flow(roi.as_int(), self.flow[roi.as_slice()])
                # Check on the success
                if not roi.succ:
                    # If not successful, don't update
                    print('not successful tracking {}'.format(roi))
                else:
                    # If successful, update
                    new_rect = self._flow_roi(roi)
                    # If we flowed the rectangle outside of the frame, cut off the part that is outside.
                    # However, if it is entirely outside, mark it as lost.
                    # If the rectangle has zero or negative height or width, it's also lost.
                    if np.any(new_rect[2:]<=0) or \
                            np.any((new_rect[:2] + new_rect[2:]) < 0) or \
                            np.any(new_rect[:2] > wh-1):
                        roi.lost = True
                        # in that case we also don't update the `roi`
                    else:
                        roi.rect = new_rect.copy()
                        roi.lefttop = np.fmax(roi.lefttop, 0)
                        roi.rightbottom = np.fmin(roi.rightbottom, wh-1 )

        succ_list = [roi.succ for roi in self.rois]

        overall_succ = self.succ_required(succ_list)
        if overall_succ:
            self.prev_frame = self.frame.copy()
            self.frame = next_frame

        self.t_update = time.perf_counter() - t00 + self.t_flow_calc
        return succ_list

    def reverse_flow(self):
        """
        computes the reverse flow, i.e. the flow from the current frame to the previous frame. Should be more or less the negative of the flow in opposite direction, i.e. `self.flow`.
        """
        if self.gray_tracker:
            rev_flow = self._flow.calc(cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY), None)
        else:
            rev_flow = self._flow.calc(self.frame, self.prev_frame, None)

        return rev_flow



    # @staticmethod
    def affine_flow(self, rect, flow, model='scale', norm=2):

        lft, tp, w, h = rect
        # if - due to rounding - the corners are outside the frame (they could be
        # by one pixel)

        # generate neighborhood from rect
        xx, yy = np.meshgrid(range(lft, lft+w+1), range(tp, tp+h+1))
        ngh = np.vstack((xx.ravel(), yy.ravel())).transpose()

        # reshape the flows
        flows_re = flow.transpose((1,0,2)).reshape((-1,2), order='F')

        # call the fitting function
        Ab, res = affine_flow(ngh, flows_re, model=model, return_full=True, norm=norm )

        quality = res.cost/ngh.shape[0]
        # check whether the affine flow was a good model
        succ = quality < self.succ_eps

        return Ab, succ, quality

    def median_flow(self, rect, flow, quantiles=(.25, .75), medflow_thresh=0.01):
        # first row is median, next two are the quantiles
#        quant_flows = np.nanquantile(flow, (.5,) + quantiles, axis=(0,1))
        med_flow = np.nanmedian(flow, axis=(0,1))
        # measure the quality by max of inter-quantile range normalized by median flow
#        quality = np.max( (quant_flows[2,:] - quant_flows[1,:]) / np.fmax(quant_flows[0,:], medflow_thresh) )
        quality = 0

        # success
        succ = quality < self.succ_eps

#        Ab = np.vstack( (np.zeros((2, 2)), quant_flows[0,:]) ).transpose()
        Ab = np.vstack( (np.zeros((2, 2)), med_flow) ).transpose()

        return Ab, succ, quality

    @staticmethod
    def flow_rectangle(roi : ROI, A=None, t=None):
        A = roi.Ab[:,:2] if A is None else A
        t = roi.Ab[:,2].squeeze() if t is None else t
        roi_rect = np.hstack([
            roi.lefttop + np.dot(A, roi.lefttop) + t,
            roi.width() + A[0, 0] * roi.width(), roi.height() + A[1, 1] * roi.height()
                            ])
        return roi_rect

    def overlay_rois(self, frame=None, color=None):
        frame = frame.copy() if frame is not None else self.frame.copy()
        colors = ((255,3,4), (3,4,250)) if color is None else color
        for ii, roi in enumerate(self.rois):
            x0,y0,x1,y1 = roi.by_corners_as_int()
            color = colors[0] if roi.success else colors[1]
            cv2_helpers.draw_roi(frame, (x0,y0), (x1,y1), box_col=color, label=roi.label)
        return frame

