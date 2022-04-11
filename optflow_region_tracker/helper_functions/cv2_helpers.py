#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
#
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 12:12:35 2018

@author: jpe
"""
#%%
import warnings
import cv2
import numpy as np

#%% CONSTANTS
vislight = (slice(360), slice(480))
infra = (slice(360, 720), vislight[1])
#%% Some convenience functions
def canvas_to_BGR(canvas):
    """Converts a plt.canvas (class matplotlib.backends.backend_*.FigureCanvas*, where * can be Mac or Agg or such
    to a numpy array in BGR color ordering, which can be used in cv2.imshow
    """
    (
        byte_string,
        wh,
    ) = canvas.print_to_buffer()  #  Returns bytestring of RGBA and (width, height)
    img = np.frombuffer(byte_string, dtype=np.uint8, count=wh[0] * wh[1] * 4).reshape(
        (wh[1], wh[0], 4)
    )[:, :, :3][
        :, :, ::-1
    ]  #  cut off A channel, then reorder 3rd dimension from RGB to BGR
    return img


def label_w_background(
    frame0,
    text1="",
    text2=0,
    bg_col=(255, 255, 255),
    txt_col=(0, 0, 0),
    display_str="{:9s} corner of region {:d}",
    origin=(0, 0),
    txt_scale=0.4,
):
    """Overlays the label with `text1` (string), `text2` (int) in `display_str` on `frame0`.
    Text is placed at `origin`, and the colors are given by `bg_col` and `txt_col`.
    Recall that in `cv2`, colors are tuples `(b,g,r)`, with `b,g,r = 0...255`.
    """
    frame0 = frame0.copy()  #  so frame0 is not overwritten
    text = display_str.format(text1, text2)
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, txt_scale, 1)
    #    print(text_size)
    cv2.rectangle(
        frame0, origin, (origin[0] + text_size[0], origin[1] + text_size[1]), bg_col, -1
    )
    cv2.putText(
        frame0,
        text,
        (origin[0], origin[1] + text_size[1]),
        cv2.FONT_HERSHEY_COMPLEX,
        txt_scale,
        txt_col,
    )
    return frame0


def draw_roi(
    initial_frame,
    topleft,
    bottomright,
    label="",
    box_col=(0, 255, 0),
    thickness=2,
    font=cv2.FONT_HERSHEY_COMPLEX,
    fontscale=0.7,
    offset=(0, 0),
):
    """draws the roi rectangle defined by `topleft` and `bottomright` onto `initial_frame`
    `label` can be either a string or an integer or anything with a `__str__` method.
    """
    cv2.rectangle(initial_frame, tuple(topleft), tuple(bottomright), box_col, thickness)
    # Put a little label with the region number, too
    cv2.putText(
        initial_frame,
        str(label),
        (int(topleft[0] + offset[0]), int(bottomright[1] + offset[1])),
        font,
        fontscale,
        box_col,
    )


#%%
def region_select_callback(
    window_name,
    initial_frame,
    lefttops,
    rightbottoms,
    str_seq=["top left", "bot right"],
    box_col=(0, 255, 0),
    label_origin=(0, 0),
):
    """Returns a function which can be used as a callback of a cv2 window to
    select regions to track

    Parameters
    ----------
    window_name : string
        The name of the cv2.namedWindow to which the callback is registered
    initial_frame : numpy.array
        The frame displayed in the window `window_name`
    lefttops, rightbottoms : List
        Variable names from the caller namespace, pointing to (ideally empty)
        lists to which the selected top left and bottom right corners can be
        attached
    box_col : Tuple of 3 uint8, optional
        Color of the boxes drawn. For cv2, that needs to be a Tuple of (B,G,R),
        with B,G,R each between 0 and 255

    Returns
    -------
    callback : function
            Function with the right signature to be used like
            ``cv2.setMouseCallback('WinName', callback)``
        Will append the coordinates of each left mouse click to `toplefts` and
        `bottomrights` in turn and display the selected regions.
        Each right click will delete the previously selected region and display it as crossed out.
        If the previous left-click only selected a left-top, but the current region is not yet completed, then only the left-top is deleted.
    """

    def callback(event, x, y, flags, param):
        # Set up two static variables (they should persist between calls to the function)
        if "_seq" not in callback.__dict__:
            callback._seq = True
            # print('Init _seq')
        if "_num_regions" not in callback.__dict__:
            callback._num_regions = 0

        if event == cv2.EVENT_LBUTTONDOWN:
            if callback._seq:
                lefttops.append([x, y])
                cv2.imshow(
                    window_name,
                    label_w_background(
                        initial_frame,
                        str_seq[1],
                        callback._num_regions,
                        origin=label_origin,
                    ),
                )
                callback._seq = not callback._seq
            else:
                rightbottoms.append([x, y])
                # Add the box to the frame
                draw_roi(
                    initial_frame,
                    lefttops[-1],
                    rightbottoms[-1],
                    box_col=box_col,
                    label=callback._num_regions,
                )
                print("New points: {}, {}.".format(lefttops[-1], rightbottoms[-1]))
                # Increment the region counter
                callback._num_regions += 1
                cv2.imshow(
                    window_name,
                    label_w_background(
                        initial_frame,
                        str_seq[0],
                        callback._num_regions,
                        origin=label_origin,
                    ),
                )
                callback._seq = not callback._seq
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(lefttops) > 0:
                if len(lefttops) > len(rightbottoms):
                    print("Deleting last left-top corner")
                    # toplefts = toplefts[:-1]  #  This makes toplefts a local variable!
                    lefttops.pop()
                    callback._seq = not callback._seq
                    cv2.imshow(
                        window_name,
                        label_w_background(
                            initial_frame,
                            str_seq[0],
                            callback._num_regions,
                            origin=label_origin,
                        ),
                    )
                else:
                    callback._num_regions -= 1
                    print("Deleting region {:d}".format(callback._num_regions))
                    #  cross it out
                    cv2.line(
                        initial_frame,
                        tuple(lefttops[-1]),
                        tuple(rightbottoms[-1]),
                        box_col,
                        2,
                    )
                    cv2.line(
                        initial_frame,
                        (lefttops[-1][0], rightbottoms[-1][1]),
                        (rightbottoms[-1][0], lefttops[-1][1]),
                        box_col,
                        2,
                    )
                    cv2.imshow(
                        window_name,
                        label_w_background(
                            initial_frame,
                            str_seq[0],
                            callback._num_regions,
                            origin=label_origin,
                        ),
                    )
                    lefttops.pop()
                    rightbottoms.pop()
            else:
                print("There was nothing to delete")

    return callback


def selectROIs(
    winname, frame, box_col=(0, 250, 0), showCrosshair=True, label_origin=(0, 0)
):
    """A replacement for cv2's cv2.selectROIs(). Difference is that you need to click for left top corner,
    then for right bottom corner, instead of dragging a rectangle

    showCrosshair is ignored, simply there for compatibility with cv2.selectROIs
    """
    tls = []  #  list for the top-left corners
    brs = []  #  list for the bottom-right corners
    #
    cv2.namedWindow(winname)
    # generate the callback function
    callback = region_select_callback(
        winname, frame, tls, brs, box_col=box_col, label_origin=label_origin
    )
    # register it with the window
    cv2.setMouseCallback(winname, callback)
    cv2.imshow(winname, label_w_background(frame, "top left", 0, origin=label_origin))
    # start the window's event loop
    print("Select regions. Press q or ESC when done.")
    _ = cv2.waitKey()

    # Once we exited, make sure that we didn't exit without selecting the last bottom right corner
    if len(brs) > len(tls):
        print(
            "[WARNING] Selected final top left without selecting corresponding bottom"
            " right."
        )
        tls = tls[:-1]

    # also, remove the callback
    cv2.setMouseCallback(winname, lambda *args: None)
    # selected_ROIs returns ROIs as list of [x,y, width, height]
    selected = [[xl, yt, xr - xl, yb - yt] for (xl, yt), (xr, yb) in zip(tls, brs)]

    return selected


#%% Region selection along line
def along_line_select_callback(
    window_name,
    initial_frame,
    startpoints,
    endpoints,
    str_seq=["start point", "end point"],
    line_col=(0, 255, 0),
    label_origin=(0, 0),
):
    """Returns a function which can be used as a callback of a cv2 window to
    select a line along which ROIs are automatically generated

    Parameters
    ----------
    window_name : string
        The name of the cv2.namedWindow to which the callback is registered
    initial_frame : numpy.array
        The frame displayed in the window `window_name`
    startpoints, endpoints : List
        Variable names from the caller namespace, pointing to (ideally empty)
        lists to which the selected start and end points of the lines can be appended
    line_col : Tuple of 3 uint8, optional
        Color of the boxes drawn. For cv2, that needs to be a Tuple of (B,G,R),
        with B,G,R each between 0 and 255

    Returns
    -------
    callback : function
            Function with the right signature to be used like
            ``cv2.setMouseCallback('WinName', callback)``
        Will append the coordinates of each left mouse click to `startpoints` and
        `endpoints` in turn and display the selected lines.
        Each right click will delete the previously selected line.
        If the previous left-click only selected a start, but the current line is not yet completed, then only the start is deleted.
    """
    display_str = "Select {:11s} of line {:d}"
    selected_point_style = cv2.MARKER_CROSS
    deleted_point_style = cv2.MARKER_TILTED_CROSS
    point_size = 15
    point_thick = 2
    # print('{}, {}'.format(bottomrights, toplefts))
    def callback(event, x, y, flags, param):
        # Set up two static variables (they should persist between calls to the function)
        if "_seq" not in callback.__dict__:
            callback._seq = True
            # print('Init _seq')
        if "_num_lines" not in callback.__dict__:
            callback._num_lines = 0

        if event == cv2.EVENT_LBUTTONDOWN:
            if callback._seq:
                startpoints.append([x, y])
                # Add the point:
                cv2.drawMarker(
                    initial_frame,
                    tuple(startpoints[-1]),
                    line_col,
                    selected_point_style,
                    point_size,
                    point_thick,
                )
                cv2.imshow(
                    window_name,
                    label_w_background(
                        initial_frame,
                        str_seq[1],
                        callback._num_lines,
                        origin=label_origin,
                        display_str=display_str,
                    ),
                )
                callback._seq = not callback._seq
            else:
                #        br_visible[:,seq[0]] = [x,y]
                endpoints.append([x, y])
                # Add the line and the point to the frame
                cv2.drawMarker(
                    initial_frame,
                    tuple(endpoints[-1]),
                    line_col,
                    selected_point_style,
                    point_size,
                    point_thick,
                )
                cv2.line(
                    initial_frame,
                    tuple(startpoints[-1]),
                    tuple(endpoints[-1]),
                    line_col,
                    2,
                )
                # Put a little label with the region number, too
                # cv2.putText(initial_frame, str(callback._num_regions),
                #             (toplefts[-1][0], bottomrights[-1][1]),
                #             cv2.FONT_HERSHEY_COMPLEX, .7, box_col)
                print("New points: {}, {}.".format(startpoints[-1], endpoints[-1]))
                # Increment the region counter
                callback._num_lines += 1
                cv2.imshow(
                    window_name,
                    label_w_background(
                        initial_frame,
                        str_seq[0],
                        callback._num_lines,
                        origin=label_origin,
                        display_str=display_str,
                    ),
                )
                callback._seq = not callback._seq
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(startpoints) > 0:
                # draw the deleted start point already
                cv2.drawMarker(
                    initial_frame,
                    tuple(startpoints[-1]),
                    line_col,
                    deleted_point_style,
                    point_size,
                    point_thick,
                )
                if len(startpoints) > len(endpoints):
                    print("Deleting last startpoint")
                    # toplefts = toplefts[:-1]  #  This makes toplefts a local variable!
                    startpoints.pop()
                    callback._seq = not callback._seq
                    cv2.imshow(
                        window_name,
                        label_w_background(
                            initial_frame,
                            str_seq[0],
                            callback._num_lines,
                            origin=label_origin,
                            display_str=display_str,
                        ),
                    )
                else:
                    # and draw the deleted endpoint, too
                    cv2.drawMarker(
                        initial_frame,
                        tuple(endpoints[-1]),
                        line_col,
                        deleted_point_style,
                        point_size,
                        point_thick,
                    )
                    callback._num_lines -= 1
                    print("Deleting line {:d}".format(callback._num_lines))
                    cv2.imshow(
                        window_name,
                        label_w_background(
                            initial_frame,
                            str_seq[0],
                            callback._num_lines,
                            origin=label_origin,
                            display_str=display_str,
                        ),
                    )
                    startpoints.pop()
                    endpoints.pop()
            else:
                print("There was nothing to delete")

    return callback


####
#%%
def _rois_along_line(startpoint, endpoint, wh, dist):
    """generates ROIs of size `wh` and with distance `dist` along the line between `start` and `end`
    `dist` is allowed to be negative, which corresponds to overlapping ROIs
    Returns
    center_points = [ np.array((x0,y0)), np.array((x1,y1)), ...]
        where the (xi, yi) are the coords of the centers
    ROIs = [ [xleft0, ytop0, width, height], [xleft1, ytop1, width, height] ,...  ]
        where the [xlefti, ytopi, width, height] are the coords of the rectangle of the ROI i
    """
    startpoint = np.array(startpoint)
    endpoint = np.array(endpoint)
    w, h = wh
    vec = endpoint - startpoint
    vec_len = np.sqrt(np.sum(vec**2))
    unit_vec = vec / vec_len
    factor = np.abs(unit_vec).max()
    L = wh[np.argmax(np.abs(vec))]
    used_up_length = L / factor
    last_start = startpoint
    center_points = []
    ROIs = []
    while used_up_length < (vec_len):
        next_center = last_start + unit_vec * (L / 2) / factor
        used_up_length += (L + dist) / factor
        last_start = last_start + (L + dist) / factor * unit_vec
        center_points.append(next_center)
        # make into ROI: [xleft, ytop, width, height]
        ROIs.append([next_center[0] - w / 2, next_center[1] - h / 2, w, h])
    if len(center_points) == 0:
        # if not even a single ROI fit, display a warning and put one in the middle
        warnings.warn(
            "Line was too short for even a single ROI; putting one in the midpoint"
        )
        next_center = (startpoint + endpoint) / 2
        center_points.append(next_center)
        ROIs.append([next_center[0] - w / 2, next_center[1] - h / 2, w, h])
    return center_points, ROIs


#%%
def selectROIs_along_line(
    winname,
    frame0,
    line_col=(0, 250, 0),
    showCrosshair=True,
    label_origin=(0, 0),
    ROIsize=(30, 30),
    dist=5,
):
    """A replacement for cv2's cv2.selectROIs(), but instead of selecting ROIs directly,
    you select start and endpoints of lines, equidistant ROIs are then generated along
    the lines.

    showCrosshair is ignored, simply there for compatibility with cv2.selectROIs
    """

    spts = []  #  list for the top-left corners
    epts = []  #  list for the bottom-right corners
    #
    cv2.namedWindow(winname)
    frame = frame0.copy()
    # generate the callback function
    callback = along_line_select_callback(
        winname, frame, spts, epts, line_col=line_col, label_origin=label_origin
    )
    # register it with the window
    cv2.setMouseCallback(winname, callback)
    cv2.imshow(
        winname,
        label_w_background(
            frame,
            "start point",
            0,
            origin=label_origin,
            display_str="Select {:s} of line {:d}",
        ),
    )
    # start the window's event loop
    print("Select regions. Press q or ESC when done.")
    _ = cv2.waitKey()

    # Once we exited, make sure that we didn't exit without selecting the last bottom right corner
    if len(spts) > len(epts):
        print(
            "[WARNING] Selected final start point without selecting corresponding end"
            " point."
        )
        epts = epts[:-1]

    # also, remove the callback
    cv2.setMouseCallback(winname, lambda *args: None)
    # selected_ROIs returns ROIs as list of [x,y, width, height]
    # selected = [
    #     [xl, yt, xr-xl, yb-yt] for (xl,yt), (xr,yb) in zip(tls, brs)
    #             ]
    ROIs = []
    for startpt, endpt in zip(spts, epts):
        _, rois = _rois_along_line(startpt, endpt, ROIsize, dist)
        ROIs += rois
    for ii, roi in enumerate(ROIs):
        roi0 = list(map(int, roi))
        draw_roi(
            frame,
            (roi0[0], roi0[1]),
            (roi0[0] + roi0[2], roi0[1] + roi0[3]),
            label=ii,
            box_col=line_col,
        )
    cv2.imshow(winname, frame)
    _ = cv2.waitKey(1)
    return ROIs


#%% Region selection with circle
def circle_select_callback(
    window_name,
    initial_frame,
    centers,
    perimeter_points,
    str_seq=["center", "pt on perimeter"],
    line_col=(0, 255, 0),
    label_origin=(0, 0),
):
    """Returns a function which can be used as a callback of a cv2 window to
    select a circle along which ROIs are generated automatically: one in the
    center, and a specified number along the perimeter

    Parameters
    ----------
    window_name : string
        The name of the cv2.namedWindow to which the callback is registered
    initial_frame : numpy.array
        The frame displayed in the window `window_name`
    centers, perimeter_points: List
        Variable names from the caller namespace, pointing to (ideally empty)
        lists to which the selected centers and radii are appended
    line_col : Tuple of 3 uint8, optional
        Color of the boxes drawn. For cv2, that needs to be a Tuple of (B,G,R),
        with B,G,R each between 0 and 255

    Returns
    -------
    callback : function
            Function with the right signature to be used like
            ``cv2.setMouseCallback('WinName', callback)``
        Will append the coordinates of each left mouse click to `centers` and
        `perimeter points` in turn and display the selected circles
        Each right click will delete the previously selected circle.
        If the previous left-click only selected a start, but the current circle is not
        yet completed, then only the center is deleted.
    """
    display_str = "Select {:11s} of circle {:d}"
    selected_center_style = cv2.MARKER_DIAMOND
    selected_perimeter_style = cv2.MARKER_CROSS
    deleted_point_style = cv2.MARKER_TILTED_CROSS
    point_size = 10
    point_thick = 2
    # print('{}, {}'.format(bottomrights, toplefts))
    def callback(event, x, y, flags, param):
        # Set up two static variables (they should persist between calls to the function)
        if "_seq" not in callback.__dict__:
            callback._seq = True
            # print('Init _seq')
        if "_num_circles" not in callback.__dict__:
            callback._num_circles = 0

        if event == cv2.EVENT_LBUTTONDOWN:
            if callback._seq:
                centers.append([x, y])
                # Add the point:
                cv2.drawMarker(
                    initial_frame,
                    tuple(centers[-1]),
                    line_col,
                    selected_center_style,
                    point_size,
                    point_thick,
                )
                cv2.imshow(
                    window_name,
                    label_w_background(
                        initial_frame,
                        str_seq[1],
                        callback._num_circles,
                        origin=label_origin,
                        display_str=display_str,
                    ),
                )
                callback._seq = not callback._seq
            else:
                #        br_visible[:,seq[0]] = [x,y]
                perimeter_points.append([x, y])
                # Add the line and the point to the frame
                cv2.drawMarker(
                    initial_frame,
                    tuple(perimeter_points[-1]),
                    line_col,
                    selected_perimeter_style,
                    point_size,
                    point_thick,
                )
                cv2.line(
                    initial_frame,
                    tuple(centers[-1]),
                    tuple(perimeter_points[-1]),
                    line_col,
                    2,
                )
                print("New points: {}, {}.".format(centers[-1], perimeter_points[-1]))
                # Increment the region counter
                callback._num_circles += 1
                cv2.imshow(
                    window_name,
                    label_w_background(
                        initial_frame,
                        str_seq[0],
                        callback._num_circles,
                        origin=label_origin,
                        display_str=display_str,
                    ),
                )
                callback._seq = not callback._seq
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(centers) > 0:
                # draw the deleted start point already
                cv2.drawMarker(
                    initial_frame,
                    tuple(centers[-1]),
                    line_col,
                    deleted_point_style,
                    point_size,
                    point_thick,
                )
                if len(centers) > len(perimeter_points):
                    print("Deleting last center")
                    # toplefts = toplefts[:-1]  #  This makes toplefts a local variable!
                    centers.pop()
                    callback._seq = not callback._seq
                    cv2.imshow(
                        window_name,
                        label_w_background(
                            initial_frame,
                            str_seq[0],
                            callback._num_circles,
                            origin=label_origin,
                            display_str=display_str,
                        ),
                    )
                else:
                    # and draw the deleted endpoint, too
                    cv2.drawMarker(
                        initial_frame,
                        tuple(perimeter_points[-1]),
                        line_col,
                        deleted_point_style,
                        point_size,
                        point_thick,
                    )
                    callback._num_circles -= 1
                    print("Deleting circle {:d}".format(callback._num_circles))
                    cv2.imshow(
                        window_name,
                        label_w_background(
                            initial_frame,
                            str_seq[0],
                            callback._num_circles,
                            origin=label_origin,
                            display_str=display_str,
                        ),
                    )
                    centers.pop()
                    perimeter_points.pop()
            else:
                print("There was nothing to delete")

    return callback


def _rois_on_circle(center, perimeter_point, wh, arc_dist):
    """generates ROIs of size `wh` and with distance `arc_dist` along the circle defined by
    `center` and `perimeter point`
    0 < `arc_dist` < 2*pi
    If `discard_outside==True`, then ROIs that fall outside the frame are discarded.
    Returns :
    center_points = [ np.array((x0,y0)), np.array((x1,y1)), ...]
        where the (xi, yi) are the coords of the centers
    ROIs = [ [xleft0, ytop0, width, height], [xleft1, ytop1, width, height] ,...  ]
        where the [xlefti, ytopi, width, height] are the coords of the rectangle of the ROI i
    """
    center = np.array(center)
    w, h = wh
    radius = np.sqrt(
        (perimeter_point[0] - center[0]) ** 2 + (perimeter_point[1] - center[1]) ** 2
    )
    theta0 = np.angle(
        perimeter_point[0] - center[0] + 1j * (perimeter_point[1] - center[1])
    )
    center_points = [
        center + radius * np.array([np.cos(theta), np.sin(theta)])
        for theta in theta0 + np.arange(0, 2 * np.pi, arc_dist)
    ]
    ROIs = [
        [next_center[0] - w / 2, next_center[1] - h / 2, w, h]
        for next_center in center_points
    ]
    return center_points, ROIs


def selectROIs_on_circle(
    winname,
    frame0,
    line_col=(0, 250, 0),
    showCrosshair=True,
    label_origin=(0, 0),
    ROIsize=(30, 30),
    num_rois=10,
    discard_outside=True,
):
    """A replacement for cv2's cv2.selectROIs().
    ROIs are selected by selecting the center of a circle, and one line on the perimeter.
    Then, `num_rois` ROIs of size `ROIsize` are generated along the circle.
    Additionally, one ROI is place in the center.

    showCrosshair is ignored, simply there for compatibility with cv2.selectROIs
    """
    spts = []  #  list for the top-left corners
    epts = []  #  list for the bottom-right corners
    arc_dist = np.pi * 2 / num_rois
    w, h = ROIsize
    H, W = frame0.shape[:2]
    #
    cv2.namedWindow(winname)
    frame = frame0.copy()
    # generate the callback function
    callback = circle_select_callback(
        winname, frame, spts, epts, line_col=line_col, label_origin=label_origin
    )
    # register it with the window
    cv2.setMouseCallback(winname, callback)
    cv2.imshow(
        winname,
        label_w_background(
            frame,
            "center",
            0,
            origin=label_origin,
            display_str="Select {:s} of circle {:d}",
        ),
    )
    # start the window's event loop
    print("Select regions. Press q or ESC when done.")
    _ = cv2.waitKey()

    # Once we exited, make sure that we didn't exit without selecting the last bottom right corner
    if len(spts) > len(epts):
        print(
            "[WARNING] Selected final center without selecting corresponding point on"
            " perimeter."
        )
        epts = epts[:-1]

    # also, remove the callback
    cv2.setMouseCallback(winname, lambda *args: None)
    # selected_ROIs returns ROIs as list of [x,y, width, height]
    # selected = [
    #     [xl, yt, xr-xl, yb-yt] for (xl,yt), (xr,yb) in zip(tls, brs)
    #             ]
    ROIs = []
    for center, perimeter_point in zip(spts, epts):
        ROIs.append([center[0] - w / 2, center[1] - h / 2, w, h])  # ROI in the center.
        _, rois = _rois_on_circle(center, perimeter_point, (w, h), arc_dist)
        if discard_outside:
            rois = [
                roi
                for roi in rois
                if not (
                    roi[0] < 0
                    or roi[1] < 0
                    or roi[0] + roi[2] > W
                    or roi[1] + roi[3] > H
                )
            ]
        ROIs += rois
    for ii, roi in enumerate(ROIs):
        roi0 = list(map(int, roi))
        draw_roi(
            frame,
            (roi0[0], roi0[1]),
            (roi0[0] + roi0[2], roi0[1] + roi0[3]),
            label=ii,
            box_col=line_col,
        )
    cv2.imshow(winname, frame)
    _ = cv2.waitKey(1)
    return ROIs


# %% Region selection by center
# NOTE: This part is a very quick and ugly adapration of the selection by circle or line
#    and could be done much simpler and cleaner
def center_select_callback(
    window_name,
    initial_frame,
    centers,
    str_seq=["center"],
    line_col=(0, 255, 0),
    label_origin=(0, 0),
):
    """Returns a function which can be used as a callback of a cv2 window to
    select a circle along which ROIs are generated automatically: one in the
    center, and a specified number along the perimeter

    Parameters
    ----------
    window_name : string
        The name of the cv2.namedWindow to which the callback is registered
    initial_frame : numpy.array
        The frame displayed in the window `window_name`
    centers: List
        Variable name from the caller namespace, pointing to (ideally empty)
        list to which the selected centers are appended
    line_col : Tuple of 3 uint8, optional
        Color of the boxes drawn. For cv2, that needs to be a Tuple of (B,G,R),
        with B,G,R each between 0 and 255

    Returns
    -------
    callback : function
            Function with the right signature to be used like
            ``cv2.setMouseCallback('WinName', callback)``
        Will append the coordinates of each left mouse click to `centers` and
        Each right click will delete the previously selected center.
    """
    display_str = "Select {:6s} of ROI {:d}"
    selected_center_style = cv2.MARKER_CROSS
    deleted_point_style = cv2.MARKER_TILTED_CROSS
    point_size = 10
    point_thick = 2
    # print('{}, {}'.format(bottomrights, toplefts))
    def callback(event, x, y, flags, param):
        # Set up static variables ( should persist between calls to the function)
        if "_num_centers" not in callback.__dict__:
            callback._num_centers = 0

        if event == cv2.EVENT_LBUTTONDOWN:
            centers.append([x, y])
            # Add the point:
            cv2.drawMarker(
                initial_frame,
                tuple(centers[-1]),
                line_col,
                selected_center_style,
                point_size,
                point_thick,
            )
            print("New points {}.".format(centers[-1]))
            # Increment the region counter
            callback._num_centers += 1
            cv2.imshow(
                window_name,
                label_w_background(
                    initial_frame,
                    str_seq[0],
                    callback._num_centers,
                    origin=label_origin,
                    display_str=display_str,
                ),
            )
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(centers) > 0:
                # draw the deleted start point already
                cv2.drawMarker(
                    initial_frame,
                    tuple(centers[-1]),
                    line_col,
                    deleted_point_style,
                    point_size,
                    point_thick,
                )

                print("Deleting last center")
                # toplefts = toplefts[:-1]  #  This makes toplefts a local variable!
                centers.pop()
                cv2.imshow(
                    window_name,
                    label_w_background(
                        initial_frame,
                        str_seq[0],
                        callback._num_centers,
                        origin=label_origin,
                        display_str=display_str,
                    ),
                )
            else:
                print("There was nothing to delete")

    return callback


#%%
def selectROIs_by_center(
    winname,
    frame0,
    line_col=(0, 250, 0),
    showCrosshair=True,
    label_origin=(0, 0),
    ROIsize=(30, 30),
    discard_outside=True,
):
    """A replacement for cv2's cv2.selectROIs().
    ROIs are selected by selecting just the center of a circle, then a ROI of size
     `ROIsize` are generated

    showCrosshair is ignored, simply there for compatibility with cv2.selectROIs
    discard_outside : bool
        If a ROI center is selected so close to the edge that one of its corners is
        outside the frame, the ROI is discarded and not recorded. Default: True
    """

    spts = []  #  list for the centers
    w, h = ROIsize
    H, W = frame0.shape[:2]
    #
    cv2.namedWindow(winname)
    frame = frame0.copy()
    # generate the callback function
    callback = center_select_callback(
        winname, frame, spts, line_col=line_col, label_origin=label_origin
    )
    # register it with the window
    cv2.setMouseCallback(winname, callback)
    cv2.imshow(
        winname,
        label_w_background(
            frame,
            "center",
            0,
            origin=label_origin,
            display_str="Select {:s} of center {:d}",
        ),
    )
    # start the window's event loop
    print("Select regions. Press q or ESC when done.")
    _ = cv2.waitKey()

    # also, remove the callback
    cv2.setMouseCallback(winname, lambda *args: None)
    # selected_ROIs returns ROIs as list of [x,y, width, height]
    # selected = [
    #     [xl, yt, xr-xl, yb-yt] for (xl,yt), (xr,yb) in zip(tls, brs)
    #             ]
    ROIs = []
    for center in spts:
        roi = [center[0] - w / 2, center[1] - h / 2, w, h]
        if discard_outside and (
            roi[0] < 0 or roi[1] < 0 or roi[0] + roi[2] > W or roi[1] + roi[3] > H
        ):
            pass
        else:
            ROIs += [roi]
    for ii, roi in enumerate(ROIs):
        roi0 = list(map(int, roi))
        draw_roi(
            frame,
            (roi0[0], roi0[1]),
            (roi0[0] + roi0[2], roi0[1] + roi0[3]),
            label=ii,
            box_col=line_col,
        )
    cv2.imshow(winname, frame)
    _ = cv2.waitKey(1)
    return ROIs
