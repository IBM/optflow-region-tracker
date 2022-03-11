#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
import numpy as np
import cv2
from matplotlib.path import Path

# %%
def random_elation(min_max_scale=(0.9, 1.1), r=1, wh=(480, 360)):
    """Returns a random elation matrix [1, 0, 0; 0, 1, 0; v1, v2, r]
    so that any vector within the frame (size `wh`) doesn't get scaled more than
    `min_max_scale`.
    We need min_max_scale[0] < r < min_max_scale[1]
    Note: It is assumed that the origin is at the center of the frame.
    """
    v = (
        np.random.rand(
            2,
        )
        - 0.5
    )
    w, h = wh
    ox, oy = w / 2, h / 2
    # extreme_point_dots = np.array(list(map(np.inner, [v]*2, np.array([[ox,oy],[-ox,oy]]) )))
    max_ip = max(np.abs(v[0] * ox + v[1] * oy), np.abs(v[0] * ox - v[1] * oy))
    v_scaled = v * min(r - min_max_scale[0], min_max_scale[1] - r) / max_ip
    return np.array([[1, 0, 0], [0, 1, 0], [v_scaled[0], v_scaled[1], r]])


def random_rot(phi_max=0.2):
    phi = (np.random.rand() - 0.5) * 2 * phi_max
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def random_perspective(
    min_max_elation_scale=(0.9, 1.1),
    max_rot=0.2,
    max_shear=1,
    min_max_scale=(0.9, 1.1),
    max_trans=(10, 10),
    wh=(480, 360),
):
    w, h = wh
    ox, oy = w / 2, h / 2
    minsc, maxsc = min_max_scale

    scale1, scale2 = np.sqrt(
        np.random.rand(
            2,
        )
        * (maxsc - minsc)
        + minsc
    )
    trans = (
        np.array(max_trans)
        * (
            np.random.rand(
                2,
            )
            - 0.5
        )
        * 2
    )

    T = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]], dtype=np.float)
    Ti = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], dtype=np.float)
    random_shear = np.eye(3)
    random_shear[0, 1] = 2 * (np.random.rand() - 0.5) / max(ox, oy) * max_shear
    random_scale1 = np.diag([scale1, 1 / scale1, 1])
    random_affine = np.block(
        [
            [scale1 * random_rot(phi_max=max_rot), trans.reshape(2, 1)],
            [np.array([0, 0, 1])],
        ]
    )
    elate = random_elation(min_max_scale=min_max_elation_scale, r=1, wh=wh)
    H = Ti @ random_affine @ random_shear @ random_scale1 @ elate @ T
    return H


# %%
def frame_generator(
    initial_frame,
    points=None,
    wh=(420, 320),
    max_translation_speed=0.5,
    min_max_elation_scale=(0.9, 1.1),
    max_rot=0.2,
    max_shear=1,
    min_max_scale=(0.9, 1.1),
    max_frames=None,
):
    """points as (x,y) in the coord system of the _generated_ frame, i.e. in
    `initial_frame` they are shifted: x_initial_frame = x + dleft etc
    """
    H, W = initial_frame.shape[:2]
    w, h = wh
    dtop, dbot, dleft, dright = map(
        int, ((H - h) / 2, (H - h) / 2, (W - w) / 2, (W - w) / 2)
    )
    cutout = (slice(dtop, H - dbot), slice(dleft, W - dright))
    if points is None:
        points = np.array(())
    initial_p = (points.astype(float) + np.array((dleft, dtop)))[np.newaxis, ...]
    # extra axis necessary for cv2.perspectiveTransform

    yield initial_frame[cutout], initial_p.squeeze() - np.array((dleft, dtop))
    frames_generated = 1

    # pick the next point to translate towards
    def gen_next_steps(prev_point, max_speed=max_translation_speed):
        # next point to head towards
        nxt_point = np.array((dright + dleft, dbot + dtop)) * np.random.rand(
            2,
        ) - np.array((dleft, dtop))
        # that results in a translation vector
        nxt_trans = nxt_point - prev_point
        # the length of the total translation is
        len_nxt_trans = np.linalg.norm(nxt_trans)
        # which means if we go in steps of max_speed...
        # (fill arange backwars so the last point is included)
        return (
            nxt_point,
            prev_point
            + np.arange(1, 0, -max_speed / len_nxt_trans)[::-1].reshape((-1, 1))
            * nxt_trans,
        )

    next_point, next_steps = gen_next_steps(np.array((0, 0)))
    while not frames_generated == max_frames:
        for next_step in next_steps:
            frame = cv2.warpAffine(
                initial_frame,
                np.hstack((np.eye(2), next_step.reshape((2, 1)))),
                dsize=(W, H),
                borderMode=cv2.BORDER_REFLECT,
            )
            RP = random_perspective(
                min_max_elation_scale=min_max_elation_scale,
                max_rot=max_rot,
                max_shear=max_shear,
                min_max_scale=min_max_scale,
                max_trans=(0, 0),
                wh=(W, H),
            )
            frame = cv2.warpPerspective(
                frame, RP, dsize=(W, H), borderMode=cv2.BORDER_REFLECT
            )
            p = cv2.perspectiveTransform(initial_p + next_step[np.newaxis, ...], RP)
            yield frame[cutout], p.squeeze() - np.array((dleft, dtop))
            frames_generated += 1
        next_point, next_steps = gen_next_steps(
            prev_point=next_point, max_speed=np.random.rand() * max_translation_speed
        )


# %%
def synthetic_reflection(
    wh, origin, strel=cv2.MORPH_ELLIPSE, el_size_wh=(5, 9), blur=False, sat_val=255
):
    """Generates a single synthetic reflection, based on the structuring element `strel`.

    You can blend with an image `image`, for instance, like this:
    >> R = random_reflection(image.shape[:2][::-1], (20,50), blur=(5,4), el_size_wh=(11,25))[:,:,np.newaxis]
    >> IR = np.uint8( R*(R/255.0) + (255-R)/255.0 * image)

    Parameters
    ----------
    wh : Iterable
        tuple (width, height) of the frame the reflection will be applied to
    origin : Iterable
        The (x,y) coordinates of the center of the reflection
    strel : cv2., optional
        The structuring element to be used, by default cv2.MORPH_ELLIPSE
    el_size_wh : tuple, optional
        The width and height of the reflection, by default (5,9)
    blur : bool or int or (int, int), optional
        If not False, a blur with kernel size (3,3) (if True), (k,k) (if int), (kx, ky)
        (if (int, int)) is applied to the structuring elemem, by default False
    sat_val : int, optional
        Value to be inserted into the saturated regions (i.e. the reflection), by default 255

    Returns
    -------
    Z : np.array of shape (wh[1], wh[0])
        Z[y,x] == 0 wherever the reflection is not
        Z[y,x] == sat_val wherever the reflection fully saturaters
        Z[y,x] == something in between 0 and sat_val, if a blur was applied
    """
    w, h = wh
    ox, oy = origin
    S = cv2.getStructuringElement(strel, el_size_wh) * sat_val
    el_h, el_w = S.shape
    Z = np.zeros((h, w), dtype=np.uint8)
    if (oy + (el_h + 1) // 2) - h < el_h and (ox + (el_w + 1) // 2) - w < el_w:
        # if the Structuring Element doesn't lie outside to the right or bottom entirely
        Z[
            max(0, oy - el_h // 2) : oy + (el_h + 1) // 2,
            max(0, ox - el_w // 2) : ox + (el_w + 1) // 2,
        ] = S[
            max(0, -oy + el_h // 2) : el_h + min(1, h - (oy + (el_h + 1) // 2)),
            max(0, -ox + el_w // 2) : el_w + min(1, w - (ox + (el_w + 1) // 2)),
        ]
    if blur:
        if blur == True:
            blur = (3, 3)
        elif isinstance(blur, int):
            blur = (blur, blur)
        Z = cv2.blur(Z, blur, borderType=cv2.BORDER_REFLECT)
    return Z


# blend like this:
# R = random_reflection(frame.shape[:2][::-1], (200,100), blur=(5,4), el_size_wh=(11,25))[:,:,np.newaxis]
# IR = np.uint8( R*(R/255.0) + (255-R)/255.0 * frame)
# plt.imshow(IR[:,:,::-1])

# %%
def polygon_rect_IoU(
    polygon_vertices_xy, rect_ltwh, test_radius=0.1, verbose_output=False
):
    """Computes the Intersection over Union for a polygon and a rectangle by rasterizing,
    i.e. (number of pixels inside both)/(number of pixels inside either).

    Parameters
    ----------
    polygon_vertices_xy : Iterable of tuples
        Vertices of polygon -- will be passed to matplotlib.path.Path constructor
    rect_ltwh : (n,4) np.array
        n rectangles, each described by `rect_ltwh[i,:]=[x_left, y_top, width, height]`
    test_radius : float, optional
        Passed to Path().contains_points as `radius` kwarg; >0 appears necessary so corners
        belong to polygon, by default 0.1
    verbose_output : bool, optional
        If True, also return the index matrices `xx`,`yy` and boolean matrices `I_r` and
        `I_p` so that `I_r[i,j]==True` iff point (xx[i,j], yy[i,j]) is inside the rectangle,
        same for `I_p`. Note that the grid `xx,yy` is only large enough to contain both
        the polygon and the rect, and it could also extend into negative indices.
        By default False

    Returns
    -------
    IoU : np.array of float in [0,1] or NaN
        IoU[i] = intersect(polygon, rect[i])/union(polyogon, rect[i]). If both shapes
        are empty, then IoU[i] = NaN.
    """

    P = Path(polygon_vertices_xy)
    rect_ltwh = np.atleast_2d(rect_ltwh)
    l, t = np.int32(np.min(rect_ltwh[:, :2], axis=0))
    r, b = np.int32(np.min(rect_ltwh[:, :2] + rect_ltwh[:, 2:], axis=0))
    # l, t, w, h = map(int,rect_ltwh)
    xmin, ymin = map(min, np.int32(P.get_extents().min), (l, t))
    xmax, ymax = map(max, np.int32(P.get_extents().max), (r, b))
    xx, yy = np.mgrid[xmin : xmax + 1, ymin : ymax + 1]
    # indicator matrices for polygon
    I_P = P.contains_points(
        np.hstack((xx.reshape((-1, 1)), yy.reshape((-1, 1)))), radius=test_radius
    ).reshape((xx.shape))
    IoU = np.full((rect_ltwh.shape[0],), fill_value=np.nan)
    I_rect = np.zeros(xx.shape + (rect_ltwh.shape[0],), dtype=bool)
    for ii, rect in enumerate(rect_ltwh):
        l, t, w, h = map(int, rect)
        # I_rect = np.zeros(xx.shape, dtype=bool)
        I_rect[l - xmin : l - xmin + w + 1, t - ymin : t - ymin + h + 1, ii] = True

        if np.any(I_rect) or np.any(I_P):
            IoU[ii] = np.count_nonzero(I_rect[:, :, ii] & I_P) / np.count_nonzero(
                I_rect[:, :, ii] | I_P
            )

    # import pdb; pdb.set_trace()
    if verbose_output:
        return xx, yy, I_P, I_rect, IoU
    else:
        return IoU


# %%
if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    H = random_perspective(wh=(128, 92), max_rot=1.5, max_trans=(3, 3))
    I = np.zeros((92, 128), dtype=np.uint8)
    o_w_h = (46, 64, 20, 20)
    I[
        o_w_h[0] - int(o_w_h[-1] / 2) : o_w_h[0] + int(o_w_h[-1] / 2),
        o_w_h[1] - int(o_w_h[-2] / 2) : o_w_h[1] + int(o_w_h[-2] / 2),
    ] = 255
    I[o_w_h[0], o_w_h[1]] = 0
    # # H = np.block( [ [np.eye(2), np.zeros((2,1))], [np.array([ (1/2)/np.sqrt(400), (1/2)/np.sqrt(400) , 1])] ] )
    # H = random_perspective(wh=(128,92), min_max_scale=(.5, 1.2), r=2)
    # T = np.array( [ [1, 0, -o_w_h[1]], [0, 1, -o_w_h[0]], [0, 0, 1]], dtype=np.float)
    # Ti = np.array( [ [1, 0, o_w_h[1]], [0, 1, o_w_h[0]], [0, 0, 1]], dtype=np.float)
    # # I2 = cv2.warpPerspective( cv2.warpAffine(I, T, (128,92)) ,H, (128,92))
    # I2 = cv2.warpPerspective(I, Ti @ H @ T, (128,92))

    I2 = cv2.warpPerspective(I, H, (128, 92))
    plt.imshow(I, alpha=1, cmap="flag")
    plt.imshow(I2, alpha=0.8)
    plt.plot(o_w_h[1], o_w_h[0], "c+")

    # writer = cv2.VideoWriter('~/tmp/test.mp4', cv2.VideoWriter_fourcc(*'avc1'), 30, (420,320), True)
    # G = frame_generator(frame, max_rot=0.1, max_shear=2, min_max_elation_scale=(.9,1.05), min_max_scale=(1,1), max_translation_speed=5)
    # for ii in range(300):
    #     writer.write(next(G))
    # writer.release()

# points = np.array(((200,200),(175,190)))
# f0 = frame.copy()
# for p in points:
#     cv2.drawMarker(f0, (p[0],p[1]), (0,255,0), markerType=cv2.MARKER_CROSS, thickness=3)
# G = frame_generator(f0, max_rot=0.1, max_shear=2, min_max_elation_scale=(.8,1.1), min_max_scale=(1,1), max_translation_speed=5, points=points)
# fig, ax = plt.subplots(4, 8, sharex=True, sharey=True)
# for a in ax.flat:
#     f, p = next(G)
#     a.imshow(f[:,:,::-1])
#     a.plot(p[:,0], p[:,1], 'wv')
