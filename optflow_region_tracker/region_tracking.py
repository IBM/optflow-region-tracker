#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
# -- works with separate vis light and NIR videos, if requested
# -- opens dialogue windows if no file paths specified
# -- just select ROIs (no "suspicious" and "healthy")
# -- computes averages directly, so much smaller memory footprint
# -- writes result into CSV file
#  author: JPE, 2020-11-04
import time, os, argparse, csv, sys
from itertools import (
    zip_longest,
)  #  neeeded to transpose a list of possibly uneven lists
import numpy as np
import matplotlib.pyplot as plt
import cv2


from IO.FileVideoStream import FileVideoStream, VideoStream
import helper_functions.cv2_helpers as cv2h
import optflow_region_tracker as ofrt

select_window_name = "Select_Regions"


class __Strings_or_Positive_Container:
    """Only for use in argparse `choices` argument!
    Accepts any positive number or a set of strings (passed to the constructor)
    """

    def __init__(self, string_choices=[]):
        self.strings = string_choices

    def __contains__(self, item):
        if item in self.strings:
            return True
        # if we're still here...
        try:
            return float(item) >= 0
        except ValueError:
            return False
        # and if we're STILL here
        return False

    def __iter__(self):
        """Because argparse tries to iterate to list all possible choices.
        No other place should this iterator ever be used!"""
        return iter(self.strings + ["Any Positive Number"])


# Set up VIDEO STREAMS
def HD_4panel(left_top_left_top_w_h=(0, 0, 0, 360, 480, 360)):
    """Applies to a 4-panel layout as e.g. Novadaq PINPOINT systems produce.
    Returns (left top panel, left top panel in grey, left center panel)"""
    xvis, yvis, xnir, ynir, w, h = left_top_left_top_w_h
    vislight = (slice(yvis, yvis + h, None), slice(xvis, xvis + w, None))
    infra = (slice(ynir, ynir + h, None), slice(xnir, xnir + w, None))

    def transform(frame):
        return (
            frame[vislight],
            cv2.cvtColor(frame[vislight], cv2.COLOR_BGR2GRAY),
            frame[infra],
        )

    return transform


def frame_and_gray(frame):
    """Returns (frame, frame in grey)"""
    return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def setup_input_stream(
    stream_type,
    skip_frames,
    frames_to_process,
    abs_fname=None,
    offset_ms=None,
    transform=None,
):
    # stream_type is ignored, it's always 'file' here
    stream = FileVideoStream(abs_fname, offset_ms=offset_ms)
    cap = VideoStream(
        stream,
        skip_frames=skip_frames,
        transform=transform,
        queue_size=32,
        max_frames=frames_to_process,
        sleep_time_if_full=0.05,
    ).start()
    return cap


def _next_frame_merged_vid(cap1):
    def _next_frame():
        if cap1.isNotDone():
            vis, vis_gray, nir = cap1.read()
        else:
            raise StopIteration
        return vis, vis_gray, nir

    return _next_frame, (cap1,)


def _next_frame_separate_vids(cap1, cap2):
    def _next_frame():
        if cap1.isNotDone() and cap2.isNotDone():
            vis, vis_gray = cap1.read()
            nir = cap2.read()
        else:
            raise StopIteration
        return vis, vis_gray, nir

    return _next_frame, (cap1, cap2)


def select_rois_manual(vis0, display_str=""):
    """
    Select ROIs by clicking left top, then right bottom on each.
    """
    initial_frame = vis0.copy()
    cv2h.label_w_background(initial_frame, display_str=display_str)
    selected_rois = cv2h.selectROIs(
        select_window_name, initial_frame, showCrosshair=True, label_origin=(100, 0)
    )
    return selected_rois


def select_rois_center(vis0, w, h):
    """Select ROIs by clicking center"""
    initial_frame = vis0.copy()
    selected_rois = cv2h.selectROIs_by_center(
        select_window_name, initial_frame, label_origin=(100, 0), ROIsize=(w, h)
    )
    return selected_rois


def select_rois_along_line(vis0, w, h, dist):
    """
    Select ROIs by clicking start and end of a line, ROIs of size `(w,h)` are generated automatically
    with distance `dist`.
    """
    initial_frame = vis0.copy()
    selected_rois = cv2h.selectROIs_along_line(
        select_window_name,
        initial_frame,
        label_origin=(100, 0),
        ROIsize=(w, h),
        dist=dist,
    )
    return selected_rois


def select_rois_on_circle(vis0, w, h, num_rois):
    """
    Select ROIs by clicking center and point on perimeter or circle, `num_rois` ROIs
    of size `(w,h)` are generated automatically at the center and along the perimeter.
    """
    initial_frame = vis0.copy()
    selected_rois = cv2h.selectROIs_on_circle(
        select_window_name,
        initial_frame,
        label_origin=(100, 0),
        ROIsize=(w, h),
        num_rois=num_rois,
    )
    return selected_rois


def plot_live_intensities(I: list, dt, ax, t0=0):
    # I is assumed to be a list of lists
    # plots the intensities in `I` into `ax`, using a time step `dt`
    # Time axis is shifted left by `t0` seconds
    ax.cla()
    for rr, rri in enumerate(I):
        lenT = len(rri)
        ax.plot(t0 + np.arange(lenT) * dt, rri, label="ROI {:d}".format(rr))
    ax.legend()
    ax.set_xlabel("time in [s]")
    ax.set_ylabel("average intensities [a.u.]")


def make_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Load video and select ROIs.         ROIs are then tracked and their"
            " brightness is stored as a CSV file, along with the initial frame and"
            " (optionally) a video of the tracking."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # how are the videos given? Give either one or two videos, or open a file dialogue
    videos = parser.add_mutually_exclusive_group()
    videos.add_argument(
        "--merged_video",
        type=str,
        help=(
            "Absolute path to the video containing BOTH, visible light and nir image,  "
            "       as it would be generated by the Stryker system"
        ),
    )
    videos.add_argument(
        "--vis_nir_videos",
        type=str,
        nargs=2,
        help=(
            "Absolute paths to the videos containing visible light and nir image,      "
            "   visible light first. This is for when two separate videos are generated"
        ),
    )
    videos.add_argument(
        "--separate_vids",
        action="store_true",
        help=(
            "If present, then two file dialogues open:        the first for the visible"
            " light video, the second for the nir video."
        ),
    )

    parser.add_argument(
        "-s",
        "--shortname",
        required=False,
        type=str,
        help="Patient name/id. Used for the filenames",
        default="no_name",
    )
    parser.add_argument(
        "--start",
        type=float,
        help="Time offset from start of video (seconds).",
        default=0.0,
    )
    parser.add_argument(
        "--end",
        type=float,
        help=(
            "Time offset from start of video to end of tracking (seconds).             "
            "                    If not given, tracking runs until stopped manually or"
            " video file ends."
        ),
    )
    parser.add_argument(
        "--fps",
        "--apparent_fps",
        type=float,
        help=(
            "In some cases, the video file reports a different framerate from what     "
            "                            is seen in Python, e.g. even though the"
            " reported framerate is 60FPS,                                 reading 60"
            " frames in Python actually corresponds to 2 seconds of video              "
            "                   (instead of just one); for such cases, pass this"
            ' "effective framerate"                                in this argument. In'
            " particular for the videos from the NOVADAQ system,                       "
            '         pass "--fps 29.95".'
        ),
    )
    parser.add_argument(
        "--panel-locations",
        type=int,
        help="""The locations of the panels for visible light and NIR respectively in the merged video.
        Specify them as
        x of left top corner of visible light, y of left top corner of visible light,
        x of left top corner of NIR, y of left top corner of NIR, width, height.
        Note: The left-top corner of the frame has coordinates 0, 0. For example (and by
         default), for the Novadaq stack set to full HD, the option would be
        --panel-locations 0 0 0 360 480 360""",
        nargs=6,
        default=(0, 0, 0, 360, 480, 360),
    )
    #
    parser.add_argument(
        "--no-tracking",
        help="If given, no actual tracking is done, i.e. the ROIs do not move.",
        action="store_true",
    )

    #  storing results and accessing previous configurations
    parser.add_argument(
        "--store_folder",
        type=str,
        help=(
            "Path where to store the resulting CSV file and the tracking vid, if"
            " applicable.         If not given, a file dialogue is opened."
        ),
    )
    parser.add_argument(
        "--fill_value",
        help=(
            "Fill value to be used for intensities of regions that are lost already"
            " when writing to CSV.                             If not given then the"
            " cells are empty."
        ),
    )
    parser.add_argument(
        "--no-tracking-vid",
        help="If given, does not store a video of the tracking.",
        action="store_true",
    )
    parser.add_argument(
        "--no-plot-stored",
        help="If given, does not stort a plot of the intensities.",
        action="store_true",
    )
    #  Show the frames as we're tracking?
    parser.add_argument(
        "--show-tracking",
        help="If given, shows the frames that are being processed.",
        action="store_true",
    )
    parser.add_argument(
        "--tracking-max-height",
        type=int,
        help=(
            "If the video stacking visible and NIR light frames is         taller than"
            " that (in pixels), it will be resized"
        ),
        default=1080,
        dest="maxheight",
    )
    parser.add_argument(
        "--tracking-max-width",
        type=int,
        help=(
            "If the video stacking visible and NIR light frames is         taller than"
            " that (in pixels), it will be resized"
        ),
        default=1920 // 3,
        dest="maxwidth",
    )
    parser.add_argument(
        "--hide-intensities",
        help="If given, does not show the live intensities in their separate window.",
        action="store_true",
    )

    # ROI selection
    choices_for_selection_frame = __Strings_or_Positive_Container(
        ["first", "brightest"]
    )
    parser.add_argument(
        "--selection_frame",
        help="",
        choices=choices_for_selection_frame,
        default="first",
    )
    roiselection = parser.add_mutually_exclusive_group()
    roiselection.add_argument(
        "--ROIs_by_center",
        help="""Select the ROIs by a single click
                            to set the center. The size of the ROI can be supplied with
                            --ROI_size""",
        action="store_true",
    )
    roiselection.add_argument(
        "--ROIs_along_line",
        help="""If given, then ROI selection proceeds like this:
                user selects beginning and end of a line
                ROIs are created along the line in uniform size and distance.
                Size and distance can be supplied with --ROI_size_distance""",
        action="store_true",
    )
    roiselection.add_argument(
        "--ROIs_on_circle",
        help="""If given, then ROI selection proceeds like this:
                user selects center and one point on perimeter of circle
                ROIs are created in the center and along the perimeter.
                Size and number or ROIs can be supplied with --ROI_size_num""",
        action="store_true",
    )
    parser.add_argument(
        "--ROI_size_distance",
        type=int,
        help=(
            "width, height, distance (all in pixels) of automatically generated ROIs.  "
            "                      Only has effect if --ROIs_along_line is specified."
        ),
        default=(30, 30, 5),
        nargs=3,
    )
    parser.add_argument(
        "--ROI_size_num",
        type=int,
        help=(
            "width (in pixels), height (in pixels),                         number of"
            " automatically generated ROIs. Only has effect if --ROIs_along_line is"
            " specified."
        ),
        default=(30, 30, 10),
        nargs=3,
    )
    parser.add_argument(
        "--ROI_size",
        type=int,
        help=(
            "width, height (in pixels) of  generated ROIs.                        Only"
            " has effect if --ROIs_by_center is specified."
        ),
        default=(30, 30),
        nargs=2,
    )

    #  Tracking algorithm params
    parser.add_argument(
        "--fit_flow_fun",
        type=str,
        help="affine or median flow",
        choices=["median", "affine"],
        default="median",
    )
    parser.add_argument(
        "--skip_frames",
        type=int,
        help="number of frames to skip before processing next one",
        default=0,
    )
    parser.add_argument(
        "--aggfunc",
        type=str,
        help=(
            "The intensities across a ROI are averages using this average, either the"
            " mean or the median."
        ),
        choices=["mean", "median"],
        default="mean",
    )
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    # CONSTANTS
    # show stats every ... frames
    stats_every = 30
    # update the intensities every ... frames
    profiles_every = stats_every  #  set to 0 for not having them at all
    # cv2 window names
    winname = "Opt Flow Tracker"
    profiles_winname = "Live Intensities"
    # colors
    overlay_color = (3, 233, 3)
    #
    tracking_vid_extra_frames_skip = 5
    #  FLAGS
    no_tracking_FLAG = args.no_tracking
    tracking_vid_FLAG = not args.no_tracking_vid
    show_tracking_FLAG = args.show_tracking
    show_intensities_FLAG = not args.hide_intensities

    offset_ms = args.start * 1000
    # Do we need to open a file dialogue for the video file(s)?
    if (args.vis_nir_videos == None) and (args.merged_video == None):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        if args.separate_vids:  #  two file dialogues
            vis_path = filedialog.askopenfilename(
                title="Select visible light video file."
            )
            nir_path = filedialog.askopenfilename(title="Select NIR video file.")
        else:
            vid_path = filedialog.askopenfilename(title="Select video file.")
    elif (args.merged_video == None) and not (args.vis_nir_videos == None):
        args.separate_vids = True
        vis_path = args.vis_nir_videos[0]
        nir_path = args.vis_nir_videos[1]
    elif (args.vis_nir_videos == None) and not (args.merged_video == None):
        args.separate_vids = False
        vid_path = args.merged_video
    else:  #  we should never end up here
        raise RuntimeError(
            message=(
                "Error with the parsing of the video path. You should never reach this"
                " part of the source code though."
            )
        )
    frames_to_process = None  #  We should get away with this
    if args.separate_vids:
        cap1 = setup_input_stream(
            "file",
            abs_fname=vis_path,
            transform=frame_and_gray,
            offset_ms=offset_ms,
            skip_frames=args.skip_frames,
            frames_to_process=frames_to_process,
        )
        cap2 = setup_input_stream(
            "file",
            abs_fname=nir_path,
            transform=None,
            offset_ms=offset_ms,
            skip_frames=args.skip_frames,
            frames_to_process=frames_to_process,
        )
        next_frame, caps = _next_frame_separate_vids(cap1, cap2)
        vid_folder = os.path.dirname(vis_path)
    else:
        cap1 = setup_input_stream(
            "file",
            abs_fname=vid_path,
            transform=HD_4panel(args.panel_locations),
            offset_ms=offset_ms,
            skip_frames=args.skip_frames,
            frames_to_process=frames_to_process,
        )
        next_frame, caps = _next_frame_merged_vid(cap1)
        vid_folder = os.path.dirname(vid_path)
    # Now that the videos are open, we can set frames_to_process to its value, in case args.end was set
    FPS_reported = caps[0]._VideoStream__cap._FileVideoStream__cap.get(cv2.CAP_PROP_FPS)
    if args.fps:  #  if framerate is specified explicitly, use it
        FPS = args.fps
    else:  #  else use the reported framerate
        FPS = FPS_reported
    tracking_FPS = FPS / (1 + args.skip_frames)
    DT = 1 / tracking_FPS  #  time step between frames
    print(
        """[INFO]
    Video FPS (reported by file): \t\t\t{:.2f}
    Video FPS (used by Python): \t\t\t{:.2f}
    Tracking FPS: \t\t\t{:.2f} ({:d} frames skipped between tracking frames)\nTime between tracked frames:\t\t{:.3f}s
    """.format(
            FPS_reported, FPS, tracking_FPS, args.skip_frames, DT
        )
    )
    if args.end is None:
        frames_to_process = np.inf  #  We should get away with this
    else:
        frames_to_process = int(
            1 + (args.end - args.start) * FPS / (1 + args.skip_frames)
        )

    # Also, parse the folder we'll store stuff in.
    if args.store_folder is None:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        store_folder = filedialog.askdirectory(
            initialdir=os.path.abspath(vid_folder),
            title="Select folder to store the generated files in.",
        )
        if not store_folder:
            store_folder = os.path.abspath(vid_folder)
    else:
        store_folder = os.path.abspath(args.store_folder)

    # Select ROIs
    vis0, vis_grey0, nir0 = next_frame()
    select_frame = vis0
    if args.selection_frame.lower() == "brightest":
        # BAD: just use NIR stream
        cap_select = setup_input_stream(
            "file",
            abs_fname=nir_path,
            transform=None,
            offset_ms=offset_ms,
            skip_frames=20,
            frames_to_process=min(frames_to_process, 1000),
        )  # BAD: set max frames manually
        max_bright = 0
        while cap_select.isNotDone():
            nir = cap_select.read()
            if nir is not None and np.mean(nir) > max_bright:
                max_bright = np.mean(nir)
                select_frame = nir.copy()
        cap_select.stop()
    else:
        try:
            offset_select = 1000 * float(args.selection_frame)
            cap_select = setup_input_stream(
                "file",
                abs_fname=nir_path,
                transform=None,
                offset_ms=offset_select,
                skip_frames=20,
                frames_to_process=min(frames_to_process, 1000),
            )  # BAD: set max frames manually
            select_frame = cap_select.read()
            cap_select.stop()
        except ValueError:
            pass

    if args.ROIs_along_line:
        w, h, dist = args.ROI_size_distance
        selected_rois = select_rois_along_line(select_frame, w, h, dist)
        confirm_ROIs_flag = True
    elif args.ROIs_on_circle:
        w, h, num_rois = args.ROI_size_num
        selected_rois = select_rois_on_circle(select_frame, w, h, num_rois)
        confirm_ROIs_flag = True
    elif args.ROIs_by_center:
        w, h = args.ROI_size
        selected_rois = select_rois_center(select_frame, w, h)
        confirm_ROIs_flag = False
    else:
        selected_rois = select_rois_manual(
            select_frame, display_str="Select the regions of interest."
        )
        confirm_ROIs_flag = False

    if confirm_ROIs_flag:
        # Show all ROIs and ask for confirmation:
        frame = select_frame.copy()
        for ii, roi in enumerate(selected_rois):
            roi0 = list(map(int, roi))
            cv2h.draw_roi(
                frame,
                (roi0[0], roi0[1]),
                (roi0[0] + roi0[2], roi0[1] + roi0[3]),
                label=ii,
            )
        cv2.imshow(
            select_window_name,
            cv2h.label_w_background(
                frame0=frame,
                display_str="Press 'X' if not happy with the ROIs, SPACE if you are",
            ),
        )
        keypress = cv2.waitKey(0)
        if (keypress & 0xFF) == ord("X") or (keypress & 0xFF) == ord("x"):
            print("[INFO] ROIs not confirmed, exiting... Start over.")
            sys.exit()

    if args.aggfunc == "mean":
        aggfunc = np.nanmean
    elif args.aggfunc == "median":
        aggfunc = np.nanmedian
    else:
        raise RuntimeError(
            "An unknown aggregation function slipped through the parser somehow"
        )
    #%% Set up the TRACKER
    #  create the optical flow object
    if no_tracking_FLAG:
        zeroflow = type(
            "ZeroFlow",
            (cv2.DenseOpticalFlow,),
            {
                "calc": lambda self, img0, img1, xxx: np.zeros(
                    img0.shape[:2] + (2,), dtype=np.float
                )
            },
        )()  #  an object whose calc() method returns only zeros of the appropriate shape
        tracker = ofrt.optflow_tracker(
            zeroflow,
            vis_grey0,
            selected_rois,
            fit_flow_fun="median",
            label="zero flow tracker",
            gray_tracker=False,
        )
    else:
        dis = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)
        #  and the tracker itself
        tracker = ofrt.optflow_tracker(
            dis,
            vis_grey0,
            selected_rois,
            fit_flow_fun=args.fit_flow_fun,
            label=" ".join(("DIS_flow", args.fit_flow_fun)),
            gray_tracker=False,
        )

    # Let's hope we get away without preallocation
    ROI_intensities = [[aggfunc(nir0[roi.as_slice()])] for roi in tracker.rois]
    ROI_positions = [[roi.by_corners()] for roi in tracker.rois]

    # Store the initial frame
    initial_frame = tracker.overlay_rois(
        frame=select_frame.copy(), color=(overlay_color, (3, 4, 250))
    )
    cv2.imwrite(
        os.path.join(store_folder, args.shortname + "_InitialFrame.png"), initial_frame
    )

    # if tracking video, then open a capture device
    if tracking_vid_FLAG:
        out_file = os.path.join(store_folder, args.shortname + "_Tracking.mp4")
        writer = cv2.VideoWriter(
            out_file,
            cv2.VideoWriter_fourcc(*"avc1"),
            max(1, FPS // (tracking_vid_extra_frames_skip + args.skip_frames + 1)),
            tuple(vis0.shape[:2][::-1]),
            True,
        )
        writer.write(initial_frame)  #  write initial frame
    # if live intensities to be shown, create the figure window
    if show_intensities_FLAG:
        intensity_fig, intensity_ax = plt.subplots()
    else:
        intensity_fig = intensity_ax = None

    #%% TRACKING loop
    extra_ctr = 0  #  counter for extra frames to skip before VideoWriter gets another frame to write
    pp = 1  #  processed frame (initial one counted)
    t0 = time.perf_counter()
    t00 = time.perf_counter()

    while True:
        # read next frame
        try:
            vis, vis_grey, nir = next_frame()
        except StopIteration:
            print(
                "[INFO] Acquiring next frame failed, likely the end of the video was"
                " reached"
            )
            #  just to be safe:
            for cap in caps:
                cap.stop()
            break

        # update tracker
        succ = tracker.update(vis_grey)

        # update the ROI intensities and positions
        all_rois_lost = True
        for rr, roi in enumerate(tracker.rois):
            if not roi.lost:
                ROI_positions[rr].append(roi.by_corners())
                ROI_intensities[rr].append(aggfunc(nir[roi.as_slice()]))
                all_rois_lost = False

        # update output videos and frame counts
        if tracking_vid_FLAG:
            frame = tracker.overlay_rois(frame=vis, color=(overlay_color, (3, 4, 250)))
            if extra_ctr == tracking_vid_extra_frames_skip:
                writer.write(frame)
                extra_ctr = 0
            else:
                extra_ctr += 1
        if show_tracking_FLAG:
            frame = np.vstack(
                (
                    tracker.overlay_rois(frame=vis, color=(overlay_color, (3, 4, 250))),
                    tracker.overlay_rois(frame=nir, color=(overlay_color, (3, 4, 250))),
                )
            )
            h, w = frame.shape[:2]
            if (h > args.maxheight) or (w > args.maxwidth):
                rescale_factor = min(args.maxheight / h, args.maxwidth / w)
                frame = cv2.resize(frame, (0, 0), fx=rescale_factor, fy=rescale_factor)
            cv2h.label_w_background(frame, display_str="press s to stop")
            cv2.imshow(winname, frame)
            k = cv2.waitKey(1)
            if k == 115:  # s key to select
                print("Pressed {} - processing stopped ".format(k))
                #  stop all the captures
                for cap in caps:
                    cap.stop()
            elif k == -1:  # normally -1 returned,so don't print it
                pass
            else:
                print("Pressed {} - not mapped".format(k))  # else print its value

        pp += 1

        # print stats
        if not (pp % stats_every):
            t1 = time.perf_counter()
            print(
                "Processed {:d} frames, {} to go; avg over last {:d} frames: {:.3f} FPS"
                .format(
                    pp, frames_to_process - pp, stats_every, stats_every / (t1 - t0)
                )
            )
            t0 = t1

        # update live intensities
        if show_intensities_FLAG and (not (pp % profiles_every)):
            plot_live_intensities(ROI_intensities, DT, intensity_ax)
            cv2.imshow(profiles_winname, cv2h.canvas_to_BGR(intensity_fig.canvas))
            if pp == profiles_every:
                #  The first time the profiles are displayed, moves the window
                # with the profiles to the right so it's not over the tracking window
                cv2.moveWindow(profiles_winname, args.maxwidth + 2, 0)  #

        if pp >= frames_to_process:
            print("[INFO] requested number of frames processed")
            # stop all the captures, then break.
            for cap in caps:
                cap.stop()
            break

        if all_rois_lost:
            print("[INFO] All ROIs are lost, tracking is stopped.")
            # stop all the captures, then break.
            for cap in caps:
                cap.stop()
            break

    if tracking_vid_FLAG:
        writer.release()

    if not args.no_plot_stored:
        # move the legend outside
        intensity_ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        plt.tight_layout()
        plt.savefig(
            os.path.join(store_folder, args.shortname + "_Intensities_plot.PNG")
        )

    # Write the csvfile
    csvfilename = os.path.join(store_folder, args.shortname + "_Intensities.csv")
    with open(csvfilename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        # write the column headers
        csvwriter.writerow(["Time"] + [f"ROI {rr}" for rr in range(len(selected_rois))])
        # write each row
        for tt, ii in enumerate(
            zip_longest(*ROI_intensities, fillvalue=args.fill_value)
        ):
            csvwriter.writerow((tt * DT,) + ii)

    print(
        "[INFO] Wrote {:d} rows and {:d} columns in file {}\n".format(
            tt, len(selected_rois), csvfilename
        )
    )
