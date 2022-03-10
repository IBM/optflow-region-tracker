#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 12:12:35 2018
@author: jpe
"""
import cv2
from threading import Thread
from queue import Queue, Empty
import warnings, time

class FileVideoStream:
    def __init__(self,filename,offset_ms=0,cap_backend=None):
        if cap_backend==None:
            self.__cap = cv2.VideoCapture(filename)
        else:  #  e.g. `cv2.VideoCapture(fname, cv2.CAP_FFMPEG)` if backend to be enforced
            self.__cap = cv2.VideoCapture(filename, cap_backend)
        # skip ahead
        self.__cap.set(cv2.CAP_PROP_POS_MSEC, offset_ms)

    def read(self):
        return self.__cap.read()

    def grab(self):
        return self.__cap.grab()

    def release(self):
        return self.__cap.release()

class VideoStream:
    """
    Attributes
    ----------
        read_frames : int
            Number of frames that have been read, (transformed, ) and placed into the queue
        Q : queue.Queue
            The queue into which elements are placed. Don't call it directly though, use ``read()`` instead.
        thread : threading.Thread
            The thread in which the frames are read (and transformed). Don't interact with it directly, use
            ``start()``, ``stop()`` and ``isNotDone()`` instead.

    Methods
    -------
        start :
            start the thread.
        stop :
            stop the thread and release resources
        read :
            return the oldest read frame and remove it from queue
        isNotDone :
            query if there are still frames to be consumed, either because they're in the queue or still being read

    Examples
    --------
    import FileVideoStream as FV
    # Absolute path to the video file
    fname = './video.MP4'
    # define the stream
    stream = FV.FileVideoStream(fname, offset_ms=1400)
    # this function will be applied to every frame before it is returned (by calling .read() )
    transform = lambda f: f[:360, :480, ...]
    # Create the capture object by wrapping the stream into VideoStream and starting it
    cap = FV.VideoStream(stream, transform=transform, max_frames=100).start()
    img = cap.read()  #  this returns the 1st frame from the video after slicing the 360x480 left-top rectangle out of it
    cap.stop()  #  once you're done.

    Inspired by Adrian Rosenbrock's `filevideostream` class (https://github.com/jrosebr1/imutils/blob/master/imutils/video/filevideostream.py)
    """
    def __init__(self, stream, skip_frames=0, transform=None,
                queue_size=16, max_frames=None, cap_backend=None, sleep_time_if_full=0.1, timeout_if_empty=1, default_if_empty=None ):
        """
        Parameters
        ----------
        filename : str
            Path to the video
        offset_ms : int, optional
            Offset to start the reading at, in ms, by default 0
        skip_frames : int, optional
            After reading one frame, skip ``skip_frames`` frames before reading the next, by default 0
        transform : function, optional
            This function is applied to every frame before it's placed into the queue (and then accessible via the ``read()`` method). Use this e.g. to
            convert to greyscale or slice the frame. If None, then no transformation is applied, by default None
        queue_size : int, optional
            The amount of elements that fit in the queue. If the queue is full, no more frames are read until the queue has room again, by default 16
        max_frames : int, optional
            Maximum number of frames to be placed into the queue. When this number is reached, the thread stops and no more frames are read and placed
            into the queue; if None, then there is no maximum, by default None
        cap_backend : int, optional
            a backend paramter that cv2.VideoCapture understands, e.g. cv2.CAP_FFMPEG. If None, then cv2 default is used, by default None
        sleep_time_if_full : float, optional
            if the queue is full, the thread sleeps for this amount of time before trying again, by default 0.1
        timeout_if_empty : float, optional
            if ``read()`` is called on an empty queue, wait for this long until giving up and throwing a warning. Typically, the frames should be read and placed
            into the queue much faster than consumed, so an empty queue should not happen regularly, by default 0.1
        default_if_empty : [type], optional
            If the queue is empty even after waiting ``timeout_if_empty`` seconds, then return this value instead, by default None
        """
        self.skip_frames = skip_frames
        self.max_frames = max_frames
        self.read_frames = 0
        self.__cap = stream

        # open the queue
        self.Q = Queue(maxsize=queue_size)
        self.__sleep_if_full = sleep_time_if_full
        self.__timeout_if_empty = timeout_if_empty
        self.__default_if_empty = default_if_empty

        # initialize the thread
        self.thread = Thread(target=self.__update, args=())
        self.thread.daemon = True

        self.__stopped = True
        self.transform = transform

    def start(self):
        """Starts the thread which is extracting (and transforming) the frames

        Returns
        -------
        FileVideoStream
            Returns ``self``
        """
        self.__stopped = False
        self.thread.start()
        return self

    def __update(self):
        """This function is executed inside the thread"""
        while not self.__stopped:
            if not self.Q.full():  #  True:  #
                ret, frame = self.__cap.read()
                self.read_frames += 1

                if not ret:
#                    raise RuntimeError()
                    print('[WARNING] Stopping stream because acquisition (`read()`) of new frame failed.')
                    self.__stopped = True
                    break

                # Apply transforms, if any
                if self.transform:
                    frame = self.transform(frame)

                self.Q.put(frame)
                # if a max number of frames to read was set and we reached it, stop now
                if (self.max_frames is not None) and (self.read_frames>=self.max_frames):
                    print('[INFO] Stopping stream because `max_frames` frames were read.')
                    self.__stopped = True
                    break
                # skip frames (if any)
                for ii in range(self.skip_frames):
                    ret = self.__cap.grab()
                if not ret:
#                    raise RuntimeError()
                    print('[WARNING] Stopping stream because acquisition (`grab()`) of new frame failed.')
                    self.__stopped = True
            else:
                time.sleep(self.__sleep_if_full)

        # if a max number of frames to read was set and we have not reached it yet but are stopping nonetheless
        if (self.max_frames is not None) and (self.read_frames<self.max_frames):
            warnings.warn('Failed to read new frame before max_frames were read.' +
                          ' Read {:d} frames, but max_frames was {:d}.'.format(
                              self.read_frames, self.max_frames))
        self.__cap.release()

    def read(self, timeout=None):
        """Removes the oldest frame from the queue and returns it

        Parameters
        ----------
        timeout : float (optional)
            If not None then overrides the default timeout duration of the Queue.get()
            call. Might be useful for first read attempt, as it sometimes takes a while
            for the capture to start. Default is None.

        Returns
        -------
        queue element
            the oldest element from the queue. If ``transform`` is not None, then this is the output of applying the transformation to a read frame.
        """
        if timeout is None: timeout=self.__timeout_if_empty
        try:
            frame = self.Q.get(timeout=timeout)
        except Empty:
            warnings.warn('Queue was empty and stayed empty for %fs. Thread might have stopped, check with `FileVideoStream.isNotDone()`. Returning default.' % self.__timeout_if_empty)
            frame = self.__default_if_empty
        return frame

    def isNotDone(self):
        """Returns True if either the thread is still running and reading, or if there are still elements left in the queue

        Returns
        -------
        Boolean
            ``self.Q.qsize()>0 or not self.__stopped``
        """
        return self.Q.qsize()>0 or not self.__stopped

    def stop(self):
        """Indicates that the thread should be stopped. You can still ``read()`` until the queue is empty."""
        print('cap.stop has been called')
        # import pdb; pdb.set_trace();
        self.__stopped = True
        self.thread.join()
