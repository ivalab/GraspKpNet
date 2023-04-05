import os

import cv2
import numpy as np
import pyrealsense2 as rs

from gknet.detectors.detector_factory import detector_factory
from gknet.opts import opts

image_ext = ["jpg", "jpeg", "png", "webp"]
video_ext = ["mp4", "mov", "avi", "mkv"]
time_stats = ["tot", "load", "pre", "net", "dec", "post", "merge"]


def normalize(x, min_d, max_min_diff):
    return 255 * (x - min_d) / max_min_diff


def demo(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if opt.demo == "video" or opt.demo[opt.demo.rfind(".") + 1 :].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == "webcam" else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            cv2.imshow("input", img)
            ret = detector.run(img)
            time_str = ""
            for stat in time_stats:
                time_str = time_str + "{} {:.3f}s |".format(stat, ret[stat])
            print(time_str)
            if cv2.waitKey(1) == 27:
                return  # esc to quit
    elif opt.demo == "webcam":
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = (
                aligned_frames.get_depth_frame()
            )  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            min_depth = np.min(depth_image[:, :])
            max_depth = np.max(depth_image[:, :])
            max_min_diff = max_depth - min_depth

            normalize = np.vectorize(normalize, otypes=[np.float32])
            depth_image = normalize(depth_image, min_depth, max_min_diff)

            inp = color_image.copy()
            inp[:, :, 2] = depth_image
            inp = inp[:, :, ::-1]

            ret = detector.run(inp)
            time_str = ""
            for stat in time_stats:
                time_str = time_str + "{} {:.3f}s |".format(stat, ret[stat])
    else:
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind(".") + 1 :].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

        for image_name in image_names:
            ret = detector.run(image_name)
            time_str = ""
            for stat in time_stats:
                time_str = time_str + "{} {:.3f}s |".format(stat, ret[stat])
            print(time_str)


if __name__ == "__main__":
    opt = opts().init()
    demo(opt)
