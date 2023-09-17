import onnxruntime as ort
import numpy as np
import cv2
from loguru import logger
from model import SiamMask
from argparse import ArgumentParser

if __name__ == "__main__":
    # Argument Parser:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--video", type=str, default="tennis.mp4", required=False)
    parser.add_argument("--output", type=str, default=None, required=False)
    parser.add_argument("--profile", action="store_true", default=False, required=False)
    args = parser.parse_args()

    logger.info("Load in model...")
    model = SiamMask(args.model, do_profiling=args.profile)

    logger.info("Load in camera feed...")
    cap = cv2.VideoCapture(args.video)

    if not args.output is None:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"MP4V"), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    x,y,w,h = 298, 109, 173, 260
    i = 0
    logger.info("Default coordinates chosen (x,y,w,h): {}, {}, {}, {}".format(x,y,w,h))

    if args.profile:
        import cupy as cp
        avg_end2end = []
        t1 = cp.cuda.Event()
        t2 = cp.cuda.Event()

    while True:
        ret, im = cap.read()
        if not ret: break

        if args.profile: t1.record()
        if i == 0:
            model.init(im, (x,y,w,h))
            track_initialized = True
        else:
            mask = model.forward(im)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt_area = [cv2.contourArea(cnt) for cnt in contours]

            if len(contours) != 0 and np.max(cnt_area) > 100:
                contour = contours[np.argmax(cnt_area)]  # use max area polygon
                polygon = contour.reshape(-1, 2)
                im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
                im = cv2.polylines(im, [polygon], True, (0,0,255), 3)
        if args.profile:
            t2.record()
            cp.cuda.runtime.deviceSynchronize()
            profile_time = cp.cuda.get_elapsed_time(t1, t2)
            avg_end2end.append(profile_time)

        if not args.output is None:
            writer.write(im)

        cv2.imshow("Annotated Frame", im)
        cv2.waitKey(1)
        i += 1

    logger.success("Written annotate video file to: {}".format(args.output))

    if args.profile:
        logger.info("\nStatistics:\n\tAverage end2end time (ms): {:.2f}\n\tInitialization inference time (ms):{:.2f}\n\tMain inference time (ms): {:.2f}".format(
            np.mean(avg_end2end),
            model.infer_init,
            np.mean(model.avg_infer_reg)
        ))

    if not args.output is None:
        writer.release()