import os
import sys
import cv2
import numpy as np

if __name__ == '__main__':
    import glob
    img_files = sorted(glob.glob(os.path.join("../SiamMask/data/tennis", '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]
    writer = cv2.VideoWriter(
        "tennis.mp4",
        cv2.VideoWriter_fourcc(*"MP4V"),
        30,
        (ims[0].shape[1], ims[0].shape[0])
    )

    for im in ims:
        writer.write(im)
    writer.release()
    exit()

    print("load in video")
    cap = cv2.VideoCapture("test.mp4")
    new_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2))
    writer = cv2.VideoWriter(
        "out.mp4",
        cv2.VideoWriter_fourcc(*"MP4V"),
        cap.get(cv2.CAP_PROP_FPS),
        new_size
    )
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, new_size)
        if i > 200 and i < 700:
            writer.write(frame)
        i += 1
        print(i)
    writer.release()
     
