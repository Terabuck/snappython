# /data/data/com.termux/files/home/dev/snappython/processor/management/commands/compound_videos.py
# Grok https://www.yeschat.ai/app/chat/3443a528ea454ca69583cc996ad5c1a8
# https://github.com/Terabuck/snappython
# Using Video-Panorama stitching approach from https://github.com/krutikabapat/Video-Panorama
import cv2
import numpy as np
import logging
from django.core.management.base import BaseCommand

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_ultrasound_region(image, threshold=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return image[y:y+h, x:x+w]

def laplacian_pyramid_blend(img1, img2, mask, levels=5, blur_ksize=7):
    # Generate Gaussian pyramids for img1, img2, and mask
    G1, G2, GM = [img1.copy()], [img2.copy()], [mask.copy()]
    for i in range(levels):
        G1.append(cv2.pyrDown(G1[-1]))
        G2.append(cv2.pyrDown(G2[-1]))
        GM.append(cv2.pyrDown(GM[-1]))
    # Generate Laplacian pyramids for img1 and img2
    L1, L2 = [G1[-1]], [G2[-1]]
    for i in range(levels-1, 0, -1):
        size = (G1[i-1].shape[1], G1[i-1].shape[0])
        L1.append(cv2.subtract(G1[i-1], cv2.pyrUp(G1[i], dstsize=size)))
        L2.append(cv2.subtract(G2[i-1], cv2.pyrUp(G2[i], dstsize=size)))
    # Blend pyramids
    LS = []
    for l1, l2, gm in zip(L1, L2, GM[::-1]):
        gm_blur = cv2.GaussianBlur(gm, (blur_ksize, blur_ksize), 0) / 255.0
        ls = l1 * (1 - gm_blur) + l2 * gm_blur
        LS.append(ls.astype(np.uint8))
    # Reconstruct
    blended = LS[0]
    for i in range(1, levels):
        size = (LS[i].shape[1], LS[i].shape[0])
        blended = cv2.pyrUp(blended, dstsize=size)
        blended = cv2.add(blended, LS[i])
    return blended

class Command(BaseCommand):
    help = 'Process an ultrasound video to create a compound view with pyramid blending'

    def add_arguments(self, parser):
        parser.add_argument('--input', type=str, default='/data/source.mp4')
        parser.add_argument('--output', type=str, default='/data/output.mp4')
        parser.add_argument('--pyr', type=float, default=0.6)
        parser.add_argument('--levels', type=int, default=5)
        parser.add_argument('--step', type=int, default=1)
        parser.add_argument('--blur', type=int, default=7)

    def handle(self, *args, **options):
        input_path = options['input']
        output_path = options['output']
        pyr = options['pyr']
        levels = options['levels']
        step = options['step']
        blur_ksize = options['blur']

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error("Error: Could not open video file %s", input_path)
            return

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info("Video properties: %d frames, %dx%d, %.2f fps", n_frames, width, height, fps)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (2 * width, height))
        if not out.isOpened():
            logger.error("Error: Could not initialize video writer for %s", output_path)
            cap.release()
            return

        success, prev = cap.read()
        if not success:
            logger.error("Error: Could not read first frame")
            cap.release()
            return

        prev = extract_ultrasound_region(prev)
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        index = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, curr = cap.read()
            if not success:
                logger.info("Reached end of video or error reading frame at index %d", index)
                break

            logger.info("Processing frame %d", index)
            curr = extract_ultrasound_region(curr)
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            try:
                kp1, des1 = sift.detectAndCompute(prev_gray, None)
                kp2, des2 = sift.detectAndCompute(curr_gray, None)
                if des1 is None or des2 is None:
                    logger.warning("No descriptors found in frame %d, skipping", index)
                    index += step
                    continue

                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
                if len(good_matches) < 4:
                    logger.warning("Not enough good matches (%d) in frame %d, skipping", len(good_matches), index)
                    index += step
                    continue

                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is None or H.shape != (3, 3):
                    logger.warning("Invalid homography matrix in frame %d, skipping", index)
                    index += step
                    continue

                H = H.astype(np.float32)
                warped_prev = cv2.warpPerspective(prev, H, (curr.shape[1] + prev.shape[1], curr.shape[0]))
                warped_prev[0:curr.shape[0], 0:curr.shape[1]] = curr

                # Create mask for blending
                mask = np.zeros_like(curr_gray)
                mask[:, :curr.shape[1]//2] = 255  # left half for prev, right half for curr
                mask = cv2.warpPerspective(mask, H, (curr.shape[1] + prev.shape[1], curr.shape[0]))

                # Laplacian pyramid blending
                blended = laplacian_pyramid_blend(warped_prev, warped_prev, mask, levels=levels, blur_ksize=blur_ksize)

                # Crop to valid region
                gray_blend = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_blend, 1, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    blended = blended[y:y+h, x:x+w]
                # Resize to output size
                blended = cv2.resize(blended, (2 * width, height))
                out.write(blended)
                prev = blended
                prev_gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logger.error("Error processing frame %d: %s", index, str(e))
            index += step

        cap.release()
        out.release()
        logger.info("Processing complete. Output saved to %s", output_path)
