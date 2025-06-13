# /data/data/com.termux/files/home/dev/snappython/processor/management/commands/compound_videos.py
# Using Video-Panorama stitching approach from https://github.com/krutikabapat/Video-Panorama
# https://github.com/Terabuck/snappython
# Grok https://www.yeschat.ai/app/chat/81edabd3e69b4d28a3bc7e576a1bc62f
# v0004-00_06_17_8_extended.png
import cv2
import numpy as np
import os
import logging
from django.conf import settings
from django.core.management.base import BaseCommand

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_ultrasound_region(image, threshold=10):
    """Extract the ultrasound image region by removing black borders."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image, (0, 0, image.shape[1], image.shape[0])
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return image[y:y+h, x:x+w], (x, y, w, h)

def create_gaussian_mask(height, width, sigma_factor=8):
    """Create a Gaussian mask for feathered blending."""
    sigma = width / sigma_factor
    x = np.arange(width, dtype=np.float32)
    mask_1d = np.exp(-((x - width/2)**2) / (2 * sigma**2))
    mask_1d /= np.max(mask_1d)
    mask = np.tile(mask_1d, (height, 1))
    return mask

def smooth_flow(flow, kernel_size=15):
    """Smooth optical flow to reduce noise."""
    flow[:,:,0] = cv2.GaussianBlur(flow[:,:,0], (kernel_size, kernel_size), 0)
    flow[:,:,1] = cv2.GaussianBlur(flow[:,:,1], (kernel_size, kernel_size), 0)
    return flow

class Command(BaseCommand):
    help = 'Process an ultrasound video to create an extended view'

    def add_arguments(self, parser):
        parser.add_argument('filename', type=str, help='Name of the input video file')

    def process_video(self, filename):
        """Process a single video file to create an extended view"""
        mp4_path = os.path.join(settings.INPUT_DIR, filename)
        video_id = os.path.splitext(filename)[0]
        
        # Initialize video capture
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            logger.error("Error: Could not open video file %s", mp4_path)
            return
        
        # Get video properties
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info("Video properties: %d frames, %dx%d", n_frames, width, height)
        
        # Read first frame
        success, prev = cap.read()
        if not success:
            logger.error("Error: Could not read first frame from %s", mp4_path)
            cap.release()
            return
        
        # Extract ultrasound region and bounding box
        prev_region, (x_prev, y_prev, w_prev, h_prev) = extract_ultrasound_region(prev)
        prev_gray = cv2.cvtColor(prev_region, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.equalizeHist(prev_gray)  # Enhance contrast
        
        # Set canvas size (2x height, 4x width of cropped first frame)
        canvas_h = h_prev * 2
        canvas_w = w_prev * 4
        accumulated = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        weights = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        
        # Initialize offsets to center the first frame
        offset_x = (canvas_w - w_prev) // 2
        offset_y = (canvas_h - h_prev) // 2
        
        # Place first frame on canvas
        mask = create_gaussian_mask(h_prev, w_prev, sigma_factor=8)
        accumulated[offset_y:offset_y + h_prev, offset_x:offset_x + w_prev] += prev_gray.astype(np.float32) * mask
        weights[offset_y:offset_y + h_prev, offset_x:offset_x + w_prev] += mask
        
        index = 1  # Start from the second frame
        amplification_factor = 5
        pyr_scale = 0.4  # Adjusted for finer motion detection
        levels = 4       # Increased for better scale handling
        winsize = 35     # Larger window for smoother flow
        
        while index < n_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, curr = cap.read()
            if not success:
                logger.info("Reached end of video or error reading frame at index %d", index)
                break
            
            logger.info("Processing frame %d", index)
            
            # Extract ultrasound region for current frame
            curr_region, (x_curr, y_curr, w_curr, h_curr) = extract_ultrasound_region(curr)
            curr_gray = cv2.cvtColor(curr_region, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.equalizeHist(curr_gray)
            
            # Resize to match first frame's dimensions
            curr_gray = cv2.resize(curr_gray, (w_prev, h_prev), interpolation=cv2.INTER_AREA)
            
            # Compute and smooth optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, pyr_scale, levels, winsize, 3, 5, 1.2, 0)
            flow = smooth_flow(flow, kernel_size=15)
            
            # Estimate motion with robust averaging
            dx = np.median(flow[:,:,0])
            dy = np.median(flow[:,:,1])
            
            # Update offsets
            offset_x += int(dx * amplification_factor)
            offset_y += int(dy * amplification_factor)
            
            # Create mask for blending
            mask = create_gaussian_mask(h_prev, w_prev, sigma_factor=8)
            
            # Place current frame on canvas with bounds checking
            x1 = max(0, min(offset_x, canvas_w - w_prev))
            x2 = x1 + w_prev
            y1 = max(0, min(offset_y, canvas_h - h_prev))
            y2 = y1 + h_prev
            
            if x2 > canvas_w or y2 > canvas_h:
                logger.warning("Frame %d exceeds canvas bounds after adjustment", index)
                continue
            
            accumulated[y1:y2, x1:x2] += curr_gray.astype(np.float32) * mask
            weights[y1:y2, x1:x2] += mask
            
            # Update previous frame
            prev_gray = curr_gray
            index += 1
        
        # Compute final image by averaging
        final_image = np.zeros_like(accumulated, dtype=np.uint8)
        mask = weights > 0
        final_image[mask] = (accumulated[mask] / weights[mask]).astype(np.uint8)
        
        # Crop to non-zero region
        non_zero_y, non_zero_x = np.where(np.any(final_image != 0, axis=1)), np.where(np.any(final_image != 0, axis=0))
        if non_zero_y[0].size > 0 and non_zero_x[0].size > 0:
            min_y, max_y = non_zero_y[0][0], non_zero_y[0][-1] + 1
            min_x, max_x = non_zero_x[0][0], non_zero_x[0][-1] + 1
            final_image = final_image[min_y:max_y, min_x:max_x]
        
        # Save the final extended view
        output_path = os.path.join(settings.OUTPUT_DIR, f"{video_id}_extended.png")
        cv2.imwrite(output_path, final_image)
        logger.info("Extended view saved to %s", output_path)
        
        # Cleanup
        cap.release()
        logger.info("Processing complete for %s", filename)

    def handle(self, *args, **options):
        filename = options['filename']
        self.process_video(filename)
