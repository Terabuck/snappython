
import os
import cv2
import numpy as np
import subprocess
import shutil
import traceback
import datetime
from PIL import Image, ImageDraw, ImageFont
from django.conf import settings
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Creates spatially compounded ultrasound images from videos using panorama stitching'

    def handle(self, *args, **options):
        os.makedirs(settings.PROCESSED_VIDEOS_DIR, exist_ok=True)
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
        os.makedirs(settings.TEMP_FRAMES_DIR, exist_ok=True)

        processed = 0
        errors = 0
        for filename in os.listdir(settings.INPUT_DIR):
            if filename.lower().endswith('.mp4'):
                self.stdout.write(f"Processing {filename} for spatial compounding...")
                try:
                    self.process_video(filename)
                    processed += 1
                except Exception as e:
                    self.stderr.write(f"Error processing {filename}: {str(e)}")
                    self.stderr.write(traceback.format_exc())
                    errors += 1
        
        if processed:
            self.stdout.write(self.style.SUCCESS(f"Successfully processed {processed} videos"))
        if errors:
            self.stdout.write(self.style.ERROR(f"Failed to process {errors} videos"))
        if not processed and not errors:
            self.stdout.write("No MP4 files found in input directory")

    def process_video(self, filename):
        mp4_path = os.path.join(settings.INPUT_DIR, filename)
        video_id = os.path.splitext(filename)[0]
        temp_dir = os.path.join(settings.TEMP_FRAMES_DIR, video_id)
        output_path = os.path.join(settings.OUTPUT_DIR, f"{video_id}_compound.png")
        
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Extract frames
            frames = self.extract_frames(mp4_path, temp_dir, num_frames=30)
            
            if not frames:
                raise ValueError("No frames extracted")
                
            # Detect and crop ultrasound window from each frame
            cropped_frames, crop_region = self.detect_and_crop_ultrasound(frames)
            
            if not cropped_frames:
                raise ValueError("No frames after cropping")
                
            # Create panorama using feature-based stitching
            compound_image = self.create_ultrasound_panorama(cropped_frames, video_id)
            
            if compound_image is None:
                raise ValueError("Panorama creation failed")
                
            # Save final composite
            compound_image.save(output_path)
            
            shutil.move(mp4_path, os.path.join(settings.PROCESSED_VIDEOS_DIR, filename))

        except Exception as e:
            self.stderr.write(f"Error processing video: {str(e)}")
            raise
        finally:
            # Clean up temporary frames
            shutil.rmtree(temp_dir, ignore_errors=True)

    def detect_and_crop_ultrasound(self, frames):
        """Detect and crop the ultrasound window from frames"""
        if not frames:
            return [], None
            
        # Use the first frame for detection
        frame = frames[0]
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # Apply adaptive thresholding to find ultrasound window
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        # Find contours to detect ultrasound region
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour that is likely the ultrasound window
        ultrasound_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area and area > (gray.size * 0.1):  # At least 10% of frame
                max_area = area
                ultrasound_contour = contour
        
        if ultrasound_contour is None:
            self.stdout.write("Using fallback cropping method")
            return self.fallback_crop(frames)
        
        # Get bounding box of ultrasound region
        x, y, w, h = cv2.boundingRect(ultrasound_contour)
        
        # Expand region slightly to ensure full capture
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        # Validate crop region
        if w <= 10 or h <= 10:
            self.stdout.write("Invalid crop region detected, using fallback")
            return self.fallback_crop(frames)
        
        crop_region = (x, y, w, h)
        self.stdout.write(f"Detected ultrasound window: {crop_region}")
        
        # Crop all frames to this region
        cropped_frames = []
        for frame in frames:
            if frame.ndim == 3:
                cropped = frame[y:y+h, x:x+w, :]
            else:
                cropped = frame[y:y+h, x:x+w]
            cropped_frames.append(cropped)
            
        return cropped_frames, crop_region

    def fallback_crop(self, frames):
        """Fallback cropping method using intensity profiling"""
        if not frames:
            return [], None
            
        first_frame = frames[0]
        if first_frame.ndim == 3:
            gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = first_frame
        
        # Create vertical and horizontal profiles
        vert_profile = np.mean(gray, axis=1)
        horiz_profile = np.mean(gray, axis=0)
        
        # Find ultrasound window boundaries
        v_threshold = np.max(vert_profile) * 0.1
        vert_mask = vert_profile > v_threshold
        if np.any(vert_mask):
            ymin = np.argmax(vert_mask)
            ymax = len(vert_mask) - np.argmax(vert_mask[::-1])
        else:
            ymin, ymax = 0, gray.shape[0]
        
        h_threshold = np.max(horiz_profile) * 0.1
        horiz_mask = horiz_profile > h_threshold
        if np.any(horiz_mask):
            xmin = np.argmax(horiz_mask)
            xmax = len(horiz_mask) - np.argmax(horiz_mask[::-1])
        else:
            xmin, xmax = 0, gray.shape[1]
        
        # Apply padding
        padding = 5
        ymin = max(0, ymin - padding)
        ymax = min(gray.shape[0], ymax + padding)
        xmin = max(0, xmin - padding)
        xmax = min(gray.shape[1], xmax + padding)
        
        # Validate region
        if ymax <= ymin or xmax <= xmin:
            ymin, ymax = 0, gray.shape[0]
            xmin, xmax = 0, gray.shape[1]
        
        crop_region = (xmin, ymin, xmax - xmin, ymax - ymin)
        self.stdout.write(f"Fallback ultrasound window: {crop_region}")
        
        # Crop all frames
        cropped_frames = []
        for frame in frames:
            if frame.ndim == 3:
                cropped = frame[ymin:ymax, xmin:xmax, :]
            else:
                cropped = frame[ymin:ymax, xmin:xmax]
            cropped_frames.append(cropped)
            
        return cropped_frames, crop_region

    def extract_frames(self, video_path, temp_dir, num_frames=30):
        """Extract frames from video"""
        # Get total frame count
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames', 
            '-of', 'default=nokey=1:noprint_wrappers=1', video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            total_frames = int(result.stdout.strip())
        except:
            total_frames = 100  # Default fallback
            
        start_frame = max(0, total_frames - num_frames)
        
        # Extract frames
        cmd = [
            'ffmpeg', '-i', video_path, 
            '-vf', f'select=gte(n\\,{start_frame})',
            '-vframes', str(num_frames),
            '-vsync', '0',
            os.path.join(temp_dir, 'frame_%04d.png')
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        frames = []
        for frame_file in sorted(os.listdir(temp_dir)):
            if frame_file.endswith('.png'):
                frame_path = os.path.join(temp_dir, frame_file)
                try:
                    img = Image.open(frame_path)
                    # Convert to RGB to ensure consistent processing
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    elif img.mode not in ['RGB', 'L']:
                        img = img.convert('RGB')
                    frames.append(np.array(img))
                except Exception as e:
                    self.stderr.write(f"Error loading frame {frame_file}: {str(e)}")
                
        return frames

    def create_ultrasound_panorama(self, frames, video_id):
        """Create a panorama using feature-based stitching"""
        if len(frames) < 2:
            return None
            
        # Initialize with the first frame
        panorama = frames[0]
        height, width = panorama.shape[:2]
        
        # Create feature detector (use ORB for efficiency)
        detector = cv2.ORB_create(1000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Prepare panorama with extra space on the right
        extra_width = width * (len(frames) - 1) // 2
        panorama = np.zeros((height, width + extra_width, 3) if frames[0].ndim == 3 
                            else (height, width + extra_width), dtype=np.uint8)
        panorama[:, :width] = frames[0]
        
        # Initialize current position
        x_offset = width
        y_offset = 0
        
        for i in range(1, len(frames)):
            frame = frames[i]
            if frame.ndim == 2:
                prev_gray = panorama[:height, :x_offset]
                curr_gray = frame
            else:
                prev_gray = cv2.cvtColor(panorama[:height, :x_offset], cv2.COLOR_RGB2GRAY)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Detect keypoints and descriptors
            kp1, des1 = detector.detectAndCompute(prev_gray, None)
            kp2, des2 = detector.detectAndCompute(curr_gray, None)
            
            if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
                # Match features
                matches = matcher.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                
                if len(matches) > 10:
                    # Extract matching points
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    # Find homography matrix
                    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    
                    if H is not None:
                        # Warp current frame
                        h, w = frame.shape[:2]
                        warped = cv2.warpPerspective(frame, H, (panorama.shape[1], panorama.shape[0]))
                        
                        # Blend with panorama
                        mask = (panorama == 0).astype(np.uint8)
                        panorama = panorama * (1 - mask) + warped * mask
                        panorama = np.uint8(panorama)
                        continue
        
            # Fallback: simple translation alignment
            try:
                # Calculate vertical shift using phase correlation
                prev_region = prev_gray[:, -50:]  # Last 50 pixels
                curr_region = curr_gray[:, :50]   # First 50 pixels
                
                # Only align if regions are valid
                if prev_region.size > 0 and curr_region.size > 0:
                    shift, _ = cv2.phaseCorrelate(prev_region.astype(np.float32), 
                                                 curr_region.astype(np.float32))
                    dy = int(round(shift[1]))
                    
                    # Apply vertical shift
                    if abs(dy) > 0:
                        frame = self.apply_vertical_shift(frame, dy)
            except:
                dy = 0
            
            # Place frame with blending
            self.blend_frame(panorama, frame, x_offset, y_offset + dy)
            x_offset += frame.shape[1] // 2
        
        # Crop to content
        panorama = self.crop_to_content(panorama)
        
        # Convert to PIL image
        panorama_img = Image.fromarray(panorama)
        
        # Add ID text
        self.add_watermark(panorama_img, video_id)
        
        return panorama_img

    def blend_frame(self, panorama, frame, x_offset, y_offset):
        """Blend frame into panorama with feathering"""
        h, w = frame.shape[:2]
        ph, pw = panorama.shape[:2]
        
        # Calculate overlap region
        x_start = max(0, x_offset)
        x_end = min(pw, x_offset + w)
        y_start = max(0, y_offset)
        y_end = min(ph, y_offset + h)
        
        # Calculate frame region
        frame_x1 = max(0, -x_offset)
        frame_x2 = min(w, pw - x_offset)
        frame_y1 = max(0, -y_offset)
        frame_y2 = min(h, ph - y_offset)
        
        # Skip if no overlap
        if (x_end <= x_start or y_end <= y_start or 
            frame_x2 <= frame_x1 or frame_y2 <= frame_y1):
            return
            
        # Extract regions
        panorama_region = panorama[y_start:y_end, x_start:x_end]
        frame_region = frame[frame_y1:frame_y2, frame_x1:frame_x2]
        
        # Create blend mask
        blend_width = min(50, frame_x2 - frame_x1)
        blend_mask = np.ones(frame_region.shape[:2], dtype=np.float32)
        
        if blend_width > 0:
            ramp = np.linspace(0, 1, blend_width)
            blend_mask[:, :blend_width] = ramp[:, np.newaxis]
        
        # Blend
        if panorama_region.ndim == 3 and frame_region.ndim == 3:
            for c in range(3):
                panorama_region[:, :, c] = (
                    panorama_region[:, :, c] * (1 - blend_mask) + 
                    frame_region[:, :, c] * blend_mask
                )
        elif panorama_region.ndim == 2 and frame_region.ndim == 2:
            panorama_region[:, :] = (
                panorama_region * (1 - blend_mask) + 
                frame_region * blend_mask
            )
    
    def apply_vertical_shift(self, frame, dy):
        """Apply vertical shift to frame with zero-padding"""
        if abs(dy) < 1:
            return frame
            
        dy = int(round(dy))
        if frame.ndim == 3:
            height, width, channels = frame.shape
        else:
            height, width = frame.shape
            channels = 1
            
        # Create new frame with padding
        new_height = height + abs(dy)
        if dy > 0:
            # Shift downward
            if channels > 1:
                new_frame = np.zeros((new_height, width, channels), dtype=frame.dtype)
                new_frame[dy:, :, :] = frame
            else:
                new_frame = np.zeros((new_height, width), dtype=frame.dtype)
                new_frame[dy:, :] = frame
        else:
            # Shift upward
            if channels > 1:
                new_frame = np.zeros((new_height, width, channels), dtype=frame.dtype)
                new_frame[:height, :, :] = frame
            else:
                new_frame = np.zeros((new_height, width), dtype=frame.dtype)
                new_frame[:height, :] = frame
                
        return new_frame

    def crop_to_content(self, image):
        """Crop image to non-black content"""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
            
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add 5% margin
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(image.shape[1] - x, w + 2 * margin_x)
        h = min(image.shape[0] - y, h + 2 * margin_y)
        
        return image[y:y+h, x:x+w]

    def enhance_ultrasound(self, img_np):
        """Advanced ultrasound-specific enhancement"""
        try:
            # Convert to LAB color space
            if img_np.ndim == 3:
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L-channel
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l_enhanced = clahe.apply(l)
                
                # Merge channels
                lab_enhanced = cv2.merge([l_enhanced, a, b])
                enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            else:
                # Grayscale enhancement
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(img_np)
            
            return enhanced
        except:
            return img_np

    def add_watermark(self, img_pil, video_id):
        """Add visible ID text with timestamp"""
        try:
            # Create drawing context
            draw = ImageDraw.Draw(img_pil)
            
            # Use large, readable font
            font_size = max(20, int(img_pil.height * 0.03))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            # Create text with ID and timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            text = f"ID: {video_id} | {timestamp}"
            
            # Position at bottom center
            text_width = draw.textlength(text, font=font)
            position = ((img_pil.width - text_width) // 2, 
                        img_pil.height - font_size - 20)
            
            # Add background for readability
            bg_box = (
                position[0] - 10,
                position[1] - 5,
                position[0] + text_width + 10,
                position[1] + font_size + 5
            )
            draw.rectangle(bg_box, fill="black")
            
            # Add text
            draw.text(position, text, fill="white", font=font)
            
            return img_pil
        except Exception as e:
            self.stderr.write(f"Error adding watermark: {str(e)}")
            return img_pil