# /data/data/com.termux/files/home/dev/snappython/processor/management/commands/compound_videos.py
# Claude https://www.yeschat.ai/app/chat/982f0e446a1d4d5a931a6b256b0cb8dd
# https://github.com/Terabuck/snappython
# Using Video-Panorama stitching approach from https://github.com/krutikabapat/Video-Panorama
import os
import cv2
import numpy as np
import subprocess
import shutil
import traceback
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops, ImageDraw, ImageFont
from django.conf import settings
from django.core.management.base import BaseCommand
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class Command(BaseCommand):
    help = 'Creates spatially compounded ultrasound images from videos using Video-Panorama stitching'

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
            # Extract frames with reasonable density for panorama stitching
            frames = self.extract_frames(mp4_path, temp_dir, num_frames=30)
            
            if not frames:
                raise ValueError("No frames extracted")
                
            # Detect and crop ultrasound window
            cropped_frames, crop_region = self.detect_and_crop_ultrasound(frames)
            
            if not cropped_frames:
                raise ValueError("No frames after cropping")
                
            # Video panorama stitching
            panorama = self.create_video_panorama(cropped_frames, video_id=video_id)
            
            if panorama is None:
                raise ValueError("Panorama creation failed")
                
            # Enhanced post-processing for ultrasound
            enhanced = self.ultrasound_enhance(panorama)
            
            if enhanced is None:
                self.stderr.write("Enhancement failed, saving unenhanced image")
                panorama.save(output_path)
            else:
                enhanced.save(output_path)
            
            shutil.move(mp4_path, os.path.join(settings.PROCESSED_VIDEOS_DIR, filename))

        except Exception as e:
            self.stderr.write(f"Error processing video: {str(e)}")
            raise
        finally:
            # Clean up temporary frames
            shutil.rmtree(temp_dir, ignore_errors=True)

    def detect_and_crop_ultrasound(self, frames):
        """Enhanced ultrasound window detection with fan-shaped region analysis"""
        if not frames:
            return [], None
            
        # Use multiple frames for robust detection
        detection_frames = frames[:min(5, len(frames))]
        
        best_crop = None
        best_score = -1
        
        for frame in detection_frames:
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            
            # Enhance contrast for better detection
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Multiple threshold approaches
            methods = [
                lambda x: cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                lambda x: cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
                lambda x: cv2.threshold(x, 25, 255, cv2.THRESH_BINARY)[1]
            ]
            
            for method in methods:
                try:
                    thresh = method(enhanced)
                    
                    # Morphological operations to clean up
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    
                    # Find contours
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area < gray.size * 0.15:  # Too small
                            continue
                            
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Validate ultrasound characteristics
                        aspect_ratio = w / h
                        if not (0.8 <= aspect_ratio <= 2.5):  # Typical ultrasound aspect ratios
                            continue
                            
                        # Check for fan-shaped characteristics
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 0
                        
                        # Score based on size, position, and shape
                        score = area * solidity
                        if y < gray.shape[0] * 0.1:  # Prefer regions not at very top
                            score *= 1.2
                        if 0.3 <= solidity <= 0.9:  # Good solidity for ultrasound
                            score *= 1.5
                            
                        if score > best_score:
                            best_score = score
                            best_crop = (x, y, w, h)
                            
                except Exception as e:
                    continue
        
        if best_crop is None:
            self.stdout.write("Using intelligent fallback cropping")
            return self.intelligent_fallback_crop(frames)
        
        # Expand and validate crop region
        x, y, w, h = best_crop
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frames[0].shape[1] - x, w + 2 * padding)
        h = min(frames[0].shape[0] - y, h + 2 * padding)
        
        crop_region = (x, y, w, h)
        self.stdout.write(f"Detected ultrasound window: {crop_region}")
        
        # Crop all frames
        cropped_frames = []
        for frame in frames:
            if frame.ndim == 3:
                cropped = frame[y:y+h, x:x+w, :]
            else:
                cropped = frame[y:y+h, x:x+w]
            cropped_frames.append(cropped)
            
        return cropped_frames, crop_region

    def intelligent_fallback_crop(self, frames):
        """Intelligent fallback using intensity profiles and edge detection"""
        if not frames:
            return [], None
            
        first_frame = frames[0]
        if first_frame.ndim == 3:
            gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = first_frame
        
        # Enhanced edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Combine intensity and edge information
        combined = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)
        
        # Create profiles with enhanced weighting
        vert_profile = np.mean(combined, axis=1)
        horiz_profile = np.mean(combined, axis=0)
        
        # Apply Gaussian smoothing to profiles
        vert_profile = ndimage.gaussian_filter1d(vert_profile, sigma=2)
        horiz_profile = ndimage.gaussian_filter1d(horiz_profile, sigma=2)
        
        # Dynamic thresholding
        v_threshold = np.percentile(vert_profile, 75)
        h_threshold = np.percentile(horiz_profile, 75)
        
        # Find boundaries with hysteresis
        vert_mask = vert_profile > v_threshold
        horiz_mask = horiz_profile > h_threshold
        
        # Get boundaries
        if np.any(vert_mask):
            vert_indices = np.where(vert_mask)[0]
            ymin, ymax = vert_indices[0], vert_indices[-1]
        else:
            ymin, ymax = 0, gray.shape[0]
            
        if np.any(horiz_mask):
            horiz_indices = np.where(horiz_mask)[0]
            xmin, xmax = horiz_indices[0], horiz_indices[-1]
        else:
            xmin, xmax = 0, gray.shape[1]
        
        # Apply reasonable padding
        padding = 15
        ymin = max(0, ymin - padding)
        ymax = min(gray.shape[0], ymax + padding)
        xmin = max(0, xmin - padding)
        xmax = min(gray.shape[1], xmax + padding)
        
        crop_region = (xmin, ymin, xmax - xmin, ymax - ymin)
        self.stdout.write(f"Intelligent fallback crop: {crop_region}")
        
        # Crop all frames
        cropped_frames = []
        for frame in frames:
            if frame.ndim == 3:
                cropped = frame[ymin:ymax, xmin:xmax, :]
            else:
                cropped = frame[ymin:ymax, xmin:xmax]
            cropped_frames.append(cropped)
            
        return cropped_frames, crop_region

    def create_video_panorama(self, frames, video_id="Untitled"):
        """Create panorama using Video-Panorama approach with SIFT features and homography"""
        if not frames or len(frames) < 2:
            self.stderr.write("Not enough frames for panorama creation")
            return None
            
        self.stdout.write(f"Creating panorama from {len(frames)} frames using SIFT matching")
        
        # Convert frames to grayscale for feature detection
        gray_frames = []
        for frame in frames:
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            gray_frames.append(gray)
        
        # Initialize with first frame
        panorama = frames[0].copy()
        if panorama.ndim == 2:
            panorama = cv2.cvtColor(panorama, cv2.COLOR_GRAY2RGB)
        elif panorama.shape[2] == 4:  # RGBA
            panorama = cv2.cvtColor(panorama, cv2.COLOR_RGBA2RGB)
        
        # Initialize SIFT detector
        try:
            sift = cv2.SIFT_create(nfeatures=1000)
        except AttributeError:
            # Fallback for older OpenCV versions
            sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)
        
        # FLANN matcher parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        successful_stitches = 0
        
        for i in range(1, len(frames)):
            self.stdout.write(f"Stitching frame {i+1}/{len(frames)}")
            
            try:
                # Get current frame
                current_frame = frames[i].copy()
                if current_frame.ndim == 2:
                    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2RGB)
                elif current_frame.shape[2] == 4:  # RGBA
                    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGBA2RGB)
                
                # Convert panorama to grayscale for feature detection
                panorama_gray = cv2.cvtColor(panorama, cv2.COLOR_RGB2GRAY)
                current_gray = gray_frames[i]
                
                # Detect keypoints and descriptors
                kp1, des1 = sift.detectAndCompute(panorama_gray, None)
                kp2, des2 = sift.detectAndCompute(current_gray, None)
                
                if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                    self.stdout.write(f"Insufficient features in frame {i}, skipping")
                    continue
                
                # Match features
                matches = flann.knnMatch(des1, des2, k=2)
                
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                
                if len(good_matches) < 10:
                    self.stdout.write(f"Insufficient good matches ({len(good_matches)}) for frame {i}")
                    continue
                
                # Extract matched points
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                H, mask = cv2.findHomography(dst_pts, src_pts, 
                                           cv2.RANSAC, 
                                           ransacReprojThreshold=5.0,
                                           confidence=0.99,
                                           maxIters=2000)
                
                if H is None:
                    self.stdout.write(f"Could not find homography for frame {i}")
                    continue
                
                # Warp current frame
                h1, w1 = panorama.shape[:2]
                h2, w2 = current_frame.shape[:2]
                
                # Get corners of current frame
                corners_current = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
                corners_transformed = cv2.perspectiveTransform(corners_current, H)
                
                # Calculate output size
                all_corners = np.concatenate([
                    np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
                    corners_transformed
                ], axis=0)
                
                [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
                [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
                
                # Translation to keep everything positive
                translation_dist = [-x_min, -y_min]
                H_translation = np.array([[1, 0, translation_dist[0]], 
                                        [0, 1, translation_dist[1]], 
                                        [0, 0, 1]])
                
                # Warp images
                output_size = (x_max - x_min, y_max - y_min)
                panorama_warped = cv2.warpPerspective(panorama, H_translation, output_size)
                current_warped = cv2.warpPerspective(current_frame, H_translation.dot(H), output_size)
                
                # Create masks for blending
                mask_panorama = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
                mask_current = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
                
                # Fill masks
                mask_panorama[translation_dist[1]:translation_dist[1]+h1, 
                            translation_dist[0]:translation_dist[0]+w1] = 255
                
                # Create mask for current frame based on non-zero pixels
                current_gray_warped = cv2.cvtColor(current_warped, cv2.COLOR_RGB2GRAY)
                mask_current[current_gray_warped > 0] = 255
                
                # Find overlap region
                overlap = cv2.bitwise_and(mask_panorama, mask_current)
                
                # Blend images
                if np.any(overlap):
                    # Use feathering for smooth blending
                    overlap_region = overlap > 0
                    
                    # Create distance transforms for smooth blending
                    dist_panorama = cv2.distanceTransform(mask_panorama, cv2.DIST_L2, 5)
                    dist_current = cv2.distanceTransform(mask_current, cv2.DIST_L2, 5)
                    
                    # Normalize distances in overlap region
                    total_dist = dist_panorama + dist_current
                    alpha = np.zeros_like(dist_panorama)
                    alpha[total_dist > 0] = dist_panorama[total_dist > 0] / total_dist[total_dist > 0]
                    
                    # Apply blending
                    result = panorama_warped.copy()
                    for c in range(3):
                        blended_channel = (alpha[..., np.newaxis] * panorama_warped[..., c] + 
                                         (1 - alpha[..., np.newaxis]) * current_warped[..., c])
                        result[overlap_region, c] = blended_channel[overlap_region, 0]
                    
                    # Add non-overlapping regions
                    non_overlap_current = (mask_current > 0) & (mask_panorama == 0)
                    result[non_overlap_current] = current_warped[non_overlap_current]
                    
                    panorama = result
                else:
                    # Simple combination if no overlap
                    combined_mask = cv2.bitwise_or(mask_panorama, mask_current)
                    result = np.zeros_like(panorama_warped)
                    result[mask_panorama > 0] = panorama_warped[mask_panorama > 0]
                    result[mask_current > 0] = current_warped[mask_current > 0]
                    panorama = result
                
                successful_stitches += 1
                self.stdout.write(f"Successfully stitched frame {i}")
                
            except Exception as e:
                self.stderr.write(f"Error stitching frame {i}: {str(e)}")
                continue
        
        if successful_stitches == 0:
            self.stderr.write("No frames could be stitched, returning first frame")
            result_img = Image.fromarray(frames[0])
        else:
            self.stdout.write(f"Successfully stitched {successful_stitches} frames")
            
            # Convert final result
            if panorama.dtype != np.uint8:
                panorama = np.clip(panorama, 0, 255).astype(np.uint8)
            
            # Convert to PIL Image
            if panorama.ndim == 3:
                result_img = Image.fromarray(panorama, mode='RGB')
            else:
                result_img = Image.fromarray(panorama, mode='L')
        
        # Add metadata overlay
        result_img = self.add_metadata_overlay(result_img, video_id, len(frames))
        
        return result_img

    def add_metadata_overlay(self, img, video_id, frame_count):
        """Add metadata overlay to the panorama image"""
        try:
            # Convert to RGB for text overlay
            if img.mode == 'L':
                img = img.convert('RGB')
            
            draw = ImageDraw.Draw(img)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
                except:
                    font = ImageFont.load_default()
            
            # Create text with background
            text_lines = [
                f"ID: {video_id}",
                f"Frames: {frame_count}",
                f"Method: Video-Panorama SIFT Stitching"
            ]
            
            y_offset = 10
            for line in text_lines:
                # Calculate text size
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Draw background rectangle
                bg_box = (10, y_offset, 10 + text_width + 10, y_offset + text_height + 6)
                draw.rectangle(bg_box, fill="black")
                
                # Draw text
                draw.text((15, y_offset + 3), line, fill="white", font=font)
                
                y_offset += text_height + 10
                
        except Exception as e:
            self.stderr.write(f"Error adding metadata overlay: {str(e)}")
        
        return img

    def extract_frames(self, video_path, temp_dir, num_frames=30):
        """Extract frames from video with improved error handling and sampling"""
        # Get video info (total frames and duration)
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames,duration', 
            '-of', 'default=nokey=1:noprint_wrappers=1', video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        try:
            lines = result.stdout.strip().split('\
')
            total_frames = int(lines[0]) if lines[0].isdigit() else None
            duration = float(lines[1]) if len(lines) > 1 and lines[1].replace('.','').isdigit() else None
        except:
            total_frames = None
            duration = None
        
        # If we can't get frame count, estimate from duration
        if total_frames is None and duration is not None:
            # Assume 30 fps if we can't get exact frame count
            total_frames = int(duration * 30)
            self.stdout.write(f"Estimated {total_frames} frames from duration {duration}s")
        elif total_frames is None:
            total_frames = 100  # Final fallback
            self.stdout.write("Using fallback frame count of 100")
        else:
            self.stdout.write(f"Video has {total_frames} total frames")
        
        # Calculate uniform sampling across video
        if total_frames > num_frames:
            frame_step = total_frames // num_frames
            start_frame = max(0, (total_frames - num_frames * frame_step) // 2)
        else:
            frame_step = 1
            start_frame = 0
        
        # Extract frames with uniform sampling
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'select=not(mod(n\\\\,{frame_step}))',
            '-vframes', str(num_frames),
            '-vsync', '0', '-q:v', '2',
            os.path.join(temp_dir, 'frame_%05d.png')
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            self.stdout.write("Frame extraction successful")
        except subprocess.CalledProcessError:
            # Fallback extraction
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vframes', str(num_frames),
                '-q:v', '2',
                os.path.join(temp_dir, 'frame_%05d.png')
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                self.stdout.write("Fallback extraction successful")
            except:
                raise RuntimeError("All frame extraction methods failed")
        
        # Load extracted frames
        frames = []
        frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])
        
        if not frame_files:
            raise RuntimeError("No frames were extracted")
        
        self.stdout.write(f"Loading {len(frame_files)} extracted frames...")
        
        for frame_file in frame_files:
            frame_path = os.path.join(temp_dir, frame_file)
            try:
                img = Image.open(frame_path)
                
                # Ensure consistent color mode
                if img.mode == 'RGBA':
                    # Convert RGBA to RGB with white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                    img = background
                elif img.mode == 'P':
                    img = img.convert('RGB')
                elif img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                
                # Convert to numpy array
                frame_array = np.array(img)
                
                # Validate frame dimensions
                if frame_array.size == 0:
                    self.stderr.write(f"Empty frame: {frame_file}")
                    continue
                    
                frames.append(frame_array)
                
            except Exception as e:
                self.stderr.write(f"Error loading frame {frame_file}: {str(e)}")
                continue
        
        if not frames:
            raise RuntimeError("No valid frames could be loaded")
        
        self.stdout.write(f"Successfully loaded {len(frames)} frames")
        return frames

    def ultrasound_enhance(self, img_pil):
        """Specialized enhancement for ultrasound panorama images"""
        if img_pil is None:
            return None
            
        try:
            # Convert to grayscale if needed
            if img_pil.mode == 'RGB':
                gray = img_pil.convert('L')
            else:
                gray = img_pil
            
            # Convert to numpy for processing
            img_array = np.array(gray)
            
            # Apply CLAHE for local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
            enhanced = clahe.apply(img_array)
            
            # Noise reduction while preserving edges
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Edge enhancement
            edges = cv2.Laplacian(denoised, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
            
            # Combine original with edge enhancement
            alpha = 0.7
            combined = cv2.addWeighted(denoised, alpha, edges, 1-alpha, 0)
            
            # Gamma correction for ultrasound
            gamma = 1.2
            lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
            gamma_corrected = cv2.LUT(combined, lookup_table)
            
            # Convert back to PIL
            result = Image.fromarray(gamma_corrected, mode='L')
            
            # Additional PIL-based enhancements
            # Contrast enhancement
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(1.3)
            
            # Sharpening
            result = result.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
            
            return result
            
        except Exception as e:
            self.stderr.write(f"Enhancement error: {str(e)}")
            return img_pil