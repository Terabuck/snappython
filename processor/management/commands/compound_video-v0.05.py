# /data/data/com.termux/files/home/dev/snappython/processor/management/commands/compound_videos.py

import os
import cv2
import numpy as np
import subprocess
import shutil
import traceback
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops, ImageDraw, ImageFont
from django.conf import settings
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Creates spatially compounded ultrasound images from videos'

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
            frames = self.extract_frames(mp4_path, temp_dir, num_frames=50)
            
            if not frames:
                raise ValueError("No frames extracted")
                
            # Detect and crop ultrasound window from each frame
            cropped_frames, crop_region = self.detect_and_crop_ultrasound(frames)
            
            if not cropped_frames:
                raise ValueError("No frames after cropping")
                
            # Build compounded image with spatial compounding
            compound_image = self.build_spatial_compound(cropped_frames, filename)
            
            if compound_image is None:
                raise ValueError("Compound image creation failed")
                
            # Enhance final composite
            enhanced = self.enhance_unified(compound_image)
            
            if enhanced is None:
                self.stderr.write("Enhancement failed, saving unenhanced image")
                compound_image.save(output_path)
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

    def extract_frames(self, video_path, temp_dir, num_frames=50):
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

    def build_spatial_compound(self, frames, filename):
        """Build spatially compounded image using weighted averaging"""
        if not frames:
            return None
            
        # Get dimensions from first frame
        height, width = frames[0].shape[0], frames[0].shape[1]
        is_color = frames[0].ndim == 3
        
        # Create accumulator arrays
        if is_color:
            accumulator = np.zeros((height, width, 3), dtype=np.float32)
        else:
            accumulator = np.zeros((height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        
        # Process each frame
        for frame in frames:
            frame = frame.astype(np.float32)
            
            # Create ultrasound content mask (ignore black background)
            if is_color:
                gray = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = frame
            mask = (gray > 20).astype(np.float32)
            
            # Add to accumulator
            if is_color:
                accumulator += frame * mask[..., None]
            else:
                accumulator += frame * mask
            weights += mask
        
        # Normalize composite
        weights[weights == 0] = 1e-10  # Avoid division by zero
        if is_color:
            composite = accumulator / weights[..., None]
        else:
            composite = accumulator / weights
        
        # Convert to uint8
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        img = Image.fromarray(composite)
        
        # Add filename as text annotation
        try:
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            draw.text((10, 10), f"File: {filename}", fill="white", font=font)
        except:
            self.stderr.write("Failed to add filename annotation")
        
        return img

    def enhance_unified(self, img_pil):
        """Apply unified enhancement pipeline with error handling"""
        if img_pil is None:
            return None
            
        try:
            # Create a copy for surface line detection
            original = img_pil.copy()
            
            # Convert to grayscale for processing
            if img_pil.mode == 'RGB':
                gray = img_pil.convert('L')
            else:
                gray = img_pil
                
            # Detect surface lines (dotted lines at tissue interfaces)
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_mask = edges.point(lambda x: 0 if x < 50 else 255)
            
            # Ultrasound-specific enhancement
            smooth = gray.filter(ImageFilter.GaussianBlur(radius=0.8))
            
            # Extract detail layers
            medium_detail = ImageChops.difference(gray, smooth)
            medium_detail = medium_detail.point(lambda x: min(x * 1.8, 255))
            fine_detail = ImageChops.difference(gray, gray.filter(ImageFilter.GaussianBlur(radius=0.5)))
            fine_detail = fine_detail.point(lambda x: min(x * 2.5, 255))
            
            # Combine details
            base = ImageChops.add(gray, medium_detail, scale=1.5)
            base = ImageChops.add(base, fine_detail, scale=1.5)
            
            # Contrast enhancement
            base = ImageOps.autocontrast(base, cutoff=1.0)  # Less aggressive contrast
            
            # Final sharpening (mild)
            base = base.filter(ImageFilter.UnsharpMask(
                radius=0.8, 
                percent=120, 
                threshold=3
            ))
            
            # Enhance surface lines by combining with edge mask
            base = ImageChops.screen(base, edge_mask)
            
            # For color images, combine enhanced luminance with original color
            if img_pil.mode == 'RGB':
                enhanced_rgb = base.convert('RGB')
                # Blend to preserve some original color information
                result = Image.blend(original, enhanced_rgb, alpha=0.5)
            else:
                result = base
                
            return result
            
        except Exception as e:
            self.stderr.write(f"Enhancement error: {str(e)}")
            return img_pil  # Return original if enhancement fails