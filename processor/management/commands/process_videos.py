# /data/data/com.termux/files/home/dev/snappython/processor/management/commands/process_videos.py

import os
import numpy as np
import subprocess
import shutil
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops
from django.conf import settings
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Processes ultrasound videos by aligning and enhancing the last 15 frames'

    def handle(self, *args, **options):
        # Create directories if they don't exist
        os.makedirs(settings.PROCESSED_VIDEOS_DIR, exist_ok=True)
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
        os.makedirs(settings.TEMP_FRAMES_DIR, exist_ok=True)

        processed = 0
        for filename in os.listdir(settings.INPUT_DIR):
            if filename.lower().endswith('.mp4'):
                self.stdout.write(f"Processing {filename}...")
                try:
                    self.process_video(filename)
                    processed += 1
                except Exception as e:
                    self.stderr.write(f"Error processing {filename}: {str(e)}")
        
        if processed:
            self.stdout.write(self.style.SUCCESS(f"Successfully processed {processed} videos"))
        else:
            self.stdout.write("No MP4 files found in input directory")

    def process_video(self, filename):
        # Setup paths
        mp4_path = os.path.join(settings.INPUT_DIR, filename)
        video_id = os.path.splitext(filename)[0]
        temp_dir = os.path.join(settings.TEMP_FRAMES_DIR, video_id)
        output_path_base = os.path.join(settings.OUTPUT_DIR, video_id)
        
        # Create temp directory for frames
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Extract frames
            frames, is_color = self.extract_frames(mp4_path, temp_dir, num_frames=15)
            
            if not frames:
                raise ValueError("No frames extracted")
            
            # Detect ultrasound region
            ultrasound_region = self.detect_ultrasound_region(frames)
            
            if ultrasound_region is None:
                self.stdout.write("Could not detect ultrasound region, processing full frame")
                # Fallback to full frame processing
                aligned = self.align_frames(frames, is_color)
                base_image = self.enhance_image(aligned, is_color)
                self.generate_comparisons(base_image, output_path_base)
            else:
                # Process ultrasound region separately
                self.stdout.write(f"Detected ultrasound region: {ultrasound_region}")
                processed_ultrasound = self.process_ultrasound_region(frames, ultrasound_region, is_color)
                
                # Create composite with static overlay
                base_frame = Image.fromarray(frames[0])
                base_image = self.create_composite_image(base_frame, processed_ultrasound, ultrasound_region)
                self.generate_comparisons(base_image, output_path_base)
            
            # Move processed video
            shutil.move(mp4_path, os.path.join(settings.PROCESSED_VIDEOS_DIR, filename))

        finally:
            # Clean up temporary frames
            shutil.rmtree(temp_dir, ignore_errors=True)

    def detect_ultrasound_region(self, frames, min_change_threshold=5):
        """Detect ultrasound region using frame differences without OpenCV"""
        if len(frames) < 2:
            return None
            
        # Convert to grayscale for analysis
        gray_frames = [self.to_grayscale(frame) for frame in frames]
        
        # Create a difference accumulator
        diff_accumulator = np.zeros_like(gray_frames[0], dtype=np.float32)
        
        # Compare each frame to the first frame
        for frame in gray_frames[1:]:
            diff = np.abs(frame.astype(np.float32) - gray_frames[0].astype(np.float32))
            diff_accumulator += diff
            
        # Normalize and threshold
        diff_accumulator /= len(frames) - 1
        threshold = np.percentile(diff_accumulator, 25)  # Only consider top 75% differences
        change_mask = diff_accumulator > threshold
        
        # Find bounding box of changed region
        rows = np.any(change_mask, axis=1)
        cols = np.any(change_mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, 0)
        xmin, xmax = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, 0)
        
        # Check if we found a valid region
        if ymax <= ymin or xmax <= xmin:
            return None
            
        # Add padding (10% of width/height)
        width = xmax - xmin
        height = ymax - ymin
        pad_x = int(width * 0.1)
        pad_y = int(height * 0.1)
        xmin = max(0, xmin - pad_x)
        ymin = max(0, ymin - pad_y)
        xmax = min(frames[0].shape[1], xmax + pad_x - 160)
        ymax = min(frames[0].shape[0], ymax + pad_y)
        
        return (xmin, ymin, xmax - xmin, ymax - ymin)

    def process_ultrasound_region(self, frames, region, is_color):
        """Process only the ultrasound region"""
        x, y, w, h = region
        ultrasound_frames = [frame[y:y+h, x:x+w] for frame in frames]
        
        # Align and enhance the ultrasound region
        aligned = self.align_frames(ultrasound_frames, is_color)
        enhanced = self.enhance_image(aligned, is_color)
        
        return enhanced

    def create_composite_image(self, base_frame, processed_ultrasound, region):
        """Combine processed ultrasound with static overlay from base frame"""
        x, y, w, h = region
        # Convert to same mode if needed
        if base_frame.mode != processed_ultrasound.mode:
            processed_ultrasound = processed_ultrasound.convert(base_frame.mode)
        
        # Create copy of base frame and paste processed ultrasound
        composite = base_frame.copy()
        composite.paste(processed_ultrasound, (x, y))
        return composite

    def extract_frames(self, video_path, temp_dir, num_frames=15):
        """Extract the last 'num_frames' frames with color detection"""
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
            # Fallback if frame count unavailable
            self.stderr.write("Couldn't get frame count, using fallback method")
            total_frames = 50
        
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
        
        # Load frames and detect color
        frames = []
        has_color = False
        for frame_file in sorted(os.listdir(temp_dir)):
            if frame_file.endswith('.png'):
                frame_path = os.path.join(temp_dir, frame_file)
                img = Image.open(frame_path)
                frames.append(np.array(img))
                
                # Improved color detection for Doppler
                if not has_color and img.mode == 'RGB':
                    rgb = np.array(img)
                    
                    # 1. Check overall color difference
                    rg_diff = np.abs(rgb[...,0].astype(np.float32) - rgb[...,1].astype(np.float32))
                    rb_diff = np.abs(rgb[...,0].astype(np.float32) - rgb[...,2].astype(np.float32))
                    
                    # 2. Doppler-specific check: look for saturated color pixels
                    # - Red Doppler: High R, low G/B
                    red_doppler = (rgb[...,0] > 200) & (rgb[...,1] < 50) & (rgb[...,2] < 50)
                    # - Blue Doppler: High B, low R/G
                    blue_doppler = (rgb[...,2] > 200) & (rgb[...,0] < 50) & (rgb[...,1] < 50)
                    
                    # 3. Combined detection
                    if (np.mean(rg_diff) > 1.5 or 
                        np.mean(rb_diff) > 1.5 or 
                        np.any(red_doppler) or 
                        np.any(blue_doppler)):
                        has_color = True
                        self.stdout.write(f"Color Doppler detected in {frame_file}")
        
        return frames, has_color

    def align_frames(self, frames, is_color):
        """Align frames using FFT-based phase correlation on central region"""
        if len(frames) < 2:
            return frames  # No alignment needed for single frame
            
        # Create grayscale versions for alignment
        ref_gray = self.to_grayscale(frames[0])
        aligned = [frames[0]]
        
        # Calculate central region coordinates (200x200 pixels)
        height, width = ref_gray.shape
        cx, cy = width//2, height//2
        crop_size = 200
        y1, y2 = max(0, cy - crop_size//2), min(height, cy + crop_size//2)
        x1, x2 = max(0, cx - crop_size//2), min(width, cx + crop_size//2)
        
        # Extract central region from reference
        ref_center = ref_gray[y1:y2, x1:x2]
        
        for frame in frames[1:]:
            try:
                # Compute phase correlation on central regions
                frame_gray = self.to_grayscale(frame)
                frame_center = frame_gray[y1:y2, x1:x2]
                shift = self.phase_correlation(ref_center, frame_center)
                
                # Apply shift with bounds checking
                aligned_frame = self.apply_shift(frame, shift)
                aligned.append(aligned_frame)
            except Exception as e:
                self.stderr.write(f"Alignment error: {e}, using original frame")
                aligned.append(frame)  # Fallback to original
                
        return aligned

    def to_grayscale(self, frame):
        """Convert frame to grayscale for alignment"""
        if frame.ndim == 2:
            return frame
        return np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    def phase_correlation(self, ref, target):
        """Pure numpy implementation of phase correlation"""
        # Compute FFTs
        fft_ref = np.fft.fft2(ref)
        fft_target = np.fft.fft2(target)
        
        # Compute cross-power spectrum
        cross_power = (fft_ref * np.conj(fft_target)) / (np.abs(fft_ref * np.conj(fft_target)) + 1e-10)
        cross_power[np.isnan(cross_power)] = 0
        
        # Inverse FFT to get phase correlation
        pc = np.abs(np.fft.ifft2(cross_power))
        
        # Find peak location
        peak = np.unravel_index(np.argmax(pc), pc.shape)
        
        # Calculate shifts (accounting for FFT wrap-around)
        shifts = []
        for idx, size in zip(peak, ref.shape):
            if idx > size // 2:
                shifts.append(idx - size)
            else:
                shifts.append(idx)
                
        return shifts[0], shifts[1]  # (dy, dx)

    def apply_shift(self, frame, shift, max_shift=7):
        """Apply shift with bounds checking"""
        dy, dx = shift
        
        # Constrain shifts to reasonable values
        if abs(dy) > max_shift or abs(dx) > max_shift:
            return frame
        
        # Apply shift using vectorized operation
        return np.roll(frame, shift=(int(dy), int(dx)), axis=(0,1))

    def enhance_image(self, frames, is_color):
        """Unified processing pipeline for both B-mode and Color Doppler"""
        # Temporal averaging
        stack = np.stack(frames, axis=0)
        avg_frame = np.mean(stack, axis=0).astype(np.uint8)
        
        # Convert to PIL for enhancement
        img_pil = Image.fromarray(avg_frame, 'RGB' if avg_frame.ndim == 3 else 'L')
        
        # Unified enhancement pipeline
        return self.enhance_unified(img_pil)

    def enhance_unified(self, img_pil):
        """Single enhancement pipeline for all image types"""
        # Preserve original for final color combination
        original = img_pil.copy()
        
        # Convert to grayscale for processing
        if img_pil.mode == 'RGB':
            gray = img_pil.convert('L')
        else:
            gray = img_pil
            
        # 1. Ultrasound-specific enhancement
        # -- Mild blur to reduce noise while preserving edges
        smooth = gray.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # 2. Extract detail layers
        # -- Medium-scale details (tissue boundaries)
        medium_detail = ImageChops.difference(gray, smooth)
        medium_detail = medium_detail.point(lambda x: min(x * 1.8, 255))
        
        # -- Fine-scale details (texture)
        fine_detail = ImageChops.difference(gray, gray.filter(ImageFilter.GaussianBlur(radius=0.8)))
        fine_detail = fine_detail.point(lambda x: min(x * 2.5, 255))
        
        # 3. Combine detail layers
        base = ImageChops.add(gray, medium_detail, scale=1.5)
        base = ImageChops.add(base, fine_detail, scale=1.5)
        
        # 4. Contrast enhancement
        base = ImageOps.autocontrast(base, cutoff=0.5)
        
        # 5. Final sharpening
        base = base.filter(ImageFilter.UnsharpMask(
            radius=1.0, 
            percent=150, 
            threshold=2
        ))
        
        # 6. For color images, combine enhanced luminance with original color
        if img_pil.mode == 'RGB':
            # Convert enhanced grayscale to RGB
            enhanced_rgb = base.convert('RGB')
            
            # Blend with original color
            return Image.blend(original, enhanced_rgb, alpha=0.4)
        
        return base

    def generate_comparisons(self, base_image, output_path_base):
        """Generate multiple output images with different filters for comparison"""
        # Original enhanced image
        base_image.save(f"{output_path_base}_enhanced_base.png")