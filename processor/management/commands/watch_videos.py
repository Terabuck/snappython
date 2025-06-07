# processor/management/commands/watch_videos.py
import os
import time
import shutil
import logging
from django.core.management.base import BaseCommand
from django.conf import settings
from django.core.management import call_command

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Watches for new ultrasound videos and triggers processing'
    
    def handle(self, *args, **options):
        # Load already processed videos
        processed = set()
        if os.path.exists(settings.PROCESSED_LOG):
            with open(settings.PROCESSED_LOG, 'r') as f:
                processed = set(f.read().splitlines())
        
        self.stdout.write("Starting video watcher...")
        self.stdout.write(f"Monitoring: {settings.INPUT_DIR_MY_USG}")
        
        while True:
            try:
                new_files = []
                for filename in os.listdir(settings.INPUT_DIR_MY_USG):
                    if (filename.lower().endswith('.mp4') 
                            and filename not in processed
                            and os.path.isfile(os.path.join(settings.INPUT_DIR_MY_USG, filename))):
                        new_files.append(filename)
                
                if new_files:
                    self.stdout.write(f"Found {len(new_files)} new video(s)")
                    
                    for filename in new_files:
                        try:
                            src = os.path.join(settings.INPUT_DIR_MY_USG, filename)
                            dest = os.path.join(settings.INPUT_DIR, filename)
                            
                            # Copy to processing directory
                            shutil.copy2(src, dest)
                            self.stdout.write(f"Copied {filename} to processing queue")
                            
                            # Trigger processing
                            call_command('process_videos')
                            
                            # Log processed file
                            processed.add(filename)
                            with open(settings.PROCESSED_LOG, 'a') as f:
                                f.write(f"{filename}\n")
                                
                            # Remove the temporary copy
                            os.remove(dest)
                            
                        except Exception as e:
                            logger.error(f"Error processing {filename}: {str(e)}")
                
                time.sleep(10)  # Check every 10 seconds
                
            except KeyboardInterrupt:
                self.stdout.write("Stopping video watcher...")
                break
            except Exception as e:
                logger.error(f"Watcher error: {str(e)}")
                time.sleep(30)