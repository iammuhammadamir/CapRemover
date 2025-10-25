import subprocess
import os

def compress_videos(propainter_output, diffueraser_output, mask_video, overlay_video=None, debug=False):
    """Compress videos using H.265 encoding if debug is True"""
    if not debug:
        return
    
    videos = [propainter_output, diffueraser_output, mask_video]
    if overlay_video is not None:
        videos.append(overlay_video)
    
    for video_path in videos:
        if os.path.exists(video_path):
            # Create temporary compressed file
            temp_path = video_path.replace('.mp4', '_temp.mp4')
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-c:v', 'libx265',
                '-crf', '23',
                '-preset', 'medium',
                '-y',  # Overwrite output file
                temp_path
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Replace original with compressed version
            os.replace(temp_path, video_path)
