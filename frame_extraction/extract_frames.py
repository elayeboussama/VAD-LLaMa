import cv2
import os
import glob
from tqdm import tqdm

def extract_frames(video_path, save_dir, filename_tmpl='img_{:08d}.jpg'):
    """Extract frames from a video.
    
    Args:
        video_path (str): Path to video file
        save_dir (str): Directory to save frames
        filename_tmpl (str): Template for frame filenames
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame
        save_path = os.path.join(save_dir, filename_tmpl.format(frame_idx))
        cv2.imwrite(save_path, frame)
        
        frame_idx += 1
        
    cap.release()
    return frame_idx

def process_videos(video_dir, save_dir, filename_tmpl='img_{:08d}.jpg'):
    """Process all videos in a directory.
    
    Args:
        video_dir (str): Directory containing videos
        save_dir (str): Directory to save frames
        filename_tmpl (str): Template for frame filenames
    """
    video_list_path = os.path.join(video_dir, 'annotations.txt')
    with open(video_list_path, 'w') as video_list_file:
        # Process abnormal videos
        abnormal_dir = os.path.join(video_dir, 'abnormal')
        abnormal_categories = [d for d in os.listdir(abnormal_dir) if os.path.isdir(os.path.join(abnormal_dir, d))]
        
        for category in abnormal_categories:
            category_path = os.path.join(abnormal_dir, category)
            video_files = glob.glob(os.path.join(category_path, '*.mp4'))
            
            for video_path in tqdm(video_files, desc=f"Processing {category}"):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                save_path = os.path.join(save_dir, category, video_name)
                
                # Extract frames
                n_frames = extract_frames(video_path, save_path, filename_tmpl)
                print(f"Extracted {n_frames} frames from {video_path}")
                video_list_file.write(f"{filename_tmpl}\n")

        # Process normal videos
        normal_dir = os.path.join(video_dir, 'normal')
        video_files = glob.glob(os.path.join(normal_dir, '*.mp4'))
        
        for video_path in tqdm(video_files, desc="Processing normal videos"):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            save_path = os.path.join(save_dir, 'Training_Normal_Videos_Anomaly', video_name)
            
            # Extract frames
            n_frames = extract_frames(video_path, save_path, filename_tmpl)
            print(f"Extracted {n_frames} frames from {video_path}")
            video_list_file.write(f"{video_path}\n")

if __name__ == '__main__':
    # Configure these paths according to your setup
    video_dir = 'data/ucf'  # Directory containing your videos
    save_dir = 'data/ucf/frames'     # Directory to save extracted frames
    
    process_videos(video_dir, save_dir)
