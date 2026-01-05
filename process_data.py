
import json
import ast
import cv2
import subprocess
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm

def time_to_seconds(time_str: str) -> float:
    """
    Convert a time string in format 'HH:MM:SS.mmm' or 'MM:SS.mmm' to total seconds.
    
    Args:
        time_str (str): The timestamp string to convert.
        
    Returns:
        float: Total seconds.
    """
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    else:
        m, s = parts
        return int(m) * 60 + float(s)

def get_fps(video_path: str) -> float:
    """
    Get the frame rate (FPS) of a video file.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        float: The FPS of the video. Returns 30.0 if FPS cannot be determined or is invalid.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0

def process_panda70m(target_videos=None, min_score=0.25):
    """
    Process the Panda-70M dataset (validation split) to extract video URLs and segment metadata.
    
    This function streams the dataset from Hugging Face, filters segments based on matching score 
    (though currently disabled in code), and collects unique video URLs until the target count 
    is reached. It saves metadata to 'data/panda70m/video_urls.json' and 'data/panda70m/segments_metadata.json'.
    
    Args:
        target_videos (int, optional): The target number of valid videos to collect. If None, continues indefinetly (or until dataset exhaustion).
        min_score (float, optional): The minimum matching score for a segment to be included. Defaults to 0.25.
        
    Returns:
        tuple: A tuple containing two lists:
            - all_videos (list): List of dictionaries containing video metadata (ID, URL, segment count).
            - all_segments (list): List of dictionaries containing segment details (start/end times, caption, score).
    """
    print("=" * 70)
    print("Processing Panda-70M Dataset")
    print("=" * 70)

    # Load from HF (partial) or local CSV/Parquet for full
    dataset = load_dataset("multimodalart/panda-70m", split="validation", streaming=True)

    all_videos = []
    all_segments = []
    video_url_map = {}  # video_id -> url

    output_dir = Path("data/panda70m")
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_rows = 0
    for example in tqdm(dataset):
        if target_videos and len(all_videos) >= target_videos:
            break

        video_id = example['videoID']
        url = example['url']

        video_url_map[video_id] = url

        timestamps = ast.literal_eval(example['timestamp'])  # list of [start, end]
        captions = ast.literal_eval(example['caption'])
        scores = ast.literal_eval(example['matching_score'])

        video_segments = []
        for i, (ts, cap, score) in enumerate(zip(timestamps, captions, scores)):

            if score < min_score:
                continue
            
            start_time, end_time = ts
            segment_info = {
                'segment_id': f"{video_id}_seg_{i:04d}",
                'video_id': video_id,
                'url': url,
                'start_time': start_time,
                'end_time': end_time,
                'start_sec': time_to_seconds(start_time),
                'end_sec': time_to_seconds(end_time),
                'caption': cap,
                'matching_score': score
            }
            video_segments.append(segment_info)
            all_segments.append(segment_info)

        if video_segments:
            all_videos.append({
                'video_id': video_id,
                'url': url,
                'num_segments': len(video_segments)
            })

        processed_rows += 1

    # Save metadata
    videos_file = output_dir / "video_urls.json"
    with open(videos_file, 'w') as f:
        json.dump(all_videos, f, indent=2)

    segments_file = output_dir / "segments_metadata.json"
    with open(segments_file, 'w') as f:
        json.dump(all_segments, f, indent=2, ensure_ascii=False)

    print(f"Processed {processed_rows} rows to find {len(all_videos)} valid videos")
    print(f"Saved {len(all_videos)} videos and {len(all_segments)} segments")

    return all_videos, all_segments

def download_videos(video_list, max_videos=10, output_dir="data/panda70m/videos"):
    """
    Download a list of videos using yt-dlp.
    
    This function takes a list of video metadata and downloads them to the specified directory.
    It handles checks for existing files, verifies yt-dlp installation, and attempts to merge
    best quality video and audio streams into an MP4 container.
    
    Args:
        video_list (list): List of dictionaries containing video metadata (must have 'url' and 'video_id').
        max_videos (int, optional): Maximum number of videos to download from the list. Defaults to 100.
        output_dir (str, optional): Directory to save downloaded videos. Defaults to "data/panda70m/videos".
        
    Returns:
        tuple: (successful_count, failed_count)
    """
    print("\n" + "=" * 70)
    print(f"Downloading Videos (max: {max_videos})")
    print("=" * 70)
    
    # Check if yt-dlp is installed
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("\n❌ yt-dlp is not installed!")
        print("\nPlease install it using:")
        print("  pip install yt-dlp")
        print("  # or")
        print("  conda install -c conda-forge yt-dlp")
        return 0, 0
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    skipped = 0
    
    for i, video in enumerate(video_list[:max_videos]):
        video_id = video['video_id']
        url = video['url']
        
        output_path = output_dir / f"{video_id}.mp4"
        
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ [{i+1}/{max_videos}] {video_id} already exists ({file_size:.1f} MB)")
            skipped += 1
            successful += 1 # Treating existing as success
            continue
        
        print(f"\n⬇️  [{i+1}/{max_videos}] Downloading {video_id}...")
        print(f"    URL: {url}")
        
        # yt-dlp command with quality settings - use direct command, not python -m
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            "--no-warnings",
            "--quiet",  # Suppress progress output
            "--progress",  # But show progress bar
            url
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per video
            )
            
            if result.returncode == 0 and output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                print(f"    ✓ Downloaded successfully ({file_size:.1f} MB)")
                successful += 1
            else:
                print(f"    ❌ Failed to download")
                if result.stderr:
                    # Show only first 200 chars of error
                    error_msg = result.stderr.strip()[:200]
                    print(f"    Error: {error_msg}")
                failed += 1
        
        except subprocess.TimeoutExpired:
            print(f"    ❌ Download timeout (>10 minutes)")
            failed += 1
        except Exception as e:
            print(f"    ❌ Error: {e}")
            failed += 1
    
    print(f"\n" + "=" * 70)
    print(f"Download Summary:")
    print(f"  ✓ Successful: {successful}")
    print(f"  ⏭️  Skipped (already exist): {skipped}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📁 Videos saved to: {output_dir}")
    print("=" * 70)
    
    return successful, failed

def main():
    """
    Main entry point for the script.
    
    Interactively asks the user for the number of videos to find and whether to proceed with downloading.
    """
    num = input("Target number of videos to find? (default: 100): ")
    target_videos = int(num) if num else 100

    videos, segments = process_panda70m(target_videos=target_videos)

    response = input("Download videos now? (yes/no): ").lower()
    if response in ['yes', 'y']:
        download_videos(videos, max_videos=len(videos))

if __name__ == "__main__":
    main()