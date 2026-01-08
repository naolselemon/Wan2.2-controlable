
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def load_midas_model(device='cuda'):
    print("Loading MiDaS model...")
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    # Try moving to the requested device; if CUDA init fails, fall back to CPU.
    try:
        midas.to(device)
    except RuntimeError as e:
        print(f"Warning: failed to initialize device '{device}': {e}. Falling back to CPU.")
        device = 'cpu'
        midas.to(device)

    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    print(f"✓ MiDaS {model_type} loaded on {device}")
    return midas, transform, device

def extract_depth(frame, model, transform, device='cuda', target_size=(360, 640)):
    """Extract depth with spatial downsampling"""
    
    # Original depth extraction
    input_batch = transform(frame).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size,  # ← Resize to 360p instead of original
            mode="bicubic",
            align_corners=False,
        ).squeeze()
     
    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map = (depth_map * 255).astype(np.uint8)
    
    return depth_map

def extract_edges(frame):
    """Extract edges using Canny edge detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def extract_optical_flow(prev_frame, curr_frame):
    """Extract optical flow between two frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    
    # Visualize flow
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*prev_frame.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return flow, flow_vis

def get_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0

def process_single_segment(video_path, segment, fps, output_dir, midas_model, midas_transform, 
                           device='cuda', sample_rate=4, target_size=(360, 640)):
    """Process segment and extract control metrics - OPTIMIZED VERSION"""
    video_id = segment['video_id']
    segment_idx = segment['segment_id']
    start_frame = int(segment['start_sec'] * fps)
    end_frame = int(segment['end_sec'] * fps)
    num_frames = end_frame - start_frame
    
    output_path = Path(output_dir) / video_id
    output_path.mkdir(parents=True, exist_ok=True) 
    
    # Open video ONCE, outside the loop
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ❌ Failed to open video: {video_path}")
        return False
    
    # Collect all frames in memory-efficient arrays
    depth_maps = []
    edge_maps = []
    flow_maps = []
    
    prev_frame = None
    
    # Calculate which frames to process
    frames_to_process = list(range(0, num_frames, sample_rate))
    pbar = tqdm(total=len(frames_to_process), desc=f"Segment {segment_idx}", unit="frame")
    
    # Process ONLY sampled frames
    for frame_offset in frames_to_process:
        # Seek to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + frame_offset)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame once for all extractions
        frame_resized = cv2.resize(frame, target_size[::-1])  # (W, H) - OpenCV uses (W, H)
        
        
        depth_map = extract_depth(frame_resized, midas_model, midas_transform, device)
        depth_maps.append(depth_map)  
        
        # Extract edges - store as bool (8x smaller than uint8)
        edges = extract_edges(frame_resized)
        edge_maps.append(edges > 0)  # Convert to boolean
        
        # Extract optical flow
        if prev_frame is not None:
            flow, _ = extract_optical_flow(prev_frame, frame_resized)
            flow_maps.append(flow.astype(np.float16))  
        
        prev_frame = frame_resized.copy()
        pbar.update(1)
    
    pbar.close()
    cap.release() 
    
    # Save as compressed NPZ (all controls in ONE file per segment)
    np.savez_compressed(
        output_path / f"segment_{segment_idx}_controls.npz",
        depth=np.stack(depth_maps),
        edges=np.stack(edge_maps),
        flow=np.stack(flow_maps) if flow_maps else np.array([]),
        metadata=np.array([{
            'sample_rate': sample_rate,
            'target_size': target_size,
            'original_frames': num_frames
        }], dtype=object)
    )
    
    return True

def process_dataset(video_dir="data/panda70m/videos", segment_file="data/panda70m/segments_metadata.json", output_dir="data/panda70m/control_signals", sample_rate=4):
    with open(segment_file) as f:
        all_segments = json.load(f)

    segments_by_video = defaultdict(list)
    for seg in all_segments:
        segments_by_video[seg['video_id']].append(seg)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    midas_model, midas_transform, device = load_midas_model(device)

    for video_id, segments in segments_by_video.items():
        video_path = Path(video_dir) / f"{video_id}.mp4"
        if not video_path.exists():
            print(f"Skipping {video_id} - video not found")
            continue

        fps = get_fps(str(video_path))  # From helper above
        print(f"Processing {video_id} ({len(segments)} segments, FPS: {fps:.2f})")

        for seg in segments:
            process_single_segment(video_path, seg, fps, output_dir, midas_model, midas_transform, device=device, sample_rate=sample_rate)

if __name__ == "__main__":
    process_dataset()