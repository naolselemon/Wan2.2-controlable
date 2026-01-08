
import json
from collections import defaultdict

with open('data/panda70m/segments_metadata.json', 'r') as f:
    data = json.load(f)

videos = defaultdict(list)
for seg in data:
    videos[seg['video_id']].append(seg)

print("=" * 60)
for video_id, segs in list(videos.items())[:20]:  # Show first 20
    num_segs = len(segs)
    avg_dur = sum(s['end_sec'] - s['start_sec'] for s in segs) / num_segs
    avg_score = sum(s['matching_score'] for s in segs) / num_segs
    print(f"Video {video_id}: {num_segs} segments, avg dur {avg_dur:.1f}s, avg score {avg_score:.3f}")

print("\nOVERALL:")
print(f"Total unique videos: {len(videos)}")
print(f"Total segments: {len(data)}")
print(f"Avg segments per video: {len(data)/len(videos):.1f}")