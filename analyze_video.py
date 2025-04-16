import os
import json
import csv
import argparse
import cv2
import numpy as np
import yt_dlp
import tempfile # Needed for temporary file handling
import logging
from skimage.metrics import structural_similarity as ssim

# Import GCS utilities
import gcs_utils

# Configure logging (optional but helpful)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_VIDEO_DURATION_SECONDS = 600 # 10 min
MIN_TIME_BETWEEN_SCENE_CHANGE = 1 # eliminate scene changes less than x second due to grid effect
SCENE_CHANGE_SENSITIVITY = 0.8 # -1 < x < 1, higher = more sensitive to scene change

# --- Constants ---
# Define GCS paths (these can be prefixes/folders within your bucket)
GCS_VIDEOS_PREFIX = "videos/"
GCS_RESULTS_BLOB = "db/results.json"

# Function to convert seconds to mm:ss format (ex: 123 -> 2:03)
def s2mmss(seconds):
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"

# --- Scene Change Analysis Functions (can tweak threshold) ---
# histogram
def is_scene_change(prev_frame, current_frame, threshold=SCENE_CHANGE_SENSITIVITY):
    prev_hist = cv2.calcHist([prev_frame], [0, 1], None, [50, 60], [0, 180, 0, 256])
    current_hist = cv2.calcHist([current_frame], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(prev_hist, prev_hist)
    cv2.normalize(current_hist, current_hist)
    score = cv2.compareHist(prev_hist, current_hist, cv2.HISTCMP_CORREL)
    return score < threshold

# ssim
# def is_scene_change(prev_frame, current_frame, threshold=0.90):
#     gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#     gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
#     # Slight blur to suppress noise-based differences
#     gray_prev = cv2.GaussianBlur(gray_prev, (5, 5), 0)
#     gray_curr = cv2.GaussianBlur(gray_curr, (5, 5), 0)

#     score, _ = ssim(gray_prev, gray_curr, full=True)
#     return score < threshold

def get_url_key(youtube_url):
    # ... (keep existing implementation)
    if "v=" in youtube_url:
        return youtube_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_url:
        return youtube_url.split("youtu.be/")[1].split("?")[0]
    return youtube_url # Consider adding better error handling for invalid URLs

def crop_black_bars(frame, threshold=10, tolerance=0.99):
    # ... (keep existing implementation)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    left = 0
    for col in range(w):
        col_pixels = gray[:, col]
        if np.mean(col_pixels < threshold) < tolerance:
            break
        left += 1
    right = w
    for col in reversed(range(w)):
        col_pixels = gray[:, col]
        if np.mean(col_pixels < threshold) < tolerance:
            break
        right -= 1
    cropped = frame[:, left:right]
    return cropped

# --- Modified GCS-Aware Functions ---
def download_youtube_video_gcs_old(url, url_key, gcs_client, gcs_bucket):
    """
    Downloads video if not in GCS, uploads it, and returns GCS blob name and title.
    """
    gcs_blob_name = f"{GCS_VIDEOS_PREFIX}{url_key}.mp4"

    # 1. Check if video already exists in GCS
    if gcs_utils.blob_exists(gcs_bucket, gcs_blob_name):
        logger.info(f"Video already exists in GCS: {gcs_blob_name}")
        # Try to get title from existing results in GCS
        results_data = gcs_utils.download_json_blob(gcs_bucket, GCS_RESULTS_BLOB)
        # print(f"results_data: {results_data}")
        # print(f"url_key: {url_key}")
        for i, item in enumerate(results_data):
            if item['ytKey'] == url_key:
                # print(f"title: {results_data[url_key]['title']}")
                # print(f"title.get: {results_data[url_key].get('title', 'foobar')}")
                return gcs_blob_name, item['title']
                # return gcs_blob_name, results_data[url_key].get('title', 'Existing Video')
        return gcs_blob_name, "Existing Video (Title Unknown)"

    # 1b. Video not in GCS - Fetch metadata first to check duration
    logger.info(f"Video not found in GCS. Fetching metadata for {url}...")
    ydl_opts_meta = {'quiet': True, 'noplaylist': True} # Options for metadata fetch
    try:
        with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
            info = ydl.extract_info(url, download=False) # *** download=False ***
            duration = info.get('duration')
            title = info.get('title', 'Unknown Title') # Get title from metadata

            if duration is None:
                logger.warning(f"Could not extract duration for {url}. Proceeding with download cautiously.")
            elif duration > MAX_VIDEO_DURATION_SECONDS:
                error_message = f"Eek! Video '{title}' duration ({duration}s) exceeds the maximum allowed limit of {MAX_VIDEO_DURATION_SECONDS}s. Download aborted."
                logger.error(error_message)
                # Optionally, inform the Streamlit user more directly here if needed,
                # but returning None should be handled by app.py
                return None, "Video is too long (needs to be under 10 min)" # Indicate failure due to duration
            logger.info(f"Video '{title}' duration ({duration}s) is within limit. Proceeding with download.")

    except yt_dlp.utils.DownloadError as e:
         logger.error(f"yt-dlp error fetching metadata for {url}: {e}")
         # You might want to inform the user specifically about download errors
         return None, None # Indicate failure
    except Exception as e:
        logger.error(f"Unexpected error fetching metadata for {url}: {e}", exc_info=True)
        return None, None # Indicate failure

    # 2. Download locally to a temporary file
    logger.info(f"Video not found in GCS. Downloading {url}...")
    title = "Unknown Title"
    with tempfile.TemporaryDirectory() as tmpdir:
        local_temp_path_tmpl = os.path.join(tmpdir, f"{url_key}.%(ext)s")
        ydl_opts = {
            'outtmpl': local_temp_path_tmpl,
            'format': 'best[ext=mp4]', # Ensure mp4 format
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # Construct the actual downloaded filename (yt-dlp might change it slightly)
                downloaded_filename = ydl.prepare_filename(info)
                title = info.get('title', 'Unknown Title')

                # Check if download actually happened and file exists
                if not os.path.exists(downloaded_filename):
                     logger.error(f"yt-dlp reported success but file not found: {downloaded_filename}")
                     raise FileNotFoundError(f"Downloaded file missing: {downloaded_filename}")

                logger.info(f"Downloaded '{title}' to {downloaded_filename}")

                # 3. Upload the temporary file to GCS
                logger.info(f"Uploading {downloaded_filename} to GCS bucket {gcs_bucket.name} as {gcs_blob_name}...")
                if gcs_utils.upload_blob(gcs_bucket, downloaded_filename, gcs_blob_name):
                    logger.info("Upload successful.")
                    return gcs_blob_name, title
                else:
                    logger.error("Upload to GCS failed.")
                    # Decide how to handle upload failure - maybe raise an exception?
                    raise Exception(f"Failed to upload {downloaded_filename} to GCS.")

        except Exception as e:
            logger.error(f"Error during download/upload for {url}: {e}", exc_info=True)
            raise # Re-raise the exception to be caught by the caller

    # Should not reach here if successful or if an error occurred and was raised
    return None, title # Indicate failure

# return: gcs_blob_name, video title or error message, clip start, clip end, total video duration
def download_youtube_video_gcs(url, url_key, gcs_client, gcs_bucket, start_time=None, end_time=None):
    """
    Downloads video if not in GCS, uploads it, and returns GCS blob name, title, and optional clip range.
    If the video exceeds the max duration, a middle 10-minute clip is downloaded.
    """
    gcs_blob_name = f"{GCS_VIDEOS_PREFIX}{url_key}.mp4"
    clip_start = None
    clip_end = None

    if gcs_utils.blob_exists(gcs_bucket, gcs_blob_name):
        logger.info(f"Video already exists in GCS: {gcs_blob_name}")
        results_data = gcs_utils.download_json_blob(gcs_bucket, GCS_RESULTS_BLOB)
        for i, item in enumerate(results_data):
            if item['ytKey'] == url_key:
                clip_start = item['clipStart'] if item.get('clipStart') is not None else 0
                return gcs_blob_name, item['title'], clip_start, None, None
        return gcs_blob_name, "Existing Video (Title Unknown)", None, None, None # if video title unknown but video file exists

    logger.info(f"Video not found in GCS. Fetching metadata for {url}...")
    ydl_opts_meta = {'quiet': True, 'noplaylist': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get('duration')
            title = info.get('title', 'Unknown Title')

            if duration is None:
                logger.warning(f"Could not extract duration for {url}. Proceeding with download cautiously.")
            elif duration > MAX_VIDEO_DURATION_SECONDS:
                clip_duration = MAX_VIDEO_DURATION_SECONDS
                clip_start = (duration - clip_duration) // 2
                clip_end = clip_start + clip_duration
                logger.info(f"Video too long. Will clip from {clip_start}s to {clip_end}s.")
                start_time = clip_start
                end_time = clip_end

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp error fetching metadata for {url}: {e}")
        return None, None, None, None, None
    except Exception as e:
        logger.error(f"Unexpected error fetching metadata for {url}: {e}", exc_info=True)
        return None, None, None, None, None

    title = "Unknown Title"
    with tempfile.TemporaryDirectory() as tmpdir:
        local_temp_path_tmpl = os.path.join(tmpdir, f"{url_key}.%(ext)s")
        ydl_opts = {
            'outtmpl': local_temp_path_tmpl,
            'format': 'best[ext=mp4]'
        }

        if start_time is not None and end_time is not None:
            ydl_opts['download_ranges'] = {'ranges': [(start_time, end_time)]}

        try:
            # Add download range to ydl_opts only if clip_start and clip_end are defined
            if clip_start is not None and clip_end is not None:
                ydl_opts['download_ranges'] = lambda *_: [{"start_time": clip_start, "end_time": clip_end}]

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_filename = ydl.prepare_filename(info)
                title = info.get('title', 'Unknown Title')

                if not os.path.exists(downloaded_filename):
                    logger.error(f"yt-dlp reported success but file not found: {downloaded_filename}")
                    raise FileNotFoundError(f"Downloaded file missing: {downloaded_filename}")

                logger.info(f"Downloaded '{title}' to {downloaded_filename}")

                logger.info(f"Uploading {downloaded_filename} to GCS bucket {gcs_bucket.name} as {gcs_blob_name}...")
                if gcs_utils.upload_blob(gcs_bucket, downloaded_filename, gcs_blob_name):
                    logger.info("Upload successful.")
                    print("return 5")
                    return gcs_blob_name, title, clip_start, clip_end, duration
                else:
                    logger.error("Upload to GCS failed.")
                    raise Exception(f"Failed to upload {downloaded_filename} to GCS.")



        except Exception as e:
            logger.error(f"Error during download/upload for {url}: {e}", exc_info=True)
            raise

    return None, title, clip_start, clip_end, duration


def analyze_video_gcs(gcs_blob_name, gcs_client, gcs_bucket, clip_start):
    """
    Downloads video from GCS to a temporary file and analyzes it.
    """
    if not gcs_bucket or not gcs_blob_name:
         logger.error("GCS bucket or blob name not provided for analysis.")
         return None # Or raise error

    with tempfile.TemporaryDirectory() as tmpdir:
        local_temp_path = os.path.join(tmpdir, os.path.basename(gcs_blob_name)) # Use basename for temp file

        # 1. Download video from GCS
        logger.info(f"Retrieving {gcs_blob_name} from data store to {local_temp_path} for analysis...")
        if not gcs_utils.download_blob(gcs_bucket, gcs_blob_name, local_temp_path):
            logger.error(f"Failed to download {gcs_blob_name} from data store.")
            return None # Or raise error

        logger.info(f"Starting analysis for {local_temp_path}...")

        # 2. Analyze the local temporary video file (existing logic)
        cap = cv2.VideoCapture(local_temp_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file: {local_temp_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
             logger.warning(f"Video {local_temp_path} has invalid FPS ({fps}) or frame count ({frame_count}). Skipping analysis.")
             cap.release()
             return None # Cannot calculate duration etc.

        duration = frame_count / fps

        prev_frame, prev_gray = None, None
        scene_changes = 0
        scene_change_timestamps = []
        last_scene_change_in_seconds = 0
        saturation_values = []
        motion_scores = []
        edge_counts = []
        object_counts = []

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            try:
                frame_resized = cv2.resize(frame, (320, 180))
                frame_resized = crop_black_bars(frame_resized)
                if frame_resized is None or frame_resized.size == 0:
                    logger.warning(f"Frame {frame_num} ({s2mmss(frame_num/fps)}) resulted in empty frame after cropping, skipping.")
                    continue

                # Check if gray exists from previous iteration before shape comparison
                if prev_gray is not None and 'gray' in locals() and gray.shape != prev_gray.shape:
                    logger.warning(f"Frame {frame_num} ({s2mmss(frame_num/fps)}) shape mismatch after crop/resize, resetting prev_gray.")
                    prev_gray = None # Reset to avoid error, might slightly affect motion calc for this frame pair

                hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
                s = hsv[:, :, 1]
                # Handle potential empty 's' if frame is fully black/white after processing
                valid_s = s[s > 50]
                saturation_values.append(np.mean(valid_s) if valid_s.size > 0 else np.mean(s) if s.size > 0 else 0)

                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

                # Dynamism
                if prev_gray is not None and gray.shape == prev_gray.shape:
                    diff = cv2.absdiff(gray, prev_gray)
                    # Added check for diff size to prevent errors on empty diffs
                    if diff.size > 0:
                         motion_score = np.mean(diff**1.3) + np.std(diff**1.3)
                         motion_scores.append(motion_score)
                    else:
                         motion_scores.append(0) # Append 0 if diff is empty

                # Scene Change
                if prev_frame is not None:
                    # Ensure prev_frame has compatible dimensions before comparison
                    if prev_frame.shape == frame_resized.shape and prev_frame.size > 0 and frame_resized.size > 0:
                         if is_scene_change(prev_frame, frame_resized): 
                            curr_frame_in_seconds = frame_num /fps
                            # print(f"detect scene change: {s2mmss(curr_frame_in_seconds)}")
                            if curr_frame_in_seconds - last_scene_change_in_seconds >= MIN_TIME_BETWEEN_SCENE_CHANGE:
                                scene_changes += 1
                                last_scene_change_in_seconds = curr_frame_in_seconds
                                scene_change_timestamps.append(s2mmss(curr_frame_in_seconds + clip_start))
                    else:
                         logger.warning(f"Frame {frame_num} ({s2mmss(frame_num/fps)}) shape mismatch for scene change detection, skipping comparison.")


                # Visual Complexity (Edge Count)
                edges = cv2.Canny(gray, 100, 200)
                edge_counts.append(np.count_nonzero(edges))

                # Contour-based object approximation
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100] # Keep threshold relative to resized frame
                object_counts.append(len(large_contours))

                prev_frame = frame_resized.copy() # Use copy to avoid issues
                prev_gray = gray.copy()

            except cv2.error as e:
                logger.error(f"OpenCV error processing frame {frame_num}: {e}", exc_info=True)
                # Decide whether to continue or stop analysis
                continue # Skip this frame

        cap.release()
        logger.info(f"Finished analysis for {local_temp_path}")

        # 3. Calculate metrics (handle potential division by zero or empty lists)
        avg_saturation_raw = np.mean(saturation_values) if saturation_values else 0
        normalized_saturation = min((avg_saturation_raw / 200) * 100, 100) # Keep normalization

        avg_motion_raw = np.mean(motion_scores) if motion_scores else 0
        normalized_motion = min((avg_motion_raw / 200) * 100, 100) # Keep normalization

        avg_edge_count = np.mean(edge_counts) if edge_counts else 0
        # max_edge_count = np.max(edge_counts) if edge_counts else 0 # Max might not be needed
        # normalized_visual_complexity = min((avg_edge_count / 1000) * 100, 100) # Keep normalization

        avg_object_count = np.mean(object_counts) if object_counts else 0
        max_object_count = np.max(object_counts) if object_counts else 0

        scenes_per_minute = (scene_changes / (duration / 60)) if duration > 0 else 0
        avg_scene_duration = (duration / scene_changes) if scene_changes > 0 else duration

        print(scene_change_timestamps)

        return {
            'duration': duration,
            'scene_count': scene_changes,
            'scenes_per_minute': scenes_per_minute,
            'avg_scene_duration': avg_scene_duration,
            'avg_saturation': normalized_saturation,
            'motion_dynamism': normalized_motion,
            'avg_object_count': round(avg_object_count, 2),
            'max_object_count': int(max_object_count),
            'scene_change_timestamps': scene_change_timestamps
        }

# --- GCS Results Loading/Saving --- 

def load_results_gcs(gcs_client, gcs_bucket):
    """Loads results dictionary from results.json in GCS."""
    # logger.info(f"Loading results from GCS blob: {GCS_RESULTS_BLOB}")
    results = gcs_utils.download_json_blob(gcs_bucket, GCS_RESULTS_BLOB)
    if results is None:
        logger.error("Failed to download or parse results JSON from GCS.")
        return {} # Return empty dict on error or if not found
    # logger.info(f"Loaded {len(results)} results from GCS.")
    return results

def save_results_gcs(data, gcs_client, gcs_bucket):
    """Saves results dictionary to results.json in GCS."""
    # logger.info(f"Saving {len(data)} results to GCS blob: {GCS_RESULTS_BLOB}")
    success = gcs_utils.upload_json_blob(gcs_bucket, data, GCS_RESULTS_BLOB)
    if success:
        logger.info("Results saved successfully to GCS.")
    else:
        logger.error("Failed to save results to GCS.")
    # Removed CSV saving logic - app.py will display from the loaded GCS data
    # save_csv_from_json(RESULTS_FILE, RESULTS_FILE_CSV) # Remove this line
    return success