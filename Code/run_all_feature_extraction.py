# Crime_Detection/Code/run_all_feature_extraction.py

import os
import sys
import logging
from tqdm import tqdm
from datetime import datetime

# Adjust sys.path to import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import BASE_PATHS, FEATURE_PATHS, OUTPUT_DIRS # Import OUTPUT_DIRS
from utils import setup_logger, get_timestamp, find_files_in_directory

# Import individual extraction functions and their classes
from feature_extraction.text_extractor import extract_text_features, RobustTextFeatureExtractor
from feature_extraction.audio_extractor import extract_audio_features, AudioFeatureExtractor
from feature_extraction.video_extractor import extract_video_features, VideoFeatureExtractor

def main():
    # --- Setup Logging ---
    log_dir = OUTPUT_DIRS['logs']
    os.makedirs(log_dir, exist_ok=True)
    timestamp = get_timestamp()
    
    # Configure the main logger for this run
    run_logger = setup_logger('FeatureExtractionRunner', os.path.join(log_dir, f'feature_extraction_run_{timestamp}.log'))
    run_logger.info("="*80)
    run_logger.info("STARTING ALL FEATURE EXTRACTION PIPELINES")
    run_logger.info(f"Log file: {os.path.join(log_dir, f'feature_extraction_run_{timestamp}.log')}")
    run_logger.info(f"Feature output base directories: {FEATURE_PATHS}")
    run_logger.info("="*80)

    # --- Initialize Feature Extractor Models (once) ---
    text_extractor = None
    audio_extractor = None
    video_extractor = None

    try:
        run_logger.info("Initializing RobustTextFeatureExtractor...")
        text_extractor = RobustTextFeatureExtractor()
        run_logger.info("RobustTextFeatureExtractor initialized.")
    except Exception as e:
        run_logger.critical(f"FATAL: Could not initialize RobustTextFeatureExtractor. Aborting. Error: {e}")
        sys.exit(1)

    try:
        run_logger.info("Initializing AudioFeatureExtractor...")
        audio_extractor = AudioFeatureExtractor()
        run_logger.info("AudioFeatureExtractor initialized.")
    except Exception as e:
        run_logger.critical(f"FATAL: Could not initialize AudioFeatureExtractor. Aborting. Error: {e}")
        sys.exit(1)

    try:
        run_logger.info("Initializing VideoFeatureExtractor...")
        video_extractor = VideoFeatureExtractor()
        run_logger.info("VideoFeatureExtractor initialized.")
    except Exception as e:
        run_logger.critical(f"FATAL: Could not initialize VideoFeatureExtractor. Aborting. Error: {e}")
        sys.exit(1)

    # --- Collect all raw files (based on video files) ---
    run_logger.info("\nScanning for raw video files to process...")
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    raw_video_files = find_files_in_directory(BASE_PATHS['raw_videos'], video_extensions)
    total_files_to_process = len(raw_video_files)
    run_logger.info(f"Found {total_files_to_process} video files in {BASE_PATHS['raw_videos']}")

    if total_files_to_process == 0:
        run_logger.warning("No video files found. Nothing to extract. Exiting.")
        return

    # --- Process Files ---
    processed_count = 0 
    skipped_count = 0 
    re_extracted_count = 0 
    failed_count = 0 

    run_logger.info("\nStarting feature extraction for all files...")
    with tqdm(total=total_files_to_process, desc="Overall Feature Extraction") as pbar:
        for relative_video_path in raw_video_files:
            # Extract range_folder and filename_base_no_ext
            range_folder = os.path.dirname(relative_video_path)
            if not range_folder:
                range_folder = "" # For top-level files

            filename_base_no_ext = os.path.splitext(os.path.basename(relative_video_path))[0]

            # Construct full raw paths (assuming consistent naming for audio/transcript)
            raw_video_path = os.path.join(BASE_PATHS['raw_videos'], relative_video_path)
            raw_audio_path = os.path.join(BASE_PATHS['raw_audios'], range_folder, f"{filename_base_no_ext}.wav")
            raw_transcript_path = os.path.join(BASE_PATHS['raw_transcripts'], range_folder, f"{filename_base_no_ext}.txt")

            # Construct output feature paths based on the project structure
            video_output_dir = os.path.join(FEATURE_PATHS['video'], range_folder, filename_base_no_ext)
            video_output_path = os.path.join(video_output_dir, f"{filename_base_no_ext}_all.npy")

            audio_output_dir = os.path.join(FEATURE_PATHS['audio'], range_folder, filename_base_no_ext)
            audio_output_path = os.path.join(audio_output_dir, f"{filename_base_no_ext}_yamnet_embeddings.npy")

            text_output_dir = os.path.join(FEATURE_PATHS['text'], range_folder) 
            text_output_path = os.path.join(text_output_dir, f"{filename_base_no_ext}_roberta_features.npy")

            # Check if all features exist
            video_exists = os.path.exists(video_output_path)
            audio_exists = os.path.exists(audio_output_path)
            text_exists = os.path.exists(text_output_path)

            status_msg = f"{relative_video_path} - "
            current_file_extracted_any = False 
            
            if video_exists and audio_exists and text_exists:
                skipped_count += 1
                status_msg += "All features already exist. Skipped."
                run_logger.debug(status_msg)
            else:
                pbar.set_description(f"Processing: {filename_base_no_ext}")
                
                # --- Text Extraction ---
                if not text_exists:
                    pbar.write(f"  Extracting TEXT for {filename_base_no_ext}...")
                    if os.path.exists(raw_transcript_path):
                        success = extract_text_features(
                            transcript_path=raw_transcript_path, 
                            output_dir=text_output_dir, 
                            filename_base=filename_base_no_ext, 
                            extractor_instance=text_extractor
                        )
                        if not success:
                            run_logger.error(f"  Failed to extract TEXT features for {filename_base_no_ext}.")
                        else:
                            current_file_extracted_any = True
                    else:
                        run_logger.warning(f"  Transcript not found for {filename_base_no_ext} at {raw_transcript_path}. Skipping text extraction.")
                
                # --- Audio Extraction ---
                if not audio_exists:
                    pbar.write(f"  Extracting AUDIO for {filename_base_no_ext}...")
                    if os.path.exists(raw_audio_path):
                        success = extract_audio_features(
                            audio_path=raw_audio_path, 
                            output_dir=audio_output_dir, 
                            filename_base=filename_base_no_ext, 
                            extractor_instance=audio_extractor
                        )
                        if not success:
                            run_logger.error(f"  Failed to extract AUDIO features for {filename_base_no_ext}.")
                        else:
                            current_file_extracted_any = True
                    else:
                        run_logger.warning(f"  Raw audio not found for {filename_base_no_ext} at {raw_audio_path}. Skipping audio extraction.")

                # --- Video Extraction ---
                if not video_exists:
                    pbar.write(f"  Extracting VIDEO for {filename_base_no_ext}...")
                    if os.path.exists(raw_video_path):
                        success = extract_video_features(
                            video_path=raw_video_path, 
                            output_dir=video_output_dir, 
                            filename_base=filename_base_no_ext, 
                            extractor_instance=video_extractor
                        )
                        if not success:
                            run_logger.error(f"  Failed to extract VIDEO features for {filename_base_no_ext}.")
                        else:
                            current_file_extracted_any = True
                    else:
                        run_logger.warning(f"  Raw video not found for {filename_base_no_ext} at {raw_video_path}. Skipping video extraction.")

                # Update counts based on the outcome of this pass
                if current_file_extracted_any:
                    re_extracted_count += 1
                
                # Re-check existence after extraction attempts to confirm if all are now present
                video_exists_after = os.path.exists(video_output_path)
                audio_exists_after = os.path.exists(audio_output_path)
                text_exists_after = os.path.exists(text_output_path)

                if video_exists_after and audio_exists_after and text_exists_after:
                    processed_count += 1
                    status_msg += "All features now present."
                else:
                    failed_count += 1
                    status_msg += "Extraction partially failed or raw file missing for some modalities."
                
                run_logger.info(status_msg)

            pbar.update(1)

    # --- Final Summary ---
    run_logger.info("\n" + "="*80)
    run_logger.info("ALL FEATURE EXTRACTION PIPELINES COMPLETED")
    run_logger.info(f"Total raw video files scanned: {total_files_to_process}")
    run_logger.info(f"Successfully processed (all features now present): {processed_count}")
    run_logger.info(f"Skipped (all features already existed at start): {skipped_count}")
    run_logger.info(f"Re-extracted (at least one missing feature was successfully extracted): {re_extracted_count}")
    run_logger.info(f"Failed (at least one feature could not be extracted or raw file missing): {failed_count}")
    run_logger.info("="*80)

if __name__ == '__main__':
    main()