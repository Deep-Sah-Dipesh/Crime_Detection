import os
import sys
import logging
from tqdm import tqdm
from datetime import datetime

# Determine the script's directory robustly
try:
    # This is the preferred method, works when __file__ is defined
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for environments where __file__ is not defined (e.g., interactive sessions, notebooks)
    # sys.argv[0] contains the path to the script being executed.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

# Add the project root (one level up from SCRIPT_DIR) to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

# Now, imports from config and utils should work correctly
from config import BASE_PATHS, FEATURE_PATHS, OUTPUT_DIRS
from utils import setup_logger, get_logger, get_timestamp, find_files_in_directory

# These imports assume 'feature_extraction' is a package discoverable from PROJECT_ROOT
from feature_extraction.text_extractor import extract_text_features, RobustTextFeatureExtractor
from feature_extraction.audio_extractor import extract_audio_features, AudioFeatureExtractor
from feature_extraction.video_extractor import extract_video_features, VideoFeatureExtractor

def main():
    log_dir = OUTPUT_DIRS['logs']
    os.makedirs(log_dir, exist_ok=True)
    timestamp = get_timestamp()
    
    # Setup logger for this script
    run_logger = setup_logger('FeatureExtractionRunner', os.path.join(log_dir, f'feature_extraction_run_{timestamp}.log'))
    run_logger.info("STARTING ALL FEATURE EXTRACTION PIPELINES")
    run_logger.info(f"Log file: {os.path.join(log_dir, f'feature_extraction_run_{timestamp}.log')}\n")

    text_extractor = RobustTextFeatureExtractor()
    audio_extractor = AudioFeatureExtractor()
    video_extractor = VideoFeatureExtractor()

    raw_videos_root = BASE_PATHS['raw_videos']
    all_video_sub_paths = find_files_in_directory(raw_videos_root, ['.mp4', '.avi', '.mov', '.mkv'])
    total_files_to_process = len(all_video_sub_paths)

    run_logger.info(f"Found {total_files_to_process} video files to process for feature extraction.\n")

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    re_extracted_count = 0

    # Ensure feature output root directories exist (e.g., .../text_features/)
    for path in FEATURE_PATHS.values():
        os.makedirs(path, exist_ok=True)

    with tqdm(total=total_files_to_process, desc="Overall Feature Extraction Progress") as pbar:
        for video_sub_path in all_video_sub_paths:
            full_video_path = os.path.join(raw_videos_root, video_sub_path)
            
            # Extract relevant path components for dynamic directory creation and filenames
            # video_sub_path example: '1-1004/About.Time.2013__#00-42-52_00-45-31_label_A.mp4'
            clip_sub_dir = os.path.dirname(video_sub_path) # e.g., '1-1004'
            clip_base_name = os.path.splitext(os.path.basename(video_sub_path))[0] # e.g., 'About.Time.2013__#00-42-52_00-45-31_label_A'

            # --- Define the actual output directories for each feature type (where the .npy files will be saved) ---
            # For Text and Audio, features go directly into the <FEATURE_TYPE>/<clip_sub_dir>/ folder
            text_features_save_dir = os.path.join(FEATURE_PATHS['text'], clip_sub_dir)
            audio_features_save_dir = os.path.join(FEATURE_PATHS['audio'], clip_sub_dir)
            
            # For Video, features go into an additional sub-directory named after the clip_base_name
            video_features_save_dir = os.path.join(FEATURE_PATHS['video'], clip_sub_dir, clip_base_name)

            # Ensure these specific directories exist before trying to save files into them
            os.makedirs(text_features_save_dir, exist_ok=True)
            os.makedirs(audio_features_save_dir, exist_ok=True)
            os.makedirs(video_features_save_dir, exist_ok=True) # Ensure video's specific clip directory exists

            # --- Define the FINAL expected feature file paths for existence checks ---
            # These are based on your project structure examples:
            # text_features/.../CLIP_NAME_roberta_features.npy
            # audio_features/.../CLIP_NAME_yamnet_embeddings.npy
            # video_features/.../CLIP_NAME/CLIP_NAME_all.npy (checking for the aggregated video feature file)
            
            text_final_feature_file = os.path.join(text_features_save_dir, f"{clip_base_name}_roberta_features.npy")
            audio_final_feature_file = os.path.join(audio_features_save_dir, f"{clip_base_name}_yamnet_embeddings.npy")
            video_final_feature_file = os.path.join(video_features_save_dir, f"{clip_base_name}_all.npy") # Assuming '_all.npy' is the main video feature file

            # Check if features already exist
            all_features_exist_initially = (
                os.path.exists(text_final_feature_file) and
                os.path.exists(audio_final_feature_file) and
                os.path.exists(video_final_feature_file)
            )

            status_msg = f"Processing {full_video_path}: "

            if all_features_exist_initially:
                status_msg += "All features already exist. Skipping."
                skipped_count += 1
                run_logger.info(status_msg)
            else:
                run_logger.info(status_msg + "Extracting missing features...")
                re_extracted_count += 1
                
                # We don't need 'extracted_any' bool here, as we update counts based on final existence check.

                if not os.path.exists(text_final_feature_file):
                    run_logger.info(f"  Extracting TEXT for {video_sub_path}...")
                    try:
                        # Pass the target save DIRECTORY and the base filename for internal construction
                        extracted_text_path = extract_text_features(full_video_path, text_features_save_dir, clip_base_name, extractor_instance=text_extractor)
                        if extracted_text_path:
                            run_logger.info(f"  TEXT features saved to {extracted_text_path}")
                        else:
                            run_logger.warning(f"  TEXT feature extraction returned no path for {video_sub_path}.")
                    except Exception as e:
                        run_logger.error(f"  Error extracting TEXT for {video_sub_path}: {e}")

                if not os.path.exists(audio_final_feature_file):
                    run_logger.info(f"  Extracting AUDIO for {video_sub_path}...")
                    try:
                        # Pass the target save DIRECTORY and the base filename for internal construction
                        extracted_audio_path = extract_audio_features(full_video_path, audio_features_save_dir, clip_base_name, extractor_instance=audio_extractor)
                        if extracted_audio_path:
                            run_logger.info(f"  AUDIO features saved to {extracted_audio_path}")
                        else:
                            run_logger.warning(f"  AUDIO feature extraction returned no path for {video_sub_path}.")
                    except Exception as e:
                        run_logger.error(f"  Error extracting AUDIO for {video_sub_path}: {e}")

                if not os.path.exists(video_final_feature_file):
                    run_logger.info(f"  Extracting VIDEO for {video_sub_path}...")
                    try:
                        # Pass the target save DIRECTORY (which includes the clip's specific folder) and the base filename
                        extracted_video_path = extract_video_features(full_video_path, video_features_save_dir, clip_base_name, extractor_instance=video_extractor)
                        if extracted_video_path:
                            run_logger.info(f"  VIDEO features saved to {extracted_video_path}")
                        else:
                            run_logger.warning(f"  VIDEO feature extraction returned no path for {video_sub_path}.")
                    except Exception as e:
                        run_logger.error(f"  Error extracting VIDEO for {video_sub_path}: {e}")

                # After extraction attempts, check final status
                video_exists_after = os.path.exists(video_final_feature_file)
                audio_exists_after = os.path.exists(audio_final_feature_file)
                text_exists_after = os.path.exists(text_final_feature_file)

                if video_exists_after and audio_exists_after and text_exists_after:
                    processed_count += 1
                    status_msg += "All features now present."
                else:
                    failed_count += 1
                    status_msg += "Extraction partially failed or raw file missing for some modalities."
                
                run_logger.info(status_msg)

            pbar.update(1)

    run_logger.info("\n" + "="*80)
    run_logger.info("ALL FEATURE EXTRACTION PIPELINES COMPLETED")
    run_logger.info(f"Total raw video files scanned: {total_files_to_process}")
    run_logger.info(f"Successfully processed (all features now present, either initially or after extraction): {processed_count + skipped_count}")
    run_logger.info(f"  (This includes {skipped_count} files that had all features already and {processed_count} files that achieved all features after extraction attempts)")
    run_logger.info(f"Attempted re-extraction (at least one missing feature was targeted): {re_extracted_count}")
    run_logger.info(f"Failed (at least one feature could not be extracted or raw file missing and remained missing): {failed_count}")
    run_logger.info("="*80)


if __name__ == "__main__":
    main()