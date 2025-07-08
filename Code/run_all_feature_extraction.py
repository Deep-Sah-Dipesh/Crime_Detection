import os
import sys
import logging
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import BASE_PATHS, FEATURE_PATHS, OUTPUT_DIRS
from utils import setup_logger, get_logger, get_timestamp, find_files_in_directory

from feature_extraction.text_extractor import extract_text_features, RobustTextFeatureExtractor
from feature_extraction.audio_extractor import extract_audio_features, AudioFeatureExtractor
from feature_extraction.video_extractor import extract_video_features, VideoFeatureExtractor

def main():
    log_dir = OUTPUT_DIRS['logs']
    os.makedirs(log_dir, exist_ok=True)
    timestamp = get_timestamp()
    
    run_logger = setup_logger('FeatureExtractionRunner', os.path.join(log_dir, f'feature_extraction_run_{timestamp}.log'))
    run_logger.info("STARTING ALL FEATURE EXTRACTION PIPELINES")
    run_logger.info(f"Log file: {os.path.join(log_dir, f'feature_extraction_run_{timestamp}.log')}\n")

    text_extractor = RobustTextFeatureExtractor()
    audio_extractor = AudioFeatureExtractor()
    video_extractor = VideoFeatureExtractor()

    raw_videos_root = BASE_PATHS['raw_videos']
    all_video_sub_paths = find_files_in_directory(raw_videos_root, ['.mp4', '.avi', '.mov'])

    if not all_video_sub_paths:
        run_logger.error(f"No video files found in {raw_videos_root}. Exiting feature extraction.")
        return

    total_files_to_process = len(all_video_sub_paths)
    processed_count = 0
    skipped_count = 0
    re_extracted_count = 0
    failed_count = 0

    run_logger.info(f"Found {total_files_to_process} raw video files to process.")
    run_logger.info("Starting feature extraction process...\n")

    with tqdm(total=total_files_to_process, desc="Overall Feature Extraction Progress", unit="file", file=sys.stdout) as pbar:
        for video_sub_path in all_video_sub_paths:
            full_video_path = os.path.join(BASE_PATHS['raw_videos'], video_sub_path)
            
            # Example video_sub_path: '1-1004/A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A.mp4'
            # clip_dir_relative: '1-1004'
            # clip_filename_base: 'A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A'
            clip_dir_relative = os.path.dirname(video_sub_path)
            clip_filename_base = os.path.splitext(os.path.basename(video_sub_path))[0]

            # Define the specific output directories for features for this clip
            # For video and audio, features go into a subfolder named after the clip_filename_base
            # For text, features go directly into the range_folder
            video_feature_output_dir = os.path.join(FEATURE_PATHS['video'], clip_dir_relative, clip_filename_base)
            audio_feature_output_dir = os.path.join(FEATURE_PATHS['audio'], clip_dir_relative, clip_filename_base)
            text_feature_output_dir = os.path.join(FEATURE_PATHS['text'], clip_dir_relative)

            # Expected final paths for checking existence
            video_final_feature_file = os.path.join(video_feature_output_dir, f"{clip_filename_base}_all.npy")
            audio_final_feature_file = os.path.join(audio_feature_output_dir, f"{clip_filename_base}_yamnet_embeddings.npy")
            text_final_feature_file = os.path.join(text_feature_output_dir, f"{clip_filename_base}_roberta_features.npy")

            all_features_exist = (
                os.path.exists(video_final_feature_file) and
                os.path.exists(audio_final_feature_file) and
                os.path.exists(text_final_feature_file)
            )

            status_msg = f"Processing {video_sub_path}: "

            if all_features_exist:
                skipped_count += 1
                status_msg += "All features already exist. Skipping."
                run_logger.info(status_msg)
            else:
                raw_audio_path = os.path.join(BASE_PATHS['raw_audios'], clip_dir_relative, f"{clip_filename_base}.wav")
                raw_transcript_path = os.path.join(BASE_PATHS['raw_transcripts'], clip_dir_relative, f"{clip_filename_base}.txt")
                
                extracted_any = False
                
                # Video Feature Extraction
                if not os.path.exists(video_final_feature_file):
                    run_logger.info(f"  Extracting video features for {clip_filename_base}...")
                    if extract_video_features(full_video_path, video_feature_output_dir, clip_filename_base, video_extractor):
                        extracted_any = True
                else:
                    run_logger.debug(f"  Video features for {clip_filename_base} already exist.")

                # Audio Feature Extraction
                if not os.path.exists(audio_final_feature_file):
                    run_logger.info(f"  Extracting audio features for {clip_filename_base}...")
                    if extract_audio_features(raw_audio_path, audio_feature_output_dir, clip_filename_base, audio_extractor):
                        extracted_any = True
                else:
                    run_logger.debug(f"  Audio features for {clip_filename_base} already exist.")

                # Text Feature Extraction
                if not os.path.exists(text_final_feature_file):
                    run_logger.info(f"  Extracting text features for {clip_filename_base}...")
                    if extract_text_features(raw_transcript_path, text_feature_output_dir, clip_filename_base, text_extractor):
                        extracted_any = True
                else:
                    run_logger.debug(f"  Text features for {clip_filename_base} already exist.")
                
                if extracted_any:
                    re_extracted_count += 1
                
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