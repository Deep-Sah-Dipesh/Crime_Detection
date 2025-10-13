# download_model.py
import tensorflow_hub as hub
import os
import shutil

print("Attempting to download the YAMNet model from TensorFlow Hub...")

try:
    # This line triggers the download to a temporary cache directory
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    print("\n✅ Model downloaded successfully to a temporary cache.")
    
    # --- This part finds the cached files ---
    
    # Get the default cache directory used by TensorFlow Hub
    cache_dir = hub.get_hub_cache_dir()
    print(f"   TensorFlow Hub cache is located at: {cache_dir}")

    # Find the specific model folder within the cache
    downloaded_model_path = os.path.join(cache_dir, "079e02c2e8642a2b8192b2d35689f41398a6558e") # This is the specific hash for YAMNet
    
    if os.path.exists(downloaded_model_path):
        # Define the destination in your project
        destination_path = os.path.join(os.getcwd(), 'local_models')
        os.makedirs(destination_path, exist_ok=True)
        
        print(f"   Copying model from cache to: {destination_path}")
        
        # Copy the entire directory contents
        for item in os.listdir(downloaded_model_path):
            s = os.path.join(downloaded_model_path, item)
            d = os.path.join(destination_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
                
        print("\n✅ Model successfully copied to your 'local_models' project folder.")
        print("You can now run the predict.py script.")
    else:
        print("\n❌ Error: Could not find the downloaded model in the cache.")
        print("   Please check your internet connection and try again.")
        print(f"   Look inside '{cache_dir}' to see if the model downloaded to a different subfolder.")

except Exception as e:
    print(f"\n❌ An error occurred during download: {e}")
    print("   This may be due to a network issue, firewall, or proxy configuration.")