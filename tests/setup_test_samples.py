import os
import shutil

def setup_test_samples():
    """Set up test sample images for model evaluation."""
    # Create test directory
    test_dir = 'sample-test'
    os.makedirs(test_dir, exist_ok=True)
    
    # Source directories
    data_dir = os.path.join('dental_ai_dataset_v4_augmented', 'three_level_classification', 'train')
    
    # Categories to copy
    categories = {
        'normal': 1,      # Copy 1 normal image
        'superficial': 1, # Copy 1 superficial caries image
        'medium': 1,      # Copy 1 medium caries image
        'deep': 1         # Copy 1 deep caries image
    }
    
    # Copy sample images
    for category, num_samples in categories.items():
        src_dir = os.path.join(data_dir, category)
        if os.path.exists(src_dir):
            try:
                # Get all images in the directory
                images = [f for f in os.listdir(src_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                
                # Copy specified number of images
                for i, img in enumerate(images[:num_samples]):
                    src_path = os.path.join(src_dir, img)
                    dst_path = os.path.join(test_dir, f'{category}_{i+1}.png')
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied {category} image: {img}")
            except Exception as e:
                print(f"Error processing {category} directory: {str(e)}")
        else:
            print(f"Warning: Could not find {src_dir}")

if __name__ == '__main__':
    setup_test_samples() 