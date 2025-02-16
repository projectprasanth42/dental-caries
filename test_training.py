import os
import sys
import torch
import platform
import gc
import nvidia_smi
import psutil
from typing import Tuple, Optional

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.detection.train import train_model as train_detection
from src.classification.train import train_model as train_classification

def optimize_p100_config() -> dict:
    """
    P100-specific optimizations.
    The P100 has 16GB memory, Pascal architecture, and 3584 CUDA cores.
    """
    return {
        'memory_config': {
            'max_memory': 16000,  # P100 has 16GB memory
            'memory_factor': 0.85,  # Keep 15% memory free for safety
            'preferred_memory_format': torch.channels_last,  # Better for Pascal architecture
        },
        'compute_config': {
            'cudnn_benchmark': True,
            'allow_tf32': True,
            'allow_fp16': True,
            'optimal_batch_size': {
                'detection': 8,     # Optimal for P100 with Mask R-CNN
                'classification': 64  # Optimal for P100 with ResNet
            }
        },
        'arch_specific': {
            'tensor_cores': False,  # P100 doesn't have tensor cores
            'compute_capability': '6.0',  # P100's compute capability
            'l2_cache_size': 4194304,  # 4MB L2 cache
            'max_threads_per_block': 1024
        }
    }

def calculate_optimal_batch_size(total_memory_mb: float, model_type: str) -> int:
    """
    Calculate optimal batch size based on available GPU memory.
    
    Args:
        total_memory_mb: Total GPU memory in MB
        model_type: 'detection' or 'classification'
    """
    # Memory requirements per sample (approximate)
    memory_per_sample = {
        'detection': 1800,     # Mask R-CNN needs more memory per sample
        'classification': 250  # ResNet needs less memory per sample
    }
    
    # Get available GPU memory (keeping 20% as buffer)
    available_memory = total_memory_mb * 0.8
    
    # Calculate batch size
    optimal_batch_size = max(1, int(available_memory / memory_per_sample[model_type]))
    
    # Limit batch size to P100 optimal values
    max_batch_size = {
        'detection': 16,
        'classification': 128
    }
    
    return min(optimal_batch_size, max_batch_size[model_type])

def setup_gpu() -> Tuple[torch.device, dict]:
    """Setup and verify GPU configuration."""
    try:
        # Initialize nvidia-smi
        nvidia_smi.nvmlInit()
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        # Get device properties
        device = torch.device('cuda')
        device_properties = torch.cuda.get_device_properties(device)
        
        # Verify if it's a P100
        is_p100 = 'P100' in device_properties.name
        print(f"GPU Information:")
        print(f"  Name: {device_properties.name}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Total Memory: {device_properties.total_memory / 1024**2:.2f} MB")
        print(f"  CUDA Cores: {device_properties.multi_processor_count}")
        print(f"  Compute Capability: {device_properties.major}.{device_properties.minor}")
        
        # Get P100 specific configurations
        p100_config = optimize_p100_config()
        
        # Set memory management flags
        torch.backends.cudnn.benchmark = p100_config['compute_config']['cudnn_benchmark']
        torch.backends.cuda.matmul.allow_tf32 = p100_config['compute_config']['allow_tf32']
        torch.backends.cudnn.allow_tf32 = p100_config['compute_config']['allow_tf32']
        
        # Set memory format for better performance on Pascal architecture
        torch.backends.cuda.preferred_memory_format = p100_config['memory_config']['preferred_memory_format']
        
        # Enable automatic mixed precision for P100
        if is_p100:
            print("Enabling P100-specific optimizations")
            # Set optimal CUDA cache size for P100
            torch.cuda.set_per_process_memory_fraction(p100_config['memory_config']['memory_factor'])
            
            # Set environment variables for P100
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            os.environ['CUDA_CACHE_DISABLE'] = '0'
            os.environ['CUDA_AUTO_TUNE'] = '1'
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Get initial memory usage
        print(f"\nInitial GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # Calculate optimal batch sizes
        p100_config['compute_config']['optimal_batch_size']['detection'] = calculate_optimal_batch_size(
            device_properties.total_memory / 1024**2, 'detection')
        p100_config['compute_config']['optimal_batch_size']['classification'] = calculate_optimal_batch_size(
            device_properties.total_memory / 1024**2, 'classification')
        
        print(f"\nOptimal Batch Sizes:")
        print(f"  Detection: {p100_config['compute_config']['optimal_batch_size']['detection']}")
        print(f"  Classification: {p100_config['compute_config']['optimal_batch_size']['classification']}")
        
        return device, p100_config
        
    except Exception as e:
        print(f"Error setting up GPU: {str(e)}")
        return torch.device('cpu'), None

def get_data_path():
    """Get the appropriate data path based on the environment."""
    if os.path.exists('/kaggle/input/augmented-dataset/dental_ai_dataset_v4_augmented'):
        # Kaggle environment
        return '/kaggle/input/augmented-dataset/dental_ai_dataset_v4_augmented'
    elif os.path.exists('/content/drive/MyDrive/dental_ai_dataset_v4_augmented'):
        # Google Colab environment
        return '/content/drive/MyDrive/dental_ai_dataset_v4_augmented'
    else:
        # Local environment
        return 'dental_ai_dataset_v4_augmented'

def monitor_gpu_usage(step: Optional[str] = None):
    """
    Monitor GPU memory usage and performance metrics.
    
    Args:
        step: Optional step name for logging
    """
    if torch.cuda.is_available():
        # Get GPU memory info
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        
        # Get GPU utilization
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        utilization = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        
        # Get temperature
        temp = nvidia_smi.nvmlDeviceGetTemperature(handle, nvidia_smi.NVML_TEMPERATURE_GPU)
        
        # Get power usage
        power = nvidia_smi.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
        
        print(f"\nGPU Metrics{f' - {step}' if step else ''}:")
        print(f"  Memory:")
        print(f"    Allocated: {allocated:.2f} MB")
        print(f"    Cached: {cached:.2f} MB")
        print(f"    Total: {info.total / 1024**2:.2f} MB")
        print(f"    Used: {info.used / 1024**2:.2f} MB")
        print(f"    Free: {info.free / 1024**2:.2f} MB")
        print(f"  Performance:")
        print(f"    GPU Utilization: {utilization.gpu}%")
        print(f"    Memory Utilization: {utilization.memory}%")
        print(f"    Temperature: {temp}°C")
        print(f"    Power Usage: {power:.2f}W")
        
        # Memory fragmentation check
        if cached > 2 * allocated:
            print("  Warning: High memory fragmentation detected")
            torch.cuda.empty_cache()

def test_training():
    """Run test training for both detection and classification models."""
    try:
        # Setup GPU
        device, p100_config = setup_gpu()
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        print(f"Using device: {device}")
        
        # Get appropriate data path
        data_dir = get_data_path()
        print(f"Using data directory: {data_dir}")
        
        # Monitor initial GPU state
        monitor_gpu_usage("Initial State")
        
        try:
            # Test detection model with optimal batch size
            print("\n=== Testing Detection Model Training ===")
            batch_size = p100_config['compute_config']['optimal_batch_size']['detection'] if p100_config else 2
            train_detection(data_dir, num_epochs=1, batch_size=batch_size, max_samples=4, device=device)
            
            # Clear GPU memory after detection training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Monitor GPU state after detection
            monitor_gpu_usage("After Detection Training")
            
        except Exception as e:
            print(f"Detection model training failed: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        try:
            # Test classification model with optimal batch size
            print("\n=== Testing Classification Model Training ===")
            batch_size = p100_config['compute_config']['optimal_batch_size']['classification'] if p100_config else 4
            train_classification(data_dir, num_epochs=1, batch_size=batch_size, max_samples=4, device=device)
            
            # Clear GPU memory after classification training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Monitor final GPU state
            monitor_gpu_usage("After Classification Training")
            
        except Exception as e:
            print(f"Classification model training failed: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
    
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            nvidia_smi.nvmlShutdown()

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models/detection', exist_ok=True)
    os.makedirs('models/classification', exist_ok=True)
    
    # Print environment information
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    test_training() 