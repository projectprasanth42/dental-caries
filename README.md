# Dental Caries Analysis and Recommendation System

A comprehensive deep learning-based system for dental caries detection, classification, and personalized recommendations from X-ray images.

## Abstract

Dental caries is one of the most prevalent oral health issues worldwide, often requiring early detection and intervention to prevent severe complications. This project presents a novel deep learning-based system for dental X-ray analysis, focusing on cavity detection, severity classification, and personalized recommendations. Using advanced models like Mask R-CNN, the system accurately detects cavities and segments their affected regions. A ResNet-50 classifier is employed to assess the severity of caries, categorizing them as normal, superficial, medium, or deep. Additionally, a fine-tuned BERT-based recommendation system generates personalized preventive advice based on the severity and potential causes.

## Project Components

1. **Detection and Segmentation (Mask R-CNN)**
   - Cavity detection in dental X-rays
   - Precise segmentation of affected regions
   - Region proposal and instance segmentation
   - Normalized bounding box coordinates

2. **Severity Classification (ResNet-50)**
   - Four-level severity classification
   - Feature extraction and transfer learning
   - Confidence scoring for predictions
   - Class-weighted loss for imbalanced data
   - Data augmentation for robustness

3. **Recommendation System (BERT)**
   - Personalized treatment recommendations
   - Context-aware advice generation
   - Severity-based recommendation prioritization
   - Five recommendation categories:
     * Routine Care
     * Preventive Measures
     * Immediate Treatment
     * Dietary Modifications
     * Oral Hygiene Education

## Project Structure

```
dental_caries_project/
├── src/
│   ├── detection/           # Mask R-CNN implementation
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── utils.py
│   ├── classification/      # ResNet-50 implementation
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── utils.py
│   ├── recommendation/      # BERT implementation
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── utils.py
│   └── web/                 # Web interface
│       ├── templates/
│       └── static/
├── models/                  # Saved model checkpoints
│   ├── detection/
│   ├── classification/
│   └── recommendation/
├── tests/                   # Unit tests
├── app.py                   # Flask application
├── requirements.txt         # Project dependencies
└── README.md               # Documentation
```

## Dataset Information

The project uses the Dental AI Dataset V4 (Augmented), which includes:

### Three-Level Classification Dataset
- Total Images: 681
- Distribution:
  * Normal: 15 images (2.2%)
  * Superficial: 204 images (30.0%)
  * Medium: 204 images (30.0%)
  * Deep: 258 images (37.9%)

### Training Configuration
- Batch Size: 16
- Learning Rate: 0.0001
- Validation Split: 20%
- Early Stopping: 10 epochs patience
- Weight Decay: 1e-4
- Data Augmentation:
  * Random rotations
  * Horizontal/vertical flips
  * Brightness/contrast adjustments
  * Gaussian noise

## Setup and Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Extract dataset:
```bash
# Extract the dataset archive
unzip dental_ai_dataset_v4_augmented.zip
```

4. Run the application:
```bash
python app.py
```

## Model Architecture

### 1. Detection Model (Mask R-CNN)
- Backbone: ResNet-50-FPN
- ROI Align for precise spatial localization
- Multi-task learning for detection and segmentation

### 2. Classification Model (ResNet-50)
- Pre-trained on ImageNet
- Fine-tuned for dental caries severity
- Custom head for 4-class classification
- Class-weighted loss for imbalance handling

### 3. Recommendation Model (BERT)
- Pre-trained BERT-base model
- Custom classification head
- Context-aware recommendation generation
- Confidence-based ranking

## Web Interface Features

1. **Upload Interface**
   - Supports PNG and JPEG formats
   - Real-time image preview
   - Drag and drop functionality

2. **Analysis Results**
   - Detection visualization
   - Severity classification
   - Confidence scores
   - Personalized recommendations

3. **Responsive Design**
   - Mobile-friendly interface
   - Real-time updates
   - Interactive visualization

## Performance Metrics

The system achieves the following performance metrics:

1. **Detection**
   - Average Precision (AP): To be evaluated
   - IoU Threshold: 0.5
   - Confidence Threshold: 0.5

2. **Classification**
   - Accuracy: To be evaluated
   - F1-Score: To be evaluated
   - Class-wise Performance: To be evaluated

3. **Recommendation**
   - Relevance Score: To be evaluated
   - User Satisfaction: To be evaluated

## Future Improvements

1. **Model Enhancements**
   - Implement cross-validation
   - Experiment with other architectures
   - Add model ensemble techniques

2. **Dataset Expansion**
   - Collect more normal cases
   - Add more variety in conditions
   - Include different X-ray types

3. **Feature Additions**
   - Treatment progress tracking
   - Multi-image analysis
   - Report generation

## License

[License information to be added]

## Citation

If you use this project, please cite:
[Citation information to be added]

## Contact

For questions or collaboration, please contact:
[Contact information to be added] 