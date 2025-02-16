# Dental Caries Analysis and Recommendation System

A comprehensive deep learning-based system for dental caries detection, classification, and personalized recommendations from X-ray images.

## Abstract

Dental caries is one of the most prevalent oral health issues worldwide, often requiring early detection and intervention to prevent severe complications. This project presents a novel deep learning-based system for dental X-ray analysis, focusing on cavity detection, severity classification, and personalized recommendations. Using advanced models like Mask R-CNN, the system accurately detects cavities and segments their affected regions. A ResNet-50 classifier is employed to assess the severity of caries, categorizing them as superficial, medium, or deep. Additionally, a fine-tuned BERT-based recommendation system generates personalized preventive advice based on the severity and potential causes, such as poor hygiene or dietary habits. 

The system automates the diagnostic process, reducing manual effort and improving accuracy in detecting and analyzing cavities. It empowers both dentists and patients by offering actionable insights, prioritizing treatment needs, and promoting better oral hygiene practices. The solution is deployable via a web-based interface, enabling remote accessibility and integration into clinical workflows. This comprehensive approach addresses limitations of traditional methods, making it a valuable tool for advancing dental care and early intervention.

## Project Components

1. **Detection and Segmentation (Mask R-CNN)**
   - Cavity detection in dental X-rays
   - Precise segmentation of affected regions
   - Region proposal and instance segmentation

2. **Severity Classification (ResNet-50)**
   - Three-level severity classification
   - Feature extraction and transfer learning
   - Confidence scoring for predictions

3. **Recommendation System (BERT)**
   - Personalized treatment recommendations
   - Context-aware advice generation
   - Integration with severity analysis

4. **Web Interface**
   - User-friendly dashboard
   - Real-time analysis capabilities
   - Report generation and visualization

## Project Structure

```
dental_caries_project/
├── data/
│   ├── dental_ai_dataset_v4_augmented/  # Preprocessed dataset
│   └── DC1000_dataset/                  # Original dataset
├── src/
│   ├── detection/                       # Mask R-CNN implementation
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── utils.py
│   ├── classification/                  # ResNet-50 implementation
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── utils.py
│   ├── recommendation/                  # BERT implementation
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── utils.py
│   └── web/                            # Web interface
│       ├── app.py
│       ├── templates/
│       └── static/
├── models/                             # Saved model checkpoints
│   ├── detection/
│   ├── classification/
│   └── recommendation/
├── tests/                              # Unit tests
│   ├── test_detection.py
│   ├── test_classification.py
│   └── test_recommendation.py
├── notebooks/                          # Jupyter notebooks
│   ├── data_analysis.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
├── requirements.txt                    # Project dependencies
├── setup.py                           # Package setup
├── .gitignore
└── README.md
```

## Dataset Information

### Original Data (`DC1000_dataset/`)
- Source: 500 panoramic dental X-ray images
- Labeled by medical professionals
- Initial distribution:
  * 51 superficial caries
  * 102 medium caries
  * 153 pulpitis
  * 204 deep caries
  * 255 additional cases from doctors

### Augmented Dataset (`dental_ai_dataset_v4_augmented/`)
Balanced dataset for multiple tasks:

1. **Binary Classification**
   - Total: 1,548 training images
   - Classes:
     * Normal: 1,290 images (83.3%)
     * Caries: 258 images (16.7%)

2. **Three-Level Classification**
   - Total: 681 training images
   - Classes:
     * Normal: 15 images (2.2%)
     * Superficial: 204 images (30.0%)
     * Medium: 204 images (30.0%)
     * Deep: 258 images (37.9%)

3. **Detailed Segmentation**
   - 344 image-mask pairs
   - 856 total regions
   - Distribution:
     * Superficial: 316 regions (36.9%)
     * Medium: 282 regions (32.9%)
     * Deep: 258 regions (30.1%)

## Setup and Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd dental_caries_project
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Architecture

### 1. Mask R-CNN
- Backbone: ResNet-101-FPN
- ROI Align for precise spatial localization
- Multi-task learning for detection and segmentation

### 2. ResNet-50 Classifier
- Pre-trained on ImageNet
- Fine-tuned for dental caries severity
- Custom head for 3-class classification

### 3. BERT Recommendation System
- Pre-trained BERT-base
- Fine-tuned on dental domain knowledge
- Custom output layer for recommendation generation

## Usage

### For Classification Tasks
1. **Binary Classification**:
   - Use `binary_classification/train/`
   - Classes: normal, caries
   - Recommended class weights: {normal: 2.5, caries: 1.0}

2. **Three-Level Classification**:
   - Use `three_level_classification/train/`
   - Classes: normal, superficial, medium, deep
   - Recommended class weights:
     * Deep: 1.0
     * Medium: 3.5
     * Superficial: 15.0
     * Normal: 20.0

### For Segmentation Tasks
1. **Binary Segmentation**:
   - Use `binary_segmentation/train/`
   - Pixel values: 0 (background), 255 (caries)

2. **Detailed Segmentation**:
   - Use `detailed_segmentation/train/`
   - Pixel values: 0, 102, 153, 255
   - Maintain severity level information

## Development Roadmap

1. Phase 1: Data Preparation and Model Development
   - Dataset preprocessing and augmentation
   - Implementation of Mask R-CNN
   - Implementation of ResNet-50 classifier

2. Phase 2: Recommendation System
   - BERT model fine-tuning
   - Integration with classification results
   - Recommendation template development

3. Phase 3: Web Interface
   - Dashboard development
   - API implementation
   - Integration testing

4. Phase 4: Deployment and Optimization
   - Model optimization
   - System integration
   - Performance testing

## Contributing

[Contribution guidelines to be added]

## License

[License information to be added]

## Citation
If you use this dataset in your research, please cite:
[Citation information to be added] 