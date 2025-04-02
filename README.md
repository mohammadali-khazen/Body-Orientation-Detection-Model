# Body Orientation Detection Model

A deep learning model that detects worker's body orientation on job sites using signal blockage by the human body. The system uses a receiving beacon mounted on the worker's chest and reference transmitting beacons to determine the worker's field of view and body orientation.

## Overview

The system works by analyzing signal blockage patterns between the chest-mounted receiving beacon and reference transmitting beacons. When a worker's body blocks signals from certain beacons, the system can determine the worker's orientation based on the angles of the detected transmitting beacons relative to the worker's estimated location.

### Features

- Uses RSSI (Received Signal Strength Indicator) data from multiple beacons
- Incorporates acceleration data for enhanced accuracy
- Predicts eight ordinal orientations (up, down, left, right, and diagonals)
- Real-time orientation detection
- Deep Neural Network (DNN) based classification

## Project Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── data/
│   │   └── data_loader.py      # Data loading and preprocessing
│   ├── features/
│   │   └── feature_engineering.py  # Feature calculation and transformation
│   ├── models/
│   │   └── orientation_model.py    # DNN model implementation
│   └── main.py                 # Main execution script
├── RSSI and Location Datasets/  # Data directory
└── models/                     # Saved model directory
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Body-Orientation-Detection-Model.git
cd Body-Orientation-Detection-Model
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate 
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your RSSI and location datasets in the `RSSI and Location Datasets/` directory.

2. Run the main script:

```bash
python src/main.py
```

The script will:

- Load and preprocess the data
- Calculate features (angles, distances, etc.)
- Train the DNN model
- Generate training history plots
- Show model evaluation metrics
- Save the trained model

## Model Architecture

The DNN model consists of:

- Input layer (12 features)
- Three hidden layers (13, 32, and 64 neurons)
- Dropout layers (0.1 rate) for regularization
- Output layer (8 classes, softmax activation)

## Features Used

The model uses the following features:

- RSSI values from three beacons
- Angles between target and transmitters
- Distances between points
- Inter-transmitter distances

## Data Preprocessing

The system performs the following preprocessing steps:

- Timestamp normalization
- RSSI value cleaning and normalization
- Acceleration data processing
- Feature scaling and encoding