# NF_MUSE Project

## Overview
This project uses a pre-trained Random Forest Model to predict Mind Wandering (MW) in real-time. If MW is detected, a beep sound will be played. This requires a Muse headband connected with the Mind Monitor application, which sends the EEG data to the target IP address of the device running the code (PC), with port number 5000 specified.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Daisuke134/NF_MUSE.git
   ```  
2. Navigate to the cloned directory:
   ```bash   
   cd NF_MUSE
   ``` 
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ``` 
## Running the Predictions
To start making real-time predictions, run:
   ```bash
   python predict.py
   ``` 
※Make sure your Muse device is connected and streaming data.

## Training and Evaluation
If you wish to evaluate the model's performance or retrain it with new data, follow these steps:

1. Download the training data CSV:
   ```bash
   curl -L -o training_data/01-18--22-41-05.csv "https://drive.google.com/uc?export=download&id=1InAOM7XTEpIAwKFhyzCfsYTwFx0rgFnz"
   ```
2. Generate training data text file from CSV:
   ```bash   
   python data.py
   ```
3. Train and evaluate the Random Forest model(This will use the generated text file to train the model and output its performance metrics):
   ```bash
　　　　　　python model.py
   ```

## Libraries Used
The project uses several libraries which need to be installed:
- numpy
- pandas
- scikit-learn
- joblib
- python-osc

These can be installed using the requirements.txt file provided in the repository.

## Contributing
Please feel free to contribute to the project by making a pull request or opening an issue.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

For any questions or support, please contact [Your Contact Information].



   


