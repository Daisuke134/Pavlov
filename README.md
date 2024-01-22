# NF_MUSE Project

## Overview
This project uses a pre-trained model to analyze EEG data from the Muse device in real-time. When Mind Wandering is detected, an alert will be triggered.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Daisuke134/NF_MUSE.git
   ```  

3. Navigate to the cloned directory:
   ```bash   
   cd NF_MUSE
   ``` 
5. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ``` 
## Running the Predictions
To start making real-time predictions, run:
   ```bash
   python predict.py
   ``` 
Make sure your Muse device is connected and streaming data.

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



   


