# French Trains Delay Prediction App
A simple train delay predictor, in python and using random forest regressor.

## Features 
A simple attempt at predicting French trains delay, based on the SNCF report from 2018 to 2023 of the delays on each train route every day. The app is built in python and allow user to open it in web browsner to :
1. Get a train delay prediction by entering the route and the month. The user can also see a chart for the principle cause of past delays for this train, as well as the perecentage of past trains with these parameters that were late.
2. Submit some data to enrich our data set, if the data is from after 2023.


## Getting Started

### Prerequisites:

Python installed on your machine
A modern web browser

### Installation

Clone the repository: bash git clone *https://github.com/juliane9/trains_delay_prediction.git*

Open app.py.

Compile and run the project:

*python app.py*

Type *http://127.0.0.1:5000* in a web browser to load the web page. "Beaware, this is a development server. Do not use it in a production deployment. Use a production WSGI server instead."


## Usage
To get a predction, enter a month in a numeric format (1 to 12) and route, in the format *departure-arrival*. Then click on predict. If a train is predicted to be delayed, the predicted amount of the delay will be shown as well as a bar graph with the past causes of delays for this route-month combination will be dispalyed. If it is predicted to be on time, the predictor will simply say so. 

To submit your data about your train delay, you need to enter the *departure station*, *arrival station*, *month of the journey*, *month* of the journey, as well as select a cause for the delay. If the cause is not known or does not fit in any of the pre-defined category, select *unknown*. Please, enter data only if it comes from a journey taken after 2023.



## Contributions and improvements of the model
Contributions to the trains predictor project are welcome ! So far, this project uses a simple Random Forest Regressor model to predict the amount of the delay, based on past causes of delay and the month of the journey for each route. 
Several improvements could be made, including (but not limited to):
- adding a "day of the week" factor to the prediction
- adding a weather feature
- a more fitted way to determine whether a train will be late (which is so far decided with an indicator type random variable, predicting a "delay" if more then 10% of the past trains for this route-month combination were late). 

### Please follow these steps to contribute:
Fork the repository. Create a new branch: git checkout -b feature/your_feature_name. Make your changes and commit them: git commit -am 'Add some feature'. Push to the branch: git push origin feature/your_feature_name. Submit a pull request.


### License
This project is licensed under the MIT License - see the LICENSE.md file for details.
