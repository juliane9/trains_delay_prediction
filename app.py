from flask import Flask, request, render_template_string
from sklearn.ensemble import RandomForestClassifier  # Assuming you've already trained this
import numpy as np

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # To be able to generate plots outside of main
import matplotlib.pyplot as plt
import os

import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# ____ UTILTY METHOD _____

# Process the data, treat it as needed, add columns and remove some
def reshape_data(df):
    # Convert to date format the date column 
    df['Date'] = pd.to_datetime(df['Date'])

    # Copy the data (in order to not modify the original dataset)
    df_sliced = df.copy()

    # Extract the month and create a column for that as we will process the delays depending on the month
    df_sliced['Month'] = df_sliced['Date'].dt.month

    # Create the 'Line' identifier column which will be used to create the different existing "routes" (route = DEPARTURE STATION - ARRIVAL STATION)
    df_sliced['Line'] = df_sliced['Gare de départ'] + " - " + df_sliced["Gare d'arrivée"]

    # Remove unnecessary columns 
    columns_to_drop = [
        "Commentaire annulations",
        "Commentaire retards au départ",
        "Commentaire retards à l'arrivée"
    ]
    # Drop the specified columns from the DataFrame
    df_sliced.drop(columns_to_drop, axis=1, inplace=True, errors='ignore') # inplace modify the existing data rather then creating a new one 

    # Drop the row if there are no trains scheduled
    df_sliced = df_sliced[df_sliced['Nombre de circulations prévues'] != 0]

    return df_sliced


# Method used to determine if "a train with this route on this month will be late"
def is_late_prediction_m(df_sliced): 
    delay_ratio = df_sliced.groupby(['Line', 'Month']).apply(
        lambda x: (x["Nombre de trains en retard à l'arrivée"]).sum() / ((x['Nombre de circulations prévues']).sum()) * 100  # 1e-8 is added to avoid division by 0
    ).reset_index(name='delay_ratio')

    # Add a column with delay ratio back into main DataFrame
    df_sliced = pd.merge(df_sliced, delay_ratio, on=['Line', 'Month'], how='left')
    df_sliced.dropna(subset=['Line_encoded', 'Month', 'delay_ratio'], inplace=True) 

    return df_sliced


# Method to predict the amount of delay, given that the train is predicted to be late, using a RandomForestRegressor.
def predict_amount_of_delay(route, month, delay_ratio_example, df_sliced):  
    # Possible causes of delay 
    cause_columns = [
        'Prct retard pour causes externes',
        'Prct retard pour cause infrastructure',
        'Prct retard pour cause gestion trafic',
        'Prct retard pour cause matériel roulant',
        'Prct retard pour cause gestion en gare et réutilisation de matériel',
        'Prct retard pour cause prise en compte voyageurs (affluence, gestions PSH, correspondances)'
    ]
  
    # Filter the dataset to include only the trains delayed by a significant amount (superior to 5 minutes)
    delayed_trains = df_sliced[df_sliced["Retard moyen de tous les trains à l'arrivée"] > 5]

    # The relevant features used to train the predictor 
    feature_columns = ['Line_encoded', 'Month', 'delay_ratio'] + cause_columns
    X_delayed = delayed_trains[feature_columns]  
    y_delayed = delayed_trains["Retard moyen des trains en retard à l'arrivée"]  # Target variable is the delay amount

    # Split the data into training and testing sets
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_delayed, y_delayed, test_size=0.2, random_state=42)

    # Train the regression model with RandomForestRegressor as this model can be used for non-linear relationship
    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train_reg, y_train_reg)

    route_encoded = encoder.transform([route]) # Encode the route
    specific_route_data = df_sliced[(df_sliced['Line'] == route) & (df_sliced['Month'] == month)]

    # Compute the mean of the cause columns for this route and month
    cause_values = specific_route_data[cause_columns].mean().tolist()

    features_for_prediction = [route_encoded[0], month, delay_ratio_example] + cause_values

    # Ensure `list_of_cause_values_for_this_route` is in the same order as `cause_columns`
    delay_amount_prediction = regressor.predict([features_for_prediction])

    prediction = "The train is predicted to be late by approximately {:.0f} minutes.".format(delay_amount_prediction[0])
    
    return prediction


# Method to calculate weighted mean                 
def weighted_mean(values, weights):
    return np.average(values, weights=weights)

# Method that generates the cause of delay graph for a route 
def generate_stacked_bar_chart(route, df, month):
    # Possible causes of delay
    cause_columns = [
        'Prct retard pour causes externes',
        'Prct retard pour cause infrastructure',
        'Prct retard pour cause gestion trafic',
        'Prct retard pour cause matériel roulant',
        'Prct retard pour cause gestion en gare et réutilisation de matériel',
        'Prct retard pour cause prise en compte voyageurs (affluence, gestions PSH, correspondances)'
    ]

    # Aggregate rows by line and month and create a weighted average over the years for the possible causes of delay.
    df_sliced['weights'] = df_sliced.groupby(['Line', 'Month'])["Nombre de trains en retard à l'arrivée"].transform('sum')
    df_sliced['weights'] = df_sliced["Nombre de trains en retard à l'arrivée"]
    grouped = df_sliced.groupby(['Line', 'Month'])

    # Utilizy method to calculate weighted average, w being the series of weight, x being the series of percentages 
    def weighted_mean(x, w):
        # Ensure there's no division by zero
        if w.sum() == 0:
            return np.nan
        return (x * w).sum() / w.sum()

    # Apply custom weighted mean function to aggregate
    df_grouped = grouped.agg({
        "Nombre de trains en retard à l'arrivée": 'sum',
        'Nombre de circulations prévues': 'sum',
        'Prct retard pour causes externes': lambda x: weighted_mean(x, df_sliced.loc[x.index, 'weights']),
        'Prct retard pour cause infrastructure': lambda x: weighted_mean(x, df_sliced.loc[x.index, 'weights']),
        'Prct retard pour cause gestion trafic': lambda x: weighted_mean(x, df_sliced.loc[x.index, 'weights']),
        'Prct retard pour cause matériel roulant': lambda x: weighted_mean(x, df_sliced.loc[x.index, 'weights']),
        'Prct retard pour cause gestion en gare et réutilisation de matériel': lambda x: weighted_mean(x, df_sliced.loc[x.index, 'weights']), 
        'Prct retard pour cause prise en compte voyageurs (affluence, gestions PSH, correspondances)': lambda x: weighted_mean(x, df_sliced.loc[x.index, 'weights'])
    }).reset_index()

    # Calculate 'cause_unknown' for each group to ensure causes of delay go to 100%
    df_grouped['cause_unknown'] = 100 - df_grouped[cause_columns].sum(axis=1)
    # Ensure 'cause_unknown' doesn't go below 0 
    df_grouped['cause_unknown'] = df_grouped['cause_unknown'].clip(lower=0)
    cause_columns.append('cause_unknown')  # Add the "cause_unknown" to the causes column
    
    try: 
        route_data = df_grouped.loc[(df_grouped['Line'] == route) & (df_grouped['Month'] == month)]
        # Get the average of percentages for the selected route
        if route_data.empty:
            raise ValueError(f"No data available for route {route} and month {month}.")

        # Aggregate data if necessary (here we take the mean, but choose the aggregation that makes sense for your data)
        route_data = route_data.mean()

        # Data for plotting
        lines = route_data.index
        fig, ax = plt.subplots()
        cum_val = np.zeros(len(lines))

        # Plot each layer of the stacked bar
        for i, cause in enumerate(cause_columns):
            ax.bar(route, route_data[cause], bottom=cum_val, label=cause)
            cum_val += route_data[cause]

        # Chart display parameters
        ax.set_ylabel('Percentage')
        ax.set_title(f'Past causes of delay for the route {route} in month {month}')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.75, 1]) # to ensure the legend appears fully

        # Directory to store the chart
        static_dir = 'static'
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)  # This will create the static directory if it doesn't exist
        
        # Save the plot
        base_path = os.path.abspath(os.path.dirname(__file__)) 
        static_path = os.path.join(base_path, 'static')  # Path to the static directory
        file_name = f"{route.replace(' - ', '_').replace(' ', '_')}_month_{month}_delay_causes.png"
        file_path = os.path.join(static_path, file_name)
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error generating chart: {e}")
 


# _____ MAIN PROGRAM _____

# Read file 
file_path = 'data/regularityTrainData.csv'

# Will not treat the problematic lines
df = pd.read_csv(file_path, on_bad_lines='skip')
df = pd.read_csv(file_path, delimiter=';')

df_sliced = reshape_data(df)

# Create the encoder : as the models need numeric data
encoder = LabelEncoder()
encoder.fit(df_sliced['Line'])

# Save the fitted encoder
joblib.dump(encoder, 'line_encoder.joblib')


df_sliced['Line_encoded'] = encoder.fit_transform(df_sliced['Line'])

# Add a column "delay_ratio" to the DataFrame
df_sliced = is_late_prediction_m(df_sliced)


# Initialize your Flask application
app = Flask(__name__, static_folder='static', static_url_path='/static')

# HTML template for the web page
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
<title>Train Delay Predictor</title>
</head>
<body>
    {% if not prediction %}
    <h2>Train Delay Prediction</h2>
    <form method="post">
        <input type="hidden" name="form_type" value="predict">
        Route: <input type="text" name="route"><br><br>
        Month: <input type="number" name="month"><br><br>
        <input type="submit" value="Predict Delay">
    </form><br><br><br>

    <h2>Help us by providing your train's experience!</h2>
    <form method="post">
        <input type="hidden" name="form_type" value="submit_data">
        <input type="text" name="departure" placeholder="DEPARTURE STATION" required><br>
        <input type="text" name="arrival" placeholder="ARRIVAL STATION" required><br>
        <input type="number" name="month" placeholder="Month of Journey" min="1" max="12" required><br>
        <input type="number" name="delay_amount" placeholder="Delay Amount in minutes" required><br>
        Cause of Delay: 
        <select name="delay_cause">
            {% for cause in cause_columns %}
            <option value="{{ cause }}">{{ cause }}</option>
            {% endfor %}
            <option value="Unknown">Unknown</option>
        </select><br>
        <input type="submit" value="Submit Data">
    </form>
    {% endif %}

    {% if prediction %}
    <h3>Prediction for route {{ route }}: {{ prediction }}</h3><br>
    <h4>In the past, {{ delay_ratio }}% of the trains were late.</h4><br><br>
    {% endif %}
    {% if image_path %}
    <h4>Past causes of delay for this route</h4>
    <img src="{{ url_for('static', filename=image_path) }}" alt="Delay Causes Chart">
    {% endif %}
    {% if feedback_message %}
        <p>{{ feedback_message }}</p>
    {% endif %}
    
</body>
</html>

'''



# Create the web page and predict the amount of delay when late
@app.route('/', methods=['GET', 'POST'])
def predict():
    global df_sliced
    prediction = ""
    image_path = ""
    route = ""
    delay_ratio_example = None
    month_error = "" 
    feedback_message = ""
    cause_columns = [
        'Prct retard pour causes externes',
        'Prct retard pour cause infrastructure',
        'Prct retard pour cause gestion trafic',
        'Prct retard pour cause matériel roulant',
        'Prct retard pour cause gestion en gare et réutilisation de matériel',
        'Prct retard pour cause prise en compte voyageurs (affluence, gestions PSH, correspondances)'
    ]
    form_type = request.form.get('form_type', 'predict')

    if request.method == 'POST':
        # Handle the preiction of delay for a route and a month
        if form_type == 'predict':
            route = request.form['route'].upper()  # Convert route to uppercase (if needed)
            month_str = request.form['month']  # Get month as string
            
            # Check if month input is a digit and within the valid range
            if month_str.isdigit():
                month = int(month_str)
                if not 1 <= month <= 12:
                    month_error = "Month should be between 1 and 12."
            else:
                month_error = "Invalid month. Please enter a numeric value."
            
            # If month is valid
            if not month_error:
                
                filtered_data = df_sliced[(df_sliced['Line'].str.upper() == route) & (df_sliced['Month'] == month)]

                if not filtered_data.empty:
                    delay_ratio_example = filtered_data['delay_ratio'].iloc[0]
                else:
                    delay_ratio_example = None

                if delay_ratio_example is not None and delay_ratio_example > 10: # Set thresold to determine if the train will be late (if more than 10% of the past trains were late, then predict this one will be late too)
                    prediction = predict_amount_of_delay(route, month, delay_ratio_example, df_sliced)
                    
                    generate_stacked_bar_chart(route, df_sliced, month)  

                    # Create path for bar graph for this route
                    image_path = f"{route.replace(' - ', '_').replace(' ', '_')}_month_{month}_delay_causes.png"

                elif delay_ratio_example is not None:
                    prediction = "The train is predicted to be on time."

                else:
                    prediction = "No data available for the selected route and month."
            else:
                prediction = month_error  # The month input is not valid
        
        # Handle the submission of data
        elif form_type == 'submit_data':
            departure = request.form['departure'].upper()
            arrival = request.form['arrival'].upper()
            month = int(request.form['month'])
            delay_amount = float(request.form['delay_amount'])
            selected_cause = request.form['delay_cause']
            
            if 1 <= month <= 12 and delay_amount >= 0:
                # Append new data to df_sliced
                if selected_cause != "unknown": 
                    new_row = {'Line': f"{departure} - {arrival}", 'Month': month, 
                            "Retard moyen de tous les trains à l'arrivée": delay_amount, 
                            'Nombre de circulations prévues': 1, 
                            "Nombre de trains en retard à l'arrivée": 1 if delay_amount > 0 else 0,  
                            selected_cause: 1
                            }
                    df_sliced = df_sliced.append(new_row, ignore_index=True)

                else:
                    new_row = {'Line': f"{departure} - {arrival}", 'Month': month, 
                            "Retard moyen de tous les trains à l'arrivée": delay_amount, 
                            'Nombre de circulations prévues': 1, 
                            "Nombre de trains en retard à l'arrivée": 1 if delay_amount > 0 else 0
                            }
                    df_sliced = df_sliced.append(new_row, ignore_index=True)
                
                # Update delay_ratio prediction based on this new information
                df_sliced.drop(columns=['delay_ratio'], inplace = True) 
                df_sliced = is_late_prediction_m(df_sliced)
                
                feedback_message = "Thank you! Your data has been submitted successfully."
            else:
                feedback_message = "Please ensure all fields are filled out correctly."


    

    return render_template_string(HTML_TEMPLATE, route=route, prediction=prediction, image_path=image_path, delay_ratio=format(delay_ratio_example, '.2f') if delay_ratio_example is not None else "N/A", feedback_message=feedback_message, cause_columns = cause_columns)


if __name__ == '__main__':
    app.run(debug=True, threaded=False)
