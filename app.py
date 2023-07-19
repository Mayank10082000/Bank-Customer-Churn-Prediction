from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model

with open('D:/Mayank/My Projects/[IMP] [DONE] Bank Customer Churn Prediction/bank_churn_lightgbm.pkl', 'rb') as f:
    lgb_model = pickle.load(f)

# Preprocessing functions
def preprocess_input(df):

    for c in df.columns:
        col_type = df[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            df[c] = df[c].astype('category')

    # return the preprocessed dataframe
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve user input from the form
    
    clientnum = int(request.form['clientnum'])
    customer_age = int(request.form['customer_age'])
    gender = request.form['gender']
    dependent_count = int(request.form['dependent_count'])
    education_level = request.form['education_level']
    marital_status = request.form['marital_status']
    income_category = request.form['income_category']
    card_category = request.form['card_category']
    months_on_book = int(request.form['months_on_book'])
    total_relationship_count = int(request.form['total_relationship_count'])
    months_inactive_12_mon = int(request.form['months_inactive_12_mon'])
    contacts_count_12_mon = int(request.form['contacts_count_12_mon'])
    credit_limit = float(request.form['credit_limit'])
    total_revolving_bal = int(request.form['total_revolving_bal'])
    avg_open_to_buy = float(request.form['avg_open_to_buy'])
    total_amt_chng_q4_q1 = float(request.form['total_amt_chng_q4_q1'])
    total_trans_amt = int(request.form['total_trans_amt'])
    total_trans_ct = int(request.form['total_trans_ct'])
    total_ct_chng_q4_q1 = float(request.form['total_ct_chng_q4_q1'])
    avg_utilization_ratio = float(request.form['avg_utilization_ratio'])

    # Create a dataframe from the user input
    data = {
        'Customer_Age': [customer_age],
        'Gender': [gender],
        'Dependent_count': [dependent_count],
        'Education_Level': [education_level],
        'Marital_Status': [marital_status],
        'Income_Category': [income_category],
        'Card_Category': [card_category],
        'Months_on_book': [months_on_book],
        'Total_Relationship_Count': [total_relationship_count],
        'Months_Inactive_12_mon': [months_inactive_12_mon],
        'Contacts_Count_12_mon': [contacts_count_12_mon],
        'Credit_Limit': [credit_limit],
        'Total_Revolving_Bal': [total_revolving_bal],
        'Avg_Open_To_Buy': [avg_open_to_buy],
        'Total_Amt_Chng_Q4_Q1': [total_amt_chng_q4_q1],
        'Total_Trans_Amt': [total_trans_amt],
        'Total_Trans_Ct': [total_trans_ct],
        'Total_Ct_Chng_Q4_Q1': [total_ct_chng_q4_q1],
        'Avg_Utilization_Ratio': [avg_utilization_ratio]
    }

    df = pd.DataFrame(data)
    print(df.info())
    # Preprocess the input data
    df_preprocessed = preprocess_input(df)

    # Perform prediction
    prediction = lgb_model.predict(df_preprocessed)

    # Convert prediction to a meaningful result
    if prediction == 'Attrited Customer':
        result = 'Attrited Customer'
    else:
        result = 'Existing Customer'

    # Return the prediction result to the user
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)