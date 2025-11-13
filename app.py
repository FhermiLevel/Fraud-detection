import streamlit as st
import pandas as pd
import numpy as np
import pickle




All_Featues = ['BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'ProviderId',
       'ProductId', 'ProductCategory', 'ChannelId', 'Value', 'PricingStrategy',
       'month', 'day', 'credit_or_debit']

CATEGORICAL_COLUMNS =['BatchId',
 'AccountId',
 'SubscriptionId',
 'CustomerId',
 'ProviderId',
 'ProductId',
 'ProductCategory',
 'ChannelId']




@st.cache_resource  #tells streamlit to load the model once and reuse them
def load_assets():
    model = pickle.load(open('Fraud_model.pkl', 'rb'))
    scaler = pickle.load(open('Fraud_scaler.pkl', 'rb'))
    encoder = pickle.load(open('Fraud_encoder.pkl', 'rb'))


    return model, encoder, scaler

model, encoder, scaler = load_assets()


def main():
    st.title("Fraud Detection Application")
    #organize inputs in a simple sidebars
    with st.sidebar:
        st.header("Input Transaction Details")
        value = st.number_input("value", min_value=2, value=200)
        month = st.slider("month", min_value=1, max_value=12, value=1)
        day = st.slider("day", min_value=1, max_value=31, value=1)
        Batch_Id = st.selectbox("BatchId", ("BatchId_36123"))
        Account_Id = st.selectbox("AccountId", ("AccountId_3957"))
        Subscription_Id = st.selectbox("SubscriptionId", ("SubscriptionId_887"))
        provider_Id = st.selectbox("ProviderId", ("ProviderId_6"))
        product_Id = st.selectbox("ProductId", ('ProductId_10', 'ProductId_6', 'ProductId_1', 'ProductId_21',
       'ProductId_3', 'ProductId_15', 'ProductId_11', 'ProductId_19',
       'ProductId_4', 'ProductId_5', 'ProductId_20', 'ProductId_9',
       'ProductId_24', 'ProductId_14', 'ProductId_2', 'ProductId_13',
       'ProductId_22', 'ProductId_8', 'ProductId_7', 'ProductId_27',
       'ProductId_12', 'ProductId_16', 'ProductId_23'))
        product_category = st.selectbox("ProductCategory",('airtime', 'financial_services', 'utility_bill', 'data_bundles',
        'tv', 'transport', 'ticket', 'movies', 'other'))
        channel_Id = st.selectbox("ChannelId", ('ChannelId_3', 'ChannelId_2', 'ChannelId_1', 'ChannelId_5'))
        pricing_strategy = st.selectbox("PricingStrategy(where 2 is online)", (2, 4, 1, 0))
        credit_or_debit = st.selectbox("credit_or_debit(Where 1 is debit and 0 is credit)", (0,1))
        customer_Id = st.selectbox("CustomerId", ("CustomerId_12345"))

    if st.button("Predict Fraud"):
        input_dict = {'BatchId': Batch_Id,
                      'AccountId': Account_Id,
                      'SubscriptionId': Subscription_Id,
                      'CustomerId': customer_Id,
                      'ProviderId': provider_Id,
                      'ProductId': product_Id,
                      'ProductCategory': product_category,
                      'ChannelId': channel_Id,
                      'Value': value,
                      'PricingStrategy': pricing_strategy,
                      'month': month,
                      'day': day,
                      'credit_or_debit': credit_or_debit
                     }
        input_df = pd.DataFrame([input_dict])
        #preprocess input data
        df_preprocessed = input_df.copy()
        for col in CATEGORICAL_COLUMNS:
            df_preprocessed[col] = encoder.fit_transform(df_preprocessed[col])
        
        all_inputs = df_preprocessed[All_Featues]

        #extract features into an array so as to be able to scale it
        all_inputs_array = all_inputs.values

        scaled_inputs_array = scaler.transform(all_inputs_array)

        #make prediction

        prediction = model.predict(scaled_inputs_array)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("The transaction is Fraudulent!")
        else:
            st.success("The transaction is Legitimate.")





if __name__ == "__main__":
    main()

    