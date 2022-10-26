import pandas as pd
import numpy as np
import joblib
import streamlit

#load the model
model=open("RandomForestClassifierModel.pkl","rb")
RFClf_model=joblib.load(model)


def rf_prediction(var_1,var_2,var_3,var_4):
    pred_arr=np.array([var_1,var_2,var_3,var_4])
    preds=pred_arr.reshape(1,-1)
    preds=preds.astype(float)
    model_prediction=RFClf_model.predict(preds)
    return model_prediction

def run():
    streamlit.title("Random Forest Classifier Model")
    html_temp="""
    
    """
    
    streamlit.markdown(html_temp)
    var_1=streamlit.text_input("member_id_fctzd")
    var_2=streamlit.text_input("days_claimdt_startpolicy")
    var_3=streamlit.text_input("days_claimdt_discharge")
    var_4=streamlit.text_input("days_claimdt_endpolicy")
    
    prediction=""
    
    if streamlit.button("Predict"):
        prediction=rf_prediction(var_1,var_2,var_3,var_4)
        streamlit.success("The prediction by Model : {}".format(prediction))    
    
if __name__=='__main__':
    run()