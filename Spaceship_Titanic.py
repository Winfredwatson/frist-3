# https://docs.streamlit.io/en/stable/api.html#streamlit.slider
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# load model
@st.cache
def load_model():
    clf = joblib.load('data/model.joblib3')
    scaler = joblib.load('data/scaler.joblib3')
    return clf, scaler
    
def convert_CryoSleep(CryoSleep1):
    return 1 if CryoSleep1 == '是' else 0

def convert_Age(Age1):
    bins = [0, 12, 18, 25, 35, 60, 100]
    return pd.cut([Age1], bins, labels=range(len(bins)-1))[0]
     

dict1 = { 'Europa':0, 'Earth':1, 'Mars':2}
def convert_HomePlanet(embark1):
    return dict1[embark1]

dict2 = { 'TRAPPIST-1e':0, 'PSO J318.5-22':1, '55 Cancri e':2}
def convert_Destination(embark2):
    return dict2[embark2]    

dict3 = { 'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'T':7}
def convert_Deck(embark3):
    return dict3[embark3]  

dict4 = { 'P(左舷)':0, 'S(右舷)':1}
def convert_Side(embark4):
    return dict4[embark4] 
    
def convert_VIP(VIP1):
    return 1 if VIP1 == '是' else 0 
    
clf, scaler = load_model()    

# 畫面設計
st.markdown('# 生存預測系統')
Deck_series = pd.Series(['A','B','C','D','E','F','G','T'])
VIP_series = pd.Series(['是', '否'])
Side_series = pd.Series(['P(左舷)', 'S(右舷)'])
CryoSleep_series = pd.Series(['是', '否'])
HomePlanet_series = pd.Series(['Europa', 'Earth', 'Mars'])
Destination_series = pd.Series(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'])

# '冷凍睡眠:', CryoSleep
CryoSleep = st.sidebar.radio('冷凍睡眠:', CryoSleep_series)

# '年齡:', Age
Age = st.sidebar.slider('年齡', 0, 100, 20)

# 'VIP:', VIP
VIP = st.sidebar.selectbox('VIP', VIP_series)

# '家園星球:', HomePlanet
HomePlanet = st.sidebar.selectbox('家園星球:', HomePlanet_series)

# '目的星球:', Destination
Destination = st.sidebar.selectbox('目的星球:', Destination_series)

# '艙等:', Deck
Deck = st.sidebar.selectbox('艙等:', Deck_series)


# '舷，分別在船身的左及右邊:', Side
Side = st.sidebar.selectbox('舷，分別在船身的左及右邊:', Side_series)



st.image('./Spaceship Titanic.jpg')

if st.sidebar.button('預測'):
    # predict
    X = []
    #Deck Cabin_num Side CryoSleep	Age	HomePlanet Destination 
    X.append([convert_Deck(Deck), convert_Side(Side), convert_CryoSleep(CryoSleep) ,convert_VIP(VIP), convert_Age(Age), convert_HomePlanet(HomePlanet),convert_Destination(Destination)])
    X=scaler.transform(np.array(X))
    
    if clf.predict(X) == 1:
        st.markdown(f'### ==> **生存, 生存機率={clf.predict_proba(X)[0][1]:.2%}**')
    else:
        st.markdown(f'### ==> **死亡, 生存機率={clf.predict_proba(X)[0][1]:.2%}**')
