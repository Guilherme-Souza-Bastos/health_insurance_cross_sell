# python Version - Python 3.9.12
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image



st.set_page_config(page_title="Health Insurance Cross Sell Results")
st.write("## Health Insurance Cross Sell Profit Results")

st.sidebar.header('How many calls should be made?')
n_calls = st.sidebar.slider('', 1, 28296, 5000)
n_calls = n_calls
#n_calls = n_calls-1

df_profit = pd.read_csv('dataset/df_profit.csv')
df_profit.columns = ["Model Profit"]
df_profit = round(df_profit/81.7,2)
#df_profit
fig = px.line(df_profit.iloc[:n_calls,0], y="Model Profit")
#fig
#st.write("Model profit was: US$ {}".format(df_profit.iloc[28296,0]))

df_random = pd.read_csv('dataset/df_random.csv')
df_random = df_random.iloc[0:28297,:]
df_random.columns = ["Random Profit"]
df_random = round(df_random/81.7,2)
fig2 = px.line(df_random.iloc[:n_calls,0], y="Random Profit")
fig2.update_traces(line_color='red')
#fig2
#st.write("Random profit was: US$ {}".format(df_random.iloc[28296,0]))

fig3 = go.Figure(data=fig.data+fig2.data)
fig3.update_xaxes(title_text="Number of Phone Calls")
fig3.update_yaxes(title_text="Profit")
fig3
st.markdown("The LightGBM Model outperformes randomly calling our customers by: **US$ {}**".format(round(df_profit.iloc[n_calls,0]-df_random.iloc[n_calls,0],2)))

st.write("## Donwload the CSV File")
df_download = pd.read_csv("dataset/df11.csv")
df_download = df_download[['response','prediction']]
df_download = df_download.iloc[0:28297,:]
df_download.columns = ["Response", "Model Prediction"]
df_download = pd.concat([df_download,df_profit,df_random], axis=1)
df_download = df_download.iloc[:n_calls,:]
# df_download.shape
st.dataframe(df_download,width=450, height=423)

st.download_button(
	label="Download data as CSV",
	data=df_download.to_csv(),
	file_name="comparison_dataframe.csv",
    mime="text/csv"
)
