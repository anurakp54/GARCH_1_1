from datetime import date, timedelta
import streamlit as st
import yfinance as yf
import numpy as np
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot as plt
import pandas as pd
import altair as alt

st.set_page_config(
    layout= "wide",
    page_title= "Multipage App",
    page_icon = "🚂",
)

st.title("Assignment FIN S53 MODULE 4")
st.subheader("By Anurak-JOE")
st.write("The volatility plot and its prediction using model GARCH (1,1)")
stock_name = st.sidebar.text_input("Put Stock Name and adding '.BK' for Thai Stock", value ="CK.BKK")

today = str(date.today())
start_date = str(date.today() - timedelta(days=1200))
# df = web.DataReader(name, data_source='yahoo', start=start_date, end=today)
df = yf.download(stock_name, start=start_date, end=today)

def _return(dataframe):
  row = dataframe.index
  for i in range(len(row)-1):
    dataframe.loc[row[i+1],'ln_return'] = 100*np.log(dataframe.loc[row[i+1],'Close']/dataframe.loc[row[i],'Close'])
    dataframe.loc[row[i+1],'pct_return'] = 100*(dataframe.loc[row[i+1],'Close']-dataframe.loc[row[i],'Close'])/dataframe.loc[row[i],'Close']
    dataframe.loc[row[i+1],'sq_ln_return'] = dataframe.loc[row[i+1],'ln_return']*(dataframe.loc[row[i+1],'ln_return'])
  return dataframe


on_click = st.button('Proceed', )

if on_click:

    dataframe = _return(df)
    source = dataframe.copy()
    print(source.reset_index())
    st.subheader("Volatility Plot")
    line = alt.Chart(source.reset_index()).mark_line(interpolate='basis').encode(
        x = alt.X('Date:N',axis = alt.Axis(labels=False)),
        y = 'ln_return:Q'
    ).properties(
        width = 600,
        height = 250
    )
    line

    # show plot PACF

    garch_11 = arch_model(dataframe['ln_return'][1::], mean='constant', vol='GARCH', p=1, q=1)
    model_fit = garch_11.fit(disp='off')
    st.subheader("GARCH Model Parameters")
    st.write(model_fit.summary())

    plot_pacf(dataframe['sq_ln_return'][1::])


    dataframe.dropna(inplace = True)
    rolling_predict = []
    test_size = dataframe.shape[0]-1

    for i in range(test_size):
      train = dataframe['ln_return'][:-(test_size-i)]
      model = arch_model(train, p=1, q=1)
      model_fit = model.fit(disp='off')
      pred = model_fit.forecast(horizon= 1)
      rolling_predict.append(np.sqrt(pred.variance.values[-1,:][0]))

    rolling_predict = pd.Series(rolling_predict,index = dataframe.index[-1*test_size:])

    #plt.figure(figsize = (15,6))
    #true, = plt.plot(dataframe['ln_return'][-(test_size):])
    #preds = plt.plot(rolling_predict)
    #plt.title(f'Volatility Prediction of {name}')
    #plt.legend('True Returns','Prediction')
    #plt.show()
    true = dataframe[-(test_size):].copy()
    line2 = alt.Chart(true.reset_index()).mark_line(interpolate='basis').encode(
        x = alt.X('Date:N',axis=alt.Axis(labels=False)),
        y = 'ln_return:Q'
    ).properties(
        width = 600,
        height = 250
    )

    predict_df = pd.DataFrame(rolling_predict, columns=['predict'])
    line3 = alt.Chart(predict_df.reset_index()).mark_line(interpolate='basis').encode(
        x=alt.X('Date:N', axis=alt.Axis(labels=False)),
        y='predict:Q',
        color = alt.value("#FFAA00")
    ).properties(
        width=600,
        height=250
    )
    line2 + line3