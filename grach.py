import pandas as pd
import numpy as np
from datetime import date, timedelta
import yfinance as yf
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot as plt

today = str(date.today())
start_date = str(date.today() - timedelta(days=1200))
# df = web.DataReader(name, data_source='yahoo', start=start_date, end=today)
name = 'PTT.BK'
df = yf.download(name, start=start_date, end=today)

def _return(dataframe):
  row = dataframe.index
  for i in range(len(row)-1):
    dataframe.loc[row[i+1],'ln_return'] = 100*np.log(dataframe.loc[row[i+1],'Close']/dataframe.loc[row[i],'Close'])
    dataframe.loc[row[i+1],'pct_return'] = 100*(dataframe.loc[row[i+1],'Close']-dataframe.loc[row[i],'Close'])/dataframe.loc[row[i],'Close']
    dataframe.loc[row[i+1],'sq_ln_return'] = dataframe.loc[row[i+1],'ln_return']*(dataframe.loc[row[i+1],'ln_return'])
  return dataframe

returns = 100*df['Close'].pct_change().dropna()
print(returns)
plt.figure(figsize = (10,4))
plt.plot(returns)

dataframe = _return(df)
source = dataframe.copy()
print(source.reset_index())
plot_pacf(dataframe['sq_ln_return'][1::])
plt.show()

garch_11= arch_model(dataframe['ln_return'][1::],mean= 'constant',vol= 'GARCH', p=1, q =1)
model_fit=garch_11.fit(disp='off')
model_fit.summary()

print(dataframe.tail())
splitdate= dataframe.index[-1]
print(splitdate)
predict= model_fit.forecast(horizon=5,start= splitdate, method = 'simulation')
print(predict.variance[splitdate:])
print(predict.variance.tail())

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
plt.figure(figsize = (15,6))
true, = plt.plot(dataframe['ln_return'][-(test_size):])
preds = plt.plot(rolling_predict)
plt.title(f'Volatility Prediction of {name}')
plt.legend('True Returns','Prediction')
plt.show()

rolling_predict.name = 'predict'
predict_df = pd.DataFrame(rolling_predict,columns=['predict'])
print(predict_df)