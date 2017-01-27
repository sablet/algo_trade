# S&P500 predict

### daily stock [Up/Down] predict by LSTM
 
* env
    * python: 3.5.2
    * OS: Ubuntu16.04
    * GPU: GeForce GTX 980
* data source   
  * S&P500 symbols list
    * from http://data.okfn.org/data/core/s-and-p-500-companies#data
  * S&P500 stcoks data (from yahoo finance)
    * daily ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'] values
* preprocess, analysis method are written main-process.ipynb
* result
  * whole direct accuracy is 0.841
  * whole profit ration is 1.011

