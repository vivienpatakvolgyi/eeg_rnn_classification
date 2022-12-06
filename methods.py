def append_time_series(df):
  X = []
  Y = []
  size= 1
  #df_y= df_y.drop('UserId', axis =1)
  for i in range(0,len(df)-size, 1):
    X.append(np.array(df[i:i+size].drop(df.columns[112:], axis =1)).reshape(112))
    label=df.values[i+size][-1].astype(float)
    Y.append(float(label))
    
  X= np.array(X)
  Y= np.array(Y)
  
  return X, Y

def MSE_R(true, prediction):
    st.write(f"MSE: %.3f" % mean_squared_error(true, prediction))
    metric = tfa.metrics.r_square.RSquare()
    metric.update_state(np.array(true), np.array(prediction))
    result = metric.result()
    st.write(f"R\u00B2: %.3f" % result.numpy())

def show_results(true, prediction):
    results = pd.DataFrame()
    results['Prediction'] = prediction
    results['True value'] = true
    st.write(results)

def predict(X_predict, y_predict, file):
    model = load_model(file)

    predicted_vals = []

    for i in range(len(X_predict)):
        pred_test = model.predict(X_predict[i])[-1]
        predicted_vals.append(pred_test)

    true = []
    prediction = []

    correct = 0
    for i in range(len(predicted_vals)):
        prediction.append(list(predicted_vals[i]).index(max(predicted_vals[i])))
        true.append(int(y_predict[i][-1]))
        if int(list(predicted_vals[i]).index(max(predicted_vals[i]))) == int(y_predict[i][-1]):
            correct += 1

    st.write("Correct predictions: ", "{:.0%}".format(correct/len(true)))

    
    show_results(true, prediction)
    MSE_R(true, prediction)