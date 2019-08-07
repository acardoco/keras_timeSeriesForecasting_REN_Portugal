import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error

columnas_factores = [3, 8, 13, 18, 23]
fichero_factores = 'excel/factor_adecuaçao.xlsx'

# specify the number of lag hours
n_hours = 1 # hasta (t - n_hours)
n_features = 4 # variables
n_obs = n_hours * n_features

ventana = 24


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def to_stationary(ts):

    ts = pd.DataFrame(ts)
    ts_log = np.log(ts)
    moving_avg = ts_log.rolling(min_periods=1, center=True, window=ventana).mean()

    # quitando rolling mean
    ts_log_moving_avg_diff = ts_log - moving_avg
    ts_log_moving_avg_diff = ts_log_moving_avg_diff.dropna()

    plt.plot(ts)
    plt.plot(ts_log_moving_avg_diff, color='green')
    plt.plot(moving_avg, color='red')
    plt.title('Estacionaria')
    plt.show()

    return ts_log_moving_avg_diff.values
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def plot_dataset(dataset):

    # specify columns to plot
    groups = np.arange(n_features)
    i = 1
    # plot each column
    plt.figure()
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(dataset[:, group])
        plt.title(group, y=0.5, loc='right')
        i += 1
    plt.show()
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def pre_proccesing_and_train(fichero_factores):

    dataframe_factores = pd.read_excel(fichero_factores, sheet_name='pasado')

    factores = dataframe_factores[['ENE18', 'MAR18', 'ABR18', 'MAY18']].values
    factores = factores.transpose()
    factores = factores.flatten()
    factores = factores[np.logical_not(np.isnan(factores))]

    temperaturas = dataframe_factores[['TEMP_ENE18', 'TEMP_MAR18', 'TEMP_ABR18', 'TEMP_MAY18']].values
    temperaturas = temperaturas.transpose()
    temperaturas = temperaturas.flatten()
    temperaturas = temperaturas[np.logical_not(np.isnan(temperaturas))]

    festivos = dataframe_factores[['FESTIVO_ENE18', 'FESTIVO_MAR18', 'FESTIVO_ABR18', 'FESTIVO_MAY18']].values
    festivos = festivos.transpose()
    festivos = festivos.flatten()
    festivos = festivos[np.logical_not(np.isnan(festivos))]

    demandas = dataframe_factores[['DEMANDA_ENE18', 'DEMANDA_MAR18', 'DEMANDA_ABR18', 'DEMANDA_MAY18']].values
    demandas = demandas.transpose()
    demandas = demandas.flatten()
    demandas = demandas[np.logical_not(np.isnan(demandas))]

    # lo vuelvo estacionario
    factores = to_stationary(factores)
    factores = factores.flatten()

    dataset = np.vstack((factores, temperaturas, festivos, demandas))
    dataset = dataset.transpose()

    values = dataset
    # integer encode direction (para strings, en este caso no hace falta)
    '''encoder = LabelEncoder()
    values[:, 3] = encoder.fit_transform(values[:, 3])'''
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_hours, 1)
    # drop columns we don't want to predict
    if n_hours == 1:
        reframed.drop(reframed.columns[[5, 6, 7]], axis=1, inplace=True)
    print(reframed.head())

    # split into train and test sets
    values = reframed.values
    n_train_hours = 2208 # 2208
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    if n_hours == 1:
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    else:
        train_X, train_y = train[:, :n_obs], train[:, -n_features]
        test_X, test_y = test[:, :n_obs], test[:, -n_features]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
        test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=0,
                        shuffle=False)
    '''# plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()'''

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    print(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    # quitamos el ultimo por el descuadre (en la fila de validacion el factor es el actual mientras
    # que en el de prediccion es el siguiente
    rmse = np.sqrt(mean_squared_error(inv_y[:-1], inv_yhat[1:]))
    print('Test RMSE: %.3f' % rmse)

    # reconstruyendo
    yhat_exp = np.exp(inv_yhat)

    # dibujamos
    plt.plot(np.exp(inv_y[:-1]))
    plt.plot(yhat_exp[1:], color='green') #subimos por el descuadre al dibujar
    plt.show()

    np.savetxt('txt/predicciones_factores_validacion.txt', yhat_exp, newline='\n')

    return model
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def predecir_siguiente(dataset):

    # los transformo para el modelo
    values = dataset
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_hours, 1)
    # drop columns we don't want to predict
    if n_hours == 1:
        reframed.drop(reframed.columns[[5, 6, 7]], axis=1, inplace=True)

    # split into train and test sets
    test = reframed.values
    if n_hours == 1:
        test_X = test[:, :-1]
        # reshape input to be 3D [samples, timesteps, features]
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    else:
        test_X = test[:, :n_obs]
        test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

    # hago la prediccion
    yhat = model.predict(test_X)
    if n_hours == 1:
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    else:
        test_X = test_X.reshape((test_X.shape[0], n_obs))
    # invert scaling for forecast
    if n_hours == 1:
        inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    else:
        inv_yhat = np.concatenate((yhat, test_X[:, -3:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # reconstruyendo
    yhat_exp = np.exp(inv_yhat)

    return yhat_exp
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def get_filas(fichero_factores, num):

    df_mes = pd.read_excel(fichero_factores, sheet_name='pasado')

    factores = df_mes['ENE18'].values
    factores = factores[np.logical_not(np.isnan(factores))]
    temperaturas_mes = df_mes['TEMP_ENE18'].values
    temperaturas_mes = temperaturas_mes[np.logical_not(np.isnan(temperaturas_mes))]
    festivos_mes = df_mes['FESTIVO_ENE18'].values
    festivos_mes = festivos_mes[np.logical_not(np.isnan(festivos_mes))]
    demandas_mes = df_mes['DEMANDA_ENE18'].values
    demandas_mes = demandas_mes[np.logical_not(np.isnan(demandas_mes))]

    dataset_mes = np.vstack((factores, temperaturas_mes, festivos_mes, demandas_mes))
    dataset_mes = dataset_mes.transpose()

    return dataset_mes[:num, :]

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def predict_enero(mes, year, hoja):

    print('*********************************************************')
    print('*********************************************************')
    print('*********************************************************')
    # *******************************************************************
    # ******************************************************************
    # ******************************************************************
    #leo los que quiero predecir (en este caso may18)
    df_mes = pd.read_excel(fichero_factores, sheet_name=hoja)
    temperaturas_mes = df_mes['TEMP_' + mes + str(year)].values
    temperaturas_mes = temperaturas_mes[np.logical_not(np.isnan(temperaturas_mes))]
    festivos_mes = df_mes['LAB_' + mes + str(year)].values
    festivos_mes = festivos_mes[np.logical_not(np.isnan(festivos_mes))]
    demandas_mes = df_mes['DEMANDA_' + mes + str(year)].values
    demandas_mes = demandas_mes[np.logical_not(np.isnan(demandas_mes))]

    # inicializo con el mes pasado de 2018
    df_pasado = pd.read_excel(fichero_factores, sheet_name='pasado')
    factores_iniciales = df_pasado[mes + '18'].values
    factores_iniciales = factores_iniciales[np.logical_not(np.isnan(factores_iniciales))]
    # lo vuelvo estacionario
    factores_iniciales = to_stationary(factores_iniciales)
    factores_iniciales = factores_iniciales.flatten()

    dataset_mes = np.vstack((factores_iniciales, temperaturas_mes, festivos_mes, demandas_mes))
    dataset_mes = dataset_mes.transpose()

    # preparo el array y predigo la siguiente hora
    fac_pred_sig = predecir_siguiente(dataset_mes)

    # y guardo
    np.savetxt('txt/predicciones_factores.txt', fac_pred_sig, newline='\n')


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
model = pre_proccesing_and_train(fichero_factores)
predict_enero('ENE', 19, 'esperado')