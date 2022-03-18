from json import load
from multiprocessing import Pool
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, InputLayer, Flatten, Reshape, Lambda, Dropout
from tensorflow import expand_dims
from scipy.stats import pearsonr
import shap
import lime
import lime.lime_tabular


CPU_CORES = 8

sb.set_theme()
# sb.set(font_scale=3)

sst_data_raw = pd.read_csv('./SST_data.csv')
soil_data_raw = pd.read_csv('./Soil_data.csv')

sst_data = sst_data_raw.drop(sst_data_raw.columns[0], axis=1)
sst_lon_original = sst_data['Lon']
sst_lat_original = sst_data['Lat']
sst_lon = ((sst_data['Lon'] - sst_data['Lon'].min()) / 2).astype('int32').to_list()
sst_lat = ((sst_data['Lat'] - sst_data['Lat'].min()) / 2).astype('int32').to_list()

sst_data = sst_data.iloc[:, 2:]
sst_data = sst_data / 12.0 # Valid range -> -12 to 12

soil_data = soil_data_raw.query('-101.5 <= Lon <= -80.5 and 35.5 <= Lat <= 48.5') # Corn belt
soil_lon = ((soil_data['Lon'] + 101.5) / 0.5).astype('int32').to_list()
soil_lat = ((soil_data['Lat'] - 35.5) / 0.5).astype('int32').to_list()
soil_data.reset_index(drop=True, inplace=True)

soil_data = soil_data.iloc[:, 2:]
soil_data = soil_data.transpose() / 740.0 # Valid range -> 0 to 740

sst_X = max(sst_lat) + 1
sst_Y = max(sst_lon) + 1
soil_X = max(soil_lat) + 1
soil_Y = max(soil_lon) + 1
print(f'SST X-Y: ({sst_X}, {sst_Y})')
print(f'Soil X-Y: ({soil_X}, {soil_Y})')

def sst_reshape(index):
    col = sst_data.iloc[:, index].to_list()
    # Embedding data in array according to lon-lat
    arr = np.zeros((sst_X, sst_Y))
    for j,r in enumerate(col):
        x = sst_lat[j]
        y = sst_lon[j]
        arr.itemset((x, y), r)
    # Flattening the array for LIME/SHAP to work
    return arr.flatten()

def soil_reshape(index):
    col = soil_data.iloc[:, index].to_list()
    # Embedding data in array according to lon-lat
    arr = np.zeros((soil_X, soil_Y))
    for j,r in enumerate(col):
        x = soil_lat[j]
        y = soil_lon[j]
        arr.itemset((x, y), r)
    # Flattening the array for LIME/SHAP to work
    return arr.flatten()

def expand_layer(tensor):
    return expand_dims(tensor, axis=3)

if __name__ == '__main__':
    # Reshaping SST Data
    # print(f'SST Data [Raw] shape: {sst_data.shape}')
    # pool = Pool(CPU_CORES)
    # sst_data_2d_frames = pool.map(sst_reshape, range(sst_data.shape[1]))
    # pool.close()
    # sst_data_2d = np.stack(sst_data_2d_frames)
    # joblib.dump(sst_data_2d, './sst_data_2d_flat.dat')
    sst_data_2d = joblib.load('./sst_data_2d_flat.dat')
    # print(f'Reshaped SST Data shape: {sst_data_2d.shape}')
    # Reshaped SST Data shape: (888, 46, 84)

    # Reshaping Soil data
    # print(f'Soil Data [Raw] shape: {soil_data.shape}')
    # pool = Pool(CPU_CORES)
    # soil_data_2d_frames = pool.map(soil_reshape, range(soil_data.shape[1]))
    # pool.close()
    # soil_data_2d = np.stack(soil_data_2d_frames)
    # joblib.dump(soil_data_2d, './soil_data_2d_flat.dat')
    # # soil_data_2d = joblib.load('./soil_data_2d_flat.dat')
    # print(f'Reshaped Soil Data shape: {soil_data_2d.shape}')
    # Reshaped Soil Data shape: (888, 27, 43)


    sst_train = sst_data_2d[:792]
    sst_validation = sst_data_2d[792:816]
    sst_test = sst_data_2d[792:885]
    soil_train = soil_data.iloc[3:795, :].reset_index(drop=True)
    soil_validation = soil_data.iloc[795:819, :].reset_index(drop=True)
    soil_test = soil_data.iloc[795:888, :].reset_index(drop=True)

    print(f'SST Train: {sst_train.shape}, Validation: {sst_validation.shape}, Test:{sst_test.shape}')
    print(f'Soil Train: {soil_train.shape}, Validation: {soil_validation.shape}, Test:{soil_test.shape}')

    '''
    # Setting up model
    model = Sequential()
    model.add(InputLayer(input_shape=(3864)))
    model.add(Reshape((46, 84)))
    model.add(Lambda(expand_layer))
    model.add(Conv2D(64, kernel_size=2, activation='tanh', padding='same'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=2, activation='tanh', padding='same'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(2250, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1125, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(sst_train, soil_train, epochs=1000, validation_data=(sst_validation, soil_validation))
    model.fit(sst_train, soil_train, epochs=2000, validation_data=(sst_test, soil_test))
    print(model.summary())
    score = model.test_on_batch(sst_test, soil_test)
    print(f'loss: {score}')
    model.save('CNN_Model')
    # exit(0)
    '''

    model = load_model('CNN_Model')
    print(model.summary())
    score = model.test_on_batch(sst_test, soil_test)
    print(f'loss: {score}')
    pred = model.predict(sst_test)

    sst_row_labels = [(x*2)+sst_data_raw['Lat'].min() for x in list(range(46))]
    sst_col_labels = [(x*2)+sst_data_raw['Lon'].min() for x in list(range(84))]


#--------------- Pred vs True -----------------------
    plt.figure(figsize=(5, 3), constrained_layout=True)
    for i in range(sst_test.shape[0]):
        sb.scatterplot(x=(soil_test.iloc[i, :]*740).to_list(), y=pred[i]*740, color='cornflowerblue')
    p = sb.lineplot(x=[0.0, 740.0], y=[0.0, 740.0], color='black')
    p.set_xlabel('Actual')
    p.set_ylabel('Predicted')
    p.set_title('Predicted vs Actual [All values]')
    p.get_figure().savefig('./plot_Shapley_CNN_pred_vs_true_all.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(5, 3), constrained_layout=True)
    for i in range(1, sst_test.shape[0], 12):
        sb.scatterplot(x=(soil_test.iloc[i, :]*740).to_list(), y=pred[i]*740, color='cornflowerblue')
    p = sb.lineplot(x=[0.0, 740.0], y=[0.0, 740.0], color='black')
    p.set_xlabel('Actual')
    p.set_ylabel('Predicted')
    p.set_title('Predicted vs Actual [May values]')
    p.get_figure().savefig('./plot_Shapley_CNN_pred_vs_true_Feb.png', dpi=300, bbox_inches='tight')
    # exit(0)

#--------------------- Pearson correlation ----------------------
    soil_test_flat_all = soil_test.to_numpy().flatten() * 740.0
    pred_flat_all = pred.flatten() * 740.0
    corr, pvalue = pearsonr(x=soil_test_flat_all, y=pred_flat_all)
    print(f'Pearson correlation [all values]\ncorr: {corr}, R^2: {corr*corr}, p-value: {pvalue}')

    soil_test_flat_feb = soil_test.to_numpy()[list(range(1, soil_test.shape[0], 12))].flatten() * 740.0
    pred_flat_feb = pred[list(range(1, pred.shape[0], 12))].flatten() * 740.0
    corr, pvalue = pearsonr(x=soil_test_flat_feb, y=pred_flat_feb)
    print(f'Pearson correlation [feb values]\ncorr: {corr}, R^2: {corr*corr}, p-value: {pvalue}')

#----------------- Mean Square Error -------------------------------
    mse_all = np.square(np.subtract(soil_test_flat_all, pred_flat_all)).mean()
    print(f'MSE [all]: {mse_all}')
    mse_feb = np.square(np.subtract(soil_test_flat_feb, pred_flat_feb)).mean()
    print(f'MSE [feb]: {mse_feb}')

    # exit(0)

#--------------------- Shapley -------------------------
    # background = sst_train[list(range(1, sst_train.shape[0], 12))] # All the February data
    # exp = shap.DeepExplainer(model, background)
    # shap_values = exp.shap_values(sst_test[list(range(1, sst_test.shape[0], 12))]) # February data in test set
    # np.save('./Shapley_CNN_values', shap_values)
    shap_values = np.load('./Shapley_CNN_values.npy')
    print(shap_values.shape)
    # exit(0)
    shap_values_avg = np.mean(shap_values, axis=0)
    shap_values_avg = np.squeeze(shap_values_avg)
    print(shap_values_avg.shape)
    # print('Shapley values min, max: ({}, {})'.format(np.amin(shap_values_avg), np.amax(shap_values_avg)))

    # Mask for setting non-existent SST locations to 0
    mask_arr = np.zeros(shape=(sst_X, sst_Y), dtype=np.int32)
    for i,j in zip(sst_lat, sst_lon):
        mask_arr.itemset((i, j), 1)

    for i in range(shap_values_avg.shape[0]):
        shap_data = shap_values_avg[i].reshape(46, 84)
        shap_data = shap_data * mask_arr
        shap_data = pd.DataFrame(shap_data)
        shap_data.set_axis(sst_row_labels, axis=0, inplace=True)
        shap_data.set_axis(sst_col_labels, axis=1, inplace=True)
        print(f'Shapley values min, max: ({shap_data.min().min()}, {shap_data.max().max()})')
        # sb.set_theme(style='ticks')
        plt.figure(figsize=(5, 3), constrained_layout=True)
        # p = sb.heatmap(shap_data, cmap='bwr', vmin=-0.1, vmax=0.1, center=0)
        p = sb.heatmap(shap_data, cmap='bwr', center=0)
        p.invert_yaxis()
        p.set_xlabel('Longitude')
        p.set_ylabel('Latitude')
        # p.set_title(f'Shap values for SST [Feb->May, Year {i+2014}]')
        p.set_title(f'Shap values for SST [May {i+2014}]')
        for index, label in enumerate(p.get_xticklabels()):
            if index % 2 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)
        for index, label in enumerate(p.get_yticklabels()):
            if index % 2 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)
        p.get_figure().savefig(f'./Shapley_values_avg_{i+2014}.png', dpi=300, bbox_inches='tight')

    shap_values_mabs = np.abs(shap_values)
    shap_values_mabs = np.mean(shap_values_mabs, axis=0)
    shap_values_mabs = np.squeeze(shap_values_mabs)

    for i in range(shap_values_mabs.shape[0]):
        shap_data = pd.DataFrame(shap_values_mabs[i].reshape(46, 84))
        shap_data.set_axis(sst_row_labels, axis=0, inplace=True)
        shap_data.set_axis(sst_col_labels, axis=1, inplace=True)
        plt.figure(figsize=(5, 3), constrained_layout=True)
        p = sb.heatmap(shap_data, cmap='bwr', center=0)
        p.invert_yaxis()
        p.set_xlabel('Longitude')
        p.set_ylabel('Latitude')
        p.set_title(f'Absolute Shap values for SST [May {i+2014}]')
        for index, label in enumerate(p.get_xticklabels()):
            if index % 2 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)
        for index, label in enumerate(p.get_yticklabels()):
            if index % 2 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)
        p.get_figure().savefig(f'./Shapley_values_mabs_{i+2014}.png', dpi=300, bbox_inches='tight')


#--------------------- LIME -------------------------------
    
    # Mask for setting non-existent SST locations to 0
    mask_arr = np.zeros(shape=(sst_X, sst_Y), dtype=np.int32)
    for i,j in zip(sst_lat, sst_lon):
        mask_arr.itemset((i, j), 1)
    
    explainer = lime.lime_tabular.LimeTabularExplainer(sst_train, mode='regression', discretize_continuous=False)
    for month in range(1, sst_test.shape[0], 12):
        exp = explainer.explain_instance(sst_test[month], model.predict, num_features=3864)
        lime_data_raw = exp.as_list()
        # print(lime_data_raw)
        lime_data = [(int(x[0]), x[1]) for x in lime_data_raw]
        lime_data.sort(key=lambda x: x[0])
        lime_data = [x[1] for x in lime_data]
        lime_data = np.reshape(lime_data, newshape=(46, 84))

        lime_data = lime_data * mask_arr

        lime_data_df = pd.DataFrame(lime_data)
        lime_data_df.set_axis(sst_row_labels, axis=0, inplace=True)
        lime_data_df.set_axis(sst_col_labels, axis=1, inplace=True)
        print(f'LIME data min, max: ({lime_data_df.min().min()}, {lime_data_df.max().max()})')
        plt.figure(figsize=(5, 3), constrained_layout=True)
        # p = sb.heatmap(lime_data_df, cmap='bwr', vmin=-0.03, vmax=0.03, center=0)
        p = sb.heatmap(lime_data_df, cmap='bwr', center=0)
        p.invert_yaxis()
        p.set_xlabel('Lon')
        p.set_ylabel('Lat')
        p.set_title(f'LIME values for SST [May {2014 + (month//12)}]')
        for index, label in enumerate(p.get_xticklabels()):
            if index % 2 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)
        for index, label in enumerate(p.get_yticklabels()):
            if index % 2 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)
        p.get_figure().savefig(f'./plot_LIME_CNN_LIME_values_{2014 + (month//12)}.png', dpi=300, bbox_inches='tight')

        lime_data_abs = pd.DataFrame(np.abs(lime_data))
        # lime_data_abs = np.abs(lime_data)
        lime_data_abs.set_axis(sst_row_labels, axis=0, inplace=True)
        lime_data_abs.set_axis(sst_col_labels, axis=1, inplace=True)
        plt.figure(figsize=(5, 3), constrained_layout=True)
        # p = sb.heatmap(lime_data_abs, cmap='bwr', vmin=0.00, vmax=0.03, center=0)
        p = sb.heatmap(lime_data_abs, cmap='bwr', center=0)
        p.invert_yaxis()
        p.set_xlabel('Lon')
        p.set_ylabel('Lat')
        p.set_title(f'Absolute LIME values for SST [May {2014 + (month//12)}]')
        for index, label in enumerate(p.get_xticklabels()):
            if index % 2 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)
        for index, label in enumerate(p.get_yticklabels()):
            if index % 2 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)
        p.get_figure().savefig(f'./plot_LIME_CNN_LIME_values_abs_{2014 + (month//12)}.png', dpi=300, bbox_inches='tight')
