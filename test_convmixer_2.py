#%%
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
import convmixer2

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
#%%
def train_cnn(training_df, test_df, params):
    """Trains and evaluates CNN on the given train and test data, respectively."""

    print("Training is starting ...")
    train_images = training_df.iloc[:, 2:].values  
    train_labels = training_df.iloc[:, 0]    
    train_prices = training_df.iloc[: ,1]

    test_images = test_df.iloc[:, 2:].values   
    test_labels = test_df.iloc[:, 0]   
    test_prices = test_df.iloc[:, 1]

    test_labels = keras.utils.np_utils.to_categorical(test_labels, params["num_classes"])
    train_labels = keras.utils.np_utils.to_categorical(train_labels, params["num_classes"])


    train_images = train_images.reshape(train_images.shape[0], params["input_w"], params["input_h"], 1)
    test_images = test_images.reshape(test_images.shape[0], params["input_w"], params["input_h"], 1)

    #predictions = transformer.transformer(train_images,train_labels,test_images,test_labels)

    convmixermodel = convmixer2.get_conv_mixer_256_8(image_size=15, num_classes=3)
    history, predictions = convmixer2.run_experiment(convmixermodel,train_images,train_labels,test_images,test_labels)


    return predictions, test_labels, test_prices

#%%
def reverse_one_hot(predictions):
    reversed_x = []
    for x in predictions:
        reversed_x.append(np.argmax(np.array(x)))
    return reversed_x
#%%
train_df = pd.read_csv("resources3/outputOfPhase2Training.csv", header=None, index_col=None, delimiter=';')
test_df = pd.read_csv("resources3/outputOfPhase2Test.csv", header=None, index_col=None, delimiter=';')

train_df = train_df.iloc[:,:-1]
test_df = test_df.iloc[:,:-1]
#%%
# drop nan values
train_df = train_df.dropna(axis=0)
test_df = test_df.dropna(axis=0)

# drop first 15 row
train_df = train_df.iloc[15:, :]
test_df = test_df.iloc[15:,:]

#%%

l0_train = train_df.loc[train_df[0] == 0]
l1_train = train_df.loc[train_df[0] == 1]
l2_train = train_df.loc[train_df[0] == 2]
l0_size = l0_train.shape[0]
l1_size = l1_train.shape[0]
l2_size = l2_train.shape[0]
#l0_l1_ratio = int((l0_size//l1_size)/4)
#l0_l2_ratio = int((l0_size//l2_size)/4)

l0_l1_ratio = (l0_size//l1_size)
l0_l2_ratio = (l0_size//l2_size)
print("Before")
print("l0_size:",l0_size,"l1_size:", l1_size,"l2_size:",l2_size)
print("l0_l1_ratio:",l0_l1_ratio,"l0_l2_ratio:", l0_l2_ratio)
#%%
l1_new = pd.DataFrame()
l2_new = pd.DataFrame()
for idx, row in train_df.iterrows():
    if row[0] == 1:
        for i in range(l0_l1_ratio):
            l1_new = l1_new.append(row)
    if row[0] == 2:
        for i in range(l0_l2_ratio):
            l2_new = l2_new.append(row)

train_df = train_df.append(l1_new)
train_df = train_df.append(l2_new)
#%%
# shuffle
train_df = shuffle(train_df)

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

print("train_df size: ", train_df.shape)
#%%
# fill params dict before call train_cnn
params = {"input_w": 15, "input_h": 15, "num_classes": 3}
#params = {"input_w": 15, "input_h": 15, "num_classes": 3, "batch_size": 1024, "epochs": 100}

predictions, test_labels, test_prices = train_cnn(train_df, test_df, params)

result_df = pd.DataFrame({"prediction": np.argmax(predictions, axis=1),
                          "test_label":np.argmax(test_labels, axis=1),
                         "test_price":test_prices})
result_df.to_csv("cnn_resultthesis9.csv", sep=';', index=None, header=None)


# %%
