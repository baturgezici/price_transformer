#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log

# %%
transformer = pd.read_csv("transformer_result2.csv")
deit = pd.read_csv("deit_result.csv")
convmixer = pd.read_csv("cnn_result16-8.csv")
cnn = pd.read_csv("convnixer_result5205.csv")

# %% 
swin = pd.read_csv("swin_pred.csv", delimiter=";")

# %%
plt.plot(range(0,100), list(map(lambda x: -1 if(x==2) else x, transformer["prediction"][:100])), label="transformer")
plt.plot(range(0,100), list(map(lambda x: -1 if(x==2) else x, convmixer["prediction"][:100])), label="convmixer")
plt.plot(range(0,100), list(map(lambda x: -1 if(x==2) else x, cnn["prediction"][:100])), label="cnn")
plt.plot(range(0,100), list(map(lambda x: -1 if(x==2) else x, deit["prediction"][:100])), label="deit")
plt.plot(range(0,100), list(map(lambda x: -1 if(x==2) else x, deit["test_label"][:100])), label="Labels")
plt.yticks([-1, 0, 1], ["Sell", "Hold", "Buy"])
plt.xlabel("index")
plt.ylabel("prediction")
plt.legend()
plt.show()

# %%
plt.plot(range(0,100), list(map(lambda x: -1 if(x==2) else x, deit["prediction"][:100])), label="deit")
plt.plot(range(0,100), list(map(lambda x: -1 if(x==2) else x, deit["test_label"][:100])), label="Labels")
plt.yticks([-1, 0, 1], ["Sell", "Hold", "Buy"])
plt.xlabel("index")
plt.ylabel("prediction")
plt.legend()
plt.show()

# %%
plt.plot(range(0,100), list(map(lambda x: -1 if(x==2) else x, cnn["prediction"][:100])), label="cnn")
plt.plot(range(0,100), list(map(lambda x: -1 if(x==2) else x, deit["test_label"][:100])), label="Labels")
plt.yticks([-1, 0, 1], ["Sell", "Hold", "Buy"])
plt.xlabel("index")
plt.ylabel("prediction")
plt.legend()
plt.show()

# %%
plt.plot(range(0,100), list(map(lambda x: -1 if(x==2) else x, convmixer["prediction"][:100])), label="convmixer")
plt.plot(range(0,100), list(map(lambda x: -1 if(x==2) else x, deit["test_label"][:100])), label="Labels")
plt.yticks([-1, 0, 1], ["Sell", "Hold", "Buy"])
plt.xlabel("index")
plt.ylabel("prediction")
plt.legend()
plt.show()

# %%
plt.plot(range(0,100), list(map(lambda x: -1 if(x==2) else x, transformer["prediction"][:100])), label="transformer")
plt.plot(range(0,100), list(map(lambda x: -1 if(x==2) else x, deit["test_label"][:100])), label="Labels")
plt.yticks([-1, 0, 1], ["Sell", "Hold", "Buy"])
plt.xlabel("index")
plt.ylabel("prediction")
plt.legend()
plt.show()

# %%
plt.plot(range(0,100), list(map(lambda x: -1 if(x==2) else x, swin["prediction"][:100])), label="transformer")
plt.plot(range(0,100), list(map(lambda x: -1 if(x==2) else x, swin["test_label"][:100])), label="Labels")
plt.yticks([-1, 0, 1], ["Sell", "Hold", "Buy"])
plt.xlabel("index")
plt.ylabel("prediction")
plt.legend()
plt.show()


# %%
def correct(x,num):
    l1 = []
    l2 = []
    for i in range(num):
        if x["prediction"][i] == x["test_label"][i]:
            l1.append(i)
            l2.append(x["prediction"][i])
    return l1, l2

# %%
plt.plot(range(0,50), transformer["prediction"][:50], label="transformer")
plt.plot(range(0,50), transformer["test_label"][:50], label="real")
l1, l2 = correct(transformer,50)
plt.scatter(l1, l2, label="Correct Points")
plt.yticks([0, 1, 2])
plt.xlabel("index")
plt.ylabel("value")
plt.legend()
plt.show()

# %%
plt.plot(range(0,50), convmixer["prediction"][:50], label="convmixer")
plt.plot(range(0,50), convmixer["test_label"][:50], label="real")
l1, l2 = correct(convmixer,50)
plt.scatter(l1, l2, label="Correct Points")
plt.yticks([0, 1, 2])
plt.xlabel("index")
plt.ylabel("value")
plt.legend()
plt.show()

# %%
plt.plot(range(0,50), cnn["prediction"][:50], label="cnn")
plt.plot(range(0,50), cnn["test_label"][:50], label="real")
l1, l2 = correct(cnn,50)
plt.scatter(l1, l2, label="Correct Points")
plt.yticks([0, 1, 2])
plt.xlabel("index")
plt.ylabel("value")
plt.legend()
plt.show()

# %%
