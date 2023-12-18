# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_deit = pd.read_csv("deit_chart_info.txt", header=None)
df_cnn = pd.read_csv("cnn_chart_info.txt", header=None)
df_transformer = pd.read_csv("transformer_chart_info.txt", header=None)
df_convmixer = pd.read_csv("convmixer_chart_info.txt", header=None)
df_swin = pd.read_csv("swin_chart_info.txt", header=None)

# %%
data_deit = np.array(df_deit)
data_cnn = np.array(df_cnn)
data_transformer = np.array(df_transformer)
data_convmixer = np.array(df_convmixer)
data_swin = np.array(df_swin)
# %%

plt.plot(range(0,28), data_swin[:,0])
plt.axhline(y=10000, color='r', linestyle='-')
plt.title("Swin Transformer")
plt.xlabel("Transaction number")
plt.ylabel("Current balance")
plt.show()


# %%

plt.plot(range(0,71), data_deit[:,0])
plt.axhline(y=10000, color='r', linestyle='-')
plt.title("DeiT")
plt.xlabel("Transaction number")
plt.ylabel("Current balance")
plt.show()

plt.plot(range(0,80), data_cnn[:,0])
plt.axhline(y=10000, color='r', linestyle='-')
plt.title("CNN")
plt.xlabel("Transaction number")
plt.ylabel("Current balance")
plt.show()

plt.plot(range(0,43), data_transformer[:,0])
plt.axhline(y=10000, color='r', linestyle='-')
plt.title("Transformer")
plt.xlabel("Transaction number")
plt.ylabel("Current balance")
plt.show()

plt.plot(range(0,38), data_convmixer[:,0])
plt.axhline(y=10000, color='r', linestyle='-')
plt.title("ConvMixer")
plt.xlabel("Transaction number")
plt.ylabel("Current balance")
plt.show()

# %%
