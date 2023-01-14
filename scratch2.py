#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logs = pd.read_csv("log.txt")


#%%
fig = plt.figure(figsize=(6,6))
ax = sns.scatterplot(x=logs["combinations"], y=(logs["execution_time"]),
                     hue=logs["processes"], legend="full", palette="flare")

# ax.set_xlim((0, 130000))
ax.set_xlabel("Number of Combinations for Cross-Correlation")

# ax.set_ylim((0, 6))
ax.set_ylabel("Execution Time (seconds)")

ax.grid(color='k', linestyle='--', linewidth=0.5)

#%%
subset = logs[logs["processes"]==12]
fig = plt.figure(figsize=(6,6))
ax = sns.scatterplot(x=subset["combinations"], y=(subset["execution_time"]))


ax.set_xlabel("Number of Combinations for Cross-Correlation")
# ax.set_xlim((0, 550000))

ax.set_ylabel("Execution Time (seconds)")
# ax.set_ylim((0, 14))

ax.grid(color='k', linestyle='--', linewidth=0.5)


#%%
fig = plt.figure(figsize=(6,6))
ax = sns.scatterplot(x=logs["num_stocks"], y=(logs["execution_time"]),
                     hue=logs["processes"], legend="full", palette="flare")

# ax.set_xlim((0, 900))
ax.set_xlabel("Number of Stocks for Cross-Correlation")

# ax.set_ylim((0, 90))
ax.set_ylabel("Execution Time (seconds)")

ax.grid(color='k', linestyle='--', linewidth=0.5)

#%%
subset = logs[logs["processes"]==12]
fig = plt.figure(figsize=(6,6))
ax = sns.scatterplot(x=subset["num_stocks"], y=(subset["execution_time"]))


ax.set_xlabel("Number of Stocks for Cross-Correlation")
# ax.set_xlim((0, 900))

ax.set_ylabel("Execution Time (seconds)")
# ax.set_ylim((0, 15))

ax.grid(color='k', linestyle='--', linewidth=0.5)