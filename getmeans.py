import pickle
import pandas as pd


with open('task3.dat', 'rb') as f:
    experiment_results = pickle.load(f)


experiment_designs = []
agg = []
for experiment_num, experiment_design, df in experiment_results:
    mean = df.iloc[2000:None:2000].mean()
    experiment_designs.append(experiment_design)
    agg.append(mean)

agg_df = pd.concat(agg, axis=1).T
print(agg_df)


with open("meansresults.html", "w") as f:
    f.write(agg_df.to_html())


result = pd.concat([pd.DataFrame(experiment_designs), agg_df], axis=1)
print(result)

result.to_csv('final.csv')
