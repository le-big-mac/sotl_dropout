import csv
import pickle
from scipy.stats import kendalltau, spearmanr

correlation_results = {}

for dataset in ["bostonHousing", "concrete", "energy", "wine-quality-red", "yacht"]:
    for t in ["", "_wd"]:
        split_values = []
        i = -1

        with open("./results{}/{}.csv".format(t, dataset), "r") as f:
            reader = csv.reader(f)
            next(f)
            for row in reader:
                if len(row) <= 2:
                    i += 1
                    split_values.append({})
                else:
                    split_values[i][row[0]] = [float(x) for x in row[1:-1]]

        results = {"spearman": [], "kendall": []}
        for split in split_values:
            spearman = {}
            kendall = {}
            for k in ["sotl", "mc_sotl", "naive_sotl"]:
                spearman[k] = tuple(spearmanr(split[k], [-x for x in split["test_loss"]]))
                kendall[k] = tuple(kendalltau(split[k], [-x for x in split["test_loss"]]))

            results["spearman"].append(spearman)
            results["kendall"].append(kendall)

        correlation_results["{}{}".format(dataset, t)] = results

with open("correlation_results.pickle", "wb+") as f:
    pickle.dump(correlation_results, f)
