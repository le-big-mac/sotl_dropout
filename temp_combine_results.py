import csv
import pickle
from scipy.stats import kendalltau, spearmanr

for dataset in ["bostonHousing", "concrete", "energy", "wine-quality-red", "yacht"]:
    for t in ["", "_wd"]:
        split_values = []
        i = -1

        with open("./results_old/{}{}.csv".format(dataset, t), "r") as f:
            reader = csv.reader(f)
            next(f)
            for row in reader:
                if len(row) <= 2:
                    i += 1
                    split_values.append({})
                elif row[0] in ["naive_sotl", "test_loss"]:
                    split_values[i]["early_stopping_{}".format(row[0])] = [float(x) for x in row[1:-1]]
                else:
                    split_values[i][row[0]] = [float(x) for x in row[1:-1]]

        naive_values = []
        i = -1
        with open("./results_old/{}{}_naive.csv".format(dataset, t), "r") as f:
            reader = csv.reader(f)
            next(f)
            for row in reader:
                if len(row) <= 2:
                    i += 1
                else:
                    split_values[i][row[0]] = [float(x) for x in row[1:-1]]

        with open("./results/{}{}.csv".format(dataset, t), "w+") as f:
            writer = csv.writer(f)
            writer.writerow(["dropout_prob", 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            for i in range(len(split_values)):
                writer.writerow([i])
                for k in ["sotl", "mc_sotl", "naive_sotl", "early_stopping_naive_sotl", "test_loss",
                          "early_stopping_test_loss"]:
                    writer.writerow([k] + split_values[i][k])