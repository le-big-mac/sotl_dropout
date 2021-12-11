import csv
import pickle
from scipy.stats import kendalltau, spearmanr
import warnings
warnings.filterwarnings("error")

dropout = (0, 0.1, 0.2, 0.3, 0.4, 0.5)
correlation_results = {}

for dataset in ["bostonHousing", "concrete", "energy", "wine-quality-red", "yacht"]:
    split_values = []
    i = -1

    with open("./results/{}.csv".format(dataset), "r") as f:
        reader = csv.reader(f)
        next(f)
        for row in reader:
            if len(row) == 1:
                i += 1
                split_values.append({})
            else:
                split_values[i][row[0]] = [float(x) for x in row[1:]]

    results = {"spearman": [], "kendall": []}
    for split in split_values:
        spearman = {}
        kendall = {}

        for e in ["", "early_stopping_"]:
            spearman["{}dropout".format(e)] = \
                spearmanr(dropout, [-x for x in split["{}test_loss".format(e)]])
            kendall["{}dropout".format(e)] = \
                kendalltau(dropout, [-x for x in split["{}test_loss".format(e)]])

            spearman["{}naive_sotl".format(e)] = \
                tuple(spearmanr(split["{}naive_sotl".format(e)], [-x for x in split["{}test_loss".format(e)]]))
            kendall["{}naive_sotl".format(e)] = \
                tuple(kendalltau(split["{}naive_sotl".format(e)], [-x for x in split["{}test_loss".format(e)]]))

            spearman["{}naive_sotl_dropout".format(e)] = \
                tuple(spearmanr(split["{}naive_sotl".format(e)], dropout))
            kendall["{}naive_sotl_dropout".format(e)] = \
                tuple(kendalltau(split["{}naive_sotl".format(e)], dropout))

        for k in ["sotl", "mc_sotl"]:
            spearman["{}_dropout".format(k)] = tuple(spearmanr(split[k], dropout))
            kendall["{}_dropout".format(k)] = tuple(kendalltau(split[k], dropout))

            for e in ["", "early_stopping_"]:
                spearman["{}{}".format(e, k)] = tuple(spearmanr(split[k], [-x for x in split["{}test_loss".format(e)]]))
                kendall["{}{}".format(e, k)] = tuple(kendalltau(split[k], [-x for x in split["{}test_loss".format(e)]]))

        results["spearman"].append(spearman)
        results["kendall"].append(kendall)

    correlation_results["{}".format(dataset)] = results

with open("correlation_results.pickle", "wb+") as f:
    pickle.dump(correlation_results, f)
