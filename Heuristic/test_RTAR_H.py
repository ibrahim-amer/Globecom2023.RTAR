from RTAR_H import RTAR_H, run_RTAR_H, convert_matchings_to_vector
import numpy as np


def main():
    tasks_budgets = [200, 15, 50, 100, 200, 300, 1000, 50]
    #############################0, 1,  2,   3,  4,  5,  6,  7,  8,  9
    workers_recruitment_costs = [5, 10, 15, 20, 50, 100, 60, 30, 80, 70, 100, 60, 30, 80, 70]
    workers_recruitment_costs = np.array(workers_recruitment_costs)
    ###############0,   1,    2,   3,   4,   5,   6,   7,   8,  9
    workers_reputations = workers_recruitment_costs / 100

    tasks_prefs = {
        0: [5, 8, 9, 6, 4, 7, 3, 2, 1, 0],
        1: [5, 8, 9, 6, 4, 7, 3, 2, 1, 0],
        2: [5, 8, 9, 6, 4, 7, 3, 2, 1, 0],
        3: [5, 8, 9, 6, 4, 7, 3, 2, 1, 0],
        4: [5, 8, 9, 6, 4, 7, 3, 2, 1, 0],
        5: [5, 8, 9, 6, 4, 7, 3, 2, 1, 0],
        6: [5, 8, 9, 6, 4, 7, 3, 2, 1, 0],
        7: [5, 8, 9, 6, 4, 7, 3, 2, 1, 0],
        8: [5, 8, 9, 6, 4, 7, 3, 2, 1, 0],
        9: [5, 8, 9, 6, 4, 7, 3, 2, 1, 0],
        10: [5, 8, 9, 6, 4, 7, 3, 2, 1, 0]
    }


    matchings = run_RTAR_H(tasks_budgets, workers_recruitment_costs, workers_reputations)
    print(matchings)
    X = convert_matchings_to_vector(matchings, len(workers_recruitment_costs), len(tasks_budgets))
    for i in range(len(workers_recruitment_costs)):
        for j in range(len(tasks_budgets)):
            print("x_{0}{1} = {2}".format(i, j, X[i*len(tasks_budgets) + j]), end=", ")
    print(X)
    #matching = RTAR_H(tasks_prefs=tasks_prefs, tasks_budgets=tasks_budgets, workers_recruitment_costs=workers_recruitment_costs)


if __name__ == "__main__":
    main()