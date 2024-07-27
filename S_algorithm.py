import pandas as pd
def find_s_algorithm(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    hypothesis = None
    for i, label in enumerate(y):
        if label == 'yes':
            hypothesis = X[i].copy()
            break
    for i, label in enumerate(y):
        if label == 'yes':
            for j in range(len(hypothesis)):
                if hypothesis[j] != X[i][j]:
                    hypothesis[j] = '?'
    return hypothesis
file_path = '/content/training.csv'
data = pd.read_csv(file_path)
specific_hypothesis = find_s_algorithm(data)
print("The most specific hypothesis found by FIND-S is:", specific_hypothesis)
