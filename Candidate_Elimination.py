import pandas as pd

def initialize_hypotheses(data):
    S = data.iloc[0, :-1].copy().values
    S[:] = 'ϕ'
    G = data.iloc[0, :-1].copy().values
    G[:] = '?'
    return S, [G]

def update_s(S, instance):
    for i in range(len(S)):
        if S[i] == 'ϕ':
            S[i] = instance[i]
        elif S[i] != instance[i]:
            S[i] = '?'
    return S

def update_g(G, instance, attributes):
    new_G = []
    for hypothesis in G:
        for i in range(len(hypothesis)):
            if hypothesis[i] == '?':
                for value in attributes[i]:
                    if value != instance[i]:
                        new_hypothesis = hypothesis.copy()
                        new_hypothesis[i] = value
                        # Convert new_hypothesis to a tuple for hashability
                        if tuple(new_hypothesis) not in [tuple(g) for g in new_G]:
                            new_G.append(new_hypothesis)
            elif hypothesis[i] != instance[i]:
                # Convert hypothesis to a tuple for hashability
                if tuple(hypothesis) not in [tuple(g) for g in new_G]:
                    new_G.append(hypothesis)
    return new_G

def candidate_elimination(data):
    attributes = [list(data[col].unique()) for col in data.columns[:-1]]
    S, G = initialize_hypotheses(data)
    for index, row in data.iterrows():
        instance = row[:-1].values
        label = row[-1]
        if label == 'yes':
            G = [g for g in G if all(g[i] == '?' or g[i] == instance[i] for i in range(len(g)))]
            S = update_s(S, instance)
        else:
            S = [s for s in S if any(s[i] != instance[i] and s[i] != '?' for i in range(len(s)))]
            G = update_g(G, instance, attributes)
    return S, G

file_path = '/content/training.csv'
data = pd.read_csv(file_path)
S, G = candidate_elimination(data)
print("The most specific hypothesis (S) is:", S)
print("The set of most general hypotheses (G) is:", G)
