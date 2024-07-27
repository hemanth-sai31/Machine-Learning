import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
data = {
    'battery_power': [842, 1021, 563, 615, 1821],
    'clock_speed': [2.2, 0.5, 0.5, 2.5, 1.2],
    'dual_sim': [0, 1, 1, 0, 1],
    'fc': [1, 0, 2, 0, 13],
    'four_g': [0, 1, 1, 1, 1],
    'int_memory': [7, 53, 41, 10, 44],
    'm_dep': [0.6, 0.7, 0.9, 0.8, 0.6],
    'mobile_wt': [188, 136, 145, 131, 141],
    'n_cores': [2, 3, 5, 6, 2],
    'pc': [2, 6, 6, 9, 14],
    'px_height': [20, 905, 1263, 1216, 1208],
    'px_width': [756, 1988, 1716, 1786, 1218],
    'ram': [2549, 2631, 2603, 2769, 1411],
    'sc_h': [9, 17, 11, 16, 8],
    'sc_w': [7, 3, 2, 8, 1],
    'talk_time': [19, 7, 9, 11, 15],
    'three_g': [0, 1, 1, 1, 1],
    'touch_screen': [0, 1, 0, 1, 0],
    'wifi': [1, 0, 1, 0, 1],
    'price_range': [1, 2, 2, 3, 1]
}
df = pd.DataFrame(data)
print(df.head())
X = df.drop('price_range', axis=1)
y = df['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
