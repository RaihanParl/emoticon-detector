import pickle

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data_dict = pickle.load(open("./data.pickle", "rb"))

data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

for i in data:

    print(len(i))
    print(i)

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True
)
model = RandomForestClassifier()
for i in x_train:
    print(len(i))
print(y_train)
model.fit(list(x_train), y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print("{}% of samples were classified correctly !".format(score * 100))

f = open("model.p", "wb")
pickle.dump({"model": model}, f)
f.close()
