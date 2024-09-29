# Mikhail
# я сделал другой код, в конце понял, что он отвечает на другой вопрос, немного отличающийся от поставленного, а время уже вышло. Мой код отвечает на вопрос, получит ли запись лайк или нет, в зависимости от просмотров или репостов) Так что рекомендации авторам тут так же не актуальны
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.DataFrame({
    "views": [10869, 9083, 5352, 4260, 5676, 8252, 3639, 7192, 2950, 15285, 3918, 42875, 78932, 7509, 3515, 13317, 27560, 5608, 21087, 12796, 2132, 5986, 8782, 4785, 6651, 6811, 1916, 41187, 1452, 4710],
    "likes": [185, 227, 25, 539, 112, 219, 2, 23, 15, 563, 1, 2, 3, 1, 2, 3, 0, 2, 3, 1, 1, 2, 3, 0, 2, 3, 1, 2, 3, 1],
    "reposts": [4, 1, 5, 5, 2, 3, 4, 4, 4, 5, 1, 1, 4, 1, 1, 4, 1, 1, 0, 0, 1, 1, 4, 1, 1, 4, 1, 1, 4, 4,],
})

data["like_conversion"] = data["likes"] / data["views"]
data["repost_conversion"] = data["reposts"] / data["views"]
data.fillna(0, inplace=True)

data["target"] = (data["likes"] > 0).astype(int)

print("Распределение классов в исходном наборе данных")
print(data["target"].value_counts())

x = data[["views"]]
y = data["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

print("Распределение классов в y_train:")
print(y_train.value_counts())

model = LogisticRegression()
model.fit(x_train, y_train)

try:
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(f"Точность модели: {accuracy:.6f}")
    print("Отчет о классификации")
    print(classification_report(y_test, y_pred))
except Exception as e:
    print(f"Ошибка при прогнозировании: {e}")

y_pred = model.predict(x_test)

accuraсy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuraсy: {accuraсy}")
print("Classification Report")
print(report)
