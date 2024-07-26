from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.linear_model import LogisticRegression

# Пример данных
messages = [
    "Win a free prize now!",
    "Subscribe to our newsletter and get an offer",
    "Let's meet for lunch tomorrow",
    "Buy now and save 50%",
    "Are you coming to the meeting?",
    "Click here to claim your free vacation",
    "This is not a spam message",
    "Get your free trial now",
    "I'll see you at the conference",
    "Cheap loans available, apply now!"
]

labels = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1 - спам, 0 - не спам

# Преобразование текстов в числовые признаки
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X, labels)


def classify_message_ml(message):
    """
    Функция для классификации текстового сообщения с использованием обученной модели.

    :param message: Текст сообщения
    :return: 'спам' или 'не спам'
    """
    X_new = vectorizer.transform([message])
    prediction = model.predict(X_new)[0]
    return "спам" if prediction == 1 else "не спам"


# Тестовые сообщения
test_messages = [
    "Win a free trip to Bahamas!",
    "Let's catch up over coffee next week",
    "Limited time offer, buy now!",
    "Are you attending the seminar?"
]

# Классифицируем каждое сообщение
for msg in test_messages:
    classification = classify_message_ml(msg)
    print(f"Сообщение: '{msg}' - Классификация: {classification}")

while True:
    input_string = input()
    classification = classify_message_ml(input_string)
    print(f"Сообщение: '{input_string}' = {classification}")



