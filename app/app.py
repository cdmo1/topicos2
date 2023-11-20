from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import json
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Carregar conjunto de dados (por exemplo, conjunto de dados Iris)
iris = load_iris()
X, y = iris.data, iris.target

# Divida o conjunto de dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# o código carrega o conjunto de dados Iris, divide-o em conjuntos de treino e teste e cria
# rotas para renderizar a página inicial e para treinar o modelo quando o formulário
# na página HTML é submetido. As métricas de desempenho (precisão, recall, F1-score e acurácia)
# são calculadas para o modelo treinado e uma matriz de confusão é gerada e convertida em uma imagem base64.
# Aqui utilizamos o Flask para criar um sistema simples de machine learning que permite treinar diferentes
# classificadores (KNN, SVM, MLP, Decision Tree e Random Forest) com o conjunto de dados Iris.
# Ele também calcula métricas de desempenho e gera uma matriz de confusão



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    data = json.loads(request.data)
    classifier = data['classifier']
    parameters = data['parameters']

    if classifier == 'KNN':
        model = KNeighborsClassifier(n_neighbors=parameters['n_neighbors'])
    elif classifier == 'SVM':
        model = SVC(C=parameters['C'], kernel=parameters['kernel'])
    elif classifier == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=tuple(map(int, parameters['hidden_layer_sizes'].split(','))))
    elif classifier == 'DT':
        model = DecisionTreeClassifier(max_depth=parameters['max_depth'])
    elif classifier == 'RF':
        model = RandomForestClassifier(n_estimators=parameters['n_estimators'])


    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    confusion = confusion_matrix(y_test, y_pred)

    # Gerar gráfico de matriz de confusão
    plt.figure(figsize=(6, 4))
    plt.imshow(confusion, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1, 2], iris.target_names)
    plt.yticks([0, 1, 2], iris.target_names)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, confusion[i, j], ha='center', va='center', color='white' if confusion[i, j] > 5 else 'black')
    plt.tight_layout()

    # Converter gráfico em imagem codificada em base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    buffer_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return jsonify({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': buffer_str
    })

if __name__ == '__main__':
    app.run(debug=True)
