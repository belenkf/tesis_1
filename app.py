
from flask import Flask, request, render_template
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Cargar y preparar el modelo
df = pd.read_csv('tu_archivo.csv')  # Asegúrate de tener tu archivo CSV en el directorio
df['PCR'] = pd.to_numeric(df['PCR'], errors='coerce')
df.fillna(df['PCR'].mean(), inplace=True)

le = LabelEncoder()
df['sexo'] = le.fit_transform(df['sexo'])

X = df.drop(columns=['Marca temporal', 'dni', 'edad', 'distrés respiratorio'])
y = df['distrés respiratorio']

dtree_model = DecisionTreeClassifier()
dtree_model.fit(X, y)

app = Flask(name)

# Página principal
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para predecir
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario
    features = [
        float(request.form['feature1']),
        float(request.form['feature2']),
        float(request.form['feature3']),
        int(request.form['sexo'])
    ]

    # Realizar la predicción
    prediction = dtree_model.predict([features])
    output = 'Sí' if prediction[0] else 'No'

    return render_template('index.html', prediction_text=f'Distrés Respiratorio: {output}')

if name == "main":
    app.run(debug=True)

