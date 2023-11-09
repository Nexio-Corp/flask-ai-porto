# cspell:disable
from flask import Flask, request, jsonify
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

# Carregar o modelo Keras
model = load_model('modelo.h5')
model.make_predict_function()  # Necessário para evitar erros no TensorFlow


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224)) # Mesmo tamanho usado no treinamento do modelo
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255.0 
    return img


@app.route('/predict', methods=['POST'])
def predict():
    # Verificar se o arquivo de imagem foi enviado com o nome 'image' no request
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'})

    file = request.files['image']

    # Verificar se o arquivo foi enviado de fato
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'})

    if file:
        # Salvar a imagem temporariamente
        file_path = 'temp.jpg'
        file.save(file_path)

        # Pré-processar a imagem
        img = preprocess_image(file_path)

        # Fazer a previsão
        prediction = model.predict(img)
        result = 'quebrada' if prediction[0][0] > 0.5 else 'usável'

        # Deletar a imagem temporária
        os.remove(file_path)

        return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)
