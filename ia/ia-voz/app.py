from flask import Flask, request, jsonify
import os
from audio import procesar_audio
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Carpeta temporal para guardar los audios
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/audio', methods=['POST'])
def recibir_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No se envió el archivo de audio'}), 400

    archivo = request.files['audio']
    
    if archivo.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    if archivo and archivo.filename.endswith('.mp3') or archivo and archivo.filename.endswith('.m4a'):
        filename = secure_filename(archivo.filename)
        ruta_audio = os.path.join(UPLOAD_FOLDER, filename)
        archivo.save(ruta_audio)

        try:
            resultado = procesar_audio(ruta_audio)
            return jsonify({'resultado': resultado})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            os.remove(ruta_audio)  # Borra el archivo temporal
    else:
        return jsonify({'error': 'Formato no permitido. Solo se acepta .mp3'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
