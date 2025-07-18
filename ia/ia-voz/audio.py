import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from whastapp import enviar_whatsapp

model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"

processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

def load_audio(file):
    waveform, sr = librosa.load(file,sr=16000)
    return torch.tensor(waveform).unsqueeze(0)

def corregir_errores(texto):
    correccions = {
        "correr": "correo",
        "correoo": "correo",
        "whatsup": "whatsapp",
        "guasap": "whatsapp",
        "güasap": "whatsapp",
        "wasap": "whatsapp",
        "imeil": "email",
        "güimail": "gmail",
        "corre": "correo",
        "guasab": "whatsapp",
        "wat sap": "whatsapp",
        "enviar un wasap": "enviar whatsapp",
        "enviar un imeil": "enviar correo"
    }

    for error, correccion in correccions.items():
        texto = texto.replace(error, correccion)

    return texto

def audio_to_text(file):
    input_audio = load_audio(file)
    inputs = processor(input_audio.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits,dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    text = transcription[0].lower()
    text_correct = corregir_errores(text)
    return text_correct

def procesar_comando_whastapp(texto):
    try:
        parte_despues_de_a = texto.split(" a ",1)[1]
        nombre_destinatario = parte_despues_de_a.split(" ",1)[0]
        cuerpo_mensaje = texto.split("que le diga ",1)[1]
        enviar_whatsapp(nombre_destinatario,cuerpo_mensaje)
    except Exception as e:
        print(e)

def detectar_comando(text):
    if "enviar whatsapp" in text or "whatsapp" in text:
        print("Acción: EL usuario quiere enviar un Whatsapp")
        procesar_comando_whastapp(text)
    elif "enviar correo" in text or "correo" in text:
        print("El usuario quiere enviar un correo")
    else:
        print("No se detecto ningun acción específica")

if __name__ == "__main__":
    route_audio = "./correo.mp3"
    text = audio_to_text(route_audio)
    print(f"Transcripción: {text}")
    detectar_comando(text)

    route_audio = "./whatsapp.mp3"
    text = audio_to_text(route_audio)
    print(f"Transcripción: {text}")
    detectar_comando(text)
