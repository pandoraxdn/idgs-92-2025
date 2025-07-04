from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

custom_responses = {
    "¿quién eres?": "Soy un asistente conversacional basado en inteligencia artificial.",
    "¿cómo te llamas?": "Puedes llamarme PandoraIA.",
    "¿cual es tu nombre?": "Puedes llamarme PandoraIA.",
    "¿qué puedes hacer?": "Puedo conversar contigo sobre muchos temas. ¡Pruébame!",
    "hola": "¡Hola! ¿En qué te puedo ayudar hoy?",
    "adiós": "¡Hasta pronto! 😊",
    "gracias": "¡De nada! Estoy para ayudarte."
}

def check_custom_response(user_input):
    user_input_lower = user_input.lower()
    for key in custom_responses:
        if key in user_input_lower:
            return custom_responses[key]
    return None

print("Cargando modelo...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
print("Modelo cargado.")

chat_history_ids = None

while True:
    user_input = input("Tú: ")

    if user_input.lower() in ['salir', 'exit', 'quit']:
        print("PandoraIA: ¡Hasta luego!")
        break

    custom_reply = check_custom_response(user_input)
    if custom_reply:
        print(f"PandoraIA: {custom_reply}")
        continue

    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {bot_response}")

