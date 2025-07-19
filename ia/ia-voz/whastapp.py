from twilio.rest import Client

def enviar_whatsapp(contacto, mensaje):
    account_sid = ""
    auth_token = ""
    twilio_number = ''

    contactos = {
        "mary": "",
        "maria": "",
    }

    numero = contactos.get(contacto.lower())

    try:
        client = Client(account_sid, auth_token)

        message = client.messages.create(
            from_=twilio_number,
            body=mensaje,
            to=numero
        )

        print(message.sid)

        return True

    except Exception as e:
        print(e)
        return False

if __name__ == "__main__":
    contact_name = "mary"
    message_body = "Mensaje de prueba"
    enviar_whatsapp(contact_name,message_body)
