import smtplib
from email.message import EmailMessage

REMITENTE = ""
PASSWORD = ""

USUARIOS = { 
    "rodrigo": ""
}

def enviar_correo(nombre_destinatario,asunto,cuerpo_mensaje):

    destinatario = USUARIOS.get(nombre_destinatario)

    msg = EmailMessage()
    msg["From"] = REMITENTE
    msg["To"] = destinatario
    msg["Subject"] = asunto
    msg.set_content(cuerpo_mensaje)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com",port=465) as smtp_server:
            smtp_server.login(REMITENTE,PASSWORD)
            smtp_server.send_message(msg)
            return (destinatario, cuerpo_mensaje)
    except smtplib.SMTPAuthenticationError as error:
        print(error)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    var = enviar_correo(
        nombre_destinatario="rodrigo",
        asunto="Nuevas actividades del día de mañana",
        cuerpo_mensaje="No se olvide la basura, y que no llegues tarde"
    )
    print(var)





