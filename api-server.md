## 🧰 1. Instalación de dependencias principales

```bash
sudo dnf update -y
sudo dnf install -y epel-release dnf-utils curl git firewalld nginx
sudo systemctl enable --now firewalld
```
---

---

## 🔥 2. Firewall

```bash
sudo firewall-cmd --add-service=http --permanent
sudo firewall-cmd --add-service=https --permanent
sudo firewall-cmd --reload
```

---

## 🍃 3. Instalar MongoDB 7

```bash
# Añadir repositorio MongoDB
sudo tee /etc/yum.repos.d/mongodb-org-8.0.repo <<EOF
[mongodb-org-8.0]
name=MongoDB Repository
baseurl=https://repo.mongodb.org/yum/redhat/9/mongodb-org/8.0/x86_64/
gpgcheck=1
enabled=1
gpgkey=https://pgp.mongodb.com/server-8.0.asc
EOF

# Instalar MongoDB
sudo dnf install -y mongodb-org
sudo systemctl enable --now mongod
```

---

## 🟩 4. Instalar Node.js y NestJS (para api1.dominio)

```bash
# Node.js 20 LTS
curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -
sudo dnf install -y nodejs

# Verificar
node -v
npm -v

# Instalar pm2
sudo npm install -g pm2
```

---

## 🧱 5. Instalar Python y Gunicorn (para api2.dominio)

```bash
sudo dnf install -y python3 python3-pip python3-virtualenv
```

---

## 📁 6. Estructura de proyectos

```bash
# Flask (api2)
sudo mkdir -p /var/www/flask-api
cd /var/www/flask-api
python3 -m venv venv
source venv/bin/activate
pip install \
  torch \
  torchvision \
  torchaudio \
  opencv-python \
  Pillow \
  matplotlib \
  easyocr \
  transformers \
  librosa \
  twilio \
  Flask \
  gunicorn

```

```python
#wsgi.py
from app import app

if __name__ == "__main__":
    app.run()
```

```bash
# NestJS (api1)
sudo mkdir -p /var/www/nest-api
cd /var/www/nest-api
# Clona tu proyecto Nest o crea uno
# git clone <repo> .  ó  nest new .
npm install
npm run build
pm2 start dist/main.js --name nest-api
pm2 startup systemd
pm2 save
```

---

## ⚙️ 7. Configuración systemd para Flask (Gunicorn)

```bash
sudo tee /etc/systemd/system/flask-api.service <<EOF
[Unit]
Description=Flask API with Gunicorn
After=network.target

[Service]
User=root
WorkingDirectory=/var/www/flask-api
ExecStart=/bin/bash -c 'cd /var/www/flask-api && source venv/bin/activate && exec gunicorn --bind 0.0.0.0:5000 wsgi:app'
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable --now flask-api
```

---

## 🌐 8. Configuración de Nginx

```bash
sudo tee /etc/nginx/conf.d/apis.conf <<EOF
server {
    listen 80;
    server_name nest.xdn.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection upgrade;
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}

server {
    listen 80;
    server_name flask.xdn.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
EOF

sudo nginx -t
sudo systemctl enable --now nginx
```

---

## 🔐 9. Instalar SSL con Certbot

```bash
sudo dnf install -y certbot python3-certbot-nginx

# Obtener certificado para ambos subdominios
sudo systemctl stop nginx
sudo certbot certonly --standalone -d nest.xdn.com.mx
sudo certbot certonly --standalone -d flask.xdn.com.mx
sudo systemctl start nginx


# Renovación automática
sudo systemctl enable --now certbot-renew.timer
```
## 🌐 10 Configuración de Nginx SSL

```bash
sudo tee /etc/nginx/conf.d/apis.conf <<EOF
# Redirección HTTP a HTTPS para nest.xdn.com.mx
server {
    listen 80;
    server_name nest.xdn.com.mx;

    return 301 https://$host$request_uri;
}

# Configuración HTTPS para nest.xdn.com.mx
server {
    listen 443 ssl http2;
    server_name nest.xdn.com.mx;

    ssl_certificate /etc/letsencrypt/live/nest.xdn.com.mx/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/nest.xdn.com.mx/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}

# Redirección HTTP a HTTPS para flask.xdn.com.mx
server {
    listen 80;
    server_name flask.xdn.com.mx;

    return 301 https://$host$request_uri;
}

# Configuración HTTPS para flask.xdn.com.mx
server {
    listen 443 ssl http2;
    server_name flask.xdn.com.mx;

    ssl_certificate /etc/letsencrypt/live/flask.xdn.com.mx/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/flask.xdn.com.mx/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}

EOF

sudo nginx -t
sudo systemctl enable --now nginx
```

## ✅ Verificación

* [https://flask.xdn.com](https://flask.xdn.com)
* [https://nest.xdn.com](https://nest.xdn.com)

---
