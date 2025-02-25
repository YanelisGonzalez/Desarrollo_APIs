# Usamos una imagen base de Python
FROM python:3.9-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos el archivo de requerimientos y lo instalamos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto de los archivos de la aplicación
COPY . .

# Exponemos el puerto en el que correrá la aplicación
EXPOSE 8000

# Comando para iniciar la aplicación con Uvicorn
CMD ["uvicorn", "app_model:app", "--host", "0.0.0.0", "--port", "8000"]
