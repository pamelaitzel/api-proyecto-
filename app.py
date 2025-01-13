from flask import Flask, render_template, request, jsonify  # Importa las funciones y clases principales de Flask.
import subprocess  # Permite ejecutar comandos del sistema operativo desde Python.
import os  # Proporciona funciones para interactuar con el sistema operativo (manejo de archivos y rutas).

app = Flask(__name__, template_folder="trabajos")  
# Crea la instancia principal de la aplicación Flask.
# Se especifica que las plantillas HTML están en una carpeta llamada "trabajos".

# Ruta para la página principal
@app.route("/")  # Define la ruta raíz ("/") que responde a solicitudes GET por defecto.
def index():  # Función que maneja las solicitudes a la ruta principal.
    return render_template("index.html")  
    # Renderiza el archivo HTML "index.html" desde la carpeta especificada como "template_folder".

# Ruta para ejecutar un notebook
@app.route("/execute", methods=["POST"])  
# Define una ruta "/execute" que acepta únicamente solicitudes POST.
def execute_notebook():  # Función que maneja las solicitudes POST a la ruta "/execute".
    notebook = request.form.get("notebook")  
    # Obtiene el nombre del notebook desde los datos enviados en el formulario (campo "notebook").
    notebook_path = os.path.join("scripts", "notebooks", notebook)  
    # Construye la ruta completa del notebook combinando las carpetas base con el nombre del archivo.

    if os.path.exists(notebook_path):  
        # Verifica si el archivo especificado existe en la ruta construida.
        try:
            # Intenta ejecutar el archivo notebook usando Python.
            output = subprocess.check_output(
                ["python", notebook_path],  
                # Ejecuta el archivo Python (notebook) en la ruta especificada.
                stderr=subprocess.STDOUT,  # Redirige los errores estándar a la salida estándar.
                text=True  # Devuelve la salida como texto en lugar de bytes.
            )
            return f"<h1>Resultados del Notebook: {notebook}</h1><pre>{output}</pre>"  
            # Devuelve una página HTML con el nombre del notebook y su salida en un bloque `<pre>`.
        except subprocess.CalledProcessError as e:  
            # Captura cualquier error que ocurra al ejecutar el comando.
            return f"<h1>Error ejecutando el Notebook: {notebook}</h1><pre>{e.output}</pre>"  
            # Devuelve una página HTML indicando que hubo un error y muestra los detalles.
    else:
        return f"<h1>Notebook no encontrado: {notebook}</h1>"  
        # Devuelve un mensaje HTML indicando que el archivo no fue encontrado.

if __name__ == "__main__":  
    # Este bloque asegura que la aplicación solo se ejecuta si el archivo es ejecutado directamente (no importado).
    app.run(debug=True)  
    # Inicia el servidor web en modo de depuración:
    # - Reinicia automáticamente al detectar cambios en el código.
    # - Muestra errores detallados en el navegador si ocurren problemas.
