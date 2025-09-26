import os
import time
from flask import Flask, render_template, request, redirect, url_for, abort, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- CONFIGURACIÓN ---
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'mp4', 'avi', 'mov', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'tu_llave_secreta_aqui' 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- FUNCIONES AUXILIARES ---
def allowed_file(filename):
    """Verifica si la extensión de un archivo es válida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def list_uploaded_files():
    """Lee la carpeta y devuelve una lista de archivos permitidos."""
    if not os.path.isdir(app.config['UPLOAD_FOLDER']):
        return []
    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
    files.sort()
    return files


# --- RUTAS DE LA APLICACIÓN ---

@app.route('/')
def index():
    filenames = list_uploaded_files()
    return render_template('index.html', filenames=filenames)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'archivos' not in request.files:
        flash('No se seleccionó ningún archivo para subir.', 'warning')
        return redirect(url_for('index'))

    files = request.files.getlist('archivos')
    for file in files:
        if file and file.filename != '':
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
            else:
                flash(f"El archivo '{secure_filename(file.filename)}' tiene una extensión no permitida.", 'danger')

    return redirect(url_for('index'))


@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except PermissionError:
            time.sleep(0.1)
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"No se pudo borrar el archivo en el segundo intento: {e}")
    
    return redirect(url_for('index'))


@app.route('/delete_all', methods=['POST'])
def delete_all_files():
    folder_path = app.config['UPLOAD_FOLDER']
    for filename in list_uploaded_files():
        filepath = os.path.join(folder_path, filename)
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"No se pudo borrar el archivo {filepath}: {e}")
    flash('Todos los archivos han sido eliminados.', 'success')
    return redirect(url_for('index'))


@app.route('/view/<filename>')
def view_file(filename):
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath) or not allowed_file(filename):
        abort(404)
    return render_template('view.html', filename=filename)


# --- CÓDIGO ANTI-CACHÉ ---
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# --- ARRANQUE DE LA APLICACIÓN ---
if __name__ == '__main__':
    app.run(debug=True)

