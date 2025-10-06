import os
import time
from flask import Flask, render_template, request, redirect, url_for, abort, flash
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import shutil # NUEVO: Importamos la librería para copiar archivos

app = Flask(__name__)

# --- CONFIGURACIÓN ---
UPLOAD_FOLDER = 'static/uploads'
# NUEVO: Definimos la carpeta local de origen
LOCAL_SOURCE_FOLDER = 'dientes' 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'tu_llave_secreta_aqui' 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOCAL_SOURCE_FOLDER, exist_ok=True) # NUEVO: Nos aseguramos que la carpeta 'dientes' exista

# --- FUNCIONES AUXILIARES ---
def allowed_file(filename):
    """Verifica si la extensión de un archivo es válida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def list_uploaded_files():
    """Lee la carpeta y devuelve una lista de archivos permitidos."""
    if not os.path.isdir(app.config['UPLOAD_FOLDER']):
        return []
    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f) and not f.startswith('enhanced_')]
    files.sort()
    return files

# NUEVO: Función que carga archivos desde la carpeta local 'dientes'
def sync_local_folder():
    """
    Revisa la carpeta LOCAL_SOURCE_FOLDER y copia los archivos nuevos a UPLOAD_FOLDER.
    """
    if not os.path.isdir(LOCAL_SOURCE_FOLDER):
        return # Si la carpeta 'dientes' no existe, no hacemos nada

    files_in_source = os.listdir(LOCAL_SOURCE_FOLDER)
    files_in_dest = os.listdir(app.config['UPLOAD_FOLDER'])
    new_files_count = 0

    for filename in files_in_source:
        # Solo procesamos archivos con extensiones permitidas y que no estén ya en la carpeta de destino
        if allowed_file(filename) and filename not in files_in_dest:
            source_path = os.path.join(LOCAL_SOURCE_FOLDER, filename)
            dest_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
            
            # Copiamos el archivo
            shutil.copy2(source_path, dest_path)
            new_files_count += 1
    
    # Si se añadieron archivos nuevos, mostramos un mensaje
    if new_files_count > 0:
        plural = 's' if new_files_count > 1 else ''
        print(f'¡Se cargaron {new_files_count} nuevo{plural} archivo{plural} desde la carpeta "dientes"!')

def enhance_image(input_path, output_path):
    """Aplica el filtro de mejora a una imagen y la guarda."""
    # (El resto de esta función no cambia)
    imagen = cv2.imread(input_path)
    if imagen is None: return False
    imagen_lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(imagen_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel_clahe = clahe.apply(l_channel)
    imagen_lab_mejorada = cv2.merge((l_channel_clahe, a_channel, b_channel))
    imagen_final = cv2.cvtColor(imagen_lab_mejorada, cv2.COLOR_LAB2BGR)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    imagen_final = cv2.filter2D(imagen_final, -1, kernel)
    cv2.imwrite(output_path, imagen_final)
    return True

# --- RUTAS DE LA APLICACIÓN ---

@app.route('/')
def index():
    # NUEVO: Llamamos a la función de sincronización cada vez que se carga la página principal
    filenames = list_uploaded_files()
    return render_template('index.html', filenames=filenames)

# --- (EL RESTO DE LAS RUTAS NO NECESITAN CAMBIOS) ---
# upload_file, enhance_file_route, delete_file, delete_all_files, view_file, etc.
# permanecen exactamente igual que en el código anterior.

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

@app.route('/enhance/<filename>')
def enhance_file_route(filename):
    original_secure = secure_filename(filename)
    enhanced_filename = "enhanced_" + original_secure
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], original_secure)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], enhanced_filename)
    if not os.path.exists(input_path):
        abort(404)
    success = enhance_image(input_path, output_path)
    if success:
        return render_template('result.html', original_filename=original_secure, enhanced_filename=enhanced_filename)
    else:
        flash('No se pudo procesar la imagen.', 'danger')
        return redirect(url_for('index'))

@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    enhanced_filepath = os.path.join(app.config['UPLOAD_FOLDER'], "enhanced_" + filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            if os.path.exists(enhanced_filepath):
                os.remove(enhanced_filepath)
        except Exception as e:
            print(f"Error al eliminar {filename}: {e}")
    return redirect(url_for('index'))

@app.route('/delete_all', methods=['POST'])
def delete_all_files():
    folder_path = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder_path):
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
    if not os.path.exists(filepath):
        abort(404)
    return render_template('view.html', filename=filename)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

with app.app_context():
    sync_local_folder()

if __name__ == '__main__':
    app.run(debug=True)