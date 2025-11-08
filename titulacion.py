import os
import time
from flask import Flask, render_template, request, redirect, url_for, abort, flash, jsonify
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
DELETED_DURING_SESSION = set()

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
        return 0
    
    files_in_source = os.listdir(LOCAL_SOURCE_FOLDER)
    files_in_dest = os.listdir(app.config['UPLOAD_FOLDER'])
    new_files_count = 0

    for filename in files_in_source:

        is_allowed = allowed_file(filename)
        is_new = filename not in files_in_dest
        is_not_ignored = filename not in DELETED_DURING_SESSION
        
        if is_allowed and is_new and is_not_ignored:
            source_path = os.path.join(LOCAL_SOURCE_FOLDER, filename)
            dest_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
            
            # Copiamos el archivo
            shutil.copy2(source_path, dest_path)
            new_files_count += 1
    
    # Si se añadieron archivos nuevos, mostramos un mensaje
    if new_files_count > 0:
        plural = 's' if new_files_count > 1 else ''
        print(f'¡Se cargaron {new_files_count} nuevo{plural} archivo{plural} desde la carpeta "dientes"!')

    return new_files_count

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
    sync_local_folder()
    filenames = list_uploaded_files()
    return render_template('index.html', filenames=filenames)

# --- (EL RESTO DE LAS RUTAS NO NECESITAN CAMBIOS) ---
# upload_file, enhance_file_route, delete_file, delete_all_files, view_file, etc.
# permanecen exactamente igual que en el código anterior.

@app.route('/sync_and_check')
def sync_and_check_route():

 new_files_count = sync_local_folder()
 return jsonify(new_files_found=new_files_count)

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
                DELETED_DURING_SESSION.discard(filename)
            else:
                flash(f"El archivo '{secure_filename(file.filename)}' tiene una extensión no permitida.", 'danger')
    return redirect(url_for('index'))

@app.route('/enhance/<path:filename>')
def enhance_file_route(filename):
    enhanced_filename = "enhanced_" + filename
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], enhanced_filename)
    filenames = list_uploaded_files()
    prev_filename = None
    next_filename = None
    try:
        current_index= filenames.index(filename)
        if len(filename) > 1:
            prev_index = (current_index - 1) % len(filenames)
            next_index = (current_index + 1) % len(filenames)
            prev_filename = filenames[prev_index]
            next_filename = filenames[next_index]
        else:
            flash('No se pudo procesar la imagen', 'danger')
            return redirect(url_for('index'))
    except ValueError:
        abort(404)
    if not os.path.exists(input_path):
        abort(404)
    success = enhance_image(input_path, output_path)
    if success:
        return render_template('result.html', original_filename=filename, enhanced_filename=enhanced_filename, prev_filename=prev_filename, next_filename=next_filename)
    else:
        flash('No se pudo procesar la imagen.', 'danger')
        return redirect(url_for('index'))

@app.route('/delete/<path:filename>', methods=['POST'])
def delete_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    enhanced_filepath = os.path.join(app.config['UPLOAD_FOLDER'], "enhanced_" + filename)
    
    files_to_remove = [filepath, enhanced_filepath]
    
    for f in files_to_remove:
        if os.path.exists(f):
            # Lógica de reintento: Intentar borrar durante 1 segundo
            attempts = 0
            while attempts < 10:
                try:
                    os.remove(f)
                    # Si se borra, añadimos el original a la lista de ignorados
                    DELETED_DURING_SESSION.add(filename) 
                    DELETED_DURING_SESSION.add("enhanced_" + filename)
                    break # Salir del bucle si se borró
                except PermissionError:
                    attempts += 1
                    time.sleep(0.1) # Esperar 100ms y reintentar
                except Exception as e:
                    print(f"Error al eliminar {f}: {e}")
                    break # Salir del bucle si es un error diferente
            
            if os.path.exists(f):
                 print(f"No se pudo eliminar el archivo {f} después de 10 intentos.")

    return redirect(url_for('index'))

@app.route('/delete_all', methods=['POST'])
def delete_all_files():
    folder_path = app.config['UPLOAD_FOLDER']
    files_to_delete = os.listdir(folder_path)
    deleted_count = 0
    
    for filename in files_to_delete:
        filepath = os.path.join(folder_path, filename)
        
        # Lógica de reintento: Intentar borrar durante 1 segundo
        attempts = 0
        while attempts < 10:
            try:
                os.remove(filepath)
                DELETED_DURING_SESSION.add(filename)
                deleted_count += 1
                break # Salir del bucle si se borró
            except PermissionError:
                attempts += 1
                time.sleep(0.1) # Esperar 100ms y reintentar
            except Exception as e:
                print(f"No se pudo borrar el archivo {filepath}: {e}")
                break # Salir del bucle si es un error diferente

        if os.path.exists(filepath):
            print(f"No se pudo eliminar el archivo {filepath} después de 10 intentos.")

    flash(f'Se eliminaron {deleted_count} de {len(files_to_delete)} archivos.', 'success')
    return redirect(url_for('index'))

@app.route('/view/<path:filename>')
def view_file(filename):
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        abort(404)
    filenames = list_uploaded_files()
    prev_filename = None
    next_filename = None
    try:
        current_index = filenames.index(filename)
        if len(filenames) > 1:
            prev_index = (current_index - 1) % len(filenames)
            next_index = (current_index + 1) % len(filenames)
            prev_filename = filenames[prev_index]
            next_filename = filenames[next_index]
    except ValueError:
        pass
    return render_template('view.html', filename=filename, prev_filename=prev_filename, next_filename=next_filename)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


if __name__ == '__main__':
    app.run(debug=True)