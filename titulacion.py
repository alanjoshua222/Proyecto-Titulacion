import os
import time
from flask import Flask, render_template, request, redirect, url_for, abort, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import shutil 

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
def get_turesky_grade(plaque_area, tooth_area):
    if tooth_area == 0:
        return 0, 0.0
    percentage = (plaque_area / tooth_area) * 100
    grade = 0
    if percentage == 0:
        grade = 0
    elif percentage < 10:
        grade = 1
    elif percentage < 20:
        grade = 2
    elif percentage < 33.3:
        grade = 3
    elif percentage < 66.6:
        grade = 4
    else:
        grade = 5
    return grade, percentage
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
    """
    Versión 8.0: Detección por MÁSCARA DE COLOR ESPECÍFICA (Diente)
    y selección del objeto CENTRAL.
    """
    imagen = cv2.imread(input_path)
    if imagen is None: return False, 0, 0

    # --- 1. MEJORA DE ILUMINACIÓN ---
    imagen_lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(imagen_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel_clahe = clahe.apply(l_channel)
    imagen_lab_mejorada = cv2.merge((l_channel_clahe, a_channel, b_channel))
    imagen_final = cv2.cvtColor(imagen_lab_mejorada, cv2.COLOR_LAB2BGR)

    # ==========================================
    # --- 2. DETECCIÓN DEL DIENTE (NUEVA ESTRATEGIA) ---
    # ==========================================
    
    # Convertir a HSV
    hsv = cv2.cvtColor(imagen_final, cv2.COLOR_BGR2HSV)
    
    # ESTRATEGIA: El diente es Brillante (V alto) y poco Saturado (S bajo)
    # La encía es oscura (V bajo) y muy saturada (S alto)
    
    # Rango para "Materia Dental" (Blanco/Amarillento brillante)
    # H: 0-179 (Cualquier tono, aunque suele ser amarillo/naranja)
    # S: 0-100 (Saturación BAJA - esto elimina la encía roja intensa)
    # V: 60-255 (Brillo MEDIO/ALTO - esto elimina el fondo oscuro)
    lower_tooth = np.array([0, 0, 80])     
    upper_tooth = np.array([180, 120, 255]) 
    
    # Crear máscara
    mask_tooth = cv2.inRange(hsv, lower_tooth, upper_tooth)
    
    # Limpieza fuerte para eliminar puntos sueltos
    kernel = np.ones((7,7), np.uint8)
    mask_tooth = cv2.morphologyEx(mask_tooth, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_tooth = cv2.morphologyEx(mask_tooth, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Encontrar contornos
    contours_tooth, _ = cv2.findContours(mask_tooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_tooth_area = 0
    
    # --- SELECCIÓN DEL DIENTE CENTRAL ---
    if contours_tooth:
        # Centro de la imagen
        height, width = imagen.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        best_contour = None
        min_dist = float('inf')
        
        # Filtramos contornos muy pequeños (ruido)
        min_area = 2000 
        
        for c in contours_tooth:
            area = cv2.contourArea(c)
            if area > min_area:
                # Calcular centro del contorno
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Distancia al centro de la foto
                    dist = np.sqrt((cX - center_x)**2 + (cY - center_y)**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_contour = c
        
        # Si encontramos un diente central...
        if best_contour is not None:
            total_tooth_area = cv2.contourArea(best_contour)
            
            # Usamos Convex Hull SOLO en este diente para que quede suave y "redondito"
            hull = cv2.convexHull(best_contour)
            
            # Dibujamos en ROJO
            cv2.drawContours(imagen_final, [hull], -1, (0, 0, 255), 3)

    print(f"Área de diente detectada: {total_tooth_area} píxeles")

    # ==========================================
    # --- 3. DETECCIÓN DE PLACA (HSV) ---
    # ==========================================
    # Rango para la Placa (Fluorescencia)
    lower_plaque = np.array([20, 50, 80])
    upper_plaque = np.array([35, 255, 220]) 
    mask_plaque = cv2.inRange(hsv, lower_plaque, upper_plaque)
    
    contours_plaque, _ = cv2.findContours(mask_plaque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_plaque_area = 0
    for c in contours_plaque:
        total_plaque_area += cv2.contourArea(c)
        
    # Dibujamos contornos de placa en CIAN
    cv2.drawContours(imagen_final, contours_plaque, -1, (255, 255, 0), 2)

    cv2.imwrite(output_path, imagen_final)
    
    return True, total_plaque_area, total_tooth_area

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

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

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
        if len(filenames) > 1:
            prev_index = (current_index - 1) % len(filenames)
            next_index = (current_index + 1) % len(filenames)
            prev_filename = filenames[prev_index]
            next_filename = filenames[next_index]
    except ValueError:
        abort(404)
    if not os.path.exists(input_path):
        abort(404)
    success,plaque_area, tooth_area = enhance_image(input_path, output_path)
    if success:
        turesky_grade, plaque_percentage = get_turesky_grade(plaque_area, tooth_area)
        return render_template('result.html', original_filename=filename, enhanced_filename=enhanced_filename, prev_filename=prev_filename, next_filename=next_filename,tooth_area=tooth_area, plaque_area=plaque_area, plaque_percentage=plaque_percentage, turesky_grade=turesky_grade)
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
    app.run(debug=True, host='0.0.0.0')

   