import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'mp4', 'avi', 'mov', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def list_uploaded_files():
    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
    files.sort()
    return files

@app.route('/')
def index():
    # siempre listamos los archivos reales que hay en la carpeta
    files = list_uploaded_files()
    return render_template('index.html', filenames=files)

@app.route('/upload', methods=['POST'])
def upload_file():
    # POST-redirect-GET: después de guardar redirigimos a index para que muestre todos los archivos
    if 'archivos' not in request.files:
        return redirect(url_for('index'))

    files = request.files.getlist('archivos')
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
    return redirect(url_for('index'))

@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    # aseguramos el filename y que exista en la carpeta (evitamos path traversal)
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            print("Error al eliminar:", e)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
