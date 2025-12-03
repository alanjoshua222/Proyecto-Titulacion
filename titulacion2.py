import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, Scrollbar, Frame, Scale, HORIZONTAL, Toplevel
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

class DentalAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Estación de Análisis Dental - Centrado Automático")
        self.root.geometry("1400x850") 
        self.root.configure(bg="#2c3e50")

        # --- VARIABLES DE ESTADO ---
        self.original_cv_image = None
        self.resized_cv_image = None    
        self.final_result_image = None
        self.current_drawing = []       
        self.teeth_rois = []            
        self.sidebar_visible = True
        self.thumbnail_cache = [] 
        self.canvas_width = 800
        self.canvas_height = 600
        self.preview_mode = False
        self.settings_window = None

        # Variables de posicionamiento (Centrado)
        self.offset_x = 0
        self.offset_y = 0

        # Configuración Persistente
        self.var_glare = tk.IntVar(value=225) 
        self.var_sens = tk.IntVar(value=0)    

        # --- UI SETUP ---
        top_bar = tk.Frame(root, bg="#34495e", height=60)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        btn_style = {"font": ("Segoe UI", 9, "bold"), "bg": "#2980b9", "fg": "white", "padx": 10, "pady": 2}
        
        self.btn_toggle = tk.Button(top_bar, text="☰", command=self.sidebar, font=("Arial",14,"bold"), bg="#34495e", fg="white", bd=0)
        self.btn_toggle.pack(side=tk.LEFT, padx=10)                            

        tk.Label(top_bar, text="🦷 Análisis QH-T", bg="#34495e", fg="white", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=10)
        
        # Botones Principales
        tk.Button(top_bar, text="📂 Importar...", command=self.smart_import, **btn_style).pack(side=tk.LEFT, padx=10)
        tk.Button(top_bar, text="⚙️ Ajustes", command=self.open_settings_window, bg="#34495e", fg="white", font=("Segoe UI", 9, "bold"), bd=1).pack(side=tk.LEFT, padx=10)

        tk.Button(top_bar, text="↩ Deshacer", command=self.undo_last_tooth, bg="#f39c12", fg="white", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(top_bar, text="🗑 Reiniciar", command=self.reset_selection, bg="#c0392b", fg="white", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=5)

        self.btn_analyze = tk.Button(top_bar, text="⚡ EJECUTAR ANÁLISIS", command=self.run_analysis, state=tk.DISABLED, bg="#27ae60", fg="white", font=("Segoe UI", 10, "bold"), padx=15)
        self.btn_analyze.pack(side=tk.LEFT, padx=20)

        self.btn_preview = tk.Button(top_bar, text="👁 Vista Previa", command=self.toggle_preview_mode, state=tk.DISABLED, bg="#8e44ad", fg="white", font=("Segoe UI", 9, "bold"))
        self.btn_preview.pack(side=tk.LEFT, padx=10)

        self.btn_save = tk.Button(top_bar, text="💾 Descargar", command=self.save_image, state=tk.DISABLED, **btn_style)
        self.btn_save.pack(side=tk.RIGHT, padx=20)

        self.main_container = tk.Frame(root, bg="#2c3e50")
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # --- SIDEBAR ---
        self.sidebar_width = 280
        self.sidebar_frame = tk.Frame(self.main_container, width=self.sidebar_width, bg="#233140")
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar_frame.pack_propagate(False)

        tk.Label(self.sidebar_frame, text="Galería de Pacientes", bg="#233140", fg="#bdc3c7", font=("Arial", 10, "bold")).pack(pady=(15, 5))

        gallery_container = tk.Frame(self.sidebar_frame, bg="#233140")
        gallery_container.pack(fill=tk.BOTH, expand=True)

        self.scrollbar_gallery = Scrollbar(gallery_container, orient="vertical")
        self.scrollbar_gallery.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas_gallery = Canvas(gallery_container, bg="#233140", highlightthickness=0, yscrollcommand=self.scrollbar_gallery.set)
        self.canvas_gallery.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar_gallery.config(command=self.canvas_gallery.yview)

        self.scrollable_frame = Frame(self.canvas_gallery, bg="#233140")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas_gallery.configure(scrollregion=self.canvas_gallery.bbox("all")))
        self.canvas_gallery.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=self.sidebar_width-20)
        self.canvas_gallery.bind_all("<MouseWheel>", self._on_mousewheel)

        # --- CANVAS DERECHO (VISUALIZACIÓN) ---
        self.content_frame = tk.Frame(self.main_container, bg="#2c3e50")
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # El Canvas ahora tiene el tamaño fijo definido
        self.canvas_draw = tk.Canvas(self.content_frame, bg="black", width=self.canvas_width, height=self.canvas_height, cursor="crosshair")
        self.canvas_draw.pack(anchor=tk.CENTER, expand=True)
        self.canvas_draw.bind("<Button-1>", self.start_drawing)
        self.canvas_draw.bind("<B1-Motion>", self.drawing_motion)
        self.canvas_draw.bind("<ButtonRelease-1>", self.end_drawing)

        self.lbl_info = tk.Label(self.content_frame, text="Importa imágenes para comenzar.", bg="#2c3e50", fg="yellow", font=("Arial", 10))
        self.lbl_info.pack(pady=5)

    # ------------------------------------------------------------------
    # VENTANA FLOTANTE DE AJUSTES
    # ------------------------------------------------------------------
    def open_settings_window(self):
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return

        self.settings_window = Toplevel(self.root)
        self.settings_window.title("Configuración")
        self.settings_window.geometry("300x250")
        self.settings_window.configure(bg="#34495e")
        self.settings_window.resizable(False, False)
        
        try:
            x = self.root.winfo_x() + 100
            y = self.root.winfo_y() + 100
            self.settings_window.geometry(f"+{x}+{y}")
        except: pass

        tk.Label(self.settings_window, text="Parámetros del Algoritmo", bg="#34495e", fg="white", font=("Arial", 10, "bold")).pack(pady=10)

        tk.Label(self.settings_window, text="Límite de Brillo (Anti-Reflejos)", bg="#34495e", fg="#bdc3c7").pack(pady=(5,0))
        s_glare = Scale(self.settings_window, from_=150, to=255, orient=HORIZONTAL, bg="#34495e", fg="white", 
                        highlightthickness=0, variable=self.var_glare, command=self.on_slider_change)
        s_glare.pack(fill=tk.X, padx=20)
        
        tk.Label(self.settings_window, text="Sensibilidad a Placa", bg="#34495e", fg="#bdc3c7").pack(pady=(15,0))
        s_sens = Scale(self.settings_window, from_=-50, to=50, orient=HORIZONTAL, bg="#34495e", fg="white", 
                       highlightthickness=0, variable=self.var_sens, command=self.on_slider_change)
        s_sens.pack(fill=tk.X, padx=20)

        tk.Button(self.settings_window, text="Cerrar", command=self.settings_window.destroy, bg="#c0392b", fg="white").pack(pady=15)

    # ------------------------------------------------------------------
    # GESTIÓN DE IMÁGENES (CON CENTRADO)
    # ------------------------------------------------------------------
    def smart_import(self):
        pop = Toplevel(self.root)
        pop.title("Importar")
        pop.geometry("300x160")
        pop.configure(bg="#34495e")
        pop.resizable(False, False)
        try:
            x = self.root.winfo_x() + (self.root.winfo_width()//2) - 150
            y = self.root.winfo_y() + (self.root.winfo_height()//2) - 80
            pop.geometry(f"+{x}+{y}")
        except: pass

        tk.Label(pop, text="¿Qué deseas importar?", bg="#34495e", fg="white", font=("Segoe UI", 11, "bold")).pack(pady=15)
        tk.Button(pop, text="📄 Archivos (Selección múltiple)", command=lambda: [pop.destroy(), self.import_files_logic()], bg="#2980b9", fg="white", font=("Segoe UI", 10), width=25).pack(pady=5)
        tk.Button(pop, text="📁 Carpeta Completa", command=lambda: [pop.destroy(), self.import_folder_logic()], bg="#27ae60", fg="white", font=("Segoe UI", 10), width=25).pack(pady=5)

    def import_files_logic(self):
        file_paths = filedialog.askopenfilenames(title="Seleccionar Imágenes", filetypes=[("Imágenes Dentales", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")])
        if file_paths: self.process_import_list(list(file_paths))

    def import_folder_logic(self):
        folder_path = filedialog.askdirectory(title="Seleccionar Carpeta")
        if folder_path:
            valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
            files_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)]
            if files_list: self.process_import_list(files_list)
            else: messagebox.showwarning("Vacío", "No hay imágenes en esa carpeta.")

    def process_import_list(self, files_list):
        self.update_gallery(files_list)
        if len(files_list) >= 1:
            self.process_selected_image(files_list[0])
            self.lbl_info.config(text=f"Cargadas {len(files_list)} imágenes.")

    def update_gallery(self, file_list):
        for w in self.scrollable_frame.winfo_children(): w.destroy()
        self.thumbnail_cache = []
        for path in file_list:
            try:
                im = Image.open(path); im.thumbnail((180, 180)); ph = ImageTk.PhotoImage(im)
                fr = tk.Frame(self.scrollable_frame, bg="#34495e", pady=2); fr.pack(fill=tk.X, pady=2, padx=5)
                btn = tk.Button(fr, image=ph, bg="#2c3e50", bd=0, command=lambda p=path: self.process_selected_image(p))
                btn.pack()
                self.thumbnail_cache.append(ph)
            except: pass

    def process_selected_image(self, path):
        img = cv2.imread(path)
        if img is None: return
        self.original_cv_image = img
        
        # 1. Redimensionar ajustando al maximo (800x600)
        h, w = img.shape[:2]
        scale = min(self.canvas_width/w, self.canvas_height/h)
        new_w, new_h = int(w*scale), int(h*scale)
        self.resized_cv_image = cv2.resize(img, (new_w, new_h))
        
        # 2. Calcular OFFSETS para centrar
        self.offset_x = (self.canvas_width - new_w) // 2
        self.offset_y = (self.canvas_height - new_h) // 2
        
        self.reset_selection()

    def show_image_on_canvas(self, cv_img):
        i = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        itk = ImageTk.PhotoImage(Image.fromarray(i))
        self.canvas_draw.delete("all")
        # Usamos los offsets calculados para dibujar la imagen en el centro
        self.canvas_draw.create_image(self.offset_x, self.offset_y, image=itk, anchor=tk.NW)
        self.canvas_draw.image = itk 

    # ------------------------------------------------------------------
    # LÓGICA DE DIBUJO CON OFFSET (COORDENADAS CORREGIDAS)
    # ------------------------------------------------------------------
    def is_inside_image(self, x, y):
        """Verifica si el click cae dentro de la imagen centrada"""
        if self.resized_cv_image is None: return False
        h, w = self.resized_cv_image.shape[:2]
        # x debe estar entre el offset y el offset+ancho
        return self.offset_x <= x < (self.offset_x + w) and self.offset_y <= y < (self.offset_y + h)

    def start_drawing(self, event):
        if self.resized_cv_image is None: return
        if self.is_inside_image(event.x, event.y):
            self.current_drawing = [(event.x, event.y)]

    def drawing_motion(self, event):
        if self.resized_cv_image is None: return
        h, w = self.resized_cv_image.shape[:2]
        
        # Clampear coordenadas dentro de los límites de la imagen (respetando offsets)
        x = max(self.offset_x, min(event.x, self.offset_x + w - 1))
        y = max(self.offset_y, min(event.y, self.offset_y + h - 1))
        
        if self.current_drawing:
            px, py = self.current_drawing[-1]
            self.canvas_draw.create_line(px, py, x, y, fill="#00ff00", width=2, tags="temp_line")
        self.current_drawing.append((x, y))

    def end_drawing(self, event):
        if self.resized_cv_image is None or len(self.current_drawing) < 5: return
        self.drawing_motion(event) # Asegurar último punto
        self.teeth_rois.append(self.current_drawing)
        
        # Dibujar polígono en Canvas
        pts = [i for s in self.current_drawing for i in s]
        self.canvas_draw.create_polygon(pts, outline="cyan", fill="", width=2, tags="saved_tooth")
        
        self.current_drawing = []
        self.canvas_draw.delete("temp_line")
        self.btn_analyze.config(state=tk.NORMAL)
        self.btn_preview.config(state=tk.NORMAL)
        self.lbl_info.config(text=f"Dientes seleccionados: {len(self.teeth_rois)}")

    # ------------------------------------------------------------------
    # ANÁLISIS (TRADUCCIÓN CANVAS -> IMAGEN)
    # ------------------------------------------------------------------
    def detect_plaque_dynamic(self, img_bgr, roi_points_canvas):
        glare_limit = self.var_glare.get()
        sensitivity_offset = self.var_sens.get()

        mask_tooth = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        
        # IMPORTANTE: Restar offsets para pasar de coords Canvas a coords Imagen
        pts_canvas = np.array(roi_points_canvas, np.int32)
        pts_img = pts_canvas - np.array([self.offset_x, self.offset_y])
        pts_img = pts_img.reshape((-1, 1, 2))
        
        cv2.fillPoly(mask_tooth, [pts_img], 255)
        
        x, y, w, h = cv2.boundingRect(pts_img)
        # Protección por si el recorte sale de rango (aunque no debería)
        if w <= 0 or h <= 0: return np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        
        roi_color = img_bgr[y:y+h, x:x+w]
        roi_mask = mask_tooth[y:y+h, x:x+w]
        
        if roi_color.size == 0: return np.zeros(img_bgr.shape[:2], dtype=np.uint8)

        lab = cv2.cvtColor(roi_color, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        _, mask_glare = cv2.threshold(l, glare_limit, 255, cv2.THRESH_BINARY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        b_enhanced = clahe.apply(b)

        valid_pixels = b_enhanced[roi_mask > 0]
        if len(valid_pixels) == 0: return np.zeros(img_bgr.shape[:2], dtype=np.uint8)

        otsu_thresh, _ = cv2.threshold(valid_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final_thresh = max(0, min(255, otsu_thresh - sensitivity_offset))

        _, mask_plaque_roi = cv2.threshold(b_enhanced, final_thresh, 255, cv2.THRESH_BINARY)
        _, mask_gum = cv2.threshold(a, 145, 255, cv2.THRESH_BINARY)
        
        mask_plaque_roi = cv2.bitwise_and(mask_plaque_roi, cv2.bitwise_not(mask_gum))
        mask_plaque_roi = cv2.bitwise_and(mask_plaque_roi, cv2.bitwise_not(mask_glare))
        mask_plaque_roi = cv2.bitwise_and(mask_plaque_roi, mask_plaque_roi, mask=roi_mask)

        full_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        full_mask[y:y+h, x:x+w] = mask_plaque_roi
        return full_mask

    def run_analysis(self):
        if self.resized_cv_image is None or not self.teeth_rois: return
        if self.preview_mode: self.toggle_preview_mode()

        analysis_img = self.resized_cv_image.copy()
        
        for idx, points in enumerate(self.teeth_rois):
            # Detectar (Ya maneja la traducción de coords internamente)
            plaque_mask = self.detect_plaque_dynamic(analysis_img, points)
            
            # Para dibujar en la imagen, necesitamos coords imagen
            pts_canvas = np.array(points, np.int32)
            pts_img = pts_canvas - np.array([self.offset_x, self.offset_y])
            pts_img = pts_img.reshape((-1, 1, 2))
            
            cv2.polylines(analysis_img, [pts_img], True, (255, 255, 0), 2)
            
            tooth_mask = np.zeros(analysis_img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(tooth_mask, [pts_img], 255)
            tooth_area = cv2.countNonZero(tooth_mask)
            plaque_area = cv2.countNonZero(plaque_mask)
            percent = (plaque_area / tooth_area * 100) if tooth_area > 0 else 0
            grade_text, color_bgr = self.get_turesky_grade(percent)

            contours_plaque, _ = cv2.findContours(plaque_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(analysis_img, contours_plaque, -1, (0, 0, 255), 1)
            
            M = cv2.moments(pts_img)
            cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else pts_img[0][0][0]
            cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else pts_img[0][0][1]
            
            label = f"#{idx+1}: {grade_text} ({percent:.1f}%)"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(analysis_img, (cX - 10, cY - 20), (cX + w_text + 10, cY + 10), (0,0,0), -1)
            cv2.putText(analysis_img, label, (cX - 5, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

        self.final_result_image = analysis_img
        self.show_image_on_canvas(self.final_result_image)
        self.btn_save.config(state=tk.NORMAL)
        messagebox.showinfo("Listo", "Análisis finalizado.")

    def update_preview(self):
        if self.resized_cv_image is None or not self.teeth_rois: return
        preview_img = np.zeros_like(self.resized_cv_image)
        for points in self.teeth_rois:
            mask = self.detect_plaque_dynamic(self.resized_cv_image, points)
            preview_img[mask > 0] = (255, 255, 255)
            
            # Dibujar contorno referencia
            pts_canvas = np.array(points, np.int32)
            pts_img = pts_canvas - np.array([self.offset_x, self.offset_y])
            pts_img = pts_img.reshape((-1, 1, 2))
            
            cv2.polylines(preview_img, [pts_img], True, (255, 0, 0), 1)
        self.show_image_on_canvas(preview_img)

    # ------------------------------------------------------------------
    # RESTO DE UTILIDADES
    # ------------------------------------------------------------------
    def toggle_preview_mode(self):
        self.preview_mode = not self.preview_mode
        if self.preview_mode:
            self.btn_preview.config(text="❌ Salir Vista Previa", bg="#e74c3c")
            self.update_preview()
        else:
            self.btn_preview.config(text="👁 Vista Previa", bg="#8e44ad")
            self.refresh_canvas_view()

    def on_slider_change(self, val):
        if self.preview_mode: self.update_preview()

    def undo_last_tooth(self):
        if self.teeth_rois: self.teeth_rois.pop(); self.refresh_canvas_view()

    def reset_selection(self):
        self.teeth_rois = []; self.refresh_canvas_view(); self.btn_analyze.config(state=tk.DISABLED); self.btn_preview.config(state=tk.DISABLED); self.lbl_info.config(text="Reiniciado.")

    def refresh_canvas_view(self):
        if self.resized_cv_image is None: return
        self.canvas_draw.delete("all")
        self.show_image_on_canvas(self.resized_cv_image)
        for poly in self.teeth_rois:
            pts = [i for s in poly for i in s]
            self.canvas_draw.create_polygon(pts, outline="cyan", fill="", width=2, tags="saved_tooth")

    def get_turesky_grade(self, p):
        if p == 0: return "Grado 0", (0, 255, 0)
        elif p < 25: return "Grado 1", (0, 255, 255)
        elif p < 40: return "Grado 2", (0, 165, 255)
        elif p < 60: return "Grado 3", (0, 128, 255)
        elif p < 80: return "Grado 4", (0, 0, 255)
        else: return "Grado 5", (0, 0, 139)

    def save_image(self):
        if self.final_result_image is not None:
            f = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")])
            if f: cv2.imwrite(f, self.final_result_image); messagebox.showinfo("Guardado", "Listo.")
            
    def sidebar(self):
        if self.sidebar_visible: self.sidebar_frame.pack_forget(); self.sidebar_visible = False
        else: self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, before=self.content_frame); self.sidebar_visible = True

    def _on_mousewheel(self, e):
        if self.sidebar_visible: self.canvas_gallery.yview_scroll(int(-1*(e.delta/120)), "units")

if __name__ == "__main__":
    root = tk.Tk()
    app = DentalAnalyzerApp(root)
    root.mainloop()