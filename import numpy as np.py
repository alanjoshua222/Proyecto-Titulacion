import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, Scrollbar, Frame, Scale, HORIZONTAL, Toplevel
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

class DentalAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Estación de Análisis Dental - V4 Inteligente")
        self.root.geometry("1400x850") 
        self.root.configure(bg="#2c3e50")

        # --- VARIABLES DE ESTADO ---
        self.original_cv_image = None
        self.resized_cv_image = None    
        self.final_result_image = None
        
        # Estructura: [{'points': [], 'glare': 225, 'sens': 0, 'percent': 0.0, 'grade': '', 'plaque_mask': ...}]
        self.teeth_data = [] 
        
        self.current_drawing = []       
        self.selected_tooth_idx = None 
        
        self.sidebar_visible = True
        self.thumbnail_cache = [] 
        self.canvas_width = 800
        self.canvas_height = 600
        
        self.editor_window = None 
        self.offset_x = 0
        self.offset_y = 0

        # Valores por defecto
        self.default_glare = 225
        self.default_sens = 0

        # Variables de Sliders
        self.var_glare = tk.IntVar(value=self.default_glare) 
        self.var_sens = tk.IntVar(value=self.default_sens)    

        # --- UI SETUP ---
        top_bar = tk.Frame(root, bg="#34495e", height=60)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        btn_style = {"font": ("Segoe UI", 9, "bold"), "bg": "#2980b9", "fg": "white", "padx": 10, "pady": 2}
        
        self.btn_toggle = tk.Button(top_bar, text="☰", command=self.sidebar, font=("Arial",14,"bold"), bg="#34495e", fg="white", bd=0)
        self.btn_toggle.pack(side=tk.LEFT, padx=10)                            

        tk.Label(top_bar, text="🦷 Análisis V4 (Anti-Falsos Positivos)", bg="#34495e", fg="white", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=10)
        
        # Botones
        tk.Button(top_bar, text="📂 Importar...", command=self.smart_import, **btn_style).pack(side=tk.LEFT, padx=10)
        
        # Botón manual de ajustes
        tk.Button(top_bar, text="⚙️ Panel de Ajustes", command=self.open_editor_window, bg="#34495e", fg="white", font=("Segoe UI", 9, "bold"), bd=1).pack(side=tk.LEFT, padx=10)

        tk.Button(top_bar, text="↩ Borrar Sel.", command=self.delete_selected_tooth, bg="#f39c12", fg="white", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(top_bar, text="🗑 Reiniciar", command=self.reset_selection, bg="#c0392b", fg="white", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=5)

        self.btn_save = tk.Button(top_bar, text="💾 Descargar", command=self.save_image, state=tk.DISABLED, **btn_style)
        self.btn_save.pack(side=tk.RIGHT, padx=20)

        # --- LAYOUT ---
        self.main_container = tk.Frame(root, bg="#2c3e50")
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Sidebar
        self.sidebar_width = 280
        self.sidebar_frame = tk.Frame(self.main_container, width=self.sidebar_width, bg="#233140")
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar_frame.pack_propagate(False)

        tk.Label(self.sidebar_frame, text="Galería", bg="#233140", fg="#bdc3c7", font=("Arial", 10, "bold")).pack(pady=(15, 5))

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

        # Canvas
        self.content_frame = tk.Frame(self.main_container, bg="#2c3e50")
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas_draw = tk.Canvas(self.content_frame, bg="black", width=self.canvas_width, height=self.canvas_height, cursor="crosshair")
        self.canvas_draw.pack(anchor=tk.CENTER, expand=True)
        
        self.canvas_draw.bind("<Button-1>", self.on_canvas_click)
        self.canvas_draw.bind("<B1-Motion>", self.drawing_motion)
        self.canvas_draw.bind("<ButtonRelease-1>", self.end_drawing)

        self.lbl_info = tk.Label(self.content_frame, text="1. Dibuja un diente -> Se analiza automáticamente.\n2. Haz clic en el diente para corregir parámetros.", bg="#2c3e50", fg="yellow", font=("Arial", 10))
        self.lbl_info.pack(pady=5)

    # ------------------------------------------------------------------
    # MOTOR DE ANÁLISIS V4 (CORREGIDO PARA EVITAR SELECCIÓN TOTAL)
    # ------------------------------------------------------------------
    def detect_plaque_engine(self, img, pts, glare_lim, sens_offset):
        """
        Algoritmo V4: 
        1. LAB 'b' (Amarillo)
        2. HSV 's' (Saturación) para confirmar que no es blanco
        3. Estadística: Si el diente es uniforme, subir umbral drásticamente
        """
        mask_tooth = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_tooth, [pts.reshape((-1, 1, 2))], 255)
        
        x, y, w, h = cv2.boundingRect(pts.reshape((-1, 1, 2)))
        if w==0 or h==0: return np.zeros(img.shape[:2], dtype=np.uint8)
        
        roi = img[y:y+h, x:x+w]
        roi_mask = mask_tooth[y:y+h, x:x+w]
        if roi.size == 0: return np.zeros(img.shape[:2], dtype=np.uint8)

        # --- 1. PREPARACIÓN DE CANALES ---
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)

        # Anti-Reflejos (Brillo L > limite)
        _, mask_glare = cv2.threshold(l, glare_lim, 255, cv2.THRESH_BINARY)
        
        # Mejora de Contraste en B (Amarillo)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        b_enhanced = clahe.apply(b)
        
        # --- 2. ANÁLISIS ESTADÍSTICO (EL CEREBRO NUEVO) ---
        valid_px = b_enhanced[roi_mask > 0]
        if len(valid_px) == 0: return np.zeros(img.shape[:2], dtype=np.uint8)

        mean_b = np.mean(valid_px)
        std_b = np.std(valid_px)
        
        # Calcular Otsu base
        otsu_thresh, _ = cv2.threshold(valid_px, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # --- CORRECCIÓN DE UNIFORMIDAD ---
        # Si la desviación estándar es baja (< 15), el diente es de un solo color (limpio).
        # En ese caso, Otsu falla y selecciona la mitad. Forzamos un umbral alto.
        # Si sens_offset es 0 (default), aplicamos la protección. Si el usuario mueve el slider, respetamos su decisión.
        
        # Umbral Mínimo Seguro: Promedio + 1 Desviación Estándar.
        # Esto significa: "Solo marca lo que sea claramente más amarillo que el promedio del diente"
        safe_threshold = mean_b + (1.2 * std_b)
        
        # El umbral final será el mayor entre Otsu y nuestro cálculo seguro
        # Restamos el slider de sensibilidad (sens_offset positivo = baja umbral = más placa)
        final_thresh = max(otsu_thresh, safe_threshold) - sens_offset
        
        # Clampear entre 0 y 255
        final_thresh = max(0, min(255, final_thresh))

        # Aplicar Umbral
        _, mask_plaque = cv2.threshold(b_enhanced, final_thresh, 255, cv2.THRESH_BINARY)
        
        # --- 3. FILTROS EXTRA ---
        # Excluir Encía (Rojo excesivo)
        _, mask_gum = cv2.threshold(a, 145, 255, cv2.THRESH_BINARY)
        
        # Excluir zonas muy pálidas (Baja saturación) - Protege el esmalte blanco
        # Placa suele tener saturación > 20-30. Esmalte limpio < 20.
        _, mask_pale = cv2.threshold(s_chan, 25, 255, cv2.THRESH_BINARY_INV)

        # Combinar todo
        mask_plaque = cv2.bitwise_and(mask_plaque, cv2.bitwise_not(mask_gum))
        mask_plaque = cv2.bitwise_and(mask_plaque, cv2.bitwise_not(mask_glare))
        # Opcional: Descomenta la siguiente línea si quieres ser estricto con dientes muy blancos
        # mask_plaque = cv2.bitwise_and(mask_plaque, cv2.bitwise_not(mask_pale)) 
        
        mask_plaque = cv2.bitwise_and(mask_plaque, mask_plaque, mask=roi_mask)
        
        full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        full_mask[y:y+h, x:x+w] = mask_plaque
        return full_mask

    # ------------------------------------------------------------------
    # FUNCIONES DE INTERFAZ & GESTIÓN
    # ------------------------------------------------------------------
    def run_single_analysis(self, idx):
        if self.resized_cv_image is None: return
        tooth = self.teeth_data[idx]
        pts_np = np.array(tooth['points'], np.int32)
        
        mask = self.detect_plaque_engine(self.resized_cv_image, pts_np, tooth['glare'], tooth['sens'])
        
        tooth_mask = np.zeros(self.resized_cv_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(tooth_mask, [pts_np.reshape((-1, 1, 2))], 255)
        
        tooth_area = cv2.countNonZero(tooth_mask)
        plaque_area = cv2.countNonZero(mask)
        
        percent = (plaque_area / tooth_area * 100) if tooth_area > 0 else 0
        grade, color = self.get_turesky_grade(percent)
        
        tooth['percent'] = percent
        tooth['grade'] = grade
        tooth['color'] = color
        tooth['plaque_mask'] = mask

    def delete_selected_tooth(self):
        if self.selected_tooth_idx is not None:
            del self.teeth_data[self.selected_tooth_idx]
            self.selected_tooth_idx = None
            self.var_glare.set(self.default_glare)
            self.var_sens.set(self.default_sens)
            if self.editor_window is not None and self.editor_window.winfo_exists():
                self.editor_window.destroy()
                self.editor_window = None
            self.redraw_all()
            self.lbl_info.config(text="Diente eliminado correctamente.")
            if not self.teeth_data: self.btn_save.config(state=tk.DISABLED)
        else:
            messagebox.showwarning("Atención", "Primero debes seleccionar un diente.")

    def undo_last_tooth(self):
        if self.teeth_data:
            self.teeth_data.pop()
            self.selected_tooth_idx = None
            if self.editor_window is not None and self.editor_window.winfo_exists():
                self.editor_window.destroy()
                self.editor_window = None
            self.redraw_all()
            self.lbl_info.config(text="Último diente deshecho.")
            if not self.teeth_data: self.btn_save.config(state=tk.DISABLED)

    def open_editor_window(self):
        if self.editor_window is not None and self.editor_window.winfo_exists():
            self.editor_window.lift(); return
        self.editor_window = Toplevel(self.root)
        self.editor_window.title("Editor de Parámetros")
        self.editor_window.geometry("320x300")
        self.editor_window.configure(bg="#34495e")
        try: self.editor_window.geometry(f"+{self.root.winfo_x() + self.root.winfo_width() - 350}+{self.root.winfo_y() + 150}")
        except: pass
        
        self.lbl_editor_title = tk.Label(self.editor_window, text="Configuración Global", bg="#34495e", fg="white", font=("Arial", 11, "bold"))
        self.lbl_editor_title.pack(pady=10)
        
        tk.Label(self.editor_window, text="Límite de Brillo", bg="#34495e", fg="#bdc3c7").pack(pady=(5,0))
        Scale(self.editor_window, from_=150, to=255, orient=HORIZONTAL, bg="#34495e", fg="white", highlightthickness=0, variable=self.var_glare, command=self.on_slider_change).pack(fill=tk.X, padx=20)
        tk.Label(self.editor_window, text="(Bajar para borrar reflejos)", bg="#34495e", fg="#7f8c8d", font=("Arial", 8)).pack()
        
        tk.Label(self.editor_window, text="Sensibilidad a Placa", bg="#34495e", fg="#bdc3c7").pack(pady=(15,0))
        Scale(self.editor_window, from_=-50, to=50, orient=HORIZONTAL, bg="#34495e", fg="white", highlightthickness=0, variable=self.var_sens, command=self.on_slider_change).pack(fill=tk.X, padx=20)
        tk.Label(self.editor_window, text="(Subir para detectar más)", bg="#34495e", fg="#7f8c8d", font=("Arial", 8)).pack()
        self.update_editor_title()

    def update_editor_title(self):
        if self.editor_window and self.editor_window.winfo_exists():
            if self.selected_tooth_idx is not None:
                self.lbl_editor_title.config(text=f"EDITANDO DIENTE #{self.selected_tooth_idx + 1}", fg="#f1c40f")
            else:
                self.lbl_editor_title.config(text="Configuración por Defecto", fg="white")

    def on_slider_change(self, val):
        if self.selected_tooth_idx is not None:
            self.teeth_data[self.selected_tooth_idx]['glare'] = self.var_glare.get()
            self.teeth_data[self.selected_tooth_idx]['sens'] = self.var_sens.get()
            self.run_single_analysis(self.selected_tooth_idx)
            self.redraw_all()
        else:
            self.default_glare = self.var_glare.get()
            self.default_sens = self.var_sens.get()

    def on_canvas_click(self, event):
        if self.resized_cv_image is None: return
        clicked_idx = self.check_click_on_tooth(event.x, event.y)
        if clicked_idx is not None:
            self.selected_tooth_idx = clicked_idx
            tooth = self.teeth_data[clicked_idx]
            self.var_glare.set(tooth['glare'])
            self.var_sens.set(tooth['sens'])
            self.open_editor_window()
            self.update_editor_title()
            self.lbl_info.config(text=f"Editando Diente #{clicked_idx+1}.")
            self.redraw_all()
        else:
            if self.selected_tooth_idx is not None:
                self.selected_tooth_idx = None
                self.update_editor_title()
                self.var_glare.set(self.default_glare)
                self.var_sens.set(self.default_sens)
                self.redraw_all()
                self.lbl_info.config(text="Modo Dibujo.")
            if self.is_inside_image(event.x, event.y):
                self.current_drawing = [(event.x, event.y)]

    def check_click_on_tooth(self, x, y):
        click_pt = (x - self.offset_x, y - self.offset_y)
        for i in range(len(self.teeth_data)-1, -1, -1):
            pts = np.array(self.teeth_data[i]['points'], np.int32)
            if cv2.pointPolygonTest(pts, click_pt, False) >= 0: return i
        return None

    def drawing_motion(self, event):
        if not self.current_drawing: return
        h, w = self.resized_cv_image.shape[:2]
        x = max(self.offset_x, min(event.x, self.offset_x + w - 1))
        y = max(self.offset_y, min(event.y, self.offset_y + h - 1))
        px, py = self.current_drawing[-1]
        self.canvas_draw.create_line(px, py, x, y, fill="#00ff00", width=2, tags="temp_line")
        self.current_drawing.append((x, y))

    def end_drawing(self, event):
        if not self.current_drawing: return
        if len(self.current_drawing) < 5: 
            self.current_drawing = []
            self.canvas_draw.delete("temp_line")
            return
        self.drawing_motion(event)
        pts_img = [(p[0] - self.offset_x, p[1] - self.offset_y) for p in self.current_drawing]
        new_tooth = {'points': pts_img, 'glare': self.default_glare, 'sens': self.default_sens, 'percent': 0.0, 'grade': '...'}
        self.teeth_data.append(new_tooth)
        self.run_single_analysis(len(self.teeth_data)-1)
        self.current_drawing = []
        self.canvas_draw.delete("temp_line")
        self.btn_save.config(state=tk.NORMAL)
        self.redraw_all()
        self.lbl_info.config(text="Diente analizado.")

    def redraw_all(self):
        if self.resized_cv_image is None: return
        self.canvas_draw.delete("all")
        display_img = self.resized_cv_image.copy()
        for i, tooth in enumerate(self.teeth_data):
            pts_np = np.array(tooth['points'], np.int32).reshape((-1, 1, 2))
            color = (0, 255, 255) if i == self.selected_tooth_idx else (255, 255, 0)
            thick = 3 if i == self.selected_tooth_idx else 2
            cv2.polylines(display_img, [pts_np], True, color, thick)
            if 'plaque_mask' in tooth:
                cnts, _ = cv2.findContours(tooth['plaque_mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display_img, cnts, -1, (0, 0, 255), 1)
            if 'percent' in tooth:
                M = cv2.moments(pts_np)
                cx = int(M["m10"]/M["m00"]) if M["m00"]!=0 else tooth['points'][0][0]
                cy = int(M["m01"]/M["m00"]) if M["m00"]!=0 else tooth['points'][0][1]
                txt = f"#{i+1} {tooth['grade']} ({tooth['percent']:.1f}%)"
                (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display_img, (cx-10, cy-20), (cx+w+10, cy+10), (0,0,0), -1)
                cv2.putText(display_img, txt, (cx-5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tooth['color'], 1)
        self.final_result_image = display_img
        i = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        itk = ImageTk.PhotoImage(Image.fromarray(i))
        self.canvas_draw.create_image(self.offset_x, self.offset_y, image=itk, anchor=tk.NW)
        self.canvas_draw.image = itk 

    def reset_selection(self):
        self.teeth_data = []
        self.selected_tooth_idx = None
        if self.editor_window is not None and self.editor_window.winfo_exists():
            self.editor_window.destroy()
            self.editor_window = None
        self.redraw_all()
        self.btn_save.config(state=tk.DISABLED)

    def process_selected_image(self, path):
        img = cv2.imread(path)
        if img is None: return
        self.original_cv_image = img
        h, w = img.shape[:2]
        s = min(self.canvas_width/w, self.canvas_height/h)
        self.resized_cv_image = cv2.resize(img, (int(w*s), int(h*s)))
        self.offset_x = (self.canvas_width - self.resized_cv_image.shape[1]) // 2
        self.offset_y = (self.canvas_height - self.resized_cv_image.shape[0]) // 2
        self.teeth_data = []
        self.selected_tooth_idx = None
        if self.editor_window is not None and self.editor_window.winfo_exists():
            self.editor_window.destroy()
            self.editor_window = None
        self.default_glare = 225
        self.default_sens = 0
        self.var_glare.set(225)
        self.var_sens.set(0)
        self.redraw_all()

    def smart_import(self):
        pop = Toplevel(self.root); pop.title("Importar"); pop.geometry("300x160")
        try: pop.geometry(f"+{self.root.winfo_x()+300}+{self.root.winfo_y()+200}")
        except: pass
        tk.Button(pop, text="📄 Archivos", command=lambda:[pop.destroy(), self.import_files_logic()], bg="#2980b9", fg="white", width=25).pack(pady=10)
        tk.Button(pop, text="📁 Carpeta", command=lambda:[pop.destroy(), self.import_folder_logic()], bg="#27ae60", fg="white", width=25).pack(pady=5)

    def import_files_logic(self):
        fs = filedialog.askopenfilenames(filetypes=[("Img", "*.jpg *.png *.jpeg *.webp")])
        if fs: self.process_import_list(list(fs))
    def import_folder_logic(self):
        d = filedialog.askdirectory()
        if d: self.process_import_list([os.path.join(d,f) for f in os.listdir(d) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    def process_import_list(self, fs):
        for w in self.scrollable_frame.winfo_children(): w.destroy()
        self.thumbnail_cache = []
        for p in fs:
            try:
                im = Image.open(p); im.thumbnail((180,180)); ph = ImageTk.PhotoImage(im)
                fr = tk.Frame(self.scrollable_frame, bg="#34495e", pady=2); fr.pack(fill=tk.X, pady=2, padx=5)
                tk.Button(fr, image=ph, bg="#2c3e50", bd=0, command=lambda pa=p: self.process_selected_image(pa)).pack()
                self.thumbnail_cache.append(ph)
            except: pass
        if fs: self.process_selected_image(fs[0])

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
    def is_inside_image(self, x, y):
        if self.resized_cv_image is None: return False
        h, w = self.resized_cv_image.shape[:2]
        return self.offset_x <= x < (self.offset_x + w) and self.offset_y <= y < (self.offset_y + h)
    def sidebar(self):
        if self.sidebar_visible: self.sidebar_frame.pack_forget(); self.sidebar_visible = False
        else: self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, before=self.content_frame); self.sidebar_visible = True
    def _on_mousewheel(self, e):
        if self.sidebar_visible: self.canvas_gallery.yview_scroll(int(-1*(e.delta/120)), "units")

if __name__ == "__main__":
    root = tk.Tk()
    app = DentalAnalyzerApp(root)
    root.mainloop()