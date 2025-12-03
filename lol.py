import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, Scrollbar, Frame
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

class DentalAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deteccion de placa bacteriana")
        self.root.geometry("1400x800") 
        self.root.configure(bg="#2c3e50")

        # --- VARIABLES DE ESTADO ---
        self.original_cv_image = None
        self.resized_cv_image = None    
        self.final_result_image = None
        
        self.current_drawing = []       
        self.teeth_rois = []            
        
        self.sidebar_visible = True
        self.thumbnail_cache = [] 

        # Dimensiones del área de dibujo
        self.canvas_width = 800
        self.canvas_height = 600

        # --- DISEÑO (LAYOUT) ---
        top_bar = tk.Frame(root, bg="#34495e", height=60)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        btn_style = {"font": ("Segoe UI", 9, "bold"), "bg": "#2980b9", "fg": "white", "padx": 10, "pady": 2}
        
        self.btn_toggle = tk.Button(top_bar, text="☰", command=self.sidebar, font=("Arial",14,"bold"), bg="#34495e", fg="white", bd=0)
        self.btn_toggle.pack(side=tk.LEFT, padx=10)                            

        tk.Label(top_bar, text="🦷 Análisis QH-T", bg="#34495e", fg="white", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=10)
        
        # --- BOTONES DE CARGA ---
        # 1. Botón Carpeta
        tk.Button(top_bar, text="📂 Carpeta", command=self.select_folder, **btn_style).pack(side=tk.LEFT, padx=5)
        # 2. Botón Imagen Individual (NUEVO)
        tk.Button(top_bar, text="📄 Cargar Imagen", command=self.load_single_image, **btn_style).pack(side=tk.LEFT, padx=5)
        
        # --- BOTONES DE EDICIÓN ---
        tk.Button(top_bar, text="↩ Deshacer", command=self.undo_last_tooth, bg="#f39c12", fg="white", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(top_bar, text="🗑 Reiniciar", command=self.reset_selection, bg="#c0392b", fg="white", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=5)

        # Botón Analizar
        self.btn_analyze = tk.Button(top_bar, text="⚡ ANALIZAR SELECCIÓN", command=self.run_analysis, state=tk.DISABLED, bg="#27ae60", fg="white", font=("Segoe UI", 10, "bold"), padx=15)
        self.btn_analyze.pack(side=tk.LEFT, padx=20)

        # Botón Guardar
        self.btn_save = tk.Button(top_bar, text="💾 Descargar", command=self.save_image, state=tk.DISABLED, **btn_style)
        self.btn_save.pack(side=tk.RIGHT, padx=20)

        # CONTENEDOR PRINCIPAL
        self.main_container = tk.Frame(root, bg="#2c3e50")
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # PANEL IZQUIERDO (GALERÍA)
        self.sidebar_width = 240
        self.sidebar_frame = tk.Frame(self.main_container, width=self.sidebar_width, bg="#233140")
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar_frame.pack_propagate(False)

        tk.Label(self.sidebar_frame, text="Galería", bg="#233140", fg="#bdc3c7").pack(pady=5)
        
        self.canvas_gallery = Canvas(self.sidebar_frame, bg="#233140", highlightthickness=0)
        self.scrollbar_gallery = Scrollbar(self.sidebar_frame, orient="vertical", command=self.canvas_gallery.yview)
        self.scrollable_frame = Frame(self.canvas_gallery, bg="#233140")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas_gallery.configure(scrollregion=self.canvas_gallery.bbox("all")))
        self.canvas_gallery.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=self.sidebar_width-20)
        self.canvas_gallery.configure(yscrollcommand=self.scrollbar_gallery.set)
        self.canvas_gallery.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.scrollbar_gallery.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_gallery.bind_all("<MouseWheel>", self._on_mousewheel)

        # PANEL DERECHO (CANVAS)
        self.content_frame = tk.Frame(self.main_container, bg="#2c3e50")
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas_draw = tk.Canvas(self.content_frame, bg="black", width=self.canvas_width, height=self.canvas_height, cursor="crosshair")
        self.canvas_draw.pack(anchor=tk.CENTER, expand=True)
        
        self.canvas_draw.bind("<Button-1>", self.start_drawing)
        self.canvas_draw.bind("<B1-Motion>", self.drawing_motion)
        self.canvas_draw.bind("<ButtonRelease-1>", self.end_drawing)

        self.lbl_info = tk.Label(self.content_frame, text="1. Carga Imagen  ->  2. Dibuja contorno  ->  3. Analizar", bg="#2c3e50", fg="yellow", font=("Arial", 10))
        self.lbl_info.pack(pady=5)

    # ------------------------------------------------------------------
    # NUEVA FUNCION: CARGAR UNA SOLA IMAGEN
    # ------------------------------------------------------------------
    def load_single_image(self):
        """Abre un diálogo para cargar una sola imagen externa"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar Imagen Dental",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        if not file_path:
            return
        
        # Reutilizamos la lógica de procesamiento que ya tenemos
        self.process_selected_image(file_path)

    # ------------------------------------------------------------------
    # LÓGICA DE DIBUJO (COORDENADAS FIJAS)
    # ------------------------------------------------------------------
    def is_inside_image(self, x, y):
        if self.resized_cv_image is None: return False
        h, w = self.resized_cv_image.shape[:2]
        return 0 <= x < w and 0 <= y < h

    def start_drawing(self, event):
        if self.resized_cv_image is None: return
        if self.is_inside_image(event.x, event.y):
            self.current_drawing = [(event.x, event.y)]

    def drawing_motion(self, event):
        if self.resized_cv_image is None: return
        h, w = self.resized_cv_image.shape[:2]
        x = max(0, min(event.x, w-1))
        y = max(0, min(event.y, h-1))
        
        if len(self.current_drawing) > 0:
            prev_x, prev_y = self.current_drawing[-1]
            self.canvas_draw.create_line(prev_x, prev_y, x, y, fill="#00ff00", width=2, tags="temp_line")
        
        self.current_drawing.append((x, y))

    def end_drawing(self, event):
        if self.resized_cv_image is None or len(self.current_drawing) < 5: return
        
        x_start, y_start = self.current_drawing[0]
        x_end, y_end = self.current_drawing[-1]
        self.canvas_draw.create_line(x_end, y_end, x_start, y_start, fill="#00ff00", width=2, tags="temp_line")
        
        self.teeth_rois.append(self.current_drawing)
        
        flat_points = [item for sublist in self.current_drawing for item in sublist]
        self.canvas_draw.create_polygon(flat_points, outline="cyan", fill="", width=2, tags="saved_tooth")
        
        self.current_drawing = []
        self.canvas_draw.delete("temp_line")
        
        self.btn_analyze.config(state=tk.NORMAL)
        self.lbl_info.config(text=f"Dientes seleccionados: {len(self.teeth_rois)}")

    def undo_last_tooth(self):
        if self.teeth_rois:
            self.teeth_rois.pop()
            self.refresh_canvas_view()
            self.lbl_info.config(text=f"Dientes seleccionados: {len(self.teeth_rois)}")

    def reset_selection(self):
        self.teeth_rois = []
        self.current_drawing = []
        self.refresh_canvas_view()
        self.btn_analyze.config(state=tk.DISABLED)
        self.lbl_info.config(text="Selección reiniciada.")

    def refresh_canvas_view(self):
        if self.resized_cv_image is None: return
        self.canvas_draw.delete("all")
        self.show_image_on_canvas(self.resized_cv_image)
        for poly in self.teeth_rois:
            flat_points = [item for sublist in poly for item in sublist]
            self.canvas_draw.create_polygon(flat_points, outline="cyan", fill="", width=2, tags="saved_tooth")

    # ------------------------------------------------------------------
    # ANÁLISIS FIJO (PLACA DENTRO DE SELECCIÓN)
    # ------------------------------------------------------------------
    def run_analysis(self):
        if self.resized_cv_image is None or not self.teeth_rois: return
        
        analysis_img = self.resized_cv_image.copy()
        hsv_img = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2HSV)
        
        lower_plaque = np.array([20, 50, 60])
        upper_plaque = np.array([40, 255, 255])
        
        for idx, points in enumerate(self.teeth_rois):
            tooth_mask = np.zeros(analysis_img.shape[:2], dtype=np.uint8)
            pts_np = np.array(points, np.int32)
            pts_np = pts_np.reshape((-1, 1, 2))
            
            cv2.fillPoly(tooth_mask, [pts_np], 255)
            
            tooth_area = cv2.countNonZero(tooth_mask)
            
            mask_plaque_global = cv2.inRange(hsv_img, lower_plaque, upper_plaque)
            plaque_in_tooth = cv2.bitwise_and(mask_plaque_global, mask_plaque_global, mask=tooth_mask)
            plaque_in_tooth = cv2.morphologyEx(plaque_in_tooth, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            
            plaque_area = cv2.countNonZero(plaque_in_tooth)
            
            percent = (plaque_area / tooth_area * 100) if tooth_area > 0 else 0
            grade_text, color_bgr = self.get_turesky_grade(percent)
            
            cv2.polylines(analysis_img, [pts_np], True, (255, 255, 0), 2)
            contours_plaque, _ = cv2.findContours(plaque_in_tooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(analysis_img, contours_plaque, -1, (0, 0, 255), 1)
            
            M = cv2.moments(pts_np)
            cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else points[0][0]
            cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else points[0][1]
            
            label = f"#{idx+1}: {grade_text} ({percent:.1f}%)"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(analysis_img, (cX - 10, cY - 20), (cX + w_text + 10, cY + 10), (0,0,0), -1)
            cv2.putText(analysis_img, label, (cX - 5, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

        self.final_result_image = analysis_img
        self.show_image_on_canvas(self.final_result_image)
        self.btn_save.config(state=tk.NORMAL)
        messagebox.showinfo("Listo", "Análisis completado respetando selección.")

    def get_turesky_grade(self, percentage):
        if percentage == 0: return "Grado 0", (0, 255, 0)
        elif percentage < 25: return "Grado 1", (0, 255, 255)
        elif percentage < 40: return "Grado 2", (0, 165, 255)
        elif percentage < 60: return "Grado 3", (0, 128, 255)
        elif percentage < 80: return "Grado 4", (0, 0, 255)
        else: return "Grado 5", (0, 0, 139)

    # ------------------------------------------------------------------
    # GESTIÓN IMÁGENES
    # ------------------------------------------------------------------
    def process_selected_image(self, path):
        img = cv2.imread(path)
        if img is None: return
        
        self.original_cv_image = img
        self.resized_cv_image = self.resize_to_fit(img, self.canvas_width, self.canvas_height)
        
        self.reset_selection()

    def resize_to_fit(self, img, max_w, max_h):
        h, w = img.shape[:2]
        scale = min(max_w/w, max_h/h)
        nw, nh = int(w*scale), int(h*scale)
        return cv2.resize(img, (nw, nh))

    def show_image_on_canvas(self, cv_img):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.canvas_draw.delete("all")
        self.canvas_draw.create_image(0, 0, image=img_tk, anchor=tk.NW)
        self.canvas_draw.image = img_tk 

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------
    def select_folder(self):
        p = filedialog.askdirectory()
        if p: self.load_gallery(p)

    def load_gallery(self, folder_path):
        for w in self.scrollable_frame.winfo_children(): w.destroy()
        self.thumbnail_cache = []
        valid = ('.png', '.jpg', '.jpeg')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid)]
        if not files: tk.Label(self.scrollable_frame, text="Vacío", bg="#233140", fg="white").pack()
        for f in files:
            path = os.path.join(folder_path, f)
            try:
                im = Image.open(path)
                im.thumbnail((180, 180))
                ph = ImageTk.PhotoImage(im)
                self.thumbnail_cache.append(ph)
                fr = tk.Frame(self.scrollable_frame, bg="#34495e", pady=2)
                fr.pack(fill=tk.X, pady=2, padx=5)
                tk.Button(fr, image=ph, bg="#2c3e50", bd=0, command=lambda p=path: self.process_selected_image(p)).pack()
            except: pass

    def save_image(self):
        if self.final_result_image is not None:
            f = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")])
            if f: cv2.imwrite(f, self.final_result_image); messagebox.showinfo("Guardado", "Imagen guardada.")

    def sidebar(self):
        if self.sidebar_visible: self.sidebar_frame.pack_forget(); self.sidebar_visible = False
        else: self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, before=self.content_frame); self.sidebar_visible = True

    def _on_mousewheel(self, e):
        if self.sidebar_visible: self.canvas_gallery.yview_scroll(int(-1*(e.delta/120)), "units")

if __name__ == "__main__":
    root = tk.Tk()
    app = DentalAnalyzerApp(root)
    root.mainloop()