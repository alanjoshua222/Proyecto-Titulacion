import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, Scrollbar, Frame
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

class DentalAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Estación de Análisis Dental")
        self.root.geometry("1400x750") 
        self.root.configure(bg="#2c3e50")

        # Variables de estado
        self.original_cv_image = None
        self.processed_image = None
        self.binary_mask = None
        self.thumbnail_cache = [] # Para evitar que el recolector de basura borre las iconos

        # --- DISEÑO PRINCIPAL (LAYOUT) ---
        # 1. Marco Superior (Barra de Herramientas)
        top_bar = tk.Frame(root, bg="#34495e", height=60)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        btn_style = {"font": ("Segoe UI", 10, "bold"), "bg": "#2980b9", "fg": "white", "padx": 15, "pady": 5}
        
        tk.Label(top_bar, text="🦷 Dental AI Analyzer", bg="#34495e", fg="white", font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=20)
        
        # Botón para seleccionar carpeta
        tk.Button(top_bar, text="📂 Seleccionar Carpeta", command=self.select_folder, **btn_style).pack(side=tk.LEFT, padx=10)
        
        tk.Button(top_bar, text="📂 Cargar Imagen", command=self.load_image, **btn_style).pack(side=tk.LEFT, padx=20)

        # Botón de Guardar
        self.btn_save = tk.Button(top_bar, text="💾 Descargar Resultado", command=self.save_image, state=tk.DISABLED, **btn_style)
        self.btn_save.pack(side=tk.RIGHT, padx=20, pady=10)

        # 2. Contenedor Principal (Divide Izquierda y Derecha)
        main_container = tk.Frame(root, bg="#2c3e50")
        main_container.pack(fill=tk.BOTH, expand=True)

        # --- PANEL IZQUIERDO: GALERÍA ---
        self.sidebar_width = 280
        sidebar_frame = tk.Frame(main_container, width=self.sidebar_width, bg="#233140")
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        sidebar_frame.pack_propagate(False) # Evita que el frame se encoja

        # Título de la galería
        tk.Label(sidebar_frame, text="Galería de Pacientes", bg="#233140", fg="#bdc3c7").pack(pady=5)

        # Configuración del Scroll para la Galería
        self.canvas_gallery = Canvas(sidebar_frame, bg="#233140", highlightthickness=0)
        self.scrollbar_gallery = Scrollbar(sidebar_frame, orient="vertical", command=self.canvas_gallery.yview)
        self.scrollable_frame = Frame(self.canvas_gallery, bg="#233140")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas_gallery.configure(scrollregion=self.canvas_gallery.bbox("all"))
        )

        self.canvas_gallery.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=self.sidebar_width-20)
        self.canvas_gallery.configure(yscrollcommand=self.scrollbar_gallery.set)

        self.canvas_gallery.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.scrollbar_gallery.pack(side=tk.RIGHT, fill=tk.Y)

        # Habilitar scroll con rueda del ratón
        self.canvas_gallery.bind_all("<MouseWheel>", self._on_mousewheel)

        # --- PANEL DERECHO: VISUALIZACIÓN ---
        content_frame = tk.Frame(main_container, bg="#2c3e50")
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Labels de imágenes
        self.lbl_original = tk.Label(content_frame, text="Selecciona una imagen...", bg="black", fg="white")
        self.lbl_original.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)

        self.lbl_binary = tk.Label(content_frame, text="Análisis", bg="black", fg="white")
        self.lbl_binary.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5)


    def _on_mousewheel(self, event):
        self.canvas_gallery.yview_scroll(int(-1*(event.delta/120)), "units")

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path: return
        
        self.load_gallery(folder_path)

    def load_gallery(self, folder_path):
        # Limpiar galería anterior
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.thumbnail_cache = []

        # Buscar imágenes válidas
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        
        if not files:
            tk.Label(self.scrollable_frame, text="No se encontraron imágenes", bg="#233140", fg="white").pack(pady=20)
            return

        # Crear miniaturas
        for filename in files:
            full_path = os.path.join(folder_path, filename)
            
            try:
                # Cargar imagen solo para thumbnail (usando PIL para rapidez)
                pil_img = Image.open(full_path)
                pil_img.thumbnail((200, 200)) # Tamaño de miniatura
                
                tk_thumb = ImageTk.PhotoImage(pil_img)
                self.thumbnail_cache.append(tk_thumb) # Guardar referencia

                # Crear un Frame para cada item (Tarjeta)
                card = tk.Frame(self.scrollable_frame, bg="#34495e", pady=5, padx=5)
                card.pack(fill=tk.X, pady=5, padx=5)

                # Botón con la imagen (al hacer click llama a process_selected_image)
                btn = tk.Button(card, image=tk_thumb, bg="#2c3e50", borderwidth=0,
                                command=lambda p=full_path: self.process_selected_image(p))
                btn.pack()

                # Etiqueta con el nombre
                lbl = tk.Label(card, text=filename[:20], bg="#34495e", fg="white", font=("Arial", 8))
                lbl.pack(fill=tk.X)
                
            except Exception as e:
                print(f"Error cargando miniatura {filename}: {e}")

    def process_selected_image(self, path):
        """Esta función se llama cuando haces click en una miniatura"""
        print(f"Procesando: {path}")
        
        img = cv2.imread(path)
        if img is None: return

        self.btn_save.config(state=tk.DISABLED)
        
        # Reutilizamos tu lógica de procesamiento
        self.original_cv_image = self.resize_with_padding(img, target_size=(640,480))
        self.create_tooth_mask(self.original_cv_image)

    # --- TUS FUNCIONES DE PROCESAMIENTO EXISTENTES (Sin cambios mayores) ---

    def resize_with_padding(self, img, target_size=(640, 480)):
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset, y_offset = (target_w - new_w) // 2, (target_h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return canvas

    def draw_grid(self, img, grid_size=40, color=(200,200,200)):
        img_grid = img.copy()
        h, w = img_grid.shape[:2]
        for x in range(0, w, grid_size): cv2.line(img_grid, (x,0), (x,h), color, 1)
        for y in range(0, h, grid_size): cv2.line(img_grid, (0,y), (w,y), color, 1)
        return img_grid

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path: return

        img = cv2.imread(path)
        if img is None: return

        # Reiniciar estado al cargar nueva imagen
        self.btn_save.config(state=tk.DISABLED)
        
        self.original_cv_image = self.resize_with_padding(img, target_size=(640,480))
        self.create_tooth_mask(self.original_cv_image)

    def create_tooth_mask(self, img):
        # ... (Tu lógica de procesamiento de imagen se mantiene igual) ...
        blurred = cv2.GaussianBlur(img, (5,5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0,0,60])
        upper_white = np.array([180,90,255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        lower_yellow = np.array([20,50,50])
        upper_yellow = np.array([40,255,255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_combined = cv2.bitwise_or(mask_white, mask_yellow)

        kernel = np.ones((9,9), np.uint8)
        binary_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel, iterations=3)

        contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(binary_clean)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 5000:
                cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        self.binary_mask = final_mask

        # Mostrar Original
        original_grid = self.draw_grid(self.original_cv_image, color=(150, 150, 150))
        self.show_image(original_grid, self.lbl_original)

        # Proceso final
        tooth_cutout_grid = cv2.bitwise_and(self.original_cv_image, self.original_cv_image, mask=self.binary_mask)
        hsv_cutout = cv2.cvtColor(tooth_cutout_grid, cv2.COLOR_BGR2HSV)
        
        # Detección de placa
        lower_plaque = np.array([20,50,80])
        upper_plaque = np.array([35,255,220])
        mask_plaque = cv2.inRange(hsv_cutout, lower_plaque, upper_plaque)
        contours_plaque, _ = cv2.findContours(mask_plaque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_plaque_area = sum(cv2.contourArea(c) for c in contours_plaque)

        cv2.drawContours(tooth_cutout_grid, contours_plaque, -1, (0,255,255), 2)
        result_grid = self.draw_grid(tooth_cutout_grid, color=(150,150,150))
        
        self.processed_image = result_grid
        self.show_image(result_grid, self.lbl_binary)
        self.btn_save.config(state=tk.NORMAL)

    def show_image(self, cv_img, label_widget, is_gray=False):
        if is_gray: img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else: img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        label_widget.config(image=img_tk, text="")
        label_widget.image = img_tk

    def save_image(self):
        if self.processed_image is None: return
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            messagebox.showinfo("Éxito", "Imagen guardada")

if __name__ == "__main__":
    root = tk.Tk()
    app = DentalAnalyzerApp(root)
    root.mainloop()