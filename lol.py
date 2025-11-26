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
        self.thumbnail_cache = [] 
        self.is_sidebar_visible = True # Estado inicial de la barra lateral

        # --- DISEÑO PRINCIPAL ---
        
        # 1. Marco Superior (Barra de Herramientas)
        top_bar = tk.Frame(root, bg="#34495e", height=60)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        btn_style = {"font": ("Segoe UI", 10, "bold"), "bg": "#2980b9", "fg": "white", "padx": 15, "pady": 5}
        
        # NUEVO: Botón para colapsar/mostrar barra lateral
        self.btn_toggle = tk.Button(top_bar, text="☰", command=self.toggle_sidebar, 
                                    font=("Arial", 14, "bold"), bg="#34495e", fg="white", 
                                    borderwidth=0, activebackground="#2c3e50", activeforeground="white")
        self.btn_toggle.pack(side=tk.LEFT, padx=10)

        tk.Label(top_bar, text="Dental AI", bg="#34495e", fg="white", font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=5)
        
        tk.Button(top_bar, text="📂 Carpeta", command=self.select_folder, **btn_style).pack(side=tk.LEFT, padx=10)
        
        self.btn_save = tk.Button(top_bar, text="💾 Guardar", command=self.save_image, state=tk.DISABLED, **btn_style)
        self.btn_save.pack(side=tk.RIGHT, padx=20, pady=10)

        # 2. Contenedor Principal
        self.main_container = tk.Frame(root, bg="#2c3e50")
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # --- PANEL IZQUIERDO: GALERÍA (Lo guardamos en self para poder manipularlo) ---
        self.sidebar_width = 260
        self.sidebar_frame = tk.Frame(self.main_container, width=self.sidebar_width, bg="#233140")
        
        # Empaquetamos inicialmente
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar_frame.pack_propagate(False)

        # Contenido de la barra lateral
        tk.Label(self.sidebar_frame, text="Pacientes", bg="#233140", fg="#bdc3c7", font=("Arial", 10, "bold")).pack(pady=10)

        self.canvas_gallery = Canvas(self.sidebar_frame, bg="#233140", highlightthickness=0)
        self.scrollbar_gallery = Scrollbar(self.sidebar_frame, orient="vertical", command=self.canvas_gallery.yview)
        self.scrollable_frame = Frame(self.canvas_gallery, bg="#233140")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas_gallery.configure(scrollregion=self.canvas_gallery.bbox("all"))
        )

        self.canvas_gallery.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=self.sidebar_width-20)
        self.canvas_gallery.configure(yscrollcommand=self.scrollbar_gallery.set)

        self.canvas_gallery.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.scrollbar_gallery.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_gallery.bind_all("<MouseWheel>", self._on_mousewheel)

        # --- PANEL DERECHO: VISUALIZACIÓN ---
        # Guardamos en self.content_frame
        self.content_frame = tk.Frame(self.main_container, bg="#2c3e50")
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.lbl_original = tk.Label(self.content_frame, text="Selecciona imagen", bg="black", fg="white")
        self.lbl_original.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)

        self.lbl_binary = tk.Label(self.content_frame, text="Análisis", bg="black", fg="white")
        self.lbl_binary.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5)

    # --- NUEVA FUNCION: Alternar Barra Lateral ---
    def toggle_sidebar(self):
        if self.is_sidebar_visible:
            # Ocultar
            self.sidebar_frame.pack_forget()
            self.btn_toggle.config(bg="#233140") # Indicador visual (opcional)
            self.is_sidebar_visible = False
        else:
            # Mostrar (Importante: 'before' asegura que aparezca a la izquierda del contenido)
            self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, before=self.content_frame)
            self.btn_toggle.config(bg="#34495e")
            self.is_sidebar_visible = True

    def _on_mousewheel(self, event):
        if self.is_sidebar_visible: # Solo hacer scroll si es visible
            self.canvas_gallery.yview_scroll(int(-1*(event.delta/120)), "units")

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path: return
        self.load_gallery(folder_path)

    def load_gallery(self, folder_path):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.thumbnail_cache = []

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        
        if not files:
            tk.Label(self.scrollable_frame, text="Vacío", bg="#233140", fg="white").pack(pady=20)
            return

        for filename in files:
            full_path = os.path.join(folder_path, filename)
            try:
                pil_img = Image.open(full_path)
                pil_img.thumbnail((200, 200))
                tk_thumb = ImageTk.PhotoImage(pil_img)
                self.thumbnail_cache.append(tk_thumb)

                card = tk.Frame(self.scrollable_frame, bg="#34495e", pady=2, padx=2)
                card.pack(fill=tk.X, pady=4, padx=5)

                btn = tk.Button(card, image=tk_thumb, bg="#2c3e50", borderwidth=0,
                                command=lambda p=full_path: self.process_selected_image(p))
                btn.pack()
                
                # Nombre de archivo recortado
                name_short = (filename[:15] + '..') if len(filename) > 15 else filename
                tk.Label(card, text=name_short, bg="#34495e", fg="white", font=("Arial", 8)).pack(fill=tk.X)
                
            except Exception: pass

    def process_selected_image(self, path):
        img = cv2.imread(path)
        if img is None: return

        self.btn_save.config(state=tk.DISABLED)
        # Procesamiento
        self.original_cv_image = self.resize_with_padding(img, target_size=(640,480))
        self.create_tooth_mask(self.original_cv_image)

    # --- LÓGICA DE PROCESAMIENTO (Igual que antes) ---
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

    def create_tooth_mask(self, img):
        blurred = cv2.GaussianBlur(img, (5,5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # Ajustar rangos según necesidad
        mask_combined = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,0,60]), np.array([180,90,255])),
            cv2.inRange(hsv, np.array([20,50,50]), np.array([40,255,255]))
        )
        kernel = np.ones((9,9), np.uint8)
        binary_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=3)
        
        final_mask = np.zeros_like(binary_clean)
        contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 5000:
                cv2.drawContours(final_mask, [largest], -1, 255, -1)

        self.binary_mask = final_mask
        
        # Mostrar Original
        self.show_image(self.draw_grid(self.original_cv_image), self.lbl_original)

        # Placa
        tooth_cutout = cv2.bitwise_and(self.original_cv_image, self.original_cv_image, mask=self.binary_mask)
        hsv_p = cv2.cvtColor(tooth_cutout, cv2.COLOR_BGR2HSV)
        mask_p = cv2.inRange(hsv_p, np.array([20,50,80]), np.array([35,255,220]))
        cnts_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(tooth_cutout, cnts_p, -1, (0,255,255), 2)
        
        self.processed_image = self.draw_grid(tooth_cutout)
        self.show_image(self.processed_image, self.lbl_binary)
        self.btn_save.config(state=tk.NORMAL)

    def show_image(self, cv_img, label_widget):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        label_widget.config(image=img_tk, text="")
        label_widget.image = img_tk

    def save_image(self):
        if self.processed_image is None: return
        f = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")])
        if f: cv2.imwrite(f, self.processed_image); messagebox.showinfo("Listo", "Guardado")

if __name__ == "__main__":
    root = tk.Tk()
    app = DentalAnalyzerApp(root)
    root.mainloop()