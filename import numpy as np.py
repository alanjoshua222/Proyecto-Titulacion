from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.video import Video
from plyer import filechooser
from kivy.utils import get_color_from_hex

class MoviePlayerApp(App):
    def build(self):
        # Usamos un layout vertical
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # BOTÓN: Tamaño fijo (100 píxeles de alto) para que no se pierda
        self.btn = Button(
            text="SELECCIONAR ARCHIVO .MOV",
            size_hint=(1, None),
            height=100,
            background_color=get_color_from_hex('#2980b9'),
            color=(1, 1, 1, 1),
            font_size='20sp'
        )
        self.btn.bind(on_release=self.open_file_manager)

        # REPRODUCTOR: Ocupará el resto del espacio disponible
        self.video_player = Video(
            source='', 
            state='stop', 
            options={'eos': 'loop'} # Para que se repita si es pesado
        )
        
        # Primero agregamos el botón y luego el video
        self.layout.add_widget(self.btn)
        self.layout.add_widget(self.video_player)
        
        return self.layout

    def open_file_manager(self, *args):
        # Esto abre la ventana de Windows/Mac/Linux
        path = filechooser.open_file(
            title="Selecciona un video .mov pesado",
            filters=[("QuickTime Files", "*.mov")]
        )
        
        if path:
            print(f"Archivo seleccionado: {path[0]}")
            self.video_player.source = path[0]
            self.video_player.state = 'play'

if __name__ == '__main__':
    MoviePlayerApp().run()