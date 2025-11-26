import dearpygui.dearpygui as dpg

class MyApp:
    def __init__(self):
        self.counter = 0
        
    def setup(self):
        # Создание контекста
        dpg.create_context()
        
        # Создание viewport
        dpg.create_viewport(
            title='My DearPyGUI App',
            width=800,
            height=600
        )
        
        self.create_windows()
        
    def create_windows(self):
        # Главное окно
        with dpg.window(label="Main Window", width=400, height=300):
            dpg.add_text("Hello, DearPyGUI!")
            dpg.add_button(
                label="Click me!",
                callback=self.button_callback
            )
            dpg.add_input_text(
                label="Text input",
                callback=self.text_callback
            )
            self.counter_text = dpg.add_text(f"Counter: {self.counter}")
            
        # Второе окно
        with dpg.window(label="Settings", pos=(410, 0), width=390, height=300):
            dpg.add_slider_float(
                label="Scale",
                default_value=1.0,
                callback=self.slider_callback
            )
    
    def button_callback(self):
        self.counter += 1
        dpg.set_value(self.counter_text, f"Counter: {self.counter}")
    
    def text_callback(self, sender, app_data):
        print(f"Text changed: {app_data}")
    
    def slider_callback(self, sender, app_data):
        print(f"Scale changed: {app_data}")
    
    def run(self):
        self.setup()
        
        # Показ viewport и запуск
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        
        # Очистка
        dpg.destroy_context()

if __name__ == "__main__":
    app = MyApp()
    app.run()