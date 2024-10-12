import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

Custom = load_model('my_model.h5')
VGG16 = load_model('modelvgg16.h5')

classes = {1: 'Hız sınırı (20km/h)',
           2: 'Hız sınırı (30km/h)',
           3: 'Hız sınırı (50km/h)',
           4: 'Hız sınırı (60km/h)',
           5: 'Hız sınırı (70km/h)',
           6: 'Hız sınırı (80km/h)',
           7: 'Hız sınırı sonu (80km/h)',
           8: 'Hız sınırı (100km/h)',
           9: 'Hız sınırı (120km/h)',
           10: 'Geçiş yok',
           11: '3.5 tondan ağır araçların geçişi yasaktır',
           12: 'Kavşakta geçiş önceliği',
           13: 'Ana yol',
           14: 'Yol ver',
           15: 'Dur',
           16: 'Araç yok',
           17: '3.5 tondan ağır araçların girişi yasaktır',
           18: 'Giriş yok',
           19: 'Genel uyarı',
           20: 'Tehlikeli sol viraj',
           21: 'Tehlikeli sağ viraj',
           22: 'Çift viraj',
           23: 'Engebeli yol',
           24: 'Kaygan yol',
           25: 'Yol sağdan daralıyor',
           26: 'Yol çalışması',
           27: 'Trafik ışıkları',
           28: 'Yayalar',
           29: 'Çocuk geçidi',
           30: 'Bisiklet geçidi',
           31: 'Buz/kara dikkat',
           32: 'Yaban hayvanları geçidi',
           33: 'Hız ve geçiş sınırlamaları sonu',
           34: 'İleri sağa dön',
           35: 'İleri sola dön',
           36: 'Sadece ileri',
           37: 'Düz veya sağa git',
           38: 'Düz veya sola git',
           39: 'Sağa dön',
           40: 'Sola dön',
           41: 'Dönel kavşak zorunluluğu',
           42: 'Geçiş yasağı sonu',
           43: '3.5 tondan ağır araçların geçiş yasağı sonu'}

top = tk.Tk()
top.geometry('800x600')
top.title('Trafik İşareti Sınıflandırma')
top.configure(background='#3c096c')

background_image = Image.open('20.png')

def resize_background(event):
    bg_width = top.winfo_width()
    bg_height = top.winfo_height()
    resized_bg = background_image.resize((bg_width, bg_height), Image.LANCZOS)
    bg = ImageTk.PhotoImage(resized_bg)
    background_label.config(image=bg)
    background_label.image = bg

background_label = tk.Label(top)
background_label.place(relwidth=1, relheight=1)
resize_background(None)
top.bind('<Configure>', resize_background)

label = Label(top, fg="#3c096c", font=('arial', 15, 'bold'))
sign_image = Label(top)
selected_model = tk.StringVar(value='30x30')

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    
    if selected_model.get() == '30x30':
        image = image.resize((30, 30))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        model = Custom
    else:
        image = image.resize((64, 64))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        model = VGG16
    
    pred_probs = model.predict(image)
    pred_class = np.argmax(pred_probs)
    sign = classes[pred_class + 1]
    label.configure(text=sign)
    label.place(relx=0.5, rely=0.8, anchor='center')

def show_classify_button(file_path):
    classify_b = Button(top, text="Görüntüyü Sınıflandırma", command=lambda: classify(file_path), padx=0, pady=5)
    classify_b.configure(background='#3c096c', foreground='#fff', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.50)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        new_size = (int(top.winfo_width() * 0.4), int(top.winfo_height() * 0.4))  
        uploaded = uploaded.resize(new_size, Image.LANCZOS)
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        sign_image.place(relx=0.5, rely=0.55, anchor='center')  
        label.place_forget()  
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Görüntü Yükleme", command=upload_image, padx=10, pady=5)
upload.configure(background='#3c096c', foreground='#fff', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=30)


heading = Label(top, text="Trafik İşaretleri Tanıma", pady=25, width=150, font=('arial', 25, 'bold'))
heading.configure(background='#3c096c', foreground='#fff')
heading.pack()


model_frame = Frame(top, background='#3c096c', pady=6, padx=30)
model_label = Label(model_frame, text="Modeli Tanımlama:", background='#3c096c', fg="#fff", font=('arial', 12, 'bold'))
model_label.pack(side=LEFT)
Custom_button = tk.Radiobutton(model_frame, text="Custom Model", variable=selected_model, value='30x30',
                               background='#3c096c', fg="#fff", font=('arial', 12, 'bold'),
                               activeforeground="#fff", selectcolor="#725ac1")
VGG16_button = tk.Radiobutton(model_frame, text="VGG16 Model", variable=selected_model, value='64x64',
                              background='#3c096c', fg="#fff", font=('arial', 12, 'bold'),
                              activeforeground="#fff", selectcolor="#3c096c")
Custom_button.pack(side=LEFT)
VGG16_button.pack(side=LEFT)
model_frame.pack(side=TOP, pady=20)

top.mainloop()
