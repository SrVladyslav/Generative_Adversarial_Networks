import PySimpleGUI as sg
from GAN_generator import *
import cv2

# GUI
sg.theme('DarkAmber')   
layout = [  [sg.Text('Si, eso es GUI de Python, no me pegueis!')],
            [sg.Image(filename= './output/new.png', key="-IMAGE-", visible=True)],
            [sg.Button('Ok'), sg.Button('Cancel')],
            #[sg.Slider(range=(-2.0,2.0), resolution=.1, default_value=0.0, size=(20,15), enable_events=True, orientation='h',key="-S1-", font=('Helvetica', 12))]]
            [sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S1-", font=('Helvetica', 12)),
            sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S2-", font=('Helvetica', 12)),
            sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S3-", font=('Helvetica', 12)),
            sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S4-", font=('Helvetica', 12)),
            sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S5-", font=('Helvetica', 12)),
            sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S6-", font=('Helvetica', 12)),
            sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S7-", font=('Helvetica', 12)),
            sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S8-", font=('Helvetica', 12)),
            sg.Slider(range=(-10,10), default_value=0, size=(15,15), enable_events=True, orientation='v',key="-S9-", font=('Helvetica', 12))]
        
        ]

# Create the Window
window = sg.Window('Python GUI = ?', layout)

# Model init
model = GAN_generator()
model
array = [1,1,1,1,1,1,1,1,1]

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    
    # Scrollers listeners
    if event == "-S1-":
        v = values["-S1-"]
        array[0] = v
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/new.png')
        
    elif values['-S2-']:
        v = values["-S2-"]
        array[1] = v
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/new.png')
    
    elif values['-S3-']:
        v = values["-S3-"]
        array[2] = v 
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/new.png')
    
    elif values['-S4-']:
        v = values["-S4-"]
        array[3] = v 
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/new.png')
    
    elif values['-S5-']:
        v = values["-S5-"]
        array[4] = v 
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/new.png')
    
    elif values['-S6-']:
        v = values["-S6-"]
        array[5] = v
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/new.png')
    
    elif values['-S7-']:
        v = values["-S7-"]
        array[6] = v 
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/new.png')

    elif values['-S8-']:
        v = values["-S8-"]
        array[7] = v 
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/new.png')
    
    elif values['-S9-']:
        v = values["-S9-"]
        array[8] = v 
        model.predictIMG(array)
        window["-IMAGE-"].update(filename= './output/new.png')
    
window.close()