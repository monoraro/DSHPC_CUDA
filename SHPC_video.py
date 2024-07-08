"""
Implementación del SHPC de forma serial en python
"""

#Carga de librerias
import cv2
import numpy as np
from numpy import asarray
from Funciones import *
import spicy as sp
import math as mt
import time
import os
import imageio
#Funcion de reconstrucción

def tiro(holo,fx_0,fy_0,fx_tmp, fy_tmp,lamb,M,N,dx,dy,k,m,n):
    
    #Calculo de los angulos de inclinación

    theta_x=mt.asin((fx_0 - fx_tmp) * lamb /(M*dx))
    theta_y=mt.asin((fy_0 - fy_tmp) * lamb /(N*dy))

    #Creación de la fase asociada

    fase= np.exp(1j*k* ((mt.sin(theta_x) * m * dx)+ ((mt.sin(theta_y) * n * dy))))
    fase1=fase
    holo=holo*fase
    
    fase = np.angle(holo, deg=False)
    min_val = np.min(fase)
    max_val = np.max(fase)
    fase = (fase - min_val) / (max_val - min_val)
    threshold_value = 0.2
    fase = np.where(fase > threshold_value, 1, 0)
    value=np.sum(fase)
    return value, fase1

carpeta = './Imagenes/40X'
carpeta2 = 'Imagenes/40X/'
# Obtener la lista de archivos en la carpeta
archivos_en_carpeta = os.listdir(carpeta)

# Ordenar la lista de archivos alfabéticamente
archivos_ordenados = sorted(archivos_en_carpeta)
archivos_ordenados = [archivo for archivo in archivos_ordenados if archivo.endswith('.bmp')]

replica = archivos_ordenados[0]
archivo = carpeta2+str(replica)
replica = lectura(archivo)
U = asarray(replica)
tiempo_inicial = time.time()
N, M = U.shape
#Parametros del montaje

#Trabajemos todo en metros o indica porfa que unidades usas
#Que si no me enredo

# dx y dy como dice el codigo jajaja
dx = 3.75
dy = 3.75
lamb = 0.633
k= 2*np.pi/lamb
Fox= M/2
Foy= N/2
cuadrante=1
# pixeles en el eje x y y de la imagen de origen
x = np.arange(0, M, 1)
y = np.arange(0, N, 1)

#Un meshgrid para la paralelizacion
m, n = np.meshgrid(x - (M/2), y - (N/2))

G=3

#Definiendo cuadrantes, solo calidad
primer_cuadrante= np.zeros((N,M))
primer_cuadrante[0:round(N/2 - (N*0.1)),round(M/2 + (M*0.1)):M]=1
segundo_cuadrante= np.zeros((N,M))
segundo_cuadrante[0:round(N/2 -(N*0.1)),0:round(M/2 - (M*0.1))]=1
tercer_cuadrante= np.zeros((N,M))
tercer_cuadrante[round(N/2 +(N*0.1)):N,0:round(M/2 - (M*0.1))]=1
cuarto_cuadrante= np.zeros((N,M))
cuarto_cuadrante[round(N/2 +(N*0.1)):N,round(M/2 + (M*0.1)):M]=1

#Ahora a tirar fourier

fourier=np.fft.fftshift(sp.fft.fft2(np.fft.fftshift(U)))
if(cuadrante==1):
    fourier=primer_cuadrante*fourier
if(cuadrante==2):
    fourier=segundo_cuadrante*fourier
if(cuadrante==3):
    fourier=tercer_cuadrante*fourier
if(cuadrante==4):
    fourier=cuarto_cuadrante*fourier
a=amplitud(fourier)
#Calculamos la amplitud del espectro de fourier

#Encontramos la posición en x y y del máximo en el espacio de Fourier
pos_max = np.unravel_index(np.argmax(a, axis=None), a.shape)
mascara = crear_mascara_circular(U.shape,(pos_max[1],pos_max[0]),200)
#Transformada insversa de fourier
fourier= fourier*mascara
fourier=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier)))




#Ahora viene definición de parametros 

paso=0.2
fin=0
fx=pos_max[1]
fy=pos_max[0]

G_temp=G
suma_maxima=0


while fin==0:
    temp=0
    frec_esp_x=np.arange(fx-paso*G_temp,fx+paso*G_temp,paso)
    frec_esp_y=np.arange(fy-paso*G_temp,fy+paso*G_temp,paso)
    for i in range(len(frec_esp_y)):
        for j in range(len(frec_esp_x)):
            fx_temp=frec_esp_x[j]
            fy_temp=frec_esp_y[i]
            temp, faserina=tiro(fourier,Fox,Foy,fx_temp,fy_temp,lamb,M,N,dx,dy,k,m,n)
            if(temp>suma_maxima):
                x_max_out = fx_temp
                y_max_out = fy_temp
                suma_maxima = temp
    G_temp = G_temp - 1
    
    if(x_max_out == fx):
        if(y_max_out ==fy):
            fin=1
    fx=x_max_out
    fy=y_max_out

theta_x=mt.asin((Fox - fx) * lamb /(M*dx))
theta_y=mt.asin((Foy - fy) * lamb /(N*dy))
fase= np.exp(1j*k* ((mt.sin(theta_x) * m * dx)+ ((mt.sin(theta_y) * n * dy))))
holo=fourier*fase
fase = np.angle(holo, deg=False)
min_val = np.min(fase)
max_val = np.max(fase)
fase = 255*(fase - min_val) / (max_val - min_val)
video = []
tiempo_final = time.time()
tiempo= tiempo_final-tiempo_inicial
tiempo_frame=[tiempo]
archivos_ordenados.pop(0)
#Loop
G = 1
paso=0.2
#Ahora la versión dinámica
for frame in archivos_ordenados:
    
    archivo = carpeta2+str(frame)

    #Esto es lo que dijo carlos yo creo
    replica = lectura(archivo)
    #Sera que esta se puede paralelizar?

    replica=ajuste_tamano(replica)
    U = asarray(replica)
    fin=0

    tiempo_inicial= time.time()

    fourier=np.fft.fftshift(sp.fft.fft2(np.fft.fftshift(U)))  
    fourier= fourier*mascara
    fourier=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier)))
    
    while fin==0:
        temp=0
        frec_esp_x=np.arange(fx-paso*G_temp,fx+paso*G_temp,paso)
        frec_esp_y=np.arange(fy-paso*G_temp,fy+paso*G_temp,paso)
        for i in range(len(frec_esp_y)):
            for j in range(len(frec_esp_x)):
                fx_temp=frec_esp_x[j]
                fy_temp=frec_esp_y[i]
                temp, faserina=tiro(fourier,Fox,Foy,fx_temp,fy_temp,lamb,M,N,dx,dy,k,m,n)
                if(temp>suma_maxima):
                    x_max_out = fx_temp
                    y_max_out = fy_temp
                    suma_maxima = temp
        G_temp = G_temp - 1
        
        if(x_max_out == fx):
            if(y_max_out ==fy):
                fin=1
        fx=x_max_out
        fy=y_max_out
    theta_x=mt.asin((Fox - fx) * lamb /(M*dx))
    theta_y=mt.asin((Foy - fy) * lamb /(N*dy))
    fase= np.exp(1j*k* ((mt.sin(theta_x) * m * dx)+ ((mt.sin(theta_y) * n * dy))))
    holo=fourier*fase
    fase = np.angle(holo, deg=False)
    min_val = np.min(fase)
    max_val = np.max(fase)
    fase = 255*(fase - min_val) / (max_val - min_val)
    video.append(fase)
    tiempo_final = time.time()
    tiempo= tiempo_final-tiempo_inicial
    tiempo_frame.append(tiempo)

# Guardar el array en un archivo de texto
np.savetxt('tiempos_serial.txt', tiempo_frame, fmt='%f', delimiter='\t')
video = [matriz.astype(np.uint8) for matriz in video]

# Define el formato del video y crea un objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Puedes ajustar el códec según tus necesidades
video_path = 'Video_40x_serial.mp4'  # Puedes cambiar la extensión según el formato deseado (mp4, gif, etc.)
imageio.mimsave(video_path, video, fps=30)
