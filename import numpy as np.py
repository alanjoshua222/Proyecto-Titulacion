import numpy as np
import cv2 as cv
import urllib.request
import matplotlib.pyplot as plt

url = "https://i.sstatic.net/oXy5T.jpg"
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
req = urllib.request.Request(url = "https://i.sstatic.net/oXy5T.jpg", headers={'User-Agent': user_agent})
array = np.asarray(bytearray(req.read()), dtype=np.uint8)
img_monedas = cv.imdecode(array, -1,)
img_monedas = cv.cvtColor(img_monedas, cv.COLOR_BGR2RGB)
plt.imshow(img_monedas)
monedas_blur = cv.medianBlur(img_monedas, 35)
plt.imshow(monedas_blur)
monedas_gris = cv.cvtColor(monedas_blur, cv.COLOR_BGR2GRAY)
ret, monedas_umbral = cv.threshold(monedas_gris,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
plt.imshow(monedas_umbral, cmap="gray")
monedas_fondo = cv.dilate(monedas_umbral, kernel,iterations=3)
distancia_transform = cv.distanceTransform(monedas_fondo, cv.DIST_L2,5)
ret, no_fondo = cv.threshold(distancia_transform,0.7*distancia_transform.max(),255,0)
ret, no_fondo = cv.threshold(distancia_transform,0.7*distancia_transform.max(),255,0)
plt.imshow(distancia_transform, cmap="gray")
no_fondo = np.uint8(no_fondo)
region_desconocida = cv.subtract(monedas_fondo,no_fondo)
plt.imshow(region_desconocida, cmap="gray")[["introducir la descripción de la imagen aquí"][5]][5]
ret, marcadores = cv.connectedComponents(no_fondo)
# Marcamos los fondos con 1
marcadores = marcadores+1
# Marcamos las regiones deconocidas con 0
marcadores[region_desconocida==255] = 0
monedas_watersheed = cv.watershed(monedas_blur, marcadores)
plt.imshow(monedas_watersheed)
#Por último le podemos dibujar el contorno para diferenciarlo claramente en la imagen inicial
contornos, jerarquía = cv.findContours(marcadores.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

for i in range(len(contornos)):
    if jerarquía[0][i][3] == -1:
        cv.drawContours(img_monedas, contornos, i, (0, 0, 255), 10)

plt.imshow(img_monedas)