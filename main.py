import cv2
import numpy as np
from sklearn.cluster import KMeans
def image_processing(image_path):
# Завантажуємо зображення
image = cv2.imread(image_path)
original = image.copy()
# Кольорова корекція (підсилення яскравості та контрасту)
corrected_image = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
cv2.imshow('Color Corrected Image', corrected_image) # Вивід кольорово
скоригованого зображення
# Конвертація в простір HSV для подальшої обробки
hsv = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2HSV)
# Кластеризація кольорів на зображенні
pixel_values = hsv.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
k = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10,
cv2.KMEANS_RANDOM_CENTERS)
# Створення зображення з сегментованими кольорами
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(hsv.shape)
segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_HSV2BGR)
cv2.imshow('Segmented Image', segmented_image) # Вивід сегментованого
зображення
# Виявлення контурів
gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cv2.imshow('Threshold Image', thresh) # Вивід бінарного зображення для
контурів
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
# Малювання контурів і додавання тексту
for contour in contours:
cv2.drawContours(original, [contour], -1, (0, 255, 0), 2)
for contour in contours:
x, y, w, h = cv2.boundingRect(contour)
if cv2.contourArea(contour) > 800:
cv2.putText(original, "Forest", (x, y - 10),
cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
# Вивід кінцевого результату
cv2.imshow('Final Image with Contours', original)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Шлях до зображення
image_path = 'image.png'
# Обробка зображення
image_processing(image_path)