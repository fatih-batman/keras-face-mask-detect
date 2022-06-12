import tensorflow.keras
from PIL import Image, ImageOps
import numpy as numpy

numpy.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('keras_model.h5')
data = numpy.ndarray(shape=(1, 224, 224, 3), dtype=numpy.float32)
image = Image.open('maskesiz_2.jpg')
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)
image_array = numpy.asarray(image)
image.show()
normalized_image_array = (image_array.astype(numpy.float32) / 127.0) - 1
data[0] = normalized_image_array
prediction = model.predict(data)
print(prediction)
if(prediction[0][0] > prediction[0][1] ): print("Hesaplamalara göre birey: Maskeli")
else: print("Hesaplamalara göre birey: Maskesiz")
