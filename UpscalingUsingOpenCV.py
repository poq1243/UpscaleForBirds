import cv2
import matplotlib.pyplot as plt
# Read image
img = cv2.imread("Bird4_small.jpeg")
plt.imshow(img[:, :, ::-1])
plt.show()

sr = cv2.dnn_superres.DnnSuperResImpl_create()

path = "LapSRN_x8.pb"

sr.readModel(path)

sr.setModel("lapsrn", 8)

result = sr.upsample(img)

plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
# Original image
plt.imshow(img[:, :, ::-1])
plt.subplot(1, 2, 2)
# SR upscaled
plt.imshow(result[:, :, ::-1])
plt.show()
