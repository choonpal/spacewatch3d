import matplotlib.pyplot as plt
from PIL import Image

IMAGE_PATH = "data/test_images/sample1.png"

image = Image.open(IMAGE_PATH)

fig, ax = plt.subplots()
ax.imshow(image)
ax.set_title("Click image - coordinate will be printed")

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)

        print(f"Clicked: x={x}, y={y}")

        ax.plot(x, y, "ro")
        fig.canvas.draw()

fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()