from PIL import Image
from io import BytesIO
from models.pyTess import pyTessMain
from models.u2net import u2Main
import base64
from flask import jsonify
import os
import pytesseract
import cv2


from flask import Flask, request, send_file

app = Flask(__name__)


@app.route('/api/textEx')
def TextExtract():
    img = Image.open(r"./images/incImg/temp.png")

    # imgText = pyTessMain.tesseractExtract(img)

    return imgText


@app.route('/expoObjectEx', methods=['GET', 'POST'])
def ObjectExtract():
    if request.method == 'GET':
        return "You can start"
    else:
        print("Image request ====================")
        bytesOfImage = request.get_data()
        with open('image.jpeg', 'wb') as out:
            out.write(bytesOfImage)
        img = Image.open("./image.jpeg")

        result = u2Main.remove(img, model_name="u2net", alpha_matting=True)
        resultImg = Image.open(BytesIO(result)).convert("RGBA")
        # resultImg.show()

        base64Value = get_base64(resultImg)

        return jsonify({"imgUri64": base64Value})


@app.route('/expoTextEx', methods=['GET', 'POST'])
def extract_text():
    bytesOfImage = request.get_data()
    with open('image.jpeg', 'wb') as out:
        out.write(bytesOfImage)
    im = cv2.imread("./image.jpeg")
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(imgray, 180, 255, cv2.THRESH_BINARY)
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, thresh1)
    img = Image.open(filename)
    text = pytesseract.image_to_string(img)
    return text


def get_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return "data:image/png;base64," + img_str.decode()


if __name__ == "__main__":
    app.run()
