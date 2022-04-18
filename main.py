from PIL import Image
from io import BytesIO
from models.pyTess import pyTessMain
from models.u2net import u2Main

from flask import Flask, request, send_file

app = Flask(__name__)


@app.route('/api/textEx')
def TextExtract():
    img = Image.open(r"./images/incImg/temp.png")

    imgText = pyTessMain.tesseractExtract(img)

    return imgText


@app.route('/objectEx', methods=['GET', 'POST'])
def ObjectExtract():
    if request.method == 'GET':
        print("GAY")
        return "GAY"
    else:
        test = request.files.get('image','')
        newImg = Image.open(test)
        img = Image.open(r"./images/IMGin/obj4.png")

        result = u2Main.remove(newImg, model_name="u2net", alpha_matting=True)
        resultImg = Image.open(BytesIO(result)).convert("RGBA")
        #resultImg.show()
        resultImg.save(r"./images/IMGout/result2.png")

        print("PENIS ",send_file(resultImg, mimetype='image/*') )
        return send_file(resultImg, mimetype='image/*')
