from PIL import Image
from io import BytesIO
from models.pyTess import pyTessMain
from models.u2net import u2Main

imgOutPath = r"./images/IMGout/"

imgName = "text.png"
imgPath = r"./images/IMGin/" + imgName

img2Name = "obj4.png"
img2Path = r"./images/IMGin/" + img2Name

tesseractPath = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pyTessMain.setup(tesseractPath)

img = Image.open(imgPath)
img2 = Image.open(img2Path)

imgText = pyTessMain.tesseractExtract(img)

result = u2Main.remove(img2, model_name="u2net", alpha_matting=True)
resultImg = Image.open(BytesIO(result)).convert("RGBA")
resultImg.save(imgOutPath + "result.png")

resultImg.show()
