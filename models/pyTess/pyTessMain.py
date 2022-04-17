import pytesseract


def setup(path):
    # Set the path of tesseract executable
    pytesseract.pytesseract.tesseract_cmd = path


def tesseractExtract(img):
    # Extract the text from the image using pytesseract
    img_text = pytesseract.image_to_string(img)

    # Remove leading spaces in the string
    img_text = img_text.lstrip()

    # Remove trailing spaces in the string
    img_text = img_text.rstrip()

    # Display the image in the terminal
    #print(img_text)

    # Character stitching for text availability check
    no_space = img_text
    no_space = no_space.replace(" ", "")
    no_space = no_space.replace("\f", "")
    no_space = no_space.replace("\n", "")
    no_space = no_space.replace("\t", "")

    # Text availability check
    if no_space == "":
        img_text = "No text Found"

    # Function return
    return img_text
