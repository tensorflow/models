from skimage import io, transform, util
import inspect

img = io.imread('dog-miko-01.jpg')

def crop_square(img, size=512):
    print(img.shape)
    start = [0, 0]
    size_pix = 0
    h = img.shape[0]
    w = img.shape[1]
    if w < h:
        # portrait
        print('portrait')
        start[0] = (h - w) / 2
        size_pix = w
    else:
        # landscpae
        print('landscape')
        start[1] = (w - h) /2
        size_pix = h

    print(start, size_pix)


    square_img = img[start[0]:(start[0]+size_pix), start[1]:(start[1]+size_pix)]
    #print(square_img.shape)
    return transform.resize(square_img, (size, size))

tiny = crop_square(img, 32)

io.imshow(tiny)
io.show()
