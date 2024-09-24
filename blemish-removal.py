import cv2
import numpy as np

source = cv2.imread('blemish.png')
r =15

def sobelfilter(crop_img):
    sobel64x = cv2.Sobel(crop_img, cv2.CV_64F, 1,0,ksize=3)
    sobelabsx = np.abs(sobel64x)
    sobel8x = np.uint8(sobelabsx)
    grad_x = np.mean(sobel8x)

    sobel64y = cv2.Sobel(crop_img, cv2.CV_64F, 0,1,ksize=3)
    sobelabsy = np.abs(sobel64y)
    sobel8y = np.uint8(sobelabsy)
    grad_y = np.mean(sobel8y)

    return grad_x, grad_y


def appendDict(x,y):
    crop_img = source[y:(y+2*r), x:(x+2*r)]
    grad_x, grad_y =  sobelfilter(crop_img)
    return grad_x, grad_y


def selectedBlemish(x,y,r):
    crop_img = source[y:(y+2*r), x:(x+2*r)]
    return identifybestPatch(x,y,r)


def identifybestPatch(x,y,r):
    patches = {}

    key1tup = appendDict(x+2*r,y)
    patches['k1'] = (x+2*r, y, key1tup[0], key1tup[1])

    key2tup = appendDict(x+2*r,y+r)
    patches['k2'] = (x+2*r, y+r, key2tup[0], key2tup[1])

    key3tup = appendDict(x-2*r,y)
    patches['k3'] = (x-2*r, y, key3tup[0], key3tup[1])

    key4tup = appendDict(x-2*r,y-r)
    patches['k4'] = (x-2*r, y-r, key4tup[0], key4tup[1])

    key5tup = appendDict(x,y+2*r)
    patches['k5'] = (x, y+2*r, key5tup[0], key5tup[1])

    key6tup = appendDict(x+r,y+2*r)
    patches['k6'] = (x+r, y+2*r, key6tup[0], key6tup[1])

    key7tup = appendDict(x,y-2*r)
    patches['k7'] = (x, y-2*r, key7tup[0], key7tup[1])

    key8tup = appendDict(x-r,y-r*2)
    patches['k8'] = (x-r, y-r*2, key8tup[0], key8tup[1])

    findlowx = {}
    findlowy = {}
    for key, (x,y, gx, gy) in patches.items():
        findlowx[key] = gx
    for key, (x,y, gx, gy) in patches.items():
        findlowy[key] = gy

    y_key_min = min(findlowy.keys(), key = (lambda k: findlowy[k]))
    x_key_min = min(findlowx.keys(), key = (lambda k: findlowx[k]))

    if x_key_min == y_key_min:
        return patches[x_key_min][0], patches[x_key_min][1]
    else:
        return patches[x_key_min][0], patches[x_key_min][1]
    
    
def bleamish_remv(event, x,y,flags, param):
    global r, source
    if event == cv2.EVENT_FLAG_LBUTTON:
        nx, ny = selectedBlemish(x, y, r)
        
        new_test = source[ny:ny+2*r, nx:nx+2*r]
        mask = 255*np.ones(new_test.shape, new_test.dtype)
        source = cv2.seamlessClone(new_test, source, mask, (x,y), cv2.NORMAL_CLONE)
        cv2.imshow('image',source)


#img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', bleamish_remv)

while(1):
    #img2 = source.copy()
    cv2.imshow('image', source)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()