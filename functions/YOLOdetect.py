from ultralytics import YOLO

def detect(img_root):
    model = YOLO(r'./test_on_1lure.pt')
    model.predict(source=img_root, save=True, save_txt=True, save_conf=True, name=r"D:\Filea\miceie\projects\Stereo_vision_new\label_norm")

if __name__  == '__main__':
    detect(r"C:\Users\delao\Desktop\New folder\Image__2023-03-14__19-16-50.bmp")
    