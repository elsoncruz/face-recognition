import cv2
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.app import MDApp
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock

thres = 0.45 
nms_threshold = 0.2
       
classNames = []
classFile = 'kivyface/coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


configPath = 'kivyface/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'kivyface/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

class MainApp(MDApp):
    def build(self):
        layout=MDBoxLayout(orientation='vertical')
        self.image=Image()
        layout.add_widget(self.image)
        self.cap = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video,1.0/30.0)
        return layout

    def load_video(self, *args):
        success,img=self.cap.read()
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
        bbox = list(bbox)


        for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box, color=(0, 255, 0), thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

            self.image_frame=img
            buffer=cv2.flip(img,0).tostring()
            texture=Texture.create(size=(img.shape[1],img.shape[0]),colorfmt='bgr')
            texture.blit_buffer(buffer,colorfmt='bgr',bufferfmt='ubyte')
            self.image.texture=texture

if __name__=='__main__':
    MainApp().run()  