from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
import cv2
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from sface import SimpleFacerec

sfr=SimpleFacerec()
sfr.load_encoding_images("img/")


class MainApp(MDApp):
    def build(self):
        layout=MDBoxLayout(orientation='vertical')
        self.image=Image()
        layout.add_widget(self.image)
        layout.add_widget(MDRaisedButton(
            text="click",
            pos_hint={'center_x':.5,'center_y':.5},
            size_hint=(None,None))
            )
        self.cap=cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video,1.0/60.0)
        return layout
    
    def load_video(self, *args):
        ret,frame=self.cap.read()
        face_location,face_name = sfr.detect_known_faces(frame)
        for face_loc,name in zip(face_location,face_name):
            y1,x2,y2,x1=face_loc[0],face_loc[1],face_loc[2],face_loc[3]

            cv2.putText(frame,name,(x1,y1-10),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,200),4)
        self.image_frame=frame
        buffer=cv2.flip(frame,0).tostring()
        texture=Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
        texture.blit_buffer(buffer,colorfmt='bgr',bufferfmt='ubyte')
        self.image.texture=texture


if __name__=='__main__':
    MainApp().run()  
     