import numpy as np
import caffe
import cv2
import glob
 
class AgenDetector:

    age_net = None
    apparent_age_net = None
    loaded = False

    @staticmethod
    def init(age_model_file = "age.prototxt",
            age_pretrained = "dex_imdb_wiki.caffemodel",
            apparent_age_model_file = "age2.prototxt",
            apparent_age_pretrained = "dex_chalearn_iccv2015.caffemodel",
            mean_file = "ilsvrc_2012_mean.npy",
            gpu_mode = False):
        if not AgenDetector.loaded:
            AgenDetector.clean()

            #Load face detector model

            if not gpu_mode:
                caffe.set_mode_cpu()
            else:
                caffe.set_mode_gpu()

            #Load age prediction network model        
            AgenDetector.age_net = caffe.Classifier(age_model_file, age_pretrained,
                mean=np.load(mean_file).mean(1).mean(1),
                channel_swap=(2,1,0),
                raw_scale=256,
                image_dims=(224, 224))

            AgenDetector.apparent_age_net = caffe.Classifier(apparent_age_model_file, apparent_age_pretrained,
                mean=np.load(mean_file).mean(1).mean(1),
                channel_swap=(2,1,0),
                raw_scale=256,
                image_dims=(224, 224))

            AgenDetector.loaded = True

    @staticmethod
    def clean():
        AgenDetector.age_net = None
        AgenDetector.age_apparent_net = None
        AgenDetector.loaded = False

    def predict_one_face(self, input_image):
        age_prediction = AgenDetector.age_net.predict([input_image])

        age_apparent_prediction = AgenDetector.apparent_age_net.predict([input_image])

        return age_prediction[0].argmax(), age_apparent_prediction[0].argmax()

    def predict(self, image_path):
        img = caffe.io.load_image(image_path)
        return self.predict_one_face(img)


if __name__ == '__main__':

    AgenDetector.init()

    predictor = AgenDetector()

    for img in glob.glob("images/cropped_face/*.jpg"):
        results = predictor.predict(img)
        print(img, "Prediction =", results[0], "Apparent Prediction =", results[1])
