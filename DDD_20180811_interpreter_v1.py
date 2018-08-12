############### Instructions ######################
# 아래 imagePath, modelfullpath, labelfilefullpath 경로만 변경해 주면 실행 가능
# 그림 한개부터 여러장까지 가능

import numpy as np
import glob

from PIL import Image
from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper

# 올바른 실행을 위해서는 반드시
# 이미지가 저장된 경로인 imagePath, 모델이 저장된 경로인 modelfullpath,
# label.txt가 저정된 경로인 labelfilefullpath를 정확하게 입력해야 함
imagePath = "/home/jim/Desktop/TEST/test_images/*.jpg"
modelfullpath = "/home/jim/Desktop/TEST/mobilenet_v1_1.0_224_quant.tflite"
labelfilefullpath = "/home/jim/Desktop/TEST/labels.txt"
# imagePath의 모든 파일 이름(이경우 jpg만)을 testimage에 저장)
testimages=glob.glob(imagePath)


################### 라벨로드 ########################
# 라벨 로드 함수.. 차후에 본 함수와 같이 load_and_intepreter도 분리할 것
# 나중에 사용할 의도로 함수 연습하기 위해 분리한 것.
# 일단 my_labels라는 빈 array를 만들고 여기에 load_and_inteprete 함수의
# 파일 경로를 가지고와서 해당 파일의 라벨 이름들을 한 곳에 순차적으로 넣음
# 그러한 array를 my_labels로 return
def load_labels(filename):
    my_labels = []
    input_file = open(filename, 'r')
    for l in input_file:
        my_labels.append(l.strip())
    return my_labels
###################################################

################ 모델로드 및 진단 ####################
# 핵심 부분. 이 함수는 아래의 메인으로 통해 실행되며
# 위의 imagePath에 있 진단할 이미지 수에 따라 실행 횟수가 정해짐

def load_and_inteprete(imagefilename, modelpathname, labelfilename):
    # 레코드키핑용 lite는 floating 안쓰니까 필요 없
    # floating_mode음l = False

    # 텐서플로 라이트 모델 파일을 읽어들이기 위한 일종의 해석기.
    # 모델파일 경로에서 모델을 불러온 뒤 입력과 출력 정보를 확인
    interpreter = interpreter_wrapper.Interpreter(model_path=modelpathname)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # # check the type of the input tensor(플로팅 모델인지 여부 확인... 레코드 키핑용)
    # if input_details[0]['dtype'] == np.float32:
    #   floating_model = True

    # NxHxWxC, H:1, W:2
    # 모델에 입려되는 array 크기에 맞추어 이미지 크기 조절
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(imagefilename)
    img = img.resize((width, height))

    # add N dim
    # 이후 img를 input_data로 넣어 인터프레터 세팅하고
    # 진단 수행
    input_data = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(labelfilename)

    predicted = []
    predicted_labels = []

    for i in top_k:
        predicted = '{0:08.6f}'.format(float(results[i] / 255.0)) + " >>>   " + labels[i]
        predicted_labels.append(predicted)
    return predicted_labels

################### 메인 ###########################
# load_and_interprete 함수를 반복 실행한다.
# 이미지 경로의 파일 수에 따라 반복진단 수행
if __name__ == "__main__":
    for i in range(len(testimages)):
        prediction = load_and_inteprete(testimages[i], modelfullpath, labelfilefullpath)
        print("\n<<진단대상 이미지>>\n" + testimages[i] + "\n[진단결과]")

        for j in range(len(prediction)):
            print(prediction[j])
###################################################