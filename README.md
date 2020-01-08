# Tensorflow-Grad_cam
## Abstract
#### CAM(Class Activation Mapping)은 CNN 기반 네트워크에서 많은 클래스를 결정할 때, 시각적인 설명을 제공합니다. 여기서 말하는 Grad-CAM(Gradient-weighted CAM)은 CAM을 구할 때, "예측 이미지안의 중요한 부분을 강조하는 대략적인 지역 맵을 생산하기위한 마지막 컨볼루션 층으로 흘러가는", "타겟 클래스(캡션, 마스크도 가능)에 대한" gradient를 이용합니다.

##### 따라서 적용할 수 있는 범위가 넓어졌습니다.
##### 1) fully-connected 층이 있는 CNN (VGG)
##### 2) 구성된 output을 사용하는 CNN (captioning)
##### 3) multimodal inputs 이 있는 CNN (VQA) 또는 강화학습

##### 저자는 Grad-CAM을 사용함으로써 얻는 이점 5가지를 말했습니다.
##### 1) 실패해보이는 예측에 대해 이것이 왜 실패했는지 설명을 해줄 수 있습니다.
##### 2) 적대적(adversarial) 이미지에 대해도 적용이 잘됩니다.(robust)
##### 3) 이전 방법인, ILSVRC-15 weakly-supervised localization task 의 성능을 뛰어넘었습니다.
##### 4) 근본적인 모델에 대해 더 신뢰할만 합니다.
##### 5) 데이터셋의 bias를 동일시하여(identifying) 모델 일반화를 달성합니다.

본 소스코드는 포트폴리오 용으로 작성한 것이고, 실제 코드는 https://github.com/cydonia999/Grad-CAM-in-TensorFlow 에서 확인할 수 있습니다. 현재 진행하고 있는 콘크리트 구조물 결함 검출 프로젝트에서 설명가능한 AI 기술을 접목하기 위해 관련 코드를 찾아보다가 알게 되었고, 이 코드는 따로 Training Code가 존재하지 않으며 기존의 vgg16, vgg19 모델을 사용하여 바로 결과를 확인해 볼 수 있습니다.

![687474703a2f2f692e696d6775722e636f6d2f4a614762645a352e706e67](https://user-images.githubusercontent.com/48546917/72021112-9f82be00-32b0-11ea-9e3c-83e27bfae2dc.jpg)

## Install 
- tensorflow or tensorflow-gpu
- opencv-python

## Model
https://github.com/machrisaa/tensorflow-vgg
위의 사이트에서 vgg16.npy과 vgg19.npy 모델을 다운받아 tensorflow_vgg폴더에 넣어주어야 한다.

## Process
`python grad-cam-tf.py <path_to_image> <path_to_VGG16_npy> [top_n]`
- `path_to_image`: Test할 이미지 경로.
- `path_to_VGG16_npy`: 사용하고자 하는 VGG모델 경로.
- `top_n`: Optional. Grad-CAM is calculated for each 'top_n' class, which is predicted by VGG16.

실행 후 결과 이미지는 Test할 이미지 경로에 저장된다.

## Results
