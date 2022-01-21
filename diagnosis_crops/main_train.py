import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json 
from glob import glob

NUM = int(input("이미지 인덱스 입력 : "))

## 작물 상태 데이터 정보
crop = {'1':'딸기','2':'토마토','3':'파프리카','4':'오이','5':'고추','6':'시설포도'}
disease = {'1':{'a1':'딸기잿빛곰팡이병','a2':'딸기흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '2':{'a5':'토마토흰가루병','a6':'토마토잿빛곰팡이병','b2':'열과','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '3':{'a9':'파프리카흰가루병','a10':'파프리카잘록병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '4':{'a3':'오이노균병','a4':'오이흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '5':{'a7':'고추탄저병','a8':'고추흰가루병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '6':{'a11':'시설포도탄저병','a12':'시설포도노균병','b4':'일소피해','b5':'축과병'}}
risk = {'1':'초기','2':'중기','3':'말기'}

label_description = {}
for key, value in disease.items():
    label_description[f'{key}_00_0'] = f'{crop[key]}_정상'
    for disease_code in value:
        for risk_code in risk:
            label = f'{key}_{disease_code}_{risk_code}'
            label_description[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}'
list(label_description.items())[:10]

       
## 샘플 데이터 살펴보기
sample = glob('data/train/*')[NUM]
sample_csv = pd.read_csv(glob(sample+'/*.csv')[0])
sample_image = cv2.imread(glob(sample+'/*.jpg')[0])
sample_json = json.load(open(glob(sample+'/*.json')[0], 'r'))     

# Prediction data
image_id = int(sample.split('/')[-1])

df = pd.read_csv('data/train.csv')
label = df.loc[df['image'].values == image_id]['label'].values

for key, value in label_description.items():
    if(key == label):
        print('\n[System] 증상코드: {}'.format(key))
        print('[System] 증상명  : {}\n'.format(value))
        
# create figure
fig = plt.figure(figsize=(10, 5))
rows = 1
columns = 2

# reading images
img1 = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

# image
fig.add_subplot(rows, columns, 1)
plt.imshow(img1)
plt.title("Original image")

# visualize bbox
points = sample_json['annotations']['bbox'][0]
part_points = sample_json['annotations']['part']

cv2.rectangle(
    img2,
    (int(points['x']), int(points['y'])),
    (int((points['x']+points['w'])), int((points['y']+points['h']))),
    (0, 255, 0),
    2
)
for part_point in part_points:
    point = part_point
    cv2.rectangle(
        img2,
        (int(point['x']), int(point['y'])),
        (int((point['x']+point['w'])), int((point['y']+point['h']))),
        (255, 0, 0),
        1
    )
fig.add_subplot(rows, columns, 2)
plt.imshow(img2)
plt.title("Detected disease image")
plt.show()
