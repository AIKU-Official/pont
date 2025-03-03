# Pont(Personalized Font)

📢 2025년 겨울학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다
</br>🎉 2025년 겨울학기 AIKU Conference 열심히상 수상!

## 소개
> 진짜 손글씨처럼 보이는 이미지를 뽑아보자!
>
키보드 딸깍으로 손글씨를 뽑을 수 있다면 얼마나 좋을까요? 깜지를 쓸 때 더 이상 손목 통증을 느끼지 않아도 될 겁니다. 고려대의 자랑 대정일 교수님은 결석 1회를 깜지 10장으로 메꿀 수 있게 해주신다고 합니다. 과연 교수님을 속일 수 있을까요?

프로젝트 목표 :
한국어 텍스트 + 한글 음절 손글씨 이미지 1개 입력 → 손글씨 이미지의 스타일을 취하여, 텍스트를 손글씨로 변환(손글씨 이미지 생성)

- 출력 이미지는 진짜 손글씨처럼 보여야 함 → 글자에 variation이 있어야 함
- 입력한 손글씨 이미지의 스타일을 담아야 함

## 방법론
### 1. 모델링
![image](https://github.com/user-attachments/assets/bd245bb7-f48c-475a-8284-fc3e153d82e5)
- [One-DM](https://github.com/dailenson/One-DM/) 모델을 베이스로 사용하였음
    - 기존에는 영어 단어를 단위로 입력 및 출력을 수행하였는데, 한글 음절을 단위로 동작하도록 수정
    - Input Text: 생성하려는 텍스트. 기존에는 영어 단어의 각 알파벳의 폰트 이미지를 이어붙여 입력으로 사용 → 한글 음절의 폰트 이미지를 사용하도록 변경, 폰트는 Pretendard 사용
    - Input Style: 손글씨 스타일을 취할 손글씨 이미지. 기존에는 영어 단어 손글씨 이미지를 입력으로 사용 → 한글 음절 손글씨 이미지를 사용하도록 변경
    - Filtered Style: Input Style 이미지에서 high-frequency 정보를 추출한 이미지. 원본 코드에서 했던 그대로 한글 음절 손글씨 이미지에 적용

Pre-training 이후 fine-tuning 순으로 이어짐. 원본 코드는 두 과정에 같은 데이터셋을 사용하였음

### 2. 모델 훈련 및 평가
기본적인 학습 세팅은 기존 코드와 동일하게 진행

데이터셋은 annotator 68명 + 각 annoator당 한글 음절 손글씨 이미지 약 1만 장 정도로 구성

- Pre-training 1: Train에 annotator 0-4, test에 5-6
- Fine-tuning 1: Pre-training 1 + Train에 0-4, test에 5-6 → 결과가 안 좋아져서 폐기
- Fine-tuning 2: Pre-training 1 + Train에 7-11, test에 12-13 → 결과가 안 좋아져서 폐기
- Pre-training 2: Train에 annotator 0-11, test에 12-13

## 환경 설정
git clone 이후 아래 코드를 실행하여 dependency들을 설치

  ```
    cd pont
    conda env create --file environment.yaml
  ```

## 사용 방법
### 0. 모델 체크포인트 다운로드
- [구글 드라이브](https://drive.google.com/drive/folders/1ozXdHYltBdBwAfijK-rD4ESfh3XqXvBE?usp=drive_link)에서 **final_model.pt** 다운로드
- **model/ckpt** 폴더에 **final_model.pt** 저장

### 1. gradio demo로 실행하는 경우
- **app.py** 파일 실행
- 터미널에서 아래 두 링크 중 아무거나 접속 한 뒤
     -Running on local URL:  http://127.0.0.1:7860
     -Running on public URL: https://e201fae01ebf502647.gradio.live
- 브라우저에서 아래와 같은 창이 뜨면 손글씨 style 이미지 한 장과 text 입력하여 데모 사용 가능
![image](https://github.com/user-attachments/assets/99a44fec-555f-4d32-b736-9881071cb088)


### 2. 터미널에서 bash 파일로 실행하는 경우
- **inference.sh** 파일 실행 혹은 아래 코드 실행
- 이때 **input_sentence**와 **style_path**는 원하는 문구, 원하는 스타일 이미지로 수정
- 실행 시 output 폴더에 **{입력한 문구}.png**로 생성된 결과가 저장됨
  
   ```
    CUDA_VISIBLE_DEVICES=0 python inference.py \
      --one_dm model/ckpt/final_model.pt \
      --input_sentence '아이쿠 한국어 손글씨 생성 프로젝트' \
      --style_path demo/style.png \
      --output_path output
   ```


## 예시 결과
정성적으로 결과가 가장 좋아보이는 pre-training 2 모델을 사용하였음. 

- 결과 분석
    - Input Text: 모서리에 부딪힌 나는 아이쿠 아파라 라고 말했다
    - Input Style:
      </br>![image](https://github.com/user-attachments/assets/303980db-fe2c-4a4b-a5da-897fa4d22246) 
    - 출력:
      ![image](https://github.com/user-attachments/assets/055fc4ed-6baf-4b4a-9f26-7226e340eea3)
        
    - 각 음절별로 따로 forward 한 뒤 결과 이미지(128*128)를 이어 붙임
    - 공백(띄어쓰기)은 (128*64)의 흰색 이미지를 따로 만들어서 이어 붙임
    - 같은 음절(아, 라)이어도 조금씩 다른 이미지가 생성됨 → variation 확보 성공
    - 진짜 손글씨처럼 삐뚤빼뚤한 이미지 생성 성공
- 양적 분석
    - 평가 지표는 Fréchet Inception Distance(FID)를 사용: 데이터셋의 실제 정답 이미지와 생성된 이미지의 유사도를 계산 (낮을수록 good)
      
        |  | 12 | 13 |
        | --- | --- | --- |
        | finetune(50k) | 227.07 | 225.44 |      
        | pretrain(50k) | 138.18 | 134.56 |
        | pretrain(120k) | 141.37 | 142.17 |


## 팀원
![Image](https://github.com/user-attachments/assets/8f74f19f-7f41-42e0-bd0c-3696984fe9d9)
  | 팀원                            | 역할                                       |
| ----------------------------- | ---------------------------------------- |
| [문정민](https://github.com/strn18) |    data preproccess, inference 코드 정리    |
| [조윤지](https://github.com/robosun78)     |    모델 finetuning, evaluation    |
| [정다현](https://github.com/dhyun22)        |    모델 pretraining, gradio 데모   |
