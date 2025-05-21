

#  Mask Detect and Counting

YOLOv8 기반 마스크 착용 탐지 및 카운팅 시스템.
실시간 CCTV 영상에서 마스크 착용 여부를 판별하고, 트래킹을 통해 총 인원 수와 마스크 착용 인원 수를 카운팅합니다.

---

##  프로젝트 구조

```
Mask-detect-and-counting/
├── weights/                  # 학습된 YOLO 모델 가중치
├── utils/                    # 유틸 함수 및 시각화 도구
├── mask_counting.py          # 메인 실행 파일
├── main.py                   # Yolov8s 학습 프로젝트
├── requirements.txt
└── README.md
```

---

##  주요 기능

* ✅ YOLOv8을 통한 마스크 착용 여부 탐지
* ✅ SORT 알고리즘을 활용한 객체 추적 및 ID 부여
* ✅ 카테고리별 마스크 착용/미착용 인원 카운팅

---

##  설치 방법

```bash
git clone https://github.com/BicMak/Mask-detect-and-counting.git
cd Mask-detect-and-counting

# 가상 환경 설정 권장
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

```

## 🧠 모델 정보

* 사용 모델: YOLOv8 (Ultralytics)
* 클래스:

  * `0`: 마스크 착용
  * `1`: 마스크 미착용
* 학습 데이터: OPENCV mask image data
* 프레임워크: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

---

## 📎 참고 라이브러리

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [abewley/sort](https://github.com/abewley/sort)


