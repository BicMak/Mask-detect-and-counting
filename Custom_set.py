import torch
import albumentations as A
import cv2
from ultralytics.data.augment import v8_transforms
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.data.dataset import YOLODataset
import yaml
import numpy as np
from types import SimpleNamespace

hyp = {
    'mosaic': 0.0,        # 모자이크 확률
    'mixup': 0.0,        # 믹스업 확률
    'copy_paste': 0.0,    # 복사-붙여넣기 확률
    'degrees': 0.0,       # 회전 각도 ±
    'translate': 0.0,     # 평행이동 ±
    'scale': 0.0,         # 이미지 스케일 ±
    'shear': 0.0,         # 전단 각도 ±
    'perspective': 0.0,   # 원근감 ±
    'flipud': 0.0,        # 상하 반전 확률
    'fliplr': 0.5,        # 좌우 반전 확률
    'hsv_h': 0.02,       # HSV-색조 증강 
    'hsv_s': 0.7,         # HSV-채도 증강
    'hsv_v': 0.4,         # HSV-명도 증강
}

hyp_ns = IterableSimpleNamespace(**hyp)

dummy_dataset = SimpleNamespace(
    data={"flip_idx": []},  # keypoints 관련 키만 채워주면 OK
    use_keypoints=False     # detection 모드이므로 False
)

# 실제 사용할 Transform 객체 생성


class CustomAugYOLODataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.safety = A.Compose([A.RandomSizedBBoxSafeCrop(height=640,width=640,erosion_rate = 0.0,p=1)],
                                 bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

        # 더미 데이터셋 생성
        dummy_dataset = SimpleNamespace(
            data={"flip_idx": []},  # keypoints 관련 키만 채워주면 OK
            use_keypoints=False     # detection 모드이므로 False
        )

        self.yolo_aug = v8_transforms(dataset=dummy_dataset,  # None 대신 dummy_dataset 사용  
                                      imgsz=640,
                                      hyp=hyp_ns,
                                      stretch=False)         

    def __len__(self): 
        return len(self.im_files)  # img_paths 대신 im_files 사용

    def __getitem__(self, idx):
        # 1. YOLOv8 기본 증강 비활성화 (내 증강만 사용하기 위함)
        original_augment = self.augment
        self.augment = False
        
        # 2. 기본 이미지/레이블 로드 (YOLOv8 증강 없이)
        img, labels = super().__getitem__(idx)
        
        # 3. 원래 증강 설정 복원
        self.augment = original_augment
        
        # 4. 내 커스텀 증강 적용 (원래 증강 옵션이 켜져 있고 레이블이 있는 경우만)
        if original_augment and len(labels) > 0:
            try:
                # 이미지 형식 변환 (CHW -> HWC)
                if isinstance(img, torch.Tensor):
                    img_np = img.permute(1, 2, 0).cpu().numpy()
                elif img.shape[0] == 3:  # 이미 CHW 형식인 경우
                    img_np = img.transpose(1, 2, 0)
                else:
                    img_np = img  # 이미 HWC 형식인 경우
                    
                # 박스와 클래스 정보 준비
                bboxes = labels[:, 1:5].cpu().numpy() if isinstance(labels, torch.Tensor) else labels[:, 1:5]
                class_ids = labels[:, 0].cpu().numpy().astype(int).tolist() if isinstance(labels, torch.Tensor) else labels[:, 0].astype(int).tolist()
                
                # 1단계: Albumentations 증강 적용
                albu_result = self.safety(
                    image=img_np,
                    bboxes=bboxes,
                    labels=class_ids
                )
                
                # 결과 가져오기
                img_albu = albu_result['image']
                bboxes_albu = albu_result['bboxes']
                class_ids_albu = albu_result['labels']
                
                # 박스가 있는 경우만 처리
                if len(bboxes_albu) > 0:
                    # 2단계: YOLOv8 증강 적용 (v8_transforms 요구 형식 맞춰주기)
                    yolo_data = {
                        "img": img_albu,                    # 이미지 (HWC 형식)
                        "cls": np.array(class_ids_albu).reshape(-1, 1),  # 클래스 ID
                        "bboxes": np.array(bboxes_albu)     # 바운딩 박스
                    }
                    
                    yolo_result = self.yolo_aug(yolo_data)
                    
                    # 결과 가져오기
                    img_final = yolo_result['img']
                    cls_final = yolo_result['cls']
                    bboxes_final = yolo_result['bboxes']
                    
                    # 레이블 재구성 [class, x, y, w, h]
                    if cls_final.shape[0] > 0:  # 클래스가 있는지 확인
                        labels_final = np.hstack((cls_final, bboxes_final))
                        
                        # YOLOv8 학습에 필요한 형식으로 변환
                        if not isinstance(img_final, torch.Tensor):
                            if img_final.shape[0] != 3:  # HWC 형식인 경우
                                img_final = img_final.transpose(2, 0, 1)  # HWC -> CHW
                            img_final = torch.from_numpy(img_final).float()
                        
                        labels_final = torch.from_numpy(labels_final).float() if isinstance(labels_final, np.ndarray) else labels_final
                        
                        # 최종 결과 반환
                        return img_final, labels_final
                
                # 바운딩 박스가 없는 경우
                else:
                    # 이미지만 변환하고 빈 레이블 반환
                    if img_albu.shape[0] != 3:  # HWC 형식인 경우
                        img_albu = img_albu.transpose(2, 0, 1)  # HWC -> CHW
                    img_albu = torch.from_numpy(img_albu).float() if isinstance(img_albu, np.ndarray) else img_albu
                    return img_albu, torch.zeros((0, 5), dtype=torch.float32)
                    
            except Exception as e:
                print(f"증강 중 오류 발생 (idx={idx}): {e}")
                # 오류 발생 시 원본 사용
        
        # 5. 증강 적용하지 않은 경우 또는 오류 발생 시 원본 반환
        # 텐서 타입 확인 및 변환
        if not isinstance(img, torch.Tensor):
            if img.shape[0] == 3:  # CHW 형식인지 확인
                img = torch.from_numpy(img).float()
            else:  # HWC 형식인 경우
                img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels).float() if len(labels) > 0 else torch.zeros((0, 5), dtype=torch.float32)
        
        return img, labels