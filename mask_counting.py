import cv2
import numpy as np
from sort import sort
from ultralytics import YOLO


mask_tracker = sort.Sort(max_age=10,
                         min_hits = 15,
                         iou_threshold= 0.6)
non_tracker = sort.Sort(max_age=10,
                        min_hits = 15,
                        iou_threshold= 0.6) 

mask_set = set()
non_set = set()

model = YOLO('mask_checking4\weights\\best.pt')
CLASS_NAMES = model.names  

def draw_box(result:list,
             frame:np.array) -> np.array:
    """
    Draw bounding boxes around detected faces on a video frame.
    Masked faces are outlined in green, unmasked faces are outlined in red.

    Args:
        result (List[Dict[str, Any]]): A list of detection dicts, each containing:
            - 'box' (Tuple[int, int, int, int]): Coordinates of the bounding box (x1, y1, x2, y2).
            - 'mask' (bool) : True if the face is wearing a mask; False otherwise.
        frame (np.ndarray): The current video frame in BGR format.

    Returns:
        np.ndarray: The same frame array with bounding boxes and labels drawn.
    """
    
    if len(result) == 0:
        return
    else:
        input_frame = result[0]

    img = frame.copy()
    print(img.shape)
    boxes = input_frame.boxes.xyxy.cpu().numpy()
    classes = input_frame.boxes.cls.cpu().numpy() 

    for (x1, y1, x2, y2), cls_id in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        if cls_id == 0:
            color = (0, 255, 0) # green
        else:
            color = (0, 0, 255) # red

        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 라벨 텍스트
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, text_thickness)
        text_w, text_h = text_size

        # 텍스트 배경 박스
        cv2.rectangle(img,
                      (x1, y1 - text_h - 4),
                      (x1 + text_w, y1),
                      color,
                      cv2.FILLED)
        # 텍스트 그리기
        cv2.putText(img,
                    label,
                    (x1, y1 - 4),
                    font,
                    font_scale,
                    (0, 0, 0),  # 검은색 글씨
                    text_thickness,
                    cv2.LINE_AA)

    return img
    
def add_tracker(result:list,
                frame:np.array) -> np.array:
    """
    Overlay bounding-box counting information on a video frame.
    using a SORT algorithm.

    Args:
        result (List[Dict[str, Any]]): A list of detection dicts, each containing:
            - 'box' (Tuple[int, int, int, int]): Coordinates of the bounding box (x1, y1, x2, y2).
            - 'mask' (bool) : True if the face is wearing a mask; False otherwise.
        frame (np.ndarray): The current video frame in BGR format.

    Returns:
        np.ndarray: The same frame with tracking IDs and count annotations drawn on each bounding box.
    """
    
    global mask_set, non_set
    
    if len(result) == 0:
        return
    else:
        input_frame = result[0]

    frame_data = input_frame.boxes.data.cpu().numpy()
    mask_data = frame_data[frame_data[:,5]==0 , :5 ]
    non_data =  frame_data[frame_data[:,5]==1 , :5 ]

    tracks_mask = mask_tracker.update(mask_data)
    tracks_non = non_tracker.update(non_data)

    mask_ids = set(tracks_mask[:, 4].astype(int).tolist())
    non_ids = set(tracks_non[:, 4].astype(int).tolist())

    mask_set |= mask_ids
    non_set |= non_ids

    # --- 텍스트 출력 설정
    font          = cv2.FONT_HERSHEY_SIMPLEX
    font_scale    = 1
    thickness     = 1
    line_type     = cv2.LINE_AA

    text_mask = f"Mask count: {len(mask_set)}"
    text_non  = f"NoMask count: {len(non_set)}"

    cv2.putText(frame, text_mask, (10, 30), font, font_scale, (255,0,0), thickness, line_type)
    cv2.putText(frame, text_non,  (10, 60), font, font_scale, (255,0,0), thickness, line_type)

    return frame

            



if __name__ == "__main__":
    
    CONFIDENCE_THRESHOLD = 0.5  # 신뢰도 임계값 (0.0 ~ 1.0)
    IOU_THRESHOLD = 0.8         # NMS IoU 임계값 (0.0 ~ 1.0)
    MAX_DETECTIONS = 10        # 최대 감지 개수

    SAVE_VIDEO = True           # 결과 영상 저장 여부
    OUTPUT_PATH = 'result_mask_detection.mp4'  # 저장할 파일 경로
    

    # you move the video project file and call 
    source = 'videoplayback (1).mp4'

    # Create a video capture object from the VideoCapture Class.
    video_cap = cv2.VideoCapture(source)

    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info - FPS: {fps}, Size: {width}x{height}")
 

    video_writer = None
    if SAVE_VIDEO:
        # 코덱 설정: 'mp4v', 'XVID', 'MJPG' 등
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
        print(f"결과 영상 저장: {OUTPUT_PATH}")

    # Create a named window for the video display.
    win_name = 'Masked people counting'
    cv2.namedWindow(win_name)



    # Enter a while loop to read and display the video frames one at a time.
    while True:
        # Read one frame at a time using the video capture object.
        has_frame, frame = video_cap.read()
        if not has_frame:
            break
        # Display the current frame in the named window.
        
        frame_result = model(frame, 
                           conf=CONFIDENCE_THRESHOLD,    # 신뢰도 임계값
                           iou=IOU_THRESHOLD,            # NMS IoU 임계값  
                           max_det=MAX_DETECTIONS,       # 최대 감지 개수
                           verbose=False)  

        frame = add_tracker(frame_result, frame)
        boxed_frame = draw_box(frame_result,frame)
        cv2.imshow(win_name, boxed_frame)

        if SAVE_VIDEO and video_writer is not None:
            video_writer.write(boxed_frame)

        # Use the waitKey() function to monitor the keyboard for user input.
        # key = cv2.waitKey(0) will display the window indefinitely until any key is pressed.
        # key = cv2.waitKey(1) will display the window for 1 ms
        key = cv2.waitKey(1)

        # The return value of the waitKey() function indicates which key was pressed.
        # You can use this feature to check if the user selected the `q` key to quit the video stream.
        if key == ord('Q') or key == ord('q') or key == 27:
            # Exit the loop.
            break

    video_cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"결과 영상이 저장되었습니다: {OUTPUT_PATH}")
    cv2.destroyWindow(win_name)

