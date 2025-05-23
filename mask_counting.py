import cv2
import numpy as np
from sort import sort
from ultralytics import YOLO


mask_tracker = sort.Sort(max_age=2,
                         min_hits = 1,
                         iou_threshold= 0.7)
non_tracker = sort.Sort(max_age=2,
                        min_hits = 1,
                        iou_threshold= 0.7) 

mask_set = set()
non_set = set()

model = YOLO('MASK_DETECTING\mask_checking4\weights\\best.pt')
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
    font_scale    = 2
    thickness     = 2
    line_type     = cv2.LINE_AA

    text_mask = f"Mask count: {len(mask_set)}"
    text_non  = f"NoMask count: {len(non_set)}"

    cv2.putText(frame, text_mask, (10, 50), font, font_scale, (0,0,0), thickness, line_type)
    cv2.putText(frame, text_non,  (10, 110), font, font_scale, (0,0,0), thickness, line_type)

    return frame

            



if __name__ == "__main__":
    
    # you move the video project file and call 
    source = '[Asian Boss] Wuhan Citizens On Coming Out of COVID-19 Lockdown [Street Interview]ASIAN BOSS.mp4'

    # Create a video capture object from the VideoCapture Class.
    video_cap = cv2.VideoCapture(source)

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
        
        frame_result = model(frame)

        frame = add_tracker(frame_result, frame)
        boxed_frame = draw_box(frame_result,frame)
        cv2.imshow(win_name, boxed_frame)


        # Use the waitKey() function to monitor the keyboard for user input.
        # key = cv2.waitKey(0) will display the window indefinitely until any key is pressed.
        # key = cv2.waitKey(1) will display the window for 1 ms
        key = cv2.waitKey(0)

        # The return value of the waitKey() function indicates which key was pressed.
        # You can use this feature to check if the user selected the `q` key to quit the video stream.
        if key == ord('Q') or key == ord('q') or key == 27:
            # Exit the loop.
            break

    video_cap.release()
    cv2.destroyWindow(win_name)

