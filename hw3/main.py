import cv2
from detector import Detector
from reid import ReIDFeatureExtractor
from tracker import Tracker

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = Detector()
    reid_extractor = ReIDFeatureExtractor()
    tracker = Tracker()

    # save video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(f'{video_path.split(".")[0]}_output.mp4', fourcc, 30.0, (1280, 720))


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Object Detection
        current_bboxs = detector.detect(frame)
        
        # Step 2: Extract ReID Features
        features = []
        for box in current_bboxs:
            x1, y1, x2, y2 = map(int, box)
            cropped_img = frame[y1:y2, x1:x2]
            features.append(reid_extractor.extract(cropped_img))
        
        # Step 3: Update Tracker
        tracker.update(current_bboxs, features)
        print("Current tracks: ", end="")
        for track in tracker.trackers:
            print(track.track_id, end=" ")
        print("")

        # Step 4: Draw Tracks
        for track in tracker.trackers:
            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), track.color, 2)
            cv2.putText(frame, str(track.track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, track.color, 2)

        cv2.putText(frame, f'count: {tracker.next_id-1}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        out.write(frame)
        # cv2.imshow("Deep SORT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(f'-' * 30)
    print(f'Count : {tracker.next_id-1}')  # next_id now reflects the total count
    print(f'-' * 30)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("videos/easy_9.mp4")
