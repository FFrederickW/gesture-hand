import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils

def fingers_up(lm):
    fingers =[]
    
    fingers.append(lm[4].x < lm[3].y)
    
    fingers.append(lm[8].y < lm[6].y)
    
    fingers.append(lm[12].y < lm[10].y)
    
    fingers.append(lm[16].y < lm[14].y)
    
    fingers.append(lm[20].y < lm[18].y)
    
    return fingers

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame =cap.read()
    
    if not ret:
        continue
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            lm = hand_landmarks.landmark
            fingers = fingers_up(lm)
            
            if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                cv2.putText(frame, "Peace", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
    cv2.imshow('Hand recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break       
    
cap.release()
cv2.destroyAllWindows()
            