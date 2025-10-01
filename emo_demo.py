import cv2
from fer import FER

# === الإعدادات ===
WINDOW_NAME = "Emotion Detection"
SCALE = 2.0          # كم نكبّر الصورة (جرّب 1.5 أو 2.0 أو 2.5)
TARGET_W, TARGET_H = 1280, 720  # بديل مباشر: مقاس ثابت للعرض

# كشف الوجه + العواطف
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')
detector = FER(mtcnn=False)

# افتح كاميرا اللابتوب
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW يحل مشاكل على ويندوز أحيانًا
# حاول ترفع دقة الالتقاط; إن ما تغيّرت عادي بنكبّر بالإظهار
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# أنشئ النافذة بوضع قابل لتغيير الحجم
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
# مبدئيًا خلها كبيرة
cv2.resizeWindow(WINDOW_NAME, TARGET_W, TARGET_H)

is_full = False  # حالة ملء الشاشة

print("Controls:  q للخروج | f تبديل ملء الشاشة | n نافذة عادية")

while True:
    ok, frame = cap.read()
    if not ok:
        print("ما قدرت أفتح الكاميرا")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.2, 6, minSize=(100,100))
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        try:
            result = detector.detect_emotions(roi)
            if result:
                emotion, score = detector.top_emotion(roi)
                label = f"{emotion} ({score:.2f})"
            else:
                label = "Detecting..."
        except Exception:
            label = "Emotion OFF"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,255), 2)

    # === طريقتان للتكبير ===
    # (A) تكبير نسبي
    display = cv2.resize(frame, None, fx=SCALE, fy=SCALE,
                         interpolation=cv2.INTER_LINEAR)

    # (B) أو مقاس شاشة ثابت (علّق A وشغّل B لو تحب مقاس محدد)
    # display = cv2.resize(frame, (TARGET_W, TARGET_H),
    #                      interpolation=cv2.INTER_LINEAR)

    cv2.imshow(WINDOW_NAME, display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        # Fullscreen
        is_full = True
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    elif key == ord('n'):
        # نافذة عادية قابلة للتكبير بالسحب
        is_full = False
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, TARGET_W, TARGET_H)

cap.release()
cv2.destroyAllWindows()