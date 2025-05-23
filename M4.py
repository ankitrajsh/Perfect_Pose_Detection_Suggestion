import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# -------- Pose Detection Setup --------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --------- Mock Pose Suggestions ---------
# Replace these with your real pose thumbnails and guidance text
POSE_SUGGESTIONS = [
    {
        "img_path": "pose1.png",  # Provide your pose image files here
        "description": "Turn left, cross arms, slight smile"
    },
    {
        "img_path": "pose2.png",
        "description": "Raise right hand, smile naturally"
    },
    {
        "img_path": "pose3.png",
        "description": "Look forward, hands in pockets"
    }
]

class PoseSuggestionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Detection & Suggestion")

        # Webcam video label
        self.video_label = ttk.Label(root)
        self.video_label.grid(row=0, column=0, padx=10, pady=10)

        # Suggestions frame
        self.suggestion_frame = ttk.LabelFrame(root, text="Top 3 Pose Suggestions")
        self.suggestion_frame.grid(row=0, column=1, padx=10, pady=10)

        self.pose_thumbs = []
        self.pose_desc_labels = []

        for i, suggestion in enumerate(POSE_SUGGESTIONS):
            # Load pose thumbnail images
            try:
                img = Image.open(suggestion["img_path"]).resize((150, 150))
            except Exception:
                # If image not found, create blank placeholder
                img = Image.new("RGB", (150, 150), (200, 200, 200))

            img_tk = ImageTk.PhotoImage(img)
            lbl_img = ttk.Label(self.suggestion_frame, image=img_tk)
            lbl_img.image = img_tk
            lbl_img.grid(row=i, column=0, pady=5)
            self.pose_thumbs.append(lbl_img)

            # Pose description
            lbl_desc = ttk.Label(self.suggestion_frame, text=suggestion["description"], wraplength=150)
            lbl_desc.grid(row=i, column=1, padx=5)
            self.pose_desc_labels.append(lbl_desc)

        # Guidance label
        self.guidance_label = ttk.Label(root, text="Pose Guidance will appear here", font=("Arial", 12))
        self.guidance_label.grid(row=1, column=0, columnspan=2, pady=10)

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.update_video()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            self.root.after(30, self.update_video)
            return

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process pose
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                rgb_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            # Update pose guidance text dynamically (mock)
            self.guidance_label.config(text="Detected pose - Try matching: " + POSE_SUGGESTIONS[0]["description"])
        else:
            self.guidance_label.config(text="No pose detected - please step in front of camera")

        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_video)

    def on_close(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PoseSuggestionApp(root)
    root.mainloop()
