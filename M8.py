import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ----------- Pose Detection Setup -----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ----------- Sample outfit images (replace with your own) -----------
OUTFIT_IMAGES = [
    "outfit1.jpg",
    "outfit2.jpg",
    "outfit3.jpg"
]

# Camera recommendations (mock data)
CAMERA_RECOMMENDATIONS = """
- Use natural light for best results.
- Set ISO to 100.
- Use aperture f/2.8 for portrait mode.
- Avoid backlight.
"""

# ----------- Tkinter GUI Setup -----------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Photo Assistant")

        # --- Video Frame ---
        self.video_label = ttk.Label(root)
        self.video_label.grid(row=0, column=0, padx=10, pady=10)

        # --- Outfit Suggestions ---
        self.outfit_frame = ttk.LabelFrame(root, text="Outfit Suggestions")
        self.outfit_frame.grid(row=0, column=1, padx=10, pady=10)

        self.outfit_thumbs = []
        for i, img_path in enumerate(OUTFIT_IMAGES):
            img = Image.open(img_path).resize((100, 100))
            img_tk = ImageTk.PhotoImage(img)
            lbl = ttk.Label(self.outfit_frame, image=img_tk)
            lbl.image = img_tk  # keep reference
            lbl.grid(row=i, column=0, pady=5)
            self.outfit_thumbs.append(lbl)

        # --- Camera Recommendations ---
        self.cam_rec_frame = ttk.LabelFrame(root, text="Camera Recommendations")
        self.cam_rec_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.cam_rec_text = tk.Text(self.cam_rec_frame, height=6, width=40)
        self.cam_rec_text.insert(tk.END, CAMERA_RECOMMENDATIONS)
        self.cam_rec_text.config(state=tk.DISABLED)
        self.cam_rec_text.pack()

        # --- Mock Preview Area ---
        self.mock_preview_frame = ttk.LabelFrame(root, text="Mock Preview")
        self.mock_preview_frame.grid(row=1, column=1, padx=10, pady=10)

        self.mock_preview_label = ttk.Label(self.mock_preview_frame)
        self.mock_preview_label.pack()

        # Load placeholder mock preview image
        placeholder_img = Image.new("RGB", (300, 400), color="gray")
        self.mock_preview_imgtk = ImageTk.PhotoImage(placeholder_img)
        self.mock_preview_label.config(image=self.mock_preview_imgtk)

        # --- Action Buttons ---
        self.button_frame = ttk.Frame(root)
        self.button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.try_pose_btn = ttk.Button(self.button_frame, text="Try Pose", command=self.try_pose)
        self.try_pose_btn.grid(row=0, column=0, padx=5)

        self.shop_outfit_btn = ttk.Button(self.button_frame, text="Shop Outfit", command=self.shop_outfit)
        self.shop_outfit_btn.grid(row=0, column=1, padx=5)

        self.take_photo_btn = ttk.Button(self.button_frame, text="Take Photo Now", command=self.take_photo)
        self.take_photo_btn.grid(row=0, column=2, padx=5)

        # Start video capture
        self.cap = cv2.VideoCapture(0)
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            self.root.after(30, self.update_video)
            return

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process pose detection
        results = pose.process(rgb_frame)

        # Draw pose landmarks on frame
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                rgb_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2)
            )

        # Convert to ImageTk
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_video)

    def try_pose(self):
        print("Try Pose clicked!")
        # Here you could trigger pose suggestions or tutorials

    def shop_outfit(self):
        print("Shop Outfit clicked!")
        # Here you could open shopping links or show outfit details

    def take_photo(self):
        print("Take Photo clicked!")
        ret, frame = self.cap.read()
        if ret:
            filename = "captured_photo.jpg"
            cv2.imwrite(filename, frame)
            print(f"Photo saved as {filename}")

    def on_close(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
