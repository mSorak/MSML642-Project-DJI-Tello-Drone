import cv2
import torch
import numpy as np
import queue
import threading
from collections import deque
from predictions.PredictionProcessor import PredictionProcessor
import logging
import pygame


class VideoProcessor:
    def __init__(self, tello_ip, video_port, prediction_processor, frame_queue_size=10, skip_frames=3):
        self.tello_ip = tello_ip
        self.video_port = video_port
        self.prediction_processor = prediction_processor
        self.frame_queue = deque(maxlen=frame_queue_size)
        self.running = True
        self.skip_frames = skip_frames
        self.lock = threading.Lock()

        self.latest_predictions = {"detected_objects": [], "depth_map": None}

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def capture_video(self):
        frame_counter = 0
        video_url = f'udp://{self.tello_ip}:{self.video_port}'
        cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            self.logger.error("Error: Could not open video stream.")
            return

        self.logger.info("Video capture started.")
        try:
            while self.running:
                ret, frame = cap.read()
                if ret:
                    frame_counter += 1
                    if frame_counter % self.skip_frames == 0:
                        with self.lock:
                            #if len(self.frame_queue) == self.frame_queue.maxlen:
                                #self.logger.warning("Dropping oldest frame due to queue overflow.")
                            self.frame_queue.append(frame)
                else:
                    self.logger.warning("Failed to capture frame.")
                    break
        except Exception as e:
            self.logger.error(f"Error in capture_video: {e}")
        finally:
            cap.release()
            self.logger.info("Video capture stopped.")

    def process_frames(self):
        self.logger.info("Prediction processing started.")
        try:
            while self.running:
                with self.lock:
                    if not self.frame_queue:
                        continue
                    frame = self.frame_queue[-1]

                predictions = self.prediction_processor.process_frame(frame)

                with self.lock:
                    self.latest_predictions = predictions
        except Exception as e:
            self.logger.error(f"Error in process_frames: {e}")
        finally:
            self.logger.info("Prediction processing stopped.")

    def display_video_with_pygame(self):
        pygame.init()
        screen = None

        try:
            while self.running:
                if not self.frame_queue:
                    continue

                with self.lock:
                    frame = self.frame_queue[-1]
                    predictions = self.latest_predictions
                    frame = self.annotate_frame(frame, predictions["detected_objects"])

                if screen is None:
                    h, w, _ = frame.shape
                    screen = pygame.display.set_mode((w, h))
                    pygame.display.set_caption("Tello Video Stream")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))

                screen.blit(frame_surface, (0, 0))
                pygame.display.update()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        break
        except Exception as e:
            self.logger.error(f"Error in display_video_with_pygame: {e}")
        finally:
            pygame.quit()

    def stop(self):
        with self.lock:
            self.running = False
            self.frame_queue.clear()
            self.latest_predictions = {"detected_objects": [], "depth_map": None}

    def annotate_frame(self, frame, detected_objects):
        if frame is None or not isinstance(frame, np.ndarray):
            self.logger.error("Invalid frame passed for annotation.")
            return None

        annotated_frame = frame.copy()
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["bounding_box"]
            label = obj["label"]
            confidence = obj["confidence"]
            object_depth = obj["depth"]

            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            distance_text = f"{label} ({confidence:.2f}): {object_depth:.2f} depth units"

            color = (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, distance_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated_frame
