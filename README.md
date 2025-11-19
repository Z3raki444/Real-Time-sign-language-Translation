🧾 Real-Time Sign Language Translation (ASL)

This project implements a real-time American Sign Language (ASL) alphabet recognition system using a webcam and a custom-trained deep learning model. It provides a simple, beginner-friendly approach to gesture translation without relying on MediaPipe or contour-based tracking.

The system uses OpenCV for video capture and PyTorch for classification. A Convolutional Neural Network (CNN) is trained on hand images representing the ASL alphabet, and the trained model performs real-time prediction on a live video feed. The design is modular, making it easy to extend to more gestures, dynamic signs, or full-word recognition.

🚀 Features

🎥 Real-time sign detection from webcam

🤖 Custom PyTorch CNN model

✋ Recognizes ASL alphabet (A–Z)

📦 Clean and modular pipeline

📐 No MediaPipe, no contours

🧩 Easily extendable for advanced gesture recognition
