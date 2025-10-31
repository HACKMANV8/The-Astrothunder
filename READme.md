SWIFT SLOT: AI Parking Management System

A high-performance, real-time AI parking management system. This project uses a custom-trained YOLOv8n model to monitor multiple video feeds, providing live slot counts, parking timers, and navigation to the nearest available spot.

Features

Multi-Zone Monitoring: A high-performance, multi-threaded backend processes multiple video feeds (carPark10.mp4, carPark20.mp4, carPark30.mp4) in parallel [cite: uploaded:carPark10.mp4, uploaded:carPark20.mp4, uploaded:carPark30.mp4].

Real-Time Dashboard: A responsive web dashboard (features.html) displays live slot counts (Free, Occupied, Total) for the selected zone [cite: uploaded:features.html].

Live Video with AI Overlay: Streams the video feed with real-time bounding boxes, timers, and navigation lines drawn by the server.

Smart Navigation: Automatically identifies the nearest free spot (class 'free') from a pre-configured entry point and draws a guidance line on the video.

Live Parking Timers: Uses object tracking (model.track()) to assign a unique ID to each occupied spot (class 'car') and displays the "Top 3 Longest Parked" vehicles on the dashboard [cite: uploaded:api_server.py, uploaded:datasets.yaml].

Efficient Caching: The server processes each video only once in a background thread. All web users connect to an in-memory cache, making the application extremely fast and scalable.

How It Works

This project is split into a Python backend and an HTML/JS frontend.

Backend (api_server.py):

A multi-threaded Flask server that launches one background AI thread for each video source upon startup.

Each thread runs a continuous model.track() loop using the custom-trained best.pt model [cite: uploaded:best.pt].

It tracks all 'car' and 'free' objects, calculates timers, and finds the nearest free spot.

The latest video frame and JSON data (counts, timers, navigation) are stored in an in-memory cache.

Flask serves this cached data to the frontend via two endpoints: /video_feed and /api/dashboard_data.

Frontend (features.html):

A static HTML/JS dashboard that uses fetch to poll the /api/dashboard_data endpoint every 3 seconds for the latest JSON.

It displays the live stream from the /video_feed endpoint.

The selection.html page [cite: uploaded:selection.html] uses localStorage to tell features.html which parking lot data to request.

How to Run

1. Backend Server

Navigate to the Backend Folder:

cd Backend_flask_AI


Install Dependencies:
Make sure you have python, pip, and git installed.

pip install -r requirements.txt


Check Configuration (CRITICAL):

Open api_server.py.

Verify that the video paths in VIDEO_SOURCES are correct (e.g., carPark10.mp4).

You must update the (x, y) pixel coordinates in ENTRY_POINTS to match the "entrance" of each of your videos.

Run the Server:

python api_server.py


The server will start on http://127.0.0.1:5000. Keep this terminal running.

2. Frontend Application

Navigate to the Frontend Folder:
In a new terminal:

cd FRONTEND_WEB


Run the App:

The easiest way is to open the index.html file directly in your web browser (e.g., Chrome, Firefox).

Usage:

Click "Dive into Swiftslot" on the index.html page [cite: uploaded:index.html].

Log in using the mock credentials on login.html [cite: uploaded:login.html].

Select a parking lot (e.g., "North Lot") on selection.html.

View the live dashboard on features.html.