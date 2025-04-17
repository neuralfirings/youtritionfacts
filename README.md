# YouTrition Facts

View live site: [https://youtritionfacts.streamlit.app/](https://youtritionfacts.streamlit.app/)

**YouTrition Facts** is a web application that analyzes YouTube videos to provide objective, quantifiable metrics about their visual content, presented in a format inspired by nutritional labels. It aims to help users understand factors like pacing, visual complexity, color saturation, and motion dynamism, particularly in the context of children's content and developmental research.

The application uses Streamlit for the user interface, OpenCV for video analysis and Google Cloud Storage (GCS) for storing video files and analysis results.

## Features
*   **Video Metrics:** Calculates key metrics: scene length, motion, color saturation, objects on screen
*   **Clipping for Long Videos:** Automatically analyzes only a 10-minute clip from the middle of videos longer than 10 minutes to manage processing time.
*   **Persistent Storage:** Uses Google Cloud Storage (GCS) to store downloaded video clips and the analysis results database (`results.json`).
*   **Interactive Results Table:** Displays results for all analyzed videos in a sortable, searchable, and filterable table using Ag-Grid.
*   **Visual Percentile Indicators:** Table cells are color-coded based on percentile rankings for each metric relative to the analyzed dataset (e.g., red for potentially overstimulating low scene duration, green for calmer pacing).
*   **(Dev Mode):** Optional feature to re-analyze all videos in the database.