# Tamil Nadu ECMWF Rainfall - Mobile Web App

This version changes the desktop-only Python script into a Streamlit web application.

## What it does

1. Open the app from a mobile phone.
2. Select 1 to 7 forecast days.
3. Upload the Tamil Nadu village GeoJSON.
4. Download:
   - Rainfall Warning PDF
   - District Rainfall Summary PDF

The ECMWF IFS calculation and village interpolation are retained from the original script.

## Local test

Open Command Prompt in this folder:

```text
pip install -r requirements.txt
streamlit run app.py
```

Then open the displayed address.

## Important

The original script used:

D:\Arun\weather\TAMIL NADU_VILLAGES (1).geojson

The mobile version does NOT depend on that fixed Windows path. The GeoJSON is uploaded through the phone/browser.

## Free online deployment

Use Streamlit Community Cloud:

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Open Streamlit Community Cloud.
4. Connect the GitHub repository.
5. Select `app.py`.
6. Deploy.
7. Open the generated web link on your phone.

Bookmark the link on your phone for easy access.

## ECMWF note

The app downloads ECMWF Open Data when the Generate button is pressed. Processing time depends on the ECMWF download and server resources.
