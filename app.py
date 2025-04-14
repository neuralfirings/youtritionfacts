# /Users/nyl/git_projects/youtritionfacts/app.py
import streamlit as st
import os
# import subprocess # No longer needed for running the script
import json
import pandas as pd
import streamlit.components.v1 as components
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode
import logging

# Import GCS utilities and analysis functions
import gcs_utils
# Import the specific functions needed from the refactored analyze_video
from analyze_video import (
    get_url_key,
    download_youtube_video_gcs,
    analyze_video_gcs,
    load_results_gcs,
    save_results_gcs
)

# Configure logging (optional but helpful)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
# RESULTS_FILE = "./results.json" # No longer needed

# --- Streamlit App Logic ---
st.set_page_config(layout="wide", page_icon="🤔", page_title="YouTrition Facts") # must be first st command
st.title("📺 YouTrition Facts")
st.markdown("Paste a YouTube video link to analyze pacing, saturation, visual complexity, and more. I validated scene duration metrics by manually comparing a handful of videos. For the other metrics, I'll publish more detail on the code. Scroll down to see why I picked the metrics I picked.")

youtube_url = st.text_input("🎥 YouTube URL")

# --- GCS Initialization ---
# Initialize client and bucket once using caching
@st.cache_resource
def init_gcs():
    client = gcs_utils.get_gcs_client()
    if not client:
        st.error("Failed to initialize Google Cloud Storage client. Check secrets/credentials.")
        return None, None
    bucket = gcs_utils.get_bucket(client)
    if not bucket:
        st.error(f"Failed to access GCS Bucket '{gcs_utils.GCS_BUCKET_NAME}'. Check bucket name and permissions in secrets.")
        return client, None
    return client, bucket

gcs_client, gcs_bucket = init_gcs()

def s2mmss(seconds):
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"

# Function to run the analysis workflow directly
def run_analysis_workflow(url, client, bucket):
    if not client or not bucket:
        st.error("GCS client or bucket not initialized. Cannot run analysis.")
        return False, None # Indicate failure

    url_key = get_url_key(url)
    if not url_key:
        st.error("Could not extract a valid ID from the YouTube URL.")
        return False, None

    status_placeholder = st.empty() # For dynamic status updates

    try:
        # 1. Download/Upload Video
        status_placeholder.info(f"🔄 Checking/Downloading video '{url_key}'...")
        gcs_blob_name, title = download_youtube_video_gcs(url, url_key, client, bucket)
        if not gcs_blob_name:
            if title: 
                st.error(title)
            else:
                st.error("Failed to download or upload video to GCS.")
            status_placeholder.empty() # Clear status message
            return False, None

        # 2. Analyze Video from GCS
        status_placeholder.info(f"🔬 Analyzing video '{title}'...")
        analysis = analyze_video_gcs(gcs_blob_name, client, bucket)
        if not analysis:
            st.error("Video analysis failed.")
            return False, None

        # 3. Load existing results from GCS (use locking if concurrent writes are possible, but unlikely in basic Streamlit)
        status_placeholder.info("💾 Loading existing results...")
        results = load_results_gcs(client, bucket) # Load fresh results before updating

        # 4. Update results
        new_analysis_db_item = {
            "ytKey": url_key,
            "link": f"https://www.youtube.com/watch?v={url_key}",
            "title": title,
            "duration": round(analysis['duration'], 2),
            "numScenes": analysis['scene_count'],
            "spm": round(analysis['scenes_per_minute'], 2),
            "avgSceneDur": round(analysis['avg_scene_duration'], 2),
            "avgColorSaturation": round(analysis['avg_saturation'], 2),
            "avgMotionDynamism": round(analysis['motion_dynamism'], 2),
            "avgObjectCount": round(analysis['avg_object_count'], 2),
            "maxObjectCount": analysis['max_object_count'],
        }
        updated_existing = False
        for i, item in enumerate(results):
            if item['ytKey'] == url_key:
                results[i] = new_analysis_db_item
                updated_existing = True
                logger.info(f"Updated existing entry for {url_key}")
                break
        if not updated_existing:
            results.append(new_analysis_db_item) 
            logger.info(f"Appended new entry for {url_key}")

        print(f"new results: {results}")

        # 5. Save updated results to GCS
        status_placeholder.info("💾 Saving updated results to GCS...")
        if not save_results_gcs(results, client, bucket):
             st.warning("Failed to save updated results to GCS.") # Warn but maybe proceed
             # Decide if this is a critical failure

        status_placeholder.empty() # Clear status message
        return True, new_analysis_db_item #results[url_key] # Indicate success and return the new entry

    except Exception as e:
        logger.error(f"Error during analysis workflow for {url}: {e}", exc_info=True)
        st.error(f"An unexpected error occurred during analysis: {e}")
        status_placeholder.empty() # Clear status message
        return False, None


if st.button("Run Analysis"):
    if not gcs_client or not gcs_bucket:
        st.error("Cannot run analysis because GCS is not configured correctly.")
    elif youtube_url.strip() == "":
        st.warning("Please enter a YouTube URL.")
    else:
        # Run the workflow directly, no subprocess
        success, analysis_output = run_analysis_workflow(youtube_url, gcs_client, gcs_bucket)

        if success:
            st.success("Analysis complete!")
            if analysis_output:
                 st.subheader("Latest Analysis Results:")
                 st.json(analysis_output) # Display the results of the run
            # Force rerun to reload the results grid from GCS
            st.rerun()
        # Error messages are handled within run_analysis_workflow


# Load and display results from GCS
if gcs_client and gcs_bucket:
    st.subheader("📊 All Videos Analyzed")
    st.text("Color saturation and motion are normalized to 0 (low) - 100 (high)")
    results_data = load_results_gcs(gcs_client, gcs_bucket) # Load directly from GCS

    if results_data:
        df = pd.DataFrame(results_data)

        column_order = [
            'title',
            'duration',
            'avgSceneDur',
            'numScenes',
            # 'spm'
            'avgColorSaturation',
            'avgMotionDynamism',
            'avgObjectCount',
            'maxObjectCount', 'link'
        ]

        df = df[column_order]

        df.duration = df.duration.apply(s2mmss)
        

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_column("ytKey", hide=True)
        title_link_renderer=JsCode("""
            class UrlCellRenderer {
            init(params) {
                this.eGui = document.createElement('a');
                this.eGui.innerText = params.value;
                this.eGui.setAttribute('href', params.data.link);
                this.eGui.setAttribute('target', '_blank');
                this.eGui.setAttribute('style', 'text-decoration:none; color:#2a5bd7;');
            }
            getGui() {
                return this.eGui;
            }
            }
            """
        )
        # title_link_renderer=JsCode('''function(params) {console.log(params);return `<a href="${params.data.link}" target="_blank">${params.value}>params.value</a>`}''')
        gb.configure_column("title", headerName="Title",
            cellRenderer=title_link_renderer,
            maxWidth=500, #suppressSizeToFit=True
        )
            # cellRendererParams={"innerRenderer": "html"}
        # )
        gb.configure_column("duration", headerName="Video Length", resizable=False) #, cellRenderer=title_link_renderer)
        gb.configure_column("avgSceneDur", headerName="Avg Scene (sec)", sort='asc', sortIndex=0)
        gb.configure_column("numScenes", headerName="Scene Count")
        gb.configure_column("spm", hide=True, headerName="Scenes/Min")
        gb.configure_column("avgColorSaturation", headerName="Color Saturation")
        gb.configure_column("avgMotionDynamism", headerName="Motion")
        gb.configure_column("avgObjectCount", headerName="Objects on Screen (Avg)")
        gb.configure_column("maxObjectCount", headerName="Objects on Screen (Max)")
        gb.configure_column("link", hide=True, headerName="URL")
        gb.configure_grid_options(domLayout='normal')
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)
        gb.configure_default_column(resizable=True, filterable=True, sortable=True, editable=False)

        gridOptions = gb.build()

        AgGrid(
            df, # Pass the original df with 'link' column
            gridOptions=gridOptions,
            update_mode=GridUpdateMode.MODEL_CHANGED, # Standard update mode
            allow_unsafe_jscode=True,
            enable_enterprise_modules=False,
            # fit_columns_on_grid_load=False,
            height=500,
            width='100%',
            reload_data=True, # Important to reflect changes after analysis run + rerun
            key='analysis_grid' # Add a key for stability,
        )

        # citations
        st.subheader("Citations")
        st.text("Some research related to the metrics used here")
        st.markdown("""* Re: __Average Scene Duration__ - [The Immediate Impact of Different Types of Television on Young Children's Executive Function](https://pmc.ncbi.nlm.nih.gov/articles/PMC9923845/#:~:text=Children%20who%20watched%20the%20fast,attention%2C%20age%2C%20and%20television%20exposure) (paper focuses on fast paced videos, with a priamry metric as scene duration)
* Re: __Motion Dynamism__ Adding this metric since this is tangentially related to pacing
* Re: __Color Saturation__ - [Disruptive Effects of Colorful vs. Non-colorful Play Area on Structured Play—A Pilot Study with Preschoolers](https://pmc.ncbi.nlm.nih.gov/articles/PMC5083879)
* Re: __Number of Objects on Screen__ - [Effect of Repeated Exposure to the Visual Environment on Young Children's Attention](https://onlinelibrary.wiley.com/doi/10.1111/cogs.13093)
""")
        with st.expander("📄 Raw JSON"):
            st.json(results_data) # Show the raw data loaded from GCS
    else:
        st.info("No analysis history found. Run a new analysis to get started.")
else:
    st.warning("Data Store is not configured. Cannot load or save analysis history.")

