# /Users/nyl/git_projects/youtritionfacts/app.py
import streamlit as st
import os
# import subprocess # No longer needed for running the script
import json
import pandas as pd
import streamlit.components.v1 as components
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, ColumnsAutoSizeMode, DataReturnMode, JsCode
import logging

# Import GCS utilities and analysis functions
import gcs_utils
# Import the specific functions needed from the refactored analyze_video
from analyze_video import (
    get_url_key,
    download_youtube_video_gcs,
    analyze_video_gcs,
    load_results_gcs,
    save_results_gcs,
    s2mmss
)

# Configure logging (optional but helpful)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit App Config ---
st.set_page_config(layout="wide", page_icon=":material/smart_display:", page_title="YouTrition Facts") # must be first st command

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

# Load GCS results object
gcs_client, gcs_bucket = init_gcs()
results_data_len=0
if gcs_client and gcs_bucket:
    results_data = load_results_gcs(gcs_client, gcs_bucket) # Load directly from GCS
    results_data_len = len(results_data)

# # Function to convert seconds to mm:ss format (ex: 123 -> 2:03)
# def s2mmss(seconds):
#     minutes = int(seconds) // 60
#     secs = int(seconds) % 60
#     return f"{minutes:02d}:{secs:02d}"

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
        status_placeholder.info(f"🔄 Taking a look at video '{url_key}'...")
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
            "sceneChangeTimestamps": analysis['scene_change_timestamps']
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

        # print(f"new results: {results}")

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

# region CSS
st.markdown("""
<style>
.st-key-yt {
    border: 8px solid black;
    padding: 10px 14px;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    background-color: white;
    color: black;
    margin-bottom: 20px;
}
.st-key-yt h2 {
    font-size: 3rem;
    font-weight: 900;
    margin: 0;
    padding-bottom: 4px;
}
.st-key-yt .serving {
    font-size: 1rem;
    margin: 4px 0px;
}
.st-key-yt .divider-thick {
    border-top: 8px solid black;
    margin: 8px 0;
}
.st-key-yt .divider-thin {
    border-top: 1px solid black;
    margin: 4px 0;
}
            
.st-key-yt .calories {
    font-size: 22px;
    font-weight: 700;
    margin: 4px 0;
}
.st-key-yt .ag-header-cell-label {
    font-size: 3rem;
    font-weight: 900;
    margin: 0;
    padding-bottom: 4px;
    border-bottom: 1px solid black;
}
div[data-baseweb="input"] {
    border: 1px solid black !important;
    border-radius: 0px !important;
}
.stMainBlockContainer { margin-top: -50px }
</style>""", unsafe_allow_html=True)

yt_ag_css = {
    ".ag-header-cell-label": {
        "font-size": "0.8rem",
        "font-weight": "900",
        "color": "#000",
        "margin": "0",
        "padding-bottom": "4px",
        # "border-bottom": "1px solid black",
        "font-family": "'Helvetica Neue', Helvetica, Arial, sans-serif"
    },
    ".ag-header-cell": {
        "background": "#fff",
        "border-bottom": "solid 3px #000",
        "border-right": "solid 1px #000 !important",
    },
    ".ag-row": {
        "border":"none",
        "margin": "0"
    },
    ".ag-root": {
        # "border":"solid 2px #000 ",
        "margin": "0"
    },
    ".ag-cell": {
        "border-right": "solid 1px #000 !important",
        # "z-index": "1000",
        # "margin-right": "10px !important",
        "border-bottom": "solid 1px #EEE !important",
        "display": "inline-block",
        "width": "calc(100% - 10px)",
    },
}
# endregion

# Main UI (YouTrition Label)
with st.container(key="yt"): 
    st.html('<h2><span style="background: #f00; color: #fff; padding: 10px; border-radius: 15px; margin-right: 5px">You</span>Trition Facts</h2>')
    st.html("""<div class="divider-thin"></div>""")
    # st.html(f"""<span class="serving">Analyze pacing, saturation, visual complexity, and more. I validated scene duration metrics by manually comparing a handful of videos. For the other metrics, I'll publish more detail on the code. Check out the <a href="#faqs">FAQs</a> for more information on these metrics, with citations!</span>""")
    # st.html("""<div class="divider-thick"></div>""")
    youtube_url = st.text_input("🎥 Paste a YouTube URL")
    if st.button("Run Analysis"):
        if not gcs_client or not gcs_bucket:
            st.error("Cannot run analysis because GCS is not configured correctly.")
        elif youtube_url.strip() == "":
            st.warning("Please enter a YouTube URL.")
        else:
            # Run the workflow directly, no subprocess
            success, analysis_output = run_analysis_workflow(youtube_url, gcs_client, gcs_bucket)

            if success:
                st.success("Analysis complete! Refresh page to see this result updated in table below.")
                if analysis_output:
                    st.json({"analysisReport": analysis_output}) # Display the results of the run
                # Force rerun to reload the results grid from GCS
                # st.rerun()
            # Error messages are handled within     run_analysis_workflow
    
    st.html('<div class="divider-thick"></div>')

    # Load and display results from GCS
    if gcs_client and gcs_bucket:
        st.markdown(f'<div class="calories"><span style="float: left">Videos Analyzed</span><span style="float: right">{results_data_len} total</span></div>', unsafe_allow_html=True)
        # st.subheader("📊 All Videos Analyzed")
        st.html('<span class="serving">See <a href="#table-1-metrics-so-many-metrics">Metrics Table</a> for more info and juicy academic papers on how these metrics affect children\'s development.</span>')
        st.markdown("""
                    * ⬆👍🏻 Avg Scene Length: lower scene duration can overstimulate 
                    * ⬇👍🏻 Motion Dynamism: high motion dynamism can cause visual fatigue. Metrics here are normalized to 0 (paint drying) to 100 (zoomies!).
                    * ⬇👍🏻 Objects on Screen: busy environments can result in distractions and diminshed learning gains
                    * ⬇👍🏻 Color Saturation: high color use can cause visual fatigue. Metrics below are normalized to 0 (sad beige videos) to 100 (eighties are back!).
        """, unsafe_allow_html=True)
        # results_data = load_results_gcs(gcs_client, gcs_bucket) # Load directly from GCS

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
            title_link_renderer=JsCode("""class UrlCellRenderer {
                init(params) {
                    this.eGui = document.createElement('a');
                    this.eGui.innerText = params.value;
                    if (params.data.link) {                      
                        this.eGui.setAttribute('href', params.data.link);
                    }
                    else{                      
                        this.eGui.setAttribute('href', '#');
                    }
                    this.eGui.setAttribute('target', '_blank');
                    this.eGui.setAttribute('style', 'text-decoration:none; color:#2a5bd7;');
                }
                getGui() {
                    return this.eGui;
                }
            }"""    )
            # title_link_renderer=JsCode('''function(params) {console.log(params);if(params.data.link != undefined) { return `<a href="${params.data.link}" target="_blank">${params.value}</a>`} else { return params.value }}''')
            gb.configure_column("title", headerName="Title",
                cellRenderer=title_link_renderer,
                width=200,
                resizable=True,
                pinned="left",
                autoSizeStrategy=ColumnsAutoSizeMode.NO_AUTOSIZE
            )
                # cellRendererParams={"innerRenderer": "html"}
            # )
            gb.configure_column("duration", headerName="Video Length") #, cellRenderer=title_link_renderer)
            gb.configure_column("avgSceneDur", width=210, headerName="Avg Scene Length (sec)", sort='asc', sortIndex=0)
            gb.configure_column("numScenes", hide=True, headerName="Scene Count")
            gb.configure_column("spm", hide=True, headerName="Scenes/Min")
            gb.configure_column("avgColorSaturation", width=180, headerName="Color Saturation")
            gb.configure_column("avgMotionDynamism", width=180, headerName="Motion Dynamism")
            gb.configure_column("avgObjectCount", width=210, headerName="Avg Objects on Screen")
            gb.configure_column("maxObjectCount",  hide=True, width=210, headerName="Max Objects on Screen")
            gb.configure_column("link", hide=True, headerName="URL")
            gb.configure_grid_options(domLayout='normal')
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)
            gb.configure_default_column(width=150, resizable=True, filterable=True, sortable=True, editable=False)#, suppressAutoSize=True)

            gridOptions = gb.build()

            search_query = st.text_input("🔍 Search Videos", "")
            gridOptions['quickFilterText'] = search_query
            gridOptions['autoSizeStrategy'] = False #ColumnsAutoSizeMode.NO_AUTOSIZE
            # print(gridOptions)
            # gridOptions["autoSizeAllColumns"] = False
            AgGrid(
                df, # Pass the original df with 'link' column
                custom_css=yt_ag_css,
                gridOptions=gridOptions,
                update_mode=GridUpdateMode.MODEL_CHANGED, # Standard update mode
                allow_unsafe_jscode=True,
                # enable_enterprise_modules=False,
                columns_auto_size_mode=ColumnsAutoSizeMode.NO_AUTOSIZE,
                # fit_columns_on_grid_load=False,
                height=500,
                width='100%',
                reload_data=True, # Important to reflect changes after analysis run + rerun
                key='analysis_grid', # Add a key for stability,
            )

        else:
            st.info("No analysis history found. Run a new analysis to get started.")
    else:
        st.warning("Data Store is not configured. Cannot load or save analysis history.")

# FAQs
st.subheader("FAQs")
st.markdown(
"""
1. **Why these metrics?**<br />
I want to find ways to analyze videos in an objective, quantifiable way. I researched (well, asked ChatGPT in deep research mode) to come up with a list of metrics for determining the quality of a YouTube video from an early childhood development standpoint, which I then supplemented with my own Google searches. You can find the full list of metrics below in Table 1. There are so many metrics that I can implement, I picked a few that was easy to do and comes up a lot in discussions with other parents and teachers. 
3. **Does this tell me which videos are bad?**<br />
Not exactly. Similar to Nutrition Facts on the back of packaged foods, this tool is meant to shed some light on what goes into a video, and you can use that data to inform whether something works or doesn't work for your specific use case. In foods, there's nothing inherently "good" or "bad" about 12g of carbs, it's just a data (em.. datum) that describes the thing. I link papers I've found related to each metric. You can read them yourself and come to your own conclusions. 
2. **Why did you make this?**<br />
I often hear people talking about the "good ol' days" when kids videos were less flashy and "better," and I wanted to see if this was true. People will also say this content creator is great, or that content creator is bad, and I'm curious what the actual data shows. I also wanted to try a project with Python CV (computer vision), and wanted to see if AI can write the code for me. 
4. **And...?**<br />
AI was able to generate.. maybe 80% of the code, and then I had to do maybe 80% of the debugging on said code. That said, with this project, because it's more self contained AI was pretty good at the narrow goals (e.g., coming up with a method to detect scene changes). What I've noticed AI struggles with is understanding the limitations, and more importantly how to work around the limitations, of specific development frameworks. Sometimes, it would just make up library functions or parameters that does not exist. For example, I used Streamlit to generate the front end and host the page, and it kept giving me wrong code for how to show links in a table. I ended up figure out how to do this the old fashioned way, which is to find a forum post where a developer had the same issue and copy/paste. 
5. **What's next?**<br />
Next up, I'd like to create a way to show a "composite" score which combines all these metrics. I also want to create an interface for showing metrics related to one video at a time. I should probably also work on a more mobile friendly UI. And of course implement more metrics, which I will probably just ask my handy dandy coding assistant to crawl through the table below and come up with methods to analyze these metrics.
""", unsafe_allow_html=True)

st.markdown(
"""
##### Table 1. Metrics! So many metrics!
| **Domain**   | **Implemented** | **Metric**                               | **How to Measure**                                               | **Developmental Relevance**                                           |
|--------------|-----------------|------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------------|
| Attention    | x               | Average Scene Length                     | Calculate average scene length. Here, I use seconds.              | Rapid scene changes can disrupt sustained attention and overstimulate children, while slower pacing helps maintain focus.[^2] [^8] |
| Attention    | x               | Motion Dynamism                          | Analyze motion rapidity within a scene.      | Fast paced videos (defined in terms of scene change  mostly, but researchers also discussion motion within a scene) led to children performing worse on executive function tasks[^8]. Fast paced videos can also cause visual fatigue[^9]|
| Attention    | x               | Number of Objects on Screen                        | Look at number of distinct objects on the screen, generate average across all frames | Busier classroom environments resulted in children spending more time off task and demonstrated smaller learning gains, suggesting screen clutter may similarly distract young viewers.[^6] |
| Attention    | x               | Color Saturation                         | Look at intensity of colors                                     | High color saturation may disrupt structured play and attention, as found in a study comparing colorful and non-colorful play areas for preschoolers.[^7] Excessive use of color also can cause visual fatique[^9] |
| Attention    |                 | Content Coherence                        | Assess narrative logic; count disconnected segments.             | Coherent and logically sequenced stories support sustained attention and narrative understanding, aligning with best practices for educational video design.[^1] |
| Attention    | x               | Video Length (Age-Appropriate Duration)  | Compare video length to age-appropriate guidelines.              | Videos should match young children's limited attention spans to support learning without fatigue or distraction.[^4] |

<details>
<summary>...see more domains for the future</summary>

| **Domain**   | **Implemented** | **Metric**                               | **How to Measure**                                               | **Developmental Relevance**                                           |
|--------------|-----------------|------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------------|
| Cognitive    |                 | Educational Content & Objectives         | Check for explicit learning goals or curriculum-based content.   | Content with clear educational goals supports school readiness and cognitive development, as emphasized by NIH findings on early learning stimulation.[^5] |
| Cognitive    |                 | Interactive Prompting                    | Count prompts/questions directed at the viewer.                  | Prompting viewers to think and respond fosters active cognitive engagement and problem-solving, supporting recommendations for effective learning video design.[^1] |
| Cognitive    |                 | Repetition for Reinforcement             | Tally repeated key words/concepts.                               | Repetition enhances memory retention and concept mastery, aligning with principles for maximizing learning through video content.[^1] |
| Cognitive    |                 | Realism vs. Fantastical Content          | Rate realism vs. fantasy; note presence of surreal elements.     | Realistic content is easier to comprehend and process, while fantastical elements may overstimulate young viewers and hinder understanding, as discussed in digital media impact studies.[^2] |
| Language     |                 | Vocabulary Diversity                     | Analyze transcript for unique words/new word use.                | Exposure to diverse vocabulary promotes robust language development in early years, consistent with research on media and child development.[^3] |
| Language     |                 | Speech Pace & Clarity                    | Calculate words/minute; rate clarity.                            | Slower, clearer speech improves comprehension and language acquisition in young viewers, as shown in video learning research.[^1] |
| Language     |                 | Interactive Dialogue                     | Count call-and-response moments and pauses.                      | Opportunities for verbal interaction, even if simulated, can strengthen expressive language skills in early learners.[^3] |
| Language     |                 | Visual Language Supports                 | Note use of captions, labels, or relevant images.                | Pairing spoken words with visual cues like text or imagery enhances word recognition and comprehension, in line with multimedia learning principles.[^1] |
| Emotional    |                 | Pro-Social/Emotional Lessons             | Count instances of sharing, emotional naming, empathy.           | Exposure to prosocial behaviors through media helps children learn empathy, cooperation, and emotional regulation strategies.[^3] |
| Emotional    |                 | Calm Tone & Emotional Safety             | Assess tone, volume, sudden sounds; rate on calmness scale.      | Calm and predictable media environments help support emotional regulation and prevent overstimulation, a concern raised in studies on media’s impact on young brains.[^2] |
| Emotional    |                 | Conflict & Violence Content              | Count conflicts/violence; review resolution quality.             | Frequent or poorly resolved conflict in media can model aggression, while peaceful resolution promotes healthy social-emotional development.[^3] |
| Emotional    |                 | Emotional Engagement & Empathy           | Count emotion words, emotional scenes, empathetic moments.       | Content that models and names emotions builds emotional awareness and empathy, foundational skills in early childhood development.[^3] |

</details>

[^1]: Brame C. J. (2016). Effective Educational Videos: Principles and Guidelines for Maximizing Student Learning from Video Content. CBE life sciences education, 15(4), es6. https://doi.org/10.1187/cbe.16-03-0125
[^2]: Hutton, J.S., Piotrowski, J.T., Bagot, K. et al. Digital Media and Developing Brains: Concerns and Opportunities. Curr Addict Rep 11, 287–298 (2024). https://doi.org/10.1007/s40429-024-00545-3
[^3]: Hadders-Algra M. (2020). Interactive media use and early childhood development. Jornal de pediatria, 96(3), 273-275. https://doi.org/10.1016/j.jped.2019.05.001
[^4]: Canadian Paediatric Society, Digital Health Task Force, Ottawa, Ontario (2017). Screen time and young children: Promoting health and development in a digital world. Paediatrics & child health, 22(8), 461–477. https://doi.org/10.1093/pch/pxx123                        
[^5]: Eunice Kennedy Shriver National Institute of Child Health and Human Development. (2018). Media exposure and early child development workshop (NIH Publication No. 18-XX). U.S. Department of Health and Human Services. https://www.nichd.nih.gov/sites/default/files/2018-06/Media_Exp_Early_Child_Dev_Work.pdf
[^6]: Fisher, A. V., Godwin, K. E., & Seltman, H. (2014). Visual Environment, Attention Allocation, and Learning in Young Children: When Too Much of a Good Thing May Be Bad. Psychological Science, 25(7), 1362-1370. https://doi.org/10.1177/0956797614533801
[^7]: Stern-Ellran, K., Zilcha-Mano, S., Sebba, R., & Levit Binnun, N. (2016). Disruptive Effects of Colorful vs. Non-colorful Play Area on Structured Play—A Pilot Study with Preschoolers. *Frontiers in Psychology*, 7, 1661. https://doi.org/10.3389/fpsyg.2016.01661
[^8]: Lillard, A. S., & Peterson, J. (2011). The immediate impact of different types of television on young children's executive function. Pediatrics, 128(4), 644–649. https://doi.org/10.1542/peds.2010-1919
[^9]: Argilés, M., Fonts, E., Pérez-Mañá, L., Martinez-Navarro, B., Sora-Domenjó, C., Pérez-Cabré, E., Sunyer-Grau, B., Rovira-Gay, C., Molins-Pitarch, C., & Quevedo-Junyent, L. (2024). Effects of colour and scene dynamism on visual fatigue in animated films. Scientific reports, 14(1), 26683. https://doi.org/10.1038/s41598-024-78329-y
""", unsafe_allow_html=True)

# with st.expander("📄 Raw JSON"):
#     st.json(results_data) # Show the raw data loaded from GCS