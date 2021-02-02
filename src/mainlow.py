import pandas as pd
import streamlit as st
from aitextgen import aitextgen
import random
import time
#import streamlit.report_thread as ReportThread
#from streamlit.server.server import Server
import SessionState

def get_session_state(rando):
    session_state = SessionState.get(random_number=random.random(), randart='',
                    randtitle='', prompt='', nsamples='', length_gen='', temperature='', topk='', topp='', no_repeat_ngram_size='', songfirstline='', gentext='', firstgen=0, ttg=0)
    return session_state

@st.cache()
def cacherando():
    rando = random_number=random.random()
    return rando

def display_app_header(main_txt,sub_txt,is_sidebar = False):
    """
    function to display major headers at user interface
    Parameters
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed 
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <div style = "background.color:#3c403f  ; padding:15px">
    <h2 style = "color:white; text_align:center;"> {main_txt} </h2>
    <p style = "color:white; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)

def display_side_panel_header(txt):
    """
    function to display minor headers at side panel
    Parameters
    ----------
    txt: str -> the text to be displayed
    """
    st.sidebar.markdown(f'## {txt}')

@st.cache(allow_output_mutation=True, max_entries=1) #ttl=1200,
def load_aitextgen():
    return aitextgen(model="bigjoedata/rockbot-scratch",
                    config="config.json",
                    vocab_file="vocab.json",
                    merges_file="merges.json",
                    special_tokens_map="special_tokens_map.json",
                    custom_tokenizer=True,
                    tokenizer="tokenizer_config.json") # This is a GPT-2 Model built from scratch with custom vocabulary; it is not fine-tuned on anything
    #return aitextgen(model="bigjoedata/rockbot", num_workers=1) # This is fine-tuned on the 124M token GPT-2 Model
    # return aitextgen(model="bigjoedata/rockbot-distilgpt2") # This is 60% lighter due to being fine-tuned on the reduced Huggingface distilgpt2 Model
    # return aitextgen(model="bigjoedata/rockbot355M")

@st.cache()
def artistsload():
    df=pd.read_parquet('theartists.parquet')
    #df = df.iloc[:, 0].apply(lambda x: x.upper())
    return df

@st.cache(max_entries=1)
def setart(df, rando): #session_id):
    randart=random.randint(0, len(df))
    return randart

@st.cache(max_entries=1)
def settitle(rando): #(session_id):
    sampletitles=[
        "Love Is A Vampire",
        "The Cards Are Against Humanity",
        "My Grandmother Likes My Music",
        "Call A Doctor, It Is Urgent",
        "So, That Just Happened",
        "Dogs Versus Cats",
        "Parties Are Overrated",
        "I Believe That Is Butter",
        "Panic In The Grocery Store",
        "He's Not A Suspect Yet",
        "Jumping To The Moon",
        "My Father Enjoys Scotch",
        "My Goodness You Are Silly",
        "The Clouds Are Bright",
        "I Love People",
        "Notorious Kangaroos"
    ]
    randtitle = random.choice(sampletitles)
    return randtitle

def generate_text(ai, prefix, nsamples, length_gen, temperature, topk, topp, no_repeat_ngram_size):
    batch_size = min(nsamples, 5)
    return ai.generate(
        n=nsamples,
        #batch_size=1,
        batch_size=nsamples, # disable to reduce memory usage
        prompt=prefix,
        max_length=length_gen,
        temperature=temperature,
        top_k=topk,
        top_p=topp,
        #repetition_penalty=4.0, # disabled to reduce CPU/GPU usage
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=True, 
        return_as_list=True,
        eos_token = "<|endoftext|>",
        bos_token = "<|startoftext|>",
        to_gpu=False
        #num_workers=1 # This parameter doesn't seem to affect CPU usage for generation
    )

def main():
    st.set_page_config(page_title='Rockbot') #layout='wide', initial_sidebar_state='auto'
    rando = cacherando()
    session_state = get_session_state(rando)
    main_txt = """🎸 🥁 Rockbot 🎤 🎧"""
    sub_txt = "Just have fun"
    subtitle = """
A [GPT-2](https://openai.com/blog/better-language-models/) based lyrics generator fine-tuned on the writing styles of nearly 20k songs by 418 artists across MANY genres (not just rock), as well as a few poets and comedians.

**Instructions:** Type in a fake song title, pick an artist, click "Generate". Adjust settings on the left

Most language models are imprecise and Rockbot is no exception. You may see NSFW lyrics unexpectedly. I have made no attempts to censor. Generated lyrics may be repetitive and/or incoherent at times, but hopefully you'll encounter something interesting or memorable.

Oh, and generation is resource intense and can be slow. I set governors on song length to keep generation time somewhat reasonable. You may adjust song length and other parameters on the left or check out [Github](https://github.com/bigjoedata/rockbot) to spin up your own Rockbot.

Just have fun.
        """
    display_app_header(main_txt,sub_txt,is_sidebar = False)
    st.markdown(subtitle)
    #session_id = ReportThread.get_report_ctx().session_id
    artists = artistsload()
    session_state.randart = setart(artists, rando) #, session_id)
    session_state.randtitle = settitle(rando) #session_id)
    session_state.songtitle = st.text_input('Your Fake Song Title (Type in your own!):', value=session_state.randtitle).lower()
    session_state.artist = st.selectbox("in the style of: ", artists, session_state.randart)
    session_state.songfirstline = st.text_input('The First Line Of The Song (OPTIONAL):').lower()
    session_state.prompt = session_state.songtitle + "\nBY\n" + session_state.artist + "\n" + session_state.songfirstline
    display_side_panel_header("Rockbot!")
    st.sidebar.markdown("""
                        [Github](https://github.com/bigjoedata/rockbot)  
                        [Primary Model](https://huggingface.co/bigjoedata/rockbot)""")

    display_side_panel_header("Configuration")
    session_state.nsamples = st.sidebar.slider("Number of Songs To Generate: ", 1, 3, 1)
    session_state.length_gen = st.sidebar.select_slider(
    "Song Length (i.e., words parts) Note: This is set low for demo purposes; songs may end abruptly: ", [r * 64 for r in range(1, 7)], 192
    ) # Max is really 1024 with this model but set lower here to reduce computational intensity
    display_side_panel_header("Fine-Tuning")

    session_state.temperature = st.sidebar.slider("Choose temperature. Higher means more creative (crazier): ", 0.0, 3.0, 0.7, 0.1)
    session_state.topk = st.sidebar.slider("Choose Top K, the number of words considered at each step. Higher is more diverse; 0 means infinite:", 0, 50, 40)
    session_state.topp = st.sidebar.slider("Choose Top P. Limits next word choice to higher probability; lower allows more flexibility:", 0.0, 1.0, 0.9, 0.05)
    session_state.no_repeat_ngram_size = st.sidebar.slider("No Repeat N-Gram Size. Eliminates repeated phrases of N Length", 0, 6, 2)

    with st.spinner("Initial model loading, please be patient"):
        ai = load_aitextgen()

    if st.button('Generate My Songs!'):
        session_state.firstgen += 1
        with st.spinner("Generating songs, please be patient, this can take quite a while. If you adjust anything, you may need to start from scratch."):
            start = time.time()
            generated = generate_text(ai, session_state.prompt, session_state.nsamples, session_state.length_gen, session_state.temperature,
                session_state.topk, session_state.topp, session_state.no_repeat_ngram_size)
            end = time.time()
            session_state.ttg = str(round(end - start)) + "s"
        st.header("Your songs")
        sep = '<|endoftext|>'
        for gen in generated:
            session_state.gentext = f"{gen.split('<|endoftext|>', 1)[0]}".replace(
                    session_state.prompt, "**" + session_state.songtitle + " BY " + session_state.artist + "**\n" + session_state.songfirstline + " ").replace(
                    "\n","<br>") + "<hr>"
    if session_state.firstgen > 0:
        st.markdown(session_state.gentext,  unsafe_allow_html=True)
        st.markdown("⏲️ Time To Generate: " + session_state.ttg)
    else:
        pass

if __name__ == "__main__":
    main()