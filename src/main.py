import pandas as pd
import streamlit as st
from aitextgen import aitextgen
import random

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

@st.cache(allow_output_mutation=True, ttl=120000, max_entries=1)
def load_aitextgen():
    # return aitextgen(model="bigjoedata/rockbot") # This is fine-tuned on the 355M token GPT-2 Model
    return aitextgen(model="bigjoedata/rockbot-distilgpt2") # This is 60% lighter due to being fine-tuned on the reduced Huggingface distilgpt2 Model

@st.cache
def artistsload():
    df=pd.read_parquet('theartists.parquet')
    df = df.iloc[:, 0].apply(lambda x: x.upper())
    return df

@st.cache(ttl=1200, max_entries=1)
def setseeds(df):
    randart=random.randint(0, len(df))
    sampletitles=[
        'Love Is A Vampire',
        'The Cards Are Against Humanity',
        'My Grandmother Likes My Music',
        'Call A Doctor, It Is Urgent',
        'So, That Just Happened',
        'Dogs Versus Cats',
        'Entropy Is Overrated',
        'I Believe That Is Butter',
        'Panic In The Grocery Store'
    ]
    randtitle = random.choice(sampletitles)
    return randart, randtitle

def generate_text(ai, prefix, nsamples, length_gen, temperature, topk, topp, no_repeat_ngram_size):
    nsamples = min(nsamples, 5)
    return ai.generate(
        n=nsamples,
        batch_size=nsamples, # disabled to reduce memory usage
        prompt=prefix,
        max_length=length_gen,
        temperature=temperature,
        #top_k=topk,
        top_p=topp,
        #repetition_penalty=4.0, # disabled to reduce CPU/GPU usage
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=True, 
        return_as_list=True,
        eos_token = "<|endoftext|>",
        bos_token = "<|startoftext|>",
        #num_workers=1 # This parameter doesn't seem to affect CPU usage for generation
    )

def main():
    main_txt = """🎸 🥁 Rockbot 🎤 🎧"""
    sub_txt = ""
    subtitle = """
            A [DistilGPT-2](https://huggingface.co/distilgpt2) based lyrics generator fine-tuned on the writing styles of 16000 songs by 270 artists across MANY genres (not just rock).

            **Instructions:** Type in a fake song title, pick an artist, click "Generate".

            Note: Due to the nature of language models, lyrics bleed across artists and you may see NSFW lyrics unexpectedly (e.g., from The Beatles), especially if you change the configuration to allow more entropy. I have made no attempt to censor lyrics whatsoever.

            Finally, these lyrics are computer generated. Not all of these will be non-repetitive and/or coherent. Just have fun.

            [Github Repository](https://github.com/bigjoedata/rockbot)

            [GPT-2 124M version Model page on Hugging Face](https://huggingface.co/bigjoedata/rockbot)

            [DistilGPT2 version Model page on Hugging Face](https://huggingface.co/bigjoedata/rockbot-distilgpt2/)
        """
    display_app_header(main_txt,sub_txt,is_sidebar = False)
    st.markdown(subtitle)

    artists = artistsload()
    randart, randtitle = setseeds(artists)
    
    songtitle = st.text_input('Your Fake Song Title (Type in your own!):', value=randtitle).upper()
    artist = st.selectbox("in the style of: ", artists, randart)

    prompt = songtitle.title() + "\nBY\n" + artist.title() + "\n"

    with st.spinner("Initial model loading, please be patient"):
        ai = load_aitextgen()
    display_side_panel_header("Configuration")
    nsamples = st.sidebar.slider("Number of Songs To Generate: ", 1, 10, 5)
    length_gen = st.sidebar.select_slider(
        "Song Length (i.e., words/word-pairs) Caution: Larger lengths slow generation considerably: ", [r * 64 for r in range(1, 9)], 256
    ) # Max is really 1024 with this model but set at 512 here to reduce max memory consumption
    display_side_panel_header("Fine-Tuning")
    temperature = st.sidebar.slider("Choose temperature. Higher means more creative (crazier): ", 0.0, 1.0, 0.7, 0.1)
    topk = st.sidebar.slider("Choose Top K. Limits next word choice to top k guesses; higher is more random:", 0, 50, 40)
    topp = st.sidebar.slider("Choose Top P. Limits next word choice to higher probability; lower is more random:", 0.0, 1.0, 0.9, 0.05)
    no_repeat_ngram_size = st.sidebar.slider("No Repeat N-Gram Size. Eliminates repeated phrases of N Length", 0, 6, 3)

    if st.button('Generate My Songs!'):
        with st.spinner("Generating songs, please be patient, this can take a while..."):
            generated = generate_text(ai, prompt, nsamples, length_gen, temperature, topk, topp, no_repeat_ngram_size)
            st.balloons()

        st.header("Your songs")

        sep = '<|endoftext|>'

        for gen in generated:
            gentext = f"{gen.split(sep, 1)[0]}".replace(prompt, "**" + songtitle.upper() + " BY " + artist.upper() + "**\n").replace("\n","<br>") + "<hr>"
            st.markdown(gentext,  unsafe_allow_html=True)

if __name__ == "__main__":
    main()