# https://huggingface.co/blog/how-to-generate

import pandas as pd
import streamlit as st
from aitextgen import aitextgen
import random

def showhelpselect():
    showhelp = st.sidebar.checkbox(' ℹ Show Help')
    if showhelp:
        st.sidebar.info('Help appears in these blue boxes')
    return showhelp

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

@st.cache(allow_output_mutation=True, ttl=600, max_entries=1)
def load_aitextgen():
    return aitextgen(model="bigjoedata/rockbot")

@st.cache
def artistsload():
    df=pd.read_parquet('theartists.parquet')
    randart=random.randint(0, len(df))
    #df = df.iloc[:, 0].apply(lambda x: x.title())
    #df = df.iloc[1:].reset_index(drop=True)
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
    return df, randart, randtitle

def generate_text(ai, prefix, nsamples, length_gen, temperature, topk, topp, no_repeat_ngram_size):
    return ai.generate(
        n=nsamples,
        #batch_size=nsamples, # disabled to reduce memory usage
        prompt=prefix,
        max_length=length_gen,
        temperature=temperature,
        #top_k=topk,
        top_p=topp,
        #repetition_penalty=4.0, # disabled to reduce memory usage
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=True, 
        return_as_list=True,
        eos_token = "<|endoftext|>",
        bos_token = "<|startoftext|>"
    )



def main():
    main_txt = """🎸 🥁 Rockbot 🎤 🎧"""
    sub_txt = ""
    subtitle = """
        **Instructions:** Type in a fake song title, pick an artist, click "Generate".
        
        Note: Due to the nature of language models, lyrics bleed across artists and you may see NSFW lyrics unexpectedly (e.g., from The Beatles), especially if you change the configuration to allow more entropy. I have made no attempt to censor lyrics whatsoever.

        Finally, these lyrics are computer generated. Not all of these will be non-repetitive and/or coherent. Just have fun.

        [Repository](https://github.com/bigjoedata/rockbot)

        [Model page on Hugging Face](https://huggingface.co/bigjoedata/rockbot)

        🎹 🪘 🎷 🎺 🪗  🪕 🎻
        """
    display_app_header(main_txt,sub_txt,is_sidebar = False)
    st.markdown(subtitle)

    artists, randart, randtitle = artistsload()
    songtitle = st.text_input('Your Fake Song Title (Type in your own!):', value=randtitle).title()
    #songtitle = songtitle.split()[:4]

    artist = st.selectbox("in the style of: ", artists, randart)

    prompt = songtitle + "\nBY\n" + artist + "\n"
    yoursong = songtitle + " BY " + artist

    #st.text(prompt)

    with st.spinner("Initial model loading, please be patient"):
        ai = load_aitextgen()
    display_side_panel_header("Configuration")
    #st.sidebar.subheader("Configuration")
    nsamples = st.sidebar.number_input("Number of Songs To Generate: ", 1, 10, 5)
    #st.sidebar.info("Number of Songs To Generate")
    length_gen = st.sidebar.select_slider(
        "Song Length (i.e., words/word-pairs) Caution: Larger lengths slow generation considerably: ", [r * 64 for r in range(1, 9)], 256
    ) # Max is really 1024 with this model but set at 512 here to reduce max memory consumption
    display_side_panel_header("Fine-Tuning")
    temperature = st.sidebar.slider("Choose temperature. Higher means more creative (crazier): ", 0.0, 1.0, 0.8, 0.1)
    topk = st.sidebar.slider("Choose top_k. Limits next word choice to top k guesses; higher is more random:", 0, 50, 25)
    topp = st.sidebar.slider("Choose top_p. Limits next word choice to higher probability; lower is more random:", 0.0, 1.0, 0.9, 0.05)
    no_repeat_ngram_size = st.sidebar.slider("No Repeat N-Gram Size. Eliminates repeated phrases of N Length", 0, 6, 3)

    if st.button('Generate My Songs!'):
        with st.spinner("Generating songs, please be patient, this can take a minute or two..."):
            generated = generate_text(ai, prompt, nsamples, length_gen, temperature, topk, topp, no_repeat_ngram_size)
            st.balloons()

        st.header("Your songs")

        sep = '<|endoftext|>'

        for gen in generated:
            gentext = f"{gen.split(sep, 1)[0]}".replace(prompt, "**" + songtitle + " BY " + artist + "**\n").replace("\n","<br>") + "<hr>"
            st.markdown(gentext,  unsafe_allow_html=True)

if __name__ == "__main__":
    main()