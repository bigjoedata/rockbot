
# 🎸 🥁 Rockbot 🎤 🎧 
A [GPT-2](https://openai.com/blog/better-language-models/) based lyrics generator fine-tuned on the writing styles of 16000 songs by 270 artists across MANY genres (not just rock).

**Instructions:** Type in a fake song title, pick an artist, click "Generate".

Most language models are imprecise and Rockbot is no exception. You may see NSFW lyrics unexpectedly. I have made no attempts to censor. Generated lyrics may be repetitive and/or incoherent at times, but hopefully you'll encounter something interesting or memorable.

Oh, and generation is resource intense and can be slow. I set governors on song length to keep generation time somewhat reasonable. You may adjust song length and other parameters on the left or check out [Github](https://github.com/bigjoedata/rockbot) to spin up your own Rockbot.

Just have fun.

[Demo](https://share.streamlit.io/bigjoedata/rockbot/main/src/main.py) Adjust settings to increase speed

[Github](https://github.com/bigjoedata/rockbot)

[GPT-2 124M version Model page on Hugging Face](https://huggingface.co/bigjoedata/rockbot)

[DistilGPT2 version Model page on Hugging Face](https://huggingface.co/bigjoedata/rockbot-distilgpt2/) This is leaner with the tradeoff being that the lyrics are more simplistic.

🎹 🪘 🎷 🎺 🪗  🪕 🎻
## Background
With the shutdown of [Google Play Music](https://en.wikipedia.org/wiki/Google_Play_Music) I used Google's takeout function to gather the metadata from artists I've listened to over the past several years. I wanted to take advantage of this bounty to build something fun. I scraped the top 50 lyrics for artists I'd listened to at least once from [Genius](https://genius.com/), then fine tuned [GPT-2's](https://openai.com/blog/better-language-models/) 124M token model using the [AITextGen](https://github.com/minimaxir/aitextgen) framework after considerable post-processing. For more on generation, see [here.](https://huggingface.co/blog/how-to-generate)

### Full Tech Stack
[Google Play Music](https://en.wikipedia.org/wiki/Google_Play_Music)  (R.I.P.). 
[Python](https://www.python.org/). 
[Streamlit](https://www.streamlit.io/). 
[GPT-2](https://openai.com/blog/better-language-models/). 
[AITextGen](https://github.com/minimaxir/aitextgen). 
[Pandas](https://pandas.pydata.org/). 
[LyricsGenius](https://lyricsgenius.readthedocs.io/en/master/). 
[Google Colab](https://colab.research.google.com/) (GPU based Training). 
[Knime](https://www.knime.com/) (data cleaning). 


## How to Use The Model
Please refer to [AITextGen](https://github.com/minimaxir/aitextgen) for much better documentation.

### Training Parameters Used

    ai.train("lyrics.txt",
             line_by_line=False,
             from_cache=False,
             num_steps=10000,
             generate_every=2000,
             save_every=2000,
             save_gdrive=False,
             learning_rate=1e-3,
             batch_size=3,
             eos_token="<|endoftext|>",
             #fp16=True
             )
###  To Use


    Generate With Prompt (Use Title Case):
    Song Name
    BY
    Artist Name
 
