# 🎸 🥁 Rockbot 🎤 🎧 
A [GPT-2](https://huggingface.co/blog/how-to-generate) based lyrics generator fine-tuned on the writing styles of 16000 songs by 270 artists across MANY genres (not just rock).

**Instructions:** Type in a fake song title, pick an artist, and fake lyrics will be generated. The generator will be pre-seeded with a random title & artist initially. 

Note: Due to the nature of language models, lyrics bleed across artists and you may see NSFW lyrics unexpectedly (e.g., from The Beatles). I have made no attempt to censor lyrics whatsoever.

Finally, these lyrics are computer generated. Not all of these will be non-repetitive and/or coherent. Just have fun.

[Repository](https://github.com/bigjoedata/rockbot)
[Model page on Hugging Face](https://huggingface.co/bigjoedata/rockbot)
🎹 🪘 🎷 🎺 🪗  🪕 🎻
## Background
With the shutdown of [Google Play Music](https://en.wikipedia.org/wiki/Google_Play_Music) I used Google's takeout function to gather the metadata from artists I've listened to over the past several years. I wanted to take advantage of this bounty to build something fun. I scraped the top 50 lyrics for artists I'd listened to at least once from [Genius](https://genius.com/), then fine tuned [GPT-2's](https://openai.com/blog/better-language-models/) 124M token model using the [AITextGen](https://github.com/minimaxir/aitextgen) framework after considerable post-processing.

### Full Tech Stack
[Google Play Music (R.I.P.)](https://en.wikipedia.org/wiki/Google_Play_Music)
[Python](https://www.python.org/)
[Streamlit](https://www.streamlit.io/)
[GPT-2](https://openai.com/blog/better-language-models/)
[AITextGen](https://github.com/minimaxir/aitextgen)
[Pandas](https://pandas.pydata.org/)
[LyricsGenius](https://lyricsgenius.readthedocs.io/en/master/)
[Google Colab](https://colab.research.google.com/) (GPU based Training)
[Knime](https://www.knime.com/) (data cleaning)


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
 
