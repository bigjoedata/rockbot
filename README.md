🎹 🪘 🎷 🎺 🪗  🪕 🎻
## Rockbot Background
Two of my passions are music and data! I realized I had a bounty of metadata from artists I've listened to over the past several years and I decided to take advantage to build something fun. I scraped the top 50 lyrics for artists I'd listened to at least once from [Genius](https://genius.com/), added some other selected top artists, did a ton of post-processing and trained a [GPT-2's](https://openai.com/blog/better-language-models/) based model from scratch using the [AITextGen](https://github.com/minimaxir/aitextgen) framework. The UI / back end is built in [Streamlit](https://www.streamlit.io/) The vocabulary was built from scratch, rather than fine-tuned off an existing model. I also fine-tuned a GPT-2 based model available [here](https://huggingface.co/bigjoedata/rockbot) but this model weighs in at a fraction of the size.

A demo is available [here](https://share.streamlit.io/bigjoedata/rockbot/main/src/main.py) Generation is resource intense and can be slow in the demo. I set governors on song length to keep generation time somewhat reasonable. You may adjust song length and other parameters on the left or check out [Github](https://github.com/bigjoedata/rockbot) to spin up your own Rockbot.

Data Prep Cleaning Notes:
- Removed duplicate lyrics from each song
- Deduped similar songs based on overall similarity to remove cover versions
- Removed as much noise / junk as possible. There is still some.
- Added tokens to delineate song
- Used language to remove non-English versions of songs
- Many others!

### Tech Stack and technical notes

 - [Python](https://www.python.org/). 
 - [Streamlit](https://www.streamlit.io/). 
 - [GPT-2](https://openai.com/blog/better-language-models/). 
 - [AITextGen](https://github.com/minimaxir/aitextgen).
 - [LyricsGenius](https://lyricsgenius.readthedocs.io/en/master/)   (retrieving lyrics for training).
 - [Knime](https://www.knime.com/) (data cleaning and post processing)
 - [GPT-2 generation](https://huggingface.co/blog/how-to-generate)

## How to Use The Model
Please refer to [AITextGen](https://github.com/minimaxir/aitextgen) and [Huggingface](https://huggingface.co/) for much better documentation.

    Generate With Prompt (Use lower case for Song Name, First Line):
    Song Name
    BY
    Artist Name (Use unmodified from [Github](https://github.com/bigjoedata/rockbot/blob/main/theartists.parquet)
    Beginning of song
 
## Spin up your own with Docker
Running your own is very easy. Visit my [Streamlit-Plus repository](https://github.com/bigjoedata/streamlit-plus) for more details on the image build

 - Install [Docker Compose](https://docs.docker.com/compose/install/)
 - Follow the following steps
```
git clone https://github.com/bigjoedata/rockbot
cd rockbot
nano docker-compose.yml # Edit environmental variables for max song length and max songs to generate to match your computing power (higher is more resource intensive)
docker-compose up -d # launch in daemon (background) mode
```