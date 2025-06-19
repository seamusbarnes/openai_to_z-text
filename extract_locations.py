#!/usr/bin/env python
# coding: utf-8


"""
# Extract and plot latitude/longitude coordinates of locations extracted from text

## Data
Book title: *Up the Amazon and Madeira rivers, through Bolivia and Peru*
Author: *Edward D. Mathews*
Publication date: *1879*

## Terms
- `region of interest` (`roi`): The areas of Peru, Bolivia and Brazil (exlcuding other references that appear in the text to locations e.g. London, China, India)
- `text_chronological`: The chronological appearence of locations in the text
- `journey_chronological`: The chronologcical appearance of locations during the exploration
- `sentence_block`: <= 5 sentence block centred on the possible location reference in the text

## Pipeline
1. Extract and basic clean text
2. Perform NLP on text
3. Identify possible references to locations in the NLP'ed text
4. Fuzzy match possible locations and correct
4. Create dataset of chronologically (in terms of the text) ordered locations with their `sentence_blocks`.
5. Do some fancy opneai api call thingy on the `sentence_blocks`, to see get structured data from the blocks:
    - Is this a real location, or did the NLP make a mistake?
    - Is this location in the region of interest `roi`?
    - Is this location `journey_chronological`?
6. Plot the real locations in the `roi` on a satellite/cartoon map of Brazil using folium
"""


import os
import numpy as np
import pandas as pd
import re
import json

from collections import Counter

import spacy
import nltk
from nltk.tokenize import sent_tokenize

from rapidfuzz import fuzz

import folium
import matplotlib.pyplot as plt

import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut


# Download punkt tokenizer if not already done
nltk.download('punkt', quiet=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

#####
# FILE PATH SETUP (adjust as needed)
TEXT_FILENAME = "Up the Amazon and Madeira Rivers through Bolivia and Peru copy.txt"
COORDS_FILENAME = "coords.json"

CWD = os.getcwd()

PATH_TO_RAW_DATA_DIR = os.path.join(CWD, "data", "raw")
PATH_TO_PROCESSED_DATA_DIR = os.path.join(CWD, "data", "processed")

PATH_TO_DATA = os.path.join(PATH_TO_RAW_DATA_DIR, TEXT_FILENAME)
PATH_TO_COORDS_JSON = os.path.join(PATH_TO_PROCESSED_DATA_DIR, COORDS_FILENAME)

# -- Read the input text
with open(PATH_TO_DATA, 'r') as file:
    text = file.read().replace('\n', ' ')

#####
# Build a list of words and their character offsets
matches = list(re.finditer(r'\S+', text))
words = [m.group(0) for m in matches]
word_starts = [m.start() for m in matches]
word_ends = [m.end() for m in matches]

#####
# Helper function: Find word index for a char position
def find_word_idx(char_idx, starts):
    for i, s in enumerate(starts):
        if char_idx < s:
            return max(i-1, 0)
    return len(starts)-1

#####
# Run spaCy NER
doc = nlp(text)

# Extract (location_text, start_char, end_char) for locations
location_spans = [
    (ent.text, ent.start_char, ent.end_char)
    for ent in doc.ents
    if ent.label_ in ("GPE", "LOC")
]

#####
# Build blocks (word windowed, guaranteed to contain full location)
window = 30 * 2  # words before and after

blocks = []
for loc_text, start_char, end_char in location_spans:
    start_word_idx = find_word_idx(start_char, word_starts)
    end_word_idx = find_word_idx(end_char-1, word_starts)
    w0 = max(start_word_idx - window, 0)
    w1 = min(end_word_idx + window + 1, len(words))  # +1 so slice includes last word

    block_words = words[w0:w1]
    # For possible highlighting, join the entity words
    entity_words = words[start_word_idx:end_word_idx+1]
    entity_str = ' '.join(entity_words)

    # Optional: mark the location in the block
    sentence_block = ' '.join(block_words)
    # Try to only mark first occurrence so as not to mess with repeated mentions in block
    sentence_block = sentence_block.replace(entity_str, f"<<{entity_str}>>", 1)

    blocks.append({
        'location': loc_text,
        'sentence_block': sentence_block,
        'start_word_index': start_word_idx,
        'end_word_index': end_word_idx,
        'location_start_char': start_char,
        'location_end_char': end_char
    })

df_blocks = pd.DataFrame(blocks)


SIMILARITY_THRESHOLD = 90  # tune as needed

# 1. Gather unique location names
all_locations = [block['location'] for block in blocks]

clusters = []
canonical_names = []
for loc in all_locations:
    matched = False
    for i, canon in enumerate(canonical_names):
        score = fuzz.token_sort_ratio(loc, canon)
        if score >= SIMILARITY_THRESHOLD:
            clusters[i].append(loc)
            matched = True
            break
    if not matched:
        canonical_names.append(loc)
        clusters.append([loc])

# 2. For each cluster, pick a canonical name (most common)
for i, c in enumerate(clusters):
    most_common = Counter(c).most_common(1)[0][0]
    canonical_names[i] = most_common

# --- STEP 3: Build your mapping and annotate blocks afterwards
location_to_canon = {}
for canon, members in zip(canonical_names, clusters):
    for m in members:
        location_to_canon[m] = canon

# Assign canonical_location to each block
for block in blocks:
    block['canonical_location'] = location_to_canon[block['location']]


locations = [block["canonical_location"] for block in blocks]
print(len(blocks))
print(len(locations))
print(len(set(locations)))


# Example output: Print a few blocks
for i in range(len(df_blocks)):  # adjust as you wish
    row = df_blocks.iloc[i]
    if row["location"] == "Cachimayo":
        print('-'*60)
        print("Location:", row["location"])
        print("Block:")
        print(row["sentence_block"])
        print(" ")
# To save: df_blocks.to_csv('location_blocks.csv')


def get_location_coords(location_set, verbose=False):
    # Filter out None or empty/whitespace locations early
    location_set = set(filter(lambda x: x and x.strip(), location_set))

    geolocator = Nominatim(user_agent="generic_name")
    location_coord_dict = {}
    n_locations = len(location_set)

    t0 = time.perf_counter()
    for count, location in enumerate(location_set, start=1):
        t1 = time.perf_counter()
        print(f"Getting coords for '{location}'; {count}/{n_locations}")
        try:
            loc = geolocator.geocode(location)
            if loc is not None:
                coords = (loc.latitude, loc.longitude)
                print(f"  {location}: {coords}")
                location_coord_dict[location] = coords
            else:
                print(f"  Location not found: {location}")
        except GeocoderUnavailable as e:
            if verbose:
                print(f"  Geocoder is unavailable: {e}")
            else:
                print("  Geocoder is unavailable")
        except GeocoderTimedOut as e:
            if verbose:
                print(f"  Geocoder timed out: {e}")
            else:
                print("  Geocoder timed out")
        except AttributeError as e:
            if verbose:
                print(f"  Attribute error (possibly bad result) for '{location}': {e}")
            else:
                print("  Attribute error (possibly bad result) for '{location}'")
                
        t_elapsed = time.perf_counter() - t1
        total_elapsed = time.perf_counter() - t0
        print(f"  Time for this location: {t_elapsed:.2f} s\n")
        print(f"  Runtime so far: {total_elapsed:.2f} s\n")

    t_total = time.perf_counter() - t0
    print(f"Total time for all locations: {t_total:.2f} s")
    return location_coord_dict

coords_dict = get_location_coords(set(locations))


# row = df_blocks[df_blocks["location"]=="shaly"]
# sentence = row["sentence_block"]
# print(sentence)


with open(PATH_TO_COORDS_JSON, "w") as f:
    json.dump(coords_dict, f, indent=4)


# Now, create a Folium map centered somewhere (e.g., the average of your lat/lon)
if coords_dict:
    lats = [coords[0] for coords in coords_dict.values()]
    lons = [coords[1] for coords in coords_dict.values()]
    avg_lat = sum(lats) / len(lats)
    avg_lon = sum(lons) / len(lons)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=2)
else:
    m = folium.Map(location=[0, 0], zoom_start=2)  # fallback

# Add markers
for name, coords in coords_dict.items():
    folium.Marker(
        location=coords,
        popup=name,
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

m


