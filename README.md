# openai_to_z-text

## Data to extract from text snippets

### Data Quality and Enrichment

1. Classify Each Place Type: “Is ‘{place}’ from the following context a city, river, region, infrastructure project, or something else?”
2. Resolve Ambiguity and Validate Locations: Let the model help spot false positives or ambiguous place names. “Does the following refer to a real place in South America, or something else? Place: {place}. Context: {snippet}”
3. Enhance Data with Additional Info. Extract date/time, mode of travel, or companions: “Extract the date (if any), means of transport, and any people mentioned in this passage. Passage: {snippet}”

### Thematic Tagging and Annotation

1. Sentiment and Experience Analysis: For each snippet, classify the author’s tone: positive, negative, curious, afraid, bored? “Determine the author’s sentiment toward {place} in the following passage: {snippet}”
2. Automatic Thematic Coding: Classify snippet themes (e.g. nature, danger, disease, commerce, indigenous, technical, travel hardship, etc.)
3. “Which of these themes best describes the main topic of the following passage? [List: travel, exploration, trade, health, wildlife, engineering, conflict, etc.] Passage: {snippet}”

### Summarization and Higher-level Structure

1. Segment Summaries: For each clustered group of snippets/geographic region/period, ask for mini-summaries. “Summarize the author’s experiences with {place} as described in these passages: {compile all related snippets here}”
2. Extracted Journeys/Routes: Infer and summarize the author’s overall route: “Given these passages in chronological order, reconstruct the sequence of places visited and summarize the main journey segments.”
3. Extract Key Events: “Identify any major events described in the following text: {snippet} (e.g. illness, accident, meeting, technical challenge).”

### Analytical Questions

1. Compare Places: “How does the author’s tone or experiences differ between {place A} and {place B}?”
2. Entity Linking or Event Extraction: “List all people and organizations mentioned in conjunction with {place}. What are their roles or significance?” 5. Map-Driven or Timeline Insights
3. For each marker or region on your Folium map, generate a tooltip summary: “Create a short informative blurb about {place} the author visited, based on the following text: {snippet}”

### Connecting to Broader Knowledge

1. Enrich your data by connecting place mentions to Wikipedia/Gazetteers—ask GPT for modern equivalents, alternate names, or notable facts.
