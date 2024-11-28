from transformers import pipeline

# Charger le pipeline de summarization avec le modèle choisi
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Texte à résumer
text = """
The Eiffel Tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, 
and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. 
It was the first structure to reach a height of 300 metres. Excluding transmitters, 
the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.
"""

# Résumé
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print("Résumé :", summary[0]['summary_text'])