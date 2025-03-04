import spacy
import requests

nlp = spacy.load("en_core_web_sm")

def extract_meaningful_concepts(text):
    doc = nlp(text)
    return [token.text.lower() for token in doc 
            if token.pos_ in {"NOUN", "PROPN", "VERB"} and not token.is_stop]
    
    
def get_conceptnet_edges(word):
    url = f"http://api.conceptnet.io/c/en/{word}"
    response = requests.get(url).json()
    return len(response.get('edges', []))  # Count relations


def select_richest_concepts(text, top_n=3):
    concepts = extract_meaningful_concepts(text)
    if (top_n is None) or (top_n > len(concepts)):
        top_n = len(concepts)
    concept_scores = {word: get_conceptnet_edges(word) for word in concepts}
    sorted_list = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_concepts, top_scores = [item[0] for item in sorted_list], [item[1] for item in sorted_list]
    return top_concepts, top_scores
# Output: [('processing', 120), ('Transformer', 80), ('revolutionized', 50)]


def get_main_concept(text):
    doc = nlp(text)
    candidate_concepts = {token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN"}}
    
    concept_scores = {word: get_conceptnet_edges(word) for word in candidate_concepts}
    main_concept = max(concept_scores, key=concept_scores.get)  # Concept with max relations

    return main_concept, concept_scores[main_concept]


### function to test whether the final concepts are equivalent to the tokens produced by the tokenizer.

if __name__ =='__main__':
    prompt = "Describe the ideal qualities of a leader in a team."
    input_concepts = select_richest_concepts(prompt, top_n=5)
    print("input_concepts:", input_concepts)
    