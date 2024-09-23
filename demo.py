import ollama
import re
from collections import defaultdict

class LLM:
    def __init__(self, model_name="llama2:70b"):
        self.model = model_name

    def run_llm(self, prompt):
        response = ollama.generate(model=self.model, prompt=prompt)["response"]
        return response
        

    def format_llm_res(self, res:str):
        pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
        relations_with_scores = defaultdict()
        for match in re.finditer(pattern, res):
            relation = match.group("relation").strip()
            if ';' in relation:
                continue
            score = match.group("score")
            if not relation or not score:
                return False, "output uncompleted.."
            try:
                score = float(score)
            except ValueError:
                return False, "Invalid score"
            relations_with_scores[relation] = score

        if not relations_with_scores:
            return False, "No relations found"
        
        return True, relations_with_scores
if __name__ == "__main__":
    rel_proj_question = "Which entities are connected to %s by relation %s?"

    extract_relation_prompt = """Please retrieve %s relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of %s relations is 1).
    Q: Name the president of the country whose main spoken language was Brahui in 1980?
    Topic Entity: Brahui Language
    Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
    A: 1. {language.human_language.main_country (Score: 0.4))}: This relation is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980.
    2. {language.human_language.countries_spoken_in (Score: 0.3)}: This relation is also relevant as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president.
    3. {base.rosetta.languoid.parent (Score: 0.2)}: This relation is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question.

    Q: """
    q = rel_proj_question % ("USA", "capital_of")
    p = extract_relation_prompt % (2, 2) + q + "\nRelations: capital_of, title_of, son_of, city_of, type_of." + "\nA: "
    llm = LLM()
    res = llm.run_llm(p)
    print(res)
