rel_proj_prompt = """
Please score the below %s relations based on similarity to '%s'. Each score is in [0, 1], the sum of all scores is 1.
%s
Your answer is:
"""

rel_proj_prompt0 = """
set A={%s}.
Please score the relations in set A based on similarity to %s.
your answer should follow these rules:
1) Example of answer: "1. spouse_of (Score: 0.3)\n2. lives_with (Score: 0.2)\n"
2) Each relation's score should be within the [0,1] range, The sum of all scores should be 1.
3) You're forbidden to answer any relation that is not in set A.
4) Your answer should mainly contain relation and score instead of explanation. Keep it short.
5) Max relation number in your answer is %s.
Your answer is:
"""

rel_proj_question = "Which entities are connected to %s by relation %s?"


extract_relation_prompt = """ Please retrieve at most %s relations (separated by semicolon) that contribute to the question Q and rate their contribution on a scale from 0 to 1 (the sum of the scores of %s relations is 1).
Please mark that your answer should follow these rules: 
1) Each relation's score should be within the [0,1] range;
2) The sum of these %s scores should be 1; 
3) The output should follow this format: {1. relation_name (Score: number)\n2. relation_name (Score: number)\n}. For example, {1. concept:professionusestool (Score: 0.3)\n2. concept:coachwontrophy (Score: 0.2)\n}".
Q: %s
Relations: %s.
%s
Your answer is:
"""
extract_relation_prompt2 = """ Please retrieve at most %s relations from R that contribute to the question Q and rate their contribution on a scale from 0 to 1.
your answer should follow the format of this example: "{1. spouse_of (Score: 0.3)\n2. lives_with (Score: 0.2)\n}". 
Q: %s
R: %s.
Your answer is:
"""

extract_relation_prompt1 = """ Please retrieve at most %s relations (separated by semicolon) that contribute to the question Q and rate their contribution on a scale from 0 to 1.
Please mark that your answer should follow these rules: 
1) Each relation's score should be within the [0,1] range;
2) The sum of these scores should be 1; 
3) The output should follow this format: {1. relation_name (Score: number)\n2. relation_name (Score: number)\n}. For example, {1. concept:professionusestool (Score: 0.3)\n2. concept:coachwontrophy (Score: 0.2)\n}".
Q: %s
Relations: %s.
Your answer is:
"""

extract_relation_prompt0 = """Please retrieve %s relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of %s relations is 1) after you see an example.
Here is the example. Example Q: Name the president of the country whose main spoken language was Brahui in 1980?
Example relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
Example A: 1. {language.human_language.main_country (Score: 0.4))}: This relation is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980.
2. {language.human_language.countries_spoken_in (Score: 0.3)}: This relation is also relevant as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president.
3. {base.rosetta.languoid.parent (Score: 0.2)}: This relation is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question.

Now please finish the task. Q: """



