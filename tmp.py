from llm import LLM_vllm
llm = LLM_vllm("llama3:8b")
prompt = """question=Which entities are connected to concept_televisionstation_ketc by relation concept:agentcontrols_reverse?
candidate_answers=[concept_company_pbs, concept_museum_pbs, concept_website_new_york_american]"""
res = llm.run(prompt, "score")
print("tst")
