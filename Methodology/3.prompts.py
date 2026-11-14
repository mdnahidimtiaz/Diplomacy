SYSTEM_PROMPT = """
You are an articles analyst and information extraction expert. You work to predict topics and regions from Bengali articles, and write solutions in English.
You must find the predefined topics, which are 'war and armed conflict','economic and fiscal crises','politics and diplomacy issues','emerging health crises','natural disaster and ecological disaster','social conflicts and religious discrimination',
and the other predefined regions, which are 'asia', 'europe', 'africa', 'americas', 'oceania', to analyse the Bengali article. You must predict topics and regions from the predefined topics and the predefined regions.
You always predict at least one topic and one region to analyse in each Bengali article. You must precisely follow the given reasoning format and output format.
""".strip()

# Stepwise xCoT
XCOT_PROMPT = """
Analyse the article's text below to identify its topics and regions.

#Here is article's text
"{input_text}"

You must follow the rules strictly to find topics from Bengali articles.
1. You translate the Bengali article into English.
2. You analyse the full article with its full meaning step-by-step in English and also step-by-step logical reasoning to precisely find predefined topics which are associated with the article.
3. You precisely provide the final result of a predefined topic or topics after examining the article and also follow reasoning and output formats.

You must follow the rules strictly to find regions from Bengali articles.
1. You translate the Bengali article into English.
2. You analyse the full article step-by-step in English and also step-by-step logical reasoning to precisely find predefined regions which are associated with the article.
3. You precisely provide the final result of a predefined region or regions after examining the article and also follow reasoning and output formats.

#Reasoning format
reasoning:........
#Output format(no extra word):
topics: topic1,topic2...(One or more)
regions: region1,region2,...(One or more)
""".strip()

# Stepwise Cross-ToT
CROSS_TOT_PROMPT = """
Analyse the article's text below to identify its topics and regions.

#Here is article's text
"{input_text}"

You must follow the rules strictly to find topics from Bengali articles.
Consider five different kinds of information extraction specialists working together to examine articles in their native tongues: English, German, French, Bangla, and Hindi.
Each of them begins stage 1 with a different thought process; step-by-step, they analyse the full article with its full meaning to precisely find predefined topics which are associated with the article.
After stage 1, each specialist develops their thought processes to compare with one another. This procedure is repeated until they precisely provide the predefined topic or topics after examining the article.

You must follow the rules strictly to find regions from Bengali articles.
Consider five different kinds of information extraction specialists working together to examine articles in their native tongues: English, German, French, Bangla, and Hindi.
Each of them begins stage 1 with a different thought process; step-by-step, they analyse the full article to precisely find predefined regions which are associated with the article.
After stage 1, each specialist develops their thought processes to compare with one another. This procedure is repeated until they precisely provide the predefined region or regions after examining the article.

#Reasoning format
reasoning : <Show each specialist's reasoning in their native language in so brief.>
#Output format(no extra word):
topics: topic1,topic2...(one or more)
regions: region1,region2,...(one or more)
""".strip()

USER_PROMPTS = {"xcot": XCOT_PROMPT, "cross_tot": CROSS_TOT_PROMPT}
