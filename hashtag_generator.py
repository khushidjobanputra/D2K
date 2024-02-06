from rake_nltk import Rake
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TagGenerator:
    def __init__(self):
        nltk.download('punkt')
        self.rake = Rake()
        self.tokenizer = AutoTokenizer.from_pretrained("fabiochiu/t5-base-tag-generation")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("fabiochiu/t5-base-tag-generation")

    def extract_keywords(self, text):
        self.rake.extract_keywords_from_text(text)
        ranked_list = self.rake.get_ranked_phrases_with_scores()
        keyword_list = []
        for keyword in ranked_list:
            keyword_updated = keyword[1].split()
            keyword_updated_string = " ".join(keyword_updated[:2])
            keyword_list.append(keyword_updated_string)
            if len(keyword_list) > 9:
                break
        return keyword_list

    def generate_tags(self, keyword_list):
        abcd = " ".join(map(str, keyword_list))
        inputs = self.tokenizer([abcd], max_length=512, truncation=True, return_tensors="pt")
        output = self.model.generate(**inputs, num_beams=8, do_sample=True, min_length=10,
                                     max_length=64)
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        tags = list(set(decoded_output.strip().split(", ")))
        return tags

# Example usage:
my_text = """
Python is a high-level, interpreted, general-purpose programming language. Its
design philosophy emphasizes code readability with the use of significant
indentation. Python is dynamically-typed and garbage-collected.
"""

tag_generator = TagGenerator()
keywords = tag_generator.extract_keywords(my_text)

tags = tag_generator.generate_tags(keywords)

print(tags)