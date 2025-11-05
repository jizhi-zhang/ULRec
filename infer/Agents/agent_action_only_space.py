import re, string, os
from typing import List, Union, Literal
from enum import Enum
import random
import tiktoken
from langchain.llms.base import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain.agents.react.base import DocstoreExplorer
from langchain.docstore.base import Docstore
from langchain.prompts import PromptTemplate
from Agents.llm import AnyOpenAILLM
from Agents.prompts import reflect_prompt, action_only_agent_prompt, react_reflect_agent_prompt, REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER
from Agents.prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt, COT_INSTRUCTION, COT_REFLECT_INSTRUCTION
from Agents.fewshots_action_only import WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT, WEBTHINK_SIMPLE7, WEBTHINK_SIMPLE8, WEBTHINK_SIMPLE9, WEBTHINK_SIMPLE10, WEBTHINK_SIMPLE11
from collections import defaultdict


GENRE_MOVIE = ['Animation', "Children's", 'Comedy', 'Adventure', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Sci-Fi', 'Documentary', 'War', 'Musical', 'Mystery', 'Film-Noir', 'Western']
GENRE_STEAM = ['Accounting', 'Simulation', 'Adventure', 'Free to Play', 'Massively Multiplayer', 'Sports', 'Racing', 'Indie', 'Software Training', 'Early Access', 'Strategy', 'RPG', 'Action', 'Education', 'Casual']
GENRE_AMAZON = ['Self-Esteem', 'Gaming', 'British & Irish', 'Business Technology', 'Graphic Design', 'Television', 'Words, Language & Grammar', 'Antiques & Collectibles', 'Business & Money', 'Evolution', 'Mountaineering', 'Education & Reference', 'Outdoor Cooking', 'Personal Transformation', 'History', 'Historical Study & Educational Resources', 'Middle East', 'Memoirs', 'Northeast', 'Science Fiction & Fantasy', 'Bible Study & Reference', 'Worship & Devotion', 'Cooking by Ingredient', 'Comic Books', 'Fairy Tales, Folk Tales & Myths', 'Pets & Animal Care', 'Australia & Oceania', 'Publishers', 'Social Sciences', 'Kitchen Appliances', 'Law Practice', 'Special Diet', 'Success', 'Fashion', 'Travel', 'Astronomy & Space Science', "Children's Books", 'Science, Nature & How It Works', 'Mysteries & Thrillers', 'Business', 'Reference', 'Cookbooks, Food & Wine', 'Sports & Outdoors', 'Action & Adventure', 'History & Philosophy', 'Travel Writing', 'Motivational', 'Death & Grief', 'Administration & Medicine Economics', 'Other Media', 'Photography & Video', 'Buddhism', 'Movies', 'Religious Studies', 'Nature & Ecology', 'Main Courses & Side Dishes', 'Beverages & Wine', 'Bibles', 'Ancient & Medieval Literature', 'Puzzles & Games', 'Constitutional Law', 'Canning & Preserving', 'Australia & South Pacific', 'Christian Books & Bibles', 'Needlecrafts & Textile Crafts', 'Mystery & Thrillers', 'Miscellaneous', 'Time Travel', 'New Age & Spirituality', 'Family Relationships', 'Hinduism', "Children's Health", 'Biography & History', 'Software', 'Business Culture', 'Diets & Weight Loss', 'Lesbian, Gay, Bisexual & Transgender Books', 'Americas', 'Hunting & Fishing', 'Mystery', 'Military', 'Biographies', 'Addiction & Recovery', 'Literary', 'Creativity', 'Paranormal & Urban', 'Graphic Novels', 'Industries', 'Stress Management', 'Historical Fiction', 'Historical', 'Atheism', 'Performing Arts', 'Parenting', 'Health, Fitness & Dieting', 'Automotive', 'Dictionaries & Thesauruses', 'Hiking & Camping', 'Processes & Infrastructure', 'Business & Finance', 'Writing, Research & Publishing Guides', 'Test Preparation', 'Clean & Wholesome', 'Hypnosis', 'Gardening & Landscape Design', 'Holidays', 'Leaders & Notable People', 'Thrillers & Suspense', 'Catholicism', 'Happiness', 'New England', 'Computers & Technology', 'Short Stories & Anthologies', 'Computer Science', 'Criminal Law', 'Arts, Music & Photography', 'Behavioral Sciences', 'Other Diets', 'Romantic Comedy', 'Anthologies', 'Home Improvement & Design', 'Teen & Young Adult', 'Cooking Education & Reference', 'Medical Books', 'Atlases & Maps', 'Job Hunting & Careers', 'Individual Artists', 'Architecture', 'Judaism', 'United States', 'Schools & Teaching', 'Regency', 'Ethnic & National', 'Dramas & Plays', 'Diseases & Physical Ailments', 'Programming', 'Operating Systems', 'Biological Sciences', 'Psychology', 'Engineering & Transportation', 'Engineering', 'Russia', 'Hardware & DIY', 'Science & Math', 'Fantasy', 'Africa', 'Marketing & Sales', 'Songbooks', 'Medicine & Health Sciences', 'Individual Sports', 'Small Business & Entrepreneurship', 'Drawing', 'Catalogs & Directories', 'Real Estate', 'Biographies & Memoirs', 'Poetry', 'Golf', 'Guitars & Fretted Instruments', 'Management & Leadership', 'Humor & Satire', 'Water Sports', 'Transportation', 'Humorous', 'Law', 'New, Used & Rental Textbooks', 'Comic Strips', 'Regional & International', 'LGBT', 'Religion & Spirituality', 'Activities, Crafts & Games', 'Sociology', 'Experiments, Instruments & Measurement', 'Nutrition', 'Christian Denominations & Sects', 'Churches & Church Leadership', 'Alternative Medicine', 'Canada', 'Games & Strategy Guides', 'South America', 'Animals', 'Ministry & Evangelism', 'Baking', 'War', 'Decorative Arts & Design', 'Protestantism', 'Early Learning', 'Coaching', 'Studying & Workbooks', 'Music', 'Etiquette', 'Politics & Social Sciences', 'Science & Mathematics', 'Europe', 'Humanities', 'Western', 'College & High School', 'Networking & Cloud Computing', 'Arts & Photography', 'Geography & Cultures', 'Science Fiction', 'Investing', 'Earth Sciences', 'Baseball', 'Asia', 'Occult & Paranormal', 'Physics', 'Regional U.S.', '</span>', 'Economics', 'Anthropology', 'Contemporary', 'Arts & Literature', 'Foreign Language Study & Reference', 'Vampires', 'True Crime', 'Romance', 'Specific Groups', 'Cars, Trains & Things That Go', 'Crafts, Hobbies & Home', 'Psychology & Counseling', 'Mathematics', 'Other Religions, Practices & Sacred Texts', 'Finance', 'Professionals & Academics', 'Sports', 'Religions', 'Archaeology', 'Chemistry', 'Web Development & Design', 'Crafts & Hobbies', 'Coming of Age', 'Holidays & Celebrations', 'Food, Lodging & Transportation', 'Classics', 'Administrative Law', 'Cooking Methods', 'Education & Teaching', 'How To Create Comics & Manga', 'History & Criticism', 'Databases & Big Data', 'Genre Fiction', 'Radio', 'Exercise & Fitness', 'Business & Professional Growth', 'Self-Help', 'Mental Health', "Women's Fiction", 'Humor & Entertainment', 'Comics & Graphic Novels', 'Mythology & Folk Tales', 'Ancient Civilizations', 'Ancient & Classical', 'Personal Health', 'Skills', 'Conflict Management', 'Theology', 'Field Guides', 'Safety & First Aid', 'Manga', 'Education', 'Mysteries & Detectives', 'Encyclopedias & Subject Guides', 'State & Local', 'Humor', 'Growing Up & Facts of Life', 'Parenting & Relationships', 'Christian Living', 'World Literature', 'New Adult & College', 'Gothic', 'Multicultural', 'Relationships', 'Erotica', 'Guitars', 'Romantic Suspense', 'Philosophy', 'Paranormal', 'Travelers & Explorers', 'Mystery, Thriller & Suspense', 'World', 'Other Eastern Religions & Sacred Texts', 'Medicine', 'Suspense', 'Urban Life', 'Politics & Government', 'Business & Professional', 'Personal Finance', 'Literature & Fiction', 'Sustainable Living', 'Essays & Correspondence']

import re, string, os
from typing import List, Union, Literal
from enum import Enum
import tiktoken
from langchain.llms.base import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain.agents.react.base import DocstoreExplorer
from langchain.docstore.base import Docstore
from langchain.prompts import PromptTemplate
from Agents.llm import AnyOpenAILLM
# from Agents.prompts import reflect_prompt, react_agent_prompt, react_reflect_agent_prompt, REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER
# from Agents.prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt, COT_INSTRUCTION, COT_REFLECT_INSTRUCTION
from Agents.agent_base_action_only_space import ReactAgent, parse_action, format_step, truncate_scratchpad
from collections import defaultdict
import random

class ReflexionStrategy(Enum):
    """
    NONE: No reflection
    LAST_ATTEMPT: Use last reasoning trace in context 
    REFLEXION: Apply reflexion to the next reasoning trace 
    LAST_ATTEMPT_AND_REFLEXION: Use last reasoning trace in context and apply reflexion to the next reasoning trace 
    """
    NONE = 'base'
    REFLEXION = 'reflexion'



class ActOnlyAgent(ReactAgent):
    def __init__(self,
                 task,
                 idxs: list, 
                 args, 
                 rec_env,
                 grounding_model,
                 max_steps: int = 30,
                 agent_prompt: PromptTemplate = action_only_agent_prompt,
                #  reflect_prompt: PromptTemplate = reflect_prompt,
                 react_llm: AnyOpenAILLM = AnyOpenAILLM(
                                             temperature=0,
                                             max_tokens=3000,
                                             model_name="gpt-3.5-turbo-16k",
                                             model_kwargs={"stop": "\n"},
                                             openai_api_key=os.environ['OPENAI_API_KEY'],
                                             openai_api_base = os.environ['OPENAI_API_BASE']),
                #  reflect_llm: AnyOpenAILLM = AnyOpenAILLM(
                #                                temperature=0,
                #                                max_tokens=3000,
                #                                model_name="gpt-3.5-turbo-16k",
                #                                openai_api_key=os.environ['OPENAI_API_KEY'],
                #                                openai_api_base = os.environ['OPENAI_API_BASE']),
                #  reflections_memory = None,
                 ) -> None:
        
        super().__init__(task, idxs, args, rec_env, grounding_model, max_steps, agent_prompt, react_llm)
        # self.reflect_llm = reflect_llm
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = REFLECTIONS
        # if reflections_memory == None:
        #     self.reflections: list = []
        # else:
        #     self.reflections = reflections_memory
        # self.reflections_str: dict = {}
        self.infos = {}
        self.final_infos = {}
        self.batch_size = args.batch_size
        self.enc = tiktoken.encoding_for_model("text-davinci-003")
    
    def run(self, reset = True, reflect_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION, outfilename='') -> None:
        
        for i in range(0, len(self.idxs), self.batch_size):
            temp_idxs = self.idxs[i: i+self.batch_size]
            
            print(f'temp_idxs:{temp_idxs}')
            
            self.single_run(temp_idxs, reset)
            
            self._build_info(temp_idxs)
        
        self.final_infos['trajs'] = self.infos
        return self.final_infos
    
    def _build_agent_prompt(self, idxs) -> str:
        prompts = [self.agent_prompt.format(
                            examples = self.react_examples,
                            # reflections = self.reflections_str[id],
                            question = self.task[id],
                            scratchpad = truncate_scratchpad(self.scratchpad[id],tokenizer=self.enc)) for id in idxs]
        return prompts
    
    def _build_info(self, idxs) -> str:
        for id in idxs:
            userid = self.userids[id]
            self.infos[id] = {}
            prompt = self.agent_prompt.format(
                                examples = self.react_examples,
                                question = '',
                                scratchpad = '')
            traj = self.task[id] + self.scratchpad[id]
            self.infos[id].update({'userid': userid, 'prompt': prompt, 'traj': traj, 'traj_by_line': traj.split('\n')})


### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")




def format_last_attempt(task: str,
                        idxs, 
                        scratchpad: str,
                        header: str = LAST_TRIAL_HEADER):
    strs = {}
    for id in idxs:
        strs[id] =  header + f'Question: {task[id]}\n' + truncate_scratchpad(scratchpad[id], tokenizer=gpt2_enc).strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'
    return strs


def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)