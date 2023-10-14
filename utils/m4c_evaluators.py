# Copyright (c) Facebook, Inc. and its affiliates.
# -*- coding: UTF-8 -*-
import re
import json
import pythia.utils.llama2_gen as llm_gen
from pythia.utils.llava_gen import llava_infer
from pythia.utils.logger import Logger
from PIL import Image
from typing import List, Optional
import spacy
import torch


open_pos = ["NOUN", "VERB", "ADJ", "ADV", "NUM"]
open_pos_ocr = ["NOUN", "VERB", "ADJ", "ADV", "NUM", "PART", "PROPN", "SYM", "X", "INTJ"]

class EvalAIAnswerProcessor:
    """
    Processes an answer similar to Eval AI
        copied from
        https://github.com/facebookresearch/pythia/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897
    """

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile("(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile("(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


class TextVQAAccuracyEvaluator:
    def __init__(self):
        self.answer_processor = EvalAIAnswerProcessor()
        self.nlp = spacy.load("en_core_web_sm")

    def _compute_answer_scores(self, raw_answers):
        """
        compute the accuracy (soft score) of human answers
        """
        answers = [self.answer_processor(a) for a in raw_answers]
        assert len(answers) == 10
        gt_answers = list(enumerate(answers))
        unique_answers = set(answers)
        unique_answer_scores = {}

        for unique_answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [
                    item for item in gt_answers if item != gt_answer
                ]
                matching_answers = [
                    item for item in other_answers if item[1] == unique_answer
                ]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            unique_answer_scores[unique_answer] = sum(accs) / len(accs)

        return unique_answer_scores

    def answer_extraction(self, caption,  num_question_generation=30):
        cap_use = ""
        # print(caption)
        caption = caption
        ans_to_cap_dict = {}
        answers = []
        for cap_idx, cap in enumerate(caption):
            # print(cap)
            cap_use += cap
            cap = cap.strip().strip(".")
            # print(cap)
            cap = self.nlp(cap)
            for token in cap:  # Noun /Verb/Adj//NUM
                if token.pos_ in open_pos:
                    if token.text.lower() not in ans_to_cap_dict:
                        ans_to_cap_dict[token.text.lower()] = [cap_idx]
                    else:
                        if cap_idx not in ans_to_cap_dict[token.text.lower()]:
                            ans_to_cap_dict[token.text.lower()].append(cap_idx)
                    answers.append(token.text)
            for ent in cap.ents:

                if ent.text not in answers:
                    if ent.text.lower() not in ans_to_cap_dict:
                        ans_to_cap_dict[ent.text.lower()] = [cap_idx]
                    else:
                        if cap_idx not in ans_to_cap_dict[ent.text.lower()]:
                            ans_to_cap_dict[ent.text.lower()].append(cap_idx)
                    answers.append(ent.text)
            for chunk in cap.noun_chunks:
                if len(chunk.text.split()) < 4:
                    if chunk.text.lower() not in ans_to_cap_dict:
                        ans_to_cap_dict[chunk.text.lower()] = [cap_idx]
                    else:
                        if cap_idx not in ans_to_cap_dict[chunk.text.lower()]:
                            ans_to_cap_dict[chunk.text.lower()].append(cap_idx)
                    #                 print(chunk.text)
                    answers.append(chunk.text)
        answers = sorted(answers, key=answers.count, reverse=True)
        real_answers = []
        for i in answers:
            i = i + "."
            if i not in real_answers:
                real_answers.append(i)

        contexts_for_question_generation = []
        answers = []
        for ans in real_answers[
            :num_question_generation
        ]:  # Generate questions for 30 answers with max frequencies.
            contexts_for_question_generation.append(
                "answer: %s  context: %s." % (ans, cap_use)
            )
            answers.append(ans)
        contexts_for_question_generation.append(
            "answer: %s  context: %s." % ("yes.", cap_use)
        )
        answers.append("yes.")
        return contexts_for_question_generation, answers, ans_to_cap_dict
    
    # QA gen
    def forward_qa_generation(self, samples, question_generation_tokenizer, question_generation_model):
        caption = samples["captions"][0]
        (
            contexts_for_question_generation,
            answers,
            ans_to_cap_dict,
        ) = self.answer_extraction(caption)
        # print("*******************************")
        # print(contexts_for_question_generation)
        # print("*******************************")
        inputs = question_generation_tokenizer(
            contexts_for_question_generation,
            padding="longest",
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        ).to(question_generation_model.device)
        question_size = inputs.input_ids.shape[0]
        cur_b = 0
        true_input_size = 10
        outputs_list = []
        question_generation_model = question_generation_model.to(inputs.input_ids.device)
        while cur_b < question_size:
            outputs = question_generation_model.module.generate(
                input_ids=inputs.input_ids[cur_b : cur_b + true_input_size],
                attention_mask=inputs.attention_mask[cur_b : cur_b + true_input_size],
                num_beams=3,
                max_length=30,
            )
            questions = question_generation_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            outputs_list += questions
            cur_b += true_input_size
        questions = outputs_list
        samples["questions"] = questions
        samples["answers"] = answers
        samples["ans_to_cap_dict"] = ans_to_cap_dict
        # results.append({"question_id": ques_id, "question":questions,"answer":answers})
        return samples

    def generate_caption(self, model_blip, caption_processor, pil_image: Image) -> str:
        assert model_blip is not None, "No caption model loaded."
        # self._prepare_caption()
        inputs = caption_processor(images=pil_image, return_tensors="pt").to(model_blip.device)
        # if not self.config.caption_model_name.startswith('git-'):
        # inputs = inputs.to(torch.device('cuda:0'))
        # print(inputs.device)
        # print(model_blip.device)
        tokens = model_blip.module.generate(**inputs, max_new_tokens=32)
        return caption_processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    # blip caption
    def interrogate_fast(self, model_blip, caption_processor, image: Image, max_flavors: int=32, caption: Optional[str]=None) -> str:
        """Fast mode simply adds the top ranked terms after a caption. It generally results in 
        better similarity between generated prompt and image than classic mode, but the prompts
        are less readable."""
        caption = caption or self.generate_caption(model_blip, caption_processor, image)
        # image_features = self.image_to_features(image)
        # merged = _merge_tables([self.artists, self.flavors, self.mediums, self.movements, self.trendings], self)
        # tops = merged.rank(image_features, max_flavors)
        return caption

    def choose_words(self, ocr):
        doc = self.nlp(ocr)
        ocr_in = []
        for token in doc:
            if token.pos_ in open_pos_ocr:
                ocr_in.append(token.text)
        ocr_cap_in = ", ".join([x.lower() for x in ocr_in[:50]])
        return ocr_cap_in

    def create_llm_prompt(self, samples, num_question_per_img=30):
        syn_question_queid = samples["questions"]
        syn_ans_queid = samples["answers"]
        Task_Prompt = ""
        num_question_generation = len(syn_question_queid) if len(syn_question_queid) <=num_question_per_img else num_question_per_img
        for idx in range(num_question_generation):
            Task_Prompt += "Question: "
            Task_Prompt += syn_question_queid[idx]
            Task_Prompt += "\n"
            Task_Prompt += "Answer: "
            Task_Prompt += syn_ans_queid[idx]
            Task_Prompt += "\n"
        
        samples["Task_Prompt"] = Task_Prompt
        # print(Task_Prompt)
        return Task_Prompt

    def eval_pred_list(self, pred_list, writer, json_for_caption, json_for_qa, json_for_llm, \
    model_vlm, tokenizer_vlm, image_processor, model_blip, caption_processor, tokenizer_t5, model_t5, model_llama2, tokenizer_llama2, \
    image_name, question_str, all_gt_answers, ocr_tokens, img):
        pred_scores = []
        # print('1:',all_gt_answers)
        # print('2:',ocr_tokens)
        # print("Begin eval...")
        if ocr_tokens:
            for entry,ocr_token,single_image in zip(pred_list,ocr_tokens,img):
                # print("3:",entry['questions'])
                pred_answer = self.answer_processor(entry['pred_answer'])
                unique_answer_scores = self._compute_answer_scores(
                    entry['gt_answers']
                )
                candidates = ""
                for key,values in unique_answer_scores.items():
                    candidates += f"{key}: {values}, "


                # vlm caption
                if model_vlm:
                    caption = {}
                    cap = {"captions":[[]]}
                    ocr_token= ", ".join([x.lower() for x in ocr_token])
                    # single caption
                    ocr_cap_in = self.choose_words(ocr_token)
                    cap["captions"][0].append(llava_infer(single_image,ocr_cap_in[:-2],tokenizer_vlm, model_vlm, image_processor))
                    caption["captions"] = cap["captions"][0][0]
                    caption["question"] = entry['questions']
                    caption['candidates'] = candidates[:-2]
                    caption['pred_answer'] = pred_answer
                    print("*********************************************")
                    print(caption)

                    # exit()
                    

                # blip caption    
                if model_blip:
                    # should_pass = False
                    # try:
                    #     with open(json_for_caption,'r') as caption_gen_file:
                    #         for file in caption_gen_file:
                    #             file_caption = json.loads(file)
                    #             if file_caption['image_path'] == image_name[0]:
                    #                 should_pass = True
                    #                 break
                    # except json.JSONDecodeError as e:
                    #     print(f"save ERORR:{e}")
                    # if should_pass:
                    #     continue
                    caption = {}
                    cap = {"captions":[[]]}
                    cap["captions"][0].append(self.interrogate_fast(model_blip,caption_processor,single_image))
                    pattern = re.compile(r'^[-._?¿!@^*()~]')
                    ocr_token = [x for x in ocr_token if not pattern.match(x)]
                    ocr_cap_in = ", ".join([x.lower() for x in ocr_token[:50]])
                    # ocr_word_list = ocr_cap_in.split(',')
                    # ocr_cap_in = ' '.join(['"{}"'.format(word) for word in ocr_word_list])
                    cap["captions"][0][0] += " with "
                    # ocr_cap_in = self.choose_words(ocr_cap_in)
                    cap["captions"][0][0] += ocr_cap_in
                    # print(image_name[0])
                    # print(cap["captions"][0][0])
                    caption['image_path'] = image_name[0]
                    caption['ocr_tokens'] = ocr_cap_in
                    caption["captions"] = cap["captions"][0][0]
                    # try:
                    #     with open(json_for_caption,'r') as caption_gen_file:
                    #         pre_caption = json.load(caption_gen_file)
                    # except json.JSONDecodeError as e:
                    #     print(f"load ERORR:{e}")
                    # if pre_caption == '':
                    #     pre_caption = {}
                    # pre_caption.update(caption)
                    caption["question"] = entry['questions']
                    caption['candidates'] = candidates[:-2]
                    caption['pred_answer'] = pred_answer
                    print("*********************************************")
                    print(caption)
                    # print(pre_caption)
                    
                    try:
                        with open(json_for_caption,'a+') as caption_gen_file:
                            json.dump(caption,caption_gen_file)
                            caption_gen_file.write('\n')
                    except json.JSONDecodeError as e:
                        print(f"save ERORR:{e}")
                    # print("cap:",cap)
                    
                if model_t5:
                    cap = {"captions":[[]]}
                    with open(json_for_caption,'r') as caption_gen_file:
                        for sig_img in caption_gen_file:
                            single_img = json.loads(sig_img)
                            # print(single_img)
                            if single_img == {}:
                                continue
                            # print("img_name1:",image_name[0])
                            # print("img_name2:",list(single_img.keys())[0])
                            if list(single_img.keys())[0] == image_name[0]:
                                cap["captions"][0].append(single_img.get(image_name[0]))
                                break
                                # pre_caption = json.load(caption_gen_file)

                    single_qa_gen = self.forward_qa_generation(cap,tokenizer_t5,model_t5)
                    single_qa_gen['image_path'] = image_name[0]
                    single_qa_gen['candidates'] = candidates
                    print("*********************************************")
                    print("single_qa_gen:",single_qa_gen)
                    with open(json_for_qa,'a+') as qa_gen_file:
                        json.dump(single_qa_gen,qa_gen_file)
                        qa_gen_file.write('\n')
                    # writer.write("single_qa_gen:")
                    # writer.write(single_qa_gen)
                
                if model_llama2:
                    context_prompt = cap["captions"][0][0]
                    task_prompt = self.create_llm_prompt(single_qa_gen)
                    LLMPrompt = (
                        "Context: "
                        + context_prompt
                        + "\n"
                        + task_prompt
                        + "======\n"
                        + "Question: "
                        + entry["questions"]
                        + "\n"
                        + "Candidates: "
                        + candidates[:-1]
                        + "\n"
                        + "Answer: "
                        + entry['pred_answer']
                        + "\n"
                        + "The number in parentheses after each candidate represents its credibility. \
                        The number in parentheses after each candidate represents its credibility. The more credible the candidate, the more you should consider the choice. Based on the credibility and contexts above, please reason the answer to the following question without explanations.\n"
                        + "Question: "
                        + entry["questions"]
                        + "\n"
                        + "Answer: "
                    )
                    LLM_outputs = llm_gen.generate(
                        model=model_llama2,
                        tokenizer=tokenizer_llama2,
                        user_prompt=LLMPrompt,
                    )
                    # writer.write("Question:")
                    # writer.write(entry["questions"])
                    # writer.write("LLM:")
                    # writer.write(LLM_outputs)
                    print("LLM:",LLM_outputs.lower())
                print("pred:",pred_answer)
                print('gt:',unique_answer_scores)

                # writer.write("pred_answer:")
                # writer.write(pred_answer)
                # writer.write("gt_answers",unique_answer_scores)
                # writer.write("==================================")
                score = unique_answer_scores.get(pred_answer, 0.)

                # LLM_auxiliary
                if json_for_llm:
                    with open(json_for_llm,'r') as llm_files:
                        for llm_file in llm_files:
                            if llm_file == '\n' or llm_file == {}:
                                continue
                            llm_out = json.loads(llm_file)
                            if llm_out['image_path'] == image_name[0]:
                                llm_pred = llm_out['llm_out']
                                break
                    print("llm_pred:",llm_pred)
                    score_llm = unique_answer_scores.get(llm_pred, 0.)
                    if score_llm > score:
                        pred_scores.append(score_llm)
                    else:
                        pred_scores.append(score)
                elif model_llama2:
                    score_llm = unique_answer_scores.get(LLM_outputs.lower(), 0.)
                    if score_llm > score:
                        pred_scores.append(score_llm)
                    else:
                        pred_scores.append(score)
                else:
                    pred_scores.append(score)
            # exit()
        else:
            for entry in pred_list:
            # print("3:",entry['questions'])
                pred_answer = self.answer_processor(entry['pred_answer'])
                unique_answer_scores = self._compute_answer_scores(
                    entry['gt_answers']
                )
                
                print("pred:",pred_answer)
                print('gt:',unique_answer_scores)
                
                score = unique_answer_scores.get(pred_answer, 0.)
                pred_scores.append(score)
        # exit()
        accuracy = sum(pred_scores) / len(pred_scores)
        print("accuracy:",accuracy)
        print("**********************************************")
        return accuracy


class STVQAAccuracyEvaluator:
    def __init__(self):
        self.answer_processor = EvalAIAnswerProcessor()

    def eval_pred_list(self, pred_list):
        pred_scores = []
        for entry in pred_list:
            pred_answer = self.answer_processor(entry['pred_answer'])
            gts = [self.answer_processor(a) for a in entry['gt_answers']]
            score = (1. if pred_answer in gts else 0.)
            pred_scores.append(score)

        accuracy = sum(pred_scores) / len(pred_scores)
        return accuracy


class STVQAANLSEvaluator:
    def __init__(self):
        import editdistance  # install with `pip install editdistance`
        self.get_edit_distance = editdistance.eval

    def get_anls(self, s1, s2):
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        iou = 1 - self.get_edit_distance(s1, s2) / max(len(s1), len(s2))
        anls = iou if iou >= .5 else 0.
        return anls

    def eval_pred_list(self, pred_list):
        pred_scores = []
        for entry in pred_list:
            anls = max(
                self.get_anls(entry['pred_answer'], gt)
                for gt in entry['gt_answers']
            )
            pred_scores.append(anls)

        accuracy = sum(pred_scores) / len(pred_scores)
        return accuracy


class TextCapsBleu4Evaluator:
    def __init__(self):
        # The following script requires Java 1.8.0 and pycocotools installed.
        # The pycocoevalcap can be installed with pip as
        # pip install git+https://github.com/ronghanghu/coco-caption.git@python23
        # Original pycocoevalcap code is at https://github.com/tylin/coco-caption
        # but has no python3 support yet.
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        from pycocoevalcap.bleu.bleu import Bleu
        self.tokenizer = PTBTokenizer()
        self.scorer = Bleu(4)

    def eval_pred_list(self, pred_list):
        # Create reference and hypotheses captions.
        gts = {}
        res = {}
        for idx, entry in enumerate(pred_list):
            gts[idx] = [{'caption': a} for a in entry['gt_answers']]
            res[idx] = [{'caption': entry['pred_answer']}]

        gts = self.tokenizer.tokenize(gts)
        res = self.tokenizer.tokenize(res)
        score, _ = self.scorer.compute_score(gts, res)

        bleu4 = score[3]  # score is (Bleu-1, Bleu-2, Bleu-3, Bleu-4)
        return bleu4
