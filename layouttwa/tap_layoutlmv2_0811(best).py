# 此tap参考TWA修改！！！！！！！！！！！

from logging import captureWarnings
import os
import functools
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import collections
from transformers.models.layoutlmv2.modeling_layoutlmv2 import (
    LayoutLMv2Encoder,
    LayoutLMv2Embeddings,
    LayoutLMv2PreTrainedModel,
)

from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder
)
from transformers import (
    BertConfig, BertPreTrainedModel, BertModel, 
    ViTFeatureExtractor, ViTModel,
    LayoutLMv2Processor, LayoutLMv2Model,LayoutLMv2Config,
    )

LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP = {}

LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

from pythia.common.registry import registry
from pythia.models.base_model import BaseModel
from pythia.modules.layers import ClassifierLayer

from pythia.modules.encoders import ImageEncoder
from pythia.utils.twa_utils import add_cons_ocr_info, create_ocr_char_info

# https://discuss.pytorch.org/t/batched-index-select/9115/7

# LayoutLM
#########################################################################################################################


def batched_index_select(input, dim, index):
    views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def load_line_to_ids_dict(fname):
    """Loads a vocabulary file into a dictionary."""
    id2voc = collections.OrderedDict()
    with open(fname, "r", encoding="utf-8") as reader:
        chars = reader.readlines()
    for index, char in enumerate(chars):
        char = char.rstrip('\n')
        id2voc[index] = char
    return id2voc


@registry.register_model("mytwa")
class M4C(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.pretrain = self.config.pretrain
        self.mmt_config = LayoutLMv2Config(**self.config.mmt)
        self._datasets = registry.get("config").datasets.split(",")
        self.answer_processor = registry.get(
            self._datasets[0] + "_answer_processor"
        )

        self.con_mnt_rel = getattr(
            self.config.ocr, 'con_mnt_rel', False
        )

        self.con_remove_bbox = getattr(
            self.config.ocr, 'con_remove_bbox', False
        )
        self.use_ocr_conf = getattr(
            self.config.ocr, 'use_ocr_conf', False
        )
        # print(self.con_mnt)
        #self.id2voc_dict = load_line_to_ids_dict(fname='data/dict/term_vocab')
        if self.con_mnt_rel:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def build(self):
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []
        self._build_char()
        # split model building into several components
        self._build_txt_encoding()
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._build_mmt()
        self._build_output()
        # MLM, Contra, RPP pretrain heads
        self.cls = BertLMPredictionHead(
            self.text_bert.embeddings.word_embeddings.weight)
        self.pollute_cls = PolluteLinear()
        self.overlap_cls = OverlapLinear()

    def _build_char(self):
        char_info_params = self.config.vocab
        self.vocab_char_fusion = char_info_params.fusion
        self.char_num = char_info_params.char_num
        self.char_max_num = char_info_params.char_max_num
        self.use_char_decoding = char_info_params.use_char_decoding
        self.fb_featrue = char_info_params.features

        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        num_choices -= self.config.classifier.ocr_max_num
        self.classifier = nn.Linear(
            self.mmt_config.hidden_size,
            num_choices,
            bias=char_info_params.bias,
        )
        if 'bert' in self.fb_featrue or 'char' in self.fb_featrue or 'bert' in self.config.ocr.features or 'char' in self.config.ocr.features:
            if 'bert' in self.fb_featrue or 'char' in self.fb_featrue:
                vocabs = self.answer_processor.processor.answer_vocab.word_list
                self.vocab_char, self.vocab_char_mask, self.vocab_word_ids = add_cons_ocr_info(
                    vocabs, len(vocabs))
                #self.vocab_char, self.vocab_char_mask, nouseLabel = add_ocr_char_info(vocabs, ans_list, len(vocabs), self.char_max_num, True)
                # print(self.vocab_char.size())
                # print(self.vocab_word_ids.size())
                self.vocab_char = self.vocab_char.view(
                    self.vocab_char.size(0), -1)
                self.vocab_word_ids = self.vocab_word_ids.view(
                    self.vocab_word_ids.size(0), -1)
                # print(self.vocab_char.size())
                # print(self.vocab_word_ids.size())
            if 'char' in self.fb_featrue or 'char' in self.config.ocr.features:
                self.char_position_embedding = nn.Embedding(char_info_params.char_decoding_max_step,
                                                            self.mmt_config.hidden_size)
                self.char_embedding = nn.Embedding(
                    char_info_params.char_num, self.mmt_config.hidden_size)
            # nn.init.xavier_uniform_(self.char_position_embedding.weight)
            # nn.init.xavier_uniform_(self.char_embedding.weight)
            # if 'char' in self.fb_featrue:
            #     self.char_layernorm = BertLayerNorm(self.mmt_config.hidden_size)
            if 'char' in self.config.ocr.features:
                self.ocr_char_layernorm = BertLayerNorm(
                    self.mmt_config.hidden_size)
            # self.ocr_bert_layernorm = BertLayerNorm(self.mmt_config.hidden_size)
            # self.ocr_char_layernorm = nn.Identity()

        if not self.use_char_decoding and self.vocab_char_fusion == 'sa':
            self.vocab_scores = OcrPtrNet(self.mmt_config.hidden_size)
        elif self.vocab_char_fusion == 'ec':
            self.vocab_ec_linear = nn.Linear(
                2 * self.mmt_config.hidden_size, self.mmt_config.hidden_size)

    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768
        self.text_bert_config = LayoutLMv2Config(**self.config.text_bert)
        if self.config.text_bert_init_from_bert_base:
            # self.text_bert = TextBert.from_pretrained(
            #     'microsoft/layoutlm-base-uncased', config=self.text_bert_config
            # )
            self.text_bert = TextBert.from_pretrained(
                'microsoft/layoutlmv2-base-uncased', config=self.text_bert_config
            )
            # Use a smaller learning rate on text bert when initializing
            # from BERT_BASE
            self.finetune_modules.append({
                'module': self.text_bert,
                'lr_scale': self.config.lr_scale_text_bert,
            })
        else:
            self.writer.write('NOT initializing text_bert from BERT_BASE')
            self.text_bert = TextBert(self.text_bert_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if self.mmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            self.writer.write(
                'Projecting text_bert output to {} dim'.format(
                    self.mmt_config.hidden_size
                )
            )
            self.text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()

    def _build_obj_encoding(self):
        # vit
        self.vit_feat_extract = ViTFeatureExtractor("google/vit-base-patch16-224-in21k")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        # self.x_position_embeddings = nn.Embedding(
        #     1024, self.config.obj.mmt_in_dim
        # )
        # self.y_position_embeddings = nn.Embedding(
        #     1024, self.config.obj.mmt_in_dim
        # )
        # # object appearance feature: Faster R-CNN
        # self.obj_faster_rcnn_fc7 = ImageEncoder(
        #     encoder_type='finetune_faster_rcnn_fpn_fc7',
        #     in_dim=2048,
        #     weights_file='detectron/fc6/fc7_w.pkl',
        #     bias_file='detectron/fc6/fc7_b.pkl',
        #     model_data_dir=self.config["model_data_dir"]
        # )
        # # apply smaller lr to pretrained Faster R-CNN fc7
        # self.finetune_modules.append({
        #     'module': self.obj_faster_rcnn_fc7,
        #     'lr_scale': self.config.lr_scale_frcn,
        # })
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.config.obj.mmt_in_dim, self.mmt_config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(
            4, self.mmt_config.hidden_size
        )

        self.obj_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.obj_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.obj_drop = nn.Dropout(self.config.obj.dropout_prob)

    def _build_ocr_encoding(self):
        self.remove_ocr_fasttext = getattr(
            self.config.ocr, 'remove_ocr_fasttext', False
        )
        self.remove_ocr_phoc = getattr(
            self.config.ocr, 'remove_ocr_phoc', False
        )
        self.remove_ocr_frcn = getattr(
            self.config.ocr, 'remove_ocr_frcn', False
        )
        self.remove_ocr_semantics = getattr(
            self.config.ocr, 'remove_ocr_semantics', False
        )
        self.remove_ocr_bbox = getattr(
            self.config.ocr, 'remove_ocr_bbox', False
        )

        # Layoutlmv2
        self.ocr_layoutlmv2_config = LayoutLMv2Config(**self.config.ocr_layoutlmv2)
        self.processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
        self.layoutlmv2model = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased", config=self.ocr_layoutlmv2_config)

        # OCR appearance feature: Faster R-CNN
        self.ocr_faster_rcnn_fc7 = ImageEncoder(
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file='detectron/fc6/fc7_w.pkl',
            bias_file='detectron/fc6/fc7_b.pkl',
            model_data_dir=self.config["model_data_dir"]
        )
        self.finetune_modules.append({
            'module': self.ocr_faster_rcnn_fc7,
            'lr_scale': self.config.lr_scale_frcn,
        })

        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.ocr.mmt_in_dim, self.mmt_config.hidden_size
        )

        if self.use_ocr_conf:
            self.linear_ocr_conf_to_mmt_in = nn.Linear(
                1, self.mmt_config.hidden_size
            )
            self.ocr_conf_layer_norm = BertLayerNorm(
                self.mmt_config.hidden_size)

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(
            4, self.mmt_config.hidden_size
        )

        self.ocr_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.projection =  nn.Linear(in_features=2048, out_features=self.mmt_config.hidden_size)
        # self.ocr_layoutlmv2_normal = nn.Linear(in_features=self.config.ocr.layoutlmv2_in_dim, out_features=self.mmt_config.hidden_size)
        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)

    def _build_mmt(self):
        self.mmt = MMT(self.mmt_config)

        # allow specifying a different/scaled lr for multimodal transformer
        self.finetune_modules.append({
            'module': self.mmt,
            'lr_scale': self.config.lr_scale_mmt,
        })

        self.char_vocab_decoder = getattr(
            self.config.vocab, 'char_vocab_decoder', False
        )
        self.fb_featrue = self.config.vocab.features

    def _build_output(self):
        # dynamic OCR-copying scores with pointer network
        self.ocr_ptr_net = OcrPtrNet(**self.config.classifier.ocr_ptr_net)

        # fixed answer vocabulary scores
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        # remove the OCR copying dimensions in LoRRA's classifier output
        # (OCR copying will be handled separately)
        num_choices -= self.config.classifier.ocr_max_num

    def forward(self, sample_list):
        fwd_results = {}
        self._forward_cmbtxt_encoding(sample_list, fwd_results)
        self._forward_obj_encoding(sample_list, fwd_results)
        self._forward_ocr_encoding(sample_list, fwd_results)
        self._forward_mmt_and_output(sample_list, fwd_results)

        # only keep scores in the forward pass results
        results = {"scores": fwd_results["scores"]}
        if self.pretrain:
            results["textcls_scores"] = fwd_results['textcls_scores']
            results["pollutecls_scores"] = fwd_results['pollutecls_scores']
            results["overlapcls_scores"] = fwd_results['overlapcls_scores']
            if self.con_mnt_rel:
                results["contrastive_scores"] = fwd_results['contrastive_scores']
        return results

    def _forward_cmbtxt_encoding(self, sample_list, fwd_results):
        fwd_results['txt_inds'] = sample_list.cmb_text
        fwd_results['txt_mask'] = _get_mask_medpad(sample_list.cmb_text)
        fwd_results['txt_type_mask'] = _get_type_mask_medpad(
            sample_list.cmb_text)
        fwd_results['adv'] = sample_list.adv
        if sample_list.adv == True:
            fwd_results['adv_delta_txt'] = sample_list.adv_delta_txt
        else:
            fwd_results['adv_delta_txt'] = None

    def _forward_obj_encoding(self, sample_list, fwd_results):
        # vit
        obj_vect = self.vit_feat_extract(list(sample_list.image),return_tensors = 'pt')['pixel_values'].to(sample_list.image_feature_0.device)
        # print("1:",obj_vect.shape)#[x,3,224,224]
        obj_feat = self.vit(obj_vect).last_hidden_state
        if sample_list.adv == True:
            obj_feat = obj_feat + sample_list.adv_delta_obj
        obj_feat = obj_feat[:, :sample_list.obj_bbox_coordinates.size(1), :]
        
        # print("2:",obj_feat.shape)#[x,197,768]

        # object appearance feature: Faster R-CNN fc7
        # obj_fc6 = sample_list.image_feature_0[:,
        #                                       :sample_list.obj_bbox_coordinates.size(1), :]
        # if sample_list.adv == True:
        #     obj_fc6 = obj_fc6 + sample_list.adv_delta_obj
        # obj_fc7 = self.obj_faster_rcnn_fc7(obj_fc6)
        # obj_fc7 = F.normalize(obj_fc7, dim=-1)#[x,100,2048]

        # obj_feat = obj_fc7
        # obj_bbox = sample_list.obj_bbox_origin_coordinates
        obj_bbox = sample_list.obj_bbox_coordinates
        obj_bbox_expand = torch.zeros([obj_bbox.size(0), obj_feat.size(1)-obj_bbox.size(1), obj_bbox.size(2)])
        obj_bbox = torch.cat([obj_bbox, obj_bbox_expand], dim=1)
        # top_left_x_feat =     self.x_position_embeddings(torch.clamp(obj_bbox[:,:, 0],min = 0, max = 1023))
        # top_left_y_feat =     self.y_position_embeddings(torch.clamp(obj_bbox[:,:, 1],min = 0, max = 1023))
        # bottom_right_x_feat = self.x_position_embeddings(torch.clamp(obj_bbox[:,:, 2],min = 0, max = 1023))
        # bottom_right_y_feat = self.y_position_embeddings(torch.clamp(obj_bbox[:,:, 3],min = 0, max = 1023))
        # obj_bbox_feat = top_left_x_feat + top_left_y_feat + bottom_right_x_feat + bottom_right_y_feat
        # print(obj_bbox.shape)#[32,100,4]
        obj_mmt_in = (
            self.obj_feat_layer_norm(
                obj_feat
            ) + self.obj_bbox_layer_norm(
                self.linear_obj_bbox_to_mmt_in(obj_bbox)
            )
        )
        # obj_final_feat = self.linear_obj_feat_to_mmt_in(torch.cat([obj_feat,obj_bbox_feat], axis = -2))
        # obj_mmt_in = self.obj_feat_layer_norm(obj_final_feat)
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results['obj_mmt_in'] = obj_mmt_in

        # binary mask of valid object vs padding
        obj_nums = sample_list.image_info_0.max_features
        fwd_results['obj_mask'] = _get_mask(obj_nums, obj_mmt_in.size(1))

    def _forward_ocr_encoding(self, sample_list, fwd_results):
        # OCR FastText feature (300-dim)
        ocr_fasttext = sample_list.context_feature_0
        if sample_list.adv == True:
            ocr_fasttext = ocr_fasttext + sample_list.adv_delta_ocr_fasttext
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR PHOC feature (604-dim)
        ocr_phoc = sample_list.context_feature_1
        if sample_list.adv == True:
            ocr_phoc = ocr_phoc + sample_list.adv_delta_ocr_phoc
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        # OCR appearance feature: Faster R-CNN fc7
        if self.con_mnt_rel:
            ocr_len = int(ocr_fasttext.size(1)/2)
        else:
            ocr_len = ocr_fasttext.size(1)
        assert ocr_len == 150
        ocr_fc6 = sample_list.image_feature_1[:, :ocr_len, :]
        # if sample_list.adv == True:
        #     ocr_fc6 = ocr_fc6 + sample_list.adv_delta_ocr
        ocr_fc7 = self.ocr_faster_rcnn_fc7(ocr_fc6)
        if self.con_mnt_rel:
            ocr_fc7 = torch.cat([ocr_fc7, ocr_fc7], dim=-2)
        ocr_fc7 = F.normalize(ocr_fc7, dim=-1)# [x,200,2048]
        assert ocr_fc7.size(1) == ocr_phoc.size(1)

        # OCR order vectors (legacy from LoRRA model; set to all zeros)
        # TODO remove OCR order vectors; they are not needed
        ocr_order_vectors = torch.zeros_like(sample_list.order_vectors)

        if self.remove_ocr_fasttext:
            ocr_fasttext = torch.zeros_like(ocr_fasttext)
        if self.remove_ocr_phoc:
            ocr_phoc = torch.zeros_like(ocr_phoc)
        if self.remove_ocr_frcn:
            ocr_fc7 = torch.zeros_like(ocr_fc7)
        ocr_feat = torch.cat(
            [ocr_fasttext, ocr_phoc, ocr_fc7, ocr_order_vectors],
            dim=-1
        )# [x,200,3152-2048]
        if self.con_mnt_rel:
            if self.con_remove_bbox:
                related_bbox = torch.zeros_like(
                    sample_list.ocr_bbox_coordinates)
                related_origin = torch.zeros_like(
                    sample_list.ocr_bbox_origin)
            else:
                related_bbox = sample_list.ocr_bbox_coordinates
                related_origin = sample_list.ocr_bbox_origin
            ocr_bbox = torch.cat(
                [sample_list.ocr_bbox_coordinates, related_bbox], dim=1)
            ocr_origin = torch.cat(
                [sample_list.ocr_bbox_origin, related_origin], dim=1)
            ocr_bbox_for_bert = torch.zeros_like(sample_list.ocr_bbox_origin)
            ocr_origin_for_bert = torch.cat(
                [sample_list.ocr_bbox_origin, ocr_bbox_for_bert], dim=1)
            if self.use_ocr_conf:
                tmp_conf = sample_list.ocr_conf.unsqueeze(-1)
                ocr_conf = torch.cat([tmp_conf, tmp_conf], dim=1)

        else:
            ocr_bbox = sample_list.ocr_bbox_coordinates
            ocr_origin = sample_list.ocr_bbox_origin
            if self.use_ocr_conf:
                ocr_conf = sample_list.ocr_conf.unsqueeze(-1)
        assert ocr_feat.size(1) == ocr_bbox.size(1)
        if self.remove_ocr_semantics:
            ocr_feat = torch.zeros_like(ocr_feat)
        if self.remove_ocr_bbox:
            ocr_bbox = torch.zeros_like(ocr_bbox)
        if self.use_ocr_conf:
            # print("ocr_feat:",ocr_feat.shape)# [x,200,3152-2048]
            # print("ocr_bbox:",ocr_bbox.shape)# [x,200,4]
            # print("ocr_conf:",ocr_conf.shape)# [x,200,1]
            ocr_mmt_in = (
                self.ocr_feat_layer_norm(
                    self.linear_ocr_feat_to_mmt_in(ocr_feat)
                ) + self.ocr_bbox_layer_norm(
                    self.linear_ocr_bbox_to_mmt_in(ocr_bbox)
                ) + self.ocr_conf_layer_norm(
                    self.linear_ocr_conf_to_mmt_in(ocr_conf)
                )
            )
        else: 
            ocr_mmt_in = (
                self.ocr_feat_layer_norm(
                    self.linear_ocr_feat_to_mmt_in(ocr_feat)
                ) + self.ocr_bbox_layer_norm(
                    self.linear_ocr_bbox_to_mmt_in(ocr_bbox)
                ) 
            )
        for i in range(len(sample_list.img)):
            bbox = ocr_origin[i][~(ocr_origin[i] == torch.tensor([0, 0, 0, 0]).to(ocr_mmt_in.device)).all(dim=1)].to(ocr_mmt_in.device)
            encoding = self.processor(sample_list.img[i], sample_list.ocr_tokens[i][:ocr_mmt_in.size(1)], boxes=bbox[:len(sample_list.ocr_tokens[i])], return_tensors="pt").to(ocr_mmt_in.device)
            if i == 0:
                ocr_layoutlmv2_pixel_values = encoding['image']
            else:
                ocr_layoutlmv2_pixel_values = torch.cat((ocr_layoutlmv2_pixel_values, encoding['image']), dim=0)

        # print("2:",ocr_layoutlmv2_pixel_values.shape)#[x,3,224,224]
        # ocr_mmt_in = torch.zeros_like(ocr_mmt_in)
        # print("1,",ocr_mmt_in.shape)# [x,200,768]
        if 'char' in self.config.ocr.features:
            ocr_char = sample_list.context_char
            ocr_char_mask = sample_list.context_char_mask
            ocr_char = _char_embedding(
                self.char_embedding, self.char_position_embedding, ocr_char, ocr_char_mask)
            ocr_char = self.ocr_char_layernorm(ocr_char)
            ocr_mmt_in += ocr_char

        if 'bert' in self.config.ocr.features:

            ocr_rem_id = sample_list.word_ids
            ocr_rem_mask = _get_mask_medpad(ocr_rem_id)
            ocr_rem_type_mask = _get_type_mask_medpad_ocr(ocr_rem_id)

            if sample_list.adv == True:
                ocr_bert_out = self.text_bert(
                    image=ocr_layoutlmv2_pixel_values,
                    txt_inds=ocr_rem_id,
                    txt_mask=ocr_rem_mask,
                    txt_type_mask=ocr_rem_type_mask,
                    bbox = ocr_origin,
                    adv=True,
                    adv_delta_ocr=sample_list.adv_delta_ocr,
                )# [x,300,768]
            else:
                ocr_bert_out = self.text_bert(
                    image=ocr_layoutlmv2_pixel_values,
                    txt_inds=ocr_rem_id,
                    txt_mask=ocr_rem_mask,
                    txt_type_mask=ocr_rem_type_mask,
                    bbox = ocr_origin,
                    adv=False,
                )# [x,300,768]
            
            # ocr_bert_out = self.text_bert_out_linear(ocr_bert_out)
            # ocr_fc7 = self.ocr_feat_layer_norm(self.projection(ocr_fc7))
            # ocr_bert_out += ocr_fc7
            if ocr_bert_out.size(1) < ocr_mmt_in.size(1):
                expand_size = torch.zeros([ocr_mmt_in.size(0), ocr_mmt_in.size(1)-ocr_bert_out.size(1), ocr_mmt_in.size.size(2)]).to(ocr_mmt_in.device)
                ocr_bert_out = torch.cat([ocr_bert_out, expand_size], dim=1)
            ocr_bert_out = self.ocr_feat_layer_norm(self.text_bert_out_linear(ocr_bert_out[:, :ocr_mmt_in.size(1), :]))
            ocr_mmt_in += ocr_bert_out
        # binary mask of valid OCR vs padding
        if self.con_mnt_rel:
            ocr_nums = sample_list.context_info_0.max_features
            ori_mask = _get_mask(ocr_nums, ocr_mmt_in.size(1) / 2)
            fwd_results['ocr_mask'] = torch.cat([ori_mask, ori_mask], dim=-1)
        else:
            ocr_nums = sample_list.context_info_0.max_features
            fwd_results['ocr_mask'] = _get_mask(ocr_nums, ocr_mmt_in.size(1))

        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results['ocr_mmt_in'] = ocr_mmt_in

    def _forward_mmt(self, sample_list, fwd_results):
        # first forward the text BERT layers

        #sequence_repr = outputs[0]
        #char_sequence_repr = outputs[2]
        #char_mask = _get_mask_medpad(sample_list.ocr_text)
        #char_type_mask = _get_type_mask_medpad(sample_list.ocr_text)
       # print(char_sequence_repr.size())

        text_bert_out = self.text_bert(
            txt_inds=fwd_results['txt_inds'],
            txt_mask=fwd_results['txt_mask'],
            txt_type_mask=fwd_results['txt_type_mask'],
            bbox=None,
            adv=fwd_results['adv'],
            adv_delta_txt=fwd_results['adv_delta_txt']
        )
        fwd_results['txt_emb'] = self.text_bert_out_linear(text_bert_out[:, :fwd_results['txt_inds'].size(1), :])

        # gen bbox
        w = torch.tensor(sample_list.image_info_0["image_width"]).unsqueeze(
            -1).float().to(fwd_results['txt_emb'].device)
        h = torch.tensor(sample_list.image_info_0["image_height"]).unsqueeze(
            -1).float().to(fwd_results['txt_emb'].device)
        bbox = torch.cat([torch.tensor([0, 0, 1, 1]).view(1, 1, 4).repeat(fwd_results['txt_emb'].shape[0], fwd_results['txt_emb'].shape[1], 1).float().to(sample_list.ocr_bbox_coordinates.device),
                          sample_list.obj_bbox_coordinates, sample_list.ocr_bbox_coordinates,
                          torch.tensor([0, 0, 1, 1]).view(1, 1, 4).repeat(fwd_results['prev_inds'].shape[0], fwd_results['prev_inds'].shape[1], 1).float().to(sample_list.ocr_bbox_coordinates.device)], dim=1)
        fwd_results['bbox'] = (bbox * 1023).long().clamp(min=0, max=1023)

        if self.fb_featrue != '':
            vocab_emb = 0
            if ('char' in self.fb_featrue) and self.char_vocab_decoder:
                if self.vocab_char.device != fwd_results['txt_emb'].device:
                    self.vocab_char = self.vocab_char.to(
                        fwd_results['txt_emb'].device)
                    self.vocab_char_mask = self.vocab_char_mask.to(
                        fwd_results['txt_emb'].device)
                vocab = _char_embedding(self.char_embedding, self.char_position_embedding, self.vocab_char,
                                        self.vocab_char_mask)
                vocab = F.normalize(vocab, dim=-1)
                vocab_emb += vocab
            if 'bert' in self.fb_featrue and self.char_vocab_decoder:
                if self.vocab_word_ids.device != fwd_results['txt_emb'].device:
                    self.vocab_word_ids = self.vocab_word_ids.to(
                        fwd_results['txt_emb'].device)
                vocab_rem_id = self.vocab_word_ids
                vocab_rem_mask = _get_mask_medpad(vocab_rem_id)
                vocab_rem_type_mask = _get_type_mask_medpad_ocr(vocab_rem_id)
                vocab_bert_out = self.text_bert(
                    txt_inds=vocab_rem_id,
                    txt_mask=vocab_rem_mask,
                    txt_type_mask=vocab_rem_type_mask,
                    adv=False,
                    adv_delta_txt=None,
                )
                vocab_bert_out = self.text_bert_out_linear(vocab_bert_outtext_bert_out[:, :vocab_rem_id.size(1), :])
                vocab_emb += vocab_bert_out

            if self.char_vocab_decoder:
                fixed_ans_emb = 0.5*vocab_emb + self.classifier.weight
            else:
                fixed_ans_emb = self.classifier.weight
        else:
            fixed_ans_emb = self.classifier.weight

        mmt_results = self.mmt(
            txt_emb=fwd_results['txt_emb'],  # fwd_results['final_txt_emb'], #
            # fwd_results['final_txt_mask'], #
            txt_mask=fwd_results['txt_mask'],
            obj_emb=fwd_results['obj_mmt_in'],
            obj_mask=fwd_results['obj_mask'],
            ocr_emb=fwd_results['ocr_mmt_in'],
            ocr_mask=fwd_results['ocr_mask'],
            fixed_ans_emb=fixed_ans_emb,  # self.classifier.module.weight,
            prev_inds=fwd_results['prev_inds'],
            bbox=fwd_results['bbox'],
        )
        fwd_results.update(mmt_results)

    def _forward_output(self, sample_list, fwd_results):
        mmt_dec_output = fwd_results['mmt_dec_output']
        mmt_ocr_output = fwd_results['mmt_ocr_output']
        ocr_mask = fwd_results['ocr_mask']
        #fixed_scores = self.classifier(mmt_dec_output)
        if self.fb_featrue != '':
            vocab_emb = 0
            if 'char' in self.fb_featrue:
                if self.vocab_char.device != mmt_ocr_output.device:
                    self.vocab_char = self.vocab_char.to(mmt_ocr_output.device)
                    self.vocab_char_mask = self.vocab_char_mask.to(
                        mmt_ocr_output.device)
                vocab = _char_embedding(self.char_embedding, self.char_position_embedding, self.vocab_char,
                                        self.vocab_char_mask)
                vocab_emb += vocab
                # vocab = F.normalize(vocab, dim=-1)
                # vocab_emb += self.char_layernorm(vocab)

            if 'bert' in self.fb_featrue:
                if self.vocab_word_ids.device != mmt_ocr_output.device:
                    self.vocab_word_ids = self.vocab_word_ids.to(
                        mmt_ocr_output.device)
                vocab_rem_id = self.vocab_word_ids
                vocab_rem_mask = _get_mask_medpad(vocab_rem_id)
                vocab_rem_type_mask = _get_type_mask_medpad_ocr(vocab_rem_id)

                vocab_bert_out = self.text_bert(
                    txt_inds=vocab_rem_id,
                    txt_mask=vocab_rem_mask,
                    txt_type_mask=vocab_rem_type_mask,
                    bbox = None,
                    adv=False,
                    adv_delta_txt=None
                )
                vocab_bert_out = self.text_bert_out_linear(vocab_bert_out[:, :vocab_rem_id.size(1), :])
                vocab_bert_out = vocab_bert_out.squeeze(1)
                vocab_emb += vocab_bert_out

            if self.vocab_char_fusion == 'sa':
                vocab_scores = self.vocab_scores(mmt_dec_output, vocab_emb)
                # fixed_scores = torch.matmul(mmt_dec_output, self.classifier_weight.transpose(-1, -2)) + self.classifier_bias
                fixed_scores = self.classifier(mmt_dec_output)
                fixed_scores += vocab_scores
            elif self.vocab_char_fusion == 'ea':
                # weight = self.classifier_weight + vocab_emb
                # fixed_scores = torch.matmul(mmt_dec_output, weight.transpose(-1, -2)) + self.classifier_bias
                weight = self.classifier.weight + vocab_emb
                bias = self.classifier.bias
                fixed_scores = F.linear(mmt_dec_output, weight, bias)
            elif self.vocab_char_fusion == 'ec':
                # weight = torch.cat([self.classifier_weight, vocab_emb], dim=-1)
                # weight = self.vocab_ec_linear(weight)
                # fixed_scores = torch.matmul(mmt_dec_output, weight.transpose(-1, -2)) + self.classifier_bias
                weight = torch.cat([self.classifier.weight, vocab_emb], dim=-1)
                weight = self.vocab_ec_linear(weight)
                bias = self.classifier.bias
                fixed_scores = F.linear(mmt_dec_output, weight, bias)
            # fixed_scores /= 2.0
        else:
            fixed_scores = self.classifier(mmt_dec_output)

        dynamic_ocr_scores = self.ocr_ptr_net(
            mmt_dec_output, mmt_ocr_output, ocr_mask
        )

        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)

        fwd_results['scores'] = scores

    def _forward_pretrain_cons_output(self, sample_list, fwd_results):
        ocr_str_len = 150
        ocr_tokens_out = fwd_results['mmt_ocr_output'][:, :ocr_str_len, :]
        rel_ocr_tokens_out = fwd_results['mmt_ocr_output'][:, ocr_str_len:, :]
        assert(rel_ocr_tokens_out.size(1) == 150)
        image_features = ocr_tokens_out.reshape(-1, ocr_tokens_out.size(2))
        text_features = rel_ocr_tokens_out.reshape(
            -1, rel_ocr_tokens_out.size(2))

        # normalized features
        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        # shape = [global_batch_size, global_batch_size]
        fwd_results['contrastive_scores'] = logits_per_image

    def _forward_pretrain_output(self, sample_list, fwd_results):
        # print(fwd_results['mmt_txt_output'].size())
        # [:,sample_list.maxChar:]
        mmt_txt_output = fwd_results['mmt_txt_output']
        # print(mmt_txt_output.size())
        textcls_scores = self.cls(mmt_txt_output)
        fwd_results['textcls_scores'] = textcls_scores

    def _forward_pollute_pretrain_output(self, sample_list, fwd_results):
        # print(fwd_results['mmt_dec_output'].size())
        seq_output = fwd_results['mmt_dec_output'][:, 0]
        pollutecls_scores = self.pollute_cls(seq_output)
        fwd_results['pollutecls_scores'] = pollutecls_scores

    def _forward_overlap_pretrain_output(self, sample_list, fwd_results):
        #ocr_str_len = 100
        mmt_ocr_output = fwd_results['mmt_ocr_output']  # [:,:ocr_str_len,:]
        mmt_obj_output = fwd_results['mmt_obj_output']
        # single sample
        sampled_mmt_ocr_output = batched_index_select(
            mmt_ocr_output, 1, sample_list.overlap_ocr).squeeze(1)
        sampled_mmt_obj_output = batched_index_select(
            mmt_obj_output, 1, sample_list.overlap_obj).squeeze(1)
        overlapcls_scores = self.overlap_cls(
            sampled_mmt_obj_output, sampled_mmt_ocr_output)
        # vector
        # overlapcls_scores = self.overlap_cls(torch.cat([mmt_obj_output,mmt_ocr_output],1), \
        #     batched_index_select(torch.cat([mmt_obj_output,mmt_ocr_output],1), 1, sample_list.overlap_ocr).squeeze(1))
        fwd_results['overlapcls_scores'] = overlapcls_scores

    def _forward_mmt_and_output(self, sample_list, fwd_results):
        if self.training:
            fwd_results['prev_inds'] = sample_list.train_prev_inds.clone()
            self._forward_mmt(sample_list, fwd_results)
            self._forward_output(sample_list, fwd_results)
        else:
            dec_step_num = sample_list.train_prev_inds.size(1)
            # fill prev_inds with BOS_IDX at index 0, and zeros elsewhere
            fwd_results['prev_inds'] = torch.zeros_like(
                sample_list.train_prev_inds
            )
            fwd_results['prev_inds'][:, 0] = self.answer_processor.BOS_IDX

            # greedy decoding at test time
            for t in range(dec_step_num):
                self._forward_mmt(sample_list, fwd_results)
                self._forward_output(sample_list, fwd_results)

                # find the highest scoring output (either a fixed vocab
                # or an OCR), and add it to prev_inds for auto-regressive
                # decoding
                argmax_inds = fwd_results["scores"].argmax(dim=-1)
                fwd_results['prev_inds'][:, 1:] = argmax_inds[:, :-1]
        if self.pretrain:
            self._forward_pretrain_output(sample_list, fwd_results)
            self._forward_pollute_pretrain_output(sample_list, fwd_results)
            self._forward_overlap_pretrain_output(sample_list, fwd_results)
            if self.con_mnt_rel:
                self._forward_pretrain_cons_output(sample_list, fwd_results)

    def get_optimizer_parameters(self, config):
        optimizer_param_groups = []

        base_lr = config.optimizer_attributes.params.lr
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append({
                "params": list(m['module'].parameters()),
                "lr": base_lr * m['lr_scale']
            })
            finetune_params_set.update(list(m['module'].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups

# class LayoutlmModel(BertModel):

#     config_class = LayoutlmConfig
#     pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP
#     base_model_prefix = "bert"

#     def __init__(self, config):
#         super(LayoutlmModel, self).__init__(config)
#         self.embeddings = LayoutlmEmbeddings(config)
#         self.init_weights()

#     def forward(
#         self,
#         input_ids,
#         bbox,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#     ):
#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids)
#         if token_type_ids is None:
#             token_type_ids = torch.zeros_like(input_ids)

#         # We create a 3D attention mask from a 2D tensor mask.
#         # Sizes are [batch_size, 1, 1, to_seq_length]
#         # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
#         # this attention mask is more simple than the triangular masking of causal attention
#         # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

#         # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
#         # masked positions, this operation will create a tensor which is 0.0 for
#         # positions we want to attend and -10000.0 for masked positions.
#         # Since we are adding it to the raw scores before the softmax, this is
#         # effectively the same as removing these entirely.
#         extended_attention_mask = extended_attention_mask.to(
#             dtype=next(self.parameters()).dtype
#         )  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

#         # Prepare head mask if needed
#         # 1.0 in head_mask indicate we keep the head
#         # attention_probs has shape bsz x n_heads x N x N
#         # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
#         # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
#         if head_mask is not None:
#             if head_mask.dim() == 1:
#                 head_mask = (
#                     head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#                 )
#                 head_mask = head_mask.expand(
#                     self.config.num_hidden_layers, -1, -1, -1, -1
#                 )
#             elif head_mask.dim() == 2:
#                 head_mask = (
#                     head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
#                 )  # We can specify head_mask for each layer
#             head_mask = head_mask.to(
#                 dtype=next(self.parameters()).dtype
#             )  # switch to fload if need + fp16 compatibility
#         else:
#             head_mask = [None] * self.config.num_hidden_layers

#         embedding_output = self.embeddings(
#             input_ids, bbox, position_ids=position_ids, token_type_ids=token_type_ids
#         )
#         encoder_outputs = self.encoder(
#             embedding_output, extended_attention_mask, head_mask=head_mask
#         )
#         sequence_output = encoder_outputs[0]
#         pooled_output = self.pooler(sequence_output)

#         outputs = (sequence_output, pooled_output) + encoder_outputs[
#             1:
#         ]  # add hidden_states and attentions if they are here
#         return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class TextBert(LayoutLMv2Model):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = LayoutLMv2Embeddings(config)
        self.encoder = LayoutLMv2Encoder(config)
        self.init_weights()  
        self.post_init()
    
    def torch_int_div(self, tensor1, tensor2):
        """
        A function that performs integer division across different versions of PyTorch.
        """
        return torch.div(tensor1, tensor2, rounding_mode="floor")
    
    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + spatial_position_embeddings + token_type_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_img_embeddings(self, image, bbox, position_ids):
        visual_embeddings = self.visual_proj(self.visual(image))
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)
        embeddings = visual_embeddings + position_embeddings + spatial_position_embeddings
        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.visual_dropout(embeddings)
        return embeddings

    def _calc_visual_bbox(self, image_feature_pool_shape, bbox, device, final_shape):
        visual_bbox_x = self.torch_int_div(
            torch.arange(
                0,
                1000 * (image_feature_pool_shape[1] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            ),
            self.config.image_feature_pool_shape[1],
        )
        visual_bbox_y = self.torch_int_div(
            torch.arange(
                0,
                1000 * (self.config.image_feature_pool_shape[0] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            ),
            self.config.image_feature_pool_shape[0],
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(image_feature_pool_shape[0], 1),
                visual_bbox_y[:-1].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(image_feature_pool_shape[0], 1),
                visual_bbox_y[1:].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, bbox.size(-1))

        visual_bbox = visual_bbox.repeat(final_shape[0], 1, 1)

        return visual_bbox

    def forward(
        self, 
        txt_inds, 
        txt_mask, 
        adv=False,  
        image=None,
        txt_type_mask=None, 
        bbox=None, 
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None, 
        output_hidden_states=None, 
        return_dict=None,
        adv_delta_txt=None, 
        adv_delta_ocr=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if bbox == None:
            bbox = torch.zeros([txt_inds.size(0), txt_inds.size(1), 4])
            bbox = bbox.long().to(txt_inds.device)
        else:
            bbox[:,:, 0] = torch.clamp(bbox[:,:, 0],min = 0, max = 1023).to(txt_inds.device)
            bbox[:,:, 1] = torch.clamp(bbox[:,:, 1],min = 0, max = 1023).to(txt_inds.device)
            bbox[:,:, 2] = torch.clamp(bbox[:,:, 2],min = 0, max = 1023).to(txt_inds.device)
            bbox[:,:, 3] = torch.clamp(bbox[:,:, 3],min = 0, max = 1023).to(txt_inds.device)
            bbox = bbox.long().to(txt_inds.device)

        input_shape = txt_inds.size()
        device = txt_inds.device

        visual_shape = list(input_shape)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
        visual_shape = torch.Size(visual_shape)
        # needs a new copy of input_shape for tracing. Otherwise wrong dimensions will occur
        final_shape = list(input_shape)
        final_shape[1] += visual_shape[1]
        final_shape = torch.Size(final_shape)

        visual_bbox = self._calc_visual_bbox(self.config.image_feature_pool_shape, bbox, device, final_shape)
        final_bbox = torch.cat([bbox, visual_bbox], dim=1)

        attention_mask = txt_mask
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        visual_attention_mask = torch.ones(visual_shape, device=device)
        final_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)

        token_type_ids = txt_type_mask
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.expand(input_shape)

        visual_position_ids = torch.arange(0, visual_shape[1], dtype=torch.long, device=device).repeat(
            input_shape[0], 1
        )
        final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)

        if bbox is None:
            bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

        text_layout_emb = self._calc_text_embeddings(
            input_ids=txt_inds,
            bbox=bbox,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        if image is None:
            visual_emb = torch.zeros([text_layout_emb.size(0),49,text_layout_emb.size(2)], device=device)
            if adv == True:
                if adv_delta_txt != None:
                    text_layout_emb = text_layout_emb + adv_delta_txt
                elif adv_delta_ocr != None:
                    text_layout_emb = text_layout_emb + adv_delta_ocr
        else:
            visual_emb = self._calc_img_embeddings(
                image=image,
                bbox=visual_bbox,
                position_ids=visual_position_ids,
            )
        
        final_emb = torch.cat([text_layout_emb, visual_emb], dim=1)
        
        # encoder_inputs = self.embeddings(
        #     txt_inds, bbox=bbox, token_type_ids=txt_type_mask)

        extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers
        # bbox, position_id = self.make_positions_ids(bbox, txt_inds)
        # if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
            # if self.config.has_spatial_attention_bias:
                # final_bbox = bbox
            # if self.config.has_relative_attention_bias:
                # position_ids = self.embeddings.position_ids[:, : input_shape[1]]
                # position_ids = position_ids.expand_as(txt_inds)
                # final_position_ids = position_ids
        encoder_outputs = self.encoder(
            final_emb,
            attention_mask=extended_attention_mask,
            bbox=final_bbox,
            position_ids=final_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            head_mask=head_mask,
        )
        sequence_output = encoder_outputs[0]
        # pooled_output = self.pooler(sequence_output)

        # outputs = (sequence_output, pooled_output) + encoder_outputs[
        #     1:
        # ]  # add hidden_states and attentions if they are here

        return sequence_output

# class TextBert(BertPreTrainedModel):

#     config_class = LayoutlmConfig
#     pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP
#     base_model_prefix = "bert"

#     def __init__(self, config):
#         super(TextBert, self).__init__(config)
#         self.embeddings = LayoutlmEmbeddings(config)
#         self.encoder = BertEncoder(config)
#         self.init_weights()  
        
#     def forward(self, txt_inds, bbox, adv, txt_mask, txt_type_mask, head_mask=None, adv_delta_txt=None):
#         attention_mask = txt_mask
#         token_type_ids = txt_type_mask
#         if bbox == None:
#             bbox = torch.zeros([txt_inds.size(0), txt_inds.size(1), 4])
#         bbox = bbox.long().to(txt_inds.device)
#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids)
#         if token_type_ids is None:
#             token_type_ids = torch.zeros_like(input_ids)

#         # We create a 3D attention mask from a 2D tensor mask.
#         # Sizes are [batch_size, 1, 1, to_seq_length]
#         # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
#         # this attention mask is more simple than the triangular masking of causal attention
#         # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

#         # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
#         # masked positions, this operation will create a tensor which is 0.0 for
#         # positions we want to attend and -10000.0 for masked positions.
#         # Since we are adding it to the raw scores before the softmax, this is
#         # effectively the same as removing these entirely.
#         extended_attention_mask = extended_attention_mask.to(
#             dtype=next(self.parameters()).dtype
#         )  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#         assert not extended_attention_mask.requires_grad
#         # Prepare head mask if needed
#         # 1.0 in head_mask indicate we keep the head
#         # attention_probs has shape bsz x n_heads x N x N
#         # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
#         # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
#         if head_mask is not None:
#             if head_mask.dim() == 1:
#                 head_mask = (
#                     head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#                 )
#                 head_mask = head_mask.expand(
#                     self.config.num_hidden_layers, -1, -1, -1, -1
#                 )
#             elif head_mask.dim() == 2:
#                 head_mask = (
#                     head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
#                 )  # We can specify head_mask for each layer
#             head_mask = head_mask.to(
#                 dtype=next(self.parameters()).dtype
#             )  # switch to fload if need + fp16 compatibility
#         else:
#             head_mask = [None] * self.config.num_hidden_layers
#         encoder_inputs = self.embeddings(
#             txt_inds, token_type_ids=txt_type_mask, bbox=bbox)
#         if adv == True:
#             encoder_inputs = encoder_inputs + adv_delta_txt
        

#         # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
#         # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#         # head_mask = [None] * self.config.num_hidden_layers

#         encoder_outputs = self.encoder(
#             encoder_inputs,
#             extended_attention_mask,
#             head_mask=head_mask
#         )
#         seq_output = encoder_outputs[0]

#         return seq_output


class MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self,
                txt_emb,
                txt_mask,
                obj_emb,
                obj_mask,
                ocr_emb,
                ocr_mask,
                fixed_ans_emb,
                prev_inds,
                bbox=None):
        # Feature: TXT [32, 20, 768]) OBJ [32, 100, 768]) OCR[32, 50, 768]) Answer torch.Size([5000, 768]
        # mask: TXT [32, 20] OBJ [32, 100] OCR [32, 50]

        # build embeddings for predictions in previous decoding steps
        # fixed_ans_emb is an embedding lookup table for each fixed vocabulary
        dec_emb = self.prev_pred_embeddings(fixed_ans_emb, ocr_emb, prev_inds)

        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        dec_mask = torch.zeros(
            dec_emb.size(0),
            dec_emb.size(1),
            dtype=torch.float32,
            device=dec_emb.device
        )
        encoder_inputs = torch.cat(
            [txt_emb, obj_emb, ocr_emb, dec_emb],
            dim=1
        )
        attention_mask = torch.cat(
            [txt_mask, obj_mask, ocr_mask, dec_mask],
            dim=1
        )

        # offsets of each modality in the joint embedding space
        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        dec_max_num = dec_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num
        ocr_begin = txt_max_num + obj_max_num
        ocr_end = ocr_begin + ocr_max_num

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )
        # decoding step elements can attend to themselves in a causal manner
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = \
            _get_causal_mask(dec_max_num, encoder_inputs.device)

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_txt_output = mmt_seq_output[:, txt_begin:txt_end]
        mmt_obj_output = mmt_seq_output[:, txt_end:ocr_begin]
        mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]

        results = {
            'mmt_seq_output': mmt_seq_output,
            'mmt_txt_output': mmt_txt_output,
            'mmt_obj_output': mmt_obj_output,
            'mmt_ocr_output': mmt_ocr_output,
            'mmt_dec_output': mmt_dec_output,
        }
        return results


class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask=None):
        if attention_mask != None:
            extended_attention_mask = (1.0 - attention_mask) * -10000.0
            assert extended_attention_mask.dim() == 2
            extended_attention_mask = extended_attention_mask.unsqueeze(1)
        #extended_attention_mask = (1.0 - attention_mask) * -10000.0
        #assert extended_attention_mask.dim() == 2
        #extended_attention_mask = extended_attention_mask.unsqueeze(1)

        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)

        scores = torch.matmul(
            query_layer,
            key_layer.transpose(-1, -2)
        )
        scores = scores / math.sqrt(self.query_key_size)
        #scores = scores + extended_attention_mask
        if attention_mask != None:
            scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)

        return scores


class PrevPredEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        MAX_DEC_LENGTH = 100
        MAX_TYPE_NUM = 5
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)
        self.token_type_embeddings = nn.Embedding(MAX_TYPE_NUM, hidden_size)

        self.ans_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.ocr_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, ans_emb, ocr_emb, prev_inds):
        assert prev_inds.dim() == 2 and prev_inds.dtype == torch.long
        assert ans_emb.dim() == 2

        batch_size = prev_inds.size(0)
        seq_length = prev_inds.size(1)
        ans_num = ans_emb.size(0)

        # apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        ans_emb = self.ans_layer_norm(ans_emb)
        ocr_emb = self.ocr_layer_norm(ocr_emb)
        assert ans_emb.size(-1) == ocr_emb.size(-1)
        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)
        raw_dec_emb = _batch_gather(ans_ocr_emb_cat, prev_inds)

        # Add position and type embedding for previous predictions
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=ocr_emb.device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        # Token type ids: 0 -- vocab; 1 -- OCR
        token_type_ids = prev_inds.ge(ans_num).long()
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = position_embeddings + token_type_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = raw_dec_emb + embeddings
        return dec_emb

# pad inside each kinds


def _get_mask_medpad(token_id):
    return (token_id != 0).float()

# TMP: slow implementation


def _get_type_mask_medpad(token_id):
    # same type for ocr and obj tags to avoid out of bert pretrain vocab; test later
    type_mask = torch.ones(token_id.shape).cuda() * 1.
    type_mask[:, :20] = 0
    return type_mask.long()


def _get_type_mask_medpad_ocr(token_id):
    # same type for ocr and obj tags to avoid out of bert pretrain vocab; test later
    type_mask = torch.ones(token_id.shape).cuda() * 1.
    return type_mask.long()

# pad at the end; used anyway by obj, ocr mmt encode


def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    # non_pad_mask[:,0]=1.0
    return non_pad_mask


@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i+1):
            mask[i, j] = 1.
    return mask


def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size*length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results


"""
From VilBert, vilbert/vilbert
"""


class BertLMPredictionHead(nn.Module):
    def __init__(self, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform()

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(
            bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self):
        super(BertPredictionHeadTransform, self).__init__()
        hidden_act = "gelu"
        hidden_size = 768
        ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[hidden_act]
        else:
            self.transform_act_fn = hidden_act
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class PolluteLinear(nn.Module):
    def __init__(self, input_size=768, hidden_size=512):
        super(PolluteLinear, self).__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, 1)

    def forward(self, x):
        hidden_state = self.LayerNorm(gelu(self.dense(x)))
        return self.decoder(hidden_state)


class OverlapLinear(nn.Module):
    def __init__(self, input_size=768, hidden_size=512):
        super(OverlapLinear, self).__init__()
        self.dense = nn.Linear(input_size*2, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, 1)

    def forward(self, x, y):
        fuse = torch.cat([x, y], -1)
        hidden_state = self.LayerNorm(gelu(self.dense(fuse)))
        return self.decoder(hidden_state)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


def _char_embedding(char_embedding, char_position_embedding, _ocr_char, ocr_char_mask, mean=True):
    ocr_char_emb = char_embedding(_ocr_char)
    pos_idx = torch.arange(ocr_char_emb.size(-2), device=ocr_char_emb.device)
    pos_emb = char_position_embedding(pos_idx)

    _dim = ocr_char_emb.dim() - pos_emb.dim()
    if _dim == 2:
        pos_emb = pos_emb.unsqueeze(0).unsqueeze(0)
    elif _dim == 1:
        pos_emb = pos_emb.unsqueeze(0)
    else:
        assert False
    ocr_char_emb += pos_emb
    ocr_char_emb *= ocr_char_mask.unsqueeze(-1)
    if mean:
        ocr_char_emb = ocr_char_emb.mean(dim=-2)
    return ocr_char_emb
