# Copyright (c) Facebook, Inc. and its affiliates.
# -*- coding: UTF-8 -*-
import gc
import os
import math
import time
import random
import numpy as np
import json

import torch
from torch import optim
# from torch.cuda.amp import autocast, GradScaler
# from apex import amp
import torch.nn.functional as F
from tqdm import tqdm
import datetime

from pythia.common.meter import Meter
from pythia.common.registry import registry
from pythia.common.report import Report
from pythia.common.dataset_loader import DatasetLoader
from pythia.utils.build_utils import build_model, build_optimizer
from pythia.utils.llama2_gen import build_llama2
from pythia.utils.llava_gen import add_details
from pythia.utils.checkpoint import Checkpoint
from pythia.utils.distributed_utils import (broadcast_scalar, is_main_process,
                                            reduce_dict, synchronize)
from pythia.utils.early_stopping import EarlyStopping
from pythia.utils.general import clip_gradients, lr_lambda_update
from pythia.utils.logger import Logger
from pythia.utils.timer import Timer

from pythia.modules import losses as all_losses
from pythia.models.tap import M4C, TextBert
from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder
)
from transformers import (
    AutoTokenizer, AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig,
    BlipForConditionalGeneration, T5ForConditionalGeneration, T5Tokenizer, 
    logging
)


logging.set_verbosity_warning()

@registry.register_trainer('base_trainer')
class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.profiler = Timer()

    def load(self):
        self._init_process_group()

        self.run_type = self.config.training_parameters.get("run_type", "train")
        self.dataset_loader = DatasetLoader(self.config)
        self._datasets = self.config.datasets

        self.writer = Logger(self.config)
        registry.register("writer", self.writer)

        self.configuration = registry.get("configuration")
        self.configuration.pretty_print()

        self.config_based_setup()

        self.load_task()
        self.load_model()
        self.load_optimizer()
        self.load_extras()

    def _init_process_group(self):
        training_parameters = self.config.training_parameters
        self.local_rank = training_parameters.local_rank
        self.device = training_parameters.device

        if self.local_rank is not None and training_parameters.distributed:
            if not torch.distributed.is_nccl_available():
                raise RuntimeError(
                    "Unable to initialize process group: NCCL is not available"
                )
            torch.distributed.init_process_group(backend="nccl",timeout=datetime.timedelta(seconds=5400))
            synchronize()

        if (
            "cuda" in self.device
            and training_parameters.distributed
            and self.local_rank is not None
        ):
            self.device = torch.device("cuda", self.local_rank)

        registry.register("current_device", self.device)

    def load_task(self):
        self.writer.write("Loading datasets", "info")
        self.dataset_loader.load_datasets()

        self.train_dataset = self.dataset_loader.train_dataset
        self.val_dataset = self.dataset_loader.val_dataset

        # Total iterations for snapshot
        self.snapshot_iterations = len(self.val_dataset)
        self.snapshot_iterations //= self.config.training_parameters.batch_size

        self.test_dataset = self.dataset_loader.test_dataset

        self.train_loader = self.dataset_loader.train_loader
        self.val_loader = self.dataset_loader.val_loader
        self.test_loader = self.dataset_loader.test_loader

    def load_model(self):
        attributes = self.config.model_attributes[self.config.model]
        # Easy way to point to config for other model
        if isinstance(attributes, str):
            attributes = self.config.model_attributes[attributes]

        attributes["model"] = self.config.model

        self.dataset_loader.update_registry_for_model(attributes)
        # 加载llama2
        if self.config.training_parameters.use_llama2:
            print("Load Llama2")
            self.model_llama2, self.tokenizer= build_llama2(self.config.training_parameters.llama2_path)
            # self.model_llama2.to(self.device)
            # self.model_llama2.resize_token_embeddings(self.model_llama2.config.vocab_size + 1) 
        # 加载caption&qa生成模型
        if self.config.training_parameters.caption_gen:
            print("Load Blip for caption")
            model_blip = BlipForConditionalGeneration.from_pretrained(self.config.training_parameters.blip_caption_large_path, torch_dtype=torch.float32)
            self.caption_processor = AutoProcessor.from_pretrained(self.config.training_parameters.blip_caption_large_path)
            # caption_model.eval()
            self.model_blip = model_blip.to(self.device)
        
        if self.config.training_parameters.caption_gen_with_vlm:
            kwargs = {"device_map": "auto"}
            # default to load_4bit
            load_8bit=False
            load_4bit=True
            if load_8bit:
                kwargs['load_in_8bit'] = True
            elif load_4bit:
                kwargs['load_in_4bit'] = True
                kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                )
            else:
                kwargs['torch_dtype'] = torch.float16
            print("Load vlm for caption")
            tokenizer_vlm = AutoTokenizer.from_pretrained(self.config.training_parameters.vlm_path, use_fast=False)
            model_vlm = AutoModelForCausalLM.from_pretrained(self.config.training_parameters.vlm_path, low_cpu_mem_usage=True, **kwargs)
            
            self.tokenizer_vlm, self.model_vlm, self.image_processor, self.context_len = add_details(tokenizer_vlm,model_vlm)
        
        if self.config.training_parameters.qa_gen:
            print("Load T5 for QA generation")
            self.tokenizer_t5 = T5Tokenizer.from_pretrained(
                self.config.training_parameters.t5_large_lm_adapt_path
            )
            self.model_t5 = T5ForConditionalGeneration.from_pretrained(
                self.config.training_parameters.t5_large_lm_adapt_path
            ).to(self.device)
            cache_file = self.config.training_parameters.T5_large_QG_path
            checkpoint = torch.load(cache_file, map_location=self.device)
            state_dict = checkpoint["model"]
            self.model_t5.load_state_dict(state_dict)
        self.model = build_model(attributes)
        self.dataset_loader.clean_config(attributes)
        training_parameters = self.config.training_parameters

        data_parallel = training_parameters.data_parallel
        distributed = training_parameters.distributed

        registry.register("data_parallel", data_parallel)
        registry.register("distributed", distributed)

        if "cuda" in str(self.config.training_parameters.device):
            rank = self.local_rank if self.local_rank is not None else 0
            device_info = "CUDA Device {} is: {}".format(
                rank, torch.cuda.get_device_name(self.local_rank)
            )

            self.writer.write(device_info, log_all=True)

        self.model = self.model.to(self.device)

        self.writer.write("Torch version is: " + torch.__version__)

        if (
            "cuda" in str(self.device)
            and torch.cuda.device_count() > 1
            and data_parallel is True
        ):
            # self.model = torch.nn.DataParallel(self.model)
            torch.cuda.set_device(self.local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                check_reduction=True, find_unused_parameters=True
            )
            if self.config.training_parameters.caption_gen:
                # self.model_llama2 = torch.nn.parallel.DistributedDataParallel(
                #     self.model_llama2.cuda(), device_ids=[self.local_rank], output_device=self.local_rank,
                #     check_reduction=True, find_unused_parameters=True
                # )
                self.model_blip = torch.nn.parallel.DistributedDataParallel(
                    self.model_blip, device_ids=[self.local_rank], output_device=self.local_rank,
                    check_reduction=True, find_unused_parameters=True
                )

            if self.config.training_parameters.caption_gen_with_vlm:
                self.model_vlm = torch.nn.parallel.DistributedDataParallel(
                        self.model_vlm.cuda(), device_ids=[self.local_rank], output_device=self.local_rank,
                        check_reduction=True, find_unused_parameters=True
                    )

            if self.config.training_parameters.qa_gen:
                self.model_t5 = torch.nn.parallel.DistributedDataParallel(
                    self.model_t5, device_ids=[self.local_rank], output_device=self.local_rank,
                    check_reduction=True, find_unused_parameters=True
                )

        if (
            "cuda" in str(self.device)
            and self.local_rank is not None
            and distributed is True
        ):
            torch.cuda.set_device(self.local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                check_reduction=True, find_unused_parameters=True
            )
            if self.config.training_parameters.caption_gen:
                self.model_blip = torch.nn.parallel.DistributedDataParallel(
                        self.model_blip, device_ids=[self.local_rank], output_device=self.local_rank,
                        check_reduction=True, find_unused_parameters=True
                    )

            if self.config.training_parameters.caption_gen_with_vlm:
                self.model_vlm = torch.nn.parallel.DistributedDataParallel(
                        self.model_vlm.cuda(), device_ids=[self.local_rank], output_device=self.local_rank,
                        check_reduction=True, find_unused_parameters=True
                    )

            if self.config.training_parameters.qa_gen:
                # self.model_llama2 = torch.nn.parallel.DistributedDataParallel(
                #     self.model_llama2.cuda(), device_ids=[self.local_rank], output_device=self.local_rank,
                #     check_reduction=True, find_unused_parameters=True
                # )
                
                self.model_t5 = torch.nn.parallel.DistributedDataParallel(
                    self.model_t5, device_ids=[self.local_rank], output_device=self.local_rank,
                    check_reduction=True, find_unused_parameters=True
            )

    def load_optimizer(self):
        self.optimizer = build_optimizer(self.model, self.config)

    def load_extras(self):
        self.checkpoint = Checkpoint(self)
        self.meter = Meter()

        self.training_parameters = self.config.training_parameters

        monitored_metric = self.training_parameters.monitored_metric
        metric_minimize = self.training_parameters.metric_minimize
        should_early_stop = self.training_parameters.should_early_stop
        patience = self.training_parameters.patience

        self.log_interval = self.training_parameters.log_interval
        self.snapshot_interval = self.training_parameters.snapshot_interval
        self.max_iterations = self.training_parameters.max_iterations
        self.should_clip_gradients = self.training_parameters.clip_gradients
        self.max_epochs = self.training_parameters.max_epochs
        self.batch_size = self.training_parameters.batch_size

        self.early_stopping = EarlyStopping(
            self.model,
            self.checkpoint,
            monitored_metric,
            patience=patience,
            minimize=metric_minimize,
            should_stop=should_early_stop,
        )
        self.current_epoch = 0
        self.current_iteration = 0
        self.checkpoint.load_state_dict()

        self.not_debug = self.training_parameters.logger_level != "debug"

        self.lr_scheduler = None

        # TODO: Allow custom scheduler
        if self.training_parameters.lr_scheduler is True:
            scheduler_class = optim.lr_scheduler.LambdaLR
            scheduler_func = lambda x: lr_lambda_update(x, self.config)
            self.lr_scheduler = scheduler_class(
                self.optimizer, lr_lambda=scheduler_func
            )

    def config_based_setup(self):
        seed = self.config.training_parameters.seed
        if seed is None:
            return

        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self):
        self.writer.write("===== Model =====")
        self.writer.write(self.model)

        if "train" not in self.run_type:
            self.inference()
            return

        should_break = False

        if self.max_epochs is None:
            self.max_epochs = math.inf
        else:
            self.max_iterations = math.inf

        self.model.train()
        # self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        # ===================STEP1 定义一个分支参考jocor设置================
        # self.model2 = self.model
        # self.model2.train()

        # define drop rate schedule STEP1.5
        forget_rate = 0.1 # 外面定义
        num_gradual = 1 # 外面定义
        exponent = 1 # 外面定义
        if self.max_iterations == math.inf:
            self.rate_schedule = np.ones(self.max_epochs) * forget_rate
            self.rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)
        else:
            self.rate_schedule = np.ones(self.max_iterations) * forget_rate
            self.rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)

        self.train_timer = Timer()
        self.snapshot_timer = Timer()

        self.profile("Setup Time")
        # 加载llama2
        if self.config.training_parameters.use_llama2:
            print("Load Llama2")
            self.model_llama2, self.tokenizer= build_llama2(self.config.training_parameters.llama2_path)
            self.model_llama2.eval()
            self.model_llama2.resize_token_embeddings(self.model_llama2.config.vocab_size + 1) 
        # 加载caption&qa生成模型
        if self.config.training_parameters.qa_gen:
            print("Load T5 for QA generation")



            # self.use_llama2 = True

        torch.autograd.set_detect_anomaly(False)
        self.writer.write("Starting training...")
        # quick hack for amp delay_unscale bug
        self.optimizer.zero_grad()
        self.optimizer.step()
        # scaler = GradScaler()  # 梯度缩放
        # scaler.step(self.optimizer)
        # accum_iter = 8  # 梯度累加
        while self.current_iteration < self.max_iterations and not should_break:
            self.current_epoch += 1
            registry.register("current_epoch", self.current_epoch)

            # Seed the sampler in case if it is distributed
            self.dataset_loader.seed_sampler("train", self.current_epoch)

            if self.current_epoch > self.max_epochs:
                break

            for batch in self.train_loader:
                self.profile("Batch load time")
                self.current_iteration += 1
                self.writer.write(self.current_iteration, "debug")

                registry.register("current_iteration", self.current_iteration)

                if self.current_iteration > self.max_iterations:
                    break
                # ============================ Code for adversarial training =============
                # batch['adv'] = self.config['model_attributes']['adv'] ?????????为啥
                # print(list(batch.keys()))
                if self.config.training_parameters.adv_training:
                    batch['adv'] = True
                    # =====================Forward====================
                    prepared_batch = self.dataset_loader.prepare_batch(batch)
                    # prepared_batch2 = prepared_batch
                    self.profile("Batch prepare time")
                    # =====TARGET1: 获取BERT的embedding层将cmb_txt提前编码======
                    if self.config.training_parameters.adv_text:
                        cmb_text = prepared_batch['cmb_text']
                        txt_embeds_init = torch.zeros([cmb_text.size(0), cmb_text.size(1), 768]) #[x,170,768]
                        txt_delta = torch.zeros_like(txt_embeds_init)

                    # =====TARGET2: 获取视觉对象编码======
                    if self.config.training_parameters.adv_img:
                        cmb_text = prepared_batch['cmb_text']
                        img_embeds_init = torch.zeros([cmb_text.size(0), 197, 768])
                        img_delta = torch.zeros_like(img_embeds_init)

                    # (6.7)=====TARGET2.5: 获取OCR对象feature（运算量巨大，降低batch_size :-(） ocr_mmt_in + 对抗干扰======
                    # 获取OCR FastText 特征（300维）
                    # ocr_fasttext_embeds_init = prepared_batch['context_feature_0']
                    # ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
                    # assert ocr_fasttext_embeds_init.size(-1) == 300

                    # 获取OCR PHOC 特征（604维）
                    # ocr_phoc_embeds_init = prepared_batch['context_feature_1']
                    # ocr_phoc = F.normalize(ocr_phoc, dim=-1)
                    # assert ocr_phoc_embeds_init.size(-1) == 604

                    # 获取OCR Faster RCNN特征
                    # ocr_fc6 = prepared_batch['image_feature_1'][:, :ocr_fasttext_embeds_init.size(1), :]

                    # 获取embedding进去的bert ocr特征
                    if self.config.training_parameters.adv_ocr:
                        ocr_word_id = prepared_batch['word_ids']
                        ocr_embeds_init = torch.zeros([ocr_word_id.size(0), ocr_word_id.size(1), 768])
                        ocr_delta = torch.zeros_like(ocr_embeds_init)

                    # =====TARGET3: 初始化delta为zero向量======
                    # ocr_fasttext_delta = torch.zeros_like(ocr_fasttext_embeds_init)
                    # ocr_phoc_delta = torch.zeros_like(ocr_phoc_embeds_init)
                    # else:
                    #     print("no adv!")
                    #     break

                    # =====TARGET4: 计算样本的prob. scores，这里的分数是最初的分数，取下巧，用model_output======
                    # Arguments should be a dict at this point
                    prepared_batch['adv'] = False
                    # prepared_batch2['adv'] = False
                    model_output = self.model(prepared_batch)
                    # =====TARGET7-8: 计算BCE Loss + KL loss（需要传到losses做处理）======
                    
                    # ===================STEP2 另一个model分支重写计算一次（不知道会不会爆显。。）================
                    # model2_output = self.model2(prepared_batch2)
                    # Origin
                    if "textcls_scores" in list(model_output.keys()):
                        gt_answer_scores = model_output['scores'] #[x,12,5200]
                        # print("1:",gt_answer_scores.shape)
                        prepared_batch['gt_answer_prob'] = F.softmax(gt_answer_scores, dim=1)
                        prepared_batch['gt_answer_logprob'] = F.log_softmax(gt_answer_scores, dim=1)
                        # MLM_CLS
                        gt_textcls_scores = model_output["textcls_scores"].permute(0,2,1) #[x,35022,170]
                        # print("2:",gt_textcls_scores.shape)
                        prepared_batch['gt_textcls_prob'] = F.softmax(gt_textcls_scores, dim=1)
                        prepared_batch['gt_textcls_logprob'] = F.log_softmax(gt_textcls_scores, dim=1)
                    else:
                        gt_answer_scores = model_output['scores']
                        prepared_batch['gt_answer_prob'] = F.softmax(gt_answer_scores, dim=1)
                        prepared_batch['gt_answer_logprob'] = F.log_softmax(gt_answer_scores, dim=1)

                    # =====TARGET5: 对抗训练主循环，step超参数暂时设定为3======
                    adv_step = 3
                    for astep in range(3):
                        # requires_grad_
                        if self.config.training_parameters.adv_text:
                            txt_delta.requires_grad_()
                        if self.config.training_parameters.adv_img:
                            img_delta.requires_grad_()
                        if self.config.training_parameters.adv_ocr:
                            ocr_delta.requires_grad_()
                        # ocr_fasttext_delta.requires_grad_()
                        # ocr_phoc_delta.requires_grad_()
                    # =====TARGET6: 对原有特征加入对抗干扰需要在原来的模型中进行修改======
                        # prepared_batch['adv'] = True # 之后尽量加一个if语句进行判断
                        # prepared_batch['adv_delta_txt'] = txt_delta # txt
                        # prepared_batch['adv_delta_obj'] = 0 # img
                        # prepared_batch['adv_delta_ocr_fasttext'] = 0
                        # prepared_batch['adv_delta_ocr_phoc'] = 0
                        # prepared_batch['adv_delta_ocr'] = 0
                        # # prepared_batch['adv_step'] = 3
                        # model_adv_output1 = self.model(prepared_batch)
                    # ===================STEP3 另一个model分支重写计算一次===============
                        # self.writer.write(prepared_batch)
                        # model2_adv_output = self.model2(prepared_batch)
                    # =====TARGET7-8: 计算BCE Loss + KL loss（需要传到losses做处理）======
                        # report_adv = Report(prepared_batch, model_adv_output1) # report获取加入干扰后的loss
                    # ===================STEP4 更新损失================
                        # report_adv2 = Report(prepared_batch, model2_adv_output)
                        
                        # self.writer.write("=============new loss=================")
                        # self.writer.write(report_adv.losses)
                    # =====================update_meter====================
                    # =====TARGET9: 更新损失======
                        # self._update_meter(report, self.meter)
                        # self._update_meter(report_adv, self.meter)
                        # loss1 = self._extract_loss(report_adv) 
                    # ====再来一次====
                        prepared_batch['adv'] = True # 之后尽量加一个if语句进行判断
                        if self.config.training_parameters.adv_text:
                            prepared_batch['adv_delta_txt'] = txt_delta # txt
                        else:
                            prepared_batch['adv_delta_txt'] = 0
                        if self.config.training_parameters.adv_img:
                            prepared_batch['adv_delta_obj'] = img_delta # img
                        else:
                            prepared_batch['adv_delta_obj'] = 0 # img
                        if self.config.training_parameters.adv_ocr:
                            prepared_batch['adv_delta_ocr'] = ocr_delta # ocr
                        else:
                            prepared_batch['adv_delta_ocr'] = 0# ocr
                        prepared_batch['adv_delta_ocr_fasttext'] = 0
                        prepared_batch['adv_delta_ocr_phoc'] = 0
                        
                        # prepared_batch['adv_step'] = 3
                        model_adv_output2 = self.model(prepared_batch)
                        report_adv2 = Report(prepared_batch, model_adv_output2) # report获取加入干扰后的loss
                        self.profile("Forward time")
                        self._update_meter(report_adv2, self.meter)
                        loss2 = self._extract_loss(report_adv2) 

                        loss = loss2 / adv_step

                        # 警告：正式训练开始！
                    # =====TARGET10: 回溯(先不混合精度)======
                        loss.backward(retain_graph=True)
                        # scaler.scale(loss).backward(retain_graph=True)
                        # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        #     scaled_loss.backward(retain_graph=True)

                    # =====TARGET11: 获取delta梯度======
                        if self.config.training_parameters.adv_text:
                            txt_delta_grad = txt_delta.grad.clone().detach().float()  # [16, 170, 768]

                        # img_delta梯度
                        if self.config.training_parameters.adv_img:
                            img_delta_grad = img_delta.grad.clone().detach().float()  # [?]

                        # ocr_delta(特征)梯度
                        if self.config.training_parameters.adv_ocr:
                            ocr_delta_grad = ocr_delta.grad.clone().detach().float()  # [?]

                        # ocr fasttext梯度
                        # ocr_fasttext_delta_grad = ocr_fasttext_delta.grad.clone().detach().float()  # [?]

                        # ocr phoc梯度
                        # ocr_phoc_delta_grad = ocr_phoc_delta.grad.clone().detach().float()  # [?]
                    # =====TARGET12: 更新剪裁======
                        # # txt
                        if self.config.training_parameters.adv_text:
                            denorm_txt = torch.norm(txt_delta_grad.view(txt_delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                            denorm_txt = torch.clamp(denorm_txt, min=1e-8)
                            txt_delta_step = (1e-3 * txt_delta_grad / denorm_txt).to(txt_delta)  # opts.adv_lr_txt
                            txt_delta = (txt_delta + txt_delta_step).detach()
                        # img
                        if self.config.training_parameters.adv_img:
                            denorm_img = torch.norm(img_delta_grad.view(img_delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                            denorm_img = torch.clamp(denorm_img, min=1e-8)
                            img_delta_step = (1e-3 * img_delta_grad / denorm_img).to(img_delta)  # opts.adv_lr_txt
                            img_delta = (img_delta + img_delta_step).detach()
                        # ocr
                        if self.config.training_parameters.adv_ocr:
                            denorm_ocr = torch.norm(ocr_delta_grad.view(ocr_delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                            denorm_ocr = torch.clamp(denorm_ocr, min=1e-8)
                            ocr_delta_step = (1e-3 * ocr_delta_grad / denorm_ocr).to(ocr_delta)  # opts.adv_lr_txt
                            ocr_delta = (ocr_delta + ocr_delta_step).detach()
                        # ocr fasttext
                        # denorm_ocr_fasttext = torch.norm(ocr_fasttext_delta_grad.view(ocr_fasttext_delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                        # denorm_ocr_fasttext = torch.clamp(denorm_ocr_fasttext, min=1e-8)
                        # ocr_fasttext_delta_step = (1e-3 * ocr_fasttext_delta_grad / denorm_ocr_fasttext).to(ocr_fasttext_delta)  # opts.adv_lr_txt
                        # ocr_fasttext_delta = (ocr_fasttext_delta + ocr_fasttext_delta_step).detach()
                        # ocr phoc
                        # denorm_ocr_phoc = torch.norm(ocr_phoc_delta_grad.view(ocr_phoc_delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                        # denorm_ocr_phoc = torch.clamp(denorm_ocr_phoc, min=1e-8)
                        # ocr_phoc_delta_step = (1e-3 * ocr_phoc_delta_grad / denorm_ocr_phoc).to(ocr_phoc_delta)  # opts.adv_lr_txt
                        # ocr_phoc_delta = (ocr_phoc_delta + ocr_phoc_delta_step).detach()

                else:
                    batch['adv'] = False
                    report = self._forward_pass(batch)
                    self._update_meter(report, self.meter)
                    loss = self._extract_loss(report)
                    self._backward(loss)
                    should_break = self._logistics(report)
                    if should_break:
                        break
                    continue
                # ============================ End ================= 
                # if (self.current_iteration % accum_iter == 0) or (self.current_iteration == self.max_iterations):
                if self.should_clip_gradients:
                    clip_gradients(
                        self.model, self.current_iteration, self.writer, self.config)
                self.optimizer.step()
                self._run_scheduler()
                self.profile("Backward time")
                self.optimizer.zero_grad()
                should_break = self._logistics(report_adv2)
                # if self.should_clip_gradients:
                #     clip_gradients(self.model, self.current_iteration, self.writer, self.config)
                # # self.optimizer.step()
                # scaler.step(self.optimizer) 
                # scaler.update()
                # self._run_scheduler()
                # self.profile("Backward time")
                # # self.optimizer.zero_grad()
                # should_break = self._logistics(report_adv2)

                # self.writer.write("===loss(batch1)==")
                # self.writer.write(loss)
                # exit()

                if should_break:
                    break

        self.finalize()

    def _run_scheduler(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.current_iteration)

    def _forward_pass(self, batch):
        prepared_batch = self.dataset_loader.prepare_batch(batch)
        self.profile("Batch prepare time")
        # Arguments should be a dict at this point
        if self.config.training_parameters.use_llama2:
            model_output = self.model(prepared_batch, model_llama2=self.model_llama2, question_str=prepared_batch['question_str'], \
                ocr_tokens=prepared_batch['ocr_tokens'])
            report = Report(prepared_batch, model_output, model_llama2=self.model_llama2)
        elif self.config.training_parameters.qa_gen:
            model_output = self.model(prepared_batch, model_llama2=None, question_str=prepared_batch['question_str'], \
                ocr_tokens=prepared_batch['ocr_tokens'])
            report = Report(prepared_batch, model_output, model_llama2=None)
        else:
            model_output = self.model(prepared_batch)
            report = Report(prepared_batch, model_output)
        self.profile("Forward time")

        return report

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

        if self.should_clip_gradients:
            clip_gradients(self.model, self.current_iteration, self.writer, self.config)

        self.optimizer.step()
        self._run_scheduler()

        self.profile("Backward time")

    def _extract_loss(self, report):
        loss_dict = report.losses
        loss = sum([loss.mean() for loss in loss_dict.values()])
        return loss

    def finalize(self):
        self.writer.write("Stepping into final validation check")

        # Only do when run_type has train as it shouldn't happen on validation and inference runs
        # Inference will take care of this anyways. Also, don't run if current iteration
        # is divisble by snapshot interval as it will just be a repeat
        if "train" in self.run_type and self.current_iteration % self.snapshot_interval != 0:
            self._try_full_validation(force=True)

        self.checkpoint.restore()
        self.checkpoint.finalize()
        self.inference()

    def _update_meter(self, report, meter=None, eval_mode=False):
        if meter is None:
            meter = self.meter

        loss_dict = report.losses
        metrics_dict = report.metrics

        reduced_loss_dict = reduce_dict(loss_dict)
        reduced_metrics_dict = reduce_dict(metrics_dict)

        loss_key = report.dataset_type + "/total_loss"

        with torch.no_grad():
            reduced_loss = sum([loss.mean() for loss in reduced_loss_dict.values()])
            if hasattr(reduced_loss, "item"):
                reduced_loss = reduced_loss.item()

            registry.register(loss_key, reduced_loss)

            meter_update_dict = {loss_key: reduced_loss}
            meter_update_dict.update(reduced_loss_dict)
            meter_update_dict.update(reduced_metrics_dict)
            meter.update(meter_update_dict)

    def _logistics(self, report):
        should_print = self.current_iteration % self.log_interval == 0
        should_break = False
        extra = {}

        if should_print is True:
            if "cuda" in str(self.device):
                extra["max mem"] = torch.cuda.max_memory_allocated() / 1024
                extra["max mem"] //= 1024

            extra.update(
                {
                    "lr": "{:.5f}".format(self.optimizer.param_groups[0]["lr"]).rstrip(
                        "0"
                    ),
                    "time": self.train_timer.get_time_since_start(),
                    "eta": self._calculate_time_left(),
                }
            )

            self.train_timer.reset()

            _, meter = self.evaluate(self.val_loader, single_batch=True)
            self.meter.update_from_meter(meter)

        # Don't print train metrics if it is not log interval
        # so as to escape clutter
        self._summarize_report(
            self.meter,
            should_print=should_print,
            extra=extra,
            prefix=report.dataset_name,
        )

        should_break = self._try_full_validation()

        return should_break

    def _try_full_validation(self, force=False):
        should_break = False

        if self.current_iteration % self.snapshot_interval == 0 or force:
            self.writer.write("Evaluation time. Running on full validation set...")
            # Validation and Early stopping
            # Create a new meter for this case
            report, meter = self.evaluate(self.val_loader)

            extra = {"validation time": self.snapshot_timer.get_time_since_start()}

            stop = self.early_stopping(self.current_iteration, meter)
            stop = bool(broadcast_scalar(stop, src=0, device=self.device))

            extra.update(self.early_stopping.get_info())

            prefix = "{}: full val".format(report.dataset_name)

            self._summarize_report(meter, prefix=prefix, extra=extra)
            self.snapshot_timer.reset()
            gc.collect()

            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

            if stop is True:
                self.writer.write("Early stopping activated")
                should_break = True

        return should_break

    def evaluate(self, loader, use_tqdm=False, single_batch=False):
        meter = Meter()
        if self.config.training_parameters.caption_gen:
            empty_data={}
            with open(self.config.training_parameters.caption_json_path,"w") as pre_file:
                json.dump(empty_data,pre_file)
                pre_file.write('\n')
        if self.config.training_parameters.qa_gen:
            empty_data={}
            with open(self.config.training_parameters.qa_json_path,"w") as pre_file:
                json.dump(empty_data,pre_file)
                pre_file.write('\n')
        with torch.no_grad():
            self.model.eval()
            if self.config.training_parameters.caption_gen:
                self.model_blip.eval()
            if self.config.training_parameters.qa_gen:
                self.model_t5.eval()
            if self.config.training_parameters.use_llama2:
                self.model_llama2.eval()
            ## tqdm modified
            # use_tqdm=False
            disable_tqdm = not use_tqdm or not is_main_process()
            # disable_tqdm = False
            # disable_tqdm = single_batch
            for batch in tqdm(loader, disable=disable_tqdm):
                # report = self._forward_pass(batch)
                # 想不到吧，验证也要改 （尽量放进函数中）
                prepared_batch = self.dataset_loader.prepare_batch(batch)
                self.profile("Batch prepare time")
                # Arguments should be a dict at this point
                prepared_batch['adv'] = False
                # print(prepared_batch['image_name'])
                if self.config.training_parameters.use_llama2 and self.config.training_parameters.qa_gen:
                    print("Begin qa_llm inference...")
                    model_output = self.model(prepared_batch, writer=self.writer, model_llama2=self.model_llama2, tokenizer_llama2=self.tokenizer, \
                    model_blip=self.model_blip, caption_processor=self.caption_processor, tokenizer_t5=self.tokenizer_t5, model_t5=self.model_t5, \
                    question_str=prepared_batch['question_str'], ocr_tokens=prepared_batch['ocr_tokens'], img=prepared_batch['img'])
                    report = Report(prepared_batch, model_output)
                # #stage3 with llm llama2
                elif self.config.training_parameters.use_llama2:
                    print("Begin llm inference...")
                    model_output = self.model(prepared_batch, writer=self.writer, model_llama2=self.model_llama2, \
                    question_str=prepared_batch['question_str'], ocr_tokens=prepared_batch['ocr_tokens'], img=prepared_batch['img'])
                # stage3 with llm
                elif self.config.training_parameters.LLM_auxiliary:
                    json_for_llm = self.config.training_parameters.llm_json_path
                    image_name = prepared_batch['image_name']
                    print("LLM module loaded successfully!")
                    model_output = self.model(prepared_batch, writer=self.writer, json_for_llm=json_for_llm, \
                    image_name=image_name, question_str=prepared_batch['question_str'], ocr_tokens=prepared_batch['ocr_tokens'], img=prepared_batch['img'])
                    report = Report(prepared_batch, model_output)
                # stage 1 caption_gen with vlm
                elif self.config.training_parameters.caption_gen_with_vlm:
                    json_for_caption = self.config.training_parameters.caption_vlm_json_path
                    image_name = prepared_batch['image_name']
                    print("Begin vlm caption inference...")
                    model_output = self.model(prepared_batch, writer=self.writer, json_for_caption=json_for_caption, \
                    model_vlm=self.model_vlm, tokenizer_vlm=self.tokenizer_vlm, image_processor=self.image_processor, \
                    image_name=image_name, question_str=prepared_batch['question_str'], ocr_tokens=prepared_batch['ocr_tokens'], img=prepared_batch['img'])
                    report = Report(prepared_batch, model_output)
                # stage 1 caption_gen
                elif self.config.training_parameters.caption_gen:
                    json_for_caption = self.config.training_parameters.caption_json_path
                    image_name = prepared_batch['image_name']
                    print("Begin caption inference...")
                    model_output = self.model(prepared_batch, writer=self.writer, json_for_caption=json_for_caption, \
                    model_blip=self.model_blip, caption_processor=self.caption_processor, \
                    image_name=image_name, question_str=prepared_batch['question_str'], ocr_tokens=prepared_batch['ocr_tokens'], img=prepared_batch['img'])
                    report = Report(prepared_batch, model_output)
                # stage 2 qa_gen
                elif self.config.training_parameters.qa_gen:
                    json_for_caption = self.config.training_parameters.caption_json_path
                    json_for_qa = self.config.training_parameters.qa_json_path
                    image_name = prepared_batch['image_name']
                    print("Begin qa inference...")
                    model_output = self.model(prepared_batch, writer=self.writer, json_for_caption=json_for_caption, json_for_qa=json_for_qa, \
                    tokenizer_t5=self.tokenizer_t5, model_t5=self.model_t5, \
                    image_name=image_name, question_str=prepared_batch['question_str'], ocr_tokens=prepared_batch['ocr_tokens'], img=prepared_batch['img'])
                    report = Report(prepared_batch, model_output)
                
                else:
                    model_output = self.model(prepared_batch)
                    report = Report(prepared_batch, model_output)
                # print("Report Success!")
                self.profile("Forward time")
                self._update_meter(report, meter, eval_mode=True)

                if single_batch is True:
                    break
                
                # print("Forward Success!")
            self.model.train()

        return report, meter

    def _summarize_report(self, meter, prefix="", should_print=True, extra={}):
        if not is_main_process():
            return

        scalar_dict = meter.get_scalar_dict()
        self.writer.add_scalars(scalar_dict, registry.get("current_iteration"))

        if not should_print:
            return

        print_str = []

        if len(prefix):
            print_str += [prefix + ":"]

        print_str += ["{}/{}".format(self.current_iteration, self.max_iterations)]
        print_str += [str(meter)]
        print_str += ["{}: {}".format(key, value) for key, value in extra.items()]

        self.writer.write(meter.delimiter.join(print_str))

    def inference(self):
        if "val" in self.run_type:
            self._inference_run("val")

        if "inference" in self.run_type or "predict" in self.run_type:
            self._inference_run("test")

    def _inference_run(self, dataset_type):
        if self.config.training_parameters.evalai_inference is True:
            self.predict_for_evalai(dataset_type)
            return

        self.writer.write("Starting inference on {} set".format(dataset_type))

        report, meter = self.evaluate(
            getattr(self, "{}_loader".format(dataset_type)), use_tqdm=True
        )
        prefix = "{}: full {}".format(report.dataset_name, dataset_type)
        self._summarize_report(meter, prefix)

    def _calculate_time_left(self):
        time_taken_for_log = time.time() * 1000 - self.train_timer.start
        iterations_left = self.max_iterations - self.current_iteration
        num_logs_left = iterations_left / self.log_interval
        time_left = num_logs_left * time_taken_for_log

        snapshot_iteration = self.snapshot_iterations / self.log_interval
        snapshot_iteration *= iterations_left / self.snapshot_interval
        time_left += snapshot_iteration * time_taken_for_log

        return self.train_timer.get_time_hhmmss(gap=time_left)

    def profile(self, text):
        if self.not_debug:
            return
        self.writer.write(text + ": " + self.profiler.get_time_since_start(), "debug")
        self.profiler.reset()

    def predict_for_evalai(self, dataset_type):
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        with torch.no_grad():
            self.model.eval()
            message = "Starting {} inference for evalai".format(dataset_type)
            self.writer.write(message)

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()

                for batch in tqdm(dataloader):
                    prepared_batch = reporter.prepare_batch(batch)
                    model_output = self.model(prepared_batch)
                    report = Report(prepared_batch, model_output)
                    reporter.add_to_report(report)

            self.writer.write("Finished predicting")
            self.model.train()
