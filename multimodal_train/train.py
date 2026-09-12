import sys

from transformers import PreTrainedConfig, PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast
import zipfile
from PIL import Image
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any

class VlmConfig(PretrainedConfig):
    model_type = "vlm_model"
    has_no_defaults_at_init = True
    def __init__(self,llmModelPath,
                 visionModelPath,
                 freezeVisionModel = True,
                 imagePadNum = 49,
                **kwargs) -> None:        
        self.llmModelPath = llmModelPath
        self.visionModelPath = visionModelPath
        self.freezeVisionModel = freezeVisionModel
        self.imagePadNum = imagePadNum
        super().__init__(**kwargs)

class VLM(PreTrainedModel):
    config_class = VlmConfig
    def __init__(self, config: VlmConfig):
        super().__init__(config)
        self.config = config
        self.visionModle = AutoModel.from_pretrained(self.config.visionModelPath)
        self.visionProcessor = AutoProcessor.from_pretrained(self.config.visionModelPath)
        # self.llmModel = AutoModel.from_pretrained(self.config.llmModelPath)
        self.llmModel = AutoModelForCausalLM.from_pretrained(self.config.llmModelPath)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llmModelPath)
        self.linear1 = nn.Linear(self.visionModle.config.vision_config.hidden_size * 4, self.llmModel.config.hidden_size)
        self.linear2 = nn.Linear(self.llmModel.config.hidden_size, self.llmModel.config.hidden_size)
        if self.config.freezeVisionModel:
            for param in self.visionModle.parameters():
                param.requires_grad = False
        for param in self.llmModel.parameters():           
            param.requires_grad = False
        

    def mergeInputIdsWithImageFeatures(self, imageFeatures, inputsEmbeds, inputIds):
        numImages, numImagePatches, embedDim = imageFeatures.shape
        batchIndices, imageIndices = torch.where(inputIds == self.tokenizer('<|image_pad|>')['input_ids'][0])
        
        inputsEmbeds[batchIndices, imageIndices] = imageFeatures.view(-1, embedDim)
        return inputsEmbeds

    def forward(self, input_ids : Tensor, labels, pixel_values, attentionMask = None):
        textEmbeds : Tensor = self.llmModel.get_input_embeddings()(input_ids)
        imageEmbeds : Tensor = self.visionModle.vision_model(pixel_values).last_hidden_state

        b, s, d = imageEmbeds.shape #(batch, seq, dim)
        imageEmbeds : Tensor = imageEmbeds.reshape(b, -1, d*4) # (b, 196, d) --> (b, 49, d*4) 压缩图片tokens，SigLIP输入token数是196
        imageFeatures : Tensor = self.linear2(F.silu(self.linear1(imageEmbeds)))

        textEmbeds = textEmbeds.to(imageFeatures.dtype)
        inputsEmbeds = self.mergeInputIdsWithImageFeatures(imageFeatures, textEmbeds, input_ids)
        outputs = self.llmModel(inputs_embeds=inputsEmbeds, attention_mask=attentionMask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            lossFct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = lossFct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

class MyDataset(Dataset):
    def __init__(self, imagePath, dataPath, tokkenizer, processor, config):
        super().__init__()
        self.dataPath = dataPath
        self.imagePath = imagePath
        self.tokenizer = tokkenizer
        self.processor = processor
        self.config = config
        with open(self.dataPath, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)
        

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            imageName = sample['image']
            conversations = sample['conversations']
            questionText = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":conversations[0]['value']}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.imagePadNum)
            answerText = conversations[1]['value'] + self.tokenizer.eos_token
            questionInputIds = self.tokenizer(questionText)['input_ids']
            answerInputIds = self.tokenizer(answerText)['input_ids']
            inputIds = questionInputIds + answerInputIds
            labels = [self.tokenizer.pad_token_id] * len(questionInputIds) + answerInputIds
            inputIds = inputIds[:-1]
            labels = labels[1:]
            image = Image.open(os.path.join(self.imagePath, imageName)).convert("RGB")
            pixelValues = self.processor(text=None, images=image, return_tensors='pt')['pixel_values']
        except:
            defaultImage = Image.new('RGB', (224, 224), color='white')
            pixelValues = self.processor(text=None, images=defaultImage, return_tensors='pt')['pixel_values']
            questionText = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":"图片内容是什么\n<image>"}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.imagePadNum)
            answerText = '图片内容为空' + self.tokenizer.eos_token
            questionInputIds = self.tokenizer(questionText)['input_ids']
            answerInputIds = self.tokenizer(answerText)['input_ids']
            inputIds = questionInputIds + answerInputIds
            labels = [self.tokenizer.pad_token_id] * len(questionInputIds) + answerInputIds
            inputIds = inputIds[:-1]
            labels = labels[1:]
        return {
            'input_ids': inputIds,
            'labels': labels,
            'pixel_values': pixelValues
        } 

class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])
            # pixel_values.append(torch.stack([torch.as_tensor(v) for v in feature['pixel_values']]))
            
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)}

def main(modelPath : str):
    visionModelPath = os.path.join(modelPath, 'AI-ModelScope/siglip-base-patch16-224')
    llmModelPath = os.path.join(modelPath, 'Qwen/Qwen2.5-0.5B-Instruct')
    config = VlmConfig(llmModelPath = llmModelPath, visionModelPath = visionModelPath)
    model = VLM(config).cuda()
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    images_path = os.path.join(modelPath, 'dataset/liuhaotian/LLaVA-CC3M-Pretrain-595K/images')
    data_path = os.path.join(modelPath, 'dataset/LinkSoul/Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json')
    tokenizer = AutoTokenizer.from_pretrained(config.llmModelPath)
    processor = AutoProcessor.from_pretrained(config.visionModelPath)
    output_dir = 'save/pretrain' 
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        num_train_epochs=5,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(images_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)  
    )
    
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('save/pretrain')
    trainer.save_state()
    
if __name__ == '__main__':
    main(sys.argv[1])

