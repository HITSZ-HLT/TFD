"""Full training script"""
import argparse
import json
import logging
import os
import torch
import yaml
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
						  Trainer, TrainingArguments)
import dill
from utils import ErcTextDataset, compute_metrics, get_num_classes, fairness
import warnings
warnings.filterwarnings("ignore")
import numpy as np, random, math

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
)

def getxy(fout, xfout, yfout, zfout, true_labels):
	best_X, best_Y, best_Z, best_dev_cmaf1, cmaf1_map = 0.0, 0.0, 0.0, -999999999, {}
	while True:
		recorded_x, recorded_y, recorded_z = best_X, best_Y, best_Z
		for i in range(-1, 2):
			for j in range(-1, 2):
				for k in range(-1, 2):
					if i == 0 and j == 0 and k == 0:
						continue
					cur_x, cur_y, cur_z, step = recorded_x, recorded_y, recorded_z, 0
					while True:
						key = '{:.2f}_{:.2f}_{:.2f}'.format(cur_x, cur_y, cur_z)
						if key not in cmaf1_map.keys():
							predict_labels = fout - cur_x * xfout - cur_y * yfout - cur_z * zfout
							cmaf1 = compute_metrics((predict_labels, true_labels))['f1_weighted']
							cmaf1_map[key] = cmaf1
						cmaf1 = cmaf1_map[key]
						if cmaf1 > best_dev_cmaf1:
							best_dev_cmaf1, best_X, best_Y, best_Z, step = cmaf1, cur_x, cur_y, cur_z, 0
						if step>=20:
							break
						cur_x += i * 0.1
						cur_y += j * 0.1
						cur_z += k * 0.1
						step += 1
		if recorded_x==best_X and recorded_y==best_Y and recorded_z==best_Z:
			break
	return best_X, best_Y, best_Z, fout - best_X * xfout - best_Y * yfout - best_Z * zfout

def main(
	OUTPUT_DIR: str,
	SEED: int,
	DATASET: str,
	BATCH_SIZE: int,
	model_checkpoint: str,
	roberta: str,
	speaker_mode: str,
	num_past_utterances: int,
	num_future_utterances: int,
	NUM_TRAIN_EPOCHS: int,
	WEIGHT_DECAY: float,
	WARMUP_RATIO: float,
	**kwargs,
):
	"""Perform full training with the given parameters."""

	NUM_CLASSES = get_num_classes(DATASET)

	with open(os.path.join(OUTPUT_DIR, "hp.json"), "r") as stream:
		hp_best = json.load(stream)
	LEARNING_RATE = hp_best["learning_rate"]
	logging.info(f"(LOADED) best hyper parameters: {hp_best}")

	OUTPUT_DIR = OUTPUT_DIR.replace("-seed-42", f"-seed-{SEED}")

	EVALUATION_STRATEGY = "epoch"
	LOGGING_STRATEGY = "epoch"
	SAVE_STRATEGY = "epoch"
	ROOT_DIR = "./multimodal-datasets/"

	if model_checkpoint is None:
		model_checkpoint = f"roberta-{roberta}"

	PER_DEVICE_TRAIN_BATCH_SIZE = BATCH_SIZE
	PER_DEVICE_EVAL_BATCH_SIZE = BATCH_SIZE * 2
	if torch.cuda.is_available():
		FP16 = True
	else:
		FP16 = False
	LOAD_BEST_MODEL_AT_END = True

	METRIC_FOR_BEST_MODEL = "eval_f1_weighted"
	GREATER_IS_BETTER = True

	args = TrainingArguments(
		output_dir=OUTPUT_DIR,
		evaluation_strategy=EVALUATION_STRATEGY,
		logging_strategy=LOGGING_STRATEGY,
		save_strategy=SAVE_STRATEGY,
		per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
		per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
		load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
		seed=SEED,
		fp16=FP16,
		learning_rate=LEARNING_RATE,
		num_train_epochs=NUM_TRAIN_EPOCHS,
		weight_decay=WEIGHT_DECAY,
		warmup_ratio=WARMUP_RATIO,
		metric_for_best_model=METRIC_FOR_BEST_MODEL,
		greater_is_better=GREATER_IS_BETTER,
	)

	ds_train = ErcTextDataset(
		DATASET=DATASET,
		SPLIT="train",
		speaker_mode=speaker_mode,
		num_past_utterances=num_past_utterances,
		num_future_utterances=num_future_utterances,
		model_checkpoint=model_checkpoint,
		ROOT_DIR=ROOT_DIR,
		SEED=SEED,
	)

	global ds_val, ds_val_speaker, ds_val_label, ds_val_word, ds_test, ds_test_speaker, ds_test_label, ds_test_word
	ds_val = ErcTextDataset(
		DATASET=DATASET,
		SPLIT="val",
		speaker_mode=speaker_mode,
		num_past_utterances=num_past_utterances,
		num_future_utterances=num_future_utterances,
		model_checkpoint=model_checkpoint,
		ROOT_DIR=ROOT_DIR,
		SEED=SEED,
	)

	ds_val_speaker = ErcTextDataset(
		DATASET=DATASET,
		SPLIT="val",
		speaker_mode=speaker_mode,
		num_past_utterances=num_past_utterances,
		num_future_utterances=num_future_utterances,
		model_checkpoint=model_checkpoint,
		ROOT_DIR=ROOT_DIR,
		SEED=SEED,
		mode="all",
	)

	ds_val_label = ErcTextDataset(
		DATASET=DATASET,
		SPLIT="val",
		speaker_mode=None,
		num_past_utterances=num_past_utterances,
		num_future_utterances=num_future_utterances,
		model_checkpoint=model_checkpoint,
		ROOT_DIR=ROOT_DIR,
		SEED=SEED,
		mode="all",
	)

	ds_val_word = ErcTextDataset(
		DATASET=DATASET,
		SPLIT="val",
		speaker_mode=speaker_mode,
		num_past_utterances=num_past_utterances,
		num_future_utterances=num_future_utterances,
		model_checkpoint=model_checkpoint,
		ROOT_DIR=ROOT_DIR,
		SEED=SEED,
		mode="partial",
	)
	ds_test = ErcTextDataset(
		DATASET=DATASET,
		SPLIT="test",
		speaker_mode=speaker_mode,
		num_past_utterances=num_past_utterances,
		num_future_utterances=num_future_utterances,
		model_checkpoint=model_checkpoint,
		ROOT_DIR=ROOT_DIR,
		SEED=SEED,
	)

	ds_test_speaker = ErcTextDataset(
		DATASET=DATASET,
		SPLIT="test",
		speaker_mode=speaker_mode,
		num_past_utterances=num_past_utterances,
		num_future_utterances=num_future_utterances,
		model_checkpoint=model_checkpoint,
		ROOT_DIR=ROOT_DIR,
		SEED=SEED,
		mode="all",
	)

	ds_test_label = ErcTextDataset(
		DATASET=DATASET,
		SPLIT="test",
		speaker_mode=None,
		num_past_utterances=num_past_utterances,
		num_future_utterances=num_future_utterances,
		model_checkpoint=model_checkpoint,
		ROOT_DIR=ROOT_DIR,
		SEED=SEED,
		mode="all",
	)

	ds_test_word = ErcTextDataset(
		DATASET=DATASET,
		SPLIT="test",
		speaker_mode=speaker_mode,
		num_past_utterances=num_past_utterances,
		num_future_utterances=num_future_utterances,
		model_checkpoint=model_checkpoint,
		ROOT_DIR=ROOT_DIR,
		SEED=SEED,
		mode="partial",
	)

	tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
	model = AutoModelForSequenceClassification.from_pretrained(
		model_checkpoint, num_labels=NUM_CLASSES
	)


	logging.info(f"training a full model with full data ...")

	trainer = Trainer(
		model=model,
		args=args,
		train_dataset=ds_train,
		eval_dataset=ds_val,
		tokenizer=tokenizer,
		compute_metrics=compute_metrics,
	)

	trainer.train()
	logging.info(f"eval ...")
	val_results = trainer.evaluate()
	with open(os.path.join(OUTPUT_DIR, "val-results.json"), "w") as stream:
		json.dump(val_results, stream, indent=4)
	logging.info(f"eval results of raw: {val_results['f1_weighted']}")
	val, true_val_label, _ = trainer.predict(ds_val)
	val_label, _, _ = trainer.predict(ds_val_label)
	val_word, _, _ = trainer.predict(ds_val_word)
	val_speaker, _, _ = trainer.predict(ds_val_speaker)
	x, y, z, final_val = getxy(val, val_label, val_word, val_speaker, true_val_label)
	logging.info(f"eval results of tfd: {compute_metrics((final_val, true_val_label))['f1_weighted']}")
	if len(ds_test) != 0:
		logging.info(f"test ...")
		test, true_test_label, _ = trainer.predict(ds_test)
		logging.info(f"test results of raw: {compute_metrics((test, true_test_label))['f1_weighted']}")
		test_label, _, _ = trainer.predict(ds_test_label)
		test_word, _, _ = trainer.predict(ds_test_word)
		test_speaker, _, _ = trainer.predict(ds_test_speaker)
		final_test = test - x * test_label - y * test_word - z * test_speaker
		test_results = compute_metrics((final_test, true_test_label))
		with open(os.path.join(OUTPUT_DIR, "test-results.json"), "w") as stream:
			json.dump(test_results, stream, indent=4)
		logging.info(f"test results of tfd: {test_results['f1_weighted']}")



if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="erc RoBERTa text huggingface training"
	)
	parser.add_argument("--OUTPUT-DIR", type=str)
	parser.add_argument("--SEED", type=int)

	args = parser.parse_args()
	args = vars(args)

	with open("./train-erc-text.yaml", "r") as stream:
		args_ = yaml.safe_load(stream)

	for key, val in args_.items():
		args[key] = val

	logging.info(f"arguments given to {__file__}: {args}")

	main(**args)
