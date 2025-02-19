from typing import TYPE_CHECKING, List, Optional

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_reward_model
from .trainer import CustomGRPOTrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

def run_grpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):
    tokenizer = load_tokenizer(model_args)
    model = load_model(tokenizer, model_args, finetuning_args)
    
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="grpo")
    
    reward_model = create_reward_model(model, model_args, finetuning_args)
    
    trainer = CustomGRPOTrainer(
        model=model,
        reward_funcs=reward_model,
        args=training_args,
        finetuning_args=finetuning_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=callbacks
    )
    
    train_result = trainer.train()
    trainer.save_model()
    trainer.save_state()
    
    if trainer.is_world_process_zero():
        plot_loss(training_args.output_dir, keys=["loss", "reward"])
    
    return train_result 