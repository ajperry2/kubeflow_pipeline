from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ajperry_pipeline.ml.models.transformer import build_transformer
from ajperry_pipeline.ml.data.reddit import RedditDataset
import torchtext.data.metrics as metrics
import mlflow
from datetime import datetime, timedelta
import os


def get_weights_file_path(config, epoch):
    model_folder = config["model_folder"]
    model_basename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_basename)


def make_model(config, vocab_source_len, vocab_target_len):
    return build_transformer(
        vocab_source_len,
        vocab_target_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    sos_idx = tokenizer_src.token_to_id("[SOS]")
    eos_idx = tokenizer_src.token_to_id("[EOS]")

    # precompute encoder output
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = (
            RedditDataset.make_causal_mask(decoder_input.size(1))
            .type_as(source_mask)
            .to(device)
        )
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1)
                .type_as(decoder_input)
                .fill_(next_word.item())
                .to(device),
            ],
            dim=1,
        )
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)


def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    global_step,
    num_examples=2,
    verbose=False,
):
    model.eval()
    count = 0
    source_texts = []
    expected_texts = []
    predicted_texts = []
    console_width = 80
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # b, seq_len
            encoder_mask = batch["encoder_mask"].to(device)  # b, 1, 1, seq_len
            assert len(encoder_input) == 1
            model_output = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )
            source_text = batch["input_text"][0]
            target_text = batch["output_text"][0]
            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())
            source_texts.append(source_text)
            expected_texts.append(target_text)
            predicted_texts.append(model_out_text)
            if verbose:
                print("-" * console_width)
                print(f"Source:\t{source_text}")
                print(f"Target:\t{target_text}")
                print(f"Predicted:\t{model_out_text}")
            if count == num_examples:
                break
    candidate_corpus = [
        list(model_out_text.split()) for model_out_text in predicted_texts
    ]
    # Example reference translations (each candidate can have multiple references, also of varying lengths)
    references_corpus = [[list(target_text.split())] for target_text in expected_texts]
    bleu_score = metrics.bleu_score(candidate_corpus, references_corpus, max_n=4)
    if verbose:
        print(f"BLEU Score: {bleu_score}")
    mlflow.log_metric("test_bleu", bleu_score, step=global_step)
    return bleu_score

def get_best_model(config):
    runs = mlflow.search_runs(experiment_names=[config["experiment_name"]])
    model_path = ""
    best_performance = 0
    if len(runs) > 0:
        for run in runs:
            if "performance" in run.tags and best_performance > float(
                run.tags["performance"]
            ):
                best_performance = run.tags["performance"]
                model_path = run.tags["model_path"]
    return best_performance, model_path

def test(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_file = Path(config["data_folder"]) / "reddit.csv"
    one_day_ago_delta = timedelta(days=config["test_window"])
    test_dataset = RedditDataset(
        data_file, 
        sequence_length=560, 
        is_train=False, 
        train_split_perc=0.8,
        is_date=True,
        date_split=(datetime.now() - one_day_ago_delta)
    )
    best_performance, model_path = get_best_model(config)
    if model_path != "":
        # make model
        model = make_model(
            config,
            test_dataset.input_tokenizer.get_vocab_size(),
            test_dataset.output_tokenizer.get_vocab_size(),
        ).to(device)
        run_id, artifact_path =  model_path.split(":")
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=artifact_path
        )
        assert artifact_path.endswith(".pth")
        checkpoint = torch.load(artifact_path, map_location=device)
        os.remove(artifact_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Test Examples: {len(test_dataset)}")
        test_dataloader = DataLoader(test_dataset, batch_size=1)
        test_performance = run_validation(
            model,
            test_dataloader,
            test_dataset.input_tokenizer,
            test_dataset.output_tokenizer,
            config["max_len"],
            device,
            0,
            num_examples=len(test_dataloader),
            verbose=False,
        )
        if test_performance < 0.5:
            return False
        if test_performance < best_performance-0.05:
            return False
    # No model we need to train
    return True
        
def train(config):
    with mlflow.start_run(run_name=config["experiment_name"],) as run:
        for k, v in config.items():
            mlflow.log_param(k, v)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
        data_file = Path(config["data_folder"]) / "reddit.csv"
        # Make Datasets
        train_dataset = RedditDataset(
            data_file, sequence_length=560, is_train=True, train_split_perc=0.8
        )
        mlflow.log_param("TrainingSamples", len(train_dataset))
        test_dataset = RedditDataset(
            data_file, sequence_length=560, is_train=False, train_split_perc=0.8
        )
        mlflow.log_param("TestSamples", len(test_dataset))

        train_dataloader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        # make model
        model = make_model(
            config,
            train_dataset.input_tokenizer.get_vocab_size(),
            train_dataset.output_tokenizer.get_vocab_size(),
        ).to(device)
        initial_epoch = 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], eps=1e-9)
        if config["finetune"] != "":
            run_id, artifact_path = config["finetune"].split(":")
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path,
                dst_path=artifact_path
            )
            assert artifact_path.endswith(".pth")
            checkpoint = torch.load(artifact_path, map_location=device)
            os.remove(artifact_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            initial_epoch = checkpoint['epoch']

        global_step = 0
        # best_uri = None
        best_performance, model_path = get_best_model(config)

        loss_fn = nn.CrossEntropyLoss(
            ignore_index=train_dataset.input_tokenizer.token_to_id("[PAD]"),
            label_smoothing=0.1,
        ).to(device)

        for epoch in range(initial_epoch, config["num_epochs"]):
            model.train()


            for batch in train_dataloader:
                encoder_input = batch["encoder_input"].to(device)  # b, seq_len
                decoder_input = batch["decoder_input"].to(device)  # b, seq_len
                encoder_mask = batch["encoder_mask"].to(device)  # b, 1, 1, seq_len
                decoder_mask = batch["decoder_mask"].to(
                    device
                )  # b, 1, seq_len, seq_len

                encoder_output = model.encode(
                    encoder_input, encoder_mask
                )  # b, seq_len, d_model
                decoder_output = model.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask
                )  # b, seq_len, d_model
                proj_output = model.project(decoder_output)

                label = batch["label"].to(device)  # b, seq_len

                loss = loss_fn(
                    proj_output.view(
                        -1, 
                        train_dataset.output_tokenizer.get_vocab_size()
                    ),
                    label.view(-1),
                )
                mlflow.log_metric("train_loss", loss.item(), step=global_step)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            performance = run_validation(
                model,
                test_dataloader,
                test_dataloader.dataset.input_tokenizer,
                test_dataloader.dataset.output_tokenizer,
                config["seq_len"],
                device,
                global_step,
                num_examples=config["num_examples"],
                verbose=config["verbose"],
            )
            mlflow.log_metric("train_bleu", performance, step=global_step)
            # Save Model
            if performance > best_performance:
                best_performance = performance
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{config['experiment_name']}_{run.info.run_id}.pth")
                mlflow.log_artifact(f"{config['experiment_name']}.pth")
                mlflow.set_tag("model_path", f"{run.info.run_id}:{config['experiment_name']}.pth")
                mlflow.set_tag("performance", performance)