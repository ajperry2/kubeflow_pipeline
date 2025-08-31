from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ajperry_pipeline.ml.models.transformer import build_transformer
from ajperry_pipeline.ml.data.reddit import RedditDataset
from tqdm import tqdm
import torchtext.data.metrics as metrics
import mlflow
from mlflow.tracking import MlflowClient


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
    print_msg,
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
                print_msg("-" * console_width)
                print_msg(f"Source:\t{source_text}")
                print_msg(f"Target:\t{target_text}")
                print_msg(f"Predicted:\t{model_out_text}")
            if count == num_examples:
                break
    candidate_corpus = [
        list(model_out_text.split()) for model_out_text in predicted_texts
    ]
    # Example reference translations (each candidate can have multiple references, also of varying lengths)
    references_corpus = [[list(target_text.split())] for target_text in expected_texts]
    bleu_score = metrics.bleu_score(candidate_corpus, references_corpus, max_n=4)
    if verbose:
        print_msg(f"BLEU Score: {bleu_score}")
    mlflow.log_metric("test_bleu", bleu_score, global_step=global_step)
    return bleu_score


def train(config):
    with mlflow.start_run(run_name=config["experiment_name"]) as run:
        for k, v in config.items():
            mlflow.log_param(k, v)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

        # Make Datasets
        train_dataset = RedditDataset(
            "reddit.csv", sequence_length=560, is_train=True, train_split_perc=0.8
        )
        mlflow.log_param("TrainingSamples", len(train_dataset))
        test_dataset = RedditDataset(
            "reddit.csv", sequence_length=560, is_train=False, train_split_perc=0.8
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

        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], eps=1e-9)

        initial_epoch = 0
        global_step = 0
        client = MlflowClient(mlflow.get_tracking_uri())
        previous_models = client.get_latest_versions(config["model_name"])
        # best_uri = None
        best_performance = 0
        if previous_models:
            for prev_model in previous_models:
                if "performance" in prev_model.tags and best_performance > float(
                    prev_model.tags["performance"]
                ):
                    best_performance = prev_model.tags["performance"]
                    # best_uri = prev_model.source

        loss_fn = nn.CrossEntropyLoss(
            ignore_index=train_dataset.input_tokenizer.token_to_id("[PAD]"),
            label_smoothing=0.1,
        ).to(device)

        for epoch in range(initial_epoch, config["num_epochs"]):
            model.train()
            batch_iterator = tqdm(
                train_dataloader, desc=f"Processing epoch: {epoch}", leave=False
            )

            for batch in batch_iterator:
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
                        -1, train_dataset.output_tokenizer.get_vocab_size()
                    ),
                    label.view(-1),
                )
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
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
                lambda x: batch_iterator.write(x),
                global_step,
                num_examples=config["num_examples"],
                verbose=config["verbose"],
            )
            # Save Model
            if performance > best_performance:
                mlflow.pytorch.log_model(
                    model,
                    config["model_name"],
                    registered_model_name=config["model_name"],
                )
                model_uri = f"{run.info.artifact_uri}/model"
                client = MlflowClient(mlflow.get_tracking_uri())
                model_info = client.get_latest_versions(config["model_name"])[0]
                client.set_model_version_tag(
                    name=config["model_name"],
                    version=model_info.version,
                    key="performance",
                    value=performance,
                )
                return model_uri
