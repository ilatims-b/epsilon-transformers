import pathlib
import tempfile

# import boto3
import pytest
import torch
from io import BytesIO
from dotenv import load_dotenv
import numpy as np
import shutil


from epsilon_transformers.persistence import LocalPersister, S3Persister
from epsilon_transformers.training.configs.training_configs import (
    LoggingConfig,
    OptimizerConfig,
    PersistanceConfig,
    ProcessDatasetConfig,
    TrainConfig,
)
from epsilon_transformers.training.configs.model_configs import RawModelConfig
from epsilon_transformers.training.train import train_model

# TODO: Insert check for training config path in load_model
# TODO: Add e2e training check for expected saved models
# TODO: Refactor the tests to use SimpleNN as fixture and random init the params

# TODO: Put slow tags on all s3 tests
# TODO: Write tests for local save_model overwrite protection
# TODO: Add a reset to the bucket state before running all the tests
# TODO: Move test non existing bucket into it's own test


def test_e2e_training():
    bucket_path = pathlib.Path(tempfile.mkdtemp(prefix="local-s3-bucket-"))

    model_config = RawModelConfig(
        d_vocab=2,
        d_model=100,
        n_ctx=10,
        d_head=48,
        n_head=12,
        d_mlp=12,
        n_layers=2,
    )
    optimizer_config = OptimizerConfig(
        optimizer_type="adam", learning_rate=1.06e-4, weight_decay=0.8
    )

    dataset_config = ProcessDatasetConfig(
        process="rrxor", batch_size=5, num_tokens=500, test_split=0.15
    )

    persistance_config = PersistanceConfig(
        location="s3",
        collection_location=str(bucket_path),
        checkpoint_every_n_tokens=100,
    )

    mock_config = TrainConfig(
        model=model_config,
        optimizer=optimizer_config,
        dataset=dataset_config,
        persistance=persistance_config,
        logging=LoggingConfig(project_name="local-s3-test", wandb=False),
        verbose=True,
        seed=1337,
    )
    train_model(mock_config)

    shutil.rmtree(bucket_path)


def test_s3_save_model_overwrite_protection():
    # Define a simple neural network
    bucket_path = pathlib.Path(tempfile.mkdtemp(prefix="local-s3-bucket-"))

    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    # Create an instance of the neural network
    network = SimpleNN()

    first_model_path = bucket_path / "45.pt"
    torch.save(network.state_dict(), first_model_path)

    persister = S3Persister(collection_location=bucket_path)

    with pytest.raises(ValueError):
        persister.save_model(network, 45)

    shutil.rmtree(bucket_path)


def load_local_model():
    # Define a simple neural network
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    # Create an instance of the neural network
    model = SimpleNN()

    with tempfile.TemporaryDirectory() as temp_dir:
        model_filepath = pathlib.Path(temp_dir) / "model.pt"
        torch.save(model.state_dict(), model_filepath)

        persister = LocalPersister(collection_location=pathlib.Path(temp_dir))
        loaded_model = persister.load_model(model=SimpleNN(), object_name="model.pt")

    assert torch.all(
        torch.eq(
            loaded_model.state_dict()["fc.weight"], model.state_dict()["fc.weight"]
        )
    )
    assert torch.all(
        torch.eq(loaded_model.state_dict()["fc.bias"], model.state_dict()["fc.bias"])
    )


def save_local_model():
    # Define a simple neural network
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    # Create an instance of the neural network
    model = SimpleNN()

    with tempfile.TemporaryDirectory() as temp_dir:
        persister = LocalPersister(collection_location=pathlib.Path(temp_dir))
        num_tokens = 45

        persister.save_model(model, num_tokens)

        loaded_model = SimpleNN()
        loaded_model_dict = torch.load(pathlib.Path(temp_dir) / f"{num_tokens}.pt")
        loaded_model.load_state_dict(loaded_model_dict)
    assert torch.all(
        torch.eq(
            loaded_model.state_dict()["fc.weight"], model.state_dict()["fc.weight"]
        )
    )
    assert torch.all(
        torch.eq(loaded_model.state_dict()["fc.bias"], model.state_dict()["fc.bias"])
    )


def test_save_and_load_s3_model():
    bucket_name = pathlib.Path(tempfile.mkdtemp(prefix="local-s3-bucket-"))
    with pytest.raises(AssertionError):
        S3Persister(
            collection_location=pathlib.Path("this-path-should-not-exist-99999")
        )

    persister = S3Persister(collection_location=bucket_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RawModelConfig(
        d_vocab=3,
        d_model=64,
        n_ctx=10,
        d_head=8,
        n_head=1,
        d_mlp=256,
        n_layers=4,
    ).to_hooked_transformer(seed=1337, device=device)

    # test save
    persister.save_model(model, 85)

    download_buffer = BytesIO()
    with open(bucket_name / "85.pt", "rb") as f:
        download_buffer.write(f.read())

    def _scramble_weights(model):
        for param in model.parameters():
            with torch.no_grad():
                if len(param.shape) > 1:  # Only scramble weights, not biases
                    flattened_param = param.view(-1)
                    np.random.shuffle(
                        flattened_param.numpy()
                    )  # Shuffle the flattened weights
                    param.data = flattened_param.view(param.shape)  # Restore the shape
        return model

    # Load the downloaded network
    downloaded_model = _scramble_weights(
        RawModelConfig(
            d_vocab=3,
            d_model=64,
            n_ctx=10,
            d_head=8,
            n_head=1,
            d_mlp=256,
            n_layers=4,
        ).to_hooked_transformer(seed=1337, device=device)
    )
    download_buffer.seek(0)  # Reset download buffer position to the beginning
    downloaded_model.load_state_dict(torch.load(download_buffer))

    # Assert that the downloaded network is the same as the original one
    for (name1, param1), (name2, param2) in zip(
        model.named_parameters(), downloaded_model.named_parameters()
    ):
        assert name1 == name2, "Model structure mismatch"
        assert (
            param1.shape == param2.shape
        ), f"Parameter shape mismatch for {name1} and {name2}"
        assert torch.allclose(
            param1, param2
        ), f"Parameter values mismatch for {name1} and {name2}"

    # Test save overwrite protection
    with pytest.raises(ValueError):
        persister.save_model(model, 85)

    # Test load
    loaded_model = persister.load_model(device=device, object_name="85.pt")
    for (name1, param1), (name2, param2) in zip(
        model.named_parameters(), loaded_model.named_parameters()
    ):
        assert name1 == name2, "Model structure mismatch"
        assert (
            param1.shape == param2.shape
        ), f"Parameter shape mismatch for {name1} and {name2}"
        assert torch.allclose(
            param1, param2
        ), f"Parameter values mismatch for {name1} and {name2}"

    # Delete mock bucket
    shutil.rmtree(bucket_name)


def test_s3_persistence_put_and_retrieve_object_from_bucket():
    # Define a simple neural network
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    # Create an instance of the neural network
    network = SimpleNN()

    bucket_path = pathlib.Path(tempfile.mkdtemp(prefix="local-s3-bucket-"))

    # Serialize the network to bytes
    buffer = BytesIO()
    torch.save(network.state_dict(), buffer)
    buffer.seek(0)

    (bucket_path / "model.pt").write_bytes(buffer.getvalue())

    # Download the serialized network from the bucket
    download_buffer = BytesIO((bucket_path / "model.pt").read_bytes())

    # Load the downloaded network
    downloaded_network = SimpleNN()
    download_buffer.seek(0)  # Reset download buffer position to the beginning
    downloaded_network.load_state_dict(torch.load(download_buffer))

    # Assert that the downloaded network is the same as the original one
    assert torch.all(
        torch.eq(
            network.state_dict()["fc.weight"],
            downloaded_network.state_dict()["fc.weight"],
        )
    )
    assert torch.all(
        torch.eq(
            network.state_dict()["fc.bias"], downloaded_network.state_dict()["fc.bias"]
        )
    )

    shutil.rmtree(bucket_path)


def test_s3_create_and_delete_bucket():
    # Define a simple neural network
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    # Create an instance of the neural network
    network = SimpleNN()

    buffer = BytesIO()
    torch.save(network.state_dict(), buffer)
    buffer.seek(0)

    bucket_path = pathlib.Path(tempfile.mkdtemp(prefix="local-bucket-"))

    # Serialize the network to bytes
    buffer = BytesIO()
    torch.save(network.state_dict(), buffer)

    model_file = bucket_path / "model.pt"
    with open(model_file, "wb") as f:
        f.write(buffer.getvalue())

    # Read file back
    download_buffer = BytesIO()
    with open(model_file, "rb") as f:
        download_buffer.write(f.read())
    download_buffer.seek(0)

    # Load the downloaded network
    downloaded_network = SimpleNN()
    download_buffer.seek(0)  # Reset download buffer position to the beginning
    downloaded_network.load_state_dict(torch.load(download_buffer))

    # Assert that the downloaded network is the same as the original one
    assert torch.all(
        torch.eq(
            network.state_dict()["fc.weight"],
            downloaded_network.state_dict()["fc.weight"],
        )
    )
    assert torch.all(
        torch.eq(
            network.state_dict()["fc.bias"], downloaded_network.state_dict()["fc.bias"]
        )
    )

    model_file.unlink()
    shutil.rmtree(bucket_path)

    # Assert that the bucket was deleted
    assert not bucket_path.exists(), f"Bucket {bucket_path} was not deleted"


if __name__ == "__main__":
    test_save_and_load_s3_model()
