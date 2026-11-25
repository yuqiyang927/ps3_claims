import pandas as pd

from ps3.data import create_sample_split

def test_hard():
    import ps3  # noqa

    # this is just a demo that pytest can produce good error messages just by parsing assert statements
    assert {"a": 1, "b": 2} == {"a": 1, "b": 2}


def test_sample_split():
    df = pd.DataFrame({"IDpol": range(1, 1001)})

    result = create_sample_split(df, "IDpol")

    assert "sample" in result.columns
    assert len(result) == len(df)
    assert set(result["sample"]) == {"train", "test"}


def test_training_fraction():
    df = pd.DataFrame({"IDpol": range(1, 1001)})

    result = create_sample_split(df, "IDpol", training_frac=0.8)

    train_frac = (result["sample"] == "train").mean()

    assert abs(train_frac - 0.8) < 0.05


def test_fixed_sample_split():
    df = pd.DataFrame({
        "IDpol": range(1, 1001),
        "Region": ["A", "B"] * 500
    })

    result = create_sample_split(df, ["IDpol", "Region"])

    df_shuffled = df.sample(frac=1, random_state=42)
    result_shuffled = create_sample_split(
        df_shuffled, ["IDpol", "Region"]
    )

    assert result["sample"].sort_index().equals(
        result_shuffled["sample"].sort_index()
    )