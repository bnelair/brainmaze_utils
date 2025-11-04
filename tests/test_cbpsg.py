from brainmaze_utils.annotations import load_CyberPSG
import pytest
import pandas as pd


def test_load_a2():
    df = load_CyberPSG('tests/a2.xml')
    assert isinstance(df, pd.DataFrame)
    print(df.head())
