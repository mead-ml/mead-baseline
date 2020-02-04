import pytest
from eight_mile.utils import transition_mask

IOBv = {"<PAD>": 0, "<GO>": 1, "<EOS>": 2, "B-X": 3, "I-X": 4, "B-Y": 5, "I-Y": 6, "O": 7}

BIOv = IOBv

IOBESv = {
    "<PAD>": 0,
    "<GO>": 1,
    "<EOS>": 2,
    "B-X": 3,
    "I-X": 4,
    "E-X": 5,
    "S-X": 6,
    "B-Y": 7,
    "I-Y": 8,
    "E-Y": 9,
    "S-Y": 10,
    "O": 11,
}


@pytest.fixture
def IOB():
    return transition_mask(IOBv, "IOB", IOBv["<GO>"], IOBv["<EOS>"], IOBv["<PAD>"])


@pytest.fixture
def BIO():
    return transition_mask(IOBv, "BIO", IOBv["<GO>"], IOBv["<EOS>"], IOBv["<PAD>"])


@pytest.fixture
def IOBES():
    return transition_mask(IOBESv, "IOBES", IOBESv["<GO>"], IOBESv["<EOS>"], IOBESv["<PAD>"])


def test_IOB_shape(IOB):
    assert IOB.shape == (len(IOBv), len(IOBv))


def test_BIO_shape(BIO):
    assert BIO.shape == (len(IOBv), len(IOBv))
    mask = transition_mask(IOBv, "IOB2", IOBv["<GO>"], IOBv["<EOS>"], IOBv["<PAD>"])
    assert mask.shape == (len(IOBv), len(IOBv))


def test_IOBES_shape(IOBES):
    assert IOBES.shape == (len(IOBESv), len(IOBESv))


def test_IOB_I_B_mismatch(IOB):
    assert IOB[IOBv["B-X"], IOBv["I-Y"]] == 0


def test_ION_I_I_match(IOB):
    assert IOB[IOBv["I-X"], IOBv["I-X"]] == 1


def test_IOB_I_I_mismatch(IOB):
    assert IOB[IOBv["I-Y"], IOBv["I-X"]] == 1


def test_IOB_to_pad(IOB):
    assert IOB[IOBv["<PAD>"], IOBv["O"]] == 1
    assert IOB[IOBv["<PAD>"], IOBv["I-X"]] == 1
    assert IOB[IOBv["<PAD>"], IOBv["B-X"]] == 1


def test_IOB_to_end(IOB):
    assert IOB[IOBv["<EOS>"], IOBv["O"]] == 1
    assert IOB[IOBv["<EOS>"], IOBv["I-X"]] == 1
    assert IOB[IOBv["<EOS>"], IOBv["B-X"]] == 1


def test_BIO_from_start(BIO):
    assert BIO[BIOv["B-X"], BIOv["<GO>"]] == 1
    assert BIO[BIOv["I-X"], BIOv["<GO>"]] == 0
    assert BIO[BIOv["O"], BIOv["<GO>"]] == 1


def test_IOBES_to_start(IOBES):
    assert IOBES[IOBESv["<GO>"], IOBESv["B-X"]] == 0
    assert IOBES[IOBESv["<GO>"], IOBESv["I-X"]] == 0
    assert IOBES[IOBESv["<GO>"], IOBESv["E-X"]] == 0
    assert IOBES[IOBESv["<GO>"], IOBESv["S-X"]] == 0
    assert IOBES[IOBESv["<GO>"], IOBESv["O"]] == 0
    assert IOBES[IOBESv["<GO>"], IOBESv["<EOS>"]] == 0
    assert IOBES[IOBESv["<GO>"], IOBESv["<PAD>"]] == 0
    assert IOBES[IOBESv["<GO>"], IOBESv["<GO>"]] == 0


def test_IOBES_from_end(IOBES):
    assert IOBES[IOBESv["B-X"], IOBESv["<EOS>"]] == 0
    assert IOBES[IOBESv["I-X"], IOBESv["<EOS>"]] == 0
    assert IOBES[IOBESv["E-X"], IOBESv["<EOS>"]] == 0
    assert IOBES[IOBESv["S-X"], IOBESv["<EOS>"]] == 0
    assert IOBES[IOBESv["O"], IOBESv["<EOS>"]] == 0
    assert IOBES[IOBESv["<PAD>"], IOBESv["<EOS>"]] == 0
    assert IOBES[IOBESv["<GO>"], IOBESv["<EOS>"]] == 0
    assert IOBES[IOBESv["<EOS>"], IOBESv["<EOS>"]] == 0


def test_IOBES_from_pad(IOBES):
    assert IOBES[IOBESv["B-X"], IOBESv["<PAD>"]] == 0
    assert IOBES[IOBESv["I-X"], IOBESv["<PAD>"]] == 0
    assert IOBES[IOBESv["E-X"], IOBESv["<PAD>"]] == 0
    assert IOBES[IOBESv["S-X"], IOBESv["<PAD>"]] == 0
    assert IOBES[IOBESv["O"], IOBESv["<PAD>"]] == 0
    assert IOBES[IOBESv["<GO>"], IOBESv["<PAD>"]] == 0
    assert IOBES[IOBESv["<PAD>"], IOBESv["<PAD>"]] == 1
    assert IOBES[IOBESv["<EOS>"], IOBESv["<PAD>"]] == 1


def test_IOBES_O(IOBES):
    assert IOBES[IOBESv["B-X"], IOBESv["O"]] == 1
    assert IOBES[IOBESv["I-X"], IOBESv["O"]] == 0
    assert IOBES[IOBESv["E-X"], IOBESv["O"]] == 0
    assert IOBES[IOBESv["S-X"], IOBESv["O"]] == 1
    assert IOBES[IOBESv["O"], IOBESv["O"]] == 1


def test_IOBES_B(IOBES):
    assert IOBES[IOBESv["I-X"], IOBESv["B-X"]] == 1
    assert IOBES[IOBESv["E-X"], IOBESv["B-X"]] == 1
    assert IOBES[IOBESv["I-Y"], IOBESv["B-X"]] == 0
    assert IOBES[IOBESv["E-Y"], IOBESv["B-X"]] == 0
    assert IOBES[IOBESv["S-X"], IOBESv["B-X"]] == 0
    assert IOBES[IOBESv["B-X"], IOBESv["B-X"]] == 0
    assert IOBES[IOBESv["O"], IOBESv["B-X"]] == 0


def test_IOBES_I(IOBES):
    assert IOBES[IOBESv["I-X"], IOBESv["I-X"]] == 1
    assert IOBES[IOBESv["E-X"], IOBESv["I-X"]] == 1
    assert IOBES[IOBESv["I-Y"], IOBESv["I-X"]] == 0
    assert IOBES[IOBESv["E-Y"], IOBESv["I-X"]] == 0
    assert IOBES[IOBESv["S-X"], IOBESv["I-X"]] == 0
    assert IOBES[IOBESv["B-X"], IOBESv["I-X"]] == 0
    assert IOBES[IOBESv["O"], IOBESv["I-X"]] == 0


def test_IOBES_from_E(IOBES):
    assert IOBES[IOBESv["I-X"], IOBESv["E-X"]] == 0
    assert IOBES[IOBESv["E-X"], IOBESv["E-X"]] == 0
    assert IOBES[IOBESv["S-X"], IOBESv["E-X"]] == 1
    assert IOBES[IOBESv["B-X"], IOBESv["E-X"]] == 1
    assert IOBES[IOBESv["O"], IOBESv["E-X"]] == 1


def test_IOBES_to_E(IOBES):
    assert IOBES[IOBESv["E-X"], IOBESv["B-X"]] == 1
    assert IOBES[IOBESv["E-X"], IOBESv["I-X"]] == 1
    assert IOBES[IOBESv["E-X"], IOBESv["E-X"]] == 0
    assert IOBES[IOBESv["E-X"], IOBESv["B-Y"]] == 0
    assert IOBES[IOBESv["E-X"], IOBESv["I-Y"]] == 0
    assert IOBES[IOBESv["E-X"], IOBESv["E-Y"]] == 0
    assert IOBES[IOBESv["E-X"], IOBESv["S-X"]] == 0
    assert IOBES[IOBESv["E-X"], IOBESv["S-Y"]] == 0
    assert IOBES[IOBESv["E-X"], IOBESv["O"]] == 0


def test_IOBES_S(IOBES):
    assert IOBES[IOBESv["B-X"], IOBESv["S-X"]] == 1
    assert IOBES[IOBESv["I-X"], IOBESv["S-X"]] == 0
    assert IOBES[IOBESv["E-X"], IOBESv["S-X"]] == 0
    assert IOBES[IOBESv["S-X"], IOBESv["S-X"]] == 1
    assert IOBES[IOBESv["O"], IOBESv["S-X"]] == 1
