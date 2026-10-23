"""Atheris harness for the BPE merge-rule JSON validator."""

import json
import sys

import atheris  # type: ignore

from cds.nlp.bpe import BPEMerge


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    document = fdp.ConsumeUnicodeNoSurrogates(4096)
    try:
        record = json.loads(document)
    except json.JSONDecodeError:
        return
    if not isinstance(record, dict):
        return
    try:
        BPEMerge.from_dict(record)
    except (KeyError, ValueError):
        return


atheris.Setup(sys.argv, TestOneInput)
atheris.Fuzz()
