"""Converters from a downloaded prompt-set source format into the speed sample format, one module each.

A module `<name>` defines `convert(raw: bytes) -> list[dict]` returning sample records (see
`effeval.core.promptsets`); a manifest selects it with `converter: <name>`.
"""
