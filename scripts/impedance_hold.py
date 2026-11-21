"""Foreground continuous-impedance hold; keeps its spring attached during contact.

Defaults to configs/controller.fr3.contact.json; --profile impedance selects
the previous settings. Run compliant_hold.py separately for the yielding law.
Both scripts require exclusive robot ownership.
"""
from compliant_hold import main


if __name__ == "__main__":
    raise SystemExit(main(continuous_impedance=True))
