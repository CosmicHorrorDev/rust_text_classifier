from argparse import ArgumentParser
from dataclasses import dataclass


@dataclass
class Args:
    dont_comment: bool
    verbose: int

    def __init__(self) -> None:
        parser = ArgumentParser(
            description=(
                "A Reddit Rust text classifier bot for automatically checking if text"
                " posts to r/rust are about the rust language or game"
            )
        )
        parser.add_argument(
            "--dont-comment",
            help="Just passively observe all the posts instead of leaving any comments",
            action="store_true",
        )
        # Default verbosity to INFO
        parser.add_argument(
            "--verbose",
            "-v",
            help="Increase logging verbosity",
            action="count",
            default=1,
        )

        args = parser.parse_args()

        self.dont_comment = args.dont_comment
        self.verbose = args.verbose
