from typing import Protocol


class RunPolicyProtocol(Protocol):
    def execute(self) -> None:
        ...


class StrictMPI:
    def execute():
        pass


class SupportMPI(StrictMPI):
    def execute():
        pass


class SupportMultiprocessing:
    def execute():
        pass


class Vanilla:
    def execute():
        pass
