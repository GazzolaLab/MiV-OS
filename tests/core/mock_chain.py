from mock_runner import MockRunner

from miv.core.operator.chainable import BaseChainingMixin


class MockChain(BaseChainingMixin):
    class Flag:
        def __init__(self):
            self.value = False
            self.cache_dir = "1"

        def check_cached(self):
            return self.value

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.cacher = self.Flag()

    def __repr__(self):
        return str(self.name)


class MockChainRunnable(MockChain):
    def __init__(self, name):
        super().__init__(name)
        self.runner = MockRunner()

    def run(self, save_path=None, dry_run=False, cache_dir=None):
        print("run ", self.name)
