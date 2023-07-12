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
        self.cacher = self.Flag()
        self.tag = name

    def __repr__(self):
        return str(self.tag)


class MockChainRunnable(MockChain):
    def __init__(self, name):
        super().__init__(name)
        self.runner = MockRunner()

    def run(self, save_path=None, dry_run=False, cache_dir=None, skip_plot=False):
        print("run ", self.tag)

    def set_save_path(self, *args, **kwargs):
        pass
