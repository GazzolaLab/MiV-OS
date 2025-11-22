from miv.core.operator.operator import ChainingMixin
from tests.core.mock_runner import MockRunner


class TemporaryCacher:
    def __init__(self, value: bool = False):
        self.value = value

    def check_cached(self):
        return self.value


class MockChain(ChainingMixin):
    def __init__(self, name):
        super().__init__()
        self.cacher = TemporaryCacher()
        self.tag = name

    def __repr__(self):
        return str(self.tag)


class MockChainWithoutCacher(ChainingMixin):
    def __init__(self, name):
        super().__init__()
        self.tag = name

    def __repr__(self):
        return str(self.tag)


class MockChainWithCache(ChainingMixin):
    def __init__(self, name):
        super().__init__()
        self.cacher = TemporaryCacher(True)
        self.tag = name

    def __repr__(self):
        return str(self.tag)


class MockChainRunnable(MockChain):
    def __init__(self, name):
        super().__init__(name)
        self.runner = MockRunner()
        self.run_counter = 0

    def output(self, save_path=None, dry_run=False, cache_dir=None, skip_plot=False):
        print("run ", self.tag)
        self.run_counter += 1

    def _set_save_path(self, *args, **kwargs):
        pass


class MockChainRunnableWithCache(MockChainRunnable):
    """
    Mock chain runner that only execute once.
    """

    def __init__(self, name):
        super().__init__(name)
        self.runner = MockRunner()
        self.run_counter = 0
        self.cacher = TemporaryCacher(False)

    def output(self, save_path=None, dry_run=False, cache_dir=None, skip_plot=False):
        print("run ", self.tag)
        self.cacher.value = True
        self.run_counter += 1
