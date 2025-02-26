import yaml
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Base class for all gait datasets."""

    def __init__(self, config_path):
        """Initialize the dataset with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.name = self.config['name']
        self.data_path = self.config['path']
        self.train_subjects = self._parse_subject_range(
            self.config['train_subjects'])
        self.test_subjects = self._parse_subject_range(
            self.config['test_subjects'])

    def _parse_subject_range(self, subject_range):
        """Parse subject range from configuration."""
        if isinstance(subject_range, list) and len(subject_range) == 1 and \
                '-' in subject_range[0]:
            start, end = map(int, subject_range[0].split('-'))
            return list(range(start, end + 1))
        return subject_range

    @abstractmethod
    def load_data(self, subjects=None, views=None, conditions=None):
        """Load data from the dataset."""
        pass

    @abstractmethod
    def preprocess(self, data):
        """Preprocess the loaded data."""
        pass

    def get_train_data(self, views=None, conditions=None):
        """Get training data."""
        return self.load_data(self.train_subjects, views, conditions)

    def get_test_data(self, views=None, conditions=None):
        """Get testing data."""
        return self.load_data(self.test_subjects, views, conditions)
