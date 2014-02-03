__author__ = 'simon.hughes'

from ExperimentBase import *
import CbData


class CbExperimentBase(ExperimentBase):
    __metaclass__ = ABCMeta

    def sub_dir(self):
        return "CoralBleaching"

    def get_data(self, settings):
        self.data = CbData.CbData(include_vague=True)
        return self.data

    def codes_to_filter(self):
        return self.data.sm_codes

