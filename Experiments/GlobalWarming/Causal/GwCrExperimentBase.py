import GwData
from GwExperimentBase import GwExperimentBase

class GwCrExperimentBase(GwExperimentBase):

    def codes_to_filter(self):
        return [GwData.CAUSAL]

    def get_data(self, settings):
        return GwData.GwData(directory=settings.data_directory + "\\" + GwData.FOLDER)

    def sub_dir(self):
        return GwData.FOLDER + "\\_Causal"
