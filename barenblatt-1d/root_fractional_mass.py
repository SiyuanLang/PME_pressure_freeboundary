"""Original time-slab model using the shared root-fractional mass layer.

Only normalization changes. The default raw jet retains the original direct
sine arithmetic. Waiting-time models supply a stable raw jet to the same layer.
"""
from train_first_slab_float64 import Float64TimeSlabELM
from positive_interval_mass import MASS_RULE, PositiveIntervalMassMixin


class RootFractionalMassELM(PositiveIntervalMassMixin, Float64TimeSlabELM):
    def state(self):
        s = super().state()
        s.update(root_order=self.root_order, root_scan=self.root_scan,
                 endpoint_weight=self.endpoint_weight, mass_rule=MASS_RULE,
                 model_loader='root_fractional_mass.RootFractionalMassELM.load')
        return s

    @classmethod
    def from_state(cls, data):
        model = super().from_state(data)
        model.root_order = int(data['root_order'])
        model.root_scan = int(data['root_scan'])
        model.endpoint_weight = str(data['endpoint_weight']) if 'endpoint_weight' in data else 'roots'
        return model
