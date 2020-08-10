from __future__ import print_function
import math
import re
import sys
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase

class XconfigMishTdnnfLayer(XconfigLayerBase):

    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "mish-tdnnf-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input':'[-1]',
                       'dim':-1,
                       'bottleneck-dim':-1,
                       'bypass-scale':0.66,
                       'dropout-proportion':-1.0,
                       'time-stride':1,
                       'l2-regularize':0.0,
                       'max-change': 0.75,
                       'self-repair-scale': 1.0e-05,
                       'context': 'default'}

    def set_derived_configs(self):
        pass

    def check_configs(self):
        if self.config['bottleneck-dim'] <= 0:
            raise RuntimeError("bottleneck-dim must be set and >0.")
        if self.config['dim'] <= self.config['bottleneck-dim']:
            raise RuntimeError("dim must be greater than bottleneck-dim")

        dropout = self.config['dropout-proportion']
        if dropout != -1.0 and not (dropout >= 0.0 and dropout < 1.0):
            raise RuntimeError("invalid value for dropout-proportion")

        if abs(self.config['bypass-scale']) > 1.0:
            raise RuntimeError("bypass-scale has invalid value")

        input_dim = self.descriptors['input']['dim']
        output_dim = self.config['dim']
        if output_dim != input_dim and self.config['bypass-scale'] != 0.0:
            raise RuntimeError('bypass-scale is nonzero but output-dim != input-dim: {0} != {1}'
                               ''.format(output_dim, input_dim))

        if not self.config['context'] in ['default', 'left-only', 'shift-left', 'none']:
            raise RuntimeError('context must be default, left-only shift-left or none, got {}'.format(
                self.config['context']))


    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        output_component = ''
        if self.config['bypass-scale'] != 0.0:
            # the no-op component is used to cache something that we don't want
            # to have to recompute.
            output_component = 'noop'
        elif self.config['dropout-proportion'] != -1.0:
            output_component = 'dropout'
        else:
            output_component = 'batchnorm'
        return '{0}.{1}'.format(self.name, output_component)


    def output_dim(self, auxiliary_output=None):
        return self.config['dim']

    def get_full_config(self):
        ans = []
        config_lines = self._generate_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                ans.append((config_name, line))
        return ans


    def _generate_config(self):
        configs = []
        name = self.name
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        output_dim = self.config['dim']
        bottleneck_dim = self.config['bottleneck-dim']
        bypass_scale = self.config['bypass-scale']
        dropout_proportion = self.config['dropout-proportion']
        time_stride = self.config['time-stride']
        context = self.config['context']
        if time_stride != 0 and context != 'none':
            time_offsets1 = '{0},0'.format(-time_stride)
            if context == 'default':
                time_offsets2 = '0,{0}'.format(time_stride)
            elif context == 'shift-left':
                time_offsets2 = '{0},0'.format(-time_stride)
            else:
                assert context == 'left-only'
                time_offsets2 = '0'
        else:
            time_offsets1 = '0'
            time_offsets2 = '0'
        l2_regularize = self.config['l2-regularize']
        max_change = self.config['max-change']
        self_repair_scale = self.config['self-repair-scale']

        # The first linear layer, from input-dim (spliced x2) to bottleneck-dim
        configs.append('component name={0}.linear type=TdnnComponent input-dim={1} '
                       'output-dim={2} l2-regularize={3} max-change={4} use-bias=false '
                       'time-offsets={5} orthonormal-constraint=-1.0'.format(
                           name, input_dim, bottleneck_dim, l2_regularize,
                           max_change, time_offsets1))
        configs.append('component-node name={0}.linear component={0}.linear '
                       'input={1}'.format(name, input_descriptor))

        # The affine layer, from bottleneck-dim (spliced x2) to output-dim
        configs.append('component name={0}.affine type=TdnnComponent '
                       'input-dim={1} output-dim={2} l2-regularize={3} max-change={4} '
                       'time-offsets={5}'.format(
                           name, bottleneck_dim, output_dim, l2_regularize,
                           max_change, time_offsets2))
        configs.append('component-node name={0}.affine component={0}.affine '
                       'input={0}.linear'.format(name))

        # The Mish layer
        configs.append('component name={0}.softplus type=SoftplusComponent dim={1} '
                       'self-repair-scale={2}'.format(
                           name, output_dim, self_repair_scale))
        configs.append('component-node name={0}.relu component={0}.softplus '
                       'input={0}.affine'.format(name))
        configs.append('component name={0}.tanh type=TanhComponent dim={1} '
                       'self-repair-scale={2}'.format(
                           name, output_dim, self_repair_scale))
        configs.append('component-node name={0}.tanh component={0}.tanh '
                       'input={0}.softplus'.format(name))
        configs.append('component name={0}.ep type=ElementwiseProductComponent '
                       'dim={1} '.format(name, output_dim))
        configs.append('component-node name={0}.ep component={0}.ep '
                       'input=Append({0}.affine, {0}.tanh)'.format(name))

        # The BatchNorm layer
        configs.append('component name={0}.batchnorm type=BatchNormComponent '
                       'dim={1}'.format(name, output_dim))
        configs.append('component-node name={0}.batchnorm component={0}.batchnorm '
                       'input={0}.relu'.format(name))

        if dropout_proportion != -1:
            # This is not normal dropout.  It's dropout where the mask is shared
            # across time, and (thanks to continuous=true), instead of a
            # zero-or-one scale, it's a continuously varying scale whose
            # expected value is 1, drawn from a uniform distribution over an
            # interval of a size that varies with dropout-proportion.
            configs.append('component name={0}.dropout type=GeneralDropoutComponent '
                           'dim={1} dropout-proportion={2} continuous=true'.format(
                               name, output_dim, dropout_proportion))
            configs.append('component-node name={0}.dropout component={0}.dropout '
                           'input={0}.batchnorm'.format(name))
            cur_component_type = 'dropout'
        else:
            cur_component_type = 'batchnorm'

        if bypass_scale != 0.0:
            # Add a NoOpComponent to cache the weighted sum of the input and the
            # output.  We could easily have the output of the component be a
            # Descriptor like 'Append(Scale(0.66, tdnn1.batchnorm), tdnn2.batchnorm)',
            # but if we did that and you used many of this component in sequence,
            # the weighted sums would have more and more terms as you went deeper
            # in the network.
            configs.append('component name={0}.noop type=NoOpComponent '
                           'dim={1}'.format(name, output_dim))
            configs.append('component-node name={0}.noop component={0}.noop '
                           'input=Sum(Scale({1}, {2}), {0}.{3})'.format(
                               name, bypass_scale, input_descriptor,
                               cur_component_type))

        return configs
