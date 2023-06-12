import qiime2.plugin
import qiime2.sdk

from qiime2.plugin import (Plugin, MetadataColumn, MetadataCategory,
                           NumericMetadataColumn)
from qiime2.plugin import Citations

from q2_types.feature_table import FeatureTable, Frequency
from q2_types.sample_data import SampleData
from q2_types.per_sample_sequences import SequencesWithQuality
from q2_types.feature_data import FeatureData, Taxonomy
from q2_types.ordination import PCoAResults

from differential.method import clr_lmer


plugin = Plugin(
    name='differential',
    version='0.1.0',
    website='https://github.com/mortonjt/differential',
    short_description='QIIME 2 plugin for differential abundance analysis.',
    description=('This QIIME 2 plugin provides support for performing '
                 'differential abundance  analysis on biom tables with '
                 'associated metadata.'),
    package='differential'
)

plugin.methods.register_function(
    function=clr_lmer,
    inputs={'table': FeatureTable[Frequency],
            'metadata': Metadata},
    parameters={'subject_column': MetadataColumn[Categorical],
                'formula': qiime2.plugin.Str,
                're_formula': qiime2.plugin.Str,
                'n_jobs': qiime2.plugin.Int,
                'bootstrap': qiime2.plugin.Int},
    outputs=[('results', FeatureData[Differential])],
    name='clr_lmer',
    description=('Perform linear mixed effects analysis on a biom table with '
                 'associated metadata after applying the clr transform.'),
)


qiime2.sdk.PluginManager().register_plugin(plugin)
