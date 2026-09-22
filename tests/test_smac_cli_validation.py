"""Unknown SMAC CLI arguments must fail loudly, not be swallowed."""

import pytest

from onpolicy.config import get_config
from onpolicy.scripts.train.train_smac import parse_args


def test_valid_arguments_are_accepted():
    args = parse_args(['--map_name', '3m', '--use_hetero_graph',
                       '--k_max', '4', '--top_k_filter', '2'], get_config())
    assert args.map_name == '3m'
    assert args.use_hetero_graph is True
    assert args.k_max == 4
    assert args.top_k_filter == 2


def test_misspelled_flag_raises():
    """--ally_faet_dim used to be dropped silently."""
    with pytest.raises(ValueError, match='Unknown command line arguments'):
        parse_args(['--map_name', '3m', '--ally_faet_dim', '99'], get_config())


@pytest.mark.parametrize('flag', [
    '--n_allies', '--n_enemies', '--ally_feat_dim', '--enemy_feat_dim',
])
def test_retired_dimension_flags_are_rejected(flag):
    """These are now derived from env metadata and must not be accepted."""
    with pytest.raises(ValueError, match='Unknown command line arguments'):
        parse_args(['--map_name', '3m', flag, '8'], get_config())


def test_error_message_names_the_offending_flag():
    with pytest.raises(ValueError) as excinfo:
        parse_args(['--map_name', '3m', '--not_a_real_flag', '1'], get_config())
    assert '--not_a_real_flag' in str(excinfo.value)


def test_legacy_hetero_script_invocation_now_fails():
    """The old 3m hetero script passed hand-computed dimensions."""
    legacy = [
        '--map_name', '3m', '--use_hetero_graph', '--hidden_size', '64',
        '--neighbor_dim', '8', '--agent_state_dim', '80', '--landmark_dim', '4',
        '--num_agents', '3', '--n_allies', '2', '--n_enemies', '3',
        '--ally_feat_dim', '8', '--enemy_feat_dim', '8',
    ]
    with pytest.raises(ValueError, match='Unknown command line arguments'):
        parse_args(legacy, get_config())
