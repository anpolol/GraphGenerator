from core.generator import Main

def test_init_n_params():
    params = {
        'N': 2,
    }

    gen = Main(N=params['N'])

    assert gen.N == params['N']
