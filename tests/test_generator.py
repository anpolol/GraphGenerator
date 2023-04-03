from core.generator import Main


def test_init_n_params():
    params = {
        "N": 100,
    }

    gen = Main(
        N=params["N"],
        max_d=500,
        L=3,
        etta=0.2,
        ro=1.1,
        mu=0.9,
        sigma_init=1,
        sigma_every=1,
        dim=32,
    )

    assert gen.N == params["N"]
