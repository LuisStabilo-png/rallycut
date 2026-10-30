import numpy as np

from rallycut.temporal import carregar_tcn, padronizar, prever_video, salvar_tcn, treinar_tcn


def _dados(n=3, janelas=400, canais=4):
    rng = np.random.default_rng(0)
    Xs = [rng.normal(size=(janelas, canais)) for _ in range(n)]
    ys = [np.c_[(x[:, 0] > 0).astype(float), (x[:, 1] > 1).astype(float)] for x in Xs]
    return padronizar(Xs, "video", list(range(n))), ys


def test_salvar_carregar_e_prever_video(tmp_path):
    Z, ys = _dados()
    modelo = treinar_tcn(Z, ys, passos=5, comp=100, lote=2, device="cpu")
    salvar_tcn([modelo], list("abcd"), tmp_path / "tcn.pt")
    modelos, colunas = carregar_tcn(tmp_path / "tcn.pt", "cpu")
    p = prever_video(modelos, Z[0], "cpu")
    assert colunas == list("abcd") and p.shape == (400,) and ((p >= 0) & (p <= 1)).all()
