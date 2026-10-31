from rallycut.pares import bruto_do_par, listar_pares


def _par(raiz, nome, *arquivos):
    pasta = raiz / nome
    pasta.mkdir(parents=True)
    for a in arquivos:
        (pasta / a).write_text("x")
    return pasta


def test_listar_pares_aceita_pares_e_revisados_e_pula_ignorados(tmp_path):
    _par(tmp_path, "quadra_a", "v_bruto.mp4", "v_editado.mp4")
    _par(tmp_path, "quadra_sem_editado", "v_bruto.mp4")
    _par(tmp_path, "quadra_ignorado", "v_bruto.mp4", "v_editado.mp4", "IGNORAR")
    r = _par(tmp_path, "praia_rev", "bruto.txt", "trechos.csv")
    _par(tmp_path, "praia_rev_sem_trechos", "bruto.txt")
    assert [p.name for p in listar_pares(tmp_path)] == ["praia_rev", "quadra_a"]
    (r / "bruto.txt").write_text("D:/videos/jogo_bruto.mp4\n")
    assert bruto_do_par(r).as_posix() == "D:/videos/jogo_bruto.mp4"
    assert bruto_do_par(tmp_path / "quadra_a").name == "v_bruto.mp4"
