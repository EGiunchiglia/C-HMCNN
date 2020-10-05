import os

datasets = {
    'enron_others': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/others/Enron_corr_trainvalid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/others/Enron_corr_test.arff'
    ),
    'diatoms_others': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/others/Diatoms_train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/others/Diatoms_test.arff'
    ),
    'imclef07a_others': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/others/ImCLEF07A_Train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/others/ImCLEF07A_Test.arff'
    ),
    'imclef07d_others': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/others/ImCLEF07D_Train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/others/ImCLEF07D_Test.arff'
    ),
    'reuters_others': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/others/reuters_train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/others/reuters_test.arff'
    ),
    'wipo_others': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/others/wipotrain.sparse.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/others/wipotest.sparse.arff'
    ),
    'yeast_others': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data/others/D0_yeast_GO.trainvalid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/others/D0_yeast_GO.test.arff'
    ),
    'cellcycle_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/cellcycle_FUN/cellcycle_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/cellcycle_FUN/cellcycle_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/cellcycle_FUN/cellcycle_FUN.test.arff'
    ),
    'church_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/church_FUN/church_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/church_FUN/church_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/church_FUN/church_FUN.test.arff'
    ),
    'derisi_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/derisi_FUN/derisi_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/derisi_FUN/derisi_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/derisi_FUN/derisi_FUN.test.arff'
    ),
    'eisen_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/eisen_FUN/eisen_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/eisen_FUN/eisen_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/eisen_FUN/eisen_FUN.test.arff'
    ),
    'expr_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/expr_FUN/expr_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/expr_FUN/expr_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/expr_FUN/expr_FUN.test.arff'
    ),
    'gasch1_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/gasch1_FUN/gasch1_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/gasch1_FUN/gasch1_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/gasch1_FUN/gasch1_FUN.test.arff'
    ),
    'gasch2_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/gasch2_FUN/gasch2_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/gasch2_FUN/gasch2_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/gasch2_FUN/gasch2_FUN.test.arff'
    ),
    'hom_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/hom_FUN/hom_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/hom_FUN/hom_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/hom_FUN/hom_FUN.test.arff'
    ),
    'pheno_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/pheno_FUN/pheno_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/pheno_FUN/pheno_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/pheno_FUN/pheno_FUN.test.arff'
    ),
    'seq_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/seq_FUN/seq_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/seq_FUN/seq_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/seq_FUN/seq_FUN.test.arff'
    ),
    'spo_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/spo_FUN/spo_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/spo_FUN/spo_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/spo_FUN/spo_FUN.test.arff'
    ),
    'struc_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/struc_FUN/struc_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/struc_FUN/struc_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/struc_FUN/struc_FUN.test.arff'
    ),
    'cellcycle_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/cellcycle_GO/cellcycle_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/cellcycle_GO/cellcycle_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/cellcycle_GO/cellcycle_GO.test.arff'
    ),
    'church_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/church_GO/church_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/church_GO/church_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/church_GO/church_GO.test.arff'
    ),
    'derisi_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/derisi_GO/derisi_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/derisi_GO/derisi_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/derisi_GO/derisi_GO.test.arff'
    ),
    'eisen_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/eisen_GO/eisen_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/eisen_GO/eisen_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/eisen_GO/eisen_GO.test.arff'
    ),
    'expr_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/expr_GO/expr_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/expr_GO/expr_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/expr_GO/expr_GO.test.arff'
    ),
    'gasch1_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/gasch1_GO/gasch1_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/gasch1_GO/gasch1_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/gasch1_GO/gasch1_GO.test.arff'
    ),
    'gasch2_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/gasch2_GO/gasch2_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/gasch2_GO/gasch2_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/gasch2_GO/gasch2_GO.test.arff'
    ),
    'hom_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/hom_GO/hom_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/hom_GO/hom_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/hom_GO/hom_GO.test.arff'
    ),
    'pheno_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/pheno_GO/pheno_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/pheno_GO/pheno_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/pheno_GO/pheno_GO.test.arff'
    ),
    'seq_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/seq_GO/seq_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/seq_GO/seq_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/seq_GO/seq_GO.test.arff'
    ),
    'spo_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/spo_GO/spo_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/spo_GO/spo_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/spo_GO/spo_GO.test.arff'
    ),
    'struc_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/struc_GO/struc_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/struc_GO/struc_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/struc_GO/struc_GO.test.arff'
    ),
}

new_datasets = {
    'cellcycle_yeast_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/cellcycle_yeast_FUN/cellcycle_yeast_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/cellcycle_yeast_FUN/cellcycle_yeast_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/cellcycle_yeast_FUN/cellcycle_yeast_FUN.test.arff'
    ),
    'church_yeast_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/church_yeast_FUN/church_yeast_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/church_yeast_FUN/church_yeast_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/church_yeast_FUN/church_yeast_FUN.test.arff'
    ),
    'derisi_yeast_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/derisi_yeast_FUN/derisi_yeast_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/derisi_yeast_FUN/derisi_yeast_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/derisi_yeast_FUN/derisi_yeast_FUN.test.arff'
    ),
    'eisen_yeast_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/eisen_yeast_FUN/eisen_yeast_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/eisen_yeast_FUN/eisen_yeast_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/eisen_yeast_FUN/eisen_yeast_FUN.test.arff'
    ),
    'expr_yeast_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/expr_yeast_FUN/expr_yeast_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/expr_yeast_FUN/expr_yeast_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/expr_yeast_FUN/expr_yeast_FUN.test.arff'
    ),
    'exprindiv_ara_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/exprindiv_ara_FUN/exprindiv_ara_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/exprindiv_ara_FUN/exprindiv_ara_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/exprindiv_ara_FUN/exprindiv_ara_FUN.test.arff'
    ),
    'gasch1_yeast_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/gasch1_yeast_FUN/gasch1_yeast_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/gasch1_yeast_FUN/gasch1_yeast_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/gasch1_yeast_FUN/gasch1_yeast_FUN.test.arff'
    ),
    'gasch2_yeast_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/gasch2_yeast_FUN/gasch2_yeast_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/gasch2_yeast_FUN/gasch2_yeast_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/gasch2_yeast_FUN/gasch2_yeast_FUN.test.arff'
    ),
    'hom_ara_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/hom_ara_FUN/hom_ara_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/hom_ara_FUN/hom_ara_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/hom_ara_FUN/hom_ara_FUN.test.arff'
    ),
    'hom_yeast_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/hom_yeast_FUN/hom_yeast_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/hom_yeast_FUN/hom_yeast_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/hom_yeast_FUN/hom_yeast_FUN.test.arff'
    ),
    'interpro_ara_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/interpro_ara_FUN/interpro_ara_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/interpro_ara_FUN/interpro_ara_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/interpro_ara_FUN/interpro_ara_FUN.test.arff'
    ),
    'pheno_yeast_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/pheno_yeast_FUN/pheno_yeast_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/pheno_yeast_FUN/pheno_yeast_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/pheno_yeast_FUN/pheno_yeast_FUN.test.arff'
    ),
    'scop_ara_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/scop_ara_FUN/scop_ara_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/scop_ara_FUN/scop_ara_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/scop_ara_FUN/scop_ara_FUN.test.arff'
    ),
    'seq_yeast_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/seq_yeast_FUN/seq_yeast_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/seq_yeast_FUN/seq_yeast_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/seq_yeast_FUN/seq_yeast_FUN.test.arff'
    ),
    'spo_yeast_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/spo_yeast_FUN/spo_yeast_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/spo_yeast_FUN/spo_yeast_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/spo_yeast_FUN/spo_yeast_FUN.test.arff'
    ),
    'struc_yeast_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/struc_yeast_FUN/struc_yeast_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/struc_yeast_FUN/struc_yeast_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_FUN/struc_yeast_FUN/struc_yeast_FUN.test.arff'
    ),
    'struc_ara_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/struc_ara_GO/struc_ara_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/struc_ara_GO/struc_ara_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/struc_ara_GO/struc_ara_GO.test.arff'
    ),
    'seq_ara_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/seq_ara_GO/seq_ara_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/seq_ara_GO/seq_ara_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/seq_ara_GO/seq_ara_GO.test.arff'
    ),
    'scop_ara_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/scop_ara_GO/scop_ara_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/scop_ara_GO/scop_ara_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/scop_ara_GO/scop_ara_GO.test.arff'
    ),
    'mouse_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/mouse_GO/mouse_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/mouse_GO/mouse_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/mouse_GO/mouse_GO.test.arff'
    ),
    'interpro_ara_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/interpro_ara_GO/interpro_ara_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/interpro_ara_GO/interpro_ara_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/interpro_ara_GO/interpro_ara_GO.test.arff'
    ),
    'hom_ara_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/hom_ara_GO/hom_ara_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/hom_ara_GO/hom_ara_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/hom_ara_GO/hom_ara_GO.test.arff'
    ),
    'exprindiv_ara_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/exprindiv_ara_GO/exprindiv_ara_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/exprindiv_ara_GO/exprindiv_ara_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/exprindiv_ara_GO/exprindiv_ara_GO.test.arff'
    ),
    'borat_yeast_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/borat_yeast_GO/borat_yeast_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/borat_yeast_GO/borat_yeast_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data_v2/datasets_GO/borat_yeast_GO/borat_yeast_GO.test.arff'
    ),
}