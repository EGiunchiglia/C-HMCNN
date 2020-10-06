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
    'cellcycle_FUN': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/cellcycle_FUN/cellcycle_FUN.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/cellcycle_FUN/cellcycle_FUN.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_FUN/cellcycle_FUN/cellcycle_FUN.test.arff'
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
    'cellcycle_GO': (
        True,
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/cellcycle_GO/cellcycle_GO.train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/cellcycle_GO/cellcycle_GO.valid.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/datasets_GO/cellcycle_GO/cellcycle_GO.test.arff'
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
}
