import os
import sys

experiments = sys.argv[2:]
models = "./models"
result_folder = sys.argv[1]
os.mkdir(result_folder)

for ex in experiments:
    ex_path = os.path.join("./" + result_folder, ex)
    os.mkdir(ex_path)
    log_path = os.path.join(models, 'train_' + ex + '.log')
    os.system('cp %s %s' % (log_path, ex_path))
    for dir in os.listdir(os.path.join(models, 'model_' + ex)):
        if dir.startswith("fold"):
            tgt1 = os.path.join(models, 'model_' + ex, dir, 'progression_valInd.mat')
            tgt2 = os.path.join(models, 'model_' + ex, dir, "2020.ham" + ex, 'model.pkl')
            dest = os.path.join(ex_path, dir)
            os.mkdir(dest)
            os.system('cp %s %s' % (tgt1, dest))
            os.system('cp %s %s' % (tgt2, dest))
