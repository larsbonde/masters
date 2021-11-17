import hashlib
import numpy as np
from IPython.display import clear_output


def touch_output_files(save_dir, file_name_prefix, num_folds=1):
    """initializes training output files"""
    file_paths = [save_dir / f"{file_name_prefix}_{i}.pt" for i in range(num_folds)]
    for path in file_paths:
        path.touch(mode=0o664)
        assert path.exists() is True
    return file_paths


def get_non_dupe_dir(path):
    """makes a new dir that does not exist"""
    for i in range(10000):  # have some max limit
        rand_int = np.int64(np.random.randint(10e5))
        rand_hash = hashlib.md5(rand_int).hexdigest()
        new_path = path / f"{rand_hash}"
        if new_path.exists():
            continue
        else:
            break
    new_path = path / new_path
    try:
        new_path.mkdir(mode=0o775, parents=True, exist_ok=True)
    except OSError as err:
        print(f"Could not make output directory: {new_path}. Aborting. {err}")
    return new_path


def print_loss(t_loss, v_loss, clear_print=False):
    """Print training and validation losses"""
    if clear_print is True:
        clear_output(wait=True)
    i = 0
    for t, v in zip(t_loss, v_loss):
        print(f"epoch: {i + 1} - train_loss: {t:.8f} - valid_loss: {v:.8f}")
        i += 1


def display_func(i, max_i, epoch, t_loss, v_loss, extra_print=None):
    """Display epoch progress bar and losses of previous epochs"""
    clear_output(wait=True)
    
    i += 1
    
    perc = round(i/max_i, 3)
    prog_val = int(100 * perc)
    if prog_val < 10:  # adjust size of bar depending on number of chars in % val
        corr_num = 2
    elif prog_val == 100:
        corr_num = 4
    else:
        corr_num = 3
    
    max_white = 60  # width of bar
    prog_black = int(perc * max_white) - corr_num
    prog_white = max_white - prog_black - corr_num
    
    if extra_print:
        print(extra_print)
    print(
        f"\nepoch: {epoch + 1} - n: {i}/{max_i} - ", 
        "[",
        prog_black * "=",
        f"{prog_val}%",
        prog_white * " ",
        "]",
        sep=""
    )
    if len(t_loss) > 0:
        print_loss(t_loss, v_loss)