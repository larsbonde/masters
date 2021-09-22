from IPython.display import clear_output


def print_loss(t_loss, v_loss, clear_print=False):
    """Print training and validation losses"""
    if clear_print is True:
        clear_output(wait=True)
    i = 0
    for t, v in zip(t_loss, v_loss):
        print(f"epoch: {i + 1} \ttrain_loss: {t:.8f} \tvalid_loss: {v:.8f}")
        i += 1


def display_func(i, max_i, epoch, t_loss, v_loss):
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
    
    print(
        f"\nepoch: {epoch + 1} n: {i}/{max_i} ", 
        "[",
        prog_black * "=",
        f"{prog_val}%",
        prog_white * " ",
        "]",
        sep=""
    )
    if len(t_loss) > 0:
        print_loss(t_loss, v_loss)