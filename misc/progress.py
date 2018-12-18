import sys


def progress(count, total, status='OVERALL PROGRESS'):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    fraction = ' (%s/%s)' % (count,total)
    sys.stdout.write('[%s] %s%s%s %s\r' % (bar, percents, '%', fraction, status))
    sys.stdout.flush()
